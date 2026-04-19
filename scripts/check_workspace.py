#!/usr/bin/env python3
"""Validate whether the workspace is safe for a bounded harness task."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_PROTOCOL_FILES = (
    "protocol/LOOP.md",
    "protocol/STATE_MACHINE.md",
    "protocol/SURFACES.yaml",
    "protocol/ANTI_SHIM.md",
    "protocol/HITL_POLICY.md",
    "protocol/RESEARCH_POLICY.md",
    "protocol/INTEGRITY_PASS.md",
    "protocol/RUN_CLASSES.yaml",
    "protocol/CALIBRATION.md",
    "protocol/MODEL_SURFACE_PILOT.yaml",
)


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-mode",
        default="constitution_bootstrap",
        help="Task mode to validate against (default: constitution_bootstrap).",
    )
    parser.add_argument(
        "--write-lock",
        action="store_true",
        help="Regenerate protocol/hashes.lock for the selected mode.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    surfaces = load_json(repo_root / "protocol" / "SURFACES.yaml")
    task_modes = surfaces.get("task_modes", {})
    if args.task_mode not in task_modes:
        print(f"Unknown task mode: {args.task_mode}", file=sys.stderr)
        return 2

    mode = task_modes[args.task_mode]

    if args.write_lock:
        write_hash_lock(repo_root, args.task_mode, mode)
        print(f"Wrote protocol/hashes.lock for task mode {args.task_mode}")
        return 0

    result = check_workspace(repo_root, args.task_mode, mode)
    if result.warnings:
        for warning in result.warnings:
            print(f"WARNING: {warning}")
    if result.errors:
        for error in result.errors:
            print(f"ERROR: {error}")
        return 1

    print(f"Workspace valid for task mode: {args.task_mode}")
    return 0


def check_workspace(repo_root: Path, task_mode: str, mode: dict[str, Any]) -> CheckResult:
    errors: list[str] = []
    warnings: list[str] = []

    for relpath in REQUIRED_PROTOCOL_FILES:
        if not (repo_root / relpath).exists():
            errors.append(f"required protocol file missing: {relpath}")

    changed_files = git_changed_files(repo_root)
    if changed_files is None:
        warnings.append("git status unavailable; changed-file surface checks were skipped")
        changed_files = []

    editable = tuple(mode.get("editable", []))
    immutable = tuple(mode.get("immutable", []))
    approval_only = tuple(mode.get("approval_only", []))

    for relpath in changed_files:
        if matches_any(relpath, editable):
            continue
        if matches_any(relpath, immutable):
            errors.append(
                f"changed immutable path for task mode {task_mode}: {relpath}"
            )
            continue
        if matches_any(relpath, approval_only):
            errors.append(
                f"changed approval-only path for task mode {task_mode}: {relpath}"
            )
            continue
        warnings.append(f"changed path not classified by surface policy: {relpath}")

    hash_lock_path = repo_root / "protocol" / "hashes.lock"
    if hash_lock_path.exists():
        hash_lock = load_json(hash_lock_path)
        expected = hash_lock.get("task_modes", {}).get(task_mode, {})
        for relpath, expected_hash in expected.items():
            path = repo_root / relpath
            if not path.exists():
                errors.append(f"hash-locked immutable path missing: {relpath}")
                continue
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                errors.append(
                    f"immutable hash mismatch for {relpath}: expected {expected_hash}, got {actual_hash}"
                )
    else:
        warnings.append("protocol/hashes.lock is missing; immutable hash verification skipped")

    return CheckResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def write_hash_lock(repo_root: Path, task_mode: str, mode: dict[str, Any]) -> None:
    hash_lock_path = repo_root / "protocol" / "hashes.lock"
    if hash_lock_path.exists():
        payload = load_json(hash_lock_path)
    else:
        payload = {
            "schema_version": 1,
            "task_modes": {},
        }

    payload["task_modes"][task_mode] = compute_hash_map(
        repo_root,
        mode.get("immutable", []),
    )
    hash_lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def compute_hash_map(repo_root: Path, patterns: list[str]) -> dict[str, str]:
    hash_map: dict[str, str] = {}
    for relpath in repo_files(repo_root):
        if matches_any(relpath, patterns):
            hash_map[relpath] = sha256_file(repo_root / relpath)
    return hash_map


def repo_files(repo_root: Path) -> list[str]:
    files: list[str] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        files.append(path.relative_to(repo_root).as_posix())
    return sorted(files)


def git_changed_files(repo_root: Path) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    changed: list[str] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        relpath = line[3:]
        if " -> " in relpath:
            relpath = relpath.split(" -> ", 1)[1]
        changed.append(relpath)
    return changed


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def matches_any(relpath: str, patterns: tuple[str, ...] | list[str]) -> bool:
    return any(fnmatch.fnmatch(relpath, pattern) for pattern in patterns)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
