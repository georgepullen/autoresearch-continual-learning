#!/usr/bin/env python3
"""Submit one heavyweight run through the single-run 3090 path."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "locks" / "active_run.lock"
DEFAULT_REMOTE_HOST = "3090"
DEFAULT_WAKE_ENDPOINT = os.environ.get("AUTORESEARCH_WAKE_ENDPOINT")


@dataclass(frozen=True)
class LockState:
    payload: dict[str, Any]
    stale: bool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit one frozen spec.")
    submit_parser.add_argument("--spec", required=True, help="Path to frozen spec JSON.")
    submit_parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    submit_parser.add_argument(
        "--wake-endpoint",
        default=DEFAULT_WAKE_ENDPOINT,
        help=(
            "Optional wake endpoint used before SSH submission. "
            "Defaults to $AUTORESEARCH_WAKE_ENDPOINT when set."
        ),
    )
    submit_parser.add_argument("--remote-workspace-root", default="$HOME/workspace")
    submit_parser.add_argument("--remote-artifacts-root", default="$HOME/shared/artifacts")
    submit_parser.add_argument("--task-mode")
    submit_parser.add_argument("--skip-preflight", action="store_true")
    submit_parser.add_argument("--skip-wake", action="store_true")
    submit_parser.add_argument("--skip-sync", action="store_true")
    submit_parser.add_argument("--dry-run", action="store_true")

    status_parser = subparsers.add_parser("status", help="Show the active-run lock.")
    status_parser.add_argument("--json", action="store_true")

    release_parser = subparsers.add_parser("release", help="Release the active-run lock.")
    release_parser.add_argument("--force", action="store_true")
    release_parser.add_argument("--run-id")
    release_parser.add_argument("--reason", default="manual_release")

    args = parser.parse_args()

    if args.command == "submit":
        return submit_run(args)
    if args.command == "status":
        return show_status(as_json=args.json)
    if args.command == "release":
        return release_lock(args.run_id, force=args.force, reason=args.reason)
    raise AssertionError(f"Unhandled command: {args.command}")


def submit_run(args: argparse.Namespace) -> int:
    spec_path = resolve_path(args.spec)
    spec = load_json(spec_path)
    run_id = require_string(spec, "run_id")
    task_mode = args.task_mode or require_string(spec, "task_mode")
    remote_host = args.remote_host
    repo_name = REPO_ROOT.name
    remote_repo_path = f"{args.remote_workspace_root.rstrip('/')}/{repo_name}"
    remote_run_dir = (
        f"{args.remote_artifacts_root.rstrip('/')}/{repo_name}/runs/{run_id}"
    )

    preflight_result = None
    if not args.skip_preflight:
        preflight_result = run_preflight(task_mode)

    initial_payload = {
        "schema_version": 1,
        "phase": "submission_in_progress",
        "run_id": run_id,
        "spec_path": display_path(spec_path),
        "task_mode": task_mode,
        "local_owner_host": socket.gethostname(),
        "local_owner_pid": os.getpid(),
        "acquired_at_utc": utc_now(),
        "remote_host": remote_host,
        "remote_repo_path": remote_repo_path,
        "remote_run_dir": remote_run_dir,
        "preflight": preflight_result,
    }

    acquire_lock(initial_payload)
    try:
        if not args.skip_wake and args.wake_endpoint:
            run_command(
                ["curl", "-fsS", args.wake_endpoint],
                dry_run=args.dry_run,
            )

        prep_command = ["ssh", remote_host, "bash", "-lc", f"prepare-workspace-repo {shlex.quote(repo_name)}"]
        run_command(prep_command, dry_run=args.dry_run)

        if not args.skip_sync:
            sync_repo(remote_host, remote_repo_path, dry_run=args.dry_run)

        remote_pid, remote_submit_command = submit_remote(
            remote_host=remote_host,
            remote_repo_path=remote_repo_path,
            remote_run_dir=remote_run_dir,
            spec=spec,
            spec_path=spec_path,
            dry_run=args.dry_run,
        )

        final_payload = {
            **initial_payload,
            "phase": "active_remote_run",
            "submitted_at_utc": utc_now(),
            "remote_pid": remote_pid,
            "remote_submit_command": remote_submit_command,
            "last_sync_at_utc": utc_now() if not args.skip_sync else None,
            "wake_attempted_at_utc": utc_now() if (not args.skip_wake and args.wake_endpoint) else None,
        }
        write_lock_payload(final_payload)
        print(json.dumps(final_payload, indent=2, sort_keys=True))
        return 0
    except Exception:
        clear_lock()
        raise


def show_status(*, as_json: bool) -> int:
    state = read_lock_state()
    if state is None:
        print("No active heavyweight run lock.")
        return 0

    payload = state.payload
    if state.stale:
        payload = {**payload, "stale": True}

    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for key in sorted(payload):
            print(f"{key}: {payload[key]}")
        if state.stale:
            print("stale: true")
    return 0


def release_lock(run_id: str | None, *, force: bool, reason: str) -> int:
    state = read_lock_state()
    if state is None:
        print("No active heavyweight run lock.")
        return 0

    payload = state.payload
    current_run_id = payload.get("run_id")
    if run_id and run_id != current_run_id:
        print(
            f"Lock belongs to run {current_run_id!r}, not requested run {run_id!r}.",
            file=sys.stderr,
        )
        return 1

    if payload.get("phase") == "active_remote_run" and not force:
        print(
            "Refusing to release an active_remote_run lock without --force.",
            file=sys.stderr,
        )
        return 1

    release_record = {
        **payload,
        "released_at_utc": utc_now(),
        "release_reason": reason,
    }
    clear_lock()
    print(json.dumps(release_record, indent=2, sort_keys=True))
    return 0


def acquire_lock(payload: dict[str, Any]) -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = read_lock_state()
    if existing is not None:
        if existing.stale:
            clear_lock()
        else:
            raise RuntimeError(
                "Another heavyweight run is already active. "
                "Inspect it with `python3 scripts/submit_run.py status`."
            )

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(LOCK_PATH, flags, 0o644)
    except FileExistsError as exc:
        raise RuntimeError("Failed to acquire active_run.lock") from exc

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_lock_payload(payload: dict[str, Any]) -> None:
    LOCK_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def clear_lock() -> None:
    if LOCK_PATH.exists():
        LOCK_PATH.unlink()


def read_lock_state() -> LockState | None:
    if not LOCK_PATH.exists():
        return None

    payload = load_json(LOCK_PATH)
    phase = payload.get("phase")
    local_owner_host = payload.get("local_owner_host")
    local_owner_pid = payload.get("local_owner_pid")
    stale = False

    if (
        phase == "submission_in_progress"
        and isinstance(local_owner_host, str)
        and isinstance(local_owner_pid, int)
        and local_owner_host == socket.gethostname()
        and not pid_exists(local_owner_pid)
    ):
        stale = True

    return LockState(payload=payload, stale=stale)


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def run_preflight(task_mode: str) -> dict[str, Any]:
    command = [sys.executable, "scripts/check_workspace.py", "--task-mode", task_mode]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Preflight failed before heavyweight submission:\n"
            + (result.stdout + result.stderr).strip()
        )
    return {
        "task_mode": task_mode,
        "checked_at_utc": utc_now(),
        "stdout": result.stdout.strip(),
    }


def sync_repo(remote_host: str, remote_repo_path: str, *, dry_run: bool) -> None:
    command = [
        "rsync",
        "-az",
        "--exclude",
        ".git/",
        "--exclude",
        "__pycache__/",
        "--exclude",
        "artifacts/",
        "--exclude",
        "locks/active_run.lock",
        f"{REPO_ROOT.as_posix().rstrip('/')}/",
        f"{remote_host}:{remote_repo_path.rstrip('/')}/",
    ]
    run_command(command, dry_run=dry_run)


def submit_remote(
    *,
    remote_host: str,
    remote_repo_path: str,
    remote_run_dir: str,
    spec: dict[str, Any],
    spec_path: Path,
    dry_run: bool,
) -> tuple[str | None, str]:
    remote_spec_path = f"{remote_repo_path.rstrip('/')}/{display_path(spec_path)}"
    run_command_text = require_string(spec, "command")
    remote_script = f"""
set -euo pipefail
mkdir -p "{remote_run_dir}"
cd "{remote_repo_path}"
nohup bash -lc {shlex.quote(run_command_text)} > "{remote_run_dir}/launcher.log" 2>&1 < /dev/null &
pid=$!
printf '%s\\n' "$pid" > "{remote_run_dir}/remote.pid"
cat > "{remote_run_dir}/submission.json" <<'JSON'
{json.dumps(
    {
        "run_id": spec.get("run_id"),
        "spec_path": remote_spec_path,
        "remote_command": run_command_text,
        "submitted_at_utc": utc_now(),
    },
    indent=2,
    sort_keys=True,
)}
JSON
printf '%s\\n' "$pid"
"""
    command = ["ssh", remote_host, "bash", "-lc", remote_script]
    result = run_command(command, dry_run=dry_run, capture_output=True)
    remote_pid = result.stdout.strip() if result and result.stdout else None
    return remote_pid, " ".join(shlex.quote(part) for part in command)


def run_command(
    command: list[str],
    *,
    dry_run: bool,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    if dry_run:
        print("DRY-RUN:", " ".join(shlex.quote(part) for part in command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Spec field {field!r} must be a non-empty string.")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
