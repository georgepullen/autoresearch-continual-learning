#!/usr/bin/env python3
"""Validate a run artifact and append the appropriate audit ledgers."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.schema import validate_artifact
from eval.sentinels import invalidating_findings, run_all_sentinels


VALID_DECISIONS = {"promote", "discard", "invalid", "needs_human_decision"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="Path to artifact JSON.")
    parser.add_argument("--spec", help="Path to frozen spec JSON/YAML.")
    parser.add_argument("--decision", choices=sorted(VALID_DECISIONS))
    args = parser.parse_args()

    repo_root = REPO_ROOT
    artifact_path = (repo_root / args.artifact).resolve()
    artifact = load_json(artifact_path)
    validation = validate_artifact(artifact)

    sentinel_findings: tuple[dict[str, Any], ...] = ()
    if args.spec:
        spec_path = (repo_root / args.spec).resolve()
        spec = load_json(spec_path)
        findings = run_all_sentinels(
            run_spec=spec,
            artifact=artifact,
            immutable_hashes_verified=bool(
                artifact.get("integrity", {}).get("immutable_hashes_verified", False)
            ),
        )
        sentinel_findings = tuple(
            {
                "code": finding.code,
                "message": finding.message,
                "invalidates_run": finding.invalidates_run,
            }
            for finding in findings
        )
    else:
        spec_path = None
        findings = ()

    invalidating = invalidating_findings(findings)
    parse_record = {
        "record_type": "artifact_parse",
        "recorded_at_utc": utc_now(),
        "artifact_path": display_path(artifact_path, repo_root),
        "spec_path": display_path(spec_path, repo_root) if spec_path else None,
        "run_id": artifact.get("run", {}).get("run_id"),
        "parent_champion": artifact.get("run", {}).get("parent_champion"),
        "schema_valid": validation.valid,
        "schema_errors": list(validation.errors),
        "sentinel_findings": list(sentinel_findings),
        "decision": args.decision,
    }

    append_jsonl(repo_root / "experiments" / "ledgers" / "runs.jsonl", parse_record)

    if not validation.valid or invalidating or args.decision == "invalid":
        append_jsonl(
            repo_root / "experiments" / "ledgers" / "invalid_runs.jsonl",
            parse_record,
        )
    if args.decision == "promote":
        append_jsonl(
            repo_root / "experiments" / "ledgers" / "promotions.jsonl",
            parse_record,
        )
    if args.decision == "needs_human_decision":
        append_jsonl(
            repo_root / "experiments" / "ledgers" / "human_decisions.jsonl",
            parse_record,
        )

    print(json.dumps(parse_record, indent=2, sort_keys=True))
    return 0 if validation.valid and not invalidating else 1


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
