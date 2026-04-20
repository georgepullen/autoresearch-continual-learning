#!/usr/bin/env python3
"""Closed-loop orchestration for the bounded continual-learning harness."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.protected_runner import (
    confirmation_request_path,
    confirmation_result_path,
    write_confirmation_request,
)
from method.selected_stack import load_selected_pilot_pair


CHAMPION_PATH = REPO_ROOT / "experiments" / "champion.json"
BOOTSTRAP_PATH = REPO_ROOT / "protocol" / "BOOTSTRAP.yaml"
RUN_CLASSES_PATH = REPO_ROOT / "protocol" / "RUN_CLASSES.yaml"
RUNS_LEDGER_PATH = REPO_ROOT / "experiments" / "ledgers" / "runs.jsonl"
SPECS_DIR = REPO_ROOT / "experiments" / "specs"
LOCK_PATH = REPO_ROOT / "locks" / "active_run.lock"
BOOTSTRAP_REQUEST_PATH = REPO_ROOT / "experiments" / "bootstrap" / "request.json"
SUBMISSIONS_DIR = REPO_ROOT / "experiments" / "submissions"
VALID_TERMINAL_OUTCOMES = {"promote", "discard", "invalid", "needs_human_decision"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="Report current loop state.")
    status_parser.add_argument("--json", action="store_true")

    step_parser = subparsers.add_parser("step", help="Perform one legal loop transition.")
    step_parser.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser("run", help="Advance until blocked.")
    run_parser.add_argument("--json", action="store_true")
    run_parser.add_argument("--max-steps", type=int, default=25)
    run_parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between successful transitions.",
    )

    args = parser.parse_args()
    if args.command == "status":
        return show_status(as_json=args.json)
    if args.command == "step":
        return step_loop(as_json=args.json)
    if args.command == "run":
        return run_loop(max_steps=args.max_steps, sleep_seconds=args.sleep_seconds, as_json=args.json)
    raise AssertionError(f"Unhandled command: {args.command}")


def show_status(*, as_json: bool) -> int:
    payload = inspect_loop_state()
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"startup_branch: {payload['startup_branch']}")
    print(f"next_action: {payload['next_action']}")
    print(f"champion_status: {payload['champion']['status']}")
    print(f"current_champion_run_id: {payload['champion'].get('current_run_id')}")
    print(f"selected_base_model: {payload['selected_stack']['model_id']}")
    print(f"selected_surface: {payload['selected_stack']['surface_name']}")
    if payload["active_run_lock"]:
        print("active_run_lock: present")
        print(f"active_run_phase: {payload['active_run_lock'].get('phase')}")
        print(f"active_run_id: {payload['active_run_lock'].get('run_id')}")
        if payload["active_run_lock"].get("stale"):
            print("active_run_stale: true")
    else:
        print("active_run_lock: none")
    if payload["candidate"]:
        print(f"candidate_run_id: {payload['candidate']['run_id']}")
        print(f"candidate_spec: {payload['candidate']['spec_path']}")
        print(f"candidate_artifact: {payload['candidate'].get('artifact_path')}")
        print(f"candidate_confirmation_request: {payload['candidate'].get('confirmation_request_path')}")
        print(f"candidate_confirmation_result: {payload['candidate'].get('confirmation_result_path')}")
        print(f"candidate_decision: {payload['candidate'].get('decision_outcome')}")
    else:
        print("candidate_run_id: none")
    return 0


def run_loop(*, max_steps: int, sleep_seconds: float, as_json: bool) -> int:
    transitions: list[dict[str, Any]] = []
    for _ in range(max_steps):
        transition = perform_transition()
        transitions.append(transition)
        if transition.get("status") == "blocked":
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    payload = {
        "steps_attempted": len(transitions),
        "transitions": transitions,
        "final_state": inspect_loop_state(),
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for item in transitions:
            print(f"{item['status']}: {item['action']}")
        print(f"final_next_action: {payload['final_state']['next_action']}")
    return 0


def step_loop(*, as_json: bool) -> int:
    transition = perform_transition()
    if as_json:
        print(json.dumps(transition, indent=2, sort_keys=True))
    else:
        print(f"{transition['status']}: {transition['action']}")
        if transition.get("details"):
            print(json.dumps(transition["details"], indent=2, sort_keys=True))
    return 0 if transition["status"] != "error" else 1


def perform_transition() -> dict[str, Any]:
    state = inspect_loop_state()
    lock = state["active_run_lock"]
    candidate = state["candidate"]

    if lock and lock.get("stale"):
        clear_lock()
        return transition("advanced", "released_stale_run_lock", {"run_id": lock.get("run_id")})

    if lock and lock.get("phase") == "active_remote_run":
        if remote_run_alive(lock):
            return transition("blocked", "heavyweight_run_active", {"run_id": lock.get("run_id")})
        clear_lock()
        return transition(
            "advanced",
            "released_completed_remote_run_lock",
            {"run_id": lock.get("run_id")},
        )

    if candidate:
        decision = candidate.get("decision")
        if decision and not candidate.get("decision_applied"):
            apply_decision(decision=decision, artifact_path=resolve_path(candidate["artifact_path"]))
            return transition("advanced", "applied_terminal_decision", {"run_id": candidate["run_id"]})

        if candidate.get("artifact_path") is None:
            if candidate.get("submitted_before"):
                return transition(
                    "blocked",
                    "artifact_missing_after_submission",
                    {"run_id": candidate["run_id"]},
                )
            run_command(
                [
                    sys.executable,
                    "scripts/submit_run.py",
                    "submit",
                    "--spec",
                    candidate["spec_path"],
                ]
            )
            record_submission_marker(candidate["run_id"], candidate["spec_path"])
            return transition("advanced", "submitted_frozen_spec", {"run_id": candidate["run_id"]})

        if decision is None:
            decision_payload = compute_decision(candidate)
            if is_confirmation_only_hold(decision_payload):
                request_path = write_confirmation_request(
                    run_id=candidate["run_id"],
                    artifact_path=candidate["artifact_path"],
                    spec_path=candidate["spec_path"],
                )
                return transition(
                    "blocked",
                    "awaiting_confirmation_result",
                    {
                        "run_id": candidate["run_id"],
                        "confirmation_request_path": display_path(request_path),
                    },
                )

            record_terminal_decision(
                artifact_path=resolve_path(candidate["artifact_path"]),
                spec_path=resolve_path(candidate["spec_path"]),
                outcome=decision_payload["outcome"],
            )
            apply_decision(
                decision=decision_payload,
                artifact_path=resolve_path(candidate["artifact_path"]),
            )
            return transition(
                "advanced",
                "recorded_terminal_decision",
                {
                    "run_id": candidate["run_id"],
                    "outcome": decision_payload["outcome"],
                },
            )

        if candidate.get("confirmation_requested") and not candidate.get("confirmation_result_path"):
            return transition(
                "blocked",
                "awaiting_confirmation_result",
                {"run_id": candidate["run_id"]},
            )

    if state["startup_branch"] == "bootstrap_baseline_path":
        request = ensure_bootstrap_request()
        return transition(
            "blocked",
            "bootstrap_spec_required",
            {
                "request_path": display_path(request),
                "approved_bootstrap_lane": display_path(BOOTSTRAP_PATH),
            },
        )

    if state["startup_branch"] == "governance_escalation":
        return transition("blocked", "needs_human_decision", state.get("latest_decision"))

    return transition("blocked", "awaiting_next_method_iteration", {})


def inspect_loop_state() -> dict[str, Any]:
    champion = load_json(CHAMPION_PATH)
    bootstrap = load_json(BOOTSTRAP_PATH)
    selected_stack = load_selected_pilot_pair(REPO_ROOT)
    run_classes = load_json(RUN_CLASSES_PATH)
    ledger = load_jsonl(RUNS_LEDGER_PATH)
    lock = read_lock_state()

    candidate = build_candidate_state(ledger)
    latest_decision = latest_terminal_decision(ledger)

    champion_status = champion.get("status") or "bootstrap_pending"
    current = champion.get("current_champion")
    current_run_id = current.get("run_id") if isinstance(current, Mapping) else None

    startup_branch = determine_startup_branch(
        champion_status=champion_status,
        current_run_id=current_run_id,
        lock=lock,
        candidate=candidate,
        latest_decision=latest_decision,
    )
    return {
        "startup_branch": startup_branch,
        "next_action": determine_next_action(
            startup_branch=startup_branch,
            lock=lock,
            candidate=candidate,
        ),
        "champion": {
            "status": champion_status,
            "current_run_id": current_run_id,
            "last_decision": champion.get("last_decision"),
        },
        "selected_stack": selected_stack,
        "bootstrap_lane": bootstrap.get("bootstrap_lane", {}),
        "run_class_readiness": {
            name: details.get("status")
            for name, details in run_classes.get("run_classes", {}).items()
            if isinstance(details, Mapping)
        },
        "active_run_lock": lock,
        "candidate": candidate,
        "latest_decision": latest_decision,
    }


def determine_startup_branch(
    *,
    champion_status: str,
    current_run_id: str | None,
    lock: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    latest_decision: dict[str, Any] | None,
) -> str:
    if lock is not None or candidate is not None:
        return "continue_active_run"
    if latest_decision and latest_decision.get("decision") == "needs_human_decision":
        return "governance_escalation"
    if champion_status == "bootstrap_pending" or current_run_id is None:
        return "bootstrap_baseline_path"
    return "default_method_iteration"


def determine_next_action(
    *,
    startup_branch: str,
    lock: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> str:
    if lock and lock.get("stale"):
        return "release_stale_run_lock"
    if lock:
        return "wait_for_heavyweight_run_completion"
    if candidate:
        if candidate.get("decision") and not candidate.get("decision_applied"):
            return "apply_terminal_decision"
        if candidate.get("artifact_path") is None:
            if candidate.get("submitted_before"):
                return "recover_or_sync_missing_artifact"
            return "submit_frozen_spec"
        if candidate.get("decision") is None and candidate.get("confirmation_requested"):
            return "wait_for_confirmation_result"
        if candidate.get("decision") is None:
            return "parse_and_decide_artifact"
    if startup_branch == "bootstrap_baseline_path":
        return "freeze_bootstrap_baseline_spec"
    if startup_branch == "governance_escalation":
        return "human_review_required"
    return "start_next_method_iteration"


def build_candidate_state(ledger: list[dict[str, Any]]) -> dict[str, Any] | None:
    specs = sorted(SPECS_DIR.glob("*.yaml"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not specs:
        return None

    decisions = decisions_by_run_id(ledger)
    for spec_path in specs:
        run_id = spec_path.stem
        decision = decisions.get(run_id)
        if decision and decision_applied(decision):
            continue
        artifact_path = discover_artifact(run_id)
        request_path = confirmation_request_path(run_id)
        result_path = confirmation_result_path(run_id)
        submission_path = submission_marker_path(run_id)
        return {
            "run_id": run_id,
            "spec_path": display_path(spec_path),
            "artifact_path": display_path(artifact_path) if artifact_path else None,
            "confirmation_request_path": display_path(request_path) if request_path.exists() else None,
            "confirmation_result_path": display_path(result_path) if result_path.exists() else None,
            "confirmation_requested": request_path.exists(),
            "submission_path": display_path(submission_path) if submission_path.exists() else None,
            "submitted_before": submission_path.exists(),
            "decision": decision,
            "decision_outcome": decision.get("decision") if decision else None,
            "decision_applied": decision_applied(decision),
        }
    return None


def decision_applied(decision: dict[str, Any] | None) -> bool:
    if not decision:
        return False
    champion = load_json(CHAMPION_PATH)
    last_decision = champion.get("last_decision")
    if not isinstance(last_decision, Mapping):
        return False
    return (
        last_decision.get("run_id") == decision.get("run_id")
        and last_decision.get("decision") == decision.get("decision")
    )


def compute_decision(candidate: dict[str, Any]) -> dict[str, Any]:
    champion = load_json(CHAMPION_PATH)
    champion_artifact_path = champion_current_artifact_path(champion)
    command = [
        sys.executable,
        "scripts/decide.py",
        "--artifact",
        candidate["artifact_path"],
        "--spec",
        candidate["spec_path"],
    ]
    if champion_artifact_path:
        command.extend(["--champion-artifact", champion_artifact_path])
    if candidate.get("confirmation_result_path"):
        command.extend(["--confirmation-result", candidate["confirmation_result_path"]])
    return run_command_json(command)


def champion_current_artifact_path(champion: Mapping[str, Any]) -> str | None:
    current = champion.get("current_champion")
    if not isinstance(current, Mapping):
        return None
    artifact_path = current.get("artifact_path")
    if isinstance(artifact_path, str) and artifact_path:
        return artifact_path
    run_id = current.get("run_id")
    if isinstance(run_id, str) and run_id:
        path = discover_artifact(run_id)
        if path:
            return display_path(path)
    return None


def is_confirmation_only_hold(decision_payload: Mapping[str, Any]) -> bool:
    if decision_payload.get("outcome") != "needs_human_decision":
        return False
    reasons = decision_payload.get("reasons", [])
    if not isinstance(reasons, list):
        return False
    if "confirmation_required_before_promotion" not in reasons:
        return False
    disqualifying = {
        reason
        for reason in reasons
        if isinstance(reason, str)
        and (
            reason.startswith("missing_champion_context")
            or reason.startswith("bootstrap_")
            or reason.startswith("confirmation_payload_invalid")
            or reason.startswith("non_numeric_metric_family")
        )
    }
    return not disqualifying


def record_terminal_decision(*, artifact_path: Path, spec_path: Path, outcome: str) -> None:
    run_command(
        [
            sys.executable,
            "scripts/parse_artifact.py",
            "--artifact",
            display_path(artifact_path),
            "--spec",
            display_path(spec_path),
            "--decision",
            outcome,
        ]
    )


def apply_decision(*, decision: Mapping[str, Any], artifact_path: Path) -> None:
    champion = load_json(CHAMPION_PATH)
    artifact = load_json(artifact_path)
    current = champion.get("current_champion")
    current_exists = isinstance(current, Mapping) and bool(current.get("run_id"))
    outcome = decision.get("decision") or decision.get("outcome")
    if outcome not in VALID_TERMINAL_OUTCOMES:
        raise ValueError(f"Unsupported decision payload: {decision!r}")
    run_id = decision["run_id"]

    champion["last_decision"] = {
        "run_id": run_id,
        "decision": outcome,
        "recorded_at_utc": utc_now(),
    }

    if outcome == "promote":
        promotion_source = "bootstrap_baseline" if champion.get("status") == "bootstrap_pending" else "promotion"
        champion["current_champion"] = {
            "run_id": run_id,
            "artifact_path": display_path(artifact_path),
            "method_family": artifact.get("method", {}).get("method_family"),
            "base_model": artifact.get("method", {}).get("base_model"),
            "run_class": artifact.get("run", {}).get("run_class"),
            "promoted_at_utc": utc_now(),
            "promotion_source": promotion_source,
        }
        if champion.get("status") == "bootstrap_pending":
            champion["status"] = "baseline_seeded"
            champion.setdefault("bootstrap", {})
            champion["bootstrap"]["seeded_from_run_id"] = run_id
            champion["bootstrap"]["seeded_at_utc"] = utc_now()
            if BOOTSTRAP_REQUEST_PATH.exists():
                BOOTSTRAP_REQUEST_PATH.unlink()
        else:
            champion["status"] = "active_champion"
    elif outcome in {"discard", "invalid", "needs_human_decision"} and not current_exists:
        champion["status"] = champion.get("status") or "bootstrap_pending"

    champion["updated_at_utc"] = utc_now()
    CHAMPION_PATH.write_text(json.dumps(champion, indent=2, sort_keys=True) + "\n")


def ensure_bootstrap_request() -> Path:
    if BOOTSTRAP_REQUEST_PATH.exists():
        return BOOTSTRAP_REQUEST_PATH
    bootstrap = load_json(BOOTSTRAP_PATH)
    lane = bootstrap.get("bootstrap_lane", {})
    suggested_run_id = "bootstrap-baseline-" + utc_now().replace(":", "").replace("-", "")
    payload = {
        "schema_version": 1,
        "requested_at_utc": utc_now(),
        "run_id": suggested_run_id,
        "required_method_families": lane.get("approved_method_families", []),
        "required_run_classes": lane.get("allowed_run_classes", []),
        "required_base_model": lane.get("required_base_model"),
        "required_development_pack": lane.get("required_development_pack"),
        "required_confirmation_pack_summary": lane.get("required_confirmation_pack_summary"),
        "required_train_generator_version": lane.get("required_train_generator_version"),
        "comparison_scope": lane.get("comparison_scope"),
        "baseline_ref": lane.get("baseline_ref"),
        "next_step": "Freeze one approved bootstrap baseline spec under protocol/BOOTSTRAP.yaml and then rerun `python3 scripts/run_loop.py step`.",
    }
    BOOTSTRAP_REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOOTSTRAP_REQUEST_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return BOOTSTRAP_REQUEST_PATH


def submission_marker_path(run_id: str) -> Path:
    return SUBMISSIONS_DIR / f"{run_id}.json"


def record_submission_marker(run_id: str, spec_path: str) -> Path:
    path = submission_marker_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "spec_path": spec_path,
            "submitted_at_utc": utc_now(),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def decisions_by_run_id(ledger: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for record in ledger:
        decision = record.get("decision")
        run_id = record.get("run_id")
        if decision in VALID_TERMINAL_OUTCOMES and isinstance(run_id, str):
            decisions[run_id] = record
    return decisions


def latest_terminal_decision(ledger: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in reversed(ledger):
        if record.get("decision") in VALID_TERMINAL_OUTCOMES:
            return record
    return None


def discover_artifact(run_id: str) -> Path | None:
    direct_candidates = (
        REPO_ROOT / "artifacts" / f"{run_id}.json",
        REPO_ROOT / "artifacts" / "runs" / run_id / "artifact.json",
        REPO_ROOT / "artifacts" / "runs" / run_id / "result.json",
        REPO_ROOT / "experiments" / "artifacts" / f"{run_id}.json",
    )
    for path in direct_candidates:
        if path.exists():
            return path.resolve()

    for root in (REPO_ROOT / "artifacts", REPO_ROOT / "experiments" / "artifacts"):
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            payload = safe_load_json(path)
            if payload.get("run", {}).get("run_id") == run_id:
                return path.resolve()
    return None


def read_lock_state() -> dict[str, Any] | None:
    if not LOCK_PATH.exists():
        return None
    payload = load_json(LOCK_PATH)
    if is_local_submission_stale(payload):
        return {**payload, "stale": True}
    return payload


def is_local_submission_stale(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("phase") == "submission_in_progress"
        and payload.get("local_owner_host") == os.uname().nodename
        and isinstance(payload.get("local_owner_pid"), int)
        and not pid_exists(int(payload["local_owner_pid"]))
    )


def remote_run_alive(lock: Mapping[str, Any]) -> bool:
    remote_host = lock.get("remote_host")
    remote_pid = lock.get("remote_pid")
    if not isinstance(remote_host, str) or not remote_host:
        return False
    if not isinstance(remote_pid, str) or not remote_pid.strip():
        return False
    command = [
        "ssh",
        remote_host,
        "bash",
        "-lc",
        f"kill -0 {shlex.quote(remote_pid)} >/dev/null 2>&1",
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return result.returncode == 0


def clear_lock() -> None:
    if LOCK_PATH.exists():
        LOCK_PATH.unlink()


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True, text=True)


def run_command_json(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return json.loads(result.stdout)


def transition(status: str, action: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "action": action,
        "details": details or {},
        "recorded_at_utc": utc_now(),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def safe_load_json(path: Path) -> dict[str, Any]:
    try:
        return load_json(path)
    except Exception:
        return {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / raw_path
    return path.resolve()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
