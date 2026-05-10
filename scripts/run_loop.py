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
from eval.schema import validate_artifact
from eval.sentinels import invalidating_findings, run_all_sentinels
from method.capacity import declared_trainable_parameter_count
from method.selected_stack import (
    load_champion_lane,
    load_default_surrogate_lane,
    load_lane_pair,
    load_model_lanes,
    load_selected_pilot_pair,
)
from scripts.research_state import (
    active_registry_versions,
    champion_needs_refresh,
    choose_next_branch,
    load_startup_summary,
    update_branch_card,
    update_champion_card,
)


CHAMPION_PATH = REPO_ROOT / "experiments" / "champion.json"
BOOTSTRAP_PATH = REPO_ROOT / "protocol" / "BOOTSTRAP.yaml"
RUN_CLASSES_PATH = REPO_ROOT / "protocol" / "RUN_CLASSES.yaml"
RUNS_LEDGER_PATH = REPO_ROOT / "experiments" / "ledgers" / "runs.jsonl"
SPECS_DIR = REPO_ROOT / "experiments" / "specs"
LOCK_PATH = REPO_ROOT / "locks" / "active_run.lock"
BOOTSTRAP_REQUEST_PATH = REPO_ROOT / "experiments" / "bootstrap" / "request.json"
SUBMISSIONS_DIR = REPO_ROOT / "experiments" / "submissions"
DECISIONS_APPLIED_DIR = REPO_ROOT / "experiments" / "decisions_applied"
LOCAL_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "runs"
DEFAULT_REMOTE_RUNTIME_PYTHON = "~/shared/envs/projects/autoresearch-continual-learning-env/bin/python"
DEFAULT_REMOTE_CONFIRM_PACK_DIR = "~/shared/artifacts/autoresearch-continual-learning/protected"
DEFAULT_SSH_PROXYCOMMAND = os.environ.get("AUTORESEARCH_SSH_PROXYCOMMAND")
VALID_TERMINAL_OUTCOMES = {
    "promote",
    "discard",
    "invalid",
    "needs_human_decision",
    "surrogate_pass",
}


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
    print(f"baseline_acceptance_tier: {payload['champion'].get('baseline_acceptance_tier')}")
    print(f"selected_base_model: {payload['selected_stack']['model_id']}")
    print(f"selected_surface: {payload['selected_stack']['surface_name']}")
    active_lanes = payload.get("active_model_lanes", {})
    champion_lane = active_lanes.get("champion_lane", {})
    surrogate_lane = active_lanes.get("surrogate_lane", {})
    if champion_lane:
        print(f"champion_lane_model: {champion_lane.get('model_id')}")
    if surrogate_lane:
        print(f"surrogate_lane_model: {surrogate_lane.get('model_id')}")
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
        remote_state = remote_run_state(lock)
        if remote_state["status"] == "alive":
            return transition("blocked", "heavyweight_run_active", {"run_id": lock.get("run_id")})
        if remote_state["status"] == "unreachable":
            return transition(
                "blocked",
                "remote_run_status_unreachable",
                {
                    "run_id": lock.get("run_id"),
                    "remote_host": lock.get("remote_host"),
                    "error": remote_state.get("error"),
                },
            )
        synced = sync_remote_outputs(
            run_id=str(lock.get("run_id")),
            remote_host=str(lock.get("remote_host")),
            remote_run_dir=str(lock.get("remote_run_dir")),
        )
        local_artifact = local_artifact_path(str(lock.get("run_id")))
        local_log = local_launcher_log_path(str(lock.get("run_id")))
        if not synced and not local_artifact.exists() and not local_log.exists():
            return transition(
                "blocked",
                "remote_outputs_missing",
                {
                    "run_id": lock.get("run_id"),
                    "remote_state": remote_state,
                },
            )
        clear_lock()
        return transition(
            "advanced",
            "released_completed_remote_run_lock",
            {
                "run_id": lock.get("run_id"),
                "artifact_path": display_path(local_artifact_path(str(lock.get("run_id")))),
            },
        )

    if candidate:
        decision = candidate.get("decision")
        if decision and not candidate.get("decision_applied"):
            apply_decision(decision=decision, artifact_path=resolve_path(candidate["artifact_path"]))
            return transition("advanced", "applied_terminal_decision", {"run_id": candidate["run_id"]})

        if candidate.get("artifact_path") is None:
            if candidate.get("submitted_before"):
                if recover_candidate_artifact(candidate):
                    return transition(
                        "advanced",
                        "recovered_remote_artifact",
                        {
                            "run_id": candidate["run_id"],
                            "artifact_path": display_path(local_artifact_path(candidate["run_id"])),
                        },
                    )
                if record_execution_failure(candidate):
                    return transition(
                        "advanced",
                        "recorded_execution_failure",
                        {
                            "run_id": candidate["run_id"],
                            "launcher_log_path": display_path(local_launcher_log_path(candidate["run_id"])),
                        },
                    )
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

        if candidate.get("confirmation_requested") and not candidate.get("confirmation_result_path"):
            if execute_confirmation(candidate):
                return transition(
                    "advanced",
                    "executed_protected_confirmation",
                    {
                        "run_id": candidate["run_id"],
                        "confirmation_result_path": display_path(
                            confirmation_result_path(candidate["run_id"])
                        ),
                    },
                )
            return transition(
                "blocked",
                "awaiting_confirmation_result",
                {"run_id": candidate["run_id"]},
            )

        if decision is None:
            decision_payload = compute_decision(candidate)
            if is_confirmation_only_hold(decision_payload):
                request_path = write_confirmation_request(
                    run_id=candidate["run_id"],
                    artifact_path=candidate["artifact_path"],
                    spec_path=candidate["spec_path"],
                    expected_pack_id=confirmation_pack_id_for_candidate(candidate),
                )
                return transition(
                    "blocked",
                    "awaiting_confirmation_result",
                    {
                        "run_id": candidate["run_id"],
                        "confirmation_request_path": display_path(request_path),
                    },
                )

            champion_spec_path = None
            if is_surrogate_graduation(decision_payload):
                champion_spec_path = freeze_champion_lane_spec_from_surrogate(
                    surrogate_candidate=candidate,
                    surrogate_decision=decision_payload,
                )
            record_terminal_decision(
                artifact_path=resolve_path(candidate["artifact_path"]),
                spec_path=resolve_path(candidate["spec_path"]),
                decision=decision_payload,
            )
            apply_decision(
                decision=decision_payload,
                artifact_path=resolve_path(candidate["artifact_path"]),
            )
            if champion_spec_path:
                return transition(
                    "advanced",
                    "froze_champion_lane_spec_from_surrogate",
                    {
                        "surrogate_run_id": candidate["run_id"],
                        "spec_path": champion_spec_path,
                    },
                )
            return transition(
                "advanced",
                "recorded_terminal_decision",
                {
                    "run_id": candidate["run_id"],
                    "outcome": decision_payload["outcome"],
                },
            )

    if state["startup_branch"] == "bootstrap_baseline_path":
        return transition(
            "advanced",
            "froze_bootstrap_baseline_spec",
            {
                "spec_path": freeze_bootstrap_baseline_spec(),
            },
        )

    if state["startup_branch"] == "governance_escalation":
        return transition("blocked", "needs_human_decision", state.get("latest_decision"))

    return transition(
        "advanced",
        "froze_next_method_iteration_spec",
        {
            "spec_path": freeze_next_method_iteration_spec(),
        },
    )


def inspect_loop_state() -> dict[str, Any]:
    champion = load_json(CHAMPION_PATH)
    bootstrap = load_json(BOOTSTRAP_PATH)
    legacy_selected_stack = load_selected_pilot_pair(REPO_ROOT)
    model_lanes = load_model_lanes(REPO_ROOT)
    champion_lane = load_champion_lane(REPO_ROOT)
    surrogate_lane = load_default_surrogate_lane(REPO_ROOT)
    run_classes = load_json(RUN_CLASSES_PATH)
    ledger = load_jsonl(RUNS_LEDGER_PATH)
    lock = read_lock_state()

    candidate = build_candidate_state(ledger)
    latest_decision = latest_terminal_decision(ledger)
    startup_summary = load_startup_summary(REPO_ROOT)
    champion_refresh_required = champion_needs_refresh(REPO_ROOT, champion)

    champion_status = champion.get("status") or "bootstrap_pending"
    current = champion.get("current_champion")
    current_run_id = current.get("run_id") if isinstance(current, Mapping) else None

    startup_branch = determine_startup_branch(
        champion_status=champion_status,
        current_run_id=current_run_id,
        champion_refresh_required=champion_refresh_required,
        lock=lock,
        candidate=candidate,
        latest_decision=latest_decision,
        ledger=ledger,
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
            "baseline_acceptance_tier": current.get("baseline_acceptance_tier")
            if isinstance(current, Mapping)
            else None,
            "last_decision": champion.get("last_decision"),
        },
        "selected_stack": {
            "source": "active_champion_lane",
            "model_id": champion_lane["model_id"],
            "surface_name": champion_lane["surface_name"],
        },
        "legacy_selected_pilot_stack": legacy_selected_stack,
        "active_model_lanes": {
            "active_family": model_lanes.get("active_family"),
            "champion_lane": champion_lane,
            "surrogate_lane": surrogate_lane,
        },
        "research_state": {
            "active_branch_id": startup_summary.get("active_branch_id"),
            "active_registry_versions": active_registry_versions(REPO_ROOT),
            "champion_refresh_required": champion_refresh_required,
        },
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
    champion_refresh_required: bool,
    lock: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    latest_decision: dict[str, Any] | None,
    ledger: list[dict[str, Any]],
) -> str:
    if lock is not None or candidate is not None:
        return "continue_active_run"
    if champion_refresh_required and has_failed_champion_lane_seed(ledger):
        return "default_method_iteration"
    if champion_refresh_required and latest_decision_is_stale_refresh_hold(latest_decision):
        return "bootstrap_baseline_path"
    if latest_decision and latest_decision.get("decision") == "needs_human_decision":
        return "governance_escalation"
    if champion_refresh_required:
        return "bootstrap_baseline_path"
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


def latest_decision_is_stale_refresh_hold(latest_decision: dict[str, Any] | None) -> bool:
    if not latest_decision or latest_decision.get("decision") != "needs_human_decision":
        return False
    spec_path = latest_decision.get("spec_path")
    if not isinstance(spec_path, str) or not spec_path:
        return False
    spec = safe_load_json(resolve_path(spec_path))
    data = spec.get("data", {})
    if not isinstance(data, Mapping):
        return False
    active_versions = active_registry_versions(REPO_ROOT)
    return any(
        data.get(field) != active_versions.get(field)
        for field in (
            "train_generator_version",
            "development_pack",
            "confirmation_pack_summary",
        )
    )


def latest_decision_is_failed_champion_lane_seed(
    latest_decision: dict[str, Any] | None,
) -> bool:
    if not latest_decision or latest_decision.get("decision") != "discard":
        return False
    spec_path = latest_decision.get("spec_path")
    if not isinstance(spec_path, str) or not spec_path:
        return False
    spec = safe_load_json(resolve_path(spec_path))
    model_lane = spec.get("model_lane", {})
    if not isinstance(model_lane, Mapping):
        return False
    return (
        model_lane.get("name") == load_champion_lane(REPO_ROOT)["name"]
        and model_lane.get("decision_mode") == "champion_bootstrap_or_compare"
        and spec.get("baseline_ref") == "bootstrap_baseline"
    )


def has_failed_champion_lane_seed(ledger: list[dict[str, Any]]) -> bool:
    for record in reversed(ledger):
        if latest_decision_is_failed_champion_lane_seed(record):
            return True
    return False


def build_candidate_state(ledger: list[dict[str, Any]]) -> dict[str, Any] | None:
    specs = sorted(SPECS_DIR.glob("*.yaml"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not specs:
        return None

    decisions = decisions_by_run_id(ledger)
    for spec_path in specs:
        run_id = spec_path.stem
        decision = decisions.get(run_id)
        if decision and decision_applied(run_id):
            continue
        artifact_path = discover_artifact(run_id)
        request_path = confirmation_request_path(run_id)
        result_path = confirmation_result_path(run_id)
        submission_path = submission_marker_path(run_id)
        submission = safe_load_json(submission_path) if submission_path.exists() else {}
        return {
            "run_id": run_id,
            "spec_path": display_path(spec_path),
            "artifact_path": display_path(artifact_path) if artifact_path else None,
            "confirmation_request_path": display_path(request_path) if request_path.exists() else None,
            "confirmation_result_path": display_path(result_path) if result_path.exists() else None,
            "confirmation_requested": request_path.exists(),
            "submission_path": display_path(submission_path) if submission_path.exists() else None,
            "submitted_before": submission_path.exists(),
            "remote_host": submission.get("remote_host"),
            "remote_repo_path": submission.get("remote_repo_path"),
            "remote_run_dir": submission.get("remote_run_dir"),
            "decision": decision,
            "decision_outcome": decision.get("decision") if decision else None,
            "decision_applied": decision_applied(run_id),
        }
    return None


def decision_applied(run_id: str) -> bool:
    return decision_applied_marker_path(run_id).exists()


def compute_decision(candidate: dict[str, Any]) -> dict[str, Any]:
    spec = safe_load_json(resolve_path(candidate["spec_path"]))
    model_lane = spec.get("model_lane", {})
    if (
        isinstance(model_lane, Mapping)
        and model_lane.get("decision_mode") == "surrogate_screen"
    ):
        return compute_surrogate_decision(candidate=candidate, spec=spec)

    champion = load_json(CHAMPION_PATH)
    champion_artifact_path = (
        None if champion_needs_refresh(REPO_ROOT, champion) else champion_current_artifact_path(champion)
    )
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


def compute_surrogate_decision(
    *,
    candidate: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_path = resolve_path(str(candidate["artifact_path"]))
    spec_path = resolve_path(str(candidate["spec_path"]))
    artifact = load_json(artifact_path)
    model_lane = spec.get("model_lane", {})
    validation = validate_artifact(artifact)
    findings = run_all_sentinels(
        run_spec=spec,
        artifact=artifact,
        immutable_hashes_verified=bool(
            artifact.get("integrity", {}).get("immutable_hashes_verified", False)
        ),
    )
    invalidating = invalidating_findings(findings)
    reasons: list[str] = []
    triggers: list[str] = []
    if not validation.valid:
        reasons.extend(f"schema_invalid:{error}" for error in validation.errors)
    reasons.extend(f"sentinel_invalid:{finding.code}" for finding in invalidating)
    integrity = artifact.get("integrity", {})
    if not isinstance(integrity, Mapping) or not integrity.get("preflight_passed"):
        reasons.append("preflight_not_passed")
    if not isinstance(integrity, Mapping) or not integrity.get("immutable_hashes_verified"):
        reasons.append("immutable_hashes_not_verified")
    if not isinstance(integrity, Mapping) or not integrity.get("shim_checks_passed"):
        reasons.append("shim_checks_not_passed")
    if reasons:
        outcome = "invalid"
    else:
        outcome = surrogate_gate_outcome(
            artifact=artifact,
            model_lane=model_lane if isinstance(model_lane, Mapping) else {},
            reasons=reasons,
            triggers=triggers,
        )

    return {
        "record_type": "decision",
        "recorded_at_utc": utc_now(),
        "artifact_path": display_path(artifact_path),
        "spec_path": display_path(spec_path),
        "champion_artifact_path": None,
        "confirmation_result_path": None,
        "run_id": artifact.get("run", {}).get("run_id"),
        "method_family": artifact.get("method", {}).get("method_family"),
        "run_class": artifact.get("run", {}).get("run_class"),
        "outcome": outcome,
        "reasons": reasons,
        "triggers": triggers,
    }


def surrogate_gate_outcome(
    *,
    artifact: Mapping[str, Any],
    model_lane: Mapping[str, Any],
    reasons: list[str],
    triggers: list[str],
) -> str:
    gate = model_lane.get("surrogate_gate", {})
    floors = gate.get("floors", {}) if isinstance(gate, Mapping) else {}
    if not isinstance(floors, Mapping):
        floors = {}
    metric_values = {
        "target_exact_match": artifact.get("metrics", {})
        .get("target_quality", {})
        .get("exact_match"),
        "anchor_exact_match": artifact.get("metrics", {})
        .get("interference", {})
        .get("anchor_exact_match"),
        "joint_success_rate": artifact.get("metrics", {})
        .get("interference", {})
        .get("joint_success_rate"),
    }
    failed = False
    for metric_name, floor_value in floors.items():
        observed = metric_values.get(str(metric_name))
        if not isinstance(floor_value, (int, float)):
            continue
        if not isinstance(observed, (int, float)):
            reasons.append(f"surrogate_metric_missing:{metric_name}")
            return "needs_human_decision"
        if observed < float(floor_value):
            failed = True
            reasons.append(
                f"surrogate_gate:{metric_name}_below_floor:{observed:.4f}<{float(floor_value):.4f}"
            )
        else:
            reasons.append(
                f"surrogate_gate:{metric_name}_satisfied:{observed:.4f}>={float(floor_value):.4f}"
            )
    if failed:
        triggers.append("surrogate_candidate_not_promising")
        return "discard"
    triggers.extend(["surrogate_candidate_promising", "champion_lane_run_required"])
    reasons.append("surrogate_screen_passed_champion_run_required")
    return "surrogate_pass"


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


def is_surrogate_graduation(decision_payload: Mapping[str, Any]) -> bool:
    if decision_payload.get("outcome") != "surrogate_pass":
        return False
    triggers = decision_payload.get("triggers", [])
    return isinstance(triggers, list) and "champion_lane_run_required" in triggers


def record_terminal_decision(
    *,
    artifact_path: Path,
    spec_path: Path,
    decision: Mapping[str, Any],
) -> None:
    metadata: dict[str, Any] = {}
    baseline_acceptance_tier = decision.get("baseline_acceptance_tier")
    if isinstance(baseline_acceptance_tier, str) and baseline_acceptance_tier:
        metadata["baseline_acceptance_tier"] = baseline_acceptance_tier
    for field in ("reasons", "triggers"):
        values = decision.get(field)
        if isinstance(values, list) and all(isinstance(value, str) for value in values):
            metadata[field] = values
    result = subprocess.run(
        [
            sys.executable,
            "scripts/parse_artifact.py",
            "--artifact",
            display_path(artifact_path),
            "--spec",
            display_path(spec_path),
            "--decision",
            str(decision["outcome"]),
            "--decision-metadata-json",
            json.dumps(metadata, sort_keys=True),
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode not in {0, 1}:
        raise RuntimeError((result.stderr or result.stdout).strip())


def apply_decision(*, decision: Mapping[str, Any], artifact_path: Path) -> None:
    champion = load_json(CHAMPION_PATH)
    artifact = load_json(artifact_path)
    current = champion.get("current_champion")
    current_exists = isinstance(current, Mapping) and bool(current.get("run_id"))
    outcome = decision.get("decision") or decision.get("outcome")
    if outcome not in VALID_TERMINAL_OUTCOMES:
        raise ValueError(f"Unsupported decision payload: {decision!r}")
    run_id = decision["run_id"]
    baseline_acceptance_tier = decision.get("baseline_acceptance_tier")
    if not isinstance(baseline_acceptance_tier, str):
        baseline_acceptance_tier = None
    method_family = artifact.get("method", {}).get("method_family")
    is_baseline_family = (
        isinstance(method_family, str) and method_family.startswith("baseline_")
    )

    champion["last_decision"] = {
        "run_id": run_id,
        "decision": outcome,
        "recorded_at_utc": utc_now(),
    }

    if outcome == "promote":
        bootstrap_seeded_from = champion.get("bootstrap", {}).get("seeded_from_run_id")
        is_bootstrap_seed = (
            champion.get("status") == "bootstrap_pending"
            or bootstrap_seeded_from == run_id
            or champion_needs_refresh(REPO_ROOT, champion)
        )
        if is_bootstrap_seed:
            promotion_source = "bootstrap_baseline"
        elif is_baseline_family:
            promotion_source = "baseline_refresh"
        else:
            promotion_source = "promotion"
        champion["current_champion"] = {
            "run_id": run_id,
            "artifact_path": display_path(artifact_path),
            "method_family": method_family,
            "base_model": artifact.get("method", {}).get("base_model"),
            "run_class": artifact.get("run", {}).get("run_class"),
            "promoted_at_utc": utc_now(),
            "promotion_source": promotion_source,
        }
        if is_bootstrap_seed or is_baseline_family:
            baseline_acceptance_tier = baseline_acceptance_tier or "provisional"
            champion["current_champion"]["baseline_acceptance_tier"] = baseline_acceptance_tier
            champion["status"] = "baseline_seeded"
            champion.setdefault("bootstrap", {})
            if is_bootstrap_seed:
                champion["bootstrap"]["seeded_from_run_id"] = run_id
                champion["bootstrap"]["seeded_at_utc"] = utc_now()
            champion["bootstrap"]["baseline_acceptance_tier"] = baseline_acceptance_tier
            if BOOTSTRAP_REQUEST_PATH.exists():
                BOOTSTRAP_REQUEST_PATH.unlink()
        else:
            champion["status"] = "active_champion"
    elif outcome in {"discard", "invalid", "needs_human_decision"} and not current_exists:
        champion["status"] = champion.get("status") or "bootstrap_pending"

    champion["updated_at_utc"] = utc_now()
    CHAMPION_PATH.write_text(json.dumps(champion, indent=2, sort_keys=True) + "\n")
    update_champion_card(REPO_ROOT, champion)
    update_branch_card(
        REPO_ROOT,
        method_family=method_family,
        run_id=run_id,
        outcome=str(outcome),
        baseline_acceptance_tier=baseline_acceptance_tier,
    )
    write_decision_applied_marker(
        run_id=run_id,
        outcome=outcome,
        baseline_acceptance_tier=baseline_acceptance_tier,
    )


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


def freeze_bootstrap_baseline_spec() -> str:
    bootstrap = load_json(BOOTSTRAP_PATH)
    lane = bootstrap.get("bootstrap_lane", {})
    champion = load_json(CHAMPION_PATH)
    branch = choose_next_branch(REPO_ROOT, champion)
    method_family = str(branch.get("method_family", "baseline_seq_lora_ft_v0"))
    model_lane = lane_for_branch(branch, bootstrap=True)
    declared_trainable_params = declared_trainable_parameter_count(
        method_family=method_family,
        base_model=model_lane["model_id"],
        surface_name=model_lane["surface_name"],
    )
    run_id = "bootstrap-refresh-" + utc_now().replace(":", "").replace("-", "")
    command = [
        sys.executable,
        "scripts/freeze_spec.py",
        "--run-id",
        run_id,
        "--run-class",
        model_lane["run_class"],
        "--method-family",
        method_family,
        "--base-model",
        model_lane["model_id"],
        "--task-mode",
        "constitution_bootstrap",
        "--command-mode",
        "bootstrap_baseline_v1",
        "--train-generator-version",
        str(lane.get("required_train_generator_version")),
        "--development-pack",
        str(lane.get("required_development_pack")),
        "--confirmation-pack-summary",
        str(lane.get("required_confirmation_pack_summary")),
        "--comparison-scope",
        model_lane["comparison_scope"],
        "--selection-policy",
        f"{model_lane['name']}_bootstrap_refresh",
        "--declared-trainable-params",
        str(declared_trainable_params),
        "--trainable-parameter-tolerance",
        "0",
        "--editable-path",
        "method/baselines/seq_lora_ft.py",
        "--editable-path",
        "method/trainer.py",
        "--editable-path",
        "scripts/run_bootstrap_baseline.py",
        "--baseline-ref",
        str(lane.get("baseline_ref", "bootstrap_baseline")),
        "--parent-champion",
        "no_champion_recorded",
        "--frozen-base-model",
        "--force",
    ]
    append_model_lane_args(command, model_lane)
    run_command(command)
    return f"experiments/specs/{run_id}.yaml"


def freeze_next_method_iteration_spec() -> str:
    champion = load_json(CHAMPION_PATH)
    branch = choose_next_branch(REPO_ROOT, champion)
    method_family = str(branch.get("method_family"))
    active_versions = active_registry_versions(REPO_ROOT)
    model_lane = lane_for_branch(branch, bootstrap=False)
    run_id_prefix = "mainline" if method_family == "hyper_lora_v0" else "baseline"
    run_id = f"{run_id_prefix}-{utc_now().replace(':', '').replace('-', '')}"
    command_mode = (
        "mainline_hyper_lora_v1" if method_family == "hyper_lora_v0" else "bootstrap_baseline_v1"
    )
    declared_trainable_params = declared_trainable_parameter_count(
        method_family=method_family,
        base_model=model_lane["model_id"],
        surface_name=model_lane["surface_name"],
    )
    editable_paths = [
        "method/trainer.py",
        "scripts/decide.py",
    ]
    if method_family == "hyper_lora_v0":
        editable_paths.extend(
            [
                "method/hypernet.py",
                "method/conflict_gate.py",
                "method/losses.py",
                "scripts/run_mainline_hyper_lora.py",
            ]
        )
    else:
        editable_paths.extend(
            [
                "method/baselines/seq_lora_ft.py",
                "scripts/run_bootstrap_baseline.py",
            ]
        )

    command = [
        sys.executable,
        "scripts/freeze_spec.py",
        "--run-id",
        run_id,
        "--run-class",
        model_lane["run_class"],
        "--method-family",
        method_family,
        "--base-model",
        model_lane["model_id"],
        "--task-mode",
        "constitution_bootstrap",
        "--command-mode",
        command_mode,
        "--train-generator-version",
        active_versions["train_generator_version"],
        "--development-pack",
        active_versions["development_pack"],
        "--confirmation-pack-summary",
        active_versions["confirmation_pack_summary"],
        "--comparison-scope",
        model_lane["comparison_scope"],
        "--selection-policy",
        f"{model_lane['name']}_branch_local_exploit",
        "--declared-trainable-params",
        str(declared_trainable_params),
        "--trainable-parameter-tolerance",
        "0",
    ]
    for editable_path in editable_paths:
        command.extend(["--editable-path", editable_path])
    current_champion = champion.get("current_champion", {})
    parent_champion = (
        current_champion.get("run_id", "no_champion_recorded")
        if isinstance(current_champion, Mapping)
        else "no_champion_recorded"
    )
    command.extend(["--parent-champion", parent_champion])
    append_model_lane_args(command, model_lane)
    command.extend(["--frozen-base-model", "--force"])
    run_command(command)
    return f"experiments/specs/{run_id}.yaml"


def freeze_champion_lane_spec_from_surrogate(
    *,
    surrogate_candidate: Mapping[str, Any],
    surrogate_decision: Mapping[str, Any],
) -> str:
    source_spec = safe_load_json(resolve_path(str(surrogate_candidate["spec_path"])))
    source_lane = source_spec.get("model_lane", {})
    target_lane_name = (
        source_lane.get("graduation_target_lane")
        if isinstance(source_lane, Mapping)
        else None
    )
    if not isinstance(target_lane_name, str) or not target_lane_name.strip():
        target_lane_name = load_champion_lane(REPO_ROOT)["name"]
    model_lane = load_lane_pair(target_lane_name, REPO_ROOT)
    method_family = str(source_spec.get("method_family"))
    active_versions = active_registry_versions(REPO_ROOT)
    declared_trainable_params = declared_trainable_parameter_count(
        method_family=method_family,
        base_model=model_lane["model_id"],
        surface_name=model_lane["surface_name"],
    )
    champion = load_json(CHAMPION_PATH)
    lane_context = champion_lane_reference_context(REPO_ROOT, champion)
    is_hyper_lora = method_family == "hyper_lora_v0"
    run_id_prefix = "mainline" if is_hyper_lora else "baseline"
    command_mode = "mainline_hyper_lora_v1" if is_hyper_lora else "bootstrap_baseline_v1"
    run_id = f"{run_id_prefix}-{utc_now().replace(':', '').replace('-', '')}"
    command = [
        sys.executable,
        "scripts/freeze_spec.py",
        "--run-id",
        run_id,
        "--run-class",
        model_lane["run_class"],
        "--method-family",
        method_family,
        "--base-model",
        model_lane["model_id"],
        "--task-mode",
        "constitution_bootstrap",
        "--command-mode",
        command_mode,
        "--train-generator-version",
        active_versions["train_generator_version"],
        "--development-pack",
        active_versions["development_pack"],
        "--confirmation-pack-summary",
        active_versions["confirmation_pack_summary"],
        "--comparison-scope",
        model_lane["comparison_scope"],
        "--selection-policy",
        f"{model_lane['name']}_surrogate_graduation",
        "--declared-trainable-params",
        str(declared_trainable_params),
        "--trainable-parameter-tolerance",
        "0",
        "--baseline-ref",
        lane_context["baseline_ref"],
        "--parent-champion",
        lane_context["parent_champion"],
    ]
    for editable_path in source_spec.get("editable_surface", {}).get("paths", []):
        if isinstance(editable_path, str) and editable_path:
            command.extend(["--editable-path", editable_path])
    append_model_lane_args(
        command,
        model_lane,
        source_run_id=str(surrogate_decision.get("run_id") or surrogate_candidate["run_id"]),
    )
    command.extend(["--frozen-base-model", "--force"])
    run_command(command)
    return f"experiments/specs/{run_id}.yaml"


def champion_lane_reference_context(
    repo_root: Path,
    champion: Mapping[str, Any],
) -> dict[str, str]:
    """Return baseline/parent refs for a champion-lane spec.

    A stale cross-lane champion is not valid comparison context for a Qwen
    champion bootstrap, so graduated surrogate specs must make bootstrap mode
    explicit until a same-lane champion exists.
    """

    if champion_needs_refresh(repo_root, champion):
        return {
            "baseline_ref": "bootstrap_baseline",
            "parent_champion": "no_champion_recorded",
        }
    current_champion = champion.get("current_champion", {})
    parent_champion = (
        current_champion.get("run_id", "no_champion_recorded")
        if isinstance(current_champion, Mapping)
        else "no_champion_recorded"
    )
    return {
        "baseline_ref": parent_champion,
        "parent_champion": parent_champion,
    }


def lane_for_branch(branch: Mapping[str, Any], *, bootstrap: bool) -> dict[str, Any]:
    lane_name = branch.get("bootstrap_target_lane") if bootstrap else branch.get("target_lane")
    if not isinstance(lane_name, str) or not lane_name.strip():
        lane_name = load_champion_lane(REPO_ROOT)["name"] if bootstrap else load_default_surrogate_lane(REPO_ROOT)["name"]
    return load_lane_pair(lane_name, REPO_ROOT)


def append_model_lane_args(
    command: list[str],
    model_lane: Mapping[str, Any],
    *,
    source_run_id: str | None = None,
) -> None:
    command.extend(
        [
            "--model-lane",
            str(model_lane["name"]),
            "--lane-role",
            str(model_lane.get("role", "")),
            "--lane-surface-name",
            str(model_lane["surface_name"]),
            "--lane-decision-mode",
            str(model_lane.get("decision_mode", "")),
        ]
    )
    graduation_target_lane = model_lane.get("graduation_target_lane")
    if isinstance(graduation_target_lane, str) and graduation_target_lane:
        command.extend(["--graduation-target-lane", graduation_target_lane])
    surrogate_gate = model_lane.get("surrogate_gate")
    if isinstance(surrogate_gate, Mapping):
        command.extend(["--surrogate-gate-json", json.dumps(surrogate_gate, sort_keys=True)])
    if source_run_id:
        command.extend(["--surrogate-source-run-id", source_run_id])


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


def decision_applied_marker_path(run_id: str) -> Path:
    return DECISIONS_APPLIED_DIR / f"{run_id}.json"


def write_decision_applied_marker(
    *,
    run_id: str,
    outcome: str,
    baseline_acceptance_tier: str | None = None,
) -> Path:
    path = decision_applied_marker_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "outcome": outcome,
        "applied_at_utc": utc_now(),
    }
    if baseline_acceptance_tier is not None:
        payload["baseline_acceptance_tier"] = baseline_acceptance_tier
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def recover_candidate_artifact(candidate: Mapping[str, Any]) -> bool:
    remote_host = candidate.get("remote_host")
    remote_run_dir = candidate.get("remote_run_dir")
    if not isinstance(remote_host, str) or not remote_host:
        return False
    if not isinstance(remote_run_dir, str) or not remote_run_dir:
        return False
    sync_remote_outputs(
        run_id=str(candidate["run_id"]),
        remote_host=remote_host,
        remote_run_dir=remote_run_dir,
    )
    return local_artifact_path(str(candidate["run_id"])).exists()


def record_execution_failure(candidate: Mapping[str, Any]) -> bool:
    run_id = str(candidate["run_id"])
    log_path = local_launcher_log_path(run_id)
    if not log_path.exists():
        return False
    log_text = log_path.read_text(errors="ignore")
    if "Traceback" not in log_text and "Error" not in log_text and "Exception" not in log_text:
        return False

    record = {
        "record_type": "execution_failure",
        "recorded_at_utc": utc_now(),
        "run_id": run_id,
        "spec_path": candidate.get("spec_path"),
        "artifact_path": None,
        "parent_champion": None,
        "schema_valid": False,
        "schema_errors": ["artifact_missing_after_submission"],
        "sentinel_findings": [],
        "decision": "invalid",
        "failure_reason": "remote_execution_failure",
        "launcher_log_path": display_path(log_path),
    }
    append_jsonl(RUNS_LEDGER_PATH, record)
    append_jsonl(REPO_ROOT / "experiments" / "ledgers" / "invalid_runs.jsonl", record)
    champion = load_json(CHAMPION_PATH)
    champion["last_decision"] = {
        "run_id": run_id,
        "decision": "invalid",
        "recorded_at_utc": utc_now(),
    }
    champion["updated_at_utc"] = utc_now()
    CHAMPION_PATH.write_text(json.dumps(champion, indent=2, sort_keys=True) + "\n")
    write_decision_applied_marker(run_id=run_id, outcome="invalid")
    return True


def execute_confirmation(candidate: Mapping[str, Any]) -> bool:
    run_id = str(candidate["run_id"])
    remote_host = candidate.get("remote_host")
    remote_repo_path = candidate.get("remote_repo_path")
    remote_run_dir = candidate.get("remote_run_dir")
    request_path = confirmation_request_path(run_id)
    result_path = confirmation_result_path(run_id)
    local_artifact = local_artifact_path(run_id)
    if not request_path.exists() or not local_artifact.exists():
        return False
    if not isinstance(remote_host, str) or not remote_host:
        return False
    if not isinstance(remote_repo_path, str) or not remote_repo_path:
        return False
    if not isinstance(remote_run_dir, str) or not remote_run_dir:
        return False

    remote_request = f"{remote_repo_path.rstrip('/')}/{display_path(request_path)}"
    remote_result = f"{remote_repo_path.rstrip('/')}/{display_path(result_path)}"
    request_payload = load_json(request_path)
    pack_id = (
        str(request_payload.get("pack_id")).strip()
        if isinstance(request_payload.get("pack_id"), str) and str(request_payload.get("pack_id")).strip()
        else "cl_confirm_locked_v1"
    )
    sync_path_to_remote(
        local_path=request_path,
        remote_host=remote_host,
        remote_path=remote_request,
    )

    command = ssh_command(
        remote_host,
        (
            f"cd {shlex.quote(remote_repo_path)} && "
            f"{DEFAULT_REMOTE_RUNTIME_PYTHON} eval/protected_runner.py execute-local "
            f"--request {shlex.quote(remote_request)} "
            f"--run-dir {shlex.quote(remote_run_dir)} "
            f"--output {shlex.quote(remote_result)} "
            f"--pack-path {shlex.quote(remote_confirmation_pack_path(pack_id))}"
        ),
    )
    result = subprocess.run(command, cwd=REPO_ROOT, check=False, text=True)
    if result.returncode not in {0, 1}:
        raise RuntimeError("remote confirmation execution failed")
    if remote_path_exists(remote_host=remote_host, remote_path=remote_result):
        sync_path_from_remote(
            remote_host=remote_host,
            remote_path=remote_result,
            local_path=result_path,
        )
    return result_path.exists()


def sync_remote_outputs(*, run_id: str, remote_host: str, remote_run_dir: str) -> bool:
    local_dir = LOCAL_ARTIFACTS_DIR / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    command = rsync_prefix()
    command.extend(
        [
        "--include",
        "artifact.json",
        "--include",
        "submission.json",
        "--include",
        "training_summary.json",
        "--include",
        "launcher.log",
        "--exclude",
        "*",
        f"{remote_host}:{remote_run_dir.rstrip('/')}/",
        f"{local_dir.as_posix().rstrip('/')}/",
        ]
    )
    result = subprocess.run(command, cwd=REPO_ROOT, check=False, text=True)
    return result.returncode == 0


def sync_path_to_remote(*, local_path: Path, remote_host: str, remote_path: str) -> None:
    mkdir_command = ssh_command(
        remote_host,
        f"mkdir -p {shlex.quote(str(Path(remote_path).parent))}",
    )
    subprocess.run(mkdir_command, cwd=REPO_ROOT, check=True, text=True)
    command = rsync_prefix()
    command.extend(
        [
        local_path.as_posix(),
        f"{remote_host}:{remote_path}",
        ]
    )
    subprocess.run(command, cwd=REPO_ROOT, check=True, text=True)


def sync_path_from_remote(*, remote_host: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    command = rsync_prefix()
    command.extend(
        [
        f"{remote_host}:{remote_path}",
        local_path.as_posix(),
        ]
    )
    subprocess.run(command, cwd=REPO_ROOT, check=True, text=True)


def remote_path_exists(*, remote_host: str, remote_path: str) -> bool:
    command = ssh_command(
        remote_host,
        f"test -f {shlex.quote(remote_path)}",
    )
    result = subprocess.run(command, cwd=REPO_ROOT, check=False, text=True)
    return result.returncode == 0


def local_artifact_path(run_id: str) -> Path:
    return (LOCAL_ARTIFACTS_DIR / run_id / "artifact.json").resolve()


def local_launcher_log_path(run_id: str) -> Path:
    return (LOCAL_ARTIFACTS_DIR / run_id / "launcher.log").resolve()


def confirmation_pack_id_for_candidate(candidate: Mapping[str, Any]) -> str | None:
    spec_path = candidate.get("spec_path")
    if not isinstance(spec_path, str) or not spec_path:
        return None
    spec = load_json(resolve_path(spec_path))
    data = spec.get("data", {})
    pack_id = data.get("confirmation_pack_summary")
    return pack_id if isinstance(pack_id, str) and pack_id else None


def remote_confirmation_pack_path(pack_id: str) -> str:
    return f"{DEFAULT_REMOTE_CONFIRM_PACK_DIR.rstrip('/')}/{pack_id}.json"


def ssh_prefix() -> list[str]:
    command = ["ssh"]
    if DEFAULT_SSH_PROXYCOMMAND:
        command.extend(["-o", f"ProxyCommand={DEFAULT_SSH_PROXYCOMMAND}"])
    return command


def ssh_command(remote_host: str, remote_script: str) -> list[str]:
    return [*ssh_prefix(), remote_host, f"bash -lc {shlex.quote(remote_script)}"]


def rsync_prefix() -> list[str]:
    command = ["rsync", "-az"]
    if DEFAULT_SSH_PROXYCOMMAND:
        command.extend(["-e", f"ssh -o ProxyCommand={shlex.quote(DEFAULT_SSH_PROXYCOMMAND)}"])
    return command


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


def remote_run_state(lock: Mapping[str, Any]) -> dict[str, Any]:
    remote_host = lock.get("remote_host")
    remote_pid = lock.get("remote_pid")
    remote_run_dir = lock.get("remote_run_dir")
    if not isinstance(remote_host, str) or not remote_host:
        return {"status": "missing_remote_host"}
    if not isinstance(remote_pid, str) or not remote_pid.strip():
        return {"status": "missing_remote_pid"}
    command = ssh_command(
        remote_host,
        f"kill -0 {shlex.quote(remote_pid)} >/dev/null 2>&1",
    )
    result = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        return {"status": "alive"}
    if result.returncode == 255:
        return {"status": "unreachable", "error": (result.stderr or result.stdout).strip()}
    if not isinstance(remote_run_dir, str) or not remote_run_dir:
        return {"status": "completed_or_missing", "remote_pid_returncode": result.returncode}

    files_command = ssh_command(
        remote_host,
        (
            f"test -f {shlex.quote(remote_run_dir.rstrip('/') + '/artifact.json')} "
            f"-o -f {shlex.quote(remote_run_dir.rstrip('/') + '/launcher.log')}"
        ),
    )
    files_result = subprocess.run(
        files_command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if files_result.returncode == 255:
        return {
            "status": "unreachable",
            "error": (files_result.stderr or files_result.stdout).strip(),
        }
    if files_result.returncode == 0:
        return {"status": "completed", "remote_pid_returncode": result.returncode}
    return {"status": "missing_outputs", "remote_pid_returncode": result.returncode}


def remote_run_alive(lock: Mapping[str, Any]) -> bool:
    return remote_run_state(lock).get("status") == "alive"


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


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


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
