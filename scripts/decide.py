#!/usr/bin/env python3
"""Apply the tiered promotion policy to one candidate artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.protected_runner import evaluate_confirmation_result, load_confirmation_result
from eval.schema import validate_artifact
from eval.sentinels import invalidating_findings, run_all_sentinels


VALID_OUTCOMES = {"promote", "discard", "invalid", "needs_human_decision"}


@dataclass(frozen=True)
class DecisionResult:
    outcome: str
    reasons: tuple[str, ...]
    triggers: tuple[str, ...]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--champion-artifact")
    parser.add_argument("--confirmation-result")
    parser.add_argument(
        "--promotion-config",
        default="protocol/PROMOTION.yaml",
        help="Path to promotion config (JSON-shaped YAML).",
    )
    parser.add_argument(
        "--record-ledger",
        action="store_true",
        help="Append the decision result to experiments/ledgers/runs.jsonl.",
    )
    parser.add_argument(
        "--runs-ledger",
        default="experiments/ledgers/runs.jsonl",
        help="Ledger path used when --record-ledger is set.",
    )
    args = parser.parse_args()

    artifact_path = resolve_path(args.artifact)
    spec_path = resolve_path(args.spec)
    artifact = load_json(artifact_path)
    spec = load_json(spec_path)
    config = load_json(resolve_path(args.promotion_config))
    champion_artifact = (
        load_json(resolve_path(args.champion_artifact))
        if args.champion_artifact
        else None
    )
    confirmation_result = (
        load_confirmation_result(resolve_path(args.confirmation_result))
        if args.confirmation_result
        else None
    )

    decision = decide(
        artifact=artifact,
        spec=spec,
        champion_artifact=champion_artifact,
        confirmation_result=confirmation_result,
        config=config,
    )

    payload = {
        "record_type": "decision",
        "recorded_at_utc": utc_now(),
        "artifact_path": display_path(artifact_path),
        "spec_path": display_path(spec_path),
        "champion_artifact_path": display_path(resolve_path(args.champion_artifact))
        if args.champion_artifact
        else None,
        "confirmation_result_path": display_path(resolve_path(args.confirmation_result))
        if args.confirmation_result
        else None,
        "run_id": artifact.get("run", {}).get("run_id"),
        "method_family": artifact.get("method", {}).get("method_family"),
        "run_class": artifact.get("run", {}).get("run_class"),
        "outcome": decision.outcome,
        "reasons": list(decision.reasons),
        "triggers": list(decision.triggers),
    }
    if args.record_ledger:
        append_jsonl(resolve_path(args.runs_ledger), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if decision.outcome in {"promote", "discard"} else 1


def decide(
    *,
    artifact: Mapping[str, Any],
    spec: Mapping[str, Any],
    champion_artifact: Mapping[str, Any] | None,
    confirmation_result: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> DecisionResult:
    reasons: list[str] = []
    triggers: list[str] = []

    validation = validate_artifact(artifact)
    findings = run_all_sentinels(
        run_spec=spec,
        artifact=artifact,
        immutable_hashes_verified=bool(
            artifact.get("integrity", {}).get("immutable_hashes_verified", False)
        ),
    )
    invalidating = invalidating_findings(findings)

    integrity_failure_reasons = evaluate_integrity(
        artifact=artifact,
        validation=validation,
        findings=invalidating,
        config=config,
    )
    if integrity_failure_reasons:
        reasons.extend(integrity_failure_reasons)
        return DecisionResult(outcome="invalid", reasons=tuple(reasons), triggers=tuple(triggers))

    if champion_artifact is None:
        reasons.append("missing_champion_context")
        return DecisionResult(
            outcome="needs_human_decision",
            reasons=tuple(reasons),
            triggers=tuple(triggers),
        )

    champion_validation = validate_artifact(champion_artifact)
    if not champion_validation.valid:
        reasons.extend(
            f"champion_artifact_invalid:{error}" for error in champion_validation.errors
        )
        return DecisionResult(
            outcome="needs_human_decision",
            reasons=tuple(reasons),
            triggers=tuple(triggers),
        )

    floor_eval = evaluate_scientific_floors(artifact, champion_artifact, config)
    reasons.extend(floor_eval["reasons"])
    triggers.extend(floor_eval["triggers"])
    if floor_eval["outcome"] == "discard":
        return DecisionResult("discard", tuple(reasons), tuple(triggers))
    if floor_eval["outcome"] == "needs_human_decision":
        return DecisionResult("needs_human_decision", tuple(reasons), tuple(triggers))

    win_eval = evaluate_required_wins(artifact, champion_artifact, config)
    reasons.extend(win_eval["reasons"])
    triggers.extend(win_eval["triggers"])
    if win_eval["outcome"] == "discard":
        return DecisionResult("discard", tuple(reasons), tuple(triggers))
    if win_eval["outcome"] == "needs_human_decision":
        return DecisionResult("needs_human_decision", tuple(reasons), tuple(triggers))

    cost_eval = evaluate_cost_envelopes(artifact, champion_artifact, config)
    reasons.extend(cost_eval["reasons"])
    triggers.extend(cost_eval["triggers"])
    if cost_eval["outcome"] == "discard":
        return DecisionResult("discard", tuple(reasons), tuple(triggers))
    if cost_eval["outcome"] == "needs_human_decision":
        return DecisionResult("needs_human_decision", tuple(reasons), tuple(triggers))

    confirmation_eval = evaluate_confirmation_result(confirmation_result)
    reasons.extend(confirmation_eval.errors)
    if confirmation_eval.errors and confirmation_eval.status != "pending":
        reasons.append("confirmation_payload_invalid")
        return DecisionResult(
            outcome="needs_human_decision",
            reasons=tuple(reasons),
            triggers=tuple(triggers),
        )
    if confirmation_eval.status == "pending":
        missing_outcome = (
            config.get("confirmation_policy", {}).get("missing_confirmation_outcome")
            or "needs_human_decision"
        )
        reasons.append("confirmation_required_before_promotion")
        return DecisionResult(
            outcome=missing_outcome,
            reasons=tuple(reasons),
            triggers=tuple(triggers),
        )
    if confirmation_eval.status == "fail":
        reasons.append("protected_confirmation_failed")
        triggers.append("confirm_regression_pattern")
        failed_outcome = (
            config.get("confirmation_policy", {}).get("failed_confirmation_outcome")
            or "discard"
        )
        return DecisionResult(
            outcome=failed_outcome,
            reasons=tuple(reasons),
            triggers=tuple(triggers),
        )

    reasons.append("all_integrity_and_promotion_gates_passed")
    return DecisionResult(outcome="promote", reasons=tuple(reasons), triggers=tuple(triggers))


def evaluate_integrity(
    *,
    artifact: Mapping[str, Any],
    validation: Any,
    findings: tuple[Any, ...],
    config: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    gates = config.get("integrity_gates", {})
    integrity = artifact.get("integrity", {})

    if gates.get("require_schema_valid", True) and not validation.valid:
        reasons.extend(f"schema_invalid:{error}" for error in validation.errors)

    if gates.get("require_preflight_passed", True) and not integrity.get("preflight_passed"):
        reasons.append("preflight_not_passed")
    if gates.get("require_immutable_hashes_verified", True) and not integrity.get(
        "immutable_hashes_verified"
    ):
        reasons.append("immutable_hashes_not_verified")
    if gates.get("require_shim_checks_passed", True) and not integrity.get(
        "shim_checks_passed"
    ):
        reasons.append("shim_checks_not_passed")

    invalid_codes = set(gates.get("invalidating_sentinel_codes", []))
    for finding in findings:
        if finding.code in invalid_codes:
            reasons.append(f"sentinel_invalidated:{finding.code}")

    return reasons


def evaluate_scientific_floors(
    artifact: Mapping[str, Any],
    champion_artifact: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    triggers: list[str] = []

    for rule_name, rule in config.get("scientific_floors", {}).items():
        family = rule["family"]
        tolerance = float(rule.get("tolerance", 0.0))
        borderline = float(rule.get("borderline_tolerance", tolerance))
        comparisons = compare_numeric_family(
            artifact_family=artifact.get("metrics", {}).get(family, {}),
            champion_family=champion_artifact.get("metrics", {}).get(family, {}),
        )
        if comparisons is None:
            return {
                "outcome": "needs_human_decision",
                "reasons": [f"non_numeric_metric_family:{family}"],
                "triggers": triggers,
            }

        for metric, candidate, champion in comparisons:
            delta = candidate - champion
            if delta < -borderline:
                reasons.append(
                    f"{rule_name}:{metric}:candidate_regressed_beyond_floor:{delta:.6f}"
                )
                return {"outcome": "discard", "reasons": reasons, "triggers": triggers}
            if delta < -tolerance:
                reasons.append(
                    f"{rule_name}:{metric}:candidate_is_borderline_below_floor:{delta:.6f}"
                )
                triggers.append("borderline_replay_eligible")
                return {
                    "outcome": "needs_human_decision",
                    "reasons": reasons,
                    "triggers": triggers,
                }

        reasons.append(f"{rule_name}:all_metrics_within_target_quality_floor")

    return {"outcome": "pass", "reasons": reasons, "triggers": triggers}


def evaluate_required_wins(
    artifact: Mapping[str, Any],
    champion_artifact: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    triggers: list[str] = []

    for rule_name, rule in config.get("required_win_families", {}).items():
        family = rule["family"]
        min_improvement = float(rule.get("min_improvement", 0.0))
        borderline = float(rule.get("borderline_tolerance", 0.0))
        comparisons = compare_numeric_family(
            artifact_family=artifact.get("metrics", {}).get(family, {}),
            champion_family=champion_artifact.get("metrics", {}).get(family, {}),
        )
        if comparisons is None:
            return {
                "outcome": "needs_human_decision",
                "reasons": [f"non_numeric_metric_family:{family}"],
                "triggers": triggers,
            }

        improvements = []
        borderline_only = []
        for metric, candidate, champion in comparisons:
            improvement = champion - candidate
            if improvement >= min_improvement:
                improvements.append((metric, improvement))
            elif improvement >= borderline:
                borderline_only.append((metric, improvement))

        if improvements:
            reasons.append(
                f"{rule_name}:improved_metrics:"
                + ",".join(f"{metric}={value:.6f}" for metric, value in improvements)
            )
            continue
        if borderline_only:
            reasons.append(
                f"{rule_name}:borderline_only:"
                + ",".join(f"{metric}={value:.6f}" for metric, value in borderline_only)
            )
            triggers.append("borderline_replay_eligible")
            return {
                "outcome": "needs_human_decision",
                "reasons": reasons,
                "triggers": triggers,
            }

        reasons.append(f"{rule_name}:no_required_interference_improvement")
        return {"outcome": "discard", "reasons": reasons, "triggers": triggers}

    return {"outcome": "pass", "reasons": reasons, "triggers": triggers}


def evaluate_cost_envelopes(
    artifact: Mapping[str, Any],
    champion_artifact: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    triggers: list[str] = []

    artifact_cost = artifact.get("metrics", {}).get("cost", {})
    champion_cost = champion_artifact.get("metrics", {}).get("cost", {})
    if not isinstance(artifact_cost, Mapping) or not isinstance(champion_cost, Mapping):
        return {
            "outcome": "needs_human_decision",
            "reasons": ["non_numeric_metric_family:cost"],
            "triggers": triggers,
        }

    for metric_name, rule in config.get("cost_envelopes", {}).items():
        candidate = artifact_cost.get(rule["metric"])
        champion = champion_cost.get(rule["metric"])
        if not isinstance(candidate, (int, float)) or not isinstance(champion, (int, float)):
            return {
                "outcome": "needs_human_decision",
                "reasons": [f"missing_cost_metric:{rule['metric']}"],
                "triggers": triggers,
            }
        if champion <= 0:
            return {
                "outcome": "needs_human_decision",
                "reasons": [f"invalid_champion_cost_metric:{rule['metric']}"],
                "triggers": triggers,
            }

        relative_increase = (candidate - champion) / champion
        borderline_limit = float(rule.get("borderline_relative_increase", 0.0))
        hard_limit = float(rule.get("max_relative_increase", 0.0))
        if relative_increase > borderline_limit:
            reasons.append(
                f"cost_envelope:{metric_name}:relative_increase_exceeded:{relative_increase:.6f}"
            )
            return {"outcome": "discard", "reasons": reasons, "triggers": triggers}
        if relative_increase > hard_limit:
            reasons.append(
                f"cost_envelope:{metric_name}:borderline_relative_increase:{relative_increase:.6f}"
            )
            triggers.append("borderline_replay_eligible")
            return {
                "outcome": "needs_human_decision",
                "reasons": reasons,
                "triggers": triggers,
            }

        reasons.append(
            f"cost_envelope:{metric_name}:within_limit:{relative_increase:.6f}"
        )

    return {"outcome": "pass", "reasons": reasons, "triggers": triggers}


def compare_numeric_family(
    *,
    artifact_family: Any,
    champion_family: Any,
) -> list[tuple[str, float, float]] | None:
    if not isinstance(artifact_family, Mapping) or not isinstance(champion_family, Mapping):
        return None

    comparisons: list[tuple[str, float, float]] = []
    for metric in sorted(champion_family):
        champion_value = champion_family.get(metric)
        candidate_value = artifact_family.get(metric)
        if not isinstance(champion_value, (int, float)) or not isinstance(candidate_value, (int, float)):
            return None
        comparisons.append((metric, float(candidate_value), float(champion_value)))
    return comparisons


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
