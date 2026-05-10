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


VALID_OUTCOMES = {
    "promote",
    "discard",
    "invalid",
    "needs_human_decision",
    "surrogate_pass",
}


@dataclass(frozen=True)
class DecisionResult:
    outcome: str
    reasons: tuple[str, ...]
    triggers: tuple[str, ...]
    baseline_acceptance_tier: str | None = None


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
        "--bootstrap-config",
        default="protocol/BOOTSTRAP.yaml",
        help="Path to bootstrap config (JSON-shaped YAML).",
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
    bootstrap_config = load_json(resolve_path(args.bootstrap_config))
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
        bootstrap_config=bootstrap_config,
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
    if decision.baseline_acceptance_tier is not None:
        payload["baseline_acceptance_tier"] = decision.baseline_acceptance_tier
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
    bootstrap_config: Mapping[str, Any],
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
        bootstrap_eval = evaluate_bootstrap_eligibility(
            artifact=artifact,
            spec=spec,
            bootstrap_config=bootstrap_config,
        )
        reasons.extend(bootstrap_eval["reasons"])
        triggers.extend(bootstrap_eval["triggers"])
        if bootstrap_eval["outcome"] != "pass":
            return DecisionResult(
                outcome=bootstrap_eval["outcome"],
                reasons=tuple(reasons),
                triggers=tuple(triggers),
            )
        confirmation_gate = evaluate_confirmation_gate(
            artifact=artifact,
            confirmation_result=confirmation_result,
            config=config,
            reasons=reasons,
            triggers=triggers,
            enforce_quality_gates=False,
        )
        if confirmation_gate is not None:
            return confirmation_gate
        bootstrap_seed_eval = evaluate_bootstrap_seed_gates(
            artifact=artifact,
            confirmation_result=confirmation_result,
            config=config,
        )
        reasons.extend(bootstrap_seed_eval["reasons"])
        triggers.extend(bootstrap_seed_eval["triggers"])
        if bootstrap_seed_eval["outcome"] != "pass":
            return DecisionResult(
                outcome=bootstrap_seed_eval["outcome"],
                reasons=tuple(reasons),
                triggers=tuple(triggers),
            )
        acceptance_eval = evaluate_baseline_acceptance_tier(
            artifact=artifact,
            confirmation_result=confirmation_result,
            config=config,
        )
        reasons.extend(acceptance_eval["reasons"])
        triggers.extend(acceptance_eval["triggers"])
        if acceptance_eval["outcome"] != "pass":
            return DecisionResult(
                outcome=acceptance_eval["outcome"],
                reasons=tuple(reasons),
                triggers=tuple(triggers),
            )
        reasons.append("approved_bootstrap_baseline_confirmed")
        return DecisionResult(
            outcome="promote",
            reasons=tuple(reasons),
            triggers=tuple(triggers),
            baseline_acceptance_tier=acceptance_eval["tier"],
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

    comparison_context_eval = evaluate_comparison_context(
        artifact=artifact,
        champion_artifact=champion_artifact,
    )
    if comparison_context_eval is not None:
        reasons.extend(comparison_context_eval["reasons"])
        triggers.extend(comparison_context_eval["triggers"])
        return DecisionResult(
            outcome=comparison_context_eval["outcome"],
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

    confirmation_gate = evaluate_confirmation_gate(
        artifact=artifact,
        confirmation_result=confirmation_result,
        config=config,
        reasons=reasons,
        triggers=triggers,
        enforce_quality_gates=True,
    )
    if confirmation_gate is not None:
        return confirmation_gate

    if is_baseline_method_family(artifact):
        acceptance_eval = evaluate_baseline_acceptance_tier(
            artifact=artifact,
            confirmation_result=confirmation_result,
            config=config,
        )
        reasons.extend(acceptance_eval["reasons"])
        triggers.extend(acceptance_eval["triggers"])
        if acceptance_eval["outcome"] != "pass":
            return DecisionResult(
                outcome=acceptance_eval["outcome"],
                reasons=tuple(reasons),
                triggers=tuple(triggers),
            )
        reasons.append("all_integrity_and_promotion_gates_passed")
        return DecisionResult(
            outcome="promote",
            reasons=tuple(reasons),
            triggers=tuple(triggers),
            baseline_acceptance_tier=acceptance_eval["tier"],
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
            reasons.append(f"sentinel_message:{finding.code}:{finding.message}")

    return reasons


def evaluate_bootstrap_eligibility(
    *,
    artifact: Mapping[str, Any],
    spec: Mapping[str, Any],
    bootstrap_config: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    triggers: list[str] = []

    bootstrap_lane = bootstrap_config.get("bootstrap_lane", {})
    if not isinstance(bootstrap_lane, Mapping):
        return {
            "outcome": "needs_human_decision",
            "reasons": ["missing_champion_context", "bootstrap_lane_missing"],
            "triggers": triggers,
        }

    approved_methods = bootstrap_lane.get("approved_method_families", [])
    allowed_run_classes = bootstrap_lane.get("allowed_run_classes", [])
    required_base_model = bootstrap_lane.get("required_base_model")
    required_pack = bootstrap_lane.get("required_development_pack")
    required_confirm_summary = bootstrap_lane.get("required_confirmation_pack_summary")
    required_generator = bootstrap_lane.get("required_train_generator_version")

    method_family = artifact.get("method", {}).get("method_family")
    run_class = artifact.get("run", {}).get("run_class")
    base_model = artifact.get("method", {}).get("base_model")
    development_pack = artifact.get("data", {}).get("development_pack")
    confirmation_pack_summary = artifact.get("data", {}).get("confirmation_pack_summary")
    train_generator_version = artifact.get("data", {}).get("train_generator_version")

    if not isinstance(method_family, str) or method_family not in approved_methods:
        reasons.extend(
            [
                "missing_champion_context",
                f"bootstrap_method_not_approved:{method_family!r}",
            ]
        )
        return {"outcome": "needs_human_decision", "reasons": reasons, "triggers": triggers}

    if not isinstance(run_class, str) or run_class not in allowed_run_classes:
        reasons.extend(
            [
                "missing_champion_context",
                f"bootstrap_run_class_not_approved:{run_class!r}",
            ]
        )
        return {"outcome": "needs_human_decision", "reasons": reasons, "triggers": triggers}

    if isinstance(required_base_model, str) and required_base_model and base_model != required_base_model:
        reasons.extend(
            [
                "missing_champion_context",
                "bootstrap_base_model_mismatch",
            ]
        )
        return {"outcome": "needs_human_decision", "reasons": reasons, "triggers": triggers}

    if isinstance(required_pack, str) and required_pack and development_pack != required_pack:
        reasons.extend(
            [
                "missing_champion_context",
                "bootstrap_visible_pack_mismatch",
            ]
        )
        return {"outcome": "needs_human_decision", "reasons": reasons, "triggers": triggers}

    if (
        isinstance(required_confirm_summary, str)
        and required_confirm_summary
        and confirmation_pack_summary != required_confirm_summary
    ):
        reasons.extend(
            [
                "missing_champion_context",
                "bootstrap_confirmation_summary_mismatch",
            ]
        )
        return {"outcome": "needs_human_decision", "reasons": reasons, "triggers": triggers}

    if isinstance(required_generator, str) and required_generator and train_generator_version != required_generator:
        reasons.extend(
            [
                "missing_champion_context",
                "bootstrap_train_generator_mismatch",
            ]
        )
        return {"outcome": "needs_human_decision", "reasons": reasons, "triggers": triggers}

    if spec.get("baseline_ref") not in {None, "", "bootstrap_baseline"}:
        reasons.extend(
            [
                "missing_champion_context",
                "bootstrap_baseline_ref_must_not_target_existing_champion",
            ]
        )
        return {"outcome": "needs_human_decision", "reasons": reasons, "triggers": triggers}

    reasons.append("approved_bootstrap_baseline_run_without_existing_champion")
    return {"outcome": "pass", "reasons": reasons, "triggers": triggers}


def evaluate_confirmation_gate(
    *,
    artifact: Mapping[str, Any],
    confirmation_result: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    reasons: list[str],
    triggers: list[str],
    enforce_quality_gates: bool = True,
) -> DecisionResult | None:
    expected_pack_id = artifact.get("data", {}).get("confirmation_pack_summary")
    confirmation_eval = evaluate_confirmation_result(
        confirmation_result,
        expected_pack_id=expected_pack_id if isinstance(expected_pack_id, str) and expected_pack_id else None,
    )
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
    if enforce_quality_gates:
        quality_eval = evaluate_confirmation_quality_gates(
            confirmation_result=confirmation_result,
            config=config,
        )
        reasons.extend(quality_eval["reasons"])
        triggers.extend(quality_eval["triggers"])
        if quality_eval["outcome"] != "pass":
            return DecisionResult(
                outcome=quality_eval["outcome"],
                reasons=tuple(reasons),
                triggers=tuple(triggers),
            )
    return None


def evaluate_confirmation_quality_gates(
    *,
    confirmation_result: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config.get("confirmation_quality_gates", {})
    if not isinstance(gates, Mapping) or not gates:
        return {"outcome": "pass", "reasons": [], "triggers": []}
    if confirmation_result is None:
        return {
            "outcome": "needs_human_decision",
            "reasons": ["confirmation_quality_payload_missing"],
            "triggers": [],
        }

    aggregate_metrics = confirmation_result.get("aggregate_metrics", {})
    if not isinstance(aggregate_metrics, Mapping):
        return {
            "outcome": "needs_human_decision",
            "reasons": ["confirmation_quality_aggregate_metrics_missing"],
            "triggers": [],
        }

    target_quality = aggregate_metrics.get("target_quality", {})
    interference = aggregate_metrics.get("interference", {})
    baseline_reference = aggregate_metrics.get("baseline_reference", {})
    baseline_target_quality = (
        baseline_reference.get("target_quality", {})
        if isinstance(baseline_reference, Mapping)
        else {}
    )
    baseline_interference = (
        baseline_reference.get("interference", {})
        if isinstance(baseline_reference, Mapping)
        else {}
    )

    confirmation_target_exact = safe_metric(target_quality, "exact_match")
    confirmation_anchor_exact = safe_metric(interference, "anchor_exact_match")
    confirmation_joint_success = safe_metric(interference, "joint_success_rate")
    baseline_target_exact = safe_metric(baseline_target_quality, "exact_match")
    baseline_anchor_exact = safe_metric(baseline_interference, "anchor_exact_match")

    required_metrics = {
        "confirmation_target_exact": confirmation_target_exact,
        "confirmation_anchor_exact": confirmation_anchor_exact,
        "confirmation_joint_success": confirmation_joint_success,
        "baseline_target_exact": baseline_target_exact,
        "baseline_anchor_exact": baseline_anchor_exact,
    }
    missing = [name for name, value in required_metrics.items() if value is None]
    if missing:
        return {
            "outcome": "needs_human_decision",
            "reasons": [f"missing_confirmation_quality_metric:{name}" for name in missing],
            "triggers": [],
        }

    failures: list[str] = []
    target_floor = max(
        float(gates.get("target_exact_min", 0.0)),
        baseline_target_exact
        + float(gates.get("target_exact_delta_from_baseline_reference", 0.0)),
    )
    if confirmation_target_exact < target_floor:
        failures.append(
            f"confirmation_target_exact_below_floor:{confirmation_target_exact:.4f}<{target_floor:.4f}"
        )

    anchor_floor = max(
        float(gates.get("anchor_exact_min", 0.0)),
        baseline_anchor_exact
        + float(gates.get("anchor_exact_delta_from_baseline_reference", 0.0)),
    )
    if confirmation_anchor_exact < anchor_floor:
        failures.append(
            f"confirmation_anchor_exact_below_floor:{confirmation_anchor_exact:.4f}<{anchor_floor:.4f}"
        )

    joint_floor = float(gates.get("joint_success_min", 0.0))
    if confirmation_joint_success < joint_floor:
        failures.append(
            f"confirmation_joint_success_below_floor:{confirmation_joint_success:.4f}<{joint_floor:.4f}"
        )

    if failures:
        return {
            "outcome": "discard",
            "reasons": failures,
            "triggers": ["confirm_regression_pattern"],
        }

    return {
        "outcome": "pass",
        "reasons": ["confirmation_quality_gates_passed"],
        "triggers": [],
    }


def evaluate_bootstrap_seed_gates(
    *,
    artifact: Mapping[str, Any],
    confirmation_result: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    triggers: list[str] = []
    gates = config.get("bootstrap_seed_gates", {})
    if not isinstance(gates, Mapping):
        return {"outcome": "pass", "reasons": reasons, "triggers": triggers}
    if confirmation_result is None:
        return {
            "outcome": "needs_human_decision",
            "reasons": ["missing_bootstrap_confirmation_payload"],
            "triggers": triggers,
        }

    aggregate_metrics = confirmation_result.get("aggregate_metrics", {})
    if not isinstance(aggregate_metrics, Mapping):
        return {
            "outcome": "needs_human_decision",
            "reasons": ["missing_bootstrap_confirmation_aggregate_metrics"],
            "triggers": triggers,
        }

    target_quality = aggregate_metrics.get("target_quality", {})
    interference = aggregate_metrics.get("interference", {})
    by_probe_family = aggregate_metrics.get("by_probe_family", {})
    baseline_reference = aggregate_metrics.get("baseline_reference", {})

    confirmation_target_exact = safe_metric(target_quality, "exact_match")
    confirmation_anchor_exact = safe_metric(interference, "anchor_exact_match")
    base_target_exact = safe_metric(
        baseline_reference.get("target_quality", {})
        if isinstance(baseline_reference, Mapping)
        else {},
        "exact_match",
    )
    latest_value_exact = probe_family_metric(
        by_probe_family,
        probe_families=metric_aliases(
            gates,
            "latest_value_probe_families",
            ("latest_state_after_repeated_overwrite",),
        ),
        metric_name="target_exact_match",
    )
    visible_dev_target_exact = safe_metric(artifact.get("metrics", {}).get("target_quality", {}), "exact_match")

    required_metrics = {
        "confirmation_target_exact": confirmation_target_exact,
        "confirmation_anchor_exact": confirmation_anchor_exact,
        "base_target_exact": base_target_exact,
        "latest_value_exact": latest_value_exact,
        "visible_dev_target_exact": visible_dev_target_exact,
    }
    missing = [name for name, value in required_metrics.items() if value is None]
    if missing:
        return {
            "outcome": "needs_human_decision",
            "reasons": [f"missing_bootstrap_metric:{name}" for name in missing],
            "triggers": triggers,
        }

    min_target_exact = max(
        float(gates.get("target_exact_min_abs", 0.35)),
        base_target_exact + float(gates.get("target_exact_delta_from_base", 0.25)),
    )
    if confirmation_target_exact < min_target_exact:
        reasons.append(
            f"bootstrap_target_exact_below_floor:{confirmation_target_exact:.4f}<{min_target_exact:.4f}"
        )
    if latest_value_exact < float(gates.get("latest_value_exact_min", 0.40)):
        reasons.append(
            "bootstrap_latest_value_exact_below_floor:"
            f"{latest_value_exact:.4f}<{float(gates.get('latest_value_exact_min', 0.40)):.4f}"
        )
    if confirmation_anchor_exact < float(gates.get("anchor_exact_min", 0.25)):
        reasons.append(
            "bootstrap_anchor_exact_below_floor:"
            f"{confirmation_anchor_exact:.4f}<{float(gates.get('anchor_exact_min', 0.25)):.4f}"
        )

    dev_confirm_gap = visible_dev_target_exact - confirmation_target_exact
    max_gap = float(gates.get("visible_confirm_gap_max", 0.25))
    if dev_confirm_gap > max_gap:
        reasons.append(
            f"bootstrap_visible_confirm_gap_too_large:{dev_confirm_gap:.4f}>{max_gap:.4f}"
        )

    if reasons:
        triggers.append("confirm_regression_pattern")
        return {"outcome": "discard", "reasons": reasons, "triggers": triggers}

    reasons.append("bootstrap_seed_gates_passed")
    return {"outcome": "pass", "reasons": reasons, "triggers": triggers}


def evaluate_baseline_acceptance_tier(
    *,
    artifact: Mapping[str, Any],
    confirmation_result: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    triggers: list[str] = []
    gates = config.get("baseline_acceptance_gates", {})
    if not isinstance(gates, Mapping):
        return {
            "outcome": "pass",
            "reasons": ["baseline_acceptance_policy_missing_defaulting_to_provisional"],
            "triggers": triggers,
            "tier": "provisional",
        }
    if confirmation_result is None:
        return {
            "outcome": "needs_human_decision",
            "reasons": ["missing_bootstrap_confirmation_payload"],
            "triggers": triggers,
        }

    aggregate_metrics = confirmation_result.get("aggregate_metrics", {})
    if not isinstance(aggregate_metrics, Mapping):
        return {
            "outcome": "needs_human_decision",
            "reasons": ["missing_bootstrap_confirmation_aggregate_metrics"],
            "triggers": triggers,
        }

    target_quality = aggregate_metrics.get("target_quality", {})
    interference = aggregate_metrics.get("interference", {})
    by_probe_family = aggregate_metrics.get("by_probe_family", {})

    confirmation_target_exact = safe_metric(target_quality, "exact_match")
    confirmation_anchor_exact = safe_metric(interference, "anchor_exact_match")
    latest_value_target_exact = probe_family_metric(
        by_probe_family,
        probe_families=metric_aliases(
            gates,
            "latest_value_probe_families",
            ("latest_state_after_repeated_overwrite",),
        ),
        metric_name="target_exact_match",
    )

    anchor_by_probe = extract_anchor_exact_by_probe_family(by_probe_family)
    neighbor_anchor_exact = first_available_metric(
        anchor_by_probe,
        metric_aliases(
            gates,
            "neighbor_anchor_probe_families",
            ("neighbor_retention_same_namespace",),
        ),
    )
    delayed_anchor_exact = first_available_metric(
        anchor_by_probe,
        metric_aliases(
            gates,
            "delayed_anchor_probe_families",
            ("delayed_retention_after_unrelated_updates",),
        ),
    )

    required_metrics = {
        "confirmation_target_exact": confirmation_target_exact,
        "confirmation_anchor_exact": confirmation_anchor_exact,
        "latest_value_target_exact": latest_value_target_exact,
        "neighbor_anchor_exact": neighbor_anchor_exact,
        "delayed_anchor_exact": delayed_anchor_exact,
    }
    missing = [name for name, value in required_metrics.items() if value is None]
    if missing:
        return {
            "outcome": "needs_human_decision",
            "reasons": [f"missing_accepted_baseline_metric:{name}" for name in missing],
            "triggers": triggers,
        }

    accepted_failures: list[str] = []
    if confirmation_target_exact < float(gates.get("confirmation_target_exact_min", 0.75)):
        accepted_failures.append(
            "accepted_baseline_target_exact_below_floor:"
            f"{confirmation_target_exact:.4f}<{float(gates.get('confirmation_target_exact_min', 0.75)):.4f}"
        )
    if latest_value_target_exact < float(gates.get("latest_value_target_exact_min", 0.75)):
        accepted_failures.append(
            "accepted_baseline_latest_value_target_exact_below_floor:"
            f"{latest_value_target_exact:.4f}<{float(gates.get('latest_value_target_exact_min', 0.75)):.4f}"
        )
    if confirmation_anchor_exact < float(gates.get("anchor_exact_min", 0.4167)):
        accepted_failures.append(
            "accepted_baseline_anchor_exact_below_floor:"
            f"{confirmation_anchor_exact:.4f}<{float(gates.get('anchor_exact_min', 0.4167)):.4f}"
        )
    if neighbor_anchor_exact < float(gates.get("neighbor_anchor_exact_min", 0.50)):
        accepted_failures.append(
            "accepted_baseline_neighbor_anchor_exact_below_floor:"
            f"{neighbor_anchor_exact:.4f}<{float(gates.get('neighbor_anchor_exact_min', 0.50)):.4f}"
        )
    if delayed_anchor_exact < float(gates.get("delayed_anchor_exact_min", 0.25)):
        accepted_failures.append(
            "accepted_baseline_delayed_anchor_exact_below_floor:"
            f"{delayed_anchor_exact:.4f}<{float(gates.get('delayed_anchor_exact_min', 0.25)):.4f}"
        )

    min_anchor_slice_exact = min(anchor_by_probe.values()) if anchor_by_probe else None
    if min_anchor_slice_exact is None:
        return {
            "outcome": "needs_human_decision",
            "reasons": ["missing_accepted_baseline_metric:min_anchor_slice_exact"],
            "triggers": triggers,
        }
    if min_anchor_slice_exact < float(gates.get("min_anchor_slice_exact_min", 0.25)):
        accepted_failures.append(
            "accepted_baseline_min_anchor_slice_exact_below_floor:"
            f"{min_anchor_slice_exact:.4f}<{float(gates.get('min_anchor_slice_exact_min', 0.25)):.4f}"
        )

    if bool(gates.get("require_no_zero_anchor_slice", True)):
        zero_anchor_slices = sorted(
            probe_family
            for probe_family, exact_match in anchor_by_probe.items()
            if exact_match <= 0.0
        )
        for probe_family in zero_anchor_slices:
            accepted_failures.append(
                f"accepted_baseline_zero_anchor_slice:{probe_family}"
            )

    if accepted_failures:
        reasons.append("baseline_acceptance_tier_provisional")
        reasons.extend(accepted_failures)
        return {
            "outcome": "pass",
            "reasons": reasons,
            "triggers": triggers,
            "tier": str(gates.get("default_tier", "provisional")),
        }

    reasons.append("accepted_baseline_gates_passed")
    return {
        "outcome": "pass",
        "reasons": reasons,
        "triggers": triggers,
        "tier": str(gates.get("accepted_tier", "accepted")),
    }


def is_baseline_method_family(artifact: Mapping[str, Any]) -> bool:
    method_family = artifact.get("method", {}).get("method_family")
    return isinstance(method_family, str) and method_family.startswith("baseline_")


def evaluate_comparison_context(
    *,
    artifact: Mapping[str, Any],
    champion_artifact: Mapping[str, Any],
) -> dict[str, Any] | None:
    reasons: list[str] = []
    triggers: list[str] = []

    candidate_comparison = artifact.get("comparison", {})
    champion_comparison = champion_artifact.get("comparison", {})
    if candidate_comparison.get("comparison_scope") != champion_comparison.get("comparison_scope"):
        reasons.append("comparison_context_mismatch:comparison_scope")

    candidate_data = artifact.get("data", {})
    champion_data = champion_artifact.get("data", {})
    for field in (
        "train_generator_version",
        "development_pack",
        "confirmation_pack_summary",
    ):
        if candidate_data.get(field) != champion_data.get(field):
            reasons.append(f"comparison_context_mismatch:{field}")

    if reasons:
        triggers.append("comparison_context_refresh_required")
        return {
            "outcome": "needs_human_decision",
            "reasons": reasons,
            "triggers": triggers,
        }
    return None


def evaluate_scientific_floors(
    artifact: Mapping[str, Any],
    champion_artifact: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    triggers: list[str] = []

    for rule_name, rule in config.get("scientific_floors", {}).items():
        family = rule["family"]
        metric_name = rule.get("metric")
        direction = rule.get("direction", "maximize")
        tolerance = float(rule.get("tolerance", 0.0))
        borderline = float(rule.get("borderline_tolerance", tolerance))
        comparisons = resolve_rule_comparisons(
            artifact=artifact,
            champion_artifact=champion_artifact,
            family=family,
            metric_name=metric_name,
        )
        if comparisons is None:
            missing = metric_name or family
            return {
                "outcome": "needs_human_decision",
                "reasons": [f"non_numeric_metric_family:{missing}"],
                "triggers": triggers,
            }

        for metric, candidate, champion in comparisons:
            signed_delta = directional_delta(
                candidate=float(candidate),
                champion=float(champion),
                direction=direction,
            )
            if signed_delta < -borderline:
                reasons.append(
                    f"{rule_name}:{metric}:candidate_regressed_beyond_floor:{signed_delta:.6f}"
                )
                return {"outcome": "discard", "reasons": reasons, "triggers": triggers}
            if signed_delta < -tolerance:
                reasons.append(
                    f"{rule_name}:{metric}:candidate_is_borderline_below_floor:{signed_delta:.6f}"
                )
                triggers.append("borderline_replay_eligible")
                return {
                    "outcome": "needs_human_decision",
                    "reasons": reasons,
                    "triggers": triggers,
                }

        reasons.append(f"{rule_name}:metric_floor_satisfied")

    return {"outcome": "pass", "reasons": reasons, "triggers": triggers}


def evaluate_required_wins(
    artifact: Mapping[str, Any],
    champion_artifact: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    triggers: list[str] = []

    improvements: list[tuple[str, str, float]] = []
    borderline_only: list[tuple[str, str, float]] = []

    for rule_name, rule in config.get("required_win_metrics", {}).items():
        family = rule["family"]
        metric_name = rule.get("metric")
        direction = rule.get("direction", "maximize")
        min_improvement = float(rule.get("min_improvement", 0.0))
        borderline = float(rule.get("borderline_tolerance", 0.0))
        comparisons = resolve_rule_comparisons(
            artifact=artifact,
            champion_artifact=champion_artifact,
            family=family,
            metric_name=metric_name,
        )
        if comparisons is None:
            missing = metric_name or family
            return {
                "outcome": "needs_human_decision",
                "reasons": [f"non_numeric_metric_family:{missing}"],
                "triggers": triggers,
            }

        for metric, candidate, champion in comparisons:
            improvement = directional_delta(
                candidate=float(candidate),
                champion=float(champion),
                direction=direction,
            )
            if improvement >= min_improvement:
                improvements.append((rule_name, metric, improvement))
            elif improvement >= borderline:
                borderline_only.append((rule_name, metric, improvement))

    min_required = int(config.get("min_required_improvements", 1))
    if len(improvements) >= min_required:
        reasons.append(
            "required_wins:improved_metrics:"
            + ",".join(
                f"{rule_name}:{metric}={value:.6f}"
                for rule_name, metric, value in improvements
            )
        )
        return {"outcome": "pass", "reasons": reasons, "triggers": triggers}

    if borderline_only:
        reasons.append(
            "required_wins:borderline_only:"
            + ",".join(
                f"{rule_name}:{metric}={value:.6f}"
                for rule_name, metric, value in borderline_only
            )
        )
        triggers.append("borderline_replay_eligible")
        return {
            "outcome": "needs_human_decision",
            "reasons": reasons,
            "triggers": triggers,
        }

    reasons.append("required_wins:no_required_cl_improvement")
    return {"outcome": "discard", "reasons": reasons, "triggers": triggers}


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


def safe_metric(payload: Mapping[str, Any] | None, key: str) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def metric_aliases(
    config: Mapping[str, Any],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    values = config.get(key)
    if isinstance(values, list):
        aliases = tuple(value for value in values if isinstance(value, str) and value)
        if aliases:
            return aliases
    if isinstance(values, str) and values:
        return (values,)
    return default


def first_available_metric(
    metrics_by_name: Mapping[str, float],
    aliases: tuple[str, ...],
) -> float | None:
    for alias in aliases:
        value = metrics_by_name.get(alias)
        if value is not None:
            return value
    return None


def probe_family_metric(
    by_probe_family: Mapping[str, Any] | None,
    *,
    probe_families: tuple[str, ...],
    metric_name: str,
) -> float | None:
    if not isinstance(by_probe_family, Mapping):
        return None
    for probe_family in probe_families:
        metrics = by_probe_family.get(probe_family)
        value = safe_metric(metrics if isinstance(metrics, Mapping) else None, metric_name)
        if value is not None:
            return value
    return None


def extract_anchor_exact_by_probe_family(
    by_probe_family: Mapping[str, Any] | None,
) -> dict[str, float]:
    if not isinstance(by_probe_family, Mapping):
        return {}
    extracted: dict[str, float] = {}
    for probe_family, metrics in by_probe_family.items():
        if not isinstance(probe_family, str):
            continue
        exact_match = safe_metric(metrics if isinstance(metrics, Mapping) else None, "anchor_exact_match")
        if exact_match is not None:
            extracted[probe_family] = exact_match
    return extracted


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


def resolve_rule_comparisons(
    *,
    artifact: Mapping[str, Any],
    champion_artifact: Mapping[str, Any],
    family: str,
    metric_name: str | None,
) -> list[tuple[str, float, float]] | None:
    artifact_family = artifact.get("metrics", {}).get(family, {})
    champion_family = champion_artifact.get("metrics", {}).get(family, {})
    if metric_name:
        if not isinstance(artifact_family, Mapping) or not isinstance(champion_family, Mapping):
            return None
        candidate_value = artifact_family.get(metric_name)
        champion_value = champion_family.get(metric_name)
        if not isinstance(candidate_value, (int, float)) or not isinstance(champion_value, (int, float)):
            return None
        return [(metric_name, float(candidate_value), float(champion_value))]
    return compare_numeric_family(
        artifact_family=artifact_family,
        champion_family=champion_family,
    )


def directional_delta(*, candidate: float, champion: float, direction: str) -> float:
    if direction == "maximize":
        return candidate - champion
    if direction == "minimize":
        return champion - candidate
    raise ValueError(f"Unsupported direction: {direction!r}")


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
