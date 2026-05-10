"""Aggregation and comparison helpers for evaluation metric families."""

from __future__ import annotations

from typing import Mapping

from .metrics import continual_learning_metrics


def aggregate_scalar_metrics(points: list[Mapping[str, float]]) -> dict[str, float]:
    """Compute mean values across a sequence of scalar metric dictionaries."""

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for point in points:
        for key, value in point.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1

    return {
        key: totals[key] / counts[key]
        for key in sorted(totals)
        if counts.get(key, 0) > 0
    }


def compare_metric_families(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    direction: str,
) -> dict[str, float]:
    """Return signed candidate-minus-baseline deltas with direction awareness."""

    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be 'maximize' or 'minimize'")

    deltas: dict[str, float] = {}
    for key, candidate_value in candidate.items():
        baseline_value = baseline.get(key)
        if not isinstance(candidate_value, (int, float)) or not isinstance(
            baseline_value, (int, float)
        ):
            continue
        raw_delta = float(candidate_value) - float(baseline_value)
        deltas[key] = raw_delta if direction == "maximize" else -raw_delta
    return deltas


def build_continual_learning_family(
    *,
    target_exact_match: float,
    anchor_exact_match: float,
    joint_success_rate: float,
    baseline_target_exact_match: float | None = None,
    baseline_anchor_exact_match: float | None = None,
    num_update_episodes: int | None = None,
) -> dict[str, float]:
    """Materialize the canonical CL metric family for one bounded run."""

    return continual_learning_metrics(
        target_exact_match=target_exact_match,
        anchor_exact_match=anchor_exact_match,
        joint_success_rate=joint_success_rate,
        baseline_target_exact_match=baseline_target_exact_match,
        baseline_anchor_exact_match=baseline_anchor_exact_match,
        num_update_episodes=num_update_episodes,
    )
