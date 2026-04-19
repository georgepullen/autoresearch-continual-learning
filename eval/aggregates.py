"""Aggregation and comparison helpers for evaluation metric families."""

from __future__ import annotations

from typing import Mapping


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
