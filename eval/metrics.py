"""Small evaluation metrics for bounded visible-dev and smoke runs."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from statistics import mean
from typing import Sequence


@dataclass(frozen=True)
class MetricPoint:
    """One named scalar metric."""

    name: str
    value: float


def exact_match_rate(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Compute normalized exact-match rate."""

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    if not targets:
        return 0.0

    matches = 0
    for prediction, target in zip(predictions, targets):
        if normalize_text(prediction) == normalize_text(target):
            matches += 1
    return matches / len(targets)


def mean_token_nll(token_losses: Sequence[float]) -> float:
    """Compute mean negative log-likelihood from token-level losses."""

    if not token_losses:
        return 0.0
    return float(mean(token_losses))


def perplexity_from_nll(token_losses: Sequence[float]) -> float:
    """Convert mean NLL into perplexity for reporting."""

    if not token_losses:
        return 0.0
    return float(exp(mean_token_nll(token_losses)))


def interference_score(anchor_losses: Sequence[float], baseline_anchor_losses: Sequence[float]) -> float:
    """Measure average deterioration against an anchor baseline.

    Positive values mean the candidate interfered more than the baseline.
    Negative values mean the candidate preserved anchors better.
    """

    if len(anchor_losses) != len(baseline_anchor_losses):
        raise ValueError("anchor losses and baseline anchor losses must have equal length")
    if not anchor_losses:
        return 0.0
    deltas = [candidate - baseline for candidate, baseline in zip(anchor_losses, baseline_anchor_losses)]
    return float(mean(deltas))


def normalize_text(text: str) -> str:
    """Normalize short text answers for exact-match comparison."""

    return " ".join(text.strip().lower().split())
