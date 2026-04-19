"""Evaluation utilities for the autoresearch continual-learning harness."""
"""Evaluation helpers for the continual-learning harness."""

from .aggregates import aggregate_scalar_metrics, compare_metric_families
from .metrics import MetricPoint, exact_match_rate, interference_score, mean_token_nll
from .runner import EvalBatch, EvalExample, EvalResult, evaluate_examples, load_eval_batch

__all__ = [
    "EvalBatch",
    "EvalExample",
    "EvalResult",
    "MetricPoint",
    "aggregate_scalar_metrics",
    "compare_metric_families",
    "evaluate_examples",
    "exact_match_rate",
    "interference_score",
    "load_eval_batch",
    "mean_token_nll",
]
