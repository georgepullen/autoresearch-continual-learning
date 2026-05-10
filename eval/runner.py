"""Minimal evaluation runner for bounded visible-dev and smoke checks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from .aggregates import aggregate_scalar_metrics
from .metrics import exact_match_rate, interference_score, mean_token_nll, perplexity_from_nll


PredictFn = Callable[[str], str]
ScoreFn = Callable[[str, str], float]


@dataclass(frozen=True)
class EvalExample:
    """One evaluation example for the bounded harness."""

    example_id: str
    prompt: str
    target: str
    anchor_prompt: str | None = None
    anchor_target: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class EvalBatch:
    """A named batch of evaluation examples."""

    pack_id: str
    examples: tuple[EvalExample, ...]
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class EvalResult:
    """Aggregated evaluation outcome."""

    predictions: tuple[str, ...]
    target_quality: dict[str, float]
    interference: dict[str, float]
    per_example: tuple[dict[str, Any], ...]
    by_probe_family: dict[str, dict[str, float]]
    retention_by_lag: dict[str, dict[str, float]]


def load_eval_batch(path: str | Path) -> EvalBatch:
    """Load a JSON-shaped visible-dev or smoke evaluation batch."""

    payload = json.loads(Path(path).read_text())
    pack_id = str(payload.get("pack_id", "")).strip()
    if not pack_id:
        raise ValueError("eval batch must include a non-empty pack_id")

    examples_payload = payload.get("examples")
    if not isinstance(examples_payload, list) or not examples_payload:
        raise ValueError("eval batch must contain a non-empty examples list")

    examples = tuple(
        EvalExample(
            example_id=str(item["example_id"]),
            prompt=str(item["prompt"]),
            target=str(item["target"]),
            anchor_prompt=(
                str(item["anchor_prompt"]) if item.get("anchor_prompt") is not None else None
            ),
            anchor_target=(
                str(item["anchor_target"]) if item.get("anchor_target") is not None else None
            ),
            metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else None,
        )
        for item in examples_payload
    )
    return EvalBatch(
        pack_id=pack_id,
        examples=examples,
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
    )


def evaluate_examples(
    batch: EvalBatch,
    *,
    predict_fn: PredictFn,
    score_fn: ScoreFn,
    baseline_anchor_score_fn: ScoreFn | None = None,
) -> EvalResult:
    """Evaluate one batch with a simple prediction and scoring interface."""

    predictions: list[str] = []
    anchor_predictions: list[str] = []
    token_losses: list[float] = []
    anchor_losses: list[float] = []
    baseline_anchor_losses: list[float] = []
    joint_successes: list[float] = []
    per_example: list[dict[str, Any]] = []
    by_probe_family_records: dict[str, list[dict[str, Any]]] = {}
    retention_by_lag_records: dict[str, list[dict[str, Any]]] = {}

    for example in batch.examples:
        prediction = predict_fn(example.prompt)
        predictions.append(prediction)

        target_loss = float(score_fn(prediction, example.target))
        token_losses.append(target_loss)

        target_success = 1.0 if target_loss == 0.0 else 0.0
        example_metrics: dict[str, Any] = {
            "target_nll": target_loss,
            "target_exact_match": target_success,
        }
        if example.anchor_target is not None:
            anchor_prompt = example.anchor_prompt or example.prompt
            anchor_prediction = predict_fn(anchor_prompt)
            anchor_predictions.append(anchor_prediction)
            anchor_loss = float(score_fn(anchor_prediction, example.anchor_target))
            anchor_losses.append(anchor_loss)
            anchor_success = 1.0 if anchor_loss == 0.0 else 0.0
            joint_successes.append(1.0 if target_success and anchor_success else 0.0)
            example_metrics["anchor_nll"] = anchor_loss
            example_metrics["anchor_exact_match"] = anchor_success
            if baseline_anchor_score_fn is not None:
                baseline_loss = float(
                    baseline_anchor_score_fn(anchor_prediction, example.anchor_target)
                )
                baseline_anchor_losses.append(baseline_loss)
                example_metrics["baseline_anchor_nll"] = baseline_loss
        per_example.append(example_metrics)

        if example.metadata:
            probe_family = example.metadata.get("probe_family")
            if isinstance(probe_family, str) and probe_family:
                by_probe_family_records.setdefault(probe_family, []).append(example_metrics)

            delay = example.metadata.get("delay_since_last_relevant_update")
            lag_bucket = lag_bucket_name(delay)
            if lag_bucket is not None:
                retention_by_lag_records.setdefault(lag_bucket, []).append(example_metrics)

    target_quality = {
        "exact_match": exact_match_rate(predictions, [item.target for item in batch.examples]),
        "mean_target_nll": mean_token_nll(token_losses),
        "perplexity": perplexity_from_nll(token_losses),
    }

    interference = aggregate_scalar_metrics(per_example)
    if anchor_predictions:
        interference["anchor_exact_match"] = exact_match_rate(
            anchor_predictions,
            [item.anchor_target for item in batch.examples if item.anchor_target is not None],
        )
    if joint_successes:
        interference["joint_success_rate"] = sum(joint_successes) / len(joint_successes)
    if anchor_losses and baseline_anchor_losses:
        interference["anchor_interference_delta"] = interference_score(
            anchor_losses,
            baseline_anchor_losses,
        )

    return EvalResult(
        predictions=tuple(predictions),
        target_quality=target_quality,
        interference=interference,
        per_example=tuple(per_example),
        by_probe_family={
            key: aggregate_scalar_metrics(records)
            for key, records in sorted(by_probe_family_records.items())
        },
        retention_by_lag={
            key: aggregate_scalar_metrics(records)
            for key, records in sorted(retention_by_lag_records.items())
        },
    )


def lag_bucket_name(delay: Any) -> str | None:
    if not isinstance(delay, int) or delay < 0:
        return None
    if delay == 0:
        return "lag_0"
    if delay <= 2:
        return "lag_1_to_2"
    return "lag_3_plus"
