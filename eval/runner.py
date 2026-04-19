"""Minimal evaluation runner for bounded visible-dev and smoke checks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Iterable

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
    anchor_target: str | None = None


@dataclass(frozen=True)
class EvalBatch:
    """A named batch of evaluation examples."""

    pack_id: str
    examples: tuple[EvalExample, ...]


@dataclass(frozen=True)
class EvalResult:
    """Aggregated evaluation outcome."""

    predictions: tuple[str, ...]
    target_quality: dict[str, float]
    interference: dict[str, float]
    per_example: tuple[dict[str, float], ...]


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
            anchor_target=(
                str(item["anchor_target"]) if item.get("anchor_target") is not None else None
            ),
        )
        for item in examples_payload
    )
    return EvalBatch(pack_id=pack_id, examples=examples)


def evaluate_examples(
    batch: EvalBatch,
    *,
    predict_fn: PredictFn,
    score_fn: ScoreFn,
    baseline_anchor_score_fn: ScoreFn | None = None,
) -> EvalResult:
    """Evaluate one batch with a simple prediction and scoring interface."""

    predictions: list[str] = []
    token_losses: list[float] = []
    anchor_losses: list[float] = []
    baseline_anchor_losses: list[float] = []
    per_example: list[dict[str, float]] = []

    for example in batch.examples:
        prediction = predict_fn(example.prompt)
        predictions.append(prediction)

        target_loss = float(score_fn(prediction, example.target))
        token_losses.append(target_loss)

        example_metrics = {"target_nll": target_loss}
        if example.anchor_target is not None:
            anchor_loss = float(score_fn(prediction, example.anchor_target))
            anchor_losses.append(anchor_loss)
            example_metrics["anchor_nll"] = anchor_loss
            if baseline_anchor_score_fn is not None:
                baseline_loss = float(baseline_anchor_score_fn(prediction, example.anchor_target))
                baseline_anchor_losses.append(baseline_loss)
                example_metrics["baseline_anchor_nll"] = baseline_loss
        per_example.append(example_metrics)

    target_quality = {
        "exact_match": exact_match_rate(predictions, [item.target for item in batch.examples]),
        "mean_target_nll": mean_token_nll(token_losses),
        "perplexity": perplexity_from_nll(token_losses),
    }

    interference = aggregate_scalar_metrics(per_example)
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
    )
