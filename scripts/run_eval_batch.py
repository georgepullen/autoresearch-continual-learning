#!/usr/bin/env python3
"""Run one evaluation batch with a pluggable predictor callable."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.runner import evaluate_examples, load_eval_batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True, help="Path to the eval pack JSON/YAML payload.")
    parser.add_argument(
        "--predictor",
        required=True,
        help="Dotted path to a predictor callable, e.g. package.module:function",
    )
    parser.add_argument(
        "--baseline-anchor-scorer",
        help="Optional dotted path to an anchor baseline score callable.",
    )
    parser.add_argument("--output", help="Optional path to write the result JSON.")
    args = parser.parse_args()

    batch = load_eval_batch(args.pack)
    predict_fn = load_callable(args.predictor)
    baseline_anchor_score_fn = (
        load_callable(args.baseline_anchor_scorer) if args.baseline_anchor_scorer else None
    )

    result = evaluate_examples(
        batch,
        predict_fn=predict_fn,
        score_fn=default_exact_match_score,
        baseline_anchor_score_fn=baseline_anchor_score_fn,
    )
    payload = {
        "schema_version": 1,
        "pack_id": batch.pack_id,
        "target_quality": result.target_quality,
        "interference": result.interference,
        "predictions": list(result.predictions),
        "per_example": list(result.per_example),
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    print(text)
    return 0


def load_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError(f"callable spec must look like module.submodule:function, got {spec!r}")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ValueError(f"{spec!r} did not resolve to a callable")
    return function


def default_exact_match_score(prediction: str, target: str) -> float:
    return 0.0 if normalize_text(prediction) == normalize_text(target) else 1.0


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


if __name__ == "__main__":
    raise SystemExit(main())
