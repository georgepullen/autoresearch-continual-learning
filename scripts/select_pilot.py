#!/usr/bin/env python3
"""Rank model/surface probe outputs under the pilot-selection policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from method.adapter_surface import SURFACE_LIBRARY, get_surface


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, help="Probe result JSON file.")
    parser.add_argument("--output", help="Optional output JSON path.")
    args = parser.parse_args()

    records = [load_json(Path(path)) for path in args.result]
    ranked = [rank_record(record) for record in records]
    ranked.sort(key=lambda item: item["ranking_key"])

    payload = {
        "schema_version": 1,
        "status": selection_status(ranked),
        "ranking_policy": [
            "3090_fit",
            "short_update_stability",
            "visible_dev_profile",
            "throughput",
            "simplicity",
        ],
        "ranked_candidates": ranked,
        "selected_pair": selected_pair(ranked),
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    print(text)
    return 0


def rank_record(record: dict[str, Any]) -> dict[str, Any]:
    status = record.get("status")
    probe = record.get("probe", {})
    measurements = record.get("measurements", {})
    model_id = probe.get("model_id")
    surface_name = probe.get("surface_name")
    surface = get_surface(surface_name) if isinstance(surface_name, str) else None

    fits_3090 = status in {"ok", "completed_with_invalid_rehearsals"}
    invalid_rate = float(record.get("invalid_run_rate", 1.0))
    visible_dev_score = probe.get("visible_dev_score")
    throughput = float(measurements.get("mean_train_tokens_per_second", 0.0))
    simplicity = surface.simplicity_rank if surface else len(SURFACE_LIBRARY) + 1

    return {
        "model_id": model_id,
        "surface_name": surface_name,
        "status": status,
        "fits_3090": fits_3090,
        "invalid_run_rate": invalid_rate,
        "visible_dev_score": visible_dev_score,
        "mean_train_tokens_per_second": throughput,
        "simplicity_rank": simplicity,
        "ranking_key": [
            0 if fits_3090 else 1,
            invalid_rate,
            0 if isinstance(visible_dev_score, (int, float)) else 1,
            -(float(visible_dev_score) if isinstance(visible_dev_score, (int, float)) else 0.0),
            -throughput,
            simplicity,
        ],
    }


def selection_status(ranked: list[dict[str, Any]]) -> str:
    if not ranked:
        return "no_candidates"
    if any(item["fits_3090"] for item in ranked) and not any(
        isinstance(item["visible_dev_score"], (int, float)) for item in ranked
    ):
        return "pending_visible_dev_evidence"
    return "ready_for_selection"


def selected_pair(ranked: list[dict[str, Any]]) -> dict[str, str] | None:
    if selection_status(ranked) != "ready_for_selection":
        return None
    winner = ranked[0]
    return {
        "model_id": winner["model_id"],
        "surface_name": winner["surface_name"],
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    raise SystemExit(main())
