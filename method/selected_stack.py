"""Helpers for resolving model/surface stacks and Qwen-family lanes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_selected_pilot_pair(repo_root: str | Path | None = None) -> dict[str, str]:
    """Return the selected base-model/surface pair from the protocol record."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    payload = json.loads((root / "protocol" / "MODEL_SURFACE_PILOT.yaml").read_text())
    selected = payload.get("selected_pair")
    if not isinstance(selected, dict):
        raise ValueError("protocol/MODEL_SURFACE_PILOT.yaml has no selected_pair record")

    model_id = selected.get("model_id")
    surface_name = selected.get("surface_name")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("selected_pair.model_id must be a non-empty string")
    if not isinstance(surface_name, str) or not surface_name.strip():
        raise ValueError("selected_pair.surface_name must be a non-empty string")

    return {
        "model_id": model_id,
        "surface_name": surface_name,
    }


def load_model_lanes(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Return the explicit model-lane protocol record."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    path = root / "protocol" / "MODEL_LANES.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing model lane protocol file: {path}")
    payload = json.loads(path.read_text())
    lanes = payload.get("lanes")
    if not isinstance(lanes, dict) or not lanes:
        raise ValueError("protocol/MODEL_LANES.yaml has no lanes record")
    return payload


def load_lane_pair(
    lane_name: str,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return one model lane with validated model and surface fields."""

    payload = load_model_lanes(repo_root)
    lanes = payload["lanes"]
    lane = lanes.get(lane_name)
    if not isinstance(lane, dict):
        known = ", ".join(sorted(str(name) for name in lanes))
        raise KeyError(f"unknown model lane {lane_name!r}; known lanes: {known}")

    model_id = lane.get("model_id")
    surface_name = lane.get("surface_name")
    run_class = lane.get("run_class")
    comparison_scope = lane.get("comparison_scope")
    for field_name, value in (
        ("model_id", model_id),
        ("surface_name", surface_name),
        ("run_class", run_class),
        ("comparison_scope", comparison_scope),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"lane {lane_name!r} has invalid {field_name}")

    return {
        **lane,
        "name": lane_name,
        "model_id": model_id,
        "surface_name": surface_name,
        "run_class": run_class,
        "comparison_scope": comparison_scope,
    }


def load_default_surrogate_lane(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Return the configured fast development/surrogate lane."""

    payload = load_model_lanes(repo_root)
    lane_name = payload.get("default_surrogate_lane")
    if not isinstance(lane_name, str) or not lane_name.strip():
        raise ValueError("protocol/MODEL_LANES.yaml missing default_surrogate_lane")
    return load_lane_pair(lane_name, repo_root)


def load_champion_lane(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Return the configured canonical champion lane."""

    payload = load_model_lanes(repo_root)
    lane_name = payload.get("champion_lane")
    if not isinstance(lane_name, str) or not lane_name.strip():
        raise ValueError("protocol/MODEL_LANES.yaml missing champion_lane")
    return load_lane_pair(lane_name, repo_root)
