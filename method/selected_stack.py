"""Helpers for resolving the selected implementation pilot stack."""

from __future__ import annotations

import json
from pathlib import Path


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
