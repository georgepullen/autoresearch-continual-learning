"""Tiny predictor helpers for smoke-testing the eval harness."""

from __future__ import annotations


def keyword_predictor(prompt: str) -> str:
    """Return deterministic short answers for the bootstrap visible-dev pack."""

    lowered = prompt.lower()
    if "france" in lowered:
        return "Paris"
    if "sky" in lowered:
        return "Blue"
    if "triangle" in lowered:
        return "Three"
    return ""


def zero_anchor_score(prediction: str, target: str) -> float:
    """Anchor baseline scorer used only for smoke checks."""

    return 0.0
