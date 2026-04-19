"""Helpers for protected confirmation metadata and aggregate feedback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIRM_SUMMARY = REPO_ROOT / "data" / "packs" / "confirm_locked_v1.summary.yaml"
DEFAULT_CONFIRM_HASH = REPO_ROOT / "data" / "packs" / "confirm_locked_v1.hash"


@dataclass(frozen=True)
class ConfirmationEvaluation:
    """Normalized protected-confirmation result."""

    status: str
    pack_id: str
    errors: tuple[str, ...]


def load_confirmation_contract(
    summary_path: Path = DEFAULT_CONFIRM_SUMMARY,
    hash_path: Path = DEFAULT_CONFIRM_HASH,
) -> dict[str, Any]:
    """Load the public confirmation-pack contract."""

    return {
      "summary": _load_json(summary_path),
      "hash": _load_json(hash_path),
    }


def evaluate_confirmation_result(
    result: Mapping[str, Any] | None,
    *,
    expected_pack_id: str = "confirm_locked_v1",
) -> ConfirmationEvaluation:
    """Validate a host-local aggregate confirmation result."""

    if result is None:
        return ConfirmationEvaluation(
            status="pending",
            pack_id=expected_pack_id,
            errors=("confirmation_result_missing",),
        )

    errors: list[str] = []
    pack_id = result.get("pack_id")
    if pack_id != expected_pack_id:
        errors.append(
            f"confirmation_pack_id_mismatch: expected {expected_pack_id!r}, got {pack_id!r}"
        )

    status = result.get("status")
    if status not in {"pass", "fail", "pending"}:
        errors.append(
            f"confirmation_status_invalid: expected one of 'pass', 'fail', 'pending', got {status!r}"
        )
        status = "pending"

    aggregate_metrics = result.get("aggregate_metrics")
    if aggregate_metrics is None:
        errors.append("confirmation_result_missing_aggregate_metrics")
    elif not isinstance(aggregate_metrics, Mapping):
        errors.append("confirmation_result_aggregate_metrics_must_be_mapping")

    return ConfirmationEvaluation(
        status=status,
        pack_id=pack_id if isinstance(pack_id, str) else expected_pack_id,
        errors=tuple(errors),
    )


def load_confirmation_result(path: str | Path) -> dict[str, Any]:
    """Load one aggregate confirmation result from disk."""

    return _load_json(Path(path))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
