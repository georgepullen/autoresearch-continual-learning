"""Canonical structured artifact schema for counted runs.

The harness depends on machine-readable results. This module defines the first
artifact shape and validates whether a run artifact is complete enough to count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ArtifactValidationResult:
    """Validation outcome for one artifact payload."""

    valid: bool
    errors: tuple[str, ...]


def artifact_template() -> dict[str, Any]:
    """Return the canonical top-level artifact template."""

    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "run_id": "",
            "parent_champion": "",
            "task_mode": "",
            "run_class": "",
            "command": "",
            "spec_path": "",
            "created_at_utc": "",
        },
        "method": {
            "method_family": "",
            "base_model": "",
            "editable_surface": {
                "paths": [],
                "selection_policy": "",
            },
            "declared_capacity": {
                "frozen_base_model": False,
                "trainable_parameter_count": 0,
                "trainable_parameter_tolerance": 0,
                "uses_retrieval": False,
                "helper_models": [],
                "uses_postprocessor": False,
                "notes": [],
            },
        },
        "data": {
            "train_generator_version": "",
            "development_pack": "",
            "confirmation_pack_summary": "",
        },
        "comparison": {
            "baseline_ref": "",
            "comparison_scope": "",
        },
        "metrics": {
            "target_quality": {},
            "interference": {},
            "cost": {
                "runtime_seconds": 0.0,
                "peak_vram_gb": 0.0,
            },
        },
        "integrity": {
            "preflight_passed": False,
            "immutable_hashes_verified": False,
            "shim_checks_passed": False,
        },
        "observed_capacity": {
            "declared_trainable_parameter_count": 0,
            "observed_trainable_parameter_count": 0,
            "base_model_trainable_parameter_count": 0,
            "frozen_base_behavior_verified": False,
            "optimizer_excludes_frozen_base_parameters": False,
            "used_retrieval": False,
            "helper_models": [],
            "used_postprocessor": False,
        },
    }


def validate_artifact(payload: Mapping[str, Any]) -> ArtifactValidationResult:
    """Validate the canonical artifact shape."""

    errors: list[str] = []

    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
        )

    _require_mapping(payload, "run", errors)
    _require_mapping(payload, "method", errors)
    _require_mapping(payload, "data", errors)
    _require_mapping(payload, "comparison", errors)
    _require_mapping(payload, "metrics", errors)
    _require_mapping(payload, "integrity", errors)
    _require_mapping(payload, "observed_capacity", errors)

    run = _as_mapping(payload.get("run"))
    method = _as_mapping(payload.get("method"))
    data = _as_mapping(payload.get("data"))
    comparison = _as_mapping(payload.get("comparison"))
    metrics = _as_mapping(payload.get("metrics"))
    integrity = _as_mapping(payload.get("integrity"))
    observed_capacity = _as_mapping(payload.get("observed_capacity"))

    _require_string_fields(
        run,
        [
            "run_id",
            "parent_champion",
            "task_mode",
            "run_class",
            "command",
            "spec_path",
            "created_at_utc",
        ],
        "run",
        errors,
    )
    _require_string_fields(
        method,
        ["method_family", "base_model"],
        "method",
        errors,
    )
    _require_mapping(method, "editable_surface", errors, parent="method")
    _require_mapping(method, "declared_capacity", errors, parent="method")
    _require_string_fields(
        data,
        ["train_generator_version", "development_pack", "confirmation_pack_summary"],
        "data",
        errors,
    )
    _require_string_fields(
        comparison,
        ["baseline_ref", "comparison_scope"],
        "comparison",
        errors,
    )
    _require_mapping(metrics, "target_quality", errors, parent="metrics")
    _require_mapping(metrics, "interference", errors, parent="metrics")
    _require_mapping(metrics, "cost", errors, parent="metrics")
    _require_boolean_fields(
        integrity,
        ["preflight_passed", "immutable_hashes_verified", "shim_checks_passed"],
        "integrity",
        errors,
    )

    editable_surface = _as_mapping(method.get("editable_surface"))
    declared_capacity = _as_mapping(method.get("declared_capacity"))
    cost = _as_mapping(metrics.get("cost"))

    _require_sequence_field(editable_surface, "paths", "method.editable_surface", errors)
    _require_string_fields(
        editable_surface,
        ["selection_policy"],
        "method.editable_surface",
        errors,
    )
    _require_int_field(
        declared_capacity,
        "trainable_parameter_count",
        "method.declared_capacity",
        errors,
    )
    _require_boolean_fields(
        declared_capacity,
        ["frozen_base_model", "uses_retrieval", "uses_postprocessor"],
        "method.declared_capacity",
        errors,
    )
    _require_int_field(
        declared_capacity,
        "trainable_parameter_tolerance",
        "method.declared_capacity",
        errors,
    )
    _require_sequence_field(
        declared_capacity,
        "helper_models",
        "method.declared_capacity",
        errors,
    )
    _require_sequence_field(
        declared_capacity,
        "notes",
        "method.declared_capacity",
        errors,
    )
    _require_number_fields(
        cost,
        ["runtime_seconds", "peak_vram_gb"],
        "metrics.cost",
        errors,
    )
    _require_int_field(
        observed_capacity,
        "declared_trainable_parameter_count",
        "observed_capacity",
        errors,
    )
    _require_int_field(
        observed_capacity,
        "observed_trainable_parameter_count",
        "observed_capacity",
        errors,
    )
    _require_int_field(
        observed_capacity,
        "base_model_trainable_parameter_count",
        "observed_capacity",
        errors,
    )
    _require_boolean_fields(
        observed_capacity,
        [
            "frozen_base_behavior_verified",
            "optimizer_excludes_frozen_base_parameters",
            "used_retrieval",
            "used_postprocessor",
        ],
        "observed_capacity",
        errors,
    )
    _require_sequence_field(
        observed_capacity,
        "helper_models",
        "observed_capacity",
        errors,
    )

    return ArtifactValidationResult(valid=not errors, errors=tuple(errors))


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _require_mapping(
    payload: Mapping[str, Any],
    field: str,
    errors: list[str],
    parent: str | None = None,
) -> None:
    value = payload.get(field)
    if not isinstance(value, Mapping):
        prefix = f"{parent}." if parent else ""
        errors.append(f"{prefix}{field} must be a mapping")


def _require_string_fields(
    payload: Mapping[str, Any],
    fields: Sequence[str],
    parent: str,
    errors: list[str],
) -> None:
    for field in fields:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{parent}.{field} must be a non-empty string")


def _require_boolean_fields(
    payload: Mapping[str, Any],
    fields: Sequence[str],
    parent: str,
    errors: list[str],
) -> None:
    for field in fields:
        value = payload.get(field)
        if not isinstance(value, bool):
            errors.append(f"{parent}.{field} must be a boolean")


def _require_number_fields(
    payload: Mapping[str, Any],
    fields: Sequence[str],
    parent: str,
    errors: list[str],
) -> None:
    for field in fields:
        value = payload.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{parent}.{field} must be a number")


def _require_int_field(
    payload: Mapping[str, Any],
    field: str,
    parent: str,
    errors: list[str],
) -> None:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        errors.append(f"{parent}.{field} must be a non-negative integer")


def _require_sequence_field(
    payload: Mapping[str, Any],
    field: str,
    parent: str,
    errors: list[str],
) -> None:
    value = payload.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        errors.append(f"{parent}.{field} must be a sequence")
