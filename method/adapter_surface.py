"""Editable-surface definitions for pilot and mainline method work."""

from __future__ import annotations

from dataclasses import dataclass


STANDARD_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
SPARSE_GATED_TARGET_MODULES = ("v_proj", "o_proj", "down_proj")


@dataclass(frozen=True)
class AdapterSurfaceSpec:
    """One named editable surface under consideration in the pilot matrix."""

    name: str
    layer_mode: str
    layer_count: int
    target_modules: tuple[str, ...]
    uses_per_layer_gate: bool
    simplicity_rank: int


@dataclass(frozen=True)
class ModelShape:
    """Shape metadata needed to size the adapter surface."""

    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int


SURFACE_LIBRARY: dict[str, AdapterSurfaceSpec] = {
    "top4_standard": AdapterSurfaceSpec(
        name="top4_standard",
        layer_mode="top_suffix",
        layer_count=4,
        target_modules=STANDARD_TARGET_MODULES,
        uses_per_layer_gate=False,
        simplicity_rank=1,
    ),
    "top6_standard": AdapterSurfaceSpec(
        name="top6_standard",
        layer_mode="top_suffix",
        layer_count=6,
        target_modules=STANDARD_TARGET_MODULES,
        uses_per_layer_gate=False,
        simplicity_rank=2,
    ),
    "sparse_suffix_gated_output": AdapterSurfaceSpec(
        name="sparse_suffix_gated_output",
        layer_mode="top_suffix",
        layer_count=6,
        target_modules=SPARSE_GATED_TARGET_MODULES,
        uses_per_layer_gate=True,
        simplicity_rank=3,
    ),
}


def get_surface(name: str) -> AdapterSurfaceSpec:
    """Return one built-in surface by name."""

    try:
        return SURFACE_LIBRARY[name]
    except KeyError as exc:
        known = ", ".join(sorted(SURFACE_LIBRARY))
        raise KeyError(f"unknown adapter surface {name!r}; known surfaces: {known}") from exc


def resolve_layer_indices(num_hidden_layers: int, surface: AdapterSurfaceSpec) -> tuple[int, ...]:
    """Resolve a named surface into concrete layer indices for one model."""

    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if surface.layer_count <= 0:
        raise ValueError("surface.layer_count must be positive")
    if surface.layer_mode != "top_suffix":
        raise ValueError(f"unsupported layer mode: {surface.layer_mode}")

    count = min(surface.layer_count, num_hidden_layers)
    start = num_hidden_layers - count
    return tuple(range(start, num_hidden_layers))


def estimate_lora_parameter_count(
    shape: ModelShape,
    surface: AdapterSurfaceSpec,
    *,
    rank: int,
) -> int:
    """Estimate the trainable parameter count for a LoRA-style probe surface."""

    if rank <= 0:
        raise ValueError("rank must be positive")

    layer_indices = resolve_layer_indices(shape.num_hidden_layers, surface)
    total = 0
    for module_name in surface.target_modules:
        in_features, out_features = module_dimensions(module_name, shape)
        total += len(layer_indices) * rank * (in_features + out_features)

    if surface.uses_per_layer_gate:
        total += len(layer_indices)

    return total


def module_dimensions(module_name: str, shape: ModelShape) -> tuple[int, int]:
    """Return the approximate input/output dimensions for one transformer module."""

    hidden = shape.hidden_size
    intermediate = shape.intermediate_size

    if module_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return hidden, hidden
    if module_name in {"up_proj", "gate_proj"}:
        return hidden, intermediate
    if module_name == "down_proj":
        return intermediate, hidden
    raise ValueError(f"unsupported target module: {module_name}")
