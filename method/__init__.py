"""Method-layer helpers for the continual-learning harness.

Keep package import lightweight. Control-plane scripts such as
``scripts/run_loop.py status`` and ``scripts/freeze_spec.py --help`` need access
to protocol helpers on machines that do not have the training stack installed.
Torch-backed method objects are loaded lazily when a runner actually asks for
them.
"""

from __future__ import annotations

from typing import Any

from .adapter_surface import (
    AdapterSurfaceSpec,
    ModelShape,
    SURFACE_LIBRARY,
    get_surface,
    resolve_layer_indices,
)
from .capacity import declared_trainable_parameter_count
from .episode_sampler import DeterministicEpisodeSampler, Episode, GeneratorSpec, load_generator_spec
from .selected_stack import load_selected_pilot_pair
from .trainer import BoundedArtifactTrainer, RunBudget, TrainerContext

__all__ = [
    "AdapterSurfaceSpec",
    "BoundedArtifactTrainer",
    "ConflictGate",
    "ConflictGateConfig",
    "DeterministicEpisodeSampler",
    "Episode",
    "GeneratedAdapterDelta",
    "GeneratorSpec",
    "HyperLoRAConfig",
    "HyperLoRAController",
    "LossBreakdown",
    "LossWeights",
    "ModelShape",
    "RunBudget",
    "SURFACE_LIBRARY",
    "TrainerContext",
    "get_surface",
    "declared_trainable_parameter_count",
    "load_selected_pilot_pair",
    "load_generator_spec",
    "resolve_layer_indices",
]


_LAZY_EXPORTS = {
    "ConflictGate": ("method.conflict_gate", "ConflictGate"),
    "ConflictGateConfig": ("method.conflict_gate", "ConflictGateConfig"),
    "GeneratedAdapterDelta": ("method.hypernet", "GeneratedAdapterDelta"),
    "HyperLoRAConfig": ("method.hypernet", "HyperLoRAConfig"),
    "HyperLoRAController": ("method.hypernet", "HyperLoRAController"),
    "LossBreakdown": ("method.losses", "LossBreakdown"),
    "LossWeights": ("method.losses", "LossWeights"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value
