"""Method-layer helpers for the continual-learning harness."""

from .adapter_surface import (
    AdapterSurfaceSpec,
    ModelShape,
    SURFACE_LIBRARY,
    get_surface,
    resolve_layer_indices,
)
from .conflict_gate import ConflictGate, ConflictGateConfig
from .episode_sampler import DeterministicEpisodeSampler, Episode, GeneratorSpec, load_generator_spec
from .hypernet import GeneratedAdapterDelta, HyperLoRAConfig, HyperLoRAController
from .losses import LossBreakdown, LossWeights
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
    "load_selected_pilot_pair",
    "load_generator_spec",
    "resolve_layer_indices",
]
