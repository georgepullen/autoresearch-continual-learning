"""Tiny hypernetwork controller for the first HyperLoRA-style method family."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .adapter_surface import ModelShape, get_surface, module_dimensions, resolve_layer_indices
from .conflict_gate import ConflictGate, ConflictGateConfig


@dataclass(frozen=True)
class HyperLoRAConfig:
    """Configuration for the smallest useful HyperLoRA controller."""

    surface_name: str
    model_shape: ModelShape
    context_dim: int
    rank: int
    trunk_hidden_dim: int = 256
    gate_hidden_dim: int = 128
    use_conflict_gate: bool = True


@dataclass(frozen=True)
class GeneratedAdapterDelta:
    """Structured adapter outputs for one forward pass of the hypernetwork."""

    layer_indices: tuple[int, ...]
    target_modules: tuple[str, ...]
    gates: torch.Tensor | None
    lora_a: dict[str, torch.Tensor]
    lora_b: dict[str, torch.Tensor]


class HyperLoRAController(nn.Module):
    """Generates low-rank adapter deltas over one explicit editable surface.

    The controller keeps the first method family legible:

    - the base model remains frozen
    - the editable layer/module set is explicit
    - trainable capacity is declared up front
    - optional conflict gating is surfaced as first-class metadata
    """

    def __init__(self, config: HyperLoRAConfig) -> None:
        super().__init__()
        if config.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if config.rank <= 0:
            raise ValueError("rank must be positive")
        if config.trunk_hidden_dim <= 0:
            raise ValueError("trunk_hidden_dim must be positive")
        if config.gate_hidden_dim <= 0:
            raise ValueError("gate_hidden_dim must be positive")

        self.config = config
        self.surface = get_surface(config.surface_name)
        self.layer_indices = resolve_layer_indices(
            config.model_shape.num_hidden_layers,
            self.surface,
        )
        self.target_modules = self.surface.target_modules

        self.trunk = nn.Sequential(
            nn.Linear(config.context_dim, config.trunk_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.trunk_hidden_dim, config.trunk_hidden_dim),
            nn.SiLU(),
        )

        self.a_heads = nn.ModuleDict()
        self.b_heads = nn.ModuleDict()
        for module_name in self.target_modules:
            in_features, out_features = module_dimensions(module_name, config.model_shape)
            layer_count = len(self.layer_indices)
            self.a_heads[module_name] = nn.Linear(
                config.trunk_hidden_dim,
                layer_count * config.rank * in_features,
            )
            self.b_heads[module_name] = nn.Linear(
                config.trunk_hidden_dim,
                layer_count * out_features * config.rank,
            )

        self.conflict_gate: ConflictGate | None = None
        if config.use_conflict_gate:
            self.conflict_gate = ConflictGate(
                ConflictGateConfig(
                    context_dim=config.context_dim,
                    num_layers=len(self.layer_indices),
                    num_modules=len(self.target_modules),
                    hidden_dim=config.gate_hidden_dim,
                    use_reference_delta=True,
                )
            )

    def forward(
        self,
        context_vector: torch.Tensor,
        reference_vector: torch.Tensor | None = None,
    ) -> GeneratedAdapterDelta:
        """Generate one bounded adapter update from a context embedding."""

        if context_vector.ndim != 2:
            raise ValueError("context_vector must be rank-2 [batch, context_dim]")

        features = self.trunk(context_vector)
        lora_a: dict[str, torch.Tensor] = {}
        lora_b: dict[str, torch.Tensor] = {}

        for module_name in self.target_modules:
            in_features, out_features = module_dimensions(module_name, self.config.model_shape)
            layer_count = len(self.layer_indices)
            a_values = self.a_heads[module_name](features).view(
                context_vector.shape[0],
                layer_count,
                self.config.rank,
                in_features,
            )
            b_values = self.b_heads[module_name](features).view(
                context_vector.shape[0],
                layer_count,
                out_features,
                self.config.rank,
            )
            lora_a[module_name] = a_values
            lora_b[module_name] = b_values

        gates = None
        if self.conflict_gate is not None:
            gates = self.conflict_gate(context_vector, reference_vector=reference_vector)

        return GeneratedAdapterDelta(
            layer_indices=self.layer_indices,
            target_modules=self.target_modules,
            gates=gates,
            lora_a=lora_a,
            lora_b=lora_b,
        )

    def declared_capacity_metadata(self) -> dict[str, object]:
        """Return explicit capacity metadata for specs, artifacts, and integrity checks."""

        trainable_parameter_count = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "method_family": "hyper_lora_v0",
            "surface_name": self.surface.name,
            "layer_indices": list(self.layer_indices),
            "target_modules": list(self.target_modules),
            "uses_conflict_gate": self.conflict_gate is not None,
            "trainable_parameter_count": trainable_parameter_count,
            "frozen_base_model": True,
            "rank": self.config.rank,
            "context_dim": self.config.context_dim,
        }
