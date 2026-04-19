"""Conflict-aware gating primitives for the mainline hypernetwork method."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ConflictGateConfig:
    """Configuration for the per-layer, per-module conflict gate."""

    context_dim: int
    num_layers: int
    num_modules: int
    hidden_dim: int = 128
    temperature: float = 1.0
    use_reference_delta: bool = True


class ConflictGate(nn.Module):
    """Predicts bounded gates for one editable surface.

    The gate is intentionally small and explicit. It can condition on:

    - the current update/context vector
    - an optional reference vector that represents the knowledge state to preserve
    - their delta, when enabled
    """

    def __init__(self, config: ConflictGateConfig) -> None:
        super().__init__()
        if config.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if config.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if config.num_modules <= 0:
            raise ValueError("num_modules must be positive")
        if config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if config.temperature <= 0:
            raise ValueError("temperature must be positive")

        self.config = config
        input_dim = config.context_dim * (3 if config.use_reference_delta else 2)
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.num_layers * config.num_modules),
        )

    def forward(
        self,
        context_vector: torch.Tensor,
        reference_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return gate values in ``[0, 1]`` with shape ``[batch, layers, modules]``."""

        if context_vector.ndim != 2:
            raise ValueError("context_vector must be rank-2 [batch, context_dim]")

        if reference_vector is None:
            reference_vector = torch.zeros_like(context_vector)
        elif reference_vector.shape != context_vector.shape:
            raise ValueError("reference_vector must match context_vector shape")

        pieces = [context_vector, reference_vector]
        if self.config.use_reference_delta:
            pieces.append(context_vector - reference_vector)

        features = torch.cat(pieces, dim=-1)
        logits = self.network(features)
        logits = logits.view(
            context_vector.shape[0],
            self.config.num_layers,
            self.config.num_modules,
        )
        return torch.sigmoid(logits / self.config.temperature)
