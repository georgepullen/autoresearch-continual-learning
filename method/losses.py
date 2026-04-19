"""Loss helpers for bounded continual-learning update episodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    """Relative weights for the first bounded training objective."""

    target_update: float = 1.0
    retain_anchor: float = 1.0
    interference_margin: float = 1.0
    gate_sparsity: float = 0.01


@dataclass(frozen=True)
class LossBreakdown:
    """Materialized scalar losses for logging and artifact emission."""

    total: torch.Tensor
    target_update: torch.Tensor
    retain_anchor: torch.Tensor
    interference_margin: torch.Tensor
    gate_sparsity: torch.Tensor

    def as_floats(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "target_update": float(self.target_update.detach().cpu()),
            "retain_anchor": float(self.retain_anchor.detach().cpu()),
            "interference_margin": float(self.interference_margin.detach().cpu()),
            "gate_sparsity": float(self.gate_sparsity.detach().cpu()),
        }


def masked_token_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy loss over token predictions with optional masking."""

    if logits.ndim != 3:
        raise ValueError("logits must be rank-3 [batch, seq, vocab]")
    if labels.ndim != 2:
        raise ValueError("labels must be rank-2 [batch, seq]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must agree on batch and sequence dims")

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()

    if attention_mask is not None:
        if attention_mask.shape != labels.shape:
            raise ValueError("attention_mask must match labels shape")
        shifted_mask = attention_mask[:, 1:].contiguous()
        shifted_labels = shifted_labels.masked_fill(shifted_mask == 0, ignore_index)

    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=ignore_index,
    )


def anchor_kl_loss(
    candidate_logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL loss that encourages the updated path to preserve anchor behavior."""

    if candidate_logits.shape != anchor_logits.shape:
        raise ValueError("candidate_logits and anchor_logits must have identical shape")
    if candidate_logits.ndim != 3:
        raise ValueError("candidate_logits must be rank-3 [batch, seq, vocab]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    candidate = F.log_softmax(candidate_logits / temperature, dim=-1)
    anchor = F.softmax(anchor_logits / temperature, dim=-1)
    token_kl = F.kl_div(candidate, anchor, reduction="none").sum(dim=-1)

    if attention_mask is not None:
        if attention_mask.shape != token_kl.shape:
            raise ValueError("attention_mask must match [batch, seq] token shape")
        token_kl = token_kl * attention_mask
        denom = attention_mask.sum().clamp_min(1)
        return token_kl.sum() / denom

    return token_kl.mean()


def interference_margin_loss(
    candidate_interference: torch.Tensor,
    baseline_interference: torch.Tensor,
    *,
    margin: float = 0.0,
) -> torch.Tensor:
    """Penalize interference that exceeds the baseline plus a small margin."""

    if margin < 0:
        raise ValueError("margin must be non-negative")
    return torch.relu(candidate_interference - baseline_interference + margin)


def gate_sparsity_loss(gates: torch.Tensor | None) -> torch.Tensor:
    """Small regularizer that discourages diffuse activation across the surface."""

    if gates is None:
        return torch.tensor(0.0)
    return gates.mean()


def compose_loss(
    *,
    weights: LossWeights,
    target_update: torch.Tensor,
    retain_anchor: torch.Tensor,
    candidate_interference: torch.Tensor,
    baseline_interference: torch.Tensor,
    gates: torch.Tensor | None,
    margin: float = 0.0,
) -> LossBreakdown:
    """Combine the core objective terms into one logged breakdown."""

    interference_term = interference_margin_loss(
        candidate_interference,
        baseline_interference,
        margin=margin,
    )
    gate_term = gate_sparsity_loss(gates).to(target_update.device)
    total = (
        weights.target_update * target_update
        + weights.retain_anchor * retain_anchor
        + weights.interference_margin * interference_term
        + weights.gate_sparsity * gate_term
    )
    return LossBreakdown(
        total=total,
        target_update=target_update,
        retain_anchor=retain_anchor,
        interference_margin=interference_term,
        gate_sparsity=gate_term,
    )
