"""Simple sequential LoRA fine-tuning baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from method.adapter_surface import ModelShape, get_surface, resolve_layer_indices

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class SequentialLoRAConfig:
    """Configuration for the simplest parametric baseline."""

    surface_name: str
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0


class SequentialLoRABaseline:
    """Thin sequential LoRA baseline over the same explicit surface family."""

    method_family = "baseline_seq_lora_ft_v0"

    def __init__(self, config: SequentialLoRAConfig, model_shape: ModelShape) -> None:
        if config.rank <= 0:
            raise ValueError("rank must be positive")
        if config.alpha <= 0:
            raise ValueError("alpha must be positive")
        if config.dropout < 0:
            raise ValueError("dropout must be non-negative")

        self.config = config
        self.surface = get_surface(config.surface_name)
        self.layer_indices = resolve_layer_indices(model_shape.num_hidden_layers, self.surface)

    def wrap_model(self, model: Any) -> Any:
        """Apply a standard LoRA adapter over the named editable surface."""

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError(
                "SequentialLoRABaseline.wrap_model requires `peft` to be installed in the active env."
            ) from exc

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.rank,
            lora_alpha=self.config.alpha,
            lora_dropout=self.config.dropout,
            bias="none",
            target_modules=list(self.surface.target_modules),
            layers_to_transform=list(self.layer_indices),
        )
        return get_peft_model(model, peft_config)

    def declared_capacity_metadata(self, wrapped_model: Any) -> dict[str, object]:
        trainable_parameter_count = sum(
            parameter.numel() for parameter in wrapped_model.parameters() if parameter.requires_grad
        )
        return {
            "method_family": self.method_family,
            "surface_name": self.surface.name,
            "layer_indices": list(self.layer_indices),
            "target_modules": list(self.surface.target_modules),
            "frozen_base_model": True,
            "trainable_parameter_count": trainable_parameter_count,
            "rank": self.config.rank,
            "alpha": self.config.alpha,
            "dropout": self.config.dropout,
            "uses_retrieval": False,
            "uses_postprocessor": False,
            "helper_models": [],
            "notes": [
                "Sequential LoRA baseline over the same explicit editable surface family.",
            ],
        }
