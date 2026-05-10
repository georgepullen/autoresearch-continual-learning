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
    method_family: str = "baseline_seq_lora_ft_v0"
    uses_answer_selection: bool = False


class SequentialLoRABaseline:
    """Thin sequential LoRA baseline over the same explicit surface family."""

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
            "method_family": self.config.method_family,
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
            "notes": baseline_notes(self.config),
        }


def model_shape_from_hf_config(config: Any) -> ModelShape:
    """Project a HF config object into the model-shape fields used by the harness."""

    root = getattr(config, "text_config", config)
    num_hidden_layers = first_positive_int(
        root,
        "num_hidden_layers",
        "num_layers",
        "n_layer",
    )
    hidden_size = first_positive_int(
        root,
        "hidden_size",
        "d_model",
        "n_embd",
    )
    intermediate_size = first_positive_int(
        root,
        "intermediate_size",
        "ffn_dim",
        "n_inner",
    )
    if intermediate_size is None:
        intermediate_size = hidden_size * 4

    if num_hidden_layers is None or hidden_size is None:
        raise ValueError("model config did not expose the required shape metadata")

    return ModelShape(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )


def observed_capacity_metadata(
    wrapped_model: Any,
    *,
    optimizer: Any | None = None,
) -> dict[str, object]:
    """Measure the runtime-observed capacity used by the wrapped LoRA model."""

    trainable_parameters = [
        parameter
        for parameter in wrapped_model.parameters()
        if getattr(parameter, "requires_grad", False)
    ]
    trainable_parameter_ids = {id(parameter) for parameter in trainable_parameters}
    observed_trainable_parameter_count = sum(
        int(parameter.numel()) for parameter in trainable_parameters
    )

    base_model_trainable_parameter_count = sum(
        int(parameter.numel())
        for name, parameter in wrapped_model.named_parameters()
        if getattr(parameter, "requires_grad", False) and not is_lora_parameter_name(name)
    )

    optimizer_parameter_ids: set[int] = set()
    optimizer_excludes_frozen_base_parameters = True
    if optimizer is not None:
        for group in getattr(optimizer, "param_groups", []):
            for parameter in group.get("params", []):
                parameter_id = id(parameter)
                optimizer_parameter_ids.add(parameter_id)
                if parameter_id not in trainable_parameter_ids:
                    optimizer_excludes_frozen_base_parameters = False

        if optimizer_parameter_ids != trainable_parameter_ids:
            optimizer_excludes_frozen_base_parameters = False

    return {
        "observed_trainable_parameter_count": observed_trainable_parameter_count,
        "observed_trainable_parameter_count_measured": True,
        "base_model_trainable_parameter_count": base_model_trainable_parameter_count,
        "base_model_trainable_parameter_count_measured": True,
        "frozen_base_behavior_verified": base_model_trainable_parameter_count == 0,
        "frozen_base_behavior_measured": True,
        "optimizer_excludes_frozen_base_parameters": optimizer_excludes_frozen_base_parameters,
        "optimizer_param_membership_measured": optimizer is not None,
        "used_retrieval": False,
        "helper_models": [],
        "used_postprocessor": False,
    }


def first_positive_int(config: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def is_lora_parameter_name(name: str) -> bool:
    lowered = name.lower()
    return "lora_" in lowered or ".lora" in lowered


def baseline_notes(config: SequentialLoRAConfig) -> list[str]:
    notes = ["Sequential LoRA baseline over the same explicit editable surface family."]
    if config.uses_answer_selection:
        notes.append(
            "Inference uses declared answer-set scoring over canonical answers derived from the training stream."
        )
    if config.method_family != "baseline_seq_lora_ft_v0":
        notes.append(f"Method family variant: {config.method_family}.")
    return notes
