"""Thin AlphaEdit-style baseline wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


ApplyEditFn = Callable[[Any, dict[str, Any]], Any]


@dataclass(frozen=True)
class EditWrapperConfig:
    """Shared config for thin editing-method wrappers."""

    method_family: str
    editable_surface_name: str
    helper_models: tuple[str, ...] = ()
    uses_retrieval: bool = False
    uses_postprocessor: bool = False


class AlphaEditWrapper:
    """Adapter that keeps AlphaEdit-style baselines honest inside the harness.

    This wrapper is deliberately thin:

    - it does not implement AlphaEdit itself
    - it records declared capacity and helper usage explicitly
    - it expects the actual edit function to be injected by the environment that has
      the underlying implementation available
    """

    def __init__(self, config: EditWrapperConfig, apply_edit_fn: ApplyEditFn) -> None:
        if config.method_family != "baseline_alphaedit_v0":
            raise ValueError("AlphaEditWrapper requires method_family='baseline_alphaedit_v0'")
        self.config = config
        self.apply_edit_fn = apply_edit_fn

    def apply_update(self, model: Any, edit_request: dict[str, Any]) -> Any:
        return self.apply_edit_fn(model, edit_request)

    def declared_capacity_metadata(self) -> dict[str, object]:
        return {
            "method_family": self.config.method_family,
            "surface_name": self.config.editable_surface_name,
            "frozen_base_model": False,
            "trainable_parameter_count": 0,
            "uses_retrieval": self.config.uses_retrieval,
            "uses_postprocessor": self.config.uses_postprocessor,
            "helper_models": list(self.config.helper_models),
            "notes": [
                "Thin AlphaEdit-style wrapper. External implementation must be declared explicitly.",
            ],
        }
