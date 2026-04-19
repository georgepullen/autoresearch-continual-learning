"""Thin NSE-style baseline wrapper."""

from __future__ import annotations

from typing import Any

from .alphaedit_wrapper import ApplyEditFn, EditWrapperConfig


class NSEWrapper:
    """Adapter that keeps NSE-style baselines explicit inside the harness."""

    def __init__(self, config: EditWrapperConfig, apply_edit_fn: ApplyEditFn) -> None:
        if config.method_family != "baseline_nse_v0":
            raise ValueError("NSEWrapper requires method_family='baseline_nse_v0'")
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
                "Thin NSE-style wrapper. External implementation must be declared explicitly.",
            ],
        }
