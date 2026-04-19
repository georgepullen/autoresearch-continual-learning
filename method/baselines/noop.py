"""No-op baseline for sanity and harness verification."""

from __future__ import annotations


class NoopBaseline:
    """A baseline that makes no learned update at all.

    This exists for two reasons:

    - sanity-check the harness and artifact path
    - provide a lower bound that any real method should beat
    """

    method_family = "baseline_noop_v0"

    def declared_capacity_metadata(self) -> dict[str, object]:
        return {
            "method_family": self.method_family,
            "frozen_base_model": True,
            "trainable_parameter_count": 0,
            "uses_retrieval": False,
            "uses_postprocessor": False,
            "helper_models": [],
            "notes": [
                "No-op baseline: predictions come from the unchanged frozen base.",
            ],
        }

    def apply_update(self, context: object | None = None) -> None:
        """Intentionally do nothing."""

        return None
