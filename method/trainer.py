"""Bounded training shell that can emit schema-valid run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any, Callable, Iterable

from eval.schema import artifact_template


@dataclass(frozen=True)
class RunBudget:
    """Execution limits for a bounded run."""

    max_steps: int
    max_runtime_seconds: float


@dataclass(frozen=True)
class TrainerContext:
    """Static run metadata needed to build a valid artifact."""

    run_id: str
    parent_champion: str
    task_mode: str
    run_class: str
    command: str
    spec_path: str
    method_family: str
    base_model: str
    editable_paths: tuple[str, ...]
    selection_policy: str
    declared_trainable_params: int
    train_generator_version: str
    development_pack: str
    confirmation_pack_summary: str
    baseline_ref: str
    comparison_scope: str


StepFn = Callable[[Any, int], dict[str, Any]]


class BoundedArtifactTrainer:
    """Run a bounded step function over episodes and emit an artifact summary.

    This class is intentionally small. It owns:

    - runtime budgeting
    - optional peak-VRAM capture
    - aggregation into the canonical artifact schema

    It does not own the method-specific forward/backward pass. That remains in the
    caller-supplied ``step_fn`` so later method families and baselines can share the
    same outer harness without hiding bespoke side paths.
    """

    def __init__(
        self,
        *,
        budget: RunBudget,
        context: TrainerContext,
        torch_module: Any | None = None,
    ) -> None:
        if budget.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if budget.max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")
        self.budget = budget
        self.context = context
        self.torch_module = torch_module

    def run(
        self,
        episodes: Iterable[Any],
        step_fn: StepFn,
        *,
        integrity_flags: dict[str, bool],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        start = time.perf_counter()
        records: list[dict[str, Any]] = []
        peak_vram_gb = 0.0

        if self._cuda_available():
            self.torch_module.cuda.reset_peak_memory_stats()

        for step_index, episode in enumerate(episodes):
            if step_index >= self.budget.max_steps:
                break
            if time.perf_counter() - start >= self.budget.max_runtime_seconds:
                break

            step_start = time.perf_counter()
            metrics = normalize_step_metrics(step_fn(episode, step_index))
            metrics["step_index"] = step_index
            metrics["step_runtime_seconds"] = time.perf_counter() - step_start
            records.append(metrics)

            if self._cuda_available():
                peak_vram_gb = max(
                    peak_vram_gb,
                    float(self.torch_module.cuda.max_memory_allocated()) / (1024 ** 3),
                )

        artifact = build_artifact(
            context=self.context,
            records=records,
            runtime_seconds=time.perf_counter() - start,
            peak_vram_gb=peak_vram_gb,
            integrity_flags=integrity_flags,
        )
        return artifact, records

    def _cuda_available(self) -> bool:
        return bool(self.torch_module is not None and self.torch_module.cuda.is_available())


def build_artifact(
    *,
    context: TrainerContext,
    records: list[dict[str, Any]],
    runtime_seconds: float,
    peak_vram_gb: float,
    integrity_flags: dict[str, bool],
) -> dict[str, Any]:
    """Aggregate bounded step records into the canonical schema."""

    payload = artifact_template()
    payload["run"].update(
        {
            "run_id": context.run_id,
            "parent_champion": context.parent_champion,
            "task_mode": context.task_mode,
            "run_class": context.run_class,
            "command": context.command,
            "spec_path": context.spec_path,
            "created_at_utc": utc_now(),
        }
    )
    payload["method"].update(
        {
            "method_family": context.method_family,
            "base_model": context.base_model,
            "editable_surface": {
                "paths": list(context.editable_paths),
                "selection_policy": context.selection_policy,
            },
            "declared_capacity": {
                "trainable_parameter_count": context.declared_trainable_params,
                "notes": [],
            },
        }
    )
    payload["data"].update(
        {
            "train_generator_version": context.train_generator_version,
            "development_pack": context.development_pack,
            "confirmation_pack_summary": context.confirmation_pack_summary,
        }
    )
    payload["comparison"].update(
        {
            "baseline_ref": context.baseline_ref,
            "comparison_scope": context.comparison_scope,
        }
    )
    payload["metrics"]["target_quality"] = aggregate_family(records, "target_quality")
    payload["metrics"]["interference"] = aggregate_family(records, "interference")
    payload["metrics"]["cost"] = {
        "runtime_seconds": runtime_seconds,
        "peak_vram_gb": peak_vram_gb,
    }
    payload["integrity"].update(
        {
            "preflight_passed": bool(integrity_flags.get("preflight_passed", False)),
            "immutable_hashes_verified": bool(
                integrity_flags.get("immutable_hashes_verified", False)
            ),
            "shim_checks_passed": bool(integrity_flags.get("shim_checks_passed", False)),
        }
    )
    return payload


def aggregate_family(records: list[dict[str, Any]], family: str) -> dict[str, float]:
    """Compute means for one nested metric family across step records."""

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for record in records:
        payload = record.get(family, {})
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1

    return {
        key: totals[key] / counts[key]
        for key in sorted(totals)
        if counts.get(key, 0) > 0
    }


def normalize_step_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure the step function returns a dict with the expected family keys."""

    if not isinstance(payload, dict):
        raise ValueError("step_fn must return a mapping")
    normalized = dict(payload)
    normalized.setdefault("target_quality", {})
    normalized.setdefault("interference", {})
    if not isinstance(normalized["target_quality"], dict):
        raise ValueError("target_quality metrics must be a mapping")
    if not isinstance(normalized["interference"], dict):
        raise ValueError("interference metrics must be a mapping")
    return normalized


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
