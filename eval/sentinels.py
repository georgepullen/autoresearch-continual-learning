"""Sentinel checks for undeclared capacity and integrity-breaking drift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


INVALIDATING_CODES = frozenset(
    {
        "undeclared_retrieval",
        "undeclared_helper_model",
        "undeclared_postprocessor",
        "missing_declared_capacity",
        "wrong_pack",
        "wrong_run_class",
        "immutable_drift",
    }
)


@dataclass(frozen=True)
class SentinelFinding:
    """One integrity finding emitted by the sentinel layer."""

    code: str
    message: str
    invalidates_run: bool


def run_all_sentinels(
    *,
    run_spec: Mapping[str, Any],
    artifact: Mapping[str, Any],
    immutable_hashes_verified: bool,
) -> tuple[SentinelFinding, ...]:
    """Run the first integrity checks over one spec/artifact pair."""

    findings: list[SentinelFinding] = []
    findings.extend(check_declared_capacity(run_spec))
    findings.extend(check_undeclared_capacity_flags(run_spec, artifact))
    findings.extend(check_pack_and_run_class(run_spec, artifact))

    if not immutable_hashes_verified:
        findings.append(
            SentinelFinding(
                code="immutable_drift",
                message="Immutable hash verification did not pass for this run.",
                invalidates_run=True,
            )
        )

    return tuple(findings)


def check_declared_capacity(run_spec: Mapping[str, Any]) -> tuple[SentinelFinding, ...]:
    """Ensure the run spec declares the capacity the method is allowed to use."""

    declared_capacity = run_spec.get("declared_capacity")
    if not isinstance(declared_capacity, Mapping):
        return (
            SentinelFinding(
                code="missing_declared_capacity",
                message="Run spec is missing the declared_capacity block.",
                invalidates_run=True,
            ),
        )

    required_fields = ("trainable_parameter_count", "uses_retrieval", "helper_models")
    missing = [field for field in required_fields if field not in declared_capacity]
    if missing:
        return (
            SentinelFinding(
                code="missing_declared_capacity",
                message=(
                    "Run spec declared_capacity is missing required fields: "
                    + ", ".join(missing)
                ),
                invalidates_run=True,
            ),
        )

    return ()


def check_undeclared_capacity_flags(
    run_spec: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> tuple[SentinelFinding, ...]:
    """Check simple explicit flags for undeclared extra capacity."""

    findings: list[SentinelFinding] = []
    declared_capacity = run_spec.get("declared_capacity", {})
    observed_capacity = artifact.get("observed_capacity", {})

    if _bool(observed_capacity.get("used_retrieval")) and not _bool(
        declared_capacity.get("uses_retrieval")
    ):
        findings.append(
            SentinelFinding(
                code="undeclared_retrieval",
                message="Artifact indicates retrieval usage that was not declared in the run spec.",
                invalidates_run=True,
            )
        )

    helper_models = observed_capacity.get("helper_models", [])
    declared_helpers = declared_capacity.get("helper_models", [])
    if isinstance(helper_models, list):
        undeclared_helpers = [
            helper for helper in helper_models if helper not in declared_helpers
        ]
        if undeclared_helpers:
            findings.append(
                SentinelFinding(
                    code="undeclared_helper_model",
                    message=(
                        "Artifact reports helper models not declared in the run spec: "
                        + ", ".join(sorted(str(item) for item in undeclared_helpers))
                    ),
                    invalidates_run=True,
                )
            )

    if _bool(observed_capacity.get("used_postprocessor")) and not _bool(
        declared_capacity.get("uses_postprocessor")
    ):
        findings.append(
            SentinelFinding(
                code="undeclared_postprocessor",
                message="Artifact indicates post-hoc postprocessor usage that was not declared in the run spec.",
                invalidates_run=True,
            )
        )

    return tuple(findings)


def check_pack_and_run_class(
    run_spec: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> tuple[SentinelFinding, ...]:
    """Ensure the artifact reports the intended pack and run-class identity."""

    findings: list[SentinelFinding] = []

    expected_run_class = run_spec.get("run_class")
    actual_run_class = artifact.get("run", {}).get("run_class")
    if expected_run_class and actual_run_class and expected_run_class != actual_run_class:
        findings.append(
            SentinelFinding(
                code="wrong_run_class",
                message=(
                    f"Artifact run_class {actual_run_class!r} does not match frozen spec "
                    f"run_class {expected_run_class!r}."
                ),
                invalidates_run=True,
            )
        )

    spec_data = run_spec.get("data", {})
    expected_pack = spec_data.get("development_pack") if isinstance(spec_data, Mapping) else None
    actual_pack = artifact.get("data", {}).get("development_pack")
    if expected_pack and actual_pack and expected_pack != actual_pack:
        findings.append(
            SentinelFinding(
                code="wrong_pack",
                message=(
                    f"Artifact development_pack {actual_pack!r} does not match frozen spec "
                    f"development_pack {expected_pack!r}."
                ),
                invalidates_run=True,
            )
        )

    return tuple(findings)


def invalidating_findings(
    findings: tuple[SentinelFinding, ...],
) -> tuple[SentinelFinding, ...]:
    """Return only findings that should invalidate a run."""

    return tuple(finding for finding in findings if finding.invalidates_run)


def _bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False
