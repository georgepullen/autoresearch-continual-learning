"""Helpers for protected confirmation metadata and aggregate feedback."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DATA_REGISTRY_PATH = REPO_ROOT / "data" / "registry.yaml"
DEFAULT_CONFIRM_REQUESTS_DIR = REPO_ROOT / "experiments" / "confirmation" / "requests"
DEFAULT_CONFIRM_RESULTS_DIR = REPO_ROOT / "experiments" / "confirmation" / "results"
DEFAULT_HOST_LOCAL_CONFIRM_ROOT = Path(
    "~/shared/artifacts/autoresearch-continual-learning/protected"
).expanduser()


@dataclass(frozen=True)
class ConfirmationEvaluation:
    """Normalized protected-confirmation result."""

    status: str
    pack_id: str
    errors: tuple[str, ...]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    execute_parser = subparsers.add_parser(
        "execute-local",
        help="Evaluate one saved candidate adapter on the host-local confirmation pack.",
    )
    execute_parser.add_argument("--request", required=True)
    execute_parser.add_argument("--run-dir", required=True)
    execute_parser.add_argument("--output", required=True)
    execute_parser.add_argument("--pack-path")
    execute_parser.add_argument("--max-new-tokens", type=int, default=24)

    args = parser.parse_args()
    if args.command == "execute-local":
        return execute_local_confirmation(args)
    raise AssertionError(f"Unhandled command: {args.command}")


def load_confirmation_contract(
    *,
    pack_id: str | None = None,
    summary_path: Path | None = None,
    hash_path: Path | None = None,
) -> dict[str, Any]:
    """Load the public confirmation-pack contract."""

    if summary_path is None or hash_path is None:
        summary_path, hash_path = confirmation_contract_paths(pack_id)
    return {
      "summary": _load_json(summary_path),
      "hash": _load_json(hash_path),
    }


def evaluate_confirmation_result(
    result: Mapping[str, Any] | None,
    *,
    expected_pack_id: str | None = None,
) -> ConfirmationEvaluation:
    """Validate a host-local aggregate confirmation result."""

    if expected_pack_id is None:
        expected_pack_id = default_confirmation_pack_id()
    if result is None:
        return ConfirmationEvaluation(
            status="pending",
            pack_id=expected_pack_id,
            errors=("confirmation_result_missing",),
        )

    errors: list[str] = []
    pack_id = result.get("pack_id")
    if pack_id != expected_pack_id:
        errors.append(
            f"confirmation_pack_id_mismatch: expected {expected_pack_id!r}, got {pack_id!r}"
        )

    status = result.get("status")
    if status not in {"pass", "fail", "pending"}:
        errors.append(
            f"confirmation_status_invalid: expected one of 'pass', 'fail', 'pending', got {status!r}"
        )
        status = "pending"

    aggregate_metrics = result.get("aggregate_metrics")
    if aggregate_metrics is None:
        errors.append("confirmation_result_missing_aggregate_metrics")
    elif not isinstance(aggregate_metrics, Mapping):
        errors.append("confirmation_result_aggregate_metrics_must_be_mapping")

    return ConfirmationEvaluation(
        status=status,
        pack_id=pack_id if isinstance(pack_id, str) else expected_pack_id,
        errors=tuple(errors),
    )


def load_confirmation_result(path: str | Path) -> dict[str, Any]:
    """Load one aggregate confirmation result from disk."""

    return _load_json(Path(path))


def confirmation_request_path(run_id: str) -> Path:
    return DEFAULT_CONFIRM_REQUESTS_DIR / f"{run_id}.json"


def confirmation_result_path(run_id: str) -> Path:
    return DEFAULT_CONFIRM_RESULTS_DIR / f"{run_id}.json"


def default_confirm_pack_path() -> Path:
    return default_confirm_pack_path_for(default_confirmation_pack_id())


def write_confirmation_request(
    *,
    run_id: str,
    artifact_path: str,
    spec_path: str,
    expected_pack_id: str | None = None,
) -> Path:
    if expected_pack_id is None:
        expected_pack_id = default_confirmation_pack_id()
    path = confirmation_request_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "artifact_path": artifact_path,
        "spec_path": spec_path,
        "pack_id": expected_pack_id,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def execute_local_confirmation(args: argparse.Namespace) -> int:
    from scripts.run_bootstrap_baseline import evaluate_saved_adapter, evaluate_visible_pack, load_runtime_stack

    request_path = Path(args.request).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    request = load_confirmation_result(request_path)
    requested_pack_id = (
        str(request.get("pack_id")).strip()
        if isinstance(request.get("pack_id"), str) and str(request.get("pack_id")).strip()
        else default_confirmation_pack_id()
    )
    pack_path = (
        Path(args.pack_path).expanduser().resolve()
        if args.pack_path
        else default_confirm_pack_path_for(requested_pack_id).resolve()
    )
    artifact = load_confirmation_result(run_dir / "artifact.json")
    adapter_dir = run_dir / "adapter"
    model_id = artifact.get("method", {}).get("base_model")
    method_family = artifact.get("method", {}).get("method_family")
    train_generator_version = artifact.get("data", {}).get("train_generator_version")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError("artifact missing method.base_model for confirmation execution")
    if not isinstance(method_family, str) or not method_family:
        raise RuntimeError("artifact missing method.method_family for confirmation execution")
    if not pack_path.exists():
        raise RuntimeError(f"host-local confirmation pack missing: {pack_path}")
    if not adapter_dir.exists():
        raise RuntimeError(f"adapter directory missing for confirmation execution: {adapter_dir}")

    evaluation = evaluate_saved_adapter(
        model_id=model_id,
        adapter_dir=adapter_dir,
        pack_path=pack_path,
        max_new_tokens=args.max_new_tokens,
        method_family=method_family,
        train_generator_version=(
            str(train_generator_version).strip() if isinstance(train_generator_version, str) and str(train_generator_version).strip() else None
        ),
    )
    candidate_answer_sets = {}
    if isinstance(train_generator_version, str) and train_generator_version.strip():
        from method import load_generator_spec
        from scripts.run_bootstrap_baseline import build_candidate_answer_sets, registry_path

        generator = load_generator_spec(registry_path("train_generators", train_generator_version.strip()))
        candidate_answer_sets = build_candidate_answer_sets(generator)
    from scripts.run_bootstrap_baseline import policy_for_method_family

    method_policy = policy_for_method_family(method_family)
    base_stack = load_runtime_stack(model_id, adapter_dir=None, train_mode=False)
    baseline_reference = evaluate_visible_pack(
        stack=base_stack,
        model=base_stack.model,
        development_pack_path=pack_path,
        max_new_tokens=args.max_new_tokens,
        method_policy=method_policy,
        candidate_answer_sets=candidate_answer_sets,
    )
    status = "pass"

    payload = {
        "schema_version": 1,
        "run_id": request.get("run_id"),
        "pack_id": request.get("pack_id", requested_pack_id),
        "status": status,
        "aggregate_metrics": {
            "target_quality": evaluation.get("target_quality", {}),
            "interference": evaluation.get("interference", {}),
            "by_probe_family": evaluation.get("by_probe_family", {}),
            "retention_by_lag": evaluation.get("retention_by_lag", {}),
            "baseline_reference": {
                "target_quality": baseline_reference.get("target_quality", {}),
                "interference": baseline_reference.get("interference", {}),
                "by_probe_family": baseline_reference.get("by_probe_family", {}),
                "retention_by_lag": baseline_reference.get("retention_by_lag", {}),
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_data_registry() -> dict[str, Any]:
    return _load_json(DATA_REGISTRY_PATH)


def default_confirmation_pack_id() -> str:
    registry = load_data_registry()
    active = registry.get("active", {})
    pack_id = active.get("confirmation_pack")
    if isinstance(pack_id, str) and pack_id:
        return pack_id
    raise RuntimeError("data registry does not define active.confirmation_pack")


def confirmation_contract_paths(pack_id: str | None = None) -> tuple[Path, Path]:
    resolved_pack_id = pack_id or default_confirmation_pack_id()
    registry = load_data_registry()
    packs = registry.get("packs", {})
    entry = packs.get(resolved_pack_id)
    if not isinstance(entry, Mapping):
        raise RuntimeError(f"confirmation pack {resolved_pack_id!r} missing from data registry")
    summary_path = entry.get("summary_path")
    hash_path = entry.get("hash_path")
    if not isinstance(summary_path, str) or not isinstance(hash_path, str):
        raise RuntimeError(
            f"confirmation pack {resolved_pack_id!r} is missing summary_path or hash_path"
        )
    return REPO_ROOT / summary_path, REPO_ROOT / hash_path


def default_confirm_pack_path_for(pack_id: str) -> Path:
    return DEFAULT_HOST_LOCAL_CONFIRM_ROOT / f"{pack_id}.json"


if __name__ == "__main__":
    raise SystemExit(main())
