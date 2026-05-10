#!/usr/bin/env python3
"""Freeze one run spec into experiments/specs/ for auditability."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from method.selected_stack import load_selected_pilot_pair


SCHEMA_VERSION = 1
DEFAULT_RUNTIME_PYTHON = "~/shared/envs/projects/autoresearch-continual-learning-env/bin/python"


def main() -> int:
    selected_pair = load_selected_pilot_pair(REPO_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-class", required=True)
    parser.add_argument("--method-family", required=True)
    parser.add_argument(
        "--base-model",
        default=selected_pair["model_id"],
        help=f"Base model for the run spec (default: selected pilot {selected_pair['model_id']}).",
    )
    parser.add_argument("--task-mode", default="method_iteration")
    parser.add_argument("--command")
    parser.add_argument(
        "--command-mode",
        choices=["bootstrap_baseline_v1", "mainline_hyper_lora_v1"],
        help="Optional helper for known runnable command templates.",
    )
    parser.add_argument(
        "--runtime-python",
        default=DEFAULT_RUNTIME_PYTHON,
        help="Python executable used by templated command modes.",
    )
    parser.add_argument("--train-generator-version", required=True)
    parser.add_argument("--development-pack", required=True)
    parser.add_argument("--confirmation-pack-summary", required=True)
    parser.add_argument("--comparison-scope", required=True)
    parser.add_argument("--selection-policy", required=True)
    parser.add_argument("--declared-trainable-params", required=True, type=int)
    parser.add_argument(
        "--trainable-parameter-tolerance",
        type=int,
        default=0,
        help="Allowed absolute difference between declared and observed trainable parameter counts.",
    )
    parser.add_argument("--editable-path", action="append", default=[])
    parser.add_argument("--helper-model", action="append", default=[])
    parser.add_argument("--baseline-ref")
    parser.add_argument("--parent-champion")
    parser.add_argument("--model-lane")
    parser.add_argument("--lane-role")
    parser.add_argument("--lane-surface-name")
    parser.add_argument("--lane-decision-mode")
    parser.add_argument("--graduation-target-lane")
    parser.add_argument("--surrogate-gate-json")
    parser.add_argument("--surrogate-source-run-id")
    parser.add_argument("--frozen-base-model", action="store_true")
    parser.add_argument("--uses-retrieval", action="store_true")
    parser.add_argument("--uses-postprocessor", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    champion = load_json(repo_root / "experiments" / "champion.json")
    baseline_ref = args.baseline_ref or champion_ref(champion)
    parent_champion = args.parent_champion or champion_ref(champion)

    spec = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "task_mode": args.task_mode,
        "run_class": args.run_class,
        "method_family": args.method_family,
        "base_model": args.base_model,
        "command": resolve_command(args),
        "parent_champion": parent_champion,
        "comparison_scope": args.comparison_scope,
        "baseline_ref": baseline_ref,
        "created_at_utc": utc_now(),
        "editable_surface": {
            "paths": sorted(set(args.editable_path)),
            "selection_policy": args.selection_policy,
        },
        "declared_capacity": {
            "frozen_base_model": args.frozen_base_model,
            "trainable_parameter_count": args.declared_trainable_params,
            "trainable_parameter_tolerance": args.trainable_parameter_tolerance,
            "uses_retrieval": args.uses_retrieval,
            "helper_models": sorted(set(args.helper_model)),
            "uses_postprocessor": args.uses_postprocessor,
        },
        "data": {
            "train_generator_version": args.train_generator_version,
            "development_pack": args.development_pack,
            "confirmation_pack_summary": args.confirmation_pack_summary,
        },
    }
    model_lane = build_model_lane_record(args)
    if model_lane:
        spec["model_lane"] = model_lane

    spec_path = repo_root / "experiments" / "specs" / f"{args.run_id}.yaml"
    if spec_path.exists() and not args.force:
        print(f"Refusing to overwrite existing spec: {spec_path}", file=sys.stderr)
        return 1

    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    print(spec_path.relative_to(repo_root).as_posix())
    return 0


def resolve_command(args: argparse.Namespace) -> str:
    if args.command:
        return args.command
    if args.command_mode == "bootstrap_baseline_v1":
        run_dir_expr = f'${{AUTORESEARCH_REMOTE_RUN_DIR:-artifacts/runs/{args.run_id}}}'
        return (
            f"{args.runtime_python} scripts/run_bootstrap_baseline.py "
            f"--spec experiments/specs/{args.run_id}.yaml "
            f"--run-dir {run_dir_expr}"
        )
    if args.command_mode == "mainline_hyper_lora_v1":
        run_dir_expr = f'${{AUTORESEARCH_REMOTE_RUN_DIR:-artifacts/runs/{args.run_id}}}'
        return (
            f"{args.runtime_python} scripts/run_mainline_hyper_lora.py "
            f"--spec experiments/specs/{args.run_id}.yaml "
            f"--run-dir {run_dir_expr}"
        )
    raise RuntimeError("Either --command or --command-mode must be provided.")


def build_model_lane_record(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.model_lane:
        return None
    payload: dict[str, Any] = {"name": args.model_lane}
    optional_fields = {
        "role": args.lane_role,
        "surface_name": args.lane_surface_name,
        "decision_mode": args.lane_decision_mode,
        "graduation_target_lane": args.graduation_target_lane,
        "source_run_id": args.surrogate_source_run_id,
    }
    for key, value in optional_fields.items():
        if isinstance(value, str) and value.strip():
            payload[key] = value
    if args.surrogate_gate_json:
        surrogate_gate = json.loads(args.surrogate_gate_json)
        if not isinstance(surrogate_gate, dict):
            raise ValueError("--surrogate-gate-json must decode to an object")
        payload["surrogate_gate"] = surrogate_gate
    return payload


def champion_ref(champion: dict[str, Any]) -> str:
    current = champion.get("current_champion")
    if isinstance(current, dict):
        run_id = current.get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id
    return "no_champion_recorded"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
