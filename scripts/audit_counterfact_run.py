#!/usr/bin/env python3
"""Build a sanitized per-case audit for a CounterFact bootstrap run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def exact_rate(records: Iterable[dict[str, Any]], key: str) -> float:
    values = [bool(record.get(key)) for record in records]
    return float(mean(values)) if values else 0.0


def case_id_for(example: dict[str, Any], fallback: int) -> str:
    metadata = example.get("metadata") or {}
    for key in ("source_case_id", "counterfact_case_id", "case_id"):
        value = metadata.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    return str(fallback)


def visible_records_from_artifact(
    *,
    artifact_path: Path,
    visible_pack_path: Path,
) -> list[dict[str, Any]]:
    artifact = load_json(artifact_path)
    pack = load_json(visible_pack_path)
    examples = pack.get("examples") or []
    evaluation = artifact.get("evaluation") or {}
    predictions = evaluation.get("predictions") or []
    per_example = evaluation.get("per_example") or []
    if len(examples) != len(per_example):
        raise ValueError(
            f"visible example count mismatch: pack={len(examples)} per_example={len(per_example)}"
        )
    if predictions and len(predictions) != len(per_example):
        raise ValueError(
            f"visible prediction count mismatch: predictions={len(predictions)} per_example={len(per_example)}"
        )

    records: list[dict[str, Any]] = []
    for idx, (example, metrics) in enumerate(zip(examples, per_example)):
        metadata = example.get("metadata") or {}
        record = {
            "split": "visible_dev",
            "index": idx,
            "example_id": example.get("id"),
            "case_id": case_id_for(example, idx),
            "probe_family": metadata.get("probe_family"),
            "relation_id": metadata.get("relation_id"),
            "subject": metadata.get("subject"),
            "target_new": metadata.get("target_new"),
            "target_true": metadata.get("target_true"),
            "target_exact_match": bool(metrics.get("target_exact_match")),
            "anchor_exact_match": bool(metrics.get("anchor_exact_match")),
            "target_prediction": predictions[idx] if predictions else None,
            "anchor_prediction": None,
        }
        records.append(record)
    return records


def sanitize_records_from_eval_result(
    *,
    pack_path: Path,
    eval_result: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    pack = load_json(pack_path)
    examples = pack.get("examples") or []
    predictions = eval_result.get("predictions") or []
    per_example = eval_result.get("per_example") or []
    if len(examples) != len(per_example):
        raise ValueError(
            f"locked example count mismatch: pack={len(examples)} per_example={len(per_example)}"
        )
    if predictions and len(predictions) != len(per_example):
        raise ValueError(
            f"locked prediction count mismatch: predictions={len(predictions)} per_example={len(per_example)}"
        )

    records: list[dict[str, Any]] = []
    for idx, (example, metrics) in enumerate(zip(examples, per_example)):
        metadata = example.get("metadata") or {}
        records.append(
            {
                "split": "locked_confirmation",
                "index": idx,
                "example_id": example.get("id"),
                "case_id": case_id_for(example, idx),
                "probe_family": metadata.get("probe_family"),
                "relation_id": metadata.get("relation_id"),
                "subject": metadata.get("subject"),
                "target_new": metadata.get("target_new"),
                "target_true": metadata.get("target_true"),
                "target_exact_match": bool(metrics.get("target_exact_match")),
                "anchor_exact_match": bool(metrics.get("anchor_exact_match")),
                "target_prediction": predictions[idx] if predictions else None,
                "anchor_prediction": None,
            }
        )
    return records


def candidate_answer_sets_for_version(train_generator_version: str | None) -> dict[str, tuple[str, ...]]:
    if not train_generator_version:
        return {}
    from method import load_generator_spec
    from scripts.run_bootstrap_baseline import build_candidate_answer_sets, registry_path

    generator = load_generator_spec(registry_path("train_generators", train_generator_version))
    return build_candidate_answer_sets(generator)


def audit_pack_with_adapter(
    *,
    model_id: str,
    adapter_dir: Path,
    pack_path: Path,
    split: str,
    max_new_tokens: int,
    method_family: str,
    train_generator_version: str | None,
) -> list[dict[str, Any]]:
    from eval.runner import load_eval_batch
    from scripts.profile_visible_dev import default_exact_match_score
    from scripts.run_bootstrap_baseline import (
        load_runtime_stack,
        policy_for_method_family,
        predict_short_answer,
    )

    stack = load_runtime_stack(model_id, adapter_dir=adapter_dir, train_mode=False)
    method_policy = policy_for_method_family(method_family)
    candidate_answer_sets = candidate_answer_sets_for_version(train_generator_version)
    batch = load_eval_batch(pack_path)

    records: list[dict[str, Any]] = []
    for idx, example in enumerate(batch.examples):
        target_prediction = predict_short_answer(
            stack=stack,
            model=stack.model,
            prompt=example.prompt,
            max_new_tokens=max_new_tokens,
            method_policy=method_policy,
            candidate_answer_sets=candidate_answer_sets,
        )
        anchor_prediction = None
        anchor_exact = False
        if example.anchor_target is not None:
            anchor_prediction = predict_short_answer(
                stack=stack,
                model=stack.model,
                prompt=example.anchor_prompt or example.prompt,
                max_new_tokens=max_new_tokens,
                method_policy=method_policy,
                candidate_answer_sets=candidate_answer_sets,
            )
            anchor_exact = default_exact_match_score(anchor_prediction, example.anchor_target) == 0.0

        metadata = example.metadata or {}
        records.append(
            {
                "split": split,
                "index": idx,
                "example_id": example.example_id,
                "case_id": case_id_for({"metadata": metadata}, idx),
                "probe_family": metadata.get("probe_family"),
                "relation_id": metadata.get("relation_id"),
                "subject": metadata.get("subject"),
                "target_new": metadata.get("target_new"),
                "target_true": metadata.get("target_true"),
                "target_exact_match": default_exact_match_score(
                    target_prediction, example.target
                )
                == 0.0,
                "anchor_exact_match": anchor_exact,
                "target_prediction": target_prediction,
                "anchor_prediction": anchor_prediction,
            }
        )
    return records


def family_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[str(record.get("probe_family"))].append(record)
        by_relation[str(record.get("relation_id"))].append(record)
        by_case[record_case_id(record)].append(record)

    def summarise_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(group),
            "target_exact": exact_rate(group, "target_exact_match"),
            "anchor_exact": exact_rate(group, "anchor_exact_match"),
            "joint_exact": exact_rate(
                [
                    {
                        "joint": bool(r.get("target_exact_match"))
                        and bool(r.get("anchor_exact_match"))
                    }
                    for r in group
                ],
                "joint",
            ),
        }

    case_rows = []
    for case_id, group in sorted(by_case.items(), key=lambda kv: kv[0]):
        first = group[0]
        case_rows.append(
            {
                "case_id": case_id,
                "relation_id": first.get("relation_id"),
                "subject": first.get("subject"),
                "target_new": first.get("target_new"),
                "target_true": first.get("target_true"),
                "probe_count": len(group),
                "target_exact": exact_rate(group, "target_exact_match"),
                "anchor_exact": exact_rate(group, "anchor_exact_match"),
                "predictions": [
                    {
                        "probe_family": r.get("probe_family"),
                        "target_exact_match": r.get("target_exact_match"),
                        "anchor_exact_match": r.get("anchor_exact_match"),
                        "target_prediction": r.get("target_prediction"),
                        "anchor_prediction": r.get("anchor_prediction"),
                    }
                    for r in group
                ],
            }
        )

    relation_rows = []
    for relation_id, group in sorted(by_relation.items(), key=lambda kv: kv[0]):
        row = {"relation_id": relation_id}
        row.update(summarise_group(group))
        row["case_count"] = len({str(r.get("case_id")) for r in group})
        relation_rows.append(row)
    relation_rows.sort(key=lambda row: (row["target_exact"] + row["anchor_exact"], row["n"]))

    target_prediction_counts = Counter(
        str(r.get("target_prediction") or "").strip()
        for r in records
        if r.get("target_prediction") is not None
    )
    anchor_prediction_counts = Counter(
        str(r.get("anchor_prediction") or "").strip()
        for r in records
        if r.get("anchor_prediction") is not None
    )
    target_confusions = Counter()
    anchor_confusions = Counter()
    for record in records:
        target_pred = str(record.get("target_prediction") or "").strip().lower()
        anchor_pred = str(record.get("anchor_prediction") or "").strip().lower()
        target_new = str(record.get("target_new") or "").strip().lower()
        target_true = str(record.get("target_true") or "").strip().lower()
        if target_pred == target_true and target_true:
            target_confusions["target_predicted_old_fact"] += 1
        if anchor_pred == target_new and target_new:
            anchor_confusions["anchor_predicted_edit_fact"] += 1

    return {
        "overall": summarise_group(records),
        "by_family": {family: summarise_group(group) for family, group in sorted(by_family.items())},
        "worst_relations": relation_rows[:12],
        "case_rows": case_rows,
        "top_target_predictions": target_prediction_counts.most_common(20),
        "top_anchor_predictions": anchor_prediction_counts.most_common(20),
        "confusion_counts": {
            "target_confusions": dict(target_confusions),
            "anchor_confusions": dict(anchor_confusions),
        },
    }


def record_case_id(record: Mapping[str, Any]) -> str:
    example_id = record.get("example_id")
    if isinstance(example_id, str):
        match = re.match(r"counterfact-(\d+)-", example_id)
        if match:
            return match.group(1)
    value = record.get("case_id")
    return str(value if value is not None else "")


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_json(Path(args.artifact))
    method = artifact.get("method") or {}
    data = artifact.get("data") or {}
    method_family = str(method.get("method_family") or args.method_family or "").strip()
    model_id = str(method.get("base_model") or args.model_id or "").strip()
    train_generator_version = str(
        data.get("train_generator_version") or args.train_generator_version or ""
    ).strip()
    if not method_family:
        raise ValueError("artifact missing method.method_family; pass --method-family")
    if not model_id:
        raise ValueError("artifact missing method.base_model; pass --model-id")

    if args.adapter_dir and args.recompute_visible:
        visible_records = audit_pack_with_adapter(
            model_id=model_id,
            adapter_dir=Path(args.adapter_dir),
            pack_path=Path(args.visible_pack),
            split="visible_dev",
            max_new_tokens=args.max_new_tokens,
            method_family=method_family,
            train_generator_version=train_generator_version or None,
        )
    else:
        visible_records = visible_records_from_artifact(
            artifact_path=Path(args.artifact),
            visible_pack_path=Path(args.visible_pack),
        )

    locked_records: list[dict[str, Any]] = []
    if args.locked_case_metrics:
        locked_payload = load_json(Path(args.locked_case_metrics))
        locked_records = locked_payload.get("records") or locked_payload
    elif args.locked_pack and args.adapter_dir:
        locked_records = audit_pack_with_adapter(
            model_id=model_id,
            adapter_dir=Path(args.adapter_dir),
            pack_path=Path(args.locked_pack),
            split="locked_confirmation",
            max_new_tokens=args.max_new_tokens,
            method_family=method_family,
            train_generator_version=train_generator_version or None,
        )
        if args.write_locked_case_metrics:
            write_json(
                Path(args.write_locked_case_metrics),
                {
                    "run_id": args.run_id,
                    "records": locked_records,
                },
            )

    audit = {
        "run_id": args.run_id,
        "artifact": str(args.artifact),
        "visible_pack": str(args.visible_pack),
        "locked_pack": str(args.locked_pack) if args.locked_pack else None,
        "visible": family_summary(visible_records),
        "locked_confirmation": family_summary(locked_records) if locked_records else None,
        "notes": [
            "Prompt text is intentionally omitted so locked confirmation prompts are not copied into repo artifacts.",
            "Predictions are included because they are needed to diagnose answer confusions and mode collapse.",
        ],
    }
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--visible-pack", required=True)
    parser.add_argument("--output")
    parser.add_argument("--locked-case-metrics")
    parser.add_argument("--locked-pack")
    parser.add_argument("--write-locked-case-metrics")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--recompute-visible", action="store_true")
    parser.add_argument("--method-family")
    parser.add_argument("--train-generator-version")
    parser.add_argument("--model-id")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    audit = build_audit(args)
    if args.output:
        write_json(Path(args.output), audit)

    visible = audit["visible"]["overall"]
    print(
        "visible "
        f"target={visible['target_exact']:.4f} "
        f"anchor={visible['anchor_exact']:.4f} "
        f"joint={visible['joint_exact']:.4f}"
    )
    if audit["locked_confirmation"]:
        locked = audit["locked_confirmation"]["overall"]
        print(
            "locked "
            f"target={locked['target_exact']:.4f} "
            f"anchor={locked['anchor_exact']:.4f} "
            f"joint={locked['joint_exact']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
