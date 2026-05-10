#!/usr/bin/env python3
"""Run the first real bootstrap baseline and emit a schema-valid artifact."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.runner import evaluate_examples, load_eval_batch
from eval.aggregates import build_continual_learning_family
from method import BoundedArtifactTrainer, DeterministicEpisodeSampler, RunBudget, TrainerContext
from method import load_generator_spec
from method.baselines import SequentialLoRABaseline, SequentialLoRAConfig
from method.baselines.seq_lora_ft import model_shape_from_hf_config, observed_capacity_metadata
from scripts.profile_visible_dev import (
    default_exact_match_score,
    extract_short_answer,
    infer_model_family,
    preferred_dtype,
    render_prompt,
)


DEFAULT_RUNTIME_PYTHON = "~/shared/envs/projects/autoresearch-continual-learning-env/bin/python"


@dataclass
class RuntimeStack:
    """Inference/training runtime for one base model family."""

    torch: Any
    model: Any
    processor: Any | None
    tokenizer: Any | None
    device: str
    model_family: str


@dataclass(frozen=True)
class BaselineMethodPolicy:
    """Execution policy for one baseline family."""

    method_family: str
    state_consistent_replay: bool = False
    answer_selection: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    learning_rate: float = 5e-4
    episode_passes: int = 8
    target_repeat_count: int = 2
    anchor_repeat_count: int = 2
    history_replay_episodes: int = 6
    target_history_replay_episodes: int | None = None
    anchor_history_replay_episodes: int | None = None
    history_lm_replay: bool = True
    current_lm_replay: bool = True
    final_consolidation: bool = False
    final_target_passes: int = 0
    final_anchor_passes: int = 0
    accumulate_supervision: bool = False
    supervision_batch_size: int = 0
    mixed_final_consolidation: bool = False
    relation_aware_replay: bool = False
    max_decode_tokens: int = 24
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayRecord:
    """One replayable current-state record."""

    key: str
    family: str
    kind: str
    lm_text: str
    supervision_examples: tuple[tuple[str, str], ...]
    updated_at: int


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument(
        "--run-dir",
        default=os.environ.get("AUTORESEARCH_REMOTE_RUN_DIR"),
        help="Output directory for artifact and adapter files.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--episode-passes", type=int)
    parser.add_argument(
        "--target-repeat-count",
        type=int,
        default=None,
        help="How many times to duplicate the primary target supervision prompt per episode.",
    )
    parser.add_argument(
        "--anchor-repeat-count",
        type=int,
        default=None,
        help="How many times to duplicate each retention/anchor supervision prompt per episode.",
    )
    parser.add_argument(
        "--history-replay-episodes",
        type=int,
        default=None,
        help="How many earlier edit episodes to rehearse alongside the current one.",
    )
    parser.add_argument(
        "--history-lm-replay",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to replay prior episode bulletin text alongside prior supervision prompts.",
    )
    parser.add_argument(
        "--current-lm-replay",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to include the current episode bulletin text in the LM replay loss.",
    )
    args = parser.parse_args()

    spec_path = resolve_path(args.spec)
    spec = load_json(spec_path)
    run_id = require_string(spec, "run_id")
    method_family = require_string(spec, "method_family")
    method_policy = policy_for_method_family(method_family)
    learning_rate = (
        args.learning_rate if args.learning_rate is not None else method_policy.learning_rate
    )
    episode_passes = (
        args.episode_passes if args.episode_passes is not None else method_policy.episode_passes
    )
    target_repeat_count = (
        args.target_repeat_count
        if args.target_repeat_count is not None
        else method_policy.target_repeat_count
    )
    anchor_repeat_count = (
        args.anchor_repeat_count
        if args.anchor_repeat_count is not None
        else method_policy.anchor_repeat_count
    )
    history_replay_episodes = (
        args.history_replay_episodes
        if args.history_replay_episodes is not None
        else method_policy.history_replay_episodes
    )
    target_history_replay_episodes = (
        method_policy.target_history_replay_episodes
        if method_policy.target_history_replay_episodes is not None
        else history_replay_episodes
    )
    anchor_history_replay_episodes = (
        method_policy.anchor_history_replay_episodes
        if method_policy.anchor_history_replay_episodes is not None
        else history_replay_episodes
    )
    history_lm_replay = (
        args.history_lm_replay
        if args.history_lm_replay is not None
        else method_policy.history_lm_replay
    )
    current_lm_replay = (
        args.current_lm_replay
        if args.current_lm_replay is not None
        else method_policy.current_lm_replay
    )
    run_dir = resolve_run_dir(args.run_dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = run_dir / "artifact.json"
    adapter_dir = run_dir / "adapter"
    training_summary_path = run_dir / "training_summary.json"

    started_at = time.perf_counter()
    preflight = run_workspace_preflight(require_string(spec, "task_mode"))

    generator_path = registry_path("train_generators", spec["data"]["train_generator_version"])
    development_pack_path = registry_path("packs", spec["data"]["development_pack"])
    generator = load_generator_spec(generator_path)
    sampler = DeterministicEpisodeSampler(generator, seed=0, shuffle=False)
    candidate_answer_sets = build_candidate_answer_sets(generator)

    stack = load_runtime_stack(spec["base_model"], adapter_dir=None, train_mode=True)
    baseline_reference = evaluate_visible_pack(
        stack=stack,
        model=stack.model,
        development_pack_path=development_pack_path,
        max_new_tokens=args.max_new_tokens,
        method_policy=method_policy,
        candidate_answer_sets=candidate_answer_sets,
    )
    baseline = SequentialLoRABaseline(
        SequentialLoRAConfig(
            surface_name=surface_name_for_spec(spec),
            rank=method_policy.lora_rank,
            alpha=method_policy.lora_alpha,
            method_family=method_family,
            uses_answer_selection=method_policy.answer_selection,
        ),
        model_shape=model_shape_from_hf_config(stack.model.config),
    )
    wrapped_model = baseline.wrap_model(stack.model)
    optimizer = build_optimizer(stack.torch, wrapped_model, learning_rate=learning_rate)

    budget = run_budget_for_class(spec["run_class"], len(generator.episodes))
    context = trainer_context_from_spec(spec_path=spec_path, spec=spec)
    trainer = BoundedArtifactTrainer(
        budget=budget,
        context=context,
        torch_module=stack.torch,
    )

    records_seed: list[dict[str, Any]] = []
    replay_history: list[dict[str, Any]] = []
    active_target_records: dict[str, ReplayRecord] = {}
    active_anchor_records: dict[str, ReplayRecord] = {}

    def step_fn(episode: Any, step_index: int) -> dict[str, Any]:
        replay_examples: list[tuple[str, str]] = []
        replay_texts: list[str] = []
        selected_target_keys: list[str] = []
        selected_anchor_keys: list[str] = []
        if method_policy.state_consistent_replay:
            target_record, anchor_records = build_state_replay_records(
                episode,
                target_repeat_count=target_repeat_count,
                anchor_repeat_count=anchor_repeat_count,
                updated_at=step_index,
                relation_aware_replay=method_policy.relation_aware_replay,
            )
            active_target_records[target_record.key] = target_record
            for anchor_record in anchor_records:
                active_anchor_records[anchor_record.key] = anchor_record

            selected_target_records, selected_anchor_records = select_state_replay_records(
                active_target_records=active_target_records,
                active_anchor_records=active_anchor_records,
                current_target_key=target_record.key,
                current_anchor_keys={record.key for record in anchor_records},
                target_limit=max(target_history_replay_episodes, 0),
                anchor_limit=max(anchor_history_replay_episodes, 0),
            )
            selected_target_keys = [record.key for record in selected_target_records]
            selected_anchor_keys = [record.key for record in selected_anchor_records]
            for replay_record in [*selected_target_records, *selected_anchor_records]:
                if history_lm_replay and replay_record.lm_text:
                    replay_texts.append(replay_record.lm_text)
                replay_examples.extend(replay_record.supervision_examples)
        else:
            replay_window = replay_history[-max(history_replay_episodes, 0) :]
            for replay_record in replay_window:
                replay_text = replay_record.get("fact_replay_text")
                if history_lm_replay and isinstance(replay_text, str) and replay_text.strip():
                    replay_texts.append(replay_text)
                examples = replay_record.get("supervision_examples")
                if isinstance(examples, list):
                    for prompt, answer in examples:
                        if (
                            isinstance(prompt, str)
                            and prompt.strip()
                            and isinstance(answer, str)
                            and answer.strip()
                        ):
                            replay_examples.append((prompt.strip(), answer.strip()))
        loss = train_one_episode(
            stack=stack,
            wrapped_model=wrapped_model,
            optimizer=optimizer,
            episode=episode,
            episode_passes=episode_passes,
            replay_examples=replay_examples,
            replay_texts=replay_texts,
            target_repeat_count=target_repeat_count,
            anchor_repeat_count=anchor_repeat_count,
            current_lm_replay=current_lm_replay,
            accumulate_supervision=method_policy.accumulate_supervision,
            supervision_batch_size=method_policy.supervision_batch_size,
        )
        replay_history.append(
            {
                "episode_id": episode.episode_id,
                "fact_replay_text": render_fact_replay_text(episode),
                "supervision_examples": build_episode_supervision_examples(
                    episode,
                    target_repeat_count=target_repeat_count,
                    anchor_repeat_count=anchor_repeat_count,
                ),
            }
        )
        record = {
            "episode_id": episode.episode_id,
            "loss": loss,
            "replay_examples": len(replay_examples),
            "replay_texts": len(replay_texts),
            "method_family": method_family,
            "state_consistent_replay": method_policy.state_consistent_replay,
            "lora_rank": method_policy.lora_rank,
            "lora_alpha": method_policy.lora_alpha,
            "learning_rate": learning_rate,
            "episode_passes": episode_passes,
            "target_repeat_count": target_repeat_count,
            "anchor_repeat_count": anchor_repeat_count,
            "history_replay_episodes": history_replay_episodes,
            "target_history_replay_episodes": target_history_replay_episodes,
            "anchor_history_replay_episodes": anchor_history_replay_episodes,
            "history_lm_replay": history_lm_replay,
            "current_lm_replay": current_lm_replay,
            "accumulate_supervision": method_policy.accumulate_supervision,
            "supervision_batch_size": method_policy.supervision_batch_size,
            "relation_aware_replay": method_policy.relation_aware_replay,
            "selected_replay_target_keys": selected_target_keys,
            "selected_replay_anchor_keys": selected_anchor_keys,
        }
        records_seed.append(record)
        return {
            "target_quality": {
                "episode_train_loss": loss,
            },
            "interference": {
                "anchor_count": float(len(episode.anchor_texts)),
            },
        }

    artifact, records = trainer.run(
        sampler.iter_episodes(limit=len(generator.episodes)),
        step_fn,
        integrity_flags={
            "preflight_passed": preflight["ok"],
            "immutable_hashes_verified": preflight["ok"],
            "shim_checks_passed": True,
        },
        observed_capacity=observed_capacity_metadata(
            wrapped_model,
            optimizer=optimizer,
        ),
    )

    consolidation_records: list[dict[str, Any]] = []
    if method_policy.final_consolidation:
        consolidation_records = run_final_consolidation(
            stack=stack,
            wrapped_model=wrapped_model,
            optimizer=optimizer,
            target_records=active_target_records,
            anchor_records=active_anchor_records,
            target_passes=method_policy.final_target_passes,
            anchor_passes=method_policy.final_anchor_passes,
            accumulate_supervision=method_policy.accumulate_supervision,
            supervision_batch_size=method_policy.supervision_batch_size,
            mixed_final_consolidation=method_policy.mixed_final_consolidation,
            started_at=started_at,
            max_runtime_seconds=budget.max_runtime_seconds,
        )

    evaluation = evaluate_visible_pack(
        stack=stack,
        model=wrapped_model,
        development_pack_path=development_pack_path,
        max_new_tokens=args.max_new_tokens,
        method_policy=method_policy,
        candidate_answer_sets=candidate_answer_sets,
    )
    wrapped_model.save_pretrained(adapter_dir)

    total_runtime = time.perf_counter() - started_at
    peak_vram_gb = current_peak_vram_gb(stack)
    cl_metrics = build_continual_learning_family(
        target_exact_match=float(evaluation["target_quality"].get("exact_match", 0.0)),
        anchor_exact_match=float(evaluation["interference"].get("anchor_exact_match", 0.0)),
        joint_success_rate=float(evaluation["interference"].get("joint_success_rate", 0.0)),
        baseline_target_exact_match=float(
            baseline_reference["target_quality"].get("exact_match", 0.0)
        ),
        baseline_anchor_exact_match=float(
            baseline_reference["interference"].get("anchor_exact_match", 0.0)
        ),
        num_update_episodes=len(generator.episodes),
    )
    artifact["metrics"]["continual_learning"] = cl_metrics
    artifact["metrics"]["target_quality"] = evaluation["target_quality"]
    artifact["metrics"]["interference"] = evaluation["interference"]
    artifact["metrics"]["cost"]["runtime_seconds"] = total_runtime
    artifact["metrics"]["cost"]["peak_vram_gb"] = peak_vram_gb
    artifact["supporting_files"] = {
        "adapter_dir": str(adapter_dir),
        "training_summary_path": str(training_summary_path),
    }
    artifact["evaluation"] = {
        **evaluation,
        "baseline_reference": baseline_reference,
    }

    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    training_summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "records": records,
                "episodes": records_seed,
                "consolidation": consolidation_records,
                "artifact_path": str(artifact_path),
                "adapter_dir": str(adapter_dir),
                "baseline_reference": baseline_reference,
                "evaluation": evaluation,
                "continual_learning_metrics": cl_metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "artifact_path": str(artifact_path),
                "adapter_dir": str(adapter_dir),
                "target_quality": evaluation["target_quality"],
                "interference": evaluation["interference"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def evaluate_saved_adapter(
    *,
    model_id: str,
    adapter_dir: str | Path,
    pack_path: str | Path,
    max_new_tokens: int = 24,
    method_family: str = "baseline_seq_lora_ft_v0",
    train_generator_version: str | None = None,
) -> dict[str, Any]:
    """Load a saved adapter and evaluate it on one confirmation pack."""

    stack = load_runtime_stack(model_id, adapter_dir=Path(adapter_dir), train_mode=False)
    if train_generator_version is None:
        registry = load_json(REPO_ROOT / "data" / "registry.yaml")
        active = registry.get("active", {})
        raw_version = active.get("train_generator")
        train_generator_version = str(raw_version) if isinstance(raw_version, str) and raw_version else None
    candidate_answer_sets: dict[str, tuple[str, ...]] = {}
    if train_generator_version:
        generator_path = registry_path("train_generators", train_generator_version)
        generator = load_generator_spec(generator_path)
        candidate_answer_sets = build_candidate_answer_sets(generator)
    method_policy = policy_for_method_family(method_family)
    return evaluate_pack_with_runtime_stack(
        stack=stack,
        model=stack.model,
        pack_path=Path(pack_path),
        max_new_tokens=max_new_tokens,
        method_policy=method_policy,
        candidate_answer_sets=candidate_answer_sets,
    )


def evaluate_visible_pack(
    *,
    stack: RuntimeStack,
    model: Any,
    development_pack_path: Path,
    max_new_tokens: int,
    method_policy: BaselineMethodPolicy,
    candidate_answer_sets: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    return evaluate_pack_with_runtime_stack(
        stack=stack,
        model=model,
        pack_path=development_pack_path,
        max_new_tokens=max_new_tokens,
        method_policy=method_policy,
        candidate_answer_sets=candidate_answer_sets,
    )


def evaluate_pack_with_runtime_stack(
    *,
    stack: RuntimeStack,
    model: Any,
    pack_path: Path,
    max_new_tokens: int,
    method_policy: BaselineMethodPolicy,
    candidate_answer_sets: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    batch = load_eval_batch(pack_path)

    def predict_fn(prompt: str) -> str:
        return predict_short_answer(
            stack=stack,
            model=model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            method_policy=method_policy,
            candidate_answer_sets=candidate_answer_sets,
        )

    result = evaluate_examples(
        batch,
        predict_fn=predict_fn,
        score_fn=default_exact_match_score,
    )
    return {
        "pack_id": batch.pack_id,
        "predictions": list(result.predictions),
        "per_example": list(result.per_example),
        "target_quality": result.target_quality,
        "interference": result.interference,
        "by_probe_family": result.by_probe_family,
        "retention_by_lag": result.retention_by_lag,
    }


def predict_short_answer(
    *,
    stack: RuntimeStack,
    model: Any,
    prompt: str,
    max_new_tokens: int,
    method_policy: BaselineMethodPolicy,
    candidate_answer_sets: Mapping[str, tuple[str, ...]],
) -> str:
    rendered = render_prompt(prompt, model_family=stack.model_family)
    if method_policy.answer_selection:
        candidates = candidate_answers_for_prompt(prompt, candidate_answer_sets)
        if candidates:
            return select_answer_by_logprob(
                stack=stack,
                model=model,
                rendered_prompt=rendered,
                candidates=candidates,
            )
    generation_limit = min(max_new_tokens, method_policy.max_decode_tokens)
    tokenizer = tokenizer_for_stack(stack)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", eos_token_id)

    if stack.model_family == "gemma3":
        batch = stack.processor(text=rendered, return_tensors="pt")
        batch = {key: value.to(stack.device) for key, value in batch.items()}
        generation_kwargs = {
            "max_new_tokens": generation_limit,
            "do_sample": False,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "repetition_penalty": 1.1,
            "renormalize_logits": True,
        }
        with stack.torch.no_grad():
            output = model.generate(**batch, **generation_kwargs)
        input_len = int(batch["input_ids"].shape[-1])
        generated = output[0][input_len:]
        decoded = stack.processor.decode(generated, skip_special_tokens=True)
    else:
        batch = stack.tokenizer(rendered, return_tensors="pt")
        batch = {key: value.to(stack.device) for key, value in batch.items()}
        generation_kwargs = {
            "max_new_tokens": generation_limit,
            "do_sample": False,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "repetition_penalty": 1.1,
            "renormalize_logits": True,
        }
        with stack.torch.no_grad():
            output = model.generate(**batch, **generation_kwargs)
        input_len = int(batch["input_ids"].shape[-1])
        generated = output[0][input_len:]
        decoded = stack.tokenizer.decode(generated, skip_special_tokens=True)

    return extract_short_answer(decoded)


def train_one_episode(
    *,
    stack: RuntimeStack,
    wrapped_model: Any,
    optimizer: Any,
    episode: Any,
    episode_passes: int,
    replay_examples: list[tuple[str, str]] | None = None,
    replay_texts: list[str] | None = None,
    target_repeat_count: int = 2,
    anchor_repeat_count: int = 2,
    current_lm_replay: bool = True,
    accumulate_supervision: bool = False,
    supervision_batch_size: int = 0,
) -> float:
    if episode_passes <= 0:
        raise ValueError("episode_passes must be positive")

    supervision_examples = build_episode_supervision_examples(
        episode,
        target_repeat_count=target_repeat_count,
        anchor_repeat_count=anchor_repeat_count,
    )
    if not supervision_examples:
        raise RuntimeError(f"episode {episode.episode_id} yielded no supervision examples")

    wrapped_model.train()
    observed_losses: list[float] = []
    replay_examples = replay_examples or []
    replay_texts = replay_texts or []
    for _ in range(episode_passes):
        if current_lm_replay:
            observed_losses.append(
                train_language_model_text(
                    stack=stack,
                    wrapped_model=wrapped_model,
                    optimizer=optimizer,
                    text=render_fact_replay_text(episode),
                )
            )
        if accumulate_supervision:
            ordered_examples = round_robin_examples(supervision_examples, replay_examples)
            observed_losses.extend(
                train_supervised_examples_accumulated(
                    stack=stack,
                    wrapped_model=wrapped_model,
                    optimizer=optimizer,
                    examples=ordered_examples,
                    max_examples_per_step=supervision_batch_size,
                )
            )
        else:
            for prompt, answer in supervision_examples:
                observed_losses.append(
                    train_supervised_example(
                        stack=stack,
                        wrapped_model=wrapped_model,
                        optimizer=optimizer,
                        prompt=prompt,
                        answer=answer,
                    )
                )
        for replay_text in replay_texts:
            observed_losses.append(
                train_language_model_text(
                    stack=stack,
                    wrapped_model=wrapped_model,
                    optimizer=optimizer,
                    text=replay_text,
                )
            )
        if not accumulate_supervision:
            for prompt, answer in replay_examples:
                observed_losses.append(
                    train_supervised_example(
                        stack=stack,
                        wrapped_model=wrapped_model,
                        optimizer=optimizer,
                        prompt=prompt,
                        answer=answer,
                    )
                )

    return sum(observed_losses) / len(observed_losses)


def run_final_consolidation(
    *,
    stack: RuntimeStack,
    wrapped_model: Any,
    optimizer: Any,
    target_records: Mapping[str, ReplayRecord],
    anchor_records: Mapping[str, ReplayRecord],
    target_passes: int,
    anchor_passes: int,
    accumulate_supervision: bool = False,
    supervision_batch_size: int = 0,
    mixed_final_consolidation: bool = False,
    started_at: float,
    max_runtime_seconds: float,
) -> list[dict[str, Any]]:
    """Run a bounded end-of-stream rehearsal pass over active current-state facts."""

    records: list[dict[str, Any]] = []
    if target_passes <= 0 and anchor_passes <= 0:
        return records

    ordered_targets = sorted(target_records.values(), key=lambda record: record.key)
    ordered_anchors = sorted(anchor_records.values(), key=lambda record: record.key)
    for pass_index in range(max(target_passes, anchor_passes)):
        pass_started = time.perf_counter()
        target_losses: list[float] = []
        anchor_losses: list[float] = []
        stopped_for_budget = False

        if accumulate_supervision and mixed_final_consolidation:
            target_examples = (
                tagged_replay_supervision_examples(ordered_targets, kind="target")
                if pass_index < target_passes
                else []
            )
            anchor_examples = (
                tagged_replay_supervision_examples(ordered_anchors, kind="anchor")
                if pass_index < anchor_passes
                else []
            )
            tagged_examples = round_robin_examples(anchor_examples, target_examples)
            tagged_losses = train_tagged_supervised_examples_accumulated(
                stack=stack,
                wrapped_model=wrapped_model,
                optimizer=optimizer,
                examples=tagged_examples,
                max_examples_per_step=supervision_batch_size,
            )
            target_losses.extend(
                item["loss"] for item in tagged_losses if item["kind"] == "target"
            )
            anchor_losses.extend(
                item["loss"] for item in tagged_losses if item["kind"] == "anchor"
            )
        elif pass_index < anchor_passes:
            for replay_record in ordered_anchors:
                if time.perf_counter() - started_at >= max_runtime_seconds:
                    stopped_for_budget = True
                    break
                if accumulate_supervision:
                    anchor_losses.extend(
                        train_supervised_examples_accumulated(
                            stack=stack,
                            wrapped_model=wrapped_model,
                            optimizer=optimizer,
                            examples=list(replay_record.supervision_examples),
                            max_examples_per_step=supervision_batch_size,
                        )
                    )
                else:
                    anchor_losses.extend(
                        train_replay_supervision_examples(
                            stack=stack,
                            wrapped_model=wrapped_model,
                            optimizer=optimizer,
                            replay_record=replay_record,
                        )
                    )

        if not stopped_for_budget and pass_index < target_passes:
            for replay_record in ordered_targets:
                if time.perf_counter() - started_at >= max_runtime_seconds:
                    stopped_for_budget = True
                    break
                if accumulate_supervision:
                    target_losses.extend(
                        train_supervised_examples_accumulated(
                            stack=stack,
                            wrapped_model=wrapped_model,
                            optimizer=optimizer,
                            examples=list(replay_record.supervision_examples),
                            max_examples_per_step=supervision_batch_size,
                        )
                    )
                else:
                    target_losses.extend(
                        train_replay_supervision_examples(
                            stack=stack,
                            wrapped_model=wrapped_model,
                            optimizer=optimizer,
                            replay_record=replay_record,
                        )
                    )

        records.append(
            {
                "pass_index": pass_index,
                "target_records": len(ordered_targets) if pass_index < target_passes else 0,
                "anchor_records": len(ordered_anchors) if pass_index < anchor_passes else 0,
                "target_updates": len(target_losses),
                "anchor_updates": len(anchor_losses),
                "mean_target_loss": mean_or_none(target_losses),
                "mean_anchor_loss": mean_or_none(anchor_losses),
                "pass_runtime_seconds": time.perf_counter() - pass_started,
                "stopped_for_budget": stopped_for_budget,
            }
        )
        if stopped_for_budget:
            break
    return records


def train_replay_supervision_examples(
    *,
    stack: RuntimeStack,
    wrapped_model: Any,
    optimizer: Any,
    replay_record: ReplayRecord,
) -> list[float]:
    losses: list[float] = []
    for prompt, answer in replay_record.supervision_examples:
        losses.append(
            train_supervised_example(
                stack=stack,
                wrapped_model=wrapped_model,
                optimizer=optimizer,
                prompt=prompt,
                answer=answer,
            )
        )
    return losses


def tagged_replay_supervision_examples(
    replay_records: list[ReplayRecord],
    *,
    kind: str,
) -> list[tuple[str, str, str]]:
    examples: list[tuple[str, str, str]] = []
    for replay_record in replay_records:
        for prompt, answer in replay_record.supervision_examples:
            examples.append((kind, prompt, answer))
    return examples


def round_robin_examples(*groups: list[Any]) -> list[Any]:
    queues = [list(group) for group in groups if group]
    ordered: list[Any] = []
    while queues:
        next_queues: list[list[Any]] = []
        for queue in queues:
            if queue:
                ordered.append(queue.pop(0))
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return ordered


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def encode_training_text(*, stack: RuntimeStack, text: str) -> dict[str, Any]:
    if stack.model_family == "gemma3":
        batch = stack.processor(text=text, return_tensors="pt")
    else:
        batch = stack.tokenizer(text, return_tensors="pt")
    return {key: value.to(stack.device) for key, value in batch.items()}


def encode_supervised_example(
    *,
    stack: RuntimeStack,
    prompt: str,
    answer: str,
) -> tuple[dict[str, Any], Any]:
    rendered_prompt = render_prompt(prompt, model_family=stack.model_family)
    full_text = f"{rendered_prompt} {answer}{eos_token_text(stack)}".rstrip()
    prompt_batch = encode_training_text(stack=stack, text=rendered_prompt)
    full_batch = encode_training_text(stack=stack, text=full_text)
    labels = full_batch["input_ids"].clone()
    prompt_length = int(prompt_batch["input_ids"].shape[-1])
    labels[:, :prompt_length] = -100
    return full_batch, labels


def train_supervised_example(
    *,
    stack: RuntimeStack,
    wrapped_model: Any,
    optimizer: Any,
    prompt: str,
    answer: str,
) -> float:
    encoded, labels = encode_supervised_example(
        stack=stack,
        prompt=prompt,
        answer=answer,
    )
    optimizer.zero_grad(set_to_none=True)
    outputs = wrapped_model(**encoded, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def train_supervised_examples_accumulated(
    *,
    stack: RuntimeStack,
    wrapped_model: Any,
    optimizer: Any,
    examples: list[tuple[str, str]],
    max_examples_per_step: int = 0,
) -> list[float]:
    tagged_examples = [("supervision", prompt, answer) for prompt, answer in examples]
    tagged_losses = train_tagged_supervised_examples_accumulated(
        stack=stack,
        wrapped_model=wrapped_model,
        optimizer=optimizer,
        examples=tagged_examples,
        max_examples_per_step=max_examples_per_step,
    )
    return [item["loss"] for item in tagged_losses]


def train_tagged_supervised_examples_accumulated(
    *,
    stack: RuntimeStack,
    wrapped_model: Any,
    optimizer: Any,
    examples: list[tuple[str, str, str]],
    max_examples_per_step: int = 0,
) -> list[dict[str, Any]]:
    if not examples:
        return []
    wrapped_model.train()
    chunk_size = max_examples_per_step if max_examples_per_step > 0 else len(examples)
    observed: list[dict[str, Any]] = []
    for chunk_start in range(0, len(examples), chunk_size):
        chunk = examples[chunk_start : chunk_start + chunk_size]
        optimizer.zero_grad(set_to_none=True)
        scale = 1.0 / len(chunk)
        for kind, prompt, answer in chunk:
            encoded, labels = encode_supervised_example(
                stack=stack,
                prompt=prompt,
                answer=answer,
            )
            outputs = wrapped_model(**encoded, labels=labels)
            loss = outputs.loss
            (loss * scale).backward()
            observed.append(
                {
                    "kind": kind,
                    "loss": float(loss.detach().item()),
                }
            )
        optimizer.step()
    return observed


def train_language_model_text(
    *,
    stack: RuntimeStack,
    wrapped_model: Any,
    optimizer: Any,
    text: str,
) -> float:
    encoded = encode_training_text(stack=stack, text=text)
    labels = encoded["input_ids"].clone()
    optimizer.zero_grad(set_to_none=True)
    outputs = wrapped_model(**encoded, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def build_episode_supervision_examples(
    episode: Any,
    *,
    target_repeat_count: int = 2,
    anchor_repeat_count: int = 2,
) -> list[tuple[str, str]]:
    metadata = episode.metadata if isinstance(episode.metadata, Mapping) else {}
    examples: list[tuple[str, str]] = []
    target_repeat_count = max(1, target_repeat_count)
    anchor_repeat_count = max(1, anchor_repeat_count)

    target_prompt = metadata.get("target_prompt")
    if isinstance(target_prompt, str) and target_prompt.strip():
        examples.extend([(target_prompt.strip(), episode.target_text.strip())] * target_repeat_count)
    target_prompt_variants = metadata.get("target_prompt_variants")
    if isinstance(target_prompt_variants, list):
        for prompt in target_prompt_variants:
            if isinstance(prompt, str) and prompt.strip():
                examples.append((prompt.strip(), episode.target_text.strip()))

    retention_prompts = metadata.get("retention_prompts")
    retention_prompt_variants = metadata.get("retention_prompt_variants")
    if isinstance(retention_prompts, list):
        for index, (prompt, fact_text) in enumerate(zip(retention_prompts, episode.anchor_texts)):
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            answer = answer_from_fact_text(fact_text)
            if answer:
                examples.extend([(prompt.strip(), answer)] * anchor_repeat_count)
            if (
                isinstance(retention_prompt_variants, list)
                and index < len(retention_prompt_variants)
            ):
                variant_entry = retention_prompt_variants[index]
                if isinstance(variant_entry, str):
                    variant_values = [variant_entry]
                elif isinstance(variant_entry, list):
                    variant_values = variant_entry
                else:
                    variant_values = []
                for variant in variant_values:
                    if isinstance(variant, str) and variant.strip() and answer:
                        examples.append((variant.strip(), answer))

    if examples:
        return examples

    examples.append((episode.update_text.strip(), episode.target_text.strip()))
    return examples


def render_fact_replay_text(episode: Any) -> str:
    lines = [canonicalize_replay_fact_text(episode.update_text)]
    lines.extend(text.strip() for text in episode.anchor_texts if text.strip())
    return "Bulletins:\n" + "\n".join(lines) + "\n"


def build_state_replay_records(
    episode: Any,
    *,
    target_repeat_count: int,
    anchor_repeat_count: int,
    updated_at: int,
    relation_aware_replay: bool = False,
) -> tuple[ReplayRecord, list[ReplayRecord]]:
    metadata = episode.metadata if isinstance(episode.metadata, Mapping) else {}
    family = replay_family(metadata, relation_aware=relation_aware_replay)
    target_key = target_fact_key(metadata, episode)
    target_examples = tuple(
        build_target_supervision_examples(episode, target_repeat_count=target_repeat_count)
    )
    target_record = ReplayRecord(
        key=target_key,
        family=family,
        kind="target",
        lm_text=canonicalize_replay_fact_text(episode.update_text),
        supervision_examples=target_examples,
        updated_at=updated_at,
    )
    anchor_records: list[ReplayRecord] = []
    for anchor_record in build_anchor_replay_records(
        episode,
        anchor_repeat_count=anchor_repeat_count,
        updated_at=updated_at,
        relation_aware_replay=relation_aware_replay,
        family=family,
    ):
        anchor_records.append(anchor_record)
    return target_record, anchor_records


def build_target_supervision_examples(
    episode: Any,
    *,
    target_repeat_count: int,
) -> list[tuple[str, str]]:
    metadata = episode.metadata if isinstance(episode.metadata, Mapping) else {}
    examples: list[tuple[str, str]] = []
    repeat_count = max(1, target_repeat_count)
    target_prompt = metadata.get("target_prompt")
    if isinstance(target_prompt, str) and target_prompt.strip():
        examples.extend([(target_prompt.strip(), episode.target_text.strip())] * repeat_count)
    target_prompt_variants = metadata.get("target_prompt_variants")
    if isinstance(target_prompt_variants, list):
        for prompt in target_prompt_variants:
            if isinstance(prompt, str) and prompt.strip():
                examples.append((prompt.strip(), episode.target_text.strip()))
    return examples


def build_anchor_replay_records(
    episode: Any,
    *,
    anchor_repeat_count: int,
    updated_at: int,
    relation_aware_replay: bool = False,
    family: str | None = None,
) -> list[ReplayRecord]:
    metadata = episode.metadata if isinstance(episode.metadata, Mapping) else {}
    retention_prompts = metadata.get("retention_prompts")
    retention_prompt_variants = metadata.get("retention_prompt_variants")
    family = family or replay_family(metadata, relation_aware=relation_aware_replay)
    repeat_count = max(1, anchor_repeat_count)
    records: list[ReplayRecord] = []
    if not isinstance(retention_prompts, list):
        return records
    for index, (prompt, fact_text) in enumerate(zip(retention_prompts, episode.anchor_texts)):
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        answer = answer_from_fact_text(fact_text)
        if not answer:
            continue
        examples: list[tuple[str, str]] = [(prompt.strip(), answer)] * repeat_count
        if isinstance(retention_prompt_variants, list) and index < len(retention_prompt_variants):
            variant_entry = retention_prompt_variants[index]
            if isinstance(variant_entry, str):
                variant_values = [variant_entry]
            elif isinstance(variant_entry, list):
                variant_values = variant_entry
            else:
                variant_values = []
            for variant in variant_values:
                if isinstance(variant, str) and variant.strip():
                    examples.append((variant.strip(), answer))
        key = anchor_fact_key(metadata, prompt, fact_text, index, family=family)
        records.append(
            ReplayRecord(
                key=key,
                family=family,
                kind="anchor",
                lm_text=fact_text.strip(),
                supervision_examples=tuple(examples),
                updated_at=updated_at,
            )
        )
    return records


def select_state_replay_records(
    *,
    active_target_records: Mapping[str, ReplayRecord],
    active_anchor_records: Mapping[str, ReplayRecord],
    current_target_key: str,
    current_anchor_keys: set[str],
    target_limit: int,
    anchor_limit: int,
) -> tuple[list[ReplayRecord], list[ReplayRecord]]:
    target_candidates = [
        record
        for key, record in active_target_records.items()
        if key != current_target_key
    ]
    anchor_candidates = [
        record
        for key, record in active_anchor_records.items()
        if key not in current_anchor_keys
    ]
    return (
        select_balanced_replay_subset(target_candidates, limit=target_limit),
        select_balanced_replay_subset(anchor_candidates, limit=anchor_limit),
    )


def select_balanced_replay_subset(
    records: list[ReplayRecord],
    *,
    limit: int,
) -> list[ReplayRecord]:
    if limit <= 0 or not records:
        return []
    grouped: dict[str, list[ReplayRecord]] = defaultdict(list)
    for record in records:
        grouped[record.family].append(record)
    for family_records in grouped.values():
        family_records.sort(key=lambda record: record.updated_at, reverse=True)

    selected: list[ReplayRecord] = []
    family_names = sorted(grouped)
    take_newest = True
    while len(selected) < limit:
        made_progress = False
        for family in family_names:
            family_records = grouped[family]
            if not family_records:
                continue
            index = 0 if take_newest else -1
            selected.append(family_records.pop(index))
            made_progress = True
            if len(selected) >= limit:
                break
        if not made_progress:
            break
        take_newest = not take_newest
    return selected


def canonicalize_replay_fact_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return re.sub(r"\bnow\b", "currently", stripped, flags=re.IGNORECASE)


def replay_family(metadata: Mapping[str, Any], *, relation_aware: bool = False) -> str:
    if relation_aware:
        relation_id = metadata.get("relation_id")
        source_dataset = str(metadata.get("source_dataset") or "")
        if isinstance(relation_id, str) and relation_id and "counterfact" in source_dataset:
            return f"counterfact:{relation_id}"
    value = metadata.get("interference_family")
    if isinstance(value, str) and value:
        return value
    sequence_id = metadata.get("sequence_id")
    if isinstance(sequence_id, str) and sequence_id:
        return sequence_id.split("-")[0]
    return "generic"


def target_fact_key(metadata: Mapping[str, Any], episode: Any) -> str:
    explicit = metadata.get("target_fact_key")
    if isinstance(explicit, str) and explicit:
        return explicit
    sequence_id = metadata.get("sequence_id")
    if isinstance(sequence_id, str) and sequence_id:
        return sequence_id
    target_prompt = metadata.get("target_prompt")
    if isinstance(target_prompt, str) and target_prompt:
        return target_prompt
    return episode.episode_id


def anchor_fact_key(
    metadata: Mapping[str, Any],
    prompt: str,
    fact_text: str,
    index: int,
    *,
    family: str | None = None,
) -> str:
    anchor_keys = metadata.get("anchor_fact_keys")
    if isinstance(anchor_keys, list) and index < len(anchor_keys):
        explicit = anchor_keys[index]
        if isinstance(explicit, str) and explicit:
            return explicit
    answer = answer_from_fact_text(fact_text)
    base = prompt.strip() or fact_text.strip() or str(index)
    family = family or replay_family(metadata)
    return f"{family}:{base}:{answer}"


def build_candidate_answer_sets(generator: Any) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for episode in generator.episodes:
        metadata = episode.metadata if isinstance(episode.metadata, Mapping) else {}
        family = replay_family(metadata)
        grouped[family].add(episode.target_text.strip())
        for anchor_text in getattr(episode, "anchor_texts", ()):
            answer = answer_from_fact_text(anchor_text)
            if answer:
                grouped[family].add(answer)

    all_answers = sorted({answer for answers in grouped.values() for answer in answers})
    grouped["all"] = set(all_answers)
    return {
        family: tuple(sorted(values))
        for family, values in grouped.items()
        if values
    }


def candidate_answers_for_prompt(
    prompt: str,
    candidate_answer_sets: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    lower = prompt.lower()
    if "desk" in lower:
        return candidate_answer_sets.get("clinic_routing", ())
    if "weekday" in lower or "review day" in lower or "which day" in lower:
        return candidate_answer_sets.get("permit_review", ())
    if "platform" in lower:
        return candidate_answer_sets.get("rail_platform", ())
    if "code" in lower or "queue" in lower:
        return candidate_answer_sets.get("support_queue", ())
    return candidate_answer_sets.get("all", ())


def select_answer_by_logprob(
    *,
    stack: RuntimeStack,
    model: Any,
    rendered_prompt: str,
    candidates: tuple[str, ...],
) -> str:
    best_answer = ""
    best_score = float("-inf")
    for answer in candidates:
        score = conditional_answer_logprob(
            stack=stack,
            model=model,
            rendered_prompt=rendered_prompt,
            answer=answer,
        )
        if score > best_score:
            best_score = score
            best_answer = answer
    return best_answer


def conditional_answer_logprob(
    *,
    stack: RuntimeStack,
    model: Any,
    rendered_prompt: str,
    answer: str,
) -> float:
    tokenizer = tokenizer_for_stack(stack)
    if tokenizer is None:
        raise RuntimeError("Conditional answer scoring requires a tokenizer-capable runtime stack.")

    prefix = tokenizer(rendered_prompt, return_tensors="pt")
    suffix_text = f" {answer}{eos_token_text(stack)}"
    full = tokenizer(rendered_prompt + suffix_text, return_tensors="pt")
    input_ids = full["input_ids"].to(stack.device)
    attention_mask = full["attention_mask"].to(stack.device)
    prefix_length = int(prefix["input_ids"].shape[-1])
    with stack.torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = stack.torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)

    target_ids = input_ids[:, 1:]
    start_index = max(prefix_length - 1, 0)
    suffix_log_probs = log_probs[:, start_index:, :]
    suffix_targets = target_ids[:, start_index:]
    token_scores = suffix_log_probs.gather(-1, suffix_targets.unsqueeze(-1)).squeeze(-1)
    return float(token_scores.mean().item())


def eos_token_text(stack: RuntimeStack) -> str:
    tokenizer = tokenizer_for_stack(stack)
    eos_token = getattr(tokenizer, "eos_token", None)
    return eos_token or ""


def tokenizer_for_stack(stack: RuntimeStack) -> Any | None:
    if stack.tokenizer is not None:
        return stack.tokenizer
    if stack.processor is not None and hasattr(stack.processor, "tokenizer"):
        return stack.processor.tokenizer
    return None


def policy_for_method_family(method_family: str) -> BaselineMethodPolicy:
    if method_family == "baseline_seq_lora_ft_v11_qwen35_wide_fact_replay":
        return BaselineMethodPolicy(
            method_family=method_family,
            state_consistent_replay=True,
            answer_selection=False,
            lora_rank=16,
            lora_alpha=32,
            learning_rate=2e-4,
            episode_passes=2,
            target_repeat_count=5,
            anchor_repeat_count=1,
            history_replay_episodes=3,
            target_history_replay_episodes=3,
            anchor_history_replay_episodes=3,
            history_lm_replay=False,
            current_lm_replay=True,
            final_consolidation=True,
            final_target_passes=2,
            final_anchor_passes=1,
            accumulate_supervision=True,
            supervision_batch_size=48,
            mixed_final_consolidation=True,
            relation_aware_replay=True,
            max_decode_tokens=8,
            notes=(
                "Qwen wide-surface fact-replay repair after v10 showed strong training-prompt recall but weak prompt-disjoint target transfer.",
                "Adds public current fact-text LM replay so edited facts are trained as statements, not only as prompt-answer pairs.",
                "Keeps rank-16 wide LoRA and uses target-biased final consolidation without retrieval, answer selection, visible-dev rehearsal, or locked-prompt access.",
            ),
        )
    if method_family == "baseline_seq_lora_ft_v10_qwen35_wide_stable_rehearsal":
        return BaselineMethodPolicy(
            method_family=method_family,
            state_consistent_replay=True,
            answer_selection=False,
            lora_rank=16,
            lora_alpha=32,
            learning_rate=1e-4,
            episode_passes=3,
            target_repeat_count=4,
            anchor_repeat_count=2,
            history_replay_episodes=4,
            target_history_replay_episodes=4,
            anchor_history_replay_episodes=6,
            history_lm_replay=False,
            current_lm_replay=False,
            final_consolidation=True,
            final_target_passes=3,
            final_anchor_passes=3,
            accumulate_supervision=True,
            supervision_batch_size=32,
            mixed_final_consolidation=True,
            relation_aware_replay=True,
            max_decode_tokens=8,
            notes=(
                "Qwen wide-surface stability repair after v9 collapsed every prompt to a single frequent answer.",
                "Uses lower-rank, lower-learning-rate LoRA over the same top-8 hybrid attention plus MLP surface.",
                "Balances anchor and target rehearsal in mixed final consolidation to test whether v9 failed from optimizer instability rather than the wider surface itself.",
            ),
        )
    if method_family == "baseline_seq_lora_ft_v9_qwen35_wide_suffix_recall":
        return BaselineMethodPolicy(
            method_family=method_family,
            state_consistent_replay=True,
            answer_selection=False,
            lora_rank=32,
            lora_alpha=64,
            learning_rate=7.5e-4,
            episode_passes=2,
            target_repeat_count=5,
            anchor_repeat_count=1,
            history_replay_episodes=4,
            target_history_replay_episodes=2,
            anchor_history_replay_episodes=2,
            history_lm_replay=False,
            current_lm_replay=False,
            final_consolidation=True,
            final_target_passes=3,
            final_anchor_passes=2,
            accumulate_supervision=True,
            supervision_batch_size=48,
            mixed_final_consolidation=False,
            relation_aware_replay=True,
            max_decode_tokens=8,
            notes=(
                "Qwen-only wide-suffix repair after v8 showed low training-prompt recall, not just weak paraphrase transfer.",
                "Uses the wider Qwen top-8 hybrid attention plus MLP surface to add legitimate adapter capacity without retrieval or answer selection.",
                "Cuts per-episode replay fan-out and shifts budget into end-of-stream public training-record consolidation so all 96 edits receive repeated direct supervision.",
            ),
        )
    if method_family == "baseline_seq_lora_ft_v8_qwen35_target_last_coverage":
        return BaselineMethodPolicy(
            method_family=method_family,
            state_consistent_replay=True,
            answer_selection=False,
            lora_rank=32,
            lora_alpha=64,
            learning_rate=5e-4,
            episode_passes=4,
            target_repeat_count=5,
            anchor_repeat_count=1,
            history_replay_episodes=4,
            target_history_replay_episodes=4,
            anchor_history_replay_episodes=2,
            history_lm_replay=False,
            current_lm_replay=False,
            final_consolidation=True,
            final_target_passes=2,
            final_anchor_passes=1,
            accumulate_supervision=True,
            supervision_batch_size=32,
            mixed_final_consolidation=False,
            relation_aware_replay=True,
            max_decode_tokens=8,
            notes=(
                "Qwen-only target-last coverage repair after v7 completed all edits but under-committed to target facts.",
                "Keeps anchors in replay but ends consolidation on public target records so latest edited facts are not washed out by same-relation old-answer anchors.",
                "Uses only training-generator prompts and no retrieval, answer-set postprocessor, visible-dev rehearsal, or locked-prompt access.",
            ),
        )
    if method_family == "baseline_seq_lora_ft_v7_qwen35_coverage_first":
        return BaselineMethodPolicy(
            method_family=method_family,
            state_consistent_replay=True,
            answer_selection=False,
            lora_rank=32,
            lora_alpha=64,
            learning_rate=5e-4,
            episode_passes=2,
            target_repeat_count=4,
            anchor_repeat_count=1,
            history_replay_episodes=3,
            target_history_replay_episodes=3,
            anchor_history_replay_episodes=3,
            history_lm_replay=False,
            current_lm_replay=False,
            final_consolidation=True,
            final_target_passes=1,
            final_anchor_passes=1,
            accumulate_supervision=True,
            supervision_batch_size=32,
            mixed_final_consolidation=True,
            relation_aware_replay=True,
            max_decode_tokens=8,
            notes=(
                "Qwen-only coverage-first repair after v6 spent the surrogate budget before reaching all 96 CounterFact edits.",
                "Keeps the same prompt-disjoint CounterFact substrate and Qwen hybrid LoRA surface.",
                "Cuts replay fan-out before the end-of-stream consolidation so every public edit receives direct supervision before visible evaluation.",
            ),
        )
    if method_family in {
        "baseline_seq_lora_ft_v6_rank32_relation_replay",
        "baseline_seq_lora_ft_v6_rank32_relation_replay_qwen35",
    }:
        return BaselineMethodPolicy(
            method_family=method_family,
            state_consistent_replay=True,
            answer_selection=False,
            lora_rank=32,
            lora_alpha=64,
            learning_rate=5e-4,
            episode_passes=4,
            target_repeat_count=4,
            anchor_repeat_count=3,
            history_replay_episodes=8,
            target_history_replay_episodes=8,
            anchor_history_replay_episodes=16,
            history_lm_replay=False,
            current_lm_replay=False,
            final_consolidation=True,
            final_target_passes=3,
            final_anchor_passes=3,
            accumulate_supervision=True,
            supervision_batch_size=24,
            mixed_final_consolidation=True,
            relation_aware_replay=True,
            max_decode_tokens=8,
            notes=(
                "Extends v5 with relation-aware CounterFact replay families.",
                "Weights anchor rehearsal higher after v5 cleared the locked target floor but missed joint/locality.",
                "Keeps the same selected stack and prompt-disjoint train/visible/locked substrate with no retrieval or answer postprocessor.",
            ),
        )
    if method_family == "baseline_seq_lora_ft_v5_rank32_mixed_rehearsal":
        return BaselineMethodPolicy(
            method_family=method_family,
            state_consistent_replay=True,
            answer_selection=False,
            lora_rank=32,
            lora_alpha=64,
            learning_rate=5e-4,
            episode_passes=4,
            target_repeat_count=3,
            anchor_repeat_count=2,
            history_replay_episodes=8,
            target_history_replay_episodes=8,
            anchor_history_replay_episodes=8,
            history_lm_replay=False,
            current_lm_replay=False,
            final_consolidation=True,
            final_target_passes=2,
            final_anchor_passes=2,
            accumulate_supervision=True,
            supervision_batch_size=24,
            mixed_final_consolidation=True,
            max_decode_tokens=8,
            notes=(
                "Uses rank-32 active-state replay with mixed gradient accumulation over target and anchor supervision.",
                "Removes recency-ordered single-example updates so rehearsal optimizes target plasticity and neighborhood specificity together.",
                "Final consolidation interleaves active target and anchor records without consuming visible-dev or locked confirmation prompts.",
            ),
        )
    if method_family == "baseline_seq_lora_ft_v4_rank16_final_consolidation":
        return BaselineMethodPolicy(
            method_family=method_family,
            state_consistent_replay=True,
            answer_selection=False,
            lora_rank=16,
            lora_alpha=32,
            learning_rate=5e-4,
            episode_passes=4,
            target_repeat_count=4,
            anchor_repeat_count=1,
            history_replay_episodes=6,
            target_history_replay_episodes=8,
            anchor_history_replay_episodes=8,
            history_lm_replay=False,
            current_lm_replay=True,
            final_consolidation=True,
            final_target_passes=2,
            final_anchor_passes=2,
            max_decode_tokens=8,
            notes=(
                "Uses cheaper per-episode state replay plus an end-of-stream consolidation sweep.",
                "The consolidation pass rehearses anchors before targets so the final gradient preserves edited facts without using eval surfaces.",
            ),
        )
    if method_family == "baseline_seq_lora_ft_v3_rank16_target_rehearsal":
        return BaselineMethodPolicy(
            method_family=method_family,
            state_consistent_replay=True,
            answer_selection=False,
            lora_rank=16,
            lora_alpha=32,
            learning_rate=5e-4,
            episode_passes=6,
            target_repeat_count=4,
            anchor_repeat_count=1,
            history_replay_episodes=6,
            target_history_replay_episodes=24,
            anchor_history_replay_episodes=4,
            history_lm_replay=False,
            current_lm_replay=True,
            max_decode_tokens=8,
            notes=(
                "Uses rank-16 active-state replay with a wider target rehearsal window.",
                "Keeps anchor rehearsal present but lower cost to reduce early target forgetting.",
            ),
        )
    if method_family == "baseline_seq_lora_ft_v2_rank16_target_replay":
        return BaselineMethodPolicy(
            method_family=method_family,
            state_consistent_replay=True,
            answer_selection=False,
            lora_rank=16,
            lora_alpha=32,
            learning_rate=7.5e-4,
            episode_passes=12,
            target_repeat_count=4,
            anchor_repeat_count=1,
            history_replay_episodes=4,
            max_decode_tokens=8,
            notes=(
                "Uses active-state replay with a rank-16 LoRA surface.",
                "Weights target supervision above anchors to improve plasticity on larger CounterFact lanes.",
            ),
        )
    if method_family == "baseline_seq_lora_ft_v1_state_replay":
        return BaselineMethodPolicy(
            method_family=method_family,
            state_consistent_replay=True,
            answer_selection=False,
            max_decode_tokens=8,
            notes=(
                "Uses active-state replay so superseded overwrite facts are not replayed.",
                "Trains short-answer supervision with explicit EOS termination.",
            ),
        )
    if method_family == "baseline_seq_lora_ft_v1_state_replay_answer_select":
        return BaselineMethodPolicy(
            method_family=method_family,
            state_consistent_replay=True,
            answer_selection=True,
            max_decode_tokens=8,
            notes=(
                "Uses active-state replay so superseded overwrite facts are not replayed.",
                "Scores only canonical answers derived from the training stream at inference time.",
            ),
        )
    return BaselineMethodPolicy(method_family=method_family)


def answer_from_fact_text(fact_text: str) -> str:
    stripped = fact_text.strip().rstrip(".")
    patterns = (
        r"routes to ([A-Za-z0-9-]+) Desk$",
        r"runs on ([A-Za-z0-9-]+)$",
        r"departs from Platform ([A-Za-z0-9-]+)$",
        r"uses code ([A-Za-z0-9-]+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, stripped)
        if match:
            return match.group(1)

    words = [word for word in stripped.replace(".", "").split() if word]
    if not words:
        return ""
    return words[-1]


def load_runtime_stack(
    model_id: str,
    *,
    adapter_dir: Path | None,
    train_mode: bool,
) -> RuntimeStack:
    import torch

    hub_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_family = infer_model_family(model_id)
    dtype = preferred_dtype(torch, model_family=model_family, device=device)

    processor = None
    tokenizer = None
    if model_family == "gemma3":
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        processor = load_pretrained_local_first(
            AutoProcessor.from_pretrained,
            model_id,
            token=hub_token,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        model = load_pretrained_local_first(
            Gemma3ForConditionalGeneration.from_pretrained,
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
            token=hub_token,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = load_pretrained_local_first(
            AutoTokenizer.from_pretrained,
            model_id,
            trust_remote_code=True,
            token=hub_token,
        )
        tokenizer = ensure_non_empty_tokenizer(
            tokenizer,
            AutoTokenizer.from_pretrained,
            model_id=model_id,
            trust_remote_code=True,
            token=hub_token,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = load_pretrained_local_first(
            AutoModelForCausalLM.from_pretrained,
            model_id,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            token=hub_token,
        )

    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir.as_posix(), is_trainable=train_mode)

    model.to(device)
    if train_mode:
        model.train()
    else:
        model.eval()

    return RuntimeStack(
        torch=torch,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        device=device,
        model_family=model_family,
    )


def load_pretrained_local_first(loader: Any, model_id: str, **kwargs: Any) -> Any:
    local_kwargs = dict(kwargs)
    local_kwargs["local_files_only"] = True
    try:
        return loader(model_id, **local_kwargs)
    except Exception as local_error:
        remote_kwargs = dict(kwargs)
        remote_kwargs["local_files_only"] = False
        try:
            return loader(model_id, **remote_kwargs)
        except Exception:
            raise local_error


def ensure_non_empty_tokenizer(
    tokenizer: Any,
    loader: Any,
    *,
    model_id: str,
    **kwargs: Any,
) -> Any:
    probe = tokenizer("tokenizer sanity check", return_tensors="pt")
    input_ids = probe.get("input_ids")
    if getattr(input_ids, "shape", [0, 0])[-1] > 0:
        return tokenizer
    remote_kwargs = dict(kwargs)
    remote_kwargs["local_files_only"] = False
    replacement = loader(model_id, **remote_kwargs)
    replacement_probe = replacement("tokenizer sanity check", return_tensors="pt")
    replacement_ids = replacement_probe.get("input_ids")
    if getattr(replacement_ids, "shape", [0, 0])[-1] <= 0:
        raise RuntimeError(f"tokenizer for {model_id} produced empty input_ids")
    return replacement


def build_optimizer(torch: Any, wrapped_model: Any, *, learning_rate: float) -> Any:
    parameters = [parameter for parameter in wrapped_model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("Wrapped baseline exposed no trainable parameters.")
    return torch.optim.AdamW(parameters, lr=learning_rate)


def trainer_context_from_spec(*, spec_path: Path, spec: Mapping[str, Any]) -> TrainerContext:
    declared_capacity = spec["declared_capacity"]
    editable_surface = spec["editable_surface"]
    data = spec["data"]
    return TrainerContext(
        run_id=require_string(spec, "run_id"),
        parent_champion=require_string(spec, "parent_champion"),
        task_mode=require_string(spec, "task_mode"),
        run_class=require_string(spec, "run_class"),
        command=require_string(spec, "command"),
        spec_path=display_path(spec_path),
        method_family=require_string(spec, "method_family"),
        base_model=require_string(spec, "base_model"),
        editable_paths=tuple(editable_surface.get("paths", [])),
        selection_policy=require_string(editable_surface, "selection_policy"),
        declared_trainable_params=int(declared_capacity["trainable_parameter_count"]),
        trainable_parameter_tolerance=int(declared_capacity.get("trainable_parameter_tolerance", 0)),
        frozen_base_model=bool(declared_capacity.get("frozen_base_model", False)),
        uses_retrieval=bool(declared_capacity.get("uses_retrieval", False)),
        helper_models=tuple(declared_capacity.get("helper_models", [])),
        uses_postprocessor=bool(declared_capacity.get("uses_postprocessor", False)),
        train_generator_version=require_string(data, "train_generator_version"),
        development_pack=require_string(data, "development_pack"),
        confirmation_pack_summary=require_string(data, "confirmation_pack_summary"),
        baseline_ref=require_string(spec, "baseline_ref"),
        comparison_scope=require_string(spec, "comparison_scope"),
    )


def registry_path(kind: str, key: str) -> Path:
    registry = load_json(REPO_ROOT / "data" / "registry.yaml")
    payload = registry.get(kind, {}).get(key)
    if not isinstance(payload, Mapping):
        raise KeyError(f"Unknown registry {kind!r} entry {key!r}")
    path_field = "path"
    if kind == "packs" and "path" not in payload:
        path_field = "summary_path"
    return resolve_path(str(payload[path_field]))


def run_budget_for_class(run_class: str, episode_count: int) -> RunBudget:
    run_classes = load_json(REPO_ROOT / "protocol" / "RUN_CLASSES.yaml")
    details = run_classes.get("run_classes", {}).get(run_class, {})
    max_runtime_seconds = details.get("max_runtime_seconds")
    if isinstance(max_runtime_seconds, (int, float)) and max_runtime_seconds > 0:
        resolved_runtime_seconds = float(max_runtime_seconds)
    else:
        candidate_range = details.get("candidate_budget_range_minutes", [60, 90])
        max_minutes = float(candidate_range[-1]) if candidate_range else 90.0
        resolved_runtime_seconds = max_minutes * 60.0
    return RunBudget(
        max_steps=max(1, episode_count),
        max_runtime_seconds=resolved_runtime_seconds,
    )


def run_workspace_preflight(task_mode: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "scripts/check_workspace.py", "--task-mode", task_mode],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "workspace preflight failed before bootstrap baseline run:\n"
            + (result.stdout + result.stderr).strip()
        )
    return {
        "ok": True,
        "stdout": result.stdout.strip(),
    }


def current_peak_vram_gb(stack: RuntimeStack) -> float:
    if stack.device != "cuda":
        return 0.0
    return float(stack.torch.cuda.max_memory_allocated()) / (1024 ** 3)


def resolve_run_dir(raw: str | None, run_id: str) -> Path:
    if raw:
        return Path(os.path.expandvars(raw)).expanduser().resolve()
    return (REPO_ROOT / "artifacts" / "runs" / run_id).resolve()


def selected_surface_name() -> str:
    payload = load_json(REPO_ROOT / "protocol" / "MODEL_SURFACE_PILOT.yaml")
    selected = payload.get("selected_pair", {})
    surface_name = selected.get("surface_name")
    if not isinstance(surface_name, str) or not surface_name:
        raise RuntimeError("protocol/MODEL_SURFACE_PILOT.yaml missing selected_pair.surface_name")
    return surface_name


def surface_name_for_spec(spec: Mapping[str, Any]) -> str:
    model_lane = spec.get("model_lane")
    if isinstance(model_lane, Mapping):
        surface_name = model_lane.get("surface_name")
        if isinstance(surface_name, str) and surface_name.strip():
            return surface_name
    return selected_surface_name()


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Expected non-empty string for {field!r}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
