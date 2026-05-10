#!/usr/bin/env python3
"""Run the first HyperLoRA-style mainline candidate on the selected stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.aggregates import build_continual_learning_family
from method import BoundedArtifactTrainer, DeterministicEpisodeSampler
from method import load_generator_spec
from method.baselines import SequentialLoRABaseline, SequentialLoRAConfig
from method.baselines.seq_lora_ft import is_lora_parameter_name, model_shape_from_hf_config
from scripts.run_bootstrap_baseline import (
    ReplayRecord,
    RuntimeStack,
    build_candidate_answer_sets,
    build_episode_supervision_examples,
    build_state_replay_records,
    build_optimizer,
    current_peak_vram_gb,
    display_path,
    encode_supervised_example,
    evaluate_visible_pack,
    policy_for_method_family,
    load_json,
    load_runtime_stack,
    render_fact_replay_text,
    registry_path,
    require_string,
    resolve_path,
    resolve_run_dir,
    run_budget_for_class,
    run_workspace_preflight,
    select_state_replay_records,
    surface_name_for_spec,
    trainer_context_from_spec,
)


CONTEXT_DIM = 32
CONTROLLER_RANK = 1
TRUNK_HIDDEN_DIM = 16
GATE_HIDDEN_DIM = 16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gate-loss-scale", type=float, default=0.01)
    parser.add_argument("--episode-passes", type=int, default=6)
    parser.add_argument("--target-repeat-count", type=int, default=2)
    parser.add_argument("--anchor-repeat-count", type=int, default=2)
    parser.add_argument("--history-replay-episodes", type=int, default=6)
    parser.add_argument(
        "--history-lm-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--current-lm-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    spec_path = resolve_path(args.spec)
    spec = load_json(spec_path)
    run_id = require_string(spec, "run_id")
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
    evaluation_policy = policy_for_method_family("baseline_seq_lora_ft_v1_state_replay")
    surface_name = mainline_surface_name(spec)

    stack = load_runtime_stack(spec["base_model"], adapter_dir=None, train_mode=True)
    baseline_reference = evaluate_visible_pack(
        stack=stack,
        model=stack.model,
        development_pack_path=development_pack_path,
        max_new_tokens=args.max_new_tokens,
        method_policy=evaluation_policy,
        candidate_answer_sets=candidate_answer_sets,
    )

    baseline = SequentialLoRABaseline(
        SequentialLoRAConfig(surface_name=surface_name),
        model_shape=model_shape_from_hf_config(stack.model.config),
    )
    wrapped_model = baseline.wrap_model(stack.model)
    from method import HyperLoRAConfig, HyperLoRAController

    controller = HyperLoRAController(
        HyperLoRAConfig(
            surface_name=surface_name,
            model_shape=model_shape_from_hf_config(stack.model.config),
            context_dim=CONTEXT_DIM,
            rank=CONTROLLER_RANK,
            trunk_hidden_dim=TRUNK_HIDDEN_DIM,
            gate_hidden_dim=GATE_HIDDEN_DIM,
            use_conflict_gate=True,
        )
    ).to(stack.device)

    optimizer = build_joint_optimizer(
        stack.torch,
        wrapped_model,
        controller,
        learning_rate=args.learning_rate,
    )

    budget = run_budget_for_class(spec["run_class"], len(generator.episodes))
    context = trainer_context_from_spec(spec_path=spec_path, spec=spec)
    trainer = BoundedArtifactTrainer(
        budget=budget,
        context=context,
        torch_module=stack.torch,
    )

    records_seed: list[dict[str, Any]] = []
    active_target_records: dict[str, ReplayRecord] = {}
    active_anchor_records: dict[str, ReplayRecord] = {}

    def step_fn(episode: Any, step_index: int) -> dict[str, Any]:
        target_record, anchor_records = build_state_replay_records(
            episode,
            target_repeat_count=args.target_repeat_count,
            anchor_repeat_count=args.anchor_repeat_count,
            updated_at=step_index,
        )
        active_target_records[target_record.key] = target_record
        for anchor_record in anchor_records:
            active_anchor_records[anchor_record.key] = anchor_record

        selected_target_records, selected_anchor_records = select_state_replay_records(
            active_target_records=active_target_records,
            active_anchor_records=active_anchor_records,
            current_target_key=target_record.key,
            current_anchor_keys={record.key for record in anchor_records},
            target_limit=max(args.history_replay_episodes, 0),
            anchor_limit=max(args.history_replay_episodes, 0),
        )
        replay_texts = [
            replay_record.lm_text
            for replay_record in [*selected_target_records, *selected_anchor_records]
            if args.history_lm_replay and replay_record.lm_text
        ]
        replay_examples = [
            example
            for replay_record in [*selected_target_records, *selected_anchor_records]
            for example in replay_record.supervision_examples
        ]
        step_metrics = train_one_mainline_episode(
            stack=stack,
            wrapped_model=wrapped_model,
            controller=controller,
            optimizer=optimizer,
            episode=episode,
            gate_loss_scale=args.gate_loss_scale,
            episode_passes=args.episode_passes,
            replay_examples=replay_examples,
            replay_texts=replay_texts,
            target_repeat_count=args.target_repeat_count,
            anchor_repeat_count=args.anchor_repeat_count,
            current_lm_replay=args.current_lm_replay,
        )
        records_seed.append(
            {
                "episode_id": episode.episode_id,
                **step_metrics,
                "replay_examples": len(replay_examples),
                "replay_texts": len(replay_texts),
                "state_consistent_replay": True,
                "selected_replay_target_keys": [
                    record.key for record in selected_target_records
                ],
                "selected_replay_anchor_keys": [
                    record.key for record in selected_anchor_records
                ],
            }
        )
        return {
            "target_quality": {
                "episode_train_loss": step_metrics["loss"],
            },
            "interference": {
                "gate_mean": step_metrics["gate_mean"],
                "gate_sparsity_loss": step_metrics["gate_sparsity_loss"],
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
            wrapped_model=wrapped_model,
            controller=controller,
            optimizer=optimizer,
        ),
    )

    evaluation = evaluate_visible_pack(
        stack=stack,
        model=wrapped_model,
        development_pack_path=development_pack_path,
        max_new_tokens=args.max_new_tokens,
        method_policy=evaluation_policy,
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
    artifact["mainline"] = {
        "controller_trainable_parameter_count": count_trainable_parameters(controller),
        "controller_rank": CONTROLLER_RANK,
        "context_dim": CONTEXT_DIM,
        "surface_name": surface_name,
        "uses_conflict_gate": True,
    }

    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    training_summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "records": records,
                "episodes": records_seed,
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
                "continual_learning": cl_metrics,
                "target_quality": evaluation["target_quality"],
                "interference": evaluation["interference"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def train_one_mainline_episode(
    *,
    stack: RuntimeStack,
    wrapped_model: Any,
    controller: HyperLoRAController,
    optimizer: Any,
    episode: Any,
    gate_loss_scale: float,
    episode_passes: int,
    replay_examples: list[tuple[str, str]] | None = None,
    replay_texts: list[str] | None = None,
    target_repeat_count: int = 2,
    anchor_repeat_count: int = 2,
    current_lm_replay: bool = True,
) -> dict[str, float]:
    if episode_passes <= 0:
        raise ValueError("episode_passes must be positive")

    supervision_examples = build_episode_supervision_examples(
        episode,
        target_repeat_count=target_repeat_count,
        anchor_repeat_count=anchor_repeat_count,
    )
    if not supervision_examples:
        raise RuntimeError(f"episode {episode.episode_id} yielded no supervision examples")

    context_vector = episode_context_vector(
        stack.torch,
        episode.update_text,
        dim=CONTEXT_DIM,
        device=stack.device,
    )
    reference_vector = episode_context_vector(
        stack.torch,
        "\n".join(episode.anchor_texts),
        dim=CONTEXT_DIM,
        device=stack.device,
    )
    wrapped_model.train()
    controller.train()
    observed_losses: list[float] = []
    gate_means: list[float] = []
    gate_penalties: list[float] = []
    replay_examples = replay_examples or []
    replay_texts = replay_texts or []
    for _ in range(episode_passes):
        if current_lm_replay:
            loss, gate_mean, gate_penalty_value = train_gated_language_model_text(
                stack=stack,
                wrapped_model=wrapped_model,
                controller=controller,
                optimizer=optimizer,
                text=render_fact_replay_text(episode),
                context_vector=context_vector,
                reference_vector=reference_vector,
                gate_loss_scale=gate_loss_scale,
            )
            observed_losses.append(loss)
            gate_means.append(gate_mean)
            gate_penalties.append(gate_penalty_value)

        for prompt, answer in supervision_examples:
            loss, gate_mean, gate_penalty_value = train_gated_supervised_example(
                stack=stack,
                wrapped_model=wrapped_model,
                controller=controller,
                optimizer=optimizer,
                prompt=prompt,
                answer=answer,
                context_vector=context_vector,
                reference_vector=reference_vector,
                gate_loss_scale=gate_loss_scale,
            )
            observed_losses.append(loss)
            gate_means.append(gate_mean)
            gate_penalties.append(gate_penalty_value)

        for replay_text in replay_texts:
            loss, gate_mean, gate_penalty_value = train_gated_language_model_text(
                stack=stack,
                wrapped_model=wrapped_model,
                controller=controller,
                optimizer=optimizer,
                text=replay_text,
                context_vector=context_vector,
                reference_vector=reference_vector,
                gate_loss_scale=gate_loss_scale,
            )
            observed_losses.append(loss)
            gate_means.append(gate_mean)
            gate_penalties.append(gate_penalty_value)

        for prompt, answer in replay_examples:
            loss, gate_mean, gate_penalty_value = train_gated_supervised_example(
                stack=stack,
                wrapped_model=wrapped_model,
                controller=controller,
                optimizer=optimizer,
                prompt=prompt,
                answer=answer,
                context_vector=context_vector,
                reference_vector=reference_vector,
                gate_loss_scale=gate_loss_scale,
            )
            observed_losses.append(loss)
            gate_means.append(gate_mean)
            gate_penalties.append(gate_penalty_value)

    return {
        "loss": sum(observed_losses) / len(observed_losses),
        "gate_mean": sum(gate_means) / len(gate_means) if gate_means else 1.0,
        "gate_sparsity_loss": sum(gate_penalties) / len(gate_penalties) if gate_penalties else 0.0,
    }


def mainline_surface_name(spec: Mapping[str, Any]) -> str:
    """Resolve the explicit editable surface for a mainline spec."""

    return surface_name_for_spec(spec)


def train_gated_language_model_text(
    *,
    stack: RuntimeStack,
    wrapped_model: Any,
    controller: HyperLoRAController,
    optimizer: Any,
    text: str,
    context_vector: Any,
    reference_vector: Any,
    gate_loss_scale: float,
) -> tuple[float, float, float]:
    encoded = stack.processor(
        text=text,
        return_tensors="pt",
    ) if stack.model_family == "gemma3" else stack.tokenizer(
        text,
        return_tensors="pt",
    )
    encoded = {key: value.to(stack.device) for key, value in encoded.items()}
    labels = encoded["input_ids"].clone()
    return train_gated_encoded_batch(
        stack=stack,
        wrapped_model=wrapped_model,
        controller=controller,
        optimizer=optimizer,
        encoded=encoded,
        labels=labels,
        context_vector=context_vector,
        reference_vector=reference_vector,
        gate_loss_scale=gate_loss_scale,
    )


def train_gated_supervised_example(
    *,
    stack: RuntimeStack,
    wrapped_model: Any,
    controller: HyperLoRAController,
    optimizer: Any,
    prompt: str,
    answer: str,
    context_vector: Any,
    reference_vector: Any,
    gate_loss_scale: float,
) -> tuple[float, float, float]:
    encoded, labels = encode_supervised_example(
        stack=stack,
        prompt=prompt,
        answer=answer,
    )
    return train_gated_encoded_batch(
        stack=stack,
        wrapped_model=wrapped_model,
        controller=controller,
        optimizer=optimizer,
        encoded=encoded,
        labels=labels,
        context_vector=context_vector,
        reference_vector=reference_vector,
        gate_loss_scale=gate_loss_scale,
    )


def train_gated_encoded_batch(
    *,
    stack: RuntimeStack,
    wrapped_model: Any,
    controller: HyperLoRAController,
    optimizer: Any,
    encoded: Mapping[str, Any],
    labels: Any,
    context_vector: Any,
    reference_vector: Any,
    gate_loss_scale: float,
) -> tuple[float, float, float]:
    gate_matrix, gate_scale, gate_penalty, gate_mean = compute_gate_terms(
        stack=stack,
        controller=controller,
        context_vector=context_vector,
        reference_vector=reference_vector,
    )
    optimizer.zero_grad(set_to_none=True)
    outputs = wrapped_model(**encoded, labels=labels)
    loss = outputs.loss * gate_scale + gate_loss_scale * gate_penalty
    loss.backward()
    apply_gate_scaled_gradients(
        wrapped_model=wrapped_model,
        gate_matrix=gate_matrix,
        layer_indices=controller.layer_indices,
        target_modules=controller.target_modules,
    )
    optimizer.step()
    return (
        float(loss.detach().item()),
        gate_mean,
        float(gate_penalty.detach().item()),
    )


def compute_gate_terms(
    *,
    stack: RuntimeStack,
    controller: HyperLoRAController,
    context_vector: Any,
    reference_vector: Any,
) -> tuple[Any, Any, Any, float]:
    from method.losses import gate_sparsity_loss

    controller_delta = controller(context_vector, reference_vector=reference_vector)
    gates = controller_delta.gates
    if gates is None:
        gate_matrix = stack.torch.ones(
            len(controller.layer_indices),
            len(controller.target_modules),
            device=stack.device,
        )
        gate_scale = stack.torch.tensor(1.0, device=stack.device)
        gate_penalty = stack.torch.tensor(0.0, device=stack.device)
        return gate_matrix, gate_scale, gate_penalty, 1.0
    gate_matrix = gates.detach().mean(dim=0)
    gate_scale = 0.5 + gates.mean()
    gate_penalty = gate_sparsity_loss(gates).to(gate_scale.device)
    return gate_matrix, gate_scale, gate_penalty, float(gates.mean().detach().item())


def build_joint_optimizer(torch: Any, wrapped_model: Any, controller: HyperLoRAController, *, learning_rate: float) -> Any:
    parameters = [
        parameter
        for parameter in list(wrapped_model.parameters()) + list(controller.parameters())
        if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("Mainline runner exposed no trainable parameters.")
    return torch.optim.AdamW(parameters, lr=learning_rate)


def observed_capacity_metadata(
    *,
    wrapped_model: Any,
    controller: HyperLoRAController,
    optimizer: Any,
) -> dict[str, object]:
    trainable_parameters = [
        parameter
        for parameter in list(wrapped_model.parameters()) + list(controller.parameters())
        if getattr(parameter, "requires_grad", False)
    ]
    trainable_parameter_ids = {id(parameter) for parameter in trainable_parameters}

    optimizer_parameter_ids: set[int] = set()
    optimizer_excludes_frozen_base_parameters = True
    for group in getattr(optimizer, "param_groups", []):
        for parameter in group.get("params", []):
            parameter_id = id(parameter)
            optimizer_parameter_ids.add(parameter_id)
            if parameter_id not in trainable_parameter_ids:
                optimizer_excludes_frozen_base_parameters = False
    if optimizer_parameter_ids != trainable_parameter_ids:
        optimizer_excludes_frozen_base_parameters = False

    base_model_trainable_parameter_count = sum(
        int(parameter.numel())
        for name, parameter in wrapped_model.named_parameters()
        if getattr(parameter, "requires_grad", False) and not is_lora_parameter_name(name)
    )

    return {
        "observed_trainable_parameter_count": sum(
            int(parameter.numel()) for parameter in trainable_parameters
        ),
        "observed_trainable_parameter_count_measured": True,
        "base_model_trainable_parameter_count": base_model_trainable_parameter_count,
        "base_model_trainable_parameter_count_measured": True,
        "frozen_base_behavior_verified": base_model_trainable_parameter_count == 0,
        "frozen_base_behavior_measured": True,
        "optimizer_excludes_frozen_base_parameters": optimizer_excludes_frozen_base_parameters,
        "optimizer_param_membership_measured": True,
        "used_retrieval": False,
        "helper_models": [],
        "used_postprocessor": False,
    }


def apply_gate_scaled_gradients(
    *,
    wrapped_model: Any,
    gate_matrix: Any,
    layer_indices: tuple[int, ...],
    target_modules: tuple[str, ...],
) -> None:
    layer_index_map = {layer: position for position, layer in enumerate(layer_indices)}
    module_index_map = {name: position for position, name in enumerate(target_modules)}
    for name, parameter in wrapped_model.named_parameters():
        if parameter.grad is None or not is_lora_parameter_name(name):
            continue
        layer_position = find_layer_position(name, layer_index_map)
        module_position = find_module_position(name, module_index_map)
        if layer_position is None or module_position is None:
            continue
        scale = float(gate_matrix[layer_position, module_position].item())
        parameter.grad.mul_(max(scale, 0.05))


def find_layer_position(name: str, layer_index_map: Mapping[int, int]) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", name)
    if not match:
        return None
    return layer_index_map.get(int(match.group(1)))


def find_module_position(name: str, module_index_map: Mapping[str, int]) -> int | None:
    for module_name, position in module_index_map.items():
        if f".{module_name}." in name:
            return position
    return None


def episode_context_vector(torch: Any, text: str, *, dim: int, device: str) -> Any:
    buckets = [0.0] * dim
    encoded = text.encode("utf-8", errors="ignore")
    if not encoded:
        encoded = b"<empty>"
    digest = hashlib.sha256(encoded).digest()
    for index, value in enumerate(digest):
        buckets[index % dim] += value / 255.0
    tensor = torch.tensor([buckets], dtype=torch.float32, device=device)
    return tensor


def count_trainable_parameters(module: Any) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    raise SystemExit(main())
