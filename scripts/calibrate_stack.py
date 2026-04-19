#!/usr/bin/env python3
"""Run a lightweight adapter probe to measure stack fit on the current machine."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from method.adapter_surface import ModelShape, estimate_lora_parameter_count, get_surface, resolve_layer_indices
from method.selected_stack import load_selected_pilot_pair


def main() -> int:
    selected_pair = load_selected_pilot_pair(REPO_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default=selected_pair["model_id"],
        help=f"Base model to probe (default: selected pilot {selected_pair['model_id']}).",
    )
    parser.add_argument(
        "--surface",
        default=selected_pair["surface_name"],
        help=f"Editable surface to probe (default: selected pilot {selected_pair['surface_name']}).",
    )
    parser.add_argument("--output")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - import failure is a runtime concern
        return finish(
            payload={
                "schema_version": 1,
                "status": "environment_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "recorded_at_utc": utc_now(),
            },
            output_path=args.output,
            exit_code=1,
        )

    surface = get_surface(args.surface)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "started",
        "recorded_at_utc": utc_now(),
        "host": {
            "hostname": socket.gethostname(),
            "device": device,
        },
        "probe": {
            "model_id": args.model_id,
            "surface_name": surface.name,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "train_steps": args.train_steps,
            "eval_steps": args.eval_steps,
            "repeats": args.repeats,
            "lora_rank": args.lora_rank,
            "learning_rate": args.learning_rate,
        },
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.to(device)

        shape = infer_model_shape(model.config)
        layer_indices = resolve_layer_indices(shape.num_hidden_layers, surface)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=max(args.lora_rank * 2, 1),
            lora_dropout=0.0,
            bias="none",
            target_modules=list(surface.target_modules),
            layers_to_transform=list(layer_indices),
        )
        model = get_peft_model(model, lora_config)
        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

        prompt = (
            "Calibration probe for conflict-aware continual learning. "
            "The goal is to measure fit, throughput, and runtime composition."
        )
        batch = tokenizer(
            [prompt] * args.batch_size,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.seq_len,
        )
        if "token_type_ids" not in batch:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        batch = {key: value.to(device) for key, value in batch.items()}
        batch["labels"] = batch["input_ids"].clone()

        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
        )

        rehearsals = []
        invalid_rehearsals = 0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for repeat_index in range(args.repeats):
            try:
                rehearsal = run_rehearsal(
                    torch_module=torch,
                    model=model,
                    optimizer=optimizer,
                    batch=batch,
                    train_steps=args.train_steps,
                    eval_steps=args.eval_steps,
                    seq_len=args.seq_len,
                    batch_size=args.batch_size,
                )
                rehearsal["repeat_index"] = repeat_index
                rehearsals.append(rehearsal)
            except RuntimeError as exc:
                invalid_rehearsals += 1
                rehearsals.append(
                    {
                        "repeat_index": repeat_index,
                        "status": "runtime_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if device == "cuda":
                    torch.cuda.empty_cache()

        peak_vram_gb = (
            float(torch.cuda.max_memory_allocated()) / (1024 ** 3) if device == "cuda" else 0.0
        )

        payload.update(
            {
                "status": "ok" if invalid_rehearsals == 0 else "completed_with_invalid_rehearsals",
                "model_shape": {
                    "num_hidden_layers": shape.num_hidden_layers,
                    "hidden_size": shape.hidden_size,
                    "intermediate_size": shape.intermediate_size,
                },
                "editable_surface": {
                    "name": surface.name,
                    "layer_indices": list(layer_indices),
                    "target_modules": list(surface.target_modules),
                    "uses_per_layer_gate": surface.uses_per_layer_gate,
                    "estimated_lora_probe_params": estimate_lora_parameter_count(
                        shape,
                        surface,
                        rank=args.lora_rank,
                    ),
                    "actual_trainable_params": trainable_params,
                },
                "measurements": summarize_rehearsals(rehearsals),
                "peak_vram_gb": peak_vram_gb,
                "invalid_run_rate": invalid_rehearsals / max(args.repeats, 1),
                "rehearsals": rehearsals,
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "probe_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return finish(payload=payload, output_path=args.output, exit_code=1)

    return finish(payload=payload, output_path=args.output, exit_code=0)


def run_rehearsal(
    *,
    torch_module: Any,
    model: Any,
    optimizer: Any,
    batch: dict[str, Any],
    train_steps: int,
    eval_steps: int,
    seq_len: int,
    batch_size: int,
) -> dict[str, Any]:
    train_tokens = train_steps * seq_len * batch_size
    eval_tokens = eval_steps * seq_len * batch_size

    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    train_start = time.perf_counter()
    model.train()
    for _ in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start

    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    eval_start = time.perf_counter()
    model.eval()
    last_eval_loss = None
    with torch_module.no_grad():
        for _ in range(eval_steps):
            outputs = model(**batch)
            loss = outputs.loss
            last_eval_loss = float(loss.detach().cpu())
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start

    total_seconds = train_seconds + eval_seconds
    return {
        "status": "ok",
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "total_seconds": total_seconds,
        "train_tokens_per_second": train_tokens / train_seconds if train_seconds else 0.0,
        "eval_tokens_per_second": eval_tokens / eval_seconds if eval_seconds else 0.0,
        "eval_fraction_of_runtime": eval_seconds / total_seconds if total_seconds else 0.0,
        "last_eval_loss": last_eval_loss,
    }


def summarize_rehearsals(rehearsals: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rehearsals = [item for item in rehearsals if item.get("status") == "ok"]
    if not ok_rehearsals:
        return {
            "ok_rehearsals": 0,
            "mean_train_tokens_per_second": 0.0,
            "mean_eval_tokens_per_second": 0.0,
            "mean_eval_fraction_of_runtime": 0.0,
        }

    return {
        "ok_rehearsals": len(ok_rehearsals),
        "mean_train_tokens_per_second": mean(
            item["train_tokens_per_second"] for item in ok_rehearsals
        ),
        "mean_eval_tokens_per_second": mean(
            item["eval_tokens_per_second"] for item in ok_rehearsals
        ),
        "mean_eval_fraction_of_runtime": mean(
            item["eval_fraction_of_runtime"] for item in ok_rehearsals
        ),
        "mean_total_seconds": mean(item["total_seconds"] for item in ok_rehearsals),
    }


def infer_model_shape(config: Any) -> ModelShape:
    for candidate in iter_shape_configs(config):
        num_hidden_layers = maybe_first_int(
            getattr(candidate, "num_hidden_layers", None),
            getattr(candidate, "n_layer", None),
            getattr(candidate, "num_layers", None),
        )
        hidden_size = maybe_first_int(
            getattr(candidate, "hidden_size", None),
            getattr(candidate, "d_model", None),
            getattr(candidate, "n_embd", None),
        )
        intermediate_size = maybe_first_int(
            getattr(candidate, "intermediate_size", None),
            getattr(candidate, "ffn_dim", None),
            hidden_size * 4 if hidden_size else None,
        )
        if num_hidden_layers and hidden_size and intermediate_size:
            return ModelShape(
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )

    raise ValueError("could not infer required model shape field")


def iter_shape_configs(config: Any) -> list[Any]:
    candidates = [config]
    for attr_name in (
        "text_config",
        "language_config",
        "llm_config",
        "model_config",
        "backbone_config",
        "decoder_config",
    ):
        nested = getattr(config, attr_name, None)
        if nested is not None:
            candidates.append(nested)
    return candidates


def maybe_first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int) and value > 0:
            return value
    return None


def first_int(*values: Any) -> int:
    for value in values:
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("could not infer required model shape field")


def mean(values: Any) -> float:
    values = list(values)
    return float(sum(values) / len(values))


def finish(*, payload: dict[str, Any], output_path: str | None, exit_code: int) -> int:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    print(text)
    return exit_code


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
