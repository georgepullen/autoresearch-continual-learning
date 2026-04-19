#!/usr/bin/env python3
"""Profile one base model on the bounded visible-dev pack and annotate a probe artifact."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.runner import evaluate_examples, load_eval_batch


class BaseModelPredictor:
    """Small deterministic generator for base-model visible-dev checks."""

    def __init__(self, *, model_id: str, max_new_tokens: int) -> None:
        import torch
        self._model_id = model_id
        self._model_family = infer_model_family(model_id)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._traces: dict[str, dict[str, str]] = {}
        hub_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        dtype = preferred_dtype(torch, model_family=self._model_family, device=self._device)

        if self._model_family == "gemma3":
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration

            self._processor = AutoProcessor.from_pretrained(model_id, token=hub_token)
            self._model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                dtype=dtype,
                low_cpu_mem_usage=True,
                token=hub_token,
            )
            self._prompt_style = "direct_answer"
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                token=hub_token,
            )
            if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                token=hub_token,
            )
            self._prompt_style = "qa_pair"

        self._model.to(self._device)
        self._model.eval()
        self._max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        rendered = render_prompt(prompt, model_family=self._model_family)
        decoded = self._generate(rendered)
        prediction = extract_short_answer(decoded)
        self._traces[prompt] = {
            "model_family": self._model_family,
            "prompt_style": self._prompt_style,
            "rendered_prompt": rendered,
            "raw_completion": decoded,
            "prediction": prediction,
        }
        return prediction

    def trace_for(self, prompt: str) -> dict[str, str]:
        return dict(self._traces.get(prompt, {}))

    def _generate(self, rendered: str) -> str:
        if self._model_family == "gemma3":
            batch = self._processor(text=rendered, return_tensors="pt")
            batch = {key: value.to(self._device) for key, value in batch.items()}
            with self._torch.no_grad():
                output = self._model.generate(
                    **batch,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                )
            input_len = int(batch["input_ids"].shape[-1])
            generated = output[0][input_len:]
            return self._processor.decode(generated, skip_special_tokens=True)

        batch = self._tokenizer(rendered, return_tensors="pt")
        batch = {key: value.to(self._device) for key, value in batch.items()}
        with self._torch.no_grad():
            output = self._model.generate(
                **batch,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        input_len = int(batch["input_ids"].shape[-1])
        generated = output[0][input_len:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-artifact", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    artifact_path = Path(args.probe_artifact)
    artifact = json.loads(artifact_path.read_text())
    probe = artifact.setdefault("probe", {})
    model_id = args.model_id or probe.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be provided either via --model-id or the probe artifact")

    batch = load_eval_batch(args.pack)
    predictor = BaseModelPredictor(model_id=model_id, max_new_tokens=args.max_new_tokens)
    result = evaluate_examples(
        batch,
        predict_fn=predictor,
        score_fn=default_exact_match_score,
    )

    visible_dev_score = float(result.target_quality["exact_match"])
    visible_dev_profile = {
        "pack_id": batch.pack_id,
        "score_name": "exact_match",
        "visible_dev_score": visible_dev_score,
        "target_quality": result.target_quality,
        "interference": result.interference,
        "predictions": list(result.predictions),
        "per_example": list(result.per_example),
        "generation_traces": [
            {
                "example_id": example.example_id,
                **predictor.trace_for(example.prompt),
            }
            for example in batch.examples
        ],
    }

    probe["visible_dev_score"] = visible_dev_score
    probe["visible_dev_pack_id"] = batch.pack_id
    artifact["visible_dev_profile"] = visible_dev_profile
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(visible_dev_profile, indent=2, sort_keys=True))
    return 0


def default_exact_match_score(prediction: str, target: str) -> float:
    return 0.0 if normalize_text(prediction) == normalize_text(target) else 1.0


def infer_model_family(model_id: str) -> str:
    lower = model_id.lower()
    if "gemma-3" in lower:
        return "gemma3"
    if "qwen" in lower:
        return "qwen"
    if "llama" in lower:
        return "llama"
    return "causal_lm"


def preferred_dtype(torch: Any, *, model_family: str, device: str) -> Any:
    if device != "cuda":
        return torch.float32
    if model_family == "gemma3":
        return torch.bfloat16
    return torch.float16


def render_prompt(prompt: str, *, model_family: str) -> str:
    if model_family == "gemma3":
        return f"Answer with a single word or number only. Do not explain.\nQuestion: {prompt}\nAnswer:"
    return f"Q: {prompt}\nA:"


def extract_short_answer(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.replace("\r", "")
    cleaned = truncate_at_turn_boundary(cleaned)
    cleaned = re.sub(r"^(answer\s*:|a\s*:|the answer is)\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" \t\r\n\"'`")
    if not cleaned:
        return ""

    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    candidate = first_line or cleaned
    candidate = candidate.strip(" \t\r\n\"'`.,;:!?()[]{}")
    if not candidate:
        return ""

    single_token = extract_single_token(candidate)
    if single_token:
        return single_token

    sentence = first_sentence(candidate)
    sentence = repair_doubled_leading_token(sentence)
    leading_subject = re.match(
        r"^([A-Za-z0-9-]+)\s+(?:is|are|was|were|has|have|had)\b",
        sentence,
        flags=re.IGNORECASE,
    )
    if leading_subject and leading_subject.group(1).lower() not in NON_ANSWER_LEADERS:
        return canonicalize_token(leading_subject.group(1))

    patterns = [
        r"\b(?:is|are|was|were|be|been|being|has|have|had)\s+([A-Za-z0-9-]+)\b",
        r"\b(?:appears?|looks?)\s+([A-Za-z0-9-]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)
        if match:
            return canonicalize_token(match.group(1))

    fallback = extract_single_token(sentence)
    if fallback:
        return fallback

    words = [word for word in re.split(r"\s+", sentence) if word]
    if not words:
        return ""
    return canonicalize_token(words[0])


def truncate_at_turn_boundary(text: str) -> str:
    boundaries = [
        "\nQ:",
        "\nQuestion:",
        "\nUser:",
        "\nAssistant:",
        "\\nQ:",
        "\\nQuestion:",
        "\\nUser:",
        "\\nAssistant:",
    ]
    end = len(text)
    for marker in boundaries:
        idx = text.find(marker)
        if idx != -1:
            end = min(end, idx)
    return text[:end].strip()


def first_sentence(text: str) -> str:
    match = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return match[0].strip()


def extract_single_token(text: str) -> str:
    token_match = re.fullmatch(r"([A-Za-z0-9-]+)", repair_doubled_leading_token(text))
    if token_match:
        return canonicalize_token(token_match.group(1))
    return ""


def repair_doubled_leading_token(text: str) -> str:
    match = re.match(r"^([A-Za-z]{2,})(\1)\b", text)
    if match:
        return match.group(1) + text[2 * len(match.group(1)) :]
    return text


def canonicalize_token(token: str) -> str:
    stripped = token.strip(" \t\r\n\"'`.,;:!?()[]{}")
    if not stripped:
        return ""
    lower = repair_doubled_leading_token(stripped).lower()
    return DIGIT_TO_WORD.get(lower, lower)


def normalize_text(text: str) -> str:
    return " ".join(canonicalize_token(part) for part in text.strip().split()).strip()


DIGIT_TO_WORD = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}

NON_ANSWER_LEADERS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "it",
    "they",
    "he",
    "she",
    "we",
    "i",
    "you",
}


if __name__ == "__main__":
    raise SystemExit(main())
