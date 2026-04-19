# 3090 Pilot Benchmark

This document records the three-model pilot benchmark that selected the first 3090 implementation stack for this repository.

It exists to answer one narrow question:

- which candidate base model should anchor the first autonomous continual-learning / knowledge-editing harness on the observed single-3090 machine?

This is an implementation-pilot decision record, not the final scientific judgment about the best model family for the overall research program.

## Scope

The benchmark compared the three official pilot candidates from [protocol/MODEL_SURFACE_PILOT.yaml](../protocol/MODEL_SURFACE_PILOT.yaml):

- `Qwen/Qwen3.5-4B-Base`
- `meta-llama/Llama-3.1-8B`
- `google/gemma-3-4b-pt`

All three were measured on the observed 3090 stack:

- host: `george-linux-server`
- GPU: `NVIDIA GeForce RTX 3090 24 GiB`
- editable surface: `top4_standard`
- probe skeleton: `hyper_lora_probe_v0`
- artifact source: `~/shared/artifacts/autoresearch-continual-learning/calibration/*.json`

The benchmark asked four practical questions:

1. Does the model fit and run stably on the 3090 under the fixed probe?
2. What peak VRAM does it use?
3. What training throughput does it achieve?
4. Does it answer the bounded visible-dev smoke questions correctly under a legitimate invocation path?

## Fit Results

| Model | Status | Peak VRAM (GiB) | Invalid-Run Rate | Mean Train Tokens/s |
| --- | --- | ---: | ---: | ---: |
| `google/gemma-3-4b-pt` | `ok` | `8.27` | `0.0` | `234.49` |
| `meta-llama/Llama-3.1-8B` | `ok` | `15.14` | `0.0` | `177.53` |
| `Qwen/Qwen3.5-4B-Base` | `ok` | `8.18` | `0.0` | `134.46` |

## Visible-Dev Results

The bounded visible-dev pack was `data/packs/dev_visible_v1.yaml`:

- `What is the capital city of France?`
- `What color is the daytime sky on a clear day?`
- `How many sides does a triangle have?`

Final repaired scores:

| Model | Visible-Dev Score | Predictions |
| --- | ---: | --- |
| `google/gemma-3-4b-pt` | `1.0` | `paris`, `blue`, `three` |
| `meta-llama/Llama-3.1-8B` | `1.0` | `paris`, `blue`, `three` |
| `Qwen/Qwen3.5-4B-Base` | `1.0` | `paris`, `blue`, `three` |

## Legitimacy Notes

The first visible-dev pass produced misleading scores and was not accepted as legitimate.

The investigation found harness-side issues:

- Gemma 3 needed the official `AutoProcessor` + `Gemma3ForConditionalGeneration` path rather than a generic causal-LM path.
- Gemma 3 generation on the 3090 needed `bfloat16`; `float16` collapsed to repeated `<pad>` tokens.
- Qwen needed a better prompt regime and a longer deterministic generation budget.
- The original extractor was too brittle and mis-scored valid answers like `Paris is ...` and `3`.

After those issues were fixed in [scripts/profile_visible_dev.py](../scripts/profile_visible_dev.py), all three candidates scored full marks. The probe artifacts now include `generation_traces` with the rendered prompt and raw completion so the visible-dev path is auditable.

## Selection Rule

The recorded ranking policy was:

1. 3090 fit
2. short-update stability
3. visible-dev profile
4. throughput
5. simplicity

All three candidates:

- fit on the 3090
- had `invalid_run_rate = 0.0`
- scored `1.0` on the repaired visible-dev pack

That means throughput broke the tie.

## Decision

The selected implementation pilot pair is:

- base model: `google/gemma-3-4b-pt`
- editable surface: `top4_standard`

Why this pair won:

- it fits comfortably on the 24 GiB 3090
- it matches Qwen on low VRAM while clearly beating it on throughput
- it materially beats Llama on throughput while using much less VRAM
- after the visible-dev invocation was repaired, it no longer had any bounded-answer disadvantage

## Non-Goals

This document does not claim:

- that Gemma 3 is universally the best scientific choice
- that the bootstrap visible-dev pack is the final scientific visible-dev pack
- that run-class envelopes are already frozen
- that newer multimodal families should never be tested

It only records that, for the first single-3090 pilot harness in this repository, `google/gemma-3-4b-pt` is the justified starting point.
