# Calibration Notes

This file records how `smoke_v1`, `dev_v1`, and `confirm_v1` should be calibrated on
the real 3090 stack.

It exists to keep the run-class policy honest:

- bounded classes are required
- exact minute budgets are not treated as universal truths
- calibration evidence must come from the chosen stack, not from taste

## Observed 3090 facts

Observed on `2026-04-18` and `2026-04-19` over SSH:

- host alias: `3090`
- hostname: `george-linux-server`
- GPU: `NVIDIA GeForce RTX 3090`
- GPU memory: `24 GiB`
- driver: `550.163.01`
- Python: `3.12.3`
- shared project env: `~/shared/envs/projects/conflict_aware_editing-env`
- dedicated calibration env: `~/shared/envs/projects/autoresearch-continual-learning-env`
- workflow helpers present:
  - `prepare-workspace-repo`
  - `clone-torch-train-env`
  - `show-3090-workflow`
- package support confirmed in the shared env:
  - `torch`
  - `transformers`
  - `peft`
  - `yaml`

The shared `conflict_aware_editing` env is usable for older harness work, but it was observed with
`transformers 4.23.1`, which is too old for the current pilot-model tokenizers. A dedicated writable
env was therefore cloned for this repo and upgraded for calibration work.

Observed dedicated calibration env package versions after upgrade:

- `transformers 5.5.4`
- `peft 0.19.1`
- `accelerate 1.13.0`
- `torch 2.6.0+cu124`
- `torchvision 0.21.0+cu124`

## Hub access facts

The host can reach Hugging Face, and the official pilot matrix is now runnable there:

- `Qwen/Qwen3.5-4B-Base`: accessible without a token
- `meta-llama/Llama-3.1-8B`: accessible once a Hugging Face token with approved access is provided
- `google/gemma-3-4b-pt`: accessible once a Hugging Face token with approved access is provided

This means:

- the full three-way pilot matrix has now been completed honestly
- the selected pilot pair is no longer hypothetical
- run-class calibration can now proceed on the chosen stack instead of on an unresolved matrix

## Calibration procedure

The intended procedure is:

1. Select the fixed pilot skeleton from `protocol/MODEL_SURFACE_PILOT.yaml`.
2. Run `scripts/calibrate_stack.py` on the selected candidate model/surface pairs.
3. Measure:
   - train tokens per second
   - eval tokens per second
   - peak VRAM
   - invalid-run rate across short repeated rehearsals
   - eval fraction of runtime
4. Freeze `smoke_v1`, `dev_v1`, and `confirm_v1` only after the selected pair has actual measurements on the observed stack.

The calibration script is deliberately narrow:

- it uses a fixed adapter probe rather than a changing research method
- it is intended to measure stack fit and bounded-run composition
- it does not claim scientific quality improvements on its own

## Benchmark results

The official three-model fit screen on the observed 3090 produced:

| Model | Status | Peak VRAM (GiB) | Invalid-Run Rate | Mean Train Tokens/s |
| --- | --- | ---: | ---: | ---: |
| `google/gemma-3-4b-pt` | `ok` | `8.27` | `0.0` | `234.49` |
| `meta-llama/Llama-3.1-8B` | `ok` | `15.14` | `0.0` | `177.53` |
| `Qwen/Qwen3.5-4B-Base` | `ok` | `8.18` | `0.0` | `134.46` |

The bounded visible-dev pack initially exposed harness-side invocation bugs rather than true model failures. After repairing the invocation path in `scripts/profile_visible_dev.py`, all three candidates scored `1.0` exact match on `dev_visible_v1`.

The selected-pair decision record is captured in `protocol/MODEL_SURFACE_PILOT.yaml`.

## Current status

Current status is:

- infrastructure ready for calibration probes
- host facts recorded
- dedicated calibration env created on the 3090
- full pilot matrix completed
- implementation pilot pair selected:
  - `google/gemma-3-4b-pt`
  - `top4_standard`
- run classes still defined but not frozen in `protocol/RUN_CLASSES.yaml`

This is enough to stop treating the base-model choice as open for the first implementation stack, while still keeping run-class envelopes honest until further calibration is attached.

## Post-selection calibration policy

Now that the pilot pair is selected, further calibration work should default to:

1. Selected-stack rehearsals
   - Example: `google/gemma-3-4b-pt` + `top4_standard`
   - Purpose: freeze run classes, stress the chosen stack, and support real baseline/mainline runs

2. Non-candidate stack rehearsals
   - Example: cached `google/gemma-2-2b-it`
   - Purpose: verify that a changed script or remote workflow still works end-to-end
   - Limitation: these runs do not change the selected pilot or freeze run classes

## Commands

Example local-to-remote flow:

```bash
python3 scripts/check_workspace.py --task-mode constitution_bootstrap
export AUTORESEARCH_REMOTE_HOST=3090
export AUTORESEARCH_WAKE_ENDPOINT='http://<wake-host>:<port>/wake'
curl -fsS "$AUTORESEARCH_WAKE_ENDPOINT"
ssh "$AUTORESEARCH_REMOTE_HOST" 'bash -lc "prepare-workspace-repo autoresearch-continual-learning"'
rsync -az ./ "$AUTORESEARCH_REMOTE_HOST":~/workspace/autoresearch-continual-learning/
ssh "$AUTORESEARCH_REMOTE_HOST" 'bash -lc "~/shared/envs/projects/autoresearch-continual-learning-env/bin/python ~/workspace/autoresearch-continual-learning/scripts/calibrate_stack.py --output ~/shared/artifacts/autoresearch-continual-learning/calibration/gemma3_4b_top4_probe.json"'
```

The calibration outputs should be saved under `~/shared/artifacts/...`, not in the repo.
