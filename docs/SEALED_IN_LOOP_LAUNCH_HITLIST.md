# Sealed In-Loop Launch Hitlist

This is the final preflight checklist before copying the harness into a clean
3090 workspace and starting a sealed in-loop Codex researcher.

## Current Truth

- Current accepted champion: `baseline-20260510T133850Z`
- Champion model: `Qwen/Qwen3.5-4B-Base`
- Surrogate model: `Qwen/Qwen3.5-0.8B-Base`
- Active surface: `qwen35_top8_hybrid_attention_mlp`
- Accepted method: `baseline_seq_lora_ft_v11_qwen35_wide_fact_replay`
- Public substrate: `cl_seq_train_v4_counterfact_standard`
- Visible pack: `cl_dev_visible_v4_counterfact_standard`
- Locked pack summary: `cl_confirm_locked_v4_counterfact_standard`

Protected confirmation for the accepted champion:

- target exact: `0.8125`
- anchor exact: `0.5451`
- joint exact: `0.4201`

## Pre-Copy Hygiene

- `python3 scripts/run_loop.py status --json` reports:
  - `active_run_lock: null`
  - current champion `baseline-20260510T133850Z`
  - `baseline_acceptance_tier: accepted`
  - `next_action: start_next_method_iteration`
- generated local files are absent:
  - `.DS_Store`
  - `.Rhistory`
  - `__pycache__/`
  - `*.pyc`
- `git diff --check` passes
- targeted tests pass:
  - `tests.test_bootstrap_baseline_acceptance`
  - `tests.test_qwen_model_lanes`
  - `tests.test_counterfact_v4`

## Copy Discipline

Use `ops/sealed_workspace_rsync_excludes.txt` when copying the repo into the
sealed in-loop workspace. The sealed workspace should not include:

- `.git/`
- `artifacts/`
- local active locks
- generated caches
- repo-development notes under `docs/repo-dev/`
- codepacks and deep-research planning documents
- old prompt bundles such as `repomix-output.md`

The sealed workspace should include:

- `README.md`
- `program.md`
- `docs/` current contract docs
- `protocol/`
- `data/` public manifests and public pack summaries/hashes
- `eval/`
- `method/`
- `scripts/`
- `tests/`
- `experiments/champion.json`
- `experiments/ledgers/`
- `experiments/research_state/`
- `experiments/specs/`
- `experiments/submissions/`
- `experiments/confirmation/`

Large run artifacts stay under `~/shared/artifacts`, not in the sealed repo.

## Launch Boundary

The sealed in-loop agent should be a clean 3090 Codex instance, not a personal
Mac Codex profile. It should receive the repo-local instructions and should not
inherit George's personal skills, credentials, logs, history, or profile state.

The Mac-side monitor remains responsible for:

- reviewing terminal iterations
- checking ledger/artifact deltas
- fixing narrow harness blockers
- managing stale locks
- escalating governance decisions

The in-loop agent is responsible for:

- choosing one bounded method hypothesis
- editing only approved method surfaces
- freezing one spec
- submitting one heavyweight run
- parsing and recording the result
- continuing within the loop contract

## First In-Loop Objective

The status machine currently points to `branch_hyper_lora_v0` as the first
mainline direction after the accepted Qwen baseline. This branch now screens on
`qwen35_surrogate_wide`; a passing surrogate result should freeze a matching
`qwen35_champion_wide` HyperLoRA run before any promotion claim.

The first frozen method spec should therefore resolve to:

- run class: `qwen35_surrogate_dev_v1`
- method family: `hyper_lora_v0`
- base model: `Qwen/Qwen3.5-0.8B-Base`
- surface: `qwen35_top8_hybrid_attention_mlp`
- comparison scope: `qwen35_surrogate_counterfact_v4_wide`
- declared trainable capacity: `5529352`

The first autonomous iteration should not change protocol, benchmark packs,
confirmation thresholds, locked data, or run-class policy.
