# Implementation Plan

This document turns the current design direction into a concrete implementation plan for
`autoresearch-continual-learning`.

The objective is not to recreate a general-purpose orchestration system. The objective is to build
a **serial, protocol-heavy continual-learning research harness** for autonomous iteration on a
conflict-aware LLM editing hypernetwork.

The implementation should preserve the current backbone:

- one main exploit loop
- one heavyweight run at a time
- immutable protocol and evaluation surfaces
- structured artifacts and append-only ledgers
- protected confirmation before promotion
- minimal HITL in the outer loop only

It should also incorporate the current corrections:

- optional narrow skeptic/integrity pass
- internal `plateau` and `confirm_regression_pattern` triggers
- calibrated run classes rather than speculative fixed timings
- tiered promotion with tolerance bands and one borderline replay
- base model and editable surface chosen by pilot matrix rather than hard-coded upfront

## End State

The first usable version of this repo should support:

1. reading a protocol-defined research program
2. validating immutable surfaces before each run
3. freezing one run spec at a time
4. enforcing a single active heavyweight run on the 3090
5. producing a structured run artifact
6. classifying the result as:
   - `promote`
   - `discard`
   - `invalid`
   - `needs_human_decision`
7. updating append-only ledgers
8. triggering bounded research or diagnosis memos only when plateau-style conditions occur

## Guiding Constraints

The implementation must preserve these rules:

- no parallel heavyweight runs
- no silent protocol changes
- no hidden retrieval, helper model, or postprocessor capacity in the mainline method
- no promotion from visible dev results alone
- no dataset or eval-pack drift without explicit versioning and approval

## Target Repo Shape

This is the intended near-term structure:

```text
autoresearch-continual-learning/
  README.md
  AGENTS.md
  program.md

  protocol/
    LOOP.md
    SURFACES.yaml
    RUN_CLASSES.yaml
    PROMOTION.yaml
    DATA_POLICY.md
    RESEARCH_POLICY.md
    HITL_POLICY.md
    ANTI_SHIM.md
    STATE_MACHINE.md
    INTEGRITY_PASS.md
    CALIBRATION.md
    MODEL_SURFACE_PILOT.yaml
    hashes.lock
    AGENTS.md

  method/
    __init__.py
    hypernet.py
    adapter_surface.py
    conflict_gate.py
    losses.py
    trainer.py
    episode_sampler.py
    baselines/
      __init__.py
      noop.py
      seq_lora_ft.py
      alphaedit_wrapper.py
      nse_wrapper.py
    AGENTS.md

  eval/
    __init__.py
    runner.py
    metrics.py
    aggregates.py
    schema.py
    sentinels.py

  data/
    registry.yaml
    train_generators/
      seq_train_v1.yaml
    packs/
      dev_visible_v1.yaml
      confirm_locked_v1.summary.yaml
      confirm_locked_v1.hash

  experiments/
    champion.json
    specs/
    ledgers/
      runs.jsonl
      promotions.jsonl
      research_memos.jsonl
      human_decisions.jsonl
      invalid_runs.jsonl

  scripts/
    check_workspace.py
    freeze_spec.py
    submit_run.py
    parse_artifact.py
    decide.py
    plateau_report.py
    rotate_hidden_pack.py

  artifacts/        # gitignored
  locks/
    active_run.lock
```

## Phase Plan

## Phase 0: Freeze The Constitution

Purpose:
- convert the current design docs into enforceable protocol files
- define what the agent may and may not edit
- define the terminal outcomes and internal triggers

Outputs:
- `protocol/LOOP.md`
- `protocol/SURFACES.yaml`
- `protocol/STATE_MACHINE.md`
- `protocol/ANTI_SHIM.md`
- `protocol/HITL_POLICY.md`
- `protocol/RESEARCH_POLICY.md`

Concrete work:
1. Move the high-level loop language from `program.md` into protocol files that tools can check.
2. Define immutable surfaces:
   - protocol files
   - eval runner
   - eval schema
   - visible dev pack manifest
   - confirmation-pack hash
3. Define editable surfaces:
   - `method/*`
   - approved baseline wrappers
   - new experiment specs
4. Define approval-only surfaces:
   - dataset registry
   - run classes
   - promotion thresholds
   - base model family
   - confirmation-pack versions

Exit criteria:
- a human can point at one file and answer “what is allowed to change?”
- a script can validate whether a patch violates the surface policy

## Phase 1: Build The Protocol Validators

Purpose:
- make protocol drift and anti-shim violations mechanically detectable

Outputs:
- `scripts/check_workspace.py`
- `protocol/hashes.lock`
- `eval/schema.py`
- `eval/sentinels.py`

Concrete work:
1. Implement a workspace checker that verifies:
   - required files exist
   - immutable file hashes match `protocol/hashes.lock`
   - the repo is clean enough to freeze a run spec
2. Define the structured result schema in `eval/schema.py`.
3. Add sentinel checks for:
   - undeclared retrieval
   - undeclared extra models
   - undeclared helper postprocessors
   - forbidden file drift
4. Add an artifact validity contract:
   - no artifact, no claim
   - malformed artifact -> `invalid`

Exit criteria:
- a broken protocol change fails before any heavyweight run starts

## Phase 2: Implement Run Specs, Ledgers, And Single-Run Locking

Purpose:
- make the loop reproducible and serial by construction

Outputs:
- `scripts/freeze_spec.py`
- `scripts/submit_run.py`
- `scripts/parse_artifact.py`
- `experiments/champion.json`
- `experiments/ledgers/*.jsonl`
- `locks/active_run.lock`

Concrete work:
1. Implement run-spec freezing into `experiments/specs/<run_id>.yaml`.
2. Require every run spec to include:
   - parent champion
   - run class
   - base model
   - editable paths
   - declared capacity
   - train generator version
   - eval pack version
   - budget envelope
3. Implement a lock mechanism so only one heavyweight run may be active at a time.
4. Define append-only ledgers:
   - all runs
   - promotions
   - invalid runs
   - research memos
   - human decisions

Exit criteria:
- the harness cannot accidentally launch two heavyweight runs at once
- every run leaves an auditable paper trail

## Phase 3: Calibrate Run Classes On The Real 3090 Stack

Purpose:
- define `smoke_v1`, `dev_v1`, and `confirm_v1` from actual throughput rather than guesswork

Outputs:
- `protocol/RUN_CLASSES.yaml`
- `protocol/CALIBRATION.md`

Concrete work:
1. Run a short calibration study on the 3090 for the candidate model/surface grid.
2. Measure:
   - tokens/sec or effective throughput
   - evaluation fraction of runtime
   - peak VRAM
   - invalid-run rate
3. Freeze real budgets for:
   - `smoke_v1`
   - `dev_v1`
   - `confirm_v1`
4. Record the hardware signature used to calibrate those classes.

Exit criteria:
- run classes have calibrated rather than speculative runtime envelopes
- comparisons are meaningful within each class

## Phase 4: Build The Model And Surface Pilot Matrix

Purpose:
- choose the first base model and first editable surface using evidence, not taste

Outputs:
- `protocol/MODEL_SURFACE_PILOT.yaml`
- `method/adapter_surface.py`
- baseline pilot scripts or config

Concrete work:
1. Compare at least these base-model candidates:
   - `Llama 3.1 8B base`
   - `Qwen3.5-4B-Base`
   - `Gemma 3 PT 4B`
2. Compare at least these editable surfaces:
   - top 4 blocks, standard target modules
   - top 6 blocks, standard target modules
   - sparse suffix with selected output modules plus a per-layer gate
3. Use a fixed minimal hypernetwork skeleton for the screen.
4. Select the winning pair lexicographically by:
   - fit and validity on the 3090
   - stability across short serial-update rehearsals
   - visible-dev metric profile
   - throughput
   - simplicity tie-breaker

Exit criteria:
- the first base model and editable surface are selected by pilot evidence

## Phase 5: Implement The First Mainline Method Family

Purpose:
- stand up the first conflict-aware hypernetwork method in the smallest useful form

Outputs:
- `method/hypernet.py`
- `method/conflict_gate.py`
- `method/losses.py`
- `method/trainer.py`
- `method/episode_sampler.py`

Concrete work:
1. Implement `hyper_lora_v0`:
   - frozen base
   - tiny hypernetwork
   - outputs coefficients or gates over a fixed low-rank adapter surface
2. Keep the trainable parameter budget small:
   - initial target roughly 10M to 30M trainable params
3. Train against a fixed visible training-episode generator.
4. Emit a structured summary artifact after each run.

Exit criteria:
- one mainline method can complete `smoke_v1` and `dev_v1`
- results can be compared against the champion and declared baselines

## Phase 6: Add Baselines And Foils

Purpose:
- keep the mainline method honest by comparing against simpler and non-parametric alternatives

Outputs:
- `method/baselines/noop.py`
- `method/baselines/seq_lora_ft.py`
- `method/baselines/alphaedit_wrapper.py`
- `method/baselines/nse_wrapper.py`

Concrete work:
1. Add a no-op baseline for sanity.
2. Add sequential LoRA fine-tuning as the simplest parametric baseline.
3. Add AlphaEdit and NSE style wrappers as stronger editing baselines.
4. Keep retrieval/external-memory methods out of the mainline claim, but leave room to include one
   declared foil later.

Exit criteria:
- the mainline method is compared against strong baselines rather than only against itself

## Phase 7: Implement Promotion Logic And Borderline Replay

Purpose:
- make decision logic explicit, multi-metric, and robust to borderline variance

Outputs:
- `protocol/PROMOTION.yaml`
- `scripts/decide.py`

Concrete work:
1. Implement Tier 0 integrity gates:
   - schema
   - hashes
   - declared capacity
   - wrong pack / wrong run class
2. Implement Tier 1 scientific floors with tolerance bands.
3. Implement Tier 2 required-win families.
4. Implement Tier 3 cost envelopes.
5. Add exactly one pre-registered replay for borderline runs:
   - same code
   - same spec
   - alternate approved seed only
6. Keep terminal outputs to:
   - `promote`
   - `discard`
   - `invalid`
   - `needs_human_decision`

Exit criteria:
- the harness can mechanically decide most runs
- ambiguous or borderline cases escalate in a controlled way

## Phase 8: Add Plateau And Confirmation-Regression Memos

Purpose:
- let the harness reason about stalls without collapsing into constant literature-grafting

Outputs:
- `scripts/plateau_report.py`
- `protocol/RESEARCH_POLICY.md`
- `protocol/INTEGRITY_PASS.md`

Concrete work:
1. Define internal trigger conditions:
   - `plateau`
   - `confirm_regression_pattern`
2. Implement memo generation:
   - one bounded research memo on plateau
   - one bounded diagnosis memo on repeated visible-dev vs confirm mismatch
3. Keep the research memo constrained:
   - one memo
   - limited literature/repo budget
   - at most one external mechanism admitted per cycle
4. Optionally run a narrow skeptic/integrity pass:
   - read-only
   - no heavy-run authority
   - no strategy ownership

Exit criteria:
- the system can react to stalls and overfitting patterns without turning into a paper-hunting loop

## Phase 9: Add Protected Confirmation

Purpose:
- prevent the visible dev pack from becoming the only scientific reality

Outputs:
- `data/packs/confirm_locked_v1.hash`
- `data/packs/confirm_locked_v1.summary.yaml`
- `eval/protected_runner.py` or equivalent logic in `eval/runner.py`

Concrete work:
1. Define a host-local locked confirmation manifest outside the public repo.
2. Store only:
   - hash
   - public composition summary
   - aggregate public result
3. Restrict the agent to aggregate confirmation feedback only.
4. Require confirmation before promotion.

Exit criteria:
- visible-dev improvements do not automatically become champion promotions

## First Concrete Milestone

The first milestone should be:

**M1: Constitution And Pilot Infrastructure**

Scope:
- create `protocol/`
- create `method/`
- create `eval/`
- create `experiments/`
- create `scripts/check_workspace.py`
- create `scripts/freeze_spec.py`
- create `scripts/submit_run.py`
- create `scripts/parse_artifact.py`
- create `scripts/decide.py`
- create the initial pilot-matrix and calibration docs

Success criteria:
- one candidate baseline can run through:
  - preflight
  - smoke
  - artifact parse
  - decision
  - ledger update

## Recommended Task Order

Implement in this order:

1. `protocol/STATE_MACHINE.md`
2. `protocol/SURFACES.yaml`
3. `protocol/ANTI_SHIM.md`
4. `protocol/RUN_CLASSES.yaml`
5. `protocol/PROMOTION.yaml`
6. `scripts/check_workspace.py`
7. `eval/schema.py`
8. `scripts/freeze_spec.py`
9. `scripts/submit_run.py`
10. `scripts/parse_artifact.py`
11. `scripts/decide.py`
12. `experiments/champion.json`
13. `experiments/ledgers/*.jsonl`
14. `protocol/MODEL_SURFACE_PILOT.yaml`
15. `protocol/CALIBRATION.md`

This order gives the project:
- protocol first
- then enforcement
- then run management
- then pilot selection
- then actual method work

## What Should Wait

Do not prioritize these before the phases above:

- a multi-agent literature council
- elaborate web dashboards
- generic issue-tracker orchestration
- generalized multi-host job routing
- support for many benchmark families at once
- retrieval-augmented or external-memory methods as the mainline claim

Those can come later if the harness proves it can make trustworthy progress in the narrow loop first.

## Definition Of “Concrete Progress”

The harness should be considered meaningfully alive once it can do this:

1. validate protocol hashes
2. freeze a run spec
3. acquire the single-run lock
4. run one baseline on the 3090
5. emit a valid structured artifact
6. classify the result
7. append it to the ledger

If it cannot yet do that, it is still a design project.
