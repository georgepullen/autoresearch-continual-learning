# Operating Contract

This file is the startup switchboard for a fresh agent in
`autoresearch-continual-learning`.

It is not a substitute for the canonical protocol files. It tells the agent
which sources to read first and which command exposes current loop state.

## Read First

On startup, read in this order:

1. `AGENTS.md`
2. `README.md`
3. `docs/DESIGN_PRINCIPLES.md`
4. `docs/LOOP_SPEC.md`
5. `docs/CASE_STUDY_CONFLICT_AWARE_EDITING.md`
6. `docs/GUARDRAILS_AND_ANTI_SHIM.md`
7. `protocol/OPERATING.md`
8. active protocol files:
   - `protocol/LOOP.md`
   - `protocol/STATE_MACHINE.md`
   - `protocol/SURFACES.yaml`
   - `protocol/ANTI_SHIM.md`
   - `protocol/BOOTSTRAP.yaml`
   - `protocol/HITL_POLICY.md`
   - `protocol/RESEARCH_POLICY.md`
   - `protocol/INTEGRITY_PASS.md`
   - `protocol/PROMOTION.yaml`
   - `protocol/RUN_CLASSES.yaml`
   - `protocol/CALIBRATION.md`
   - `protocol/MODEL_SURFACE_PILOT.yaml`

## Status Command

Use this command to inspect current loop state:

```bash
python3 scripts/run_loop.py status
```

The machine-facing JSON variant is:

```bash
python3 scripts/run_loop.py status --json
```

## Startup Branches

After reading the required files, inspect:

- champion state from `experiments/champion.json`
- active heavyweight run lock from `locks/active_run.lock`
- recent decision ledger from `experiments/ledgers/runs.jsonl`
- selected stack and run-class readiness from:
  - `protocol/MODEL_SURFACE_PILOT.yaml`
  - `protocol/RUN_CLASSES.yaml`

Then branch immediately into one of:

1. `bootstrap_baseline_path`
   - use when champion state is `bootstrap_pending` and no active champion exists
   - authority: `protocol/BOOTSTRAP.yaml`

2. `continue_active_run`
   - use when a heavyweight run lock exists or when an artifact/confirmation
     result is waiting to be parsed or decided
   - authority: `scripts/run_loop.py`

3. `default_method_iteration`
   - use when an active champion exists and no governance or orchestration
     block is present
   - authority:
     - `protocol/LOOP.md`
     - `protocol/PROMOTION.yaml`
     - `protocol/SURFACES.yaml`

4. `governance_escalation`
   - use only for explicit `needs_human_decision` cases or policy-edit work
   - authority:
     - `protocol/HITL_POLICY.md`
     - `protocol/SURFACES.yaml`

## No-Champion Rule

If `experiments/champion.json` reports `bootstrap_pending`, do not treat
missing champion context as ordinary ambiguity. Follow the explicit bootstrap
lane in `protocol/BOOTSTRAP.yaml`.

## Active-Run Rule

If `locks/active_run.lock` exists and is not stale, do not submit another
heavyweight run. Resume through `scripts/run_loop.py`.

## Canonical Sources

Keep this file short. Do not duplicate promotion rules, anti-shim policy, or
full editable-surface policy here. The canonical authority remains under
`protocol/`.
