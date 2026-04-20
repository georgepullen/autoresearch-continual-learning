# autoresearch-continual-learning

`autoresearch-continual-learning` is a public fork and design pivot inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

This repository is building a constrained autonomous research harness for:

- continual learning
- knowledge editing
- interference-sensitive updates
- multi-metric evaluation with regression risk
- single-GPU workflows where heavyweight runs must be serialized

The repository is focused on the research loop itself:

- bounded editable surfaces
- bounded runs
- structured artifacts
- anti-shim constraints
- promotion and discard rules

## What This Repo Is Today

This is no longer just a design-doc repo, but it is also not yet a fully self-running autonomous experimentation system.

Current state:

- a real `protocol/` constitution layer exists
- preflight, schema, parsing, submission, and decision scripts exist
- the 3090 pilot benchmark is complete
- the first implementation stack is selected:
  - base model: `google/gemma-3-4b-pt`
  - editable surface: `top4_standard`
- baseline and method scaffolding exist

Not done yet:

- a single end-to-end loop runner that keeps iterating by itself
- frozen run-class envelopes for the selected stack
- non-bootstrap visible-dev and confirmation packs
- real baseline and mainline experiment comparisons on the selected Gemma stack

So the honest description is:

**this repo is a constitution-first harness with a selected pilot stack, moving into real implementation.**

## Why This Exists

`autoresearch` is powerful because it makes autonomous research tractable by constraining the loop:

1. narrow editable surface
2. fixed evaluation surface
3. fixed compute budget
4. machine-readable result
5. keep-or-discard discipline

For continual learning and knowledge editing, those same ideas still matter, but the problem is harder:

- success is not one scalar metric
- regressions matter as much as improvements
- evaluation can be gamed more easily
- hidden capacity and shims are real failure modes
- heavyweight runs are expensive and must be serialized

This repo adapts the `autoresearch` mentality to that harder setting.

## Current 3090 Stack Decision

The first 3090 pilot benchmark compared:

- `Qwen/Qwen3.5-4B-Base`
- `meta-llama/Llama-3.1-8B`
- `google/gemma-3-4b-pt`

Outcome:

- all three fit on the observed 3090
- all three reached `1.0` on the repaired bounded visible-dev smoke pack
- Gemma won the implementation pilot because it matched the legitimacy checks and had the best throughput on the fixed probe

Selected pilot pair:

- `google/gemma-3-4b-pt`
- `top4_standard`

## First Case Study

The first target repo is:

- `conflict_aware_editing`

That case study is useful because it already has:

- explicit quality gates
- artifact contracts
- competing method families
- evaluation slices with regression risk
- a strong anti-cheating stance

The design principle here is:

**fit `conflict_aware_editing` to the best constrained autonomous research loop, not the other way around.**

## Project Stance

This repo is intentionally closer to `autoresearch` than to a general-purpose software-engineering orchestrator.

Its center of gravity is:

- hypothesis
- bounded change
- bounded run
- structured result
- promote, discard, or escalate

not:

- issue
- agent swarm
- PR theater

## Current Repository Shape

The main layers now present are:

- `protocol/`
  - loop contract
  - immutable/editable surfaces
  - anti-shim policy
  - promotion rules
  - run-class policy
  - pilot model/surface selection
- `scripts/`
  - workspace preflight
  - spec freezing
  - 3090 submission
  - artifact parsing
  - decision logic
  - pilot calibration and visible-dev profiling
- `eval/`
  - schema validation
  - metrics and aggregation
  - protected confirmation normalization
  - sentinel checks
- `method/`
  - editable-surface definitions
  - trainer shell
  - method scaffolding
  - baseline wrappers
- `experiments/`
  - champion state
  - append-only ledgers

## Attribution And Fork Boundary

This repository is a public fork of `karpathy/autoresearch`, and that inspiration should remain explicit.

What is preserved:

- tight loop constraints
- fixed-budget thinking
- structured result parsing
- human ownership of research organization

What is adapted:

- the target domain
- the evaluation complexity
- the decision logic
- the anti-shim and anti-cheat requirements

Because this is a public fork, some upstream prototype files still exist in the tree. Treat those as inherited reference material, not as the implementation of the current continual-learning harness.

## Documents

- [Design Principles](docs/DESIGN_PRINCIPLES.md)
- [Loop Specification](docs/LOOP_SPEC.md)
- [Case Study: conflict_aware_editing](docs/CASE_STUDY_CONFLICT_AWARE_EDITING.md)
- [Guardrails and Anti-Shim Rules](docs/GUARDRAILS_AND_ANTI_SHIM.md)
- [Protocol Loop](protocol/LOOP.md)
- [Protocol Surfaces](protocol/SURFACES.yaml)
- [Run Classes](protocol/RUN_CLASSES.yaml)
- [Calibration Notes](protocol/CALIBRATION.md)
- [Model/Surface Pilot Matrix](protocol/MODEL_SURFACE_PILOT.yaml)
- [Submission Flow](scripts/submit_run.py)
