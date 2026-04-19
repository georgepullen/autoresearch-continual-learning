# Implementation Index

This document indexes the current repository so implementation can start from the existing contracts rather than from ad hoc interpretation.

## Current State

This repository is currently a constitution-first harness repo, not a finished autonomous experimentation repo.

What exists now:

- design constraints
- loop contracts
- anti-shim rules
- case-study fitting guidance
- an implementation roadmap
- a research-agent instruction file
- `protocol/`, `method/`, `eval/`, `data/`, `experiments/`, `scripts/`, and `locks/`
- a selected pilot pair for the observed 3090 stack:
  - `google/gemma-3-4b-pt`
  - `top4_standard`

What does not exist yet:

- a finished end-to-end autonomous loop runner
- frozen run-class envelopes
- non-bootstrap visible-dev and confirmation packs
- real baseline and mainline experiment lanes on the selected Gemma stack

That means the next implementation work should focus on wiring the existing constitution layer into a real experimental loop rather than on creating the constitution layer from scratch.

## Repository Map

| Path | Role | Authority | Implementation Use |
| --- | --- | --- | --- |
| `AGENTS.md` | Repo working rules for agents and contributors | Mandatory local contribution policy | Read before edits; preserve upstream attribution and anti-shim emphasis |
| `README.md` | Project charter and fork boundary | High-level public framing | Use to keep implementation aligned with the repo's stated scope |
| `program.md` | Current autonomous research instructions | Interim agent runtime contract | Primary source to decompose into future `protocol/*` files |
| `docs/DESIGN_PRINCIPLES.md` | Why the adapted loop exists | Foundational design rationale | Use to justify narrow surfaces, bounded runs, structured outcomes |
| `docs/LOOP_SPEC.md` | Core loop and terminal decisions | Canonical loop contract | Source for `protocol/LOOP.md` and `protocol/STATE_MACHINE.md` |
| `docs/GUARDRAILS_AND_ANTI_SHIM.md` | Threat model and enforcement expectations | Canonical integrity contract | Source for `protocol/ANTI_SHIM.md`, `protocol/hashes.lock`, and preflight checks |
| `docs/CASE_STUDY_CONFLICT_AWARE_EDITING.md` | First-target fit and case-study boundaries | Case-study operating guidance | Source for first editable-surface policy and first dev/confirm pack assumptions |
| `docs/IMPLEMENTATION_PLAN.md` | Phased build roadmap and target repo shape | Primary build plan | Source for directory creation order, initial scripts, and milestone definition |
| `docs/DEEP_RESEARCH_PROMPT.md` | Broad research/design input brief | Background reference | Use for original problem framing, non-goals, and external-search scope |
| `docs/DEEP_RESEARCH_FOLLOWUP.md` | Corrections to the original research brief | Background refinement | Use for plateau triggers, skeptic pass, calibrated run classes, pilot matrix, and replay policy |

## Contract Hierarchy

When implementation decisions conflict, use this order:

1. `AGENTS.md` for repo-local editing rules.
2. `README.md` plus `docs/DESIGN_PRINCIPLES.md` for project scope and fork boundary.
3. `docs/LOOP_SPEC.md` plus `docs/GUARDRAILS_AND_ANTI_SHIM.md` for loop and integrity contracts.
4. `docs/CASE_STUDY_CONFLICT_AWARE_EDITING.md` for first-target operational fitting.
5. `program.md` for the current autonomous behavior before `protocol/` exists.
6. `docs/IMPLEMENTATION_PLAN.md` for build sequencing and initial file layout.
7. `docs/DEEP_RESEARCH_PROMPT.md` and `docs/DEEP_RESEARCH_FOLLOWUP.md` as rationale and refinement, not as first-line runtime contract.

## Document Dependencies

These documents already imply the first implementation split:

- `program.md` should be decomposed into machine-checkable protocol files.
- `docs/LOOP_SPEC.md` defines the outer loop shape and the four terminal outcomes.
- `docs/GUARDRAILS_AND_ANTI_SHIM.md` defines what must be enforceable by validation, hashing, and artifact checks.
- `docs/CASE_STUDY_CONFLICT_AWARE_EDITING.md` defines the first target repo's editable vs immutable surfaces.
- `docs/IMPLEMENTATION_PLAN.md` defines the first missing directories, scripts, ledgers, and lock file.

## Settled Decisions

These look implementation-ready rather than exploratory:

- The loop is serial and allows only one heavyweight run at a time.
- Public framing should stay centered on continual learning and knowledge editing, not generic orchestration.
- Promotion cannot be based on one scalar metric.
- Terminal outcomes are exactly:
  - `promote`
  - `discard`
  - `invalid`
  - `needs_human_decision`
- Immutable evaluation and protocol surfaces are first-class requirements.
- Hidden capacity, silent protocol edits, slice shopping, and artifact-free claims are invalid by design.
- The first milestone is protocol and harness infrastructure, not method novelty.

## Open Choices That Still Need Evidence

These are intentionally not frozen yet:

- the exact run-class envelopes on the selected Gemma stack
- the exact editable suffix or band
- exact `smoke_v1`, `dev_v1`, and `confirm_v1` durations
- tolerance bands for borderline promotion decisions
- the exact confirmation-pack implementation and host-local storage boundary
- whether the optional skeptic pass is needed immediately or can wait

The follow-up doc is explicit that these should be decided by pilots or calibration rather than taste. The base-model choice for the first implementation stack is no longer open; the remaining open items are run-class and evaluation-surface refinements on that stack.

## Implementation Entry Points

If implementation continues now, the most direct next build order is:

1. Freeze longer selected-stack run classes in `protocol/RUN_CLASSES.yaml`.
2. Wire the selected Gemma pair through the real baseline and mainline training path.
3. Build the end-to-end loop runner that connects:
   - preflight
   - spec freeze
   - submission
   - artifact parse
   - decision
   - champion update
4. Replace bootstrap visible-dev assumptions with the first trustworthy public dev pack and protected confirmation flow.
5. Start real baseline and mainline experiment comparisons on the selected stack.

## Practical Reading Order

For someone starting implementation, read in this order:

1. `README.md`
2. `docs/DESIGN_PRINCIPLES.md`
3. `docs/LOOP_SPEC.md`
4. `docs/GUARDRAILS_AND_ANTI_SHIM.md`
5. `docs/CASE_STUDY_CONFLICT_AWARE_EDITING.md`
6. `program.md`
7. `docs/IMPLEMENTATION_PLAN.md`
8. `docs/DEEP_RESEARCH_FOLLOWUP.md`
9. `docs/DEEP_RESEARCH_PROMPT.md`

## Bottom Line

The repository is already past the pure constitution stage.

The next code should not reopen model selection. It should:

- treat `google/gemma-3-4b-pt` + `top4_standard` as the first implementation stack
- finish run-class freezing on that stack
- wire the real experimental loop
- then begin honest baseline and method comparisons

That is the shortest path from "constitution-first harness" to "trustworthy autonomous research loop."
