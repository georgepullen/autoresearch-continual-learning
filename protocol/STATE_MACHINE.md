# State Machine

This file defines the canonical run-state vocabulary for the harness.

The goal is to make terminal decisions explicit without confusing them with internal trigger conditions or workflow stages.

## Design Rule

The harness has exactly four terminal run outcomes:

- `promote`
- `discard`
- `invalid`
- `needs_human_decision`

Do not add extra terminal outcomes such as `plateau`.

`plateau` and similar concepts are internal triggers, not terminal decisions.

## State Layers

The harness uses three layers of state:

1. execution stages
2. internal trigger conditions
3. terminal outcomes

## Execution Stages

These are not terminal outcomes. They describe where a run or candidate is in the loop.

Suggested canonical stages:

1. `preflight_pending`
2. `preflight_passed`
3. `spec_frozen`
4. `queued_for_heavyweight_run`
5. `heavyweight_run_active`
6. `artifact_pending`
7. `artifact_parsed`
8. `decision_ready`
9. `decision_recorded`

An implementation may add finer-grained transient stages, but it must preserve the four terminal outcomes and the trigger model below.

## Internal Trigger Conditions

These do not terminate the loop by themselves.

They influence whether the harness continues iterating normally, generates a bounded memo, or escalates.

### `plateau`

Used when recent valid runs fail to produce meaningful progress under the current hypothesis family.

Expected effect:

- trigger one bounded research or diagnosis memo
- do not silently invent a new method family
- do not become a terminal run outcome

### `confirm_regression_pattern`

Used when visible-dev wins repeatedly fail protected confirmation in a similar way.

Expected effect:

- trigger one bounded diagnosis memo
- re-examine overfitting, pack mismatch, or methodology assumptions
- do not become a terminal run outcome

### `borderline_replay_eligible`

Used when a valid artifact lands within pre-registered tolerance bands that permit one replay on an alternate approved seed.

Expected effect:

- allow exactly one replay under the same code and frozen spec family
- do not bypass protected confirmation
- do not become a fifth decision state

## Terminal Outcomes

### `promote`

Use when:

- the artifact is valid
- integrity gates pass
- promotion criteria pass
- protected confirmation requirements pass

Effect:

- update champion state through the harness
- record the promotion in the ledgers

### `discard`

Use when:

- the artifact is valid
- no meaningful improvement is demonstrated
- or the tradeoff is clearly not worth the added complexity or cost

Effect:

- keep champion unchanged
- record the failed candidate in the ledgers

### `invalid`

Use when:

- the run failed
- the artifact is malformed or incomplete
- immutable surfaces drifted unexpectedly
- undeclared capacity or a forbidden shim was used
- wrong pack, wrong run class, or another integrity-breaking mismatch is detected

Effect:

- candidate cannot be compared scientifically
- record the invalidation reason explicitly

### `needs_human_decision`

Use when:

- the artifact is valid
- the gain is real but the tradeoff is materially ambiguous
- methodology changes appear to be required
- or governance-level thresholds cannot be resolved automatically

Effect:

- keep champion unchanged until a human decision is recorded
- preserve the evidence in the ledgers

## Canonical Transition Shape

The intended high-level transition shape is:

1. `preflight_pending` -> `preflight_passed`
2. `preflight_passed` -> `spec_frozen`
3. `spec_frozen` -> `queued_for_heavyweight_run`
4. `queued_for_heavyweight_run` -> `heavyweight_run_active`
5. `heavyweight_run_active` -> `artifact_pending`
6. `artifact_pending` -> `artifact_parsed`
7. `artifact_parsed` -> `decision_ready`
8. `decision_ready` -> one of:
   - `promote`
   - `discard`
   - `invalid`
   - `needs_human_decision`

Internal triggers such as `plateau` may be computed from ledger history before or after a terminal outcome is recorded, but they do not replace the terminal decision.

## Human And Agent Boundaries

The agent may advance work through execution stages and terminal decisions when the protocol permits.

The human remains the authority for:

- changing governance-level policy
- changing immutable surfaces
- changing promotion rules
- resolving `needs_human_decision` cases

## Implementation Requirement

Any future code implementation must keep these layers separate:

- workflow stage
- trigger condition
- terminal outcome

If an implementation collapses them together, it becomes easier to hide ambiguity or accidentally treat research stalls as hard decisions.
