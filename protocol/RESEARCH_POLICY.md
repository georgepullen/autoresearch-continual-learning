# Research Policy

This file defines when the harness may enter bounded research or diagnosis mode.

The default loop is exploit-oriented, not literature-hunting.

## Default Stance

Do not run a standing parliament of paper scouts.

Use one main exploit agent by default.

Any research-mode expansion must be:

- explicitly triggered
- narrowly scoped
- artifact-backed where possible
- temporary rather than permanent

## Trigger Conditions

### `plateau`

Trigger when recent valid runs fail to produce meaningful progress under the current method family and run-class discipline.

Expected response:

- generate one bounded research or diagnosis memo
- keep the current champion unchanged
- admit at most one new external mechanism or research direction into the next cycle
- detect from append-only decision history rather than operator intuition alone

### `confirm_regression_pattern`

Trigger when visible-dev wins repeatedly fail protected confirmation in a similar way.

Expected response:

- generate one bounded diagnosis memo
- examine overfitting, pack mismatch, or incorrect development assumptions
- prefer diagnosis of current assumptions before importing novelty
- detect from repeated confirmation-failure patterns in recent decision records

## Memo Budget

Each trigger event should produce at most one bounded memo per cycle.

A memo should specify:

- what pattern triggered it
- what evidence was inspected
- what hypotheses are proposed
- what one next action is justified
- what evidence would falsify that action

The memo should be stored as an append-only record in the research memo ledger.

## External Idea Admission Rule

External mechanisms should not enter the codebase casually.

Require:

1. a concrete failure pattern in current runs
2. one bounded candidate mechanism that addresses that pattern
3. a clear way to test it inside existing protocol constraints

Do not admit novelty just because it sounds promising in literature.

## Non-Goal

This policy must not turn the harness into a permanent architecture-grafting loop.

Research mode exists to unstick the exploit loop, not to replace it.
