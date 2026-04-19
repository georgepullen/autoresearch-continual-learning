# HITL Policy

This file defines the minimum human-in-the-loop policy for the harness.

The project stance is:

- human in the loop for governance
- agent in the loop for execution

## Human Responsibilities

The human owns:

- the research organization
- editable and immutable surface policy
- benchmark-pack definitions
- promotion thresholds
- approval-only configuration changes
- methodology changes that alter what is being measured
- decisions on ambiguous tradeoffs recorded as `needs_human_decision`

## Agent Responsibilities

The agent owns:

- bounded hypothesis selection within the allowed research space
- bounded implementation changes
- preflight validation
- run-spec freezing
- heavyweight run submission
- artifact parsing
- ledger updates
- automatic `promote`, `discard`, or `invalid` decisions when the rules are clear

## Required Human Escalation

Escalate to `needs_human_decision` when:

- a gain is real but materially ambiguous
- runtime or memory cost rises enough that significance is unclear
- a protocol or benchmark surface needs to change
- a candidate adds meaningful complexity for marginal gains
- the harness cannot resolve a tradeoff within the configured rules

## Explicit Non-Requirements

The human is not expected to:

- micromanage each experiment
- read raw logs for every run
- choose every next hypothesis manually
- approve routine bounded method iterations individually

## Design Requirement

If the harness depends on constant human intervention for ordinary valid iterations, it is too weakly specified.

The human should intervene rarely and for governance-level reasons, not because the loop is underspecified.
