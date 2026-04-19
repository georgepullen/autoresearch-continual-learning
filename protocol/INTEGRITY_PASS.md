# Integrity Pass

This file defines the optional narrow skeptic pass for the harness.

## Role

The harness may use one cheap integrity or skeptic pass in addition to the main exploit agent.

This role is not a coequal planner and does not own strategy.

Its job is narrow:

- detect protocol drift
- detect undeclared capacity
- challenge weak promotion justifications
- flag suspicious novelty claims

## Inputs

The integrity pass should be read-only.

It may inspect:

- the frozen run spec
- immutable-surface status
- the parsed artifact
- the current champion context
- recent ledger history where needed

## Outputs

The integrity pass may emit:

- `no_integrity_objection`
- `integrity_concern`
- `promotion_justification_weak`
- `possible_protocol_change_disguised_as_method_change`

These are advisory outputs for the harness or human reviewer.

They are not terminal run outcomes.

## Limits

The integrity pass may not:

- launch heavyweight runs
- change code
- rewrite the run spec
- override the terminal state machine by itself
- become a permanent multi-agent review committee

## When To Use It

Use it sparingly, especially when:

- a candidate is near promotion
- a result looks surprisingly strong
- a methodology boundary may have been crossed
- a trigger such as `plateau` or `confirm_regression_pattern` has fired

## Design Requirement

If the exploit loop cannot function without constant skeptic intervention, the main loop is not specified tightly enough.
