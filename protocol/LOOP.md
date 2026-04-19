# Harness Loop

This file is the canonical loop contract for `autoresearch-continual-learning`.

It translates the higher-level design docs and `program.md` into the default operating loop that a Codex Agent should follow once the harness exists.

## Attribution Boundary

This repository is a public fork and design pivot inspired by `karpathy/autoresearch`.

What is preserved:

- tight loop constraints
- bounded experiment budgets
- structured result parsing
- human ownership of the research organization

What is adapted:

- the target domain is continual learning / knowledge editing
- success is multi-metric rather than scalar
- regressions and interference matter as much as gains
- anti-shim enforcement is part of the loop, not an afterthought

## Core Objective

Produce valid method improvements under fixed constraints.

Do not optimize for visible benchmark movement by any means available.

## Default Operating Mode

The intended operating model is:

- human in the loop for governance
- agent in the loop for execution

The human defines:

- editable surfaces
- immutable surfaces
- benchmark packs
- promotion rules
- when methodology changes require approval

The agent owns:

- one bounded hypothesis at a time
- bounded code or config changes
- preflight validation
- run-spec freezing
- exactly one heavyweight run submission
- artifact parsing
- ledger updates
- promotion, discard, or escalation within the allowed rules

## Preconditions Before Method Work

Before any default-loop method iteration, the agent must:

1. Read the protocol and governance files in full.
2. Identify the active task mode and allowed edit surfaces.
3. Identify the current champion or baseline artifact.
4. Identify the development pack and the protected confirmation pack.
5. Verify that the workspace passes preflight and immutable-surface checks.

If any of these are unclear, the agent must not start a heavyweight run.

## One Iteration

The default recurring loop is:

1. Choose one bounded hypothesis.
2. Make the smallest coherent change needed to test it.
3. Run lightweight checks first.
4. Freeze one run spec.
5. Submit exactly one heavyweight run.
6. Parse the structured artifact.
7. Classify the result as:
   - `promote`
   - `discard`
   - `invalid`
   - `needs_human_decision`
8. Append the result to the ledgers.
9. If the result is `promote`, advance champion state through the harness rules.
10. Continue.

## Heavyweight Run Rule

Only one heavyweight run may be active at a time.

This must be enforced by infrastructure, not by prompt etiquette.

For this project, heavyweight runs include:

- training runs
- major validation sweeps
- expensive knowledge-editing comparison batches
- other serial-budget experiments explicitly marked as heavyweight by the run-class policy

## Inner And Outer Loop

### Inner Loop

Fast iteration on a visible development pack.

Properties:

- bounded run class
- fixed manifests
- structured artifact
- keep/discard discipline

### Outer Loop

Promotion gate for promising changes.

Properties:

- protected confirmation pack
- stricter integrity requirements
- possible human review
- no silent promotion from visible-dev wins alone

## Promotion Stance

Promotion requires all of the following:

1. the artifact is valid
2. target quality is preserved or improved within the defined tolerance rules
3. at least one relevant regression or interference dimension improves
4. no forbidden shortcut or undeclared capacity was used
5. protected confirmation succeeds where required

Anything else becomes `discard`, `invalid`, or `needs_human_decision`.

## Non-Goals

This loop is not a generic software-engineering orchestrator.

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
