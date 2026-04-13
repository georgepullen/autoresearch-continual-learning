# Guardrails and Anti-Shim Rules

## Threat model

An autonomous research agent can improve headline metrics for the wrong reasons.

Common failure modes include:

- evaluation hacking
- data leakage
- hidden extra capacity
- prompt stuffing
- changing what is being measured
- narrowing the problem instead of solving it

For continual learning and knowledge editing, these risks are high because the objective already contains tradeoffs and visible benchmarks are easy to overfit.

## Non-negotiable anti-shim rules

1. **Immutable evaluation surface by default**
   - evaluation code, manifests, and core metric definitions are read-only unless the task is explicitly a protocol-change task

2. **No hidden capacity**
   - no retrieval, memory store, helper model, prompt stuffing, or post-hoc output patching unless the method explicitly claims them

3. **No per-example hand tuning**
   - preserve sets, update sets, thresholds, or target prompts must not be hand-authored from observed eval outputs

4. **No slice shopping**
   - the agent must not swap to easier manifests after observing bad outcomes

5. **No manual cherry-picking**
   - failed runs must be logged
   - comparisons must be artifact-backed

6. **No silent protocol edits**
   - if evaluation logic or locked manifests change, the run is not a method iteration; it is a protocol-change event

## Operational guardrails

The loop should enforce at least these checks:

### Edit-surface check

Before a method run, verify that only approved files changed.

### Immutable-path hash check

Compute and compare hashes for:

- evaluation docs
- locked manifests
- metric code

Any unexpected drift makes the run `invalid`.

### Artifact contract check

A run only counts if the resulting artifact:

- has the expected schema
- records the actual command and config
- includes runtime and memory data
- identifies the baseline and comparison scope

### Baseline gate

A method cannot be promoted if it fails the repo’s minimum baseline conditions.

For `conflict_aware_editing`, that means at least:

- match or exceed raw `memit` target success
- improve at least one interference metric on the same slice

### Two-pack discipline

Use:

- visible dev pack for rapid iteration
- locked or hidden confirmation pack for promotion

This reduces overfitting to the visible benchmark.

## Human escalation triggers

Automatically classify the run as `needs_human_decision` when:

- a method improves one metric family while hurting another materially
- runtime or memory cost increases significantly
- a protocol or benchmark surface needs to change
- the method adds meaningful complexity for marginal gains

## Core principle

If the loop makes cheating easy, the loop is wrong.

The guardrails must be part of the experiment system, not just a note in the README.
