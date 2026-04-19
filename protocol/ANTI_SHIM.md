# Anti-Shim Policy

This file is the operational anti-shim contract for the harness.

The core principle is simple:

If the loop makes cheating easy, the loop is wrong.

## Threat Model

An autonomous agent can improve visible metrics for the wrong reasons.

High-risk failure modes include:

- evaluation hacking
- data leakage
- hidden extra capacity
- prompt stuffing
- slice shopping after seeing bad results
- changing what is measured instead of improving the method
- patching outputs after inference

## Non-Negotiable Rules

### Immutable evaluation surface by default

Evaluation code, locked manifests, and core metric definitions are read-only unless the task is explicitly a protocol-change task.

### No hidden capacity

The method must not use undeclared:

- retrieval
- memory stores
- helper models
- prompt stuffing
- post-hoc output patching

If extra capacity is part of the claimed method, it must be explicit in the run spec and artifact.

### No per-example hand tuning

Agents must not hand-author preserve sets, update sets, thresholds, or prompts from observed evaluation failures.

### No slice shopping

Agents must not switch to easier visible manifests after observing poor outcomes.

### No cherry-picked claims

Every counted claim must be backed by a valid artifact and recorded in the ledgers.

### No silent protocol edits

If protocol files, locked manifests, or core schema/evaluation surfaces change, the work is not a default method iteration.

It is a protocol-change event and must be treated as such.

## Declaration Requirements

Every counted run must declare:

- the base model
- the editable surface
- the run class
- the packs or manifests used
- the claimed method family
- any additional capacity used

Undeclared capacity is an integrity failure.

## Invalidation Rules

The harness must classify a run as `invalid` when any of the following occurs:

- immutable surfaces drift unexpectedly
- the artifact is missing or malformed
- undeclared capacity is detected
- wrong pack or wrong run-class usage is detected
- post-hoc patching is used to improve outputs
- the recorded command or config does not match the claimed method

## Enforcement Expectations

The implemented harness should enforce at least:

1. edit-surface checking against `protocol/SURFACES.yaml`
2. immutable-path hash verification
3. structured artifact validation
4. declared-capacity checks
5. append-only ledger recording

## Interpretation Rule

When in doubt, prefer invalidating suspicious gains over accepting a possibly shimmed result.

False negatives are costly, but false positives that promote bad science are worse.
