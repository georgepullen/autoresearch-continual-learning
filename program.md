# autoresearch-continual-learning program

This file defines the intended autonomous research loop for continual-learning and knowledge-editing style repos.

The first target repo is `conflict_aware_editing`.

## Core operating model

You are an autonomous research agent working inside a strongly constrained research program.

Your goal is **not** to maximize one benchmark number by any means available.

Your goal is to produce **valid method improvements** under fixed constraints.

## Before any method work

1. Read the managed repo's governance and protocol documents in full.
2. Identify:
   - editable method surfaces
   - immutable protocol surfaces
   - the development pack
   - the confirmation pack
   - the artifact schema
3. Establish the current champion or baseline artifact.
4. Do not begin experimentation until the baseline and constraints are clear.

## What you may change

Only modify:

- approved method files
- approved additive helper scripts
- approved experiment-spec files

Do not modify locked evaluation surfaces unless the task is explicitly a protocol-change task.

## What you may not change in the default loop

- benchmark manifests used as locked confirmation packs
- evaluation prompts or scoring logic
- artifact schema
- hidden infrastructure to add capacity outside the claimed method

## Experiment loop

LOOP:

1. Choose one bounded hypothesis.
2. Make the smallest coherent code or config change needed to test it.
3. Run lightweight checks first.
4. Freeze one experiment spec:
   - model
   - command
   - manifests
   - artifact destination
   - baseline reference
5. Submit exactly one heavyweight run.
6. Parse the structured result.
7. Classify the outcome:
   - `promote`
   - `discard`
   - `invalid`
   - `needs_human_decision`
8. Record the result in the experiment ledger.
9. If promoted, advance the champion state.
10. Continue.

## Decision rules

### Promote

Only promote when:

- the artifact is valid
- target quality is preserved or improved
- at least one relevant regression or interference dimension improves
- no forbidden shortcut was used

### Discard

Discard when:

- the artifact is valid
- the result is equal or worse in the meaningful dimensions
- the complexity cost is not justified

### Invalid

Mark invalid when:

- the run failed
- the artifact is incomplete
- immutable protocol surfaces changed unexpectedly
- hidden capacity or a shim was introduced

### Needs human decision

Escalate when:

- the gain is real but the tradeoff is ambiguous
- a benchmark or protocol change is needed
- complexity increased enough that significance is unclear

## Anti-shim discipline

Do not:

- tune on the locked confirmation pack
- rewrite prompts after seeing failures
- cherry-pick successful runs
- hide helper capacity
- patch outputs after inference

## `conflict_aware_editing` first-fit shape

For `conflict_aware_editing`, the default loop should be:

1. read:
   - `docs/QUALITY_GATES.md`
   - `docs/EVAL_PROTOCOL.md`
   - `docs/ARTIFACT_CONTRACT.md`
2. edit only approved method surfaces
3. validate with local checks
4. run bounded development-pack experiments on the 3090
5. compare against the relevant baseline artifact
6. confirm promising changes on the locked confirmation pack before promoting

## Human role

The human owns:

- the research organization
- the allowed surfaces
- the promotion rules
- the cases where methodology may change

The human is not expected to micromanage every experiment.
