# Case Study: `conflict_aware_editing`

## Why this repo is a good first case

`conflict_aware_editing` is not a pure continual-learning repo, but it is close enough to the class of problems we care about:

- the method must update behavior
- the update can damage nearby behavior
- interference is part of the objective
- evaluation already has explicit anti-cheating rules

This makes it a strong first case study for an autoresearch-style loop.

## What the repo already gets right

The repo already has three strong governance documents:

- quality gates
- evaluation protocol
- artifact contract

These already forbid many common shims:

- changing slices after seeing results
- hiding extra capacity or helper models
- rewriting outputs after inference
- cherry-picking examples
- claiming results without saved artifacts

That means the autonomous loop should not replace these rules. It should operationalize them.

## Optimal fitted loop for this repo

The optimal loop for `conflict_aware_editing` is not:

- edit the whole repo
- rerun arbitrary commands
- keep anything that looks subjectively better

It should be:

1. **Read-only protocol surfaces**
   - `docs/QUALITY_GATES.md`
   - `docs/EVAL_PROTOCOL.md`
   - `docs/ARTIFACT_CONTRACT.md`
   - default fixed manifests unless explicitly in protocol-edit mode

2. **Editable method surfaces**
   - controller logic
   - wrapper-level method code
   - reusable experiment-spec or runner glue

3. **Frozen experiment spec**
   - model
   - algorithms compared
   - manifest paths
   - command
   - artifact destination

4. **Single heavyweight execution**
   - run on the 3090
   - produce one artifact

5. **Structured decision**
   - compare against the relevant baseline artifact
   - determine `promote`, `discard`, `invalid`, or `needs_human_decision`

## Recommended editable surfaces

Default autonomous method loop should allow edits in:

- `src/conflict_aware_editing/conflict_aware_memit.py`
- `src/conflict_aware_editing/scope_constrained_memit.py`
- wrapper-level helper modules in `src/conflict_aware_editing/`
- additive experiment helper scripts in `scripts/`

Default autonomous method loop should not allow edits in:

- `third_party/`
- `docs/QUALITY_GATES.md`
- `docs/EVAL_PROTOCOL.md`
- `docs/ARTIFACT_CONTRACT.md`
- locked manifest files under `configs/validation/`
- evaluation prompt definitions unless explicitly in protocol mode

## Why the loop is harder than original `autoresearch`

The original loop optimizes one scalar metric.

This repo requires balancing:

- rewrite success
- paraphrase success
- neighborhood interference
- distracting-neighborhood interference
- RippleEdits behavior
- runtime
- peak VRAM

So the autonomous loop must reason in multi-metric terms, not just “lower is better”.

## Development and confirmation packs

The fitted loop should distinguish:

### Development pack

Cheap, versioned, visible, used for frequent iteration.

For this repo, that likely means:

- `counterfact_dev_v1.json`
- `ripple_dev_v1.json`

or a future cleaner dev pack.

### Confirmation pack

Locked pack used to confirm a promising change before promotion.

For this repo, that likely means:

- `counterfact_small_v1.json`
- `ripple_small_v1.json`

### Optional hidden pack

Best practice for long-term autonomy would be a host-local held-out pack that the agent cannot casually inspect or tune on.

## Core takeaway

The right move is not to bend this repo into a generic autonomous harness.

The right move is to define:

- what must stay frozen
- what the agent may modify
- what counts as a valid experiment
- what counts as a real improvement

and then let the agent run aggressively inside that envelope.
