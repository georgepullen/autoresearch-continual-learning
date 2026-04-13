# Design Principles

## What makes `autoresearch` effective

The original `autoresearch` loop works because it sharply reduces ambiguity.

The most important principles are:

1. **Narrow editable surface**
   - the agent edits one place, not the whole repo
2. **Fixed evaluation surface**
   - the metric and evaluation code are not drifting under the agent
3. **Bounded run budget**
   - every experiment spends roughly the same scarce resource
4. **Machine-readable result**
   - the next decision can be made from structured output, not raw logs
5. **Keep-or-discard discipline**
   - branch state only advances on evidence
6. **Human edits the research organization**
   - the human shapes the instructions and constraints, not every experiment

## Why continual learning and knowledge editing are harder

Compared with a single-metric training loop, continual-learning and knowledge-editing research adds several failure modes:

- more than one success metric matters
- regression and interference matter as much as improvement
- there are more ways to hide capacity or cheat
- the line between method change and evaluation change is easier to blur
- the agent can accidentally optimize for the visible benchmark instead of the real objective

## What must change in the adapted loop

The adapted loop cannot be:

- “edit anything”
- “run one benchmark”
- “keep if one scalar improves”

Instead it should be:

1. edit only approved method surfaces
2. preserve immutable evaluation surfaces
3. run on a bounded development pack
4. emit structured multi-metric results
5. classify the outcome as:
   - `promote`
   - `discard`
   - `invalid`
   - `needs_human_decision`
6. periodically confirm against a stricter locked or hidden pack

## Guardrail stance

For this project, anti-cheating and anti-shim rules are not review extras. They are part of the research loop itself.

If a loop makes it easy for an agent to:

- hand-tune on the visible evaluation set
- sneak in retrieval or prompt stuffing
- rewrite evaluation prompts
- patch outputs after inference
- route around the claimed method

then the loop is badly designed, even if it produces superficially good numbers.
