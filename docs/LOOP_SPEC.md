# Loop Specification

## Core abstraction

The adapted loop should be:

1. read the research program
2. choose one bounded hypothesis
3. edit only approved method surfaces
4. run lightweight checks
5. freeze one bounded experiment spec
6. submit exactly one heavyweight run
7. parse the structured result
8. decide:
   - `promote`
   - `discard`
   - `invalid`
   - `needs_human_decision`
9. update the experiment ledger
10. continue

## Human in the loop

The intended model is:

- **human in the loop for governance**
- **agent in the loop for execution**

The human should define:

- editable surfaces
- immutable surfaces
- benchmark packs
- promotion criteria
- when methodology changes require review

The agent should own:

- bounded code changes
- experiment setup
- run execution
- artifact parsing
- iterative promotion or discard decisions within the allowed rules

## Loop layers

### Inner loop

Fast autonomous iteration on a development pack.

Properties:

- bounded compute budget
- fixed development manifests
- structured result parsing
- keep/discard discipline

### Outer loop

Promotion gate for method changes that appear promising.

Properties:

- locked confirmation pack
- stricter artifact validation
- possible human review
- no silent promotion from dev-only wins

## Decision states

The loop should not collapse everything into “keep” or “discard”.

Use:

- `promote`
  - method improved within the allowed contract
- `discard`
  - no meaningful improvement or harmful tradeoff
- `invalid`
  - artifact invalid, eval surface changed, hidden side channel, or run failure
- `needs_human_decision`
  - meaningful but ambiguous tradeoff or methodology change

## Heavyweight run rule

Only one heavyweight run can be active at a time.

This must be enforced by infrastructure, not by prompt instructions.

For this project, “heavyweight run” includes:

- training runs
- large validation sweeps
- major knowledge-editing comparison batches
- expensive continual-learning evaluations

## Promotion rule

The branch or champion state should only advance when:

1. the artifact is valid
2. target quality is preserved or improved
3. at least one relevant regression / interference dimension improves
4. no forbidden shortcut was used

Anything else is `discard` or `needs_human_decision`.
