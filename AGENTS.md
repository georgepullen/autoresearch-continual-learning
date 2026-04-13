# Repository Guidelines

## Mandatory Read First
Before making changes in this repository, read these files in full:

- `README.md`
- `docs/DESIGN_PRINCIPLES.md`
- `docs/LOOP_SPEC.md`
- `docs/CASE_STUDY_CONFLICT_AWARE_EDITING.md`
- `docs/GUARDRAILS_AND_ANTI_SHIM.md`

This repo is a design and contract repo for autonomous continual-learning / knowledge-editing research loops. Do not treat it as a generic coding harness.

## Working Rules
- Preserve explicit attribution to the upstream `karpathy/autoresearch` fork origin.
- Keep public language general to continual learning / knowledge editing unless a case study explicitly needs repo-specific detail.
- Prefer design contracts, schemas, and loop rules over speculative implementation sprawl.
- Do not copy large chunks of upstream prose into new documents.
- Treat anti-cheating and anti-shim constraints as first-class design requirements.

## Project Structure
- `docs/`: design docs, case studies, guardrails
- `program.md`: the evolving research-agent instruction file for the adapted loop

## Commit Style
Use concise imperative subjects such as:
- `Define continual learning loop contract`
- `Add anti-shim guardrails for autonomous editing`

