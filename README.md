# autoresearch-continual-learning

`autoresearch-continual-learning` is a public fork and design pivot inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

The goal is not to reproduce the original single-file language-model training loop. The goal is to adapt the core ideas that make `autoresearch` effective to a harder class of research problems:

- continual learning
- knowledge editing
- interference-sensitive model updates
- multi-metric evaluation with regression risks
- scarce-resource workflows where only one heavyweight run can happen at a time

## Why this exists

`autoresearch` is powerful because it makes autonomous research tractable by constraining the loop:

1. narrow editable surface
2. fixed evaluation surface
3. fixed compute budget
4. machine-readable result
5. keep-or-discard discipline

For continual learning and knowledge editing, those same principles still matter, but the problem is harder:

- the objective is not one scalar metric
- evaluation can be gamed more easily
- regressions matter as much as gains
- the agent can accidentally create shims or hidden side channels
- heavyweight runs are more expensive and must be serialized

This repo is where that adapted loop is being designed.

## First case study

The first target repo is:

- `conflict_aware_editing`

That repo is a good stress test because it already has:

- fixed evaluation slices
- explicit quality gates
- artifact contracts
- competing method families
- a strong anti-cheating stance

The design principle here is:

**fit `conflict_aware_editing` to the optimal autonomous continual-learning / knowledge-editing loop, not the other way around.**

## Project stance

This repo is intentionally closer to `autoresearch` than to a general-purpose software-engineering orchestrator.

It may borrow selected ideas from systems like Symphony or Agent Orchestrator, but only where they improve the research loop. The center of gravity is:

- hypothesis
- bounded change
- bounded run
- structured result
- promote, discard, or escalate

not:

- issue
- agent
- PR

## Current phase

The current phase is design and contract-setting, not full implementation.

The immediate output of this repo is:

- a case study for `conflict_aware_editing`
- a loop specification for autonomous continual-learning / knowledge-editing research
- anti-shim and anti-cheat guardrails
- a research-agent `program.md`

Because this repository is a public fork, it still contains some upstream prototype files from `autoresearch`. For now, treat those as inherited reference material, not as the implementation of the continual-learning loop described here.

## Attribution and fork boundary

This repository is a public fork of `karpathy/autoresearch`, and that inspiration should remain explicit.

What is being carried over:

- the idea that autonomous research gets stronger when the loop is tightly constrained
- the emphasis on fixed budgets and structured result parsing
- the use of a human-edited research instructions file as part of the system

What is changing:

- the research domain
- the evaluation complexity
- the decision rules
- the guardrails needed to prevent evaluation hacking or hidden shims

## Documents

- [Design Principles](docs/DESIGN_PRINCIPLES.md)
- [Loop Specification](docs/LOOP_SPEC.md)
- [Case Study: conflict_aware_editing](docs/CASE_STUDY_CONFLICT_AWARE_EDITING.md)
- [Guardrails and Anti-Shim Rules](docs/GUARDRAILS_AND_ANTI_SHIM.md)
