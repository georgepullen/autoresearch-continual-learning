# Follow-Up Note For Deep Research

This is a **corrective follow-up**, not a reset.

I agree with the vast majority of the prior report:
- the center of gravity should be an `autoresearch`-style continual-learning loop, not a general software-engineering orchestration platform
- the loop should be serial, protocol-heavy, and single-champion
- the mainline bet should still be a **tiny hypernetwork over a frozen base**
- minimal HITL in the outer loop is still the right direction
- a sparse research mode is still the right direction
- protected confirmation, anti-shim rules, and immutable protocol/eval surfaces are essential

What I want adjusted is **not the architecture of the plan**, but a few places where the previous report became more specific or rigid than the current evidence really supports.

Please preserve the backbone of the original plan and only revise the points below.

---

## 1. Keep “one main exploit agent”, but allow one cheap integrity/skeptic pass

I agree that I do **not** want a standing parliament of paper scouts.

However, I want you to soften the “one-agent” recommendation slightly:
- keep **one main exploit agent**
- optionally allow **one cheap integrity/skeptic pass**
- that skeptic should not be a coequal planner or a paper-hunting committee
- its role is only:
  - protocol drift detection
  - undeclared-capacity detection
  - justification skepticism on novelty or promotion

This is a very small correction, not a change in direction.

### Why I want this adjustment

The outer-shell orchestration repos consistently separate **terminal success** from **intermediate review/handoff states**. That suggests there is value in one lightweight integrity/review voice without promoting it into a permanent committee.

### Tiny codepack A: AO state vocabulary supports non-terminal review states

```ts
// /tmp/cl-repos/agent-orchestrator/packages/core/src/types.ts
export type SessionStatus =
  | "spawning"
  | "working"
  | "pr_open"
  | "ci_failed"
  | "review_pending"
  | "changes_requested"
  | "approved"
  | "mergeable"
  | "merged"
  | "cleanup"
  | "needs_input"
  | "stuck"
  | "errored"
  | "killed"
  | "idle"
  | "done"
  | "terminated";
```

### Tiny codepack B: Symphony explicitly treats success as reaching the next handoff state

```text
# /tmp/cl-repos/symphony/SPEC.md
Symphony does not require first-class tracker write APIs in the orchestrator.
- Ticket mutations ... are typically handled by the coding agent ...
- The service remains a scheduler/runner and tracker reader.
- Workflow-specific success often means "reached the next handoff state"
  (for example `Human Review`) rather than tracker terminal state `Done`.
```

### Requested adjustment

Please revise the prior recommendation to:

> Use one main exploit agent by default. Do not introduce a standing multi-agent council. If a second role is used at all, make it a narrow integrity skeptic/review pass rather than a coequal researcher.

---

## 2. Keep the four terminal outcomes, but add an internal `plateau`-style trigger concept

I agree that the final run outcomes should remain:
- `promote`
- `discard`
- `invalid`
- `needs_human_decision`

But I want you to refine the design so that `plateau` is treated as an **internal trigger condition**, not as a fifth terminal decision.

### Why I want this adjustment

The prior report already introduces plateau logic in prose, but it should be elevated into the formal internal state machine so research mode is triggered by an explicit condition rather than ad hoc operator intuition.

### Tiny codepack C: AO distinguishes terminal status from internal session state

```ts
// /tmp/cl-repos/agent-orchestrator/packages/core/src/types.ts
export const TERMINAL_STATUSES: ReadonlySet<SessionStatus> = new Set([
  "killed",
  "terminated",
  "done",
  "cleanup",
  "errored",
  "merged",
]);

export function isTerminalSession(session: {
  status: SessionStatus;
  activity: ActivityState | null;
}): boolean {
  return (
    TERMINAL_STATUSES.has(session.status) ||
    (session.activity !== null && TERMINAL_ACTIVITIES.has(session.activity))
  );
}
```

### Tiny codepack D: Symphony explicitly models continuation/retry as internal semantics

```text
# /tmp/cl-repos/symphony/SPEC.md
attempt should be passed to the template because the workflow prompt may provide different
instructions for:
- first run
- continuation run after a successful prior session
- retry after error/timeout/stall
```

### Requested adjustment

Please keep the four terminal outcomes, but explicitly define:

- `plateau` = internal trigger condition for bounded research mode
- `confirm_regression_pattern` = optional internal trigger when visible-dev wins repeatedly fail locked confirmation in the same way

Do **not** turn either into new terminal end states.

---

## 3. Keep run classes, but do not hard-freeze exact minute budgets yet

I agree strongly with:
- `smoke_v1`
- `dev_v1`
- `confirm_v1`
- “never compare across run classes”

What I want softened is the specific timing recommendation like:
- 10–15 min
- 60–90 min
- 2–3 hours

Those are good candidate defaults, but they should still be treated as **pilot-calibrated values**, not settled design facts.

### Why I want this adjustment

Karpathy’s loop proves the value of a fixed budget, but not the exact budget for this harder setting. The right bounded run duration here depends on:
- chosen base model
- chosen editable surface
- hypernetwork size
- sequence length
- evaluation pack size
- whether the loop is update-heavy, eval-heavy, or confirmation-heavy

### Tiny codepack E: official `autoresearch` proves fixed-budget discipline, not universal timing

```python
# /tmp/cl-repos/autoresearch/prepare.py
MAX_SEQ_LEN = 2048
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 40 * 524288
```

```python
# /tmp/cl-repos/autoresearch/prepare.py
@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    ...
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
```

### Tiny codepack F: `nested_learning` shows runtime depends on algorithm mode and integrity metadata

```python
# /tmp/cl-repos/nested_learning/src/nested_learning/training.py
algorithm_mode = _resolve_algorithm_mode(cfg)
_validate_algorithm_mode_constraints(cfg, algorithm_mode=algorithm_mode, distributed=distributed)
...
steps = cfg.train.steps
...
online_updates = bool(cfg.train.get("online_updates", False))
online_chunk_size = int(cfg.train.get("online_chunk_size", 0) or 0)
```

```python
# /tmp/cl-repos/nested_learning/src/nested_learning/training.py
metadata = {
    "step": step,
    "checkpoint_sha256": ckpt_hash,
    "config_sha256": config_hash,
    "algorithm_mode": str(cfg.train.get("algorithm_mode", "two_pass_stopgrad_updates")),
    "online_updates": bool(cfg.train.get("online_updates", False)),
    "use_fast_state": bool(cfg.train.get("use_fast_state", False)),
}
```

### Requested adjustment

Please revise the prior recommendation to:

> keep named run classes, but make the exact durations provisional and subject to an explicit hardware/model calibration phase on the chosen 3090-era setup.

In other words:
- preserve bounded classes
- preserve comparability within class
- weaken the exact minute counts into pilot-calibrated defaults

---

## 4. Keep lexicographic gates, but include variance bands / borderline replay policy

I agree with the broad point that:
- promotion should not collapse into one weighted scalar
- integrity and anti-shim checks should be hard gates
- target success alone is not enough

What I want added is:
- tolerance bands
- explicit borderline policy
- optional replay/re-check for near-threshold runs

### Why I want this adjustment

The current evidence base points toward **multi-metric heterogeneity**, not perfectly crisp thresholds. Some metrics should be hard-gated, but others will likely require:
- a noise tolerance
- a champion-relative confidence band
- or a replay-on-borderline rule

### Tiny codepack G: `conflict_aware_editing` already aggregates multiple metric families together

```python
# /Users/georgepullen/Documents/research/conflict_aware_editing/src/conflict_aware_editing/validation.py
class ValidationConfig:
    model_name: str = "EleutherAI/gpt-j-6B"
    ...
    algorithms: tuple[str, ...] = ("unedited", "memit", "alphaedit", "conflict_memit")

def format_validation_summary(results: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for algorithm, payload in results["algorithms"].items():
        summary[algorithm] = {
            **payload["counterfact"]["aggregate"],
            **payload["rippleedits"]["aggregate"],
        }
```

### Tiny codepack H: `incremental_learning.pytorch` shows a CL metric surface is naturally multi-dimensional

```python
# /tmp/cl-repos/incremental_learning_pytorch/inclearn/lib/metrics.py
self.metrics["accuracy"].append(...)
self.metrics["accuracy_top5"].append(...)
self.metrics["accuracy_per_class"].append(...)
self.metrics["incremental_accuracy"].append(...)
self.metrics["forgetting"].append(...)
self.metrics["cord"].append(...)
```

```python
# /tmp/cl-repos/incremental_learning_pytorch/inclearn/lib/metrics.py
results = {
    "task_id": len(self.metrics["accuracy"]) - 1,
    "accuracy": self.metrics["accuracy"][-1],
    "incremental_accuracy": self.metrics["incremental_accuracy"][-1],
    "accuracy_top5": self.metrics["accuracy_top5"][-1],
    "forgetting": self.metrics["forgetting"][-1],
    "accuracy_per_class": self.metrics["accuracy_per_class"][-1],
    "cord": self.metrics["cord"][-1]
}
```

### Requested adjustment

Please preserve the report’s lexicographic-gate philosophy, but revise it to:

> hard-gate integrity and anti-shim conditions; use lexicographic or tiered gates for core scientific metrics; and define an explicit borderline replay/tolerance policy for near-threshold runs.

This is not a request to go back to weighted-soup scoring.

---

## 5. Treat the exact base-model pick and exact editable suffix as hypotheses, not fixed defaults

I agree with the prior report that:
- the method should be **base-model agnostic in design**
- the mainline bet should be a **tiny hypernetwork over a frozen base**
- the editable surface should stay narrow

What I want changed is the confidence level around:
- `Llama 3.1 8B base` as the default first pick
- `top 6 transformer blocks only` as the default editable band

Those are both **plausible starting hypotheses**, but the current evidence does not justify treating them as settled design truths yet.

### Why I want this adjustment

The included implementation evidence is strongly **surface-agnostic** and **model-agnostic**:
- hypernetwork methods expose `layer_indices` and `target_modules`
- validation harnesses keep `model_name` in config
- that suggests the right output here is a **selection matrix** and a **pilot plan**, not a hard-coded commitment

### Tiny codepack I: `doc-to-lora` makes layer selection and target modules configurable

```python
# /tmp/cl-repos/doc-to-lora/src/ctx_to_lora/modeling/hypernet.py
@dataclass
class HypernetConfig:
    ...
    extra_modules: list[str] | None
    base_hidden_size: int
    layer_indices: Iterable[int]
    feature_sizes: tuple[dict[str, int], dict[str, int]]
    aggregator_config: AggregatorConfig
```

```python
# /tmp/cl-repos/doc-to-lora/src/ctx_to_lora/modeling/hypernet.py
self.target_modules = tuple(sorted(self.lora_config.target_modules)) if self.lora_config else None
self.num_modules = len(self.target_modules) if self.target_modules else 0
self.extra_modules = self.config.extra_modules if self.config.extra_modules else None
self.layer_indices = self.config.layer_indices
self.n_layers = len(self.layer_indices)
```

```python
# /tmp/cl-repos/doc-to-lora/src/ctx_to_lora/modeling/hypernet.py
def generate_weights(self, features, attn_mask=None, position_ids=None):
    flat_loras, flat_layernorms = self.forward(features, attn_mask, position_ids)
    return self._to_lora_dict(flat_loras), self._to_layernorm_dict(flat_layernorms)
```

### Tiny codepack J: `doc-to-lora` also keeps the base model open as a config/runtime choice

```python
# /tmp/cl-repos/doc-to-lora/src/ctx_to_lora/modeling/hypernet.py
@classmethod
def from_state_dict(cls, state_dict: dict, train: bool = True, ...):
    model_name_or_path = state_dict["base_model_name_or_path"]
    base_model = get_model(
        model_name_or_path,
        train=train,
        requires_grad=False,
        peft_config=lora_config,
        ...
    )
```

### Tiny codepack K: `conflict_aware_editing` already treats model choice as config, not doctrine

```python
# /Users/georgepullen/Documents/research/conflict_aware_editing/src/conflict_aware_editing/validation.py
class ValidationConfig:
    model_name: str = "EleutherAI/gpt-j-6B"
    revision: str | None = "float16"
    torch_dtype: str = "float16"
    ...
```

### Requested adjustment

Please revise the prior recommendation to:

> recommend a **base-model selection matrix** for the first pilot rather than one fixed model, and recommend a **narrow editable suffix/band pilot study** rather than one fixed “top 6 blocks” commitment.

Concretely, I want you to compare a small candidate set such as:
- Llama 3.1 8B base
- Qwen3.5-4B-Base
- Gemma 3 PT 4B

And I want you to propose an initial editable-surface study like:
- top 4 blocks
- top 6 blocks
- sparse suffix + selected attention/output modules

Then choose the winner based on:
- 3090 fit
- bounded-run throughput
- stability under repeated serial updates
- quality of the first visible-dev results

---

## Net instruction

Please **do not overturn the original plan**.

Instead, keep its backbone and apply these five refinements:

1. one main exploit agent, with at most one narrow skeptic/integrity pass
2. keep the four terminal run outcomes, but add internal plateau-style triggers
3. keep run classes, but calibrate their exact durations empirically
4. keep lexicographic gating, but add tolerance / borderline replay policy
5. treat the exact base-model pick and exact editable suffix as pilot hypotheses, not settled defaults

Everything else in the original report should be treated as directionally correct unless a stronger correction is clearly justified.
