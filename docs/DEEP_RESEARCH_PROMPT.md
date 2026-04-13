# Deep Research Prompt: Platinum-Standard Autoresearch Loop For Continual Learning

I want a deep, skeptical research-and-design report that helps me define the best starting point for an **autoresearch-style continual-learning research harness**.

This is **not** a generic software-engineering harness question.
This is **not** a request to preserve the structure of `conflict_aware_editing`.
This is **not** a request to maximize orchestration complexity for its own sake.

The target is much narrower and more ambitious:

> Design the optimal `autoresearch`-style loop for **autonomous iteration on a conflict-aware LLM editing hypernetwork**: a system that learns how to internalize new information into an LLM while minimizing adverse effects on the model’s existing knowledge.

The end goal is an open-source repo currently named:
- `autoresearch-continual-learning`
- GitHub: `https://github.com/georgepullen/autoresearch-continual-learning`

I want you to reason from first principles, from public repos/papers, and from the included codepacks.
Treat the included packs as high-signal local inspection evidence from a Codex agent that cloned and explored those repos directly.

## Core framing

The central design tension is this:

- Andrej Karpathy’s `autoresearch` is effective because it is **extremely constrained**.
- Continual learning for LLM knowledge editing is much harder than short-budget nanoGPT optimization.
- If we naively add a “research phase” that just keeps borrowing recent papers, the loop may degenerate into endless grafting of external ideas rather than genuine architecture iteration.
- If we naively remove human oversight completely, the loop may drift, shim the evals, or keep making taste-poor decisions.

I want you to figure out what the **platinum standard** loop should look like under those conditions.

## Hard requirements

1. The loop should be **inspired by Karpathy’s autoresearch simplicity**, but it may diverge sharply where continual learning demands it.
2. The system should assume **one heavyweight training/eval run at a time** on a single consumer GPU execution box.
3. The design should avoid relying on parallel training runs.
4. The design should support **OpenAI Codex CLI** as the agent runtime.
5. The design should avoid paid services and low-ceiling hosted tooling where possible.
6. The design should assume the project is public/open-source.
7. The design should prefer **simple but strong protocol constraints** over orchestration complexity.
8. The design should include anti-cheat / anti-shim mechanisms.
9. The design should distinguish between:
   - what the agent may edit freely
   - what is immutable protocol/eval surface
   - what can change only with explicit approval
10. The design should include a minimal but deliberate form of **HITL (Human In The Loop)** if that is likely to improve research direction quality.

## Important non-goals

Do **not** answer this as:
- “what is the best general agentic software engineering platform?”
- “what is the best extension of Symphony for all programming work?”
- “what is the best continual learning framework in the abstract?”

Do **not** overly anchor on my old repo structure.
Treat `conflict_aware_editing` as:
- evidence of the kinds of metrics/guardrails I care about
- evidence of anti-shim concerns
- evidence of messy real-world research repo failure modes
- **not** the template to preserve

## Target research problem

The target research problem is specifically this:

> Build an autonomous research loop that iterates toward a **hypernetwork-based conflict-aware editing system** that predicts how to internalize new information into an LLM without causing disproportionate forgetting, corruption, or collateral damage to existing knowledge.

This is narrower than “all continual learning for LLMs”.
Please keep the design focused on that objective.

## Special angle I want you to think hard about

Karpathy’s original loop can often interpolate “what to try next” from abundant pretraining knowledge about language-model optimization.

For **continual learning optimizations on LLMs**, especially knowledge-editing / internalization / conflict-aware retention, the prior is much sparser and noisier.

So I want you to decide whether the optimal loop should include a dedicated **research-direction stage**, and if so:
- how it should work
- when it should trigger
- what inputs it should use
- how it avoids devolving into “just import ideas from papers forever”

I am especially interested in whether the right design is something like:
- one research-direction assessor agent that sees recent literature
- one that surveys broader CL implementation repos
- one that only sees the local codebase and is biased against novelty
- one meta-assessor that combines these and either:
  - makes a direction choice, or
  - escalates to minimal HITL when taste/governance is required

But do **not** assume that multi-agent assessor setup is the answer. Evaluate it critically.

## What I want from you

Produce a concrete design for:

1. **The optimal continual-learning autoresearch loop**
2. **The minimal repo structure** for that loop
3. **The editable vs immutable surfaces**
4. **The experiment/run specification format**
5. **The result summary / ledger format**
6. **The promotion decision state machine**
7. **The research-direction stage**, if any
8. **The HITL policy**, if any
9. **The anti-cheat / anti-shim enforcement model**
10. **The recommended base model + scale ceiling** for a 3090-class setup
11. **The recommended datasets and eval sets**
12. **The tooling stack** for Codex CLI-based operation
13. **A pragmatic implementation roadmap**

## Required deliverable structure

I want your final answer to have these sections:

1. **Executive Recommendation**
   - What the loop should be
   - Whether minimal HITL is required
   - Whether a dedicated research stage is required
   - Whether multi-agent assessors are worth it

2. **Why Karpathy’s Autoresearch Works**
   - distill the principles that make it effective
   - identify which principles transfer cleanly
   - identify which ones break under continual-learning objectives

3. **Ideal Continual-Learning Inner Loop**
   - the minimal recurring loop
   - what one iteration looks like
   - what “bounded run” means here
   - what the promote/discard/invalid/needs-human-decision states should be

4. **Research-Direction Policy**
   - should there be a literature-aware stage?
   - if yes, when?
   - how do we prevent endless architecture grafting?
   - what evidence threshold is required before new external ideas enter the codebase?

5. **HITL Policy**
   - where should humans still intervene?
   - what should be fully autonomous?
   - what is the minimal HITL that materially improves outcomes?

6. **Repo Design**
   - ideal repo structure from scratch
   - core files
   - immutable surfaces
   - editable surfaces
   - experiment ledgers
   - artifact layout

7. **Model / Hardware / Data Recommendations**
   - best public small-but-strong base LLMs to consider as of March 2026
   - parameter-count ceiling suitable for a single 3090-class box
   - whether the hypernetwork should be tiny / medium / large relative to base model
   - whether datasets should stay fixed for the entire program or evolve only under approved rules

8. **Evaluation Design**
   - what metrics matter most for conflict-aware internalization
   - how to test knowledge gain vs collateral damage
   - how to design visible dev evals vs protected confirmation evals
   - how to detect shims, memorization shortcuts, or protocol gaming

9. **Tooling Recommendations**
   - specifically for Codex CLI
   - CLI-first if possible
   - public/free sources for repo/paper/web research
   - avoid paid services unless truly essential

10. **Final Blueprint**
   - one concrete repo blueprint I can build
   - one concrete iteration loop
   - one concrete governance model
   - one concrete roadmap for implementation

## Search tasks you must perform

You must go beyond the included codepacks.
Please actively search for and synthesize:

1. Additional GitHub repos relevant to:
   - continual learning for LLMs
   - hypernetworks / meta-learning for editing or internalization
   - knowledge editing with bounded collateral damage
   - parameter-efficient adaptation under continual updates
   - online / streaming / sequential adaptation

2. Public paper collections relevant to:
   - continual learning for LLMs
   - lifelong LLM agents
   - meta-learning for model updates
   - hypernetworks and adapter generation
   - catastrophic forgetting metrics
   - conflict-aware editing / model editing / memory editing

3. Up-to-date base-model options as of **March 2026**:
   - use public leaderboards, Hugging Face model pages, benchmarks, recent papers, qualitative reports
   - prioritize small models with strong reasoning/knowledge performance and plausible 3090 usability

4. Public tooling suitable for a Codex CLI agent to do:
   - web research
   - repo inspection
   - paper lookup
   - structured result logging
   without requiring paid services or brittle low-rate ceilings

## Constraints you should respect in your design

- One heavyweight run at a time
- A single execution box with consumer GPU limits
- The loop should be serial by default
- The loop should make progress via **better constraints and better decisions**, not just more orchestration
- The system should stay simple enough that a single person can operate it
- The loop should be robust against eval cheating and against “success” that comes from narrowing the task illegitimately

## Guidance on what to do with the included repos

### Primary anchors
Treat these as the highest-priority anchors:
- official `karpathy/autoresearch`
- my current design fork `autoresearch-continual-learning`
- `conflict_aware_editing` as an anti-shim / eval-governance case study
- `doc-to-lora` as inspiration for hypernetwork-mediated adaptation
- `avalanche` as a CL metric/eval vocabulary source
- `nested_learning` as a warning/example of how CL systems grow more complex

### Secondary references
Use these as background references, not as the center of gravity:
- `GMvandeVen/continual-learning`
- `arthurdouillard/incremental_learning.pytorch`
- `qianlima-lab/awesome-lifelong-llm-agent`
- `ContinualAI/continual-learning-papers`
- `openai/symphony`
- `ComposioHQ/agent-orchestrator`
- `qiuyanxin/autoship`
- `langchain-ai/open-swe`
- `All-Hands-AI/OpenHands`

## Local inspection notes from the cloned repos

These are my local findings after cloning and inspecting the repos directly. Use them as strong priors, but verify anything that is time-sensitive.

### Official `autoresearch`
- The most important thing is not the orchestrator, it is the **protocol**.
- Human edits `program.md`; agent edits the training surface.
- The loop is effective because it constrains editable surface, evaluation surface, time budget, and the keep/discard rule.
- It is outer-loop HITL, inner-loop autonomy.

### My current `autoresearch-continual-learning` fork
- This repo should become a **design-from-scratch continual-learning harness**, not an adaptation that preserves the old `conflict_aware_editing` structure.
- The strongest current ideas in the fork are the guardrail docs and the insistence on `promote` / `discard` / `invalid` / `needs_human_decision`.

### `conflict_aware_editing`
- Valuable mainly for eval governance, artifact contracts, anti-shim thinking, and fixed-slice comparisons.
- Not a clean architectural template.
- Shows the importance of immutable protocol surfaces, fixed comparison baselines, and fail-fast validation.

### `doc-to-lora`
- Useful because it explores a context-conditioned hypernetwork path that generates/update adapter weights.
- Potentially inspiring for hypernetwork-mediated internalization, but it is much more system-heavy than a Karpathy-style loop.
- Important question: what is the minimal useful subset of this idea for our target problem?

### `avalanche`
- Most useful as a vocabulary source for continual-learning metrics and evaluation plugin structure.
- Especially important for forgetting / backward transfer / forward transfer / experience-level evaluation ideas.

### `nested_learning`
- Useful as evidence for modern online/continual training structure, checkpoint-integrity metadata, and evaluation over time.
- Also useful as a cautionary example of how complexity can explode.

### `GMvandeVen/continual-learning`
- Useful for the classic continual-learning lifecycle: prepare, train, conclude, evaluate, checkpoint.
- Strong baseline reference for serial continual-learning experiments.

### `incremental_learning.pytorch`
- Useful for explicit task-by-task metric logging: incremental accuracy, forgetting, old/new accuracy, etc.

### `awesome-lifelong-llm-agent` and `continual-learning-papers`
- Useful as literature and taxonomy inputs.
- I do **not** want the loop to degenerate into endless paper grafting.
- I want you to design a better way to use these resources selectively and productively.

### `Symphony`, `Agent Orchestrator`, `Autoship`, `Open SWE`, `OpenHands`
- These are relevant as outer-shell orchestration/reference systems.
- My current belief is that they should not dominate the design because the core bottleneck is the **serial continual-learning inner loop**, not broad ticket concurrency.
- Still, they may contribute useful ideas about state tracking, review handoff, observability, or workspace discipline.

## Extra critical questions I want you to answer

1. What is the simplest loop that could actually work here?
2. What parts of Karpathy’s loop should stay unchanged?
3. What new states or phases are truly necessary for continual learning?
4. Should the training/eval datasets remain fixed for the whole research program?
5. If datasets may evolve, under what exact approval rule?
6. What would a protected confirmation benchmark look like?
7. What forms of cheating/shimming are most likely for this target problem?
8. How do we design the loop so the agent reasons from the current architecture rather than reflexively importing papers?
9. Is the right unit of improvement:
   - direct model-editing heuristics,
   - a hypernetwork that predicts updates,
   - generated adapter weights,
   - memory-controller policies,
   - or a staged combination?
10. Given a 3090-class budget, what is realistically trainable and repeatedly iterable?

## Final instruction on style

Be concrete. Be skeptical. Prefer a small number of strong design commitments over sprawling option trees.

Do not just give me a broad survey.
I want a **decision-quality blueprint** for the best starting point.

---

# Codepacks

## Codepack A: Official `karpathy/autoresearch`

# Files

## File: prepare.py
````python
"""
One-time data preparation for autoresearch experiments.
Downloads data shards and trains a BPE tokenizer.
Usage:
    python prepare.py                  # full prep (download + tokenizer)
    python prepare.py --num-shards 8   # download only 8 shards (for testing)
Data and tokenizer are stored in ~/.cache/autoresearch/.
"""
import os
import sys
import time
import math
import argparse
import pickle
from multiprocessing import Pool
import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch
# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 40 * 524288  # number of tokens for val eval
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet
VAL_SHARD = MAX_SHARD  # pinned validation shard (shard_06542)
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192
# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"
# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def download_single_shard(index):
    """Download one parquet shard with retries. Returns True on success."""
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True
    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False
def download_data(num_shards, download_workers=8):
    """Download training shards + pinned validation shard."""
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)
    # Count what's already downloaded
    existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")))
    if existing == len(ids):
        print(f"Data: all {len(ids)} shards already downloaded at {DATA_DIR}")
        return
    needed = len(ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")
    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, ids)
    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(ids)} shards ready at {DATA_DIR}")
# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------
def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]
def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from training split (all shards except pinned val shard)."""
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return
def train_tokenizer():
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards (1 train + 1 val). Download more data first.")
        sys.exit(1)
    # --- Train with rustbpe ---
    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)
    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    # Save tokenizer
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)
    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")
    # --- Build token_bytes lookup for BPB evaluation ---
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")
    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")
# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------
class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)
    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)
    def get_vocab_size(self):
        return self.enc.n_vocab
    def get_bos_token_id(self):
        return self.bos_token_id
    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids
    def decode(self, ids):
        return self.enc.decode(ids)
def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)
def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1
def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1
    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)
    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)
    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - pos
                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch
# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument("--num-shards", type=int, default=10, help="Number of training shards to download (-1 = all). Val shard is always pinned.")
    parser.add_argument("--download-workers", type=int, default=8, help="Number of parallel download workers")
    args = parser.parse_args()
    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards
    print(f"Cache directory: {CACHE_DIR}")
    print()
    # Step 1: Download data
    download_data(num_shards, download_workers=args.download_workers)
    print()
    # Step 2: Train tokenizer
    train_tokenizer()
    print()
    print("Done! Ready to train.")
````

## File: program.md
````markdown
# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
````

## File: train.py
````python
"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import gc
import math
import time
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels import get_kernel
cap = torch.cuda.get_device_capability()
# varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb
# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes
    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops
    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }
    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer
    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits
# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)
@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)
    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------
# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------
t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12
tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")
def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )
config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")
tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)
model = torch.compile(model, dynamic=False)
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch
print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")
# Schedules (all based on progress = training_time / TIME_BUDGET)
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)
# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)
    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()
    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    if step > 10:
        total_training_time += dt
    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)
    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)
    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()
    step += 1
    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break
print()  # newline after \r training log
total_tokens = step * TOTAL_BATCH_SIZE
# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
````


## Codepack B: Current `autoresearch-continual-learning` design fork

# Files

## File: docs/CASE_STUDY_CONFLICT_AWARE_EDITING.md
```markdown
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
```

## File: docs/DESIGN_PRINCIPLES.md
```markdown
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
```

## File: docs/GUARDRAILS_AND_ANTI_SHIM.md
```markdown
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
```

## File: docs/LOOP_SPEC.md
```markdown
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
```

## File: program.md
```markdown
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
```

## File: README.md
```markdown
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
```


## Codepack C: `conflict_aware_editing` governance + evaluation case study

# Files

## File: conflict_aware_editing_snippets.md
````markdown
# conflict_aware_editing snippets

These excerpts show three things we want Deep Research to internalize: strict governance docs, fail-fast slice validation, and controller-style conflict-aware editing.

### /Users/georgepullen/Documents/research/conflict_aware_editing/docs/QUALITY_GATES.md lines 1-120
```
   1 # Quality Gates
   2 
   3 ## Purpose
   4 These are the non-negotiable rules for method work in this repository. If a change breaks any gate below, it is not a valid research improvement.
   5 
   6 ## Non-Negotiable Rules
   7 - Use fixed evaluation slices only. Do not swap examples after seeing outcomes.
   8 - Compare on the same model, same slice manifests, and same metrics.
   9 - Do not use hidden retrieval, external memory, prompt stuffing, or extra helper models unless the method explicitly claims them.
  10 - Do not hand-author preserve sets, target prompts, or expected answers from the evaluation outputs.
  11 - Do not patch or rewrite generated outputs after inference.
  12 - Do not claim wins from one-off examples. Use saved run artifacts only.
  13 - Do not hide behavior changes inside `third_party/` unless the patch is clearly documented and justified.
  14 
  15 ## Required Gates For Any Claimed Improvement
  16 1. Match or exceed raw `memit` on target rewrite success.
  17 2. Improve at least one interference metric such as NKL or RippleEdits accuracy.
  18 3. Save a timestamped artifact under `~/shared/artifacts/conflict_aware_editing/validation_runs/`.
  19 4. Report runtime and peak VRAM alongside quality metrics.
  20 5. Keep the method runnable on the 24 GB 3090 path unless the change is explicitly marked as out of scope.
  21 
  22 ## Forbidden Shortcuts
  23 - Tuning thresholds on the same fixed slice used for the headline result
  24 - Discarding failed runs without noting them
  25 - Manually selecting only “easy” edits
  26 - Changing evaluation prompts per algorithm
  27 - Mixing local and remote environments in one reported result
  28 
  29 ## Review Checklist
  30 - Which baseline did this beat?
  31 - Which fixed slice was used?
  32 - What artifact path proves it?
  33 - Did the method add any hidden capacity or side channel?
  34 - Is the result still reproducible on the 3090 workflow?
```

### /Users/georgepullen/Documents/research/conflict_aware_editing/docs/EVAL_PROTOCOL.md lines 1-120
```
   1 # Evaluation Protocol
   2 
   3 ## Default Benchmark Slices
   4 Use the fixed manifests in `configs/validation/`:
   5 - `counterfact_small_v1.json`
   6 - `ripple_small_v1.json`
   7 
   8 These are the default inputs for `scripts/run_validation_batch.py`. Do not edit them to improve reported results. If a new slice is needed, create a new versioned manifest such as `counterfact_small_v2.json`.
   9 
  10 ## Required Algorithms
  11 For method development, compare against:
  12 - `unedited`
  13 - `memit`
  14 - `conflict_memit`
  15 
  16 Add `alphaedit` when testing constrained-edit baselines or when a change touches projection logic.
  17 
  18 ## Required Metrics
  19 Always log:
  20 - rewrite success
  21 - paraphrase success
  22 - neighborhood and distracting-neighborhood NKL
  23 - RippleEdits axis scores
  24 - edit runtime in seconds
  25 - peak VRAM in GB
  26 
  27 Do not report only one metric family.
  28 
  29 ## Execution Rules
  30 - Run heavy evaluations on the 3090.
  31 - Use `scripts/check_workspace.py` before starting runs.
  32 - Use the shared runner:
  33 
  34 ```bash
  35 ssh 3090 'bash -lc "source ~/.config/shared-ml-env.sh && ~/shared/envs/projects/conflict_aware_editing-env/bin/python ~/workspace/conflict_aware_editing/scripts/run_validation_batch.py"'
  36 ```
  37 
  38 - For faster iteration, narrow algorithms, not slices. Example:
  39 
  40 ```bash
  41 ... run_validation_batch.py --algorithms memit conflict_memit
  42 ```
  43 
  44 ## Interpretation Rules
  45 A method is promising only if it preserves target success and improves interference metrics on the same slice. Improvements from a single case, unlogged manual reruns, or changed prompts do not count.
```

### /Users/georgepullen/Documents/research/conflict_aware_editing/docs/ARTIFACT_CONTRACT.md lines 1-120
```
   1 # Artifact Contract
   2 
   3 ## Output Location
   4 All evaluation outputs belong under:
   5 
   6 `~/shared/artifacts/conflict_aware_editing/`
   7 
   8 Validation JSON files go in:
   9 
  10 `~/shared/artifacts/conflict_aware_editing/validation_runs/`
  11 
  12 ## Required Contents
  13 Every saved validation artifact must include:
  14 - model name, revision, and dtype
  15 - slice manifest paths or slice identifiers
  16 - algorithm list
  17 - per-case and aggregate metrics
  18 - runtime and peak VRAM
  19 - controller metadata when a controller is used
  20 - UTC timestamp in the filename
  21 
  22 The current naming pattern is:
  23 
  24 `validation_<YYYYMMDDTHHMMSSZ>.json`
  25 
  26 ## Reporting Rules
  27 - Never overwrite a prior artifact.
  28 - Reference artifact paths directly in summaries and PR notes.
  29 - If comparing methods, use artifacts from the same slice version.
  30 - Keep one result per run; do not merge numbers by hand across files.
  31 
  32 ## What Makes an Artifact Valid
  33 - It was produced by `scripts/run_validation_batch.py` or a clearly documented successor.
  34 - The config section reflects the actual run inputs.
  35 - Metrics were computed in the same run that produced the edit.
  36 - The artifact can be traced to a specific command and environment.
  37 
  38 ## Failure Handling
  39 If a run crashes, do not present partial console output as a result. Note the failure separately and rerun after fixing the issue. Successful claims must always point to a completed artifact file.
```

### /Users/georgepullen/Documents/research/conflict_aware_editing/src/conflict_aware_editing/validation.py lines 99-163
```
  99 def _load_slice_manifest(slice_path: Path) -> dict[str, Any]:
 100     with slice_path.open() as handle:
 101         return json.load(handle)
 102 
 103 
 104 def load_counterfact_records(
 105     data_dir: Path,
 106     size: int,
 107     start_index: int,
 108     slice_path: Path | None = None,
 109 ) -> list[dict[str, Any]]:
 110     ensure_counterfact_assets(data_dir)
 111     with (data_dir / "multi_counterfact.json").open() as handle:
 112         data = json.load(handle)
 113     if slice_path is None:
 114         return data[start_index : start_index + size]
 115 
 116     manifest = _load_slice_manifest(slice_path)
 117     requested_case_ids = list(manifest["case_ids"])
 118     case_ids = set(requested_case_ids)
 119     records = [record for record in data if record["case_id"] in case_ids]
 120     loaded_case_ids = {record["case_id"] for record in records}
 121     missing_case_ids = [case_id for case_id in requested_case_ids if case_id not in loaded_case_ids]
 122     if missing_case_ids:
 123         raise ValueError(
 124             f"CounterFact manifest {slice_path} references missing case IDs: {missing_case_ids}"
 125         )
 126     records.sort(key=lambda record: manifest["case_ids"].index(record["case_id"]))
 127     return records
 128 
 129 
 130 def load_ripple_records(
 131     limit_per_split: int,
 132     slice_path: Path | None = None,
 133 ) -> list[dict[str, Any]]:
 134     records: list[dict[str, Any]] = []
 135     if slice_path is not None:
 136         manifest = _load_slice_manifest(slice_path)
 137         for spec in manifest["examples"]:
 138             benchmark = load_ripple_benchmark(spec["split"])
 139             if spec["index"] >= len(benchmark):
 140                 raise ValueError(
 141                     f"Ripple manifest {slice_path} references {spec['split']}[{spec['index']}] "
 142                     f"but split size is {len(benchmark)}"
 143                 )
 144             example = dict(benchmark[spec["index"]])
 145             example["_slice_split"] = spec["split"]
 146             example["_slice_index"] = spec["index"]
 147             records.append(example)
 148         return records
 149 
 150     if limit_per_split <= 0:
 151         return records
 152     for split in ["popular", "random", "recent"]:
 153         selected = 0
 154         for example in load_ripple_benchmark(split):
 155             try:
 156                 build_ripple_rewrite_request(example, case_id=selected)
 157             except Exception:
 158                 continue
 159             records.append(example)
 160             selected += 1
 161             if selected >= limit_per_split:
 162                 break
 163     return records
```

### /Users/georgepullen/Documents/research/conflict_aware_editing/src/conflict_aware_editing/validation.py lines 612-846
```
 612 def run_validation(config: ValidationConfig) -> dict[str, Any]:
 613     artifact_root = config.artifact_root.expanduser()
 614     data_dir = artifact_root / "data"
 615     run_dir = artifact_root / "validation_runs"
 616     run_dir.mkdir(parents=True, exist_ok=True)
 617 
 618     counterfact_records = load_counterfact_records(
 619         data_dir=data_dir,
 620         size=config.counterfact_size,
 621         start_index=config.counterfact_start_index,
 622         slice_path=config.counterfact_slice_path,
 623     )
 624     ripple_records = load_ripple_records(
 625         config.ripple_examples_per_split,
 626         slice_path=config.ripple_slice_path,
 627     )
 628 
 629     model, tokenizer = load_model_and_tokenizer(config)
 630     projector: torch.Tensor | None = None
 631     scope_projection_bases: Any = None
 632     controller_config = ConflictAwareMemitConfig(
 633         candidate_scales=config.conflict_candidate_scales,
 634         max_preserve_nkl=config.conflict_max_preserve_nkl,
 635         max_preserve_prompts=config.conflict_max_preserve_prompts,
 636     )
 637     scope_controller_config = ScopeConstrainedMemitConfig(
 638         max_preserve_candidates=config.scope_max_preserve_candidates,
 639         max_update_candidates=config.scope_max_update_candidates,
 640         top_preserve=config.scope_top_preserve,
 641         top_update=config.scope_top_update,
 642         candidate_raw_blends=config.scope_candidate_raw_blends,
 643     )
 644 
 645     algorithms = list(config.algorithms)
 646     results: dict[str, Any] = {
 647         "config": {
 648             **asdict(config),
 649             "artifact_root": str(artifact_root),
 650         },
 651         "algorithms": {},
 652     }
 653 
 654     for algorithm in algorithms:
 655         print(f"Running validation for {algorithm}")
 656         algorithm_result: dict[str, Any] = {
 657             "counterfact": {"cases": []},
 658             "rippleedits": {"examples": []},
 659         }
 660 
 661         for case_index, record in enumerate(counterfact_records):
 662             _empty_peak_memory()
 663             before = evaluate_counterfact_record(model, tokenizer, record)
 664             edit_runtime = 0.0
 665             original_weights: dict[str, torch.Tensor] = {}
 666             controller_metadata: dict[str, Any] | None = None
 667 
 668             if algorithm == "memit":
 669                 edit_runtime, original_weights = _run_memit_case(
 670                     model,
 671                     tokenizer,
 672                     {"case_id": record["case_id"], **record["requested_rewrite"]},
 673                     config.memit_hparams_name,
 674                 )
 675             elif algorithm == "alphaedit":
 676                 edit_runtime, original_weights, projector = _run_alphaedit_case(
 677                     model,
 678                     tokenizer,
 679                     {"case_id": record["case_id"], **record["requested_rewrite"]},
 680                     config.alphaedit_hparams_name,
 681                     projector,
 682                 )
 683             elif algorithm == "conflict_memit":
 684                 scope = build_counterfact_control_scope(
 685                     record,
 686                     max_preserve_prompts=config.conflict_max_preserve_prompts,
 687                 )
 688                 edit_runtime, original_weights, controller_metadata = _run_conflict_memit_case(
 689                     model,
 690                     tokenizer,
 691                     {"case_id": record["case_id"], **record["requested_rewrite"]},
 692                     config.memit_hparams_name,
 693                     target_prompts=scope["target_prompts"],
 694                     target_new=scope["target_new"],
 695                     preserve_prompts=scope["preserve_prompts"],
 696                     controller_config=controller_config,
 697                 )
 698             elif algorithm == "scope_memit":
 699                 scope = build_counterfact_scope_bundle(
 700                     record,
 701                     max_preserve_candidates=config.scope_max_preserve_candidates,
 702                     max_update_candidates=config.scope_max_update_candidates,
 703                 )
 704                 (
 705                     edit_runtime,
 706                     original_weights,
 707                     controller_metadata,
 708                     scope_projection_bases,
 709                 ) = _run_scope_memit_case(
 710                     model,
 711                     tokenizer,
 712                     {"case_id": record["case_id"], **record["requested_rewrite"]},
 713                     config.memit_hparams_name,
 714                     config.alphaedit_hparams_name,
 715                     target_prompt=scope["target_prompt"],
 716                     target_new=scope["target_new"],
 717                     preserve_candidates=scope["preserve_candidates"],
 718                     update_candidates=scope["update_candidates"],
 719                     controller_config=scope_controller_config,
 720                     projection_bases=scope_projection_bases,
 721                 )
 722 
 723             after = evaluate_counterfact_record(model, tokenizer, record)
 724             peak_memory_gb = _peak_memory_gb()
 725             if original_weights:
 726                 restore_named_weights(model, original_weights)
 727 
 728             case_result = {
 729                 "case_id": record["case_id"],
 730                 "requested_rewrite": record["requested_rewrite"],
 731                 "before": _json_safe_counterfact_metrics(before),
 732                 "after": _json_safe_counterfact_metrics(after),
 733                 "summary": summarise_counterfact_case(before, after),
 734                 "edit_runtime_seconds": edit_runtime,
 735                 "peak_memory_gb": peak_memory_gb,
 736             }
 737             if controller_metadata is not None:
 738                 case_result["controller"] = controller_metadata
 739             algorithm_result["counterfact"]["cases"].append(case_result)
 740             print(f"  CounterFact case {case_index + 1}/{len(counterfact_records)} complete")
 741 
 742         for example_index, example in enumerate(ripple_records):
 743             _empty_peak_memory()
 744             edit_runtime = 0.0
 745             original_weights = {}
 746             controller_metadata = None
 747 
 748             if algorithm == "memit":
 749                 request = build_ripple_rewrite_request(example, case_id=example_index)
 750                 edit_runtime, original_weights = _run_memit_case(
 751                     model,
 752                     tokenizer,
 753                     request,
 754                     config.memit_hparams_name,
 755                 )
 756             elif algorithm == "alphaedit":
 757                 request = build_ripple_rewrite_request(example, case_id=example_index)
 758                 edit_runtime, original_weights, projector = _run_alphaedit_case(
 759                     model,
 760                     tokenizer,
 761                     request,
 762                     config.alphaedit_hparams_name,
 763                     projector,
 764                 )
 765             elif algorithm == "conflict_memit":
 766                 request = build_ripple_rewrite_request(example, case_id=example_index)
 767                 scope = build_ripple_control_scope(
 768                     example,
 769                     case_id=example_index,
 770                     max_preserve_prompts=config.conflict_max_preserve_prompts,
 771                 )
 772                 edit_runtime, original_weights, controller_metadata = _run_conflict_memit_case(
 773                     model,
 774                     tokenizer,
 775                     request,
 776                     config.memit_hparams_name,
 777                     target_prompts=scope["target_prompts"],
 778                     target_new=scope["target_new"],
 779                     preserve_prompts=scope["preserve_prompts"],
 780                     controller_config=controller_config,
 781                 )
 782             elif algorithm == "scope_memit":
 783                 request = build_ripple_rewrite_request(example, case_id=example_index)
 784                 scope = build_ripple_scope_bundle(
 785                     example,
 786                     case_id=example_index,
 787                     max_preserve_candidates=config.scope_max_preserve_candidates,
 788                     max_update_candidates=config.scope_max_update_candidates,
 789                 )
 790                 (
 791                     edit_runtime,
 792                     original_weights,
 793                     controller_metadata,
 794                     scope_projection_bases,
 795                 ) = _run_scope_memit_case(
 796                     model,
 797                     tokenizer,
 798                     request,
 799                     config.memit_hparams_name,
 800                     config.alphaedit_hparams_name,
 801                     target_prompt=scope["target_prompt"],
 802                     target_new=scope["target_new"],
 803                     preserve_candidates=scope["preserve_candidates"],
 804                     update_candidates=scope["update_candidates"],
 805                     controller_config=scope_controller_config,
 806                     projection_bases=scope_projection_bases,
 807                 )
 808 
 809             example_result = evaluate_ripple_example(
 810                 model,
 811                 tokenizer,
 812                 example,
 813                 max_new_tokens=config.max_new_tokens,
 814             )
 815             example_result["example_source"] = {
 816                 "split": example.get("_slice_split"),
 817                 "index": example.get("_slice_index"),
 818             }
 819             example_result["edit_runtime_seconds"] = edit_runtime
 820             example_result["peak_memory_gb"] = _peak_memory_gb()
 821             if original_weights:
 822                 restore_named_weights(model, original_weights)
 823             if controller_metadata is not None:
 824                 example_result["controller"] = controller_metadata
 825             algorithm_result["rippleedits"]["examples"].append(example_result)
 826             print(f"  Ripple example {example_index + 1}/{len(ripple_records)} complete")
 827 
 828         algorithm_result["counterfact"]["aggregate"] = _aggregate_counterfact(
 829             algorithm_result["counterfact"]["cases"]
 830         )
 831         algorithm_result["rippleedits"]["aggregate"] = _aggregate_ripple(
 832             algorithm_result["rippleedits"]["examples"]
 833         )
 834         results["algorithms"][algorithm] = algorithm_result
 835 
 836     return results
 837 
 838 
 839 def format_validation_summary(results: Mapping[str, Any]) -> dict[str, Any]:
 840     summary: dict[str, Any] = {}
 841     for algorithm, payload in results["algorithms"].items():
 842         summary[algorithm] = {
 843             **payload["counterfact"]["aggregate"],
 844             **payload["rippleedits"]["aggregate"],
 845         }
 846     return summary
```

### /Users/georgepullen/Documents/research/conflict_aware_editing/src/conflict_aware_editing/conflict_aware_memit.py lines 13-240
```
  13 @dataclass(frozen=True)
  14 class ConflictAwareMemitConfig:
  15     candidate_scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
  16     max_preserve_nkl: float = 1e-5
  17     min_target_success: float = 1.0
  18     max_preserve_prompts: int = 8
  19 
  20 
  21 @dataclass
  22 class ConflictAwareMemitState:
  23     original_weights: dict[str, torch.Tensor]
  24     metadata: dict[str, Any]
  25 
  26 
  27 def _model_device(model: Any) -> torch.device:
  28     return next(model.parameters()).device
  29 
  30 
  31 def _unique_nonempty(prompts: Sequence[str]) -> list[str]:
  32     seen: set[str] = set()
  33     ordered: list[str] = []
  34     for prompt in prompts:
  35         normalized = prompt.strip()
  36         if not normalized or normalized in seen:
  37             continue
  38         seen.add(normalized)
  39         ordered.append(normalized)
  40     return ordered
  41 
  42 
  43 def _tokenize_target(tokenizer: Any, target: str) -> list[int]:
  44     return tokenizer(f" {target.strip()}", add_special_tokens=False)["input_ids"]
  45 
  46 
  47 def _next_token_log_probs(model: Any, tokenizer: Any, prompts: Sequence[str]) -> list[torch.Tensor]:
  48     if not prompts:
  49         return []
  50 
  51     device = _model_device(model)
  52     batch = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
  53     prompt_lens = [len(ids) for ids in tokenizer(prompts)["input_ids"]]
  54 
  55     with torch.no_grad():
  56         logits = model(**batch).logits
  57 
  58     return [
  59         torch.nn.functional.log_softmax(logits[index, prompt_len - 1, :], dim=0).detach().cpu()
  60         for index, prompt_len in enumerate(prompt_lens)
  61     ]
  62 
  63 
  64 def _target_success_stats(
  65     model: Any,
  66     tokenizer: Any,
  67     prompts: Sequence[str],
  68     target: str,
  69 ) -> dict[str, float]:
  70     if not prompts:
  71         return {"success_rate": 0.0, "avg_nll": float("inf")}
  72 
  73     device = _model_device(model)
  74     target_tokens = _tokenize_target(tokenizer, target)
  75     formatted_prompts = [f"{prompt} {target.strip()}" for prompt in prompts]
  76     batch = tokenizer(formatted_prompts, padding=True, return_tensors="pt").to(device)
  77     prefix_lens = [len(ids) for ids in tokenizer(prompts)["input_ids"]]
  78 
  79     with torch.no_grad():
  80         logits = model(**batch).logits
  81 
  82     successes = 0
  83     total_nll = 0.0
  84     for prompt_index in range(len(prompts)):
  85         exact_match = True
  86         token_nll = 0.0
  87         for token_offset, token_id in enumerate(target_tokens):
  88             next_token_logits = logits[prompt_index, prefix_lens[prompt_index] + token_offset - 1, :]
  89             log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=0)
  90             token_nll += -log_probs[token_id].item()
  91             if next_token_logits.argmax().item() != token_id:
  92                 exact_match = False
  93         successes += int(exact_match)
  94         total_nll += token_nll / max(len(target_tokens), 1)
  95 
  96     return {
  97         "success_rate": successes / len(prompts),
  98         "avg_nll": total_nll / len(prompts),
  99     }
 100 
 101 
 102 def _preserve_nkl_stats(
 103     before_log_probs: Sequence[torch.Tensor],
 104     after_log_probs: Sequence[torch.Tensor],
 105 ) -> dict[str, float]:
 106     if not before_log_probs:
 107         return {"mean_nkl": 0.0, "max_nkl": 0.0}
 108 
 109     kls = [
 110         float(kl_div(before_log_prob, after_log_prob, log_target=True, reduction="batchmean").item())
 111         for before_log_prob, after_log_prob in zip(before_log_probs, after_log_probs)
 112     ]
 113     return {
 114         "mean_nkl": sum(kls) / len(kls),
 115         "max_nkl": max(kls),
 116     }
 117 
 118 
 119 def _apply_named_weights(model: Any, weights: Mapping[str, torch.Tensor]) -> None:
 120     named_parameters = dict(model.named_parameters())
 121     with torch.no_grad():
 122         for name, value in weights.items():
 123             named_parameters[name][...] = value.to(named_parameters[name].device)
 124 
 125 
 126 def _apply_scaled_deltas(
 127     model: Any,
 128     original_weights: Mapping[str, torch.Tensor],
 129     deltas: Mapping[str, torch.Tensor],
 130     scale: float,
 131 ) -> None:
 132     named_parameters = dict(model.named_parameters())
 133     with torch.no_grad():
 134         for name, original_value in original_weights.items():
 135             updated_value = original_value + scale * deltas[name]
 136             named_parameters[name][...] = updated_value.to(named_parameters[name].device)
 137 
 138 
 139 def build_counterfact_control_scope(
 140     record: Mapping[str, Any],
 141     *,
 142     max_preserve_prompts: int,
 143 ) -> dict[str, Any]:
 144     rewrite = record["requested_rewrite"]
 145     rewrite_prompt = rewrite["prompt"].format(rewrite["subject"])
 146     target_new = rewrite["target_new"]["str"]
 147     neighborhood_prompts = _unique_nonempty(record["neighborhood_prompts"])[:max_preserve_prompts]
 148     distracting_prompts = [
 149         f"{rewrite_prompt} {target_new}. {prompt}" for prompt in neighborhood_prompts
 150     ]
 151     preserve_prompts = _unique_nonempty([*neighborhood_prompts, *distracting_prompts])[
 152         : 2 * max_preserve_prompts
 153     ]
 154     return {
 155         "target_prompts": [rewrite_prompt],
 156         "target_new": target_new,
 157         "preserve_prompts": preserve_prompts,
 158     }
 159 
 160 
 161 def build_ripple_control_scope(
 162     example: Mapping[str, Any],
 163     *,
 164     case_id: int,
 165     max_preserve_prompts: int,
 166 ) -> dict[str, Any]:
 167     request = build_ripple_rewrite_request(example, case_id=case_id)
 168     fact_prompt = request["prompt"].format(request["subject"])
 169 
 170     preserve_prompts: list[str] = []
 171     for axis, test_phase in [
 172         ("Relation_Specificity", False),
 173         ("Logical_Generalization", False),
 174         ("Subject_Aliasing", False),
 175         ("Compositionality_I", False),
 176         ("Compositionality_II", False),
 177         ("Forgetfulness", True),
 178     ]:
 179         for testcase in example.get(axis, []):
 180             preserve_prompts.extend(
 181                 query["prompt"] for query in testcase.get("condition_queries", []) if query.get("prompt")
 182             )
 183             if test_phase:
 184                 preserve_prompts.extend(
 185                     query["prompt"] for query in testcase.get("test_queries", []) if query.get("prompt")
 186                 )
 187 
 188     preserve_prompts = [
 189         prompt for prompt in _unique_nonempty(preserve_prompts) if prompt != fact_prompt
 190     ][:max_preserve_prompts]
 191 
 192     return {
 193         "target_prompts": [fact_prompt],
 194         "target_new": request["target_new"]["str"],
 195         "preserve_prompts": preserve_prompts,
 196     }
 197 
 198 
 199 def apply_conflict_aware_memit(
 200     model: Any,
 201     tokenizer: Any,
 202     request: Mapping[str, Any],
 203     hparams_name: str,
 204     *,
 205     target_prompts: Sequence[str],
 206     target_new: str,
 207     preserve_prompts: Sequence[str],
 208     config: ConflictAwareMemitConfig,
 209 ) -> tuple[Any, ConflictAwareMemitState]:
 210     before_preserve = _next_token_log_probs(model, tokenizer, preserve_prompts)
 211     _, original_weights = apply_memit_edit(
 212         model=model,
 213         tokenizer=tokenizer,
 214         requests=[request],
 215         hparams_name=hparams_name,
 216         copy=False,
 217         return_orig_weights=True,
 218     )
 219     original_weights = {
 220         name: value.detach().cpu().clone()
 221         for name, value in original_weights.items()
 222     }
 223 
 224     named_parameters = dict(model.named_parameters())
 225     deltas = {
 226         name: (
 227             named_parameters[name].detach().cpu().to(dtype=original_value.dtype) - original_value
 228         )
 229         for name, original_value in original_weights.items()
 230     }
 231     _apply_named_weights(model, original_weights)
 232 
 233     candidates: list[dict[str, Any]] = []
 234     for scale in config.candidate_scales:
 235         _apply_scaled_deltas(model, original_weights, deltas, scale)
 236         target_stats = _target_success_stats(model, tokenizer, target_prompts, target_new)
 237         preserve_stats = _preserve_nkl_stats(
 238             before_preserve,
 239             _next_token_log_probs(model, tokenizer, preserve_prompts),
 240         )
```
````


## Codepack D: `SakanaAI/doc-to-lora`

# Files

## File: doc_to_lora_snippets.md
````markdown
# doc-to-lora snippets

These excerpts matter because they show a hypernetwork-conditioned adapter-generation path and a richer eval surface. We want Deep Research to reason about which parts are useful inspiration versus overcomplication.

### /tmp/cl-repos/doc-to-lora/src/ctx_to_lora/modeling/hypernet.py lines 61-115
```
  61 @dataclass
  62 class HypernetConfig:
  63     latent_size: int
  64     use_light_weight_lora: bool
  65     light_weight_latent_size: int
  66     per_rank_gen: bool
  67     use_per_rank_bias: bool
  68     use_bias: bool
  69     per_layer_processing: bool
  70     use_token_mixing: bool
  71     num_pre_head_layers: int
  72     dropout_rate: float
  73 
  74     lora_config: LoraConfig
  75     extra_modules: list[str] | None
  76     base_hidden_size: int
  77 
  78     layer_indices: Iterable[int]
  79     feature_sizes: tuple[dict[str, int], dict[str, int]]
  80     aggregator_config: AggregatorConfig
  81 
  82 
  83 def get_hypernet_config(
  84     model: PreTrainedModel,
  85     ctx_encoder_model_config: PretrainedConfig,
  86     hypernet_args: HypernetArguments,
  87     aggregator_args: AggregatorArguments,
  88     ctx_encoder_args: CtxEncoderArguments,
  89 ):
  90     num_modules = 0
  91     lora_config = getattr(model, "peft_config", None)
  92     if lora_config is not None:
  93         lora_config = lora_config["default"]
  94         num_modules += len(lora_config.target_modules)
  95     num_extra_modules = len(hypernet_args.extra_modules or [])
  96     indices = torch.arange(get_num_layers(model), device=model.device)
  97     return HypernetConfig(
  98         **vars(hypernet_args),
  99         base_hidden_size=model.config.hidden_size,
 100         lora_config=lora_config,
 101         layer_indices=indices,
 102         feature_sizes=get_peft_in_out_features(model, peft_config=lora_config),
 103         aggregator_config=get_aggregator_config(
 104             model,
 105             ctx_encoder_model_config,
 106             ctx_encoder_args.ctx_encoder_type == CTX_ENCODER_TYPE.PER_LAYER_ACTIVATIONS,
 107             hypernet_args.latent_size,
 108             num_modules,
 109             num_extra_modules,
 110             lora_config.r,
 111             hypernet_args.per_rank_gen,
 112             aggregator_args,
 113         ),
 114     )
 115 
```

### /tmp/cl-repos/doc-to-lora/src/ctx_to_lora/modeling/hypernet.py lines 215-360
```
 215 class HyperLoRA(nn.Module):
 216     def __init__(self, config: HypernetConfig):
 217         super().__init__()
 218 
 219         # aggregator output [bs, n_layers, n_modules, feature_dim]
 220         # by mixing the pooled features with layer embs and module embs (for pooling)
 221         # or via a perceiver w/ bottleneck size = n_modules * n_layers
 222         self.config = config
 223         logger.debug(f"HyperLoRA config: {self.config}")
 224         self.iterative_mode = False
 225         self._init_model()
 226 
 227     def _init_model(self):
 228         self.agg_config = self.config.aggregator_config
 229         self.aggregator = AGGREGATOR_CLS[self.agg_config.aggregator_type](
 230             **vars(self.agg_config)
 231         )
 232 
 233         self.lora_config = self.config.lora_config
 234         self.r = self.lora_config.r
 235 
 236         self.target_modules = (
 237             tuple(sorted(self.lora_config.target_modules)) if self.lora_config else None
 238         )
 239         self.num_modules = len(self.target_modules) if self.target_modules else 0
 240         self.extra_modules = (
 241             self.config.extra_modules if self.config.extra_modules else None
 242         )
 243         self.num_extra_modules = len(self.extra_modules) if self.extra_modules else 0
 244         self.layer_indices = self.config.layer_indices
 245         self.n_layers = len(self.layer_indices)
 246 
 247         self.d_in, self.d_out = self.config.feature_sizes
 248         self.d_latent = self.config.latent_size
 249 
 250         if self.target_modules:
 251             if self.config.per_layer_processing:
 252                 layers = [
 253                     ResMLPBlockPerLayer(
 254                         self.n_layers,
 255                         self.d_latent,
 256                         self.d_latent * 4,
 257                         self.d_latent,
 258                     )
 259                     for _ in range(self.config.num_pre_head_layers)
 260                 ]
 261             else:
 262                 layers = [
 263                     ResMLPBlock(
 264                         input_size=self.config.latent_size,
 265                         hidden_size=self.config.latent_size * 4,
 266                         output_size=self.config.latent_size,
 267                         dropout_rate=getattr(self.config, "dropout_rate", 0),
 268                     )
 269                     for _ in range(self.config.num_pre_head_layers)
 270                 ]
 271 
 272             self.layers = nn.Sequential(*layers)
 273 
 274             self.d_lora = max(self.d_in[m] + self.d_out[m] for m in self.target_modules)
 275 
 276             self.bias_A = nn.ParameterDict(
 277                 {
 278                     m: nn.Parameter(
 279                         torch.normal(
 280                             0,
 281                             0.2 / (self.d_in[m] * self.r) ** 0.5,
 282                             (self.n_layers, self.r, self.d_in[m]),
 283                         )
 284                     )
 285                     for m in self.target_modules
 286                 }
 287             )
 288             self.bias_B = nn.ParameterDict(
 289                 {
 290                     m: nn.Parameter(torch.zeros((self.n_layers, self.r, self.d_out[m])))
 291                     for m in self.target_modules
 292                 }
 293             )
 294 
 295             self.scaler_A = nn.ParameterDict(
 296                 {
 297                     m: nn.Parameter(torch.ones((1, self.n_layers, self.r, 1)))
 298                     for m in self.target_modules
 299                 }
 300             )
 301             self.scaler_B = nn.ParameterDict(
 302                 {
 303                     m: nn.Parameter(torch.zeros((1, self.n_layers, self.r, 1)))
 304                     for m in self.target_modules
 305                 }
 306             )
 307 
 308             n_modules = len(self.target_modules)
 309             # have to do this otherwise doesnt work with adamw_torch_fused
 310             # has something to do with the bias shape (n_modules r d_lora)
 311             # when n_modules == 1, adamw_torch_fused complains about device/layout
 312             # but when n_modules > 1, it works fine
 313             if n_modules == 1:
 314                 self.head = Mix(
 315                     "bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora",
 316                     weight_shape="n_layers d_latent d_lora",
 317                     bias_shape=None,  # no bias
 318                     n_layers=len(self.layer_indices),
 319                     d_latent=self.config.latent_size,
 320                     r=self.config.lora_config.r,
 321                     d_lora=self.d_lora,
 322                 )
 323             else:
 324                 self.head = Mix(
 325                     "bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora",
 326                     weight_shape="n_layers n_modules d_latent d_lora",
 327                     bias_shape=None,  # no bias
 328                     n_layers=len(self.layer_indices),
 329                     n_modules=n_modules,
 330                     d_latent=self.config.latent_size,
 331                     r=self.config.lora_config.r,
 332                     d_lora=self.d_lora,
 333                 )
 334 
 335     def get_head_bias(self):
 336         bias_dict = dict()
 337         for module in self.target_modules:
 338             bias_A = self.bias_A[module]
 339             bias_B = self.bias_B[module]
 340 
 341             bias_dict[module] = dict(A=bias_A, B=bias_B)
 342         return bias_dict
 343 
 344     def _to_lora_dict(
 345         self, flat_loras: Float[Tensor, "bs n_layers n_modules r max_io_dim"]
 346     ) -> dict[str, dict[str, Float[Tensor, "bs n_layers r _"]]]:
 347         if self.target_modules is None:
 348             return None
 349         # list of [bs, n_layers, r, in_d_outim]
 350         # and in_d_outim might vary across modules
 351         loras = unpack(
 352             flat_loras,
 353             [[] for _ in range(len(self.target_modules))],
 354             "bs n_layers * r max_io_dim",
 355         )
 356 
 357         # dict of {module:
 358         #   {A: [bs, n_layers, r, d_inim],
 359         #    B: [bs, n_layers, r, d_outim]}}
 360         lora_dict = dict()
```

### /tmp/cl-repos/doc-to-lora/src/ctx_to_lora/modeling/hypernet.py lines 430-517
```
 430     def generate_weights(
 431         self,
 432         features: Float[Tensor, "bs seq_len feature_dim"],
 433         attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
 434         position_ids: Integer[Tensor, "bs seq_len"] | None = None,
 435     ):
 436         flat_loras, flat_layernorms = self.forward(features, attn_mask, position_ids)
 437         return self._to_lora_dict(flat_loras), self._to_layernorm_dict(flat_layernorms)
 438 
 439 
 440 class ModulatedPretrainedModel(nn.Module):
 441     def __init__(
 442         self,
 443         base_model: PeftModel,
 444         hypernet_config: HypernetConfig,
 445         ctx_encoder_args: CtxEncoderArguments,
 446         use_base_input_as_ctx: bool = False,
 447         # need non-packed inputs for generation
 448         use_sequence_packing: bool = True,
 449         user_defined_scaling: float = 1,
 450         inp_compressor=None,
 451     ):
 452         assert not use_base_input_as_ctx
 453         super().__init__()
 454         self.device = base_model.device
 455         self.peft_config = base_model.peft_config["default"]
 456         self.hypernet_config = hypernet_config
 457         self.ctx_encoder_args = ctx_encoder_args
 458         self.use_base_input_as_ctx = use_base_input_as_ctx
 459         self.use_sequence_packing = use_sequence_packing
 460         self.user_defined_scaling = user_defined_scaling
 461         self.inp_compressor = inp_compressor
 462         self.model_accepts_loss_kwargs = True
 463         self.generated_loras = None
 464 
 465         self.register_module("base_model", base_model)
 466         self._init_model()
 467         self._bias_hyper_init()
 468 
 469     @classmethod
 470     def from_state_dict(
 471         cls,
 472         state_dict: dict,
 473         train: bool = True,
 474         base_model_kwargs: dict = None,
 475         use_flash_attn: bool = True,
 476         **kwargs: Any,
 477     ):
 478         lora_config = state_dict["hypernet_config"].lora_config
 479         print(f"lora_config: {lora_config}")
 480         model_name_or_path = state_dict["base_model_name_or_path"]
 481         base_model = get_model(
 482             model_name_or_path,
 483             train=train,
 484             requires_grad=False,
 485             peft_config=lora_config,
 486             model_kwargs=base_model_kwargs,
 487             use_flash_attn=use_flash_attn,
 488         )
 489         hypernet_config = state_dict["hypernet_config"]
 490         if getattr(hypernet_config, "num_pre_head_layers", None) is None:
 491             hypernet_config.num_pre_head_layers = 4
 492         if getattr(hypernet_config, "use_per_rank_bias", None) is None:
 493             hypernet_config.use_per_rank_bias = False
 494         if getattr(hypernet_config, "use_bias", None) is None:
 495             hypernet_config.use_bias = True
 496         ctx_encoder_args = state_dict["ctx_encoder_args"]
 497         model = cls(base_model, hypernet_config, ctx_encoder_args, **kwargs)
 498         model.load_state_dict(state_dict)
 499         return model
 500 
 501     def patch_lora_forward(self):
 502         layers = get_layers(self.base_model)
 503 
 504         lora_forward_fn = (
 505             lora_forward_packed if self.use_sequence_packing else lora_forward
 506         )
 507         for layer_idx in self.hypernet.layer_indices:
 508             for module_info in get_peft_modules(layers[layer_idx], self.peft_config):
 509                 name = module_info["name"]
 510                 module = module_info["module"]
 511                 if getattr(module, "patched_forward", False):
 512                     continue
 513                 logger.debug(f"Applying LoRA forward to {name}")
 514                 module.forward_orig = module.forward
 515                 module.patched_forward = True
 516                 module.forward = partial(
 517                     lora_forward_fn,
```

### /tmp/cl-repos/doc-to-lora/src/ctx_to_lora/metrics.py lines 1-163
```
   1 from collections import defaultdict
   2 from collections.abc import Callable
   3 
   4 import numpy as np
   5 import torch
   6 from rouge_score import rouge_scorer
   7 from transformers import EvalPrediction
   8 
   9 LENGTH_BINS = [
  10     # finegrain bins
  11     (0, 2**7 - 1),
  12     (2**7, 2**8 - 1),
  13     (2**8, 2**9 - 1),
  14     # coarse bins
  15     (0, 2**9 - 1),
  16     (2**9, 2**10 - 1),
  17     (2**10, 2**11 - 1),
  18     (2**11, 2**12 - 1),
  19     (2**12, 2**13 - 1),
  20     (0, 2**13 - 1),
  21     (2**13, 2**14 - 1),
  22     (2**14, 2**15 - 1),
  23     (2**15, float("inf")),
  24 ]
  25 
  26 
  27 def get_length_bin(length: int):
  28     """Get the length bin for a given length."""
  29     for i, (start, end) in enumerate(LENGTH_BINS):
  30         if start <= length < end:
  31             return (start, end)
  32 
  33 
  34 def compute_rouge(pred_texts, label_texts):
  35     out = defaultdict(list)
  36     scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
  37     for pred_text, label_text in zip(pred_texts, label_texts):
  38         scores = scorer.score(pred_text, label_text)
  39         for k, v in scores.items():
  40             out[f"{k}.f1"].append(v.fmeasure)
  41     out_mean = dict()
  42     for k in out:
  43         out_mean[k] = np.mean(out[k])
  44     return out_mean, out
  45 
  46 
  47 @torch.inference_mode()
  48 def compute_per_token_acc(shift_logits, shift_labels, valid_masks):
  49     indices = torch.where(valid_masks)
  50     acc = (shift_logits.argmax(-1) == shift_labels)[indices].float()
  51     return {
  52         "per_token_accs": acc.flatten().tolist(),
  53         "n_per_token_accs": valid_masks.sum().item(),
  54     }
  55 
  56 
  57 @torch.inference_mode()
  58 def compute_prefix_matching(shift_logits, shift_labels, valid_masks):
  59     lengths = valid_masks.sum(dim=1)
  60 
  61     is_wrong = (shift_logits.argmax(-1) != shift_labels) * valid_masks
  62     is_correct = (shift_logits.argmax(-1) == shift_labels) * valid_masks
  63     # NOTE: not reliable for multi-turn conversations
  64     # ie, all tokens in the following user's turn will be correct
  65     # still monotonically correlate with perf though
  66     wrong_pos = torch.argmax(is_wrong, dim=1) - torch.argmax(valid_masks, dim=1)
  67     perf = wrong_pos / lengths
  68 
  69     # if all tokens are correct, set to 1
  70     perf = torch.where(is_correct.sum(dim=1) == lengths, 1, perf)
  71     return {
  72         "prefix_matchings": perf.tolist(),
  73         "n_prefix_matchings": valid_masks.shape[0],
  74     }
  75 
  76 
  77 @torch.inference_mode()
  78 def compute_perplexity(shift_logits, shift_labels, valid_masks):
  79     return {"perplexities_ph": [1], "n_perplexities_ph": 1}
  80 
  81 
  82 class Evaluator:
  83     def __init__(self, metric_fns: list[Callable]):
  84         self.metric_fns = metric_fns
  85         self.reset()
  86 
  87     def reset(self):
  88         self.accum_metrics = defaultdict(lambda: list((0,)))
  89         self.count = defaultdict(lambda: list((0,)))
  90 
  91     def update(self, shift_logits, shift_labels, valid_masks, lengths=None):
  92         for metric_fn in self.metric_fns:
  93             # overall metric
  94             metric = metric_fn(shift_logits, shift_labels, valid_masks)
  95             for k, v in metric.items():
  96                 key = k if not k.startswith("n_") else k[2:]
  97                 if k.startswith("n_"):
  98                     # prefix "n_" indicates the count of the metric
  99                     self.count[key].append(v)
 100                 else:
 101                     self.accum_metrics[key] += v
 102                 for start, end in LENGTH_BINS:
 103                     key_w_len = f"{key}_len_{start}-{end}"
 104                     if key_w_len not in self.accum_metrics:
 105                         # add key here so that it shows up in the output
 106                         self.accum_metrics[key_w_len] = [0]
 107                         self.count[key_w_len] = [0]
 108             # split samples into length groups, calculate metric for each group
 109             if lengths is not None:
 110                 for start, end in LENGTH_BINS:
 111                     logits, labels, masks = [], [], []
 112 
 113                     for logit, label, m, len in zip(
 114                         shift_logits, shift_labels, valid_masks, lengths
 115                     ):
 116                         if isinstance(len, torch.Tensor):
 117                             len = len.item()
 118                         if start <= len < end:
 119                             logits.append(logit)
 120                             labels.append(label)
 121                             masks.append(m)
 122 
 123                     if not logits:
 124                         continue
 125 
 126                     metric = metric_fn(
 127                         torch.stack(logits), torch.stack(labels), torch.stack(masks)
 128                     )
 129                     for k, v in metric.items():
 130                         if k.startswith("n_"):
 131                             key = f"{k[2:]}_len_{start}-{end}"
 132                             self.count[key].append(v)
 133                         else:
 134                             key = f"{k}_len_{start}-{end}"
 135                             self.accum_metrics[key] += v
 136 
 137     def compute(self):
 138         # Get result across entire eval set
 139         result = {
 140             k: np.sum(v) / np.sum(self.count[k]) if len(v) > 1 else "None"
 141             for k, v in self.accum_metrics.items()
 142         }
 143         # Reset batch statistics
 144         self.reset()
 145         return result
 146 
 147 
 148 @torch.no_grad()
 149 def compute_metrics(
 150     eval_pred: EvalPrediction,
 151     compute_result: bool,
 152     evaluator: Evaluator,
 153 ) -> dict | None:
 154     inputs = eval_pred.inputs
 155     len_key = "ctx_ids_len" if "ctx_ids_len" in inputs else "input_ids_len"
 156     lengths = inputs[len_key]
 157     logits, labels = eval_pred.predictions, eval_pred.label_ids
 158     shift_logits = logits[..., :-1, :]
 159     shift_labels = labels[..., 1:]
 160     valid_masks = torch.where(shift_labels != -100, 1, 0)
 161     evaluator.update(shift_logits, shift_labels, valid_masks, lengths)
 162     if compute_result:
 163         return evaluator.compute()
```

### /tmp/cl-repos/doc-to-lora/run_eval.py lines 8-179
```
   8 if __name__ == "__main__":
   9     import argparse
  10 
  11     parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
  12     parser.add_argument(
  13         "--model_name_or_path",
  14         type=str,
  15         default=None,
  16         help="Evaluate a base model from HuggingFace Hub, without loading checkpoint",
  17     )
  18     parser.add_argument(
  19         "--checkpoint_path",
  20         type=str,
  21         default=None,
  22         help="Path to the checkpoint to evaluate",
  23     )
  24     parser.add_argument(
  25         "--split",
  26         type=str,
  27         choices=["validation", "test"],
  28         default="validation",
  29         help="Which split to evaluate on",
  30     )
  31     parser.add_argument(
  32         "--datasets",
  33         type=str,
  34         nargs="+",
  35         help=(
  36             "Specific datasets to evaluate on."
  37             "If not provided, uses default from args.yaml"
  38         ),
  39     )
  40     parser.add_argument(
  41         "--eval_batch_size",
  42         type=int,
  43         default=8,
  44         help="Eval batch size for teacher forcing",
  45     )
  46     parser.add_argument(
  47         "--eval_batch_size_gen",
  48         type=int,
  49         default=32,
  50         help="Eval batch size for generation",
  51     )
  52     parser.add_argument(
  53         "--max_val_samples_per_ds",
  54         type=int,
  55         default=-1,
  56         help=(
  57             "Maximum number of validation samples per dataset. "
  58             "If -1, uses values from checkpoint config."
  59         ),
  60     )
  61     parser.add_argument(
  62         "--max_test_samples_per_ds",
  63         type=int,
  64         default=500,
  65         help=(
  66             "Maximum number of validation samples per dataset. "
  67             "If -1, uses values from checkpoint config."
  68         ),
  69     )
  70     parser.add_argument(
  71         "--max_ctx_chunk_len",
  72         type=int,
  73         default=-1,
  74         help="Maximum length of context chunk for evaluation",
  75     )
  76     parser.add_argument(
  77         "--max_new_tokens",
  78         type=int,
  79         default=256,
  80         help="Maximum number of new tokens to generate during evaluation",
  81     )
  82     parser.add_argument(
  83         "--remove_context",
  84         action="store_true",
  85         help="Remove context when evaluating the base model.",
  86     )
  87     parser.add_argument(
  88         "--use_cd",
  89         action="store_true",
  90         help="Use context distillation model for evaluation.",
  91     )
  92     parser.add_argument(
  93         "--cd_update_iterations",
  94         type=int,
  95         default=20,
  96         help="Number of update iterations for context distillation during evaluation",
  97     )
  98     parser.add_argument(
  99         "--cd_use_gen_q",
 100         action="store_true",
 101         help="Use generated queries for context distillation training.",
 102     )
 103     parser.add_argument(
 104         "--q_gen_rounds",
 105         type=int,
 106         default=4,
 107         help="Number of rounds of query generation for context distillation.",
 108     )
 109     parser.add_argument(
 110         "--cd_batch_size",
 111         type=int,
 112         default=16,
 113         help="Batch size for context distillation.",
 114     )
 115     parser.add_argument(
 116         "--use_iterative_mode",
 117         action="store_true",
 118         help="Use iterative mode LoRA layer-by-layer generation",
 119     )
 120     parser.add_argument(
 121         "--use_llmlingua",
 122         action="store_true",
 123         help="Use LLMLingua compression for evaluation",
 124     )
 125     parser.add_argument(
 126         "--llmlingua_compression_rate",
 127         type=float,
 128         default=0.9,
 129         help="Compression rate for LLMLingua",
 130     )
 131     parser.add_argument(
 132         "--use_t2l",
 133         action="store_true",
 134         help="Use Text-to-LoRA model for evaluation",
 135     )
 136     parser.add_argument(
 137         "--add_ctx_to_input",
 138         action="store_true",
 139         help="Add ctx to base model's input",
 140     )
 141     parser.add_argument(
 142         "--truncate_if_too_long_inp",
 143         action="store_true",
 144         help="Truncate input sequences that are too long",
 145     )
 146     parser.add_argument(
 147         "--truncate_if_too_long_ctx",
 148         action="store_true",
 149         help="Truncate ctx sequences that are too long",
 150     )
 151     parser.add_argument(
 152         "--gen_lora_scaling",
 153         type=float,
 154         default=1.0,
 155     )
 156     parser.add_argument(
 157         "--flip_ctx_inp",
 158         action="store_true",
 159         help="Flip the order of context and input",
 160     )
 161     parser.add_argument(
 162         "--use_generative_adapter",
 163         action="store_true",
 164         help="Use generative adapter for evaluation",
 165     )
 166 
 167     cli_args = vars(parser.parse_args())
 168 
 169     if cli_args["model_name_or_path"]:
 170         assert cli_args["max_ctx_chunk_len"] <= 0, (
 171             f"Evaluating base model shouldn't be used with `max_ctx_chunk_len`"
 172         )
 173 
 174     eval_batch_size_gen = cli_args.pop("eval_batch_size_gen")
 175     eval_batch_size = cli_args.pop("eval_batch_size")
 176     run_eval(
 177         **cli_args,
 178         eval_batch_size=eval_batch_size_gen,
 179         generative=True,
```
````


## Codepack E: `ContinualAI/avalanche`

# Files

## File: avalanche_snippets.md
````markdown
# avalanche snippets

These excerpts show the best part of Avalanche for our purposes: a clean metric vocabulary for continual learning, especially forgetting/BWT/FWT and evaluation plugin composition.

### /tmp/cl-repos/avalanche/examples/eval_plugin.py lines 24-200
```
  24 from avalanche.benchmarks import nc_benchmark
  25 from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location
  26 from avalanche.evaluation.metrics import (
  27     forgetting_metrics,
  28     accuracy_metrics,
  29     labels_repartition_metrics,
  30     loss_metrics,
  31     cpu_usage_metrics,
  32     timing_metrics,
  33     gpu_usage_metrics,
  34     ram_usage_metrics,
  35     disk_usage_metrics,
  36     MAC_metrics,
  37     bwt_metrics,
  38     forward_transfer_metrics,
  39     class_accuracy_metrics,
  40     amca_metrics,
  41 )
  42 from avalanche.models import SimpleMLP
  43 from avalanche.logging import (
  44     InteractiveLogger,
  45     TextLogger,
  46     CSVLogger,
  47     TensorboardLogger,
  48 )
  49 from avalanche.training.plugins import EvaluationPlugin
  50 from avalanche.training.supervised import Naive
  51 
  52 
  53 def main(args):
  54     # --- CONFIG
  55     device = torch.device(
  56         f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
  57     )
  58     # ---------
  59 
  60     # --- TRANSFORMATIONS
  61     train_transform = transforms.Compose(
  62         [
  63             RandomCrop(28, padding=4),
  64             ToTensor(),
  65             transforms.Normalize((0.1307,), (0.3081,)),
  66         ]
  67     )
  68     test_transform = transforms.Compose(
  69         [ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
  70     )
  71     # ---------
  72 
  73     # --- BENCHMARK CREATION
  74     mnist_train = MNIST(
  75         root=default_dataset_location("mnist"),
  76         train=True,
  77         download=True,
  78         transform=train_transform,
  79     )
  80     mnist_test = MNIST(
  81         root=default_dataset_location("mnist"),
  82         train=False,
  83         download=True,
  84         transform=test_transform,
  85     )
  86     benchmark = nc_benchmark(mnist_train, mnist_test, 5, task_labels=False, seed=1234)
  87     # ---------
  88 
  89     # MODEL CREATION
  90     model = SimpleMLP(num_classes=benchmark.n_classes)
  91 
  92     # DEFINE THE EVALUATION PLUGIN AND LOGGER
  93     # The evaluation plugin manages the metrics computation.
  94     # It takes as argument a list of metrics and a list of loggers.
  95     # The evaluation plugin calls the loggers to serialize the metrics
  96     # and save them in persistent memory or print them in the standard output.
  97 
  98     # log to text file
  99     text_logger = TextLogger(open("log.txt", "a"))
 100 
 101     # print to stdout
 102     interactive_logger = InteractiveLogger()
 103 
 104     csv_logger = CSVLogger()
 105 
 106     tb_logger = TensorboardLogger()
 107 
 108     eval_plugin = EvaluationPlugin(
 109         accuracy_metrics(
 110             minibatch=True,
 111             epoch=True,
 112             epoch_running=True,
 113             experience=True,
 114             stream=True,
 115         ),
 116         loss_metrics(
 117             minibatch=True,
 118             epoch=True,
 119             epoch_running=True,
 120             experience=True,
 121             stream=True,
 122         ),
 123         class_accuracy_metrics(
 124             epoch=True, stream=True, classes=list(range(benchmark.n_classes))
 125         ),
 126         amca_metrics(),
 127         forgetting_metrics(experience=True, stream=True),
 128         bwt_metrics(experience=True, stream=True),
 129         forward_transfer_metrics(experience=True, stream=True),
 130         cpu_usage_metrics(
 131             minibatch=True,
 132             epoch=True,
 133             epoch_running=True,
 134             experience=True,
 135             stream=True,
 136         ),
 137         timing_metrics(
 138             minibatch=True,
 139             epoch=True,
 140             epoch_running=True,
 141             experience=True,
 142             stream=True,
 143         ),
 144         ram_usage_metrics(
 145             every=0.5, minibatch=True, epoch=True, experience=True, stream=True
 146         ),
 147         gpu_usage_metrics(
 148             args.cuda,
 149             every=0.5,
 150             minibatch=True,
 151             epoch=True,
 152             experience=True,
 153             stream=True,
 154         ),
 155         disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
 156         MAC_metrics(minibatch=True, epoch=True, experience=True),
 157         labels_repartition_metrics(on_train=True, on_eval=True),
 158         loggers=[interactive_logger, text_logger, csv_logger, tb_logger],
 159         collect_all=True,
 160     )  # collect all metrics (set to True by default)
 161 
 162     # CREATE THE STRATEGY INSTANCE (NAIVE)
 163     cl_strategy = Naive(
 164         model,
 165         SGD(model.parameters(), lr=0.001, momentum=0.9),
 166         CrossEntropyLoss(),
 167         train_mb_size=500,
 168         train_epochs=1,
 169         eval_mb_size=100,
 170         device=device,
 171         evaluator=eval_plugin,
 172         eval_every=1,
 173     )
 174 
 175     # TRAINING LOOP
 176     print("Starting experiment...")
 177     results = []
 178     for i, experience in enumerate(benchmark.train_stream):
 179         print("Start of experience: ", experience.current_experience)
 180         print("Current Classes: ", experience.classes_in_this_experience)
 181 
 182         # train returns a dictionary containing last recorded value
 183         # for each metric.
 184         res = cl_strategy.train(experience, eval_streams=[benchmark.test_stream])
 185         print("Training completed")
 186 
 187         print("Computing accuracy on the whole test set")
 188         # test returns a dictionary with the last metric collected during
 189         # evaluation on that stream
 190         results.append(cl_strategy.eval(benchmark.test_stream))
 191 
 192     print(f"Test metrics:\n{results}")
 193 
 194     # Dict with all the metric curves,
 195     # only available when `collect_all` is True.
 196     # Each entry is a (x, metric value) tuple.
 197     # You can use this dictionary to manipulate the
 198     # metrics without avalanche.
 199     all_metrics = cl_strategy.evaluator.get_all_metrics()
 200     print(f"Stored metrics: {list(all_metrics.keys())}")
```

### /tmp/cl-repos/avalanche/avalanche/evaluation/metrics/forgetting_bwt.py lines 41-117
```
  41 class Forgetting(Metric[Dict[int, float]]):
  42     """
  43     The standalone Forgetting metric.
  44     This metric returns the forgetting relative to a specific key.
  45     Alternatively, this metric returns a dict in which each key is associated
  46     to the forgetting.
  47     Forgetting is computed as the difference between the first value recorded
  48     for a specific key and the last value recorded for that key.
  49     The value associated to a key can be update with the `update` method.
  50 
  51     At initialization, this metric returns an empty dictionary.
  52     """
  53 
  54     def __init__(self):
  55         """
  56         Creates an instance of the standalone Forgetting metric
  57         """
  58 
  59         self.initial: Dict[int, float] = dict()
  60         """
  61         The initial value for each key.
  62         """
  63 
  64         self.last: Dict[int, float] = dict()
  65         """
  66         The last value detected for each key
  67         """
  68 
  69     def update_initial(self, k, v):
  70         self.initial[k] = v
  71 
  72     def update_last(self, k, v):
  73         self.last[k] = v
  74 
  75     def update(self, k, v, initial=False):
  76         if initial:
  77             self.update_initial(k, v)
  78         else:
  79             self.update_last(k, v)
  80 
  81     def result_key(self, k: int) -> Optional[float]:
  82         """
  83         Compute the forgetting for a specific key.
  84 
  85         :param k: the key for which returning forgetting.
  86 
  87         :return: the difference between the first and last value encountered
  88             for k, if k is not None. It returns None if k has not been updated
  89             at least twice.
  90         """
  91         assert k is not None
  92         if k in self.initial and k in self.last:
  93             return self.initial[k] - self.last[k]
  94         else:
  95             return None
  96 
  97     def result(self) -> Dict[int, float]:
  98         """
  99         Compute the forgetting for all keys.
 100 
 101         :return: A dictionary containing keys whose value has been updated
 102             at least twice. The associated value is the difference between
 103             the first and last value recorded for that key.
 104         """
 105 
 106         ik = set(self.initial.keys())
 107         both_keys = list(ik.intersection(set(self.last.keys())))
 108 
 109         forgetting: Dict[int, float] = {}
 110         for k in both_keys:
 111             forgetting[k] = self.initial[k] - self.last[k]
 112 
 113         return forgetting
 114 
 115     def reset_last(self) -> None:
 116         self.last = dict()
 117 
```

### /tmp/cl-repos/avalanche/avalanche/evaluation/metrics/forgetting_bwt.py lines 123-145
```
 123 class GenericExperienceForgetting(
 124     PluginMetric[TResult_co], Generic[TMetric, TResult_co, TResultKey_co], ABC
 125 ):
 126     """
 127     The GenericExperienceForgetting metric, describing the change in
 128     a metric detected for a certain experience. The user should
 129     subclass this and provide the desired metric.
 130 
 131     In particular, the user should override:
 132     * __init__ by calling `super` and instantiating the `self.current_metric`
 133     property as a valid avalanche metric
 134     * `metric_update`, to update `current_metric`
 135     * `metric_result` to get the result from `current_metric`.
 136     * `__str__` to define the experience forgetting  name.
 137 
 138     This plugin metric, computed separately for each experience,
 139     is the difference between the metric result obtained after
 140     first training on a experience and the metric result obtained
 141     on the same experience at the end of successive experiences.
 142 
 143     This metric is computed during the eval phase only.
 144     """
 145 
```

### /tmp/cl-repos/avalanche/avalanche/evaluation/metrics/forgetting_bwt.py lines 215-245
```
 215     def before_training_exp(self, strategy: "SupervisedTemplate") -> None:
 216         assert strategy.experience is not None
 217         self.train_exp_id = strategy.experience.current_experience
 218 
 219     def before_eval(self, strategy) -> None:
 220         self.reset_last()
 221 
 222     def before_eval_exp(self, strategy: "SupervisedTemplate") -> None:
 223         self._current_metric.reset()
 224 
 225     def after_eval_iteration(self, strategy: "SupervisedTemplate") -> None:
 226         super().after_eval_iteration(strategy)
 227         assert strategy.experience is not None
 228         self.eval_exp_id = strategy.experience.current_experience
 229         self.metric_update(strategy)
 230 
 231     def after_eval_exp(self, strategy: "SupervisedTemplate") -> MetricResult:
 232         # update experience on which training just ended
 233         self._check_eval_exp_id()
 234         if self.train_exp_id == self.eval_exp_id:
 235             self.update(self.eval_exp_id, self.metric_result(strategy), initial=True)
 236         else:
 237             # update other experiences
 238             # if experience has not been encountered in training
 239             # its value will not be considered in forgetting
 240             self.update(self.eval_exp_id, self.metric_result(strategy))
 241 
 242         return self._package_result(strategy)
 243 
 244     def after_eval(self, strategy: "SupervisedTemplate") -> MetricResult:
 245         self.eval_exp_id = -1  # reset the last experience ID
```

### /tmp/cl-repos/avalanche/avalanche/evaluation/metrics/forgetting_bwt.py lines 285-335
```
 285 class ExperienceForgetting(
 286     GenericExperienceForgetting[TaskAwareAccuracy, Dict[int, float], Optional[float]]
 287 ):
 288     """
 289     The ExperienceForgetting metric, describing the accuracy loss
 290     detected for a certain experience.
 291 
 292     This plugin metric, computed separately for each experience,
 293     is the difference between the accuracy result obtained after
 294     first training on a experience and the accuracy result obtained
 295     on the same experience at the end of successive experiences.
 296 
 297     This metric is computed during the eval phase only.
 298     """
 299 
 300     def __init__(self):
 301         """
 302         Creates an instance of the ExperienceForgetting metric.
 303         """
 304 
 305         super().__init__(TaskAwareAccuracy())
 306 
 307     def result_key(self, k: int) -> Optional[float]:
 308         """
 309         Forgetting for an experience defined by its key.
 310 
 311         See :class:`Forgetting` documentation for more detailed information.
 312 
 313         :param k: key from which to compute the forgetting.
 314         :return: the difference between the first and last value encountered
 315             for k.
 316         """
 317         return self.forgetting.result_key(k=k)
 318 
 319     def result(self) -> Dict[int, float]:
 320         """
 321         Forgetting for all experiences.
 322 
 323         See :class:`Forgetting` documentation for more detailed information.
 324 
 325         :return: A dictionary containing keys whose value has been updated
 326             at least twice. The associated value is the difference between
 327             the first and last value recorded for that key.
 328         """
 329         return self.forgetting.result()
 330 
 331     def metric_update(self, strategy):
 332         self._current_metric.update(strategy.mb_y, strategy.mb_output, 0)
 333 
 334     def metric_result(self, strategy):
 335         return self._current_metric.result(0)[0]
```
````


## Codepack F: `kmccleary3301/nested_learning`

# Files

## File: nested_learning_snippets.md
````markdown
# nested_learning snippets

These excerpts are useful because they show a more modern continual/online training and evaluation stack with integrity metadata, but also illustrate how quickly complexity can explode.

### /tmp/cl-repos/nested_learning/src/nested_learning/training.py lines 685-760
```
 685 def run_training_loop(
 686     cfg: DictConfig,
 687     *,
 688     device: torch.device,
 689     distributed: bool = False,
 690     dist_ctx: DistributedContext | None = None,
 691 ) -> Dict[str, float]:
 692     algorithm_mode = _resolve_algorithm_mode(cfg)
 693     _validate_algorithm_mode_constraints(
 694         cfg,
 695         algorithm_mode=algorithm_mode,
 696         distributed=distributed,
 697     )
 698     _validate_online_chunking_constraints(cfg)
 699     _validate_distributed_config(cfg, distributed)
 700     _validate_paper_auditing_variant(cfg)
 701     _validate_fast_state_batch_semantics(cfg)
 702     _validate_online_update_fast_state_semantics(cfg)
 703     model = build_model_from_cfg(cfg.model).to(device)
 704     train_seed = cfg.train.get("seed")
 705     deterministic = cfg.train.get("deterministic", False)
 706     if train_seed is not None:
 707         _seed_everything(int(train_seed), deterministic=bool(deterministic))
 708     model = _maybe_compile_model(model, cfg.train.get("compile"))
 709     if distributed:
 710         assert dist_ctx is not None
 711         if device.type == "cuda":
 712             idx = device.index if device.index is not None else 0
 713             model = torch.nn.parallel.DistributedDataParallel(
 714                 model,
 715                 device_ids=[idx],
 716                 output_device=idx,
 717                 find_unused_parameters=True,
 718             )
 719         else:
 720             model = torch.nn.parallel.DistributedDataParallel(
 721                 model,
 722                 find_unused_parameters=True,
 723             )
 724         base_model = model.module
 725     else:
 726         base_model = model
 727 
 728     _validate_tied_lm_head_for_paper_auditing(cfg, base_model)
 729 
 730     seed_offset = 0
 731     if train_seed is not None and dist_ctx is not None:
 732         seed_offset = dist_ctx.rank
 733     dataloader_seed = None if train_seed is None else int(train_seed) + seed_offset
 734     dataloader, sampler = build_dataloader(
 735         cfg.data,
 736         distributed=distributed,
 737         dist_ctx=dist_ctx,
 738         seed=dataloader_seed,
 739     )
 740     optimizer = _build_optimizer(base_model, cfg, device=device)
 741     autocast_factory = _make_autocast_factory(device, cfg.train.get("mixed_precision"))
 742     logger = init_logger(getattr(cfg, "logging", None), cfg)
 743     if distributed and dist_ctx is not None and dist_ctx.rank != 0:
 744         logger = NullLogger()
 745     _log_run_features(logger, base_model, cfg, optimizer, device)
 746     steps = cfg.train.steps
 747     log_interval = cfg.train.get("log_interval", 1)
 748     per_layer_teach = bool(cfg.train.get("per_layer_teach_signal", False))
 749     online_updates = bool(cfg.train.get("online_updates", False))
 750     online_chunk_size = int(cfg.train.get("online_chunk_size", 0) or 0)
 751     online_boundary_targets = bool(cfg.train.get("online_boundary_targets", False))
 752     online_carry_attention_cache = bool(cfg.train.get("online_carry_attention_cache", False))
 753     use_fast_state = bool(cfg.train.get("use_fast_state", False))
 754     fail_if_faithful_disabled = bool(cfg.train.get("fail_if_paper_faithful_disabled", False))
 755     strict_streaming = bool(cfg.train.get("strict_streaming_contract", False))
 756     if distributed and per_layer_teach:
 757         msg = "per_layer_teach_signal disabled under DDP (uses base model methods)"
 758         if fail_if_faithful_disabled or strict_streaming:
 759             raise RuntimeError(
 760                 f"{msg}. Set train.strict_streaming_contract=false and "
```

### /tmp/cl-repos/nested_learning/src/nested_learning/training.py lines 802-858
```
 802     metrics: Dict[str, float] = {}
 803     surprise_metric_getter = getattr(base_model, "get_surprise_metric", None)
 804     surprise_metric = (
 805         str(surprise_metric_getter()).strip().lower()
 806         if callable(surprise_metric_getter)
 807         else str(cfg.model.get("surprise_metric", "l2")).strip().lower()
 808     )
 809     for step in range(steps):
 810         if sampler is not None and step % len(dataloader) == 0:
 811             sampler.set_epoch(epoch)
 812             epoch += 1
 813         try:
 814             batch = next(step_iter)
 815         except StopIteration:
 816             step_iter = iter(dataloader)
 817             batch = next(step_iter)
 818         tokens = batch.to(device)
 819         fast_state = None
 820         if use_fast_state:
 821             init_fn = getattr(base_model, "init_fast_state", None)
 822             if not callable(init_fn):
 823                 raise ValueError("train.use_fast_state=true requires model.init_fast_state()")
 824             fast_state = init_fn()
 825         _apply_teach_schedule(base_model, cfg, step)
 826         update_metrics: Dict[str, float] = {}
 827         if online_updates and hasattr(base_model, "forward_with_block_outputs"):
 828             total_loss = 0.0
 829             total_tokens = 0
 830             teach_signal_norm = 0.0
 831             optimizer.zero_grad()
 832             chunk_size = online_chunk_size
 833             if chunk_size <= 0:
 834                 inferred = _infer_online_chunk_size(base_model)
 835                 chunk_size = inferred if inferred is not None else tokens.size(1)
 836             if chunk_size < 1:
 837                 print(f"[train] online_chunk_size={chunk_size} is too small; clamping to 1")
 838                 chunk_size = 1
 839             attention_cache = None
 840             if online_carry_attention_cache:
 841                 init_attention_cache = getattr(base_model, "init_attention_cache", None)
 842                 if not callable(init_attention_cache):
 843                     raise RuntimeError(
 844                         "online_carry_attention_cache=true requires model.init_attention_cache()"
 845                     )
 846                 attention_cache = init_attention_cache()
 847 
 848             chunk_iter: Iterator[tuple[torch.Tensor, torch.Tensor | None, bool]]
 849             if online_boundary_targets:
 850                 chunk_iter = _iter_online_boundary_chunks(tokens, chunk_size=chunk_size)
 851             else:
 852                 chunk_iter = (
 853                     (chunk, None, finalize_updates)
 854                     for chunk, finalize_updates in _iter_online_token_chunks(
 855                         tokens, chunk_size=chunk_size
 856                     )
 857                 )
 858             for chunk_tokens, next_tokens, finalize_updates in chunk_iter:
```

### /tmp/cl-repos/nested_learning/src/nested_learning/training.py lines 1041-1069
```
1041                 if hasattr(base_model, "pop_update_metrics"):
1042                     update_metrics = base_model.pop_update_metrics()
1043         if step % log_interval == 0:
1044             ppl = torch.exp(loss.detach()).item()
1045             metrics_payload = {
1046                 "loss": loss.item(),
1047                 "ppl": ppl,
1048                 "teach_signal_norm": teach_signal_norm,
1049             }
1050             metrics_payload.update(update_metrics)
1051             logger.log(metrics_payload, step=step)
1052             if (not distributed) or (dist_ctx and dist_ctx.rank == 0):
1053                 print(
1054                     f"[train] step={step} loss={loss.item():.4f} "
1055                     f"ppl={ppl:.2f} teach_norm={teach_signal_norm:.4f}"
1056                 )
1057             metrics = metrics_payload
1058         maybe_save_checkpoint(
1059             cfg,
1060             base_model,
1061             optimizer,
1062             step=step,
1063             total_steps=steps,
1064             distributed=distributed,
1065             dist_ctx=dist_ctx,
1066             step_offset=int(cfg.train.get("step_offset", 0) or 0),
1067         )
1068     logger.finish()
1069     return metrics
```

### /tmp/cl-repos/nested_learning/src/nested_learning/training.py lines 1492-1528
```
1492 def write_checkpoint_metadata(cfg: DictConfig, ckpt_path: Path, step: int) -> None:
1493     config_yaml = OmegaConf.to_yaml(cfg)
1494     config_path = ckpt_path.with_suffix(".yaml")
1495     config_path.write_text(config_yaml)
1496     config_hash = sha256(config_yaml.encode("utf-8")).hexdigest()
1497     ckpt_hash = _checksum_path(str(ckpt_path))
1498     sha_path = ckpt_path.with_suffix(".sha256")
1499     if ckpt_hash:
1500         sha_path.write_text(f"{ckpt_hash}  {ckpt_path.name}\n")
1501     tokenizer_path = cfg.data.get("tokenizer_path") if hasattr(cfg, "data") else None
1502     metadata = {
1503         "step": step,
1504         "checkpoint_sha256": ckpt_hash,
1505         "config_sha256": config_hash,
1506         "tokenizer_hash": _checksum_path(tokenizer_path) if tokenizer_path else None,
1507         "config_path": str(config_path),
1508         "algorithm_mode": str(cfg.train.get("algorithm_mode", "two_pass_stopgrad_updates")),
1509         "online_updates": bool(cfg.train.get("online_updates", False)),
1510         "online_boundary_targets": bool(cfg.train.get("online_boundary_targets", False)),
1511         "online_carry_attention_cache": bool(
1512             cfg.train.get("online_carry_attention_cache", False)
1513         ),
1514         "use_fast_state": bool(cfg.train.get("use_fast_state", False)),
1515         "rng_states": _capture_rng_states(),
1516     }
1517     ckpt_path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2))
1518 
1519 
1520 def verify_checkpoint_integrity(ckpt_path: Path) -> Dict[str, object]:
1521     if not ckpt_path.exists():
1522         raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
1523     meta_path = ckpt_path.with_suffix(".meta.json")
1524     if not meta_path.exists():
1525         raise FileNotFoundError(f"Metadata file {meta_path} missing")
1526     metadata = json.loads(meta_path.read_text())
1527     computed_sha = _checksum_path(str(ckpt_path))
1528     recorded_sha = metadata.get("checkpoint_sha256")
```

### /tmp/cl-repos/nested_learning/scripts/eval/continual.py lines 143-253
```
 143 def main(
 144     config: Path = typer.Option(..., help="Hydra model config for HOPE."),
 145     checkpoints: List[Path] = typer.Option(
 146         ..., help="Ordered list of checkpoints (chronological)."
 147     ),
 148     segments_yaml: Path = typer.Option(..., help="YAML describing shard directories per segment."),
 149     tokenizer_path: Path = typer.Option(..., help="SentencePiece model path (unused for now)."),
 150     batch_size: int = typer.Option(4, help="Batch size for evaluation."),
 151     max_batches: int = typer.Option(50, help="Max batches per segment (0 = entire dataset)."),
 152     device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu"),
 153     output: Path = typer.Option(Path("eval/continual_results.json")),
 154     memorize: bool = typer.Option(False, help="Enable memorization while evaluating segments."),
 155     memorize_steps: int = typer.Option(1, help="Memorization passes per batch."),
 156     memorize_no_reset: bool = typer.Option(True, help="Keep memory between segments by default."),
 157     memorize_surprise_threshold: float = typer.Option(
 158         None, help="Minimum teach-signal norm needed to memorize a batch."
 159     ),
 160     memorize_paths: str = typer.Option(
 161         "all",
 162         help=(
 163             "Comma-separated memory paths to update (e.g., 'titan,cms_fast'); "
 164             "use 'all' for default behavior."
 165         ),
 166     ),
 167     eval_state_mode: str = typer.Option(
 168         "reset_per_sample",
 169         help="Streaming eval state mode: 'reset_per_sample' or 'carry_across_samples'.",
 170     ),
 171     eval_use_fast_state: bool = typer.Option(
 172         False,
 173         help="Use model fast state during inference scoring (independent from memorization state).",
 174     ),
 175     eval_use_attention_cache: bool = typer.Option(
 176         False,
 177         help="Use attention KV cache during inference scoring.",
 178     ),
 179 ) -> None:
 180     segments = load_segments(segments_yaml)
 181     if not segments:
 182         raise typer.BadParameter("No segments found in YAML.")
 183 
 184     cfg = OmegaConf.load(config)
 185     cfg = unwrap_config(cfg)
 186     device_obj = resolve_device(device)
 187     results = []
 188 
 189     if memorize_paths.lower() == "all":
 190         allowed_paths = None
 191     else:
 192         allowed_paths = tuple(path.strip() for path in memorize_paths.split(",") if path.strip())
 193     memorize_cfg = MemorizeConfig(
 194         enabled=memorize,
 195         steps=max(1, memorize_steps),
 196         reset=not memorize_no_reset,
 197         use_correct_answer=False,
 198         surprise_threshold=memorize_surprise_threshold,
 199         paths=allowed_paths,
 200     )
 201 
 202     for step_idx, ckpt_path in enumerate(checkpoints):
 203         state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
 204         model = build_model_from_cfg(cfg.model)
 205         state_dict = state["model"] if "model" in state else state
 206         missing, unexpected = model.load_state_dict(state_dict, strict=False)
 207         if missing or unexpected:
 208             print(
 209                 "[continual] Warning: state_dict mismatch "
 210                 f"(missing={len(missing)} unexpected={len(unexpected)}) – continuing."
 211             )
 212         model = model.to(device_obj)
 213 
 214         segment_losses = {}
 215         baseline_losses = {}
 216         segment_stats = {}
 217         for segment in segments:
 218             name = segment["name"]
 219             shards_dir = Path(segment["shards_dir"])
 220             dataset = TokenShardDataset(shards_dir)
 221             loader = DataLoader(
 222                 dataset,
 223                 batch_size=batch_size,
 224                 shuffle=False,
 225                 num_workers=0,
 226                 collate_fn=collate_batch,
 227             )
 228             base_loss, mem_loss, stats = evaluate_segment(
 229                 model,
 230                 loader,
 231                 device_obj,
 232                 None if max_batches <= 0 else max_batches,
 233                 memorize_cfg,
 234                 eval_state_mode=eval_state_mode,
 235                 eval_use_fast_state=eval_use_fast_state,
 236                 eval_use_attention_cache=eval_use_attention_cache,
 237             )
 238             baseline_losses[name] = base_loss
 239             segment_losses[name] = mem_loss
 240             if stats:
 241                 segment_stats[name] = stats
 242 
 243         entry = {"checkpoint": str(ckpt_path), "segment_losses": segment_losses}
 244         if memorize_cfg.enabled:
 245             entry["segment_baseline_losses"] = baseline_losses
 246             entry["segment_memorize_delta"] = {
 247                 name: baseline_losses[name] - segment_losses[name] for name in segment_losses
 248             }
 249             if segment_stats:
 250                 entry["memorize_stats"] = segment_stats
 251         entry["eval_state_mode"] = eval_state_mode
 252         entry["eval_use_fast_state"] = bool(eval_use_fast_state)
 253         entry["eval_use_attention_cache"] = bool(eval_use_attention_cache)
```
````

