#!/usr/bin/env python3
"""Materialize the v4 CounterFact continual-learning substrate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "azhx/counterfact"
DEFAULT_CONFIG = "default"
DEFAULT_SOURCE_SPLIT = "train"
DEFAULT_PUBLIC_CASES = 96
DEFAULT_SMOKE_CASES = 12
DEFAULT_HOST_LOCAL_CONFIRM_ROOT = Path(
    "~/shared/artifacts/autoresearch-continual-learning/protected"
).expanduser()


@dataclass(frozen=True)
class CounterFactPackIds:
    """Stable ids for one CounterFact v4 lane."""

    train_generator_version: str
    visible_pack_id: str
    confirmation_pack_id: str
    train_status: str
    visible_status: str
    confirmation_summary_status: str
    scale_label: str
    purpose_label: str


ACTIVE_PACK_IDS = CounterFactPackIds(
    train_generator_version="cl_seq_train_v4_counterfact_standard",
    visible_pack_id="cl_dev_visible_v4_counterfact_standard",
    confirmation_pack_id="cl_confirm_locked_v4_counterfact_standard",
    train_status="active_public_cl_generator",
    visible_status="active_cl_visible_pack",
    confirmation_summary_status="active_confirmation_pack_summary",
    scale_label="standard",
    purpose_label="active development and protected-confirmation lane",
)

SMOKE_PACK_IDS = CounterFactPackIds(
    train_generator_version="cl_seq_train_v4_counterfact_smoke",
    visible_pack_id="cl_dev_visible_v4_counterfact_smoke",
    confirmation_pack_id="cl_confirm_locked_v4_counterfact_smoke",
    train_status="smoke_public_cl_generator",
    visible_status="smoke_cl_visible_pack",
    confirmation_summary_status="smoke_confirmation_pack_summary",
    scale_label="smoke",
    purpose_label="fast plumbing and regression-check lane",
)

LEGACY_PRESTANDARD_PACK_IDS = CounterFactPackIds(
    train_generator_version="cl_seq_train_v4_counterfact",
    visible_pack_id="cl_dev_visible_v4_counterfact",
    confirmation_pack_id="cl_confirm_locked_v4_counterfact",
    train_status="prestandard_public_cl_generator_archived_for_history",
    visible_status="prestandard_cl_visible_pack_archived_for_history",
    confirmation_summary_status="prestandard_confirmation_pack_summary_archived_for_history",
    scale_label="prestandard_smoke",
    purpose_label="historical 12-case lane used before the v4 standard scale-up",
)


@dataclass(frozen=True)
class CounterFactCase:
    """One validated CounterFact edit case."""

    row_idx: int
    case_id: int
    relation_id: str
    subject: str
    prompt: str
    target_new: str
    target_true: str
    paraphrase_prompts: tuple[str, str]
    neighborhood_prompts: tuple[str, str, str]
    generation_prompts: tuple[str, ...]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source-split", default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--public-cases", type=int, default=DEFAULT_PUBLIC_CASES)
    parser.add_argument("--smoke-cases", type=int, default=DEFAULT_SMOKE_CASES)
    parser.add_argument(
        "--no-legacy-prestandard-aliases",
        action="store_true",
        help="Do not rewrite the original v4 ids as 12-case historical aliases.",
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-scan-rows", type=int, default=20000)
    parser.add_argument(
        "--source-mode",
        choices=["auto", "datasets", "dataset-server"],
        default="auto",
        help="Prefer the optional datasets loader when installed; otherwise use the HF rows API.",
    )
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument(
        "--host-confirm-root",
        default=DEFAULT_HOST_LOCAL_CONFIRM_ROOT.as_posix(),
        help="Host-local directory where raw locked confirmation packs are written.",
    )
    parser.add_argument(
        "--host-confirm-path",
        help=(
            "Compatibility override for the active locked confirmation pack path. "
            "When omitted, --host-confirm-root/<pack_id>.json is used."
        ),
    )
    parser.add_argument("--repo-root", default=REPO_ROOT.as_posix())
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    requested_cases = max(args.public_cases, args.smoke_cases)
    cases = select_cases(
        fetch_counterfact_rows(
            dataset=args.dataset,
            config=args.config,
            split=args.source_split,
            page_size=args.page_size,
            max_scan_rows=args.max_scan_rows,
            source_mode=args.source_mode,
            request_retries=args.request_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
        ),
        count=requested_cases,
    )
    if len(cases) < requested_cases:
        raise RuntimeError(
            f"only found {len(cases)} valid CounterFact cases; requested {requested_cases}"
        )

    host_confirm_root = Path(args.host_confirm_root).expanduser()
    active_host_confirm_path = (
        Path(args.host_confirm_path).expanduser()
        if args.host_confirm_path
        else host_confirm_root / f"{ACTIVE_PACK_IDS.confirmation_pack_id}.json"
    )
    materialize_v4_files(
        repo_root=repo_root,
        cases=cases[: args.public_cases],
        dataset=args.dataset,
        config=args.config,
        source_split=args.source_split,
        host_confirm_path=active_host_confirm_path,
        pack_ids=ACTIVE_PACK_IDS,
    )
    materialize_v4_files(
        repo_root=repo_root,
        cases=cases[: args.smoke_cases],
        dataset=args.dataset,
        config=args.config,
        source_split=args.source_split,
        host_confirm_path=host_confirm_root / f"{SMOKE_PACK_IDS.confirmation_pack_id}.json",
        pack_ids=SMOKE_PACK_IDS,
    )
    if not args.no_legacy_prestandard_aliases:
        materialize_v4_files(
            repo_root=repo_root,
            cases=cases[: args.smoke_cases],
            dataset=args.dataset,
            config=args.config,
            source_split=args.source_split,
            host_confirm_path=(
                host_confirm_root / f"{LEGACY_PRESTANDARD_PACK_IDS.confirmation_pack_id}.json"
            ),
            pack_ids=LEGACY_PRESTANDARD_PACK_IDS,
        )
    print(
        json.dumps(
            {
                "active_cases": args.public_cases,
                "active_examples": args.public_cases * 3,
                "active_train_generator": f"data/train_generators/{ACTIVE_PACK_IDS.train_generator_version}.yaml",
                "active_visible_dev_pack": f"data/packs/{ACTIVE_PACK_IDS.visible_pack_id}.yaml",
                "active_confirmation_summary": f"data/packs/{ACTIVE_PACK_IDS.confirmation_pack_id}.summary.yaml",
                "active_host_confirmation_pack": str(active_host_confirm_path),
                "smoke_cases": args.smoke_cases,
                "smoke_examples": args.smoke_cases * 3,
                "smoke_train_generator": f"data/train_generators/{SMOKE_PACK_IDS.train_generator_version}.yaml",
                "smoke_visible_dev_pack": f"data/packs/{SMOKE_PACK_IDS.visible_pack_id}.yaml",
                "smoke_confirmation_summary": f"data/packs/{SMOKE_PACK_IDS.confirmation_pack_id}.summary.yaml",
                "legacy_prestandard_aliases_written": not args.no_legacy_prestandard_aliases,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def fetch_counterfact_rows(
    *,
    dataset: str,
    config: str,
    split: str,
    page_size: int,
    max_scan_rows: int,
    source_mode: str = "auto",
    request_retries: int = 3,
    retry_sleep_seconds: float = 2.0,
) -> Iterable[dict[str, Any]]:
    """Yield CounterFact rows, preferring a cached dataset loader when available."""

    if source_mode not in {"auto", "datasets", "dataset-server"}:
        raise ValueError("source_mode must be one of auto, datasets, dataset-server")

    if source_mode in {"auto", "datasets"}:
        try:
            yield from fetch_counterfact_rows_from_datasets(
                dataset=dataset,
                config=config,
                split=split,
                max_scan_rows=max_scan_rows,
            )
            return
        except ImportError:
            if source_mode == "datasets":
                raise
            print(
                "optional datasets package unavailable; falling back to dataset-server rows API",
                file=sys.stderr,
            )
        except Exception as exc:
            if source_mode == "datasets":
                raise
            print(
                f"datasets loader failed; falling back to dataset-server rows API: {exc}",
                file=sys.stderr,
            )

    yield from fetch_counterfact_rows_from_dataset_server(
        dataset=dataset,
        config=config,
        split=split,
        page_size=page_size,
        max_scan_rows=max_scan_rows,
        request_retries=request_retries,
        retry_sleep_seconds=retry_sleep_seconds,
    )


def fetch_counterfact_rows_from_datasets(
    *,
    dataset: str,
    config: str,
    split: str,
    max_scan_rows: int,
) -> Iterable[dict[str, Any]]:
    """Yield rows through the optional Hugging Face datasets package."""

    try:
        from datasets import load_dataset
    except ImportError:
        raise

    if max_scan_rows <= 0:
        raise ValueError("max_scan_rows must be positive")

    loaded = load_dataset(dataset, config, split=split)
    for row_idx, row in enumerate(loaded):
        if row_idx >= max_scan_rows:
            break
        if isinstance(row, Mapping):
            yield {
                "row_idx": row_idx,
                "row": dict(row),
            }


def fetch_counterfact_rows_from_dataset_server(
    *,
    dataset: str,
    config: str,
    split: str,
    page_size: int,
    max_scan_rows: int,
    request_retries: int = 3,
    retry_sleep_seconds: float = 2.0,
) -> Iterable[dict[str, Any]]:
    """Yield rows from the Hugging Face dataset-server JSON API."""

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if max_scan_rows <= 0:
        raise ValueError("max_scan_rows must be positive")

    offset = 0
    while offset < max_scan_rows:
        length = min(page_size, max_scan_rows - offset)
        query = urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        payload = fetch_json_url(
            url,
            request_retries=request_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            break
        for item in rows:
            if isinstance(item, Mapping):
                yield dict(item)
        if len(rows) < length:
            break
        offset += length


def fetch_json_url(
    url: str,
    *,
    request_retries: int,
    retry_sleep_seconds: float,
) -> dict[str, Any]:
    """Fetch one JSON URL with bounded retry for transient dataset-server stalls."""

    attempts = max(1, request_retries)
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(url, timeout=30) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise RuntimeError(f"dataset-server returned non-object JSON for {url}")
            return payload
        except (TimeoutError, URLError, json.JSONDecodeError) as exc:
            if attempt >= attempts:
                raise
            sleep_for = max(0.0, retry_sleep_seconds) * attempt
            print(
                f"retrying CounterFact page after transient fetch error: {exc}",
                file=sys.stderr,
            )
            if sleep_for:
                time.sleep(sleep_for)

    raise RuntimeError("unreachable retry state")


def select_cases(rows: Iterable[Mapping[str, Any]], *, count: int) -> list[CounterFactCase]:
    """Select stable, simple, ASCII CounterFact cases with broad relation coverage."""

    if count <= 0:
        raise ValueError("count must be positive")

    selected: list[CounterFactCase] = []
    reserves: list[CounterFactCase] = []
    seen_relation_ids: set[str] = set()
    seen_case_ids: set[int] = set()
    for item in rows:
        case = normalize_counterfact_case(item)
        if case is None or case.case_id in seen_case_ids:
            continue
        seen_case_ids.add(case.case_id)
        if case.relation_id not in seen_relation_ids:
            selected.append(case)
            seen_relation_ids.add(case.relation_id)
        else:
            reserves.append(case)
        if len(selected) >= count:
            return selected[:count]

    for case in reserves:
        if len(selected) >= count:
            break
        selected.append(case)
    return selected[:count]


def normalize_counterfact_case(item: Mapping[str, Any]) -> CounterFactCase | None:
    row = item.get("row")
    if not isinstance(row, Mapping):
        return None
    rewrite = row.get("requested_rewrite")
    if not isinstance(rewrite, Mapping):
        return None

    case_id = int(row.get("case_id", -1))
    row_idx = int(item.get("row_idx", -1))
    relation_id = string_value(rewrite.get("relation_id"))
    subject = string_value(rewrite.get("subject"))
    prompt_template = string_value(rewrite.get("prompt"))
    target_new = target_string(rewrite.get("target_new"))
    target_true = target_string(rewrite.get("target_true"))
    if case_id < 0 or not relation_id:
        return None
    if not valid_short_text(subject, max_words=8, max_chars=80):
        return None
    if not valid_answer(target_new) or not valid_answer(target_true):
        return None
    prompt = render_counterfact_prompt(prompt_template, subject)
    if not valid_prompt(prompt):
        return None

    paraphrases = first_valid_prompts(row.get("paraphrase_prompts"), count=2)
    neighborhoods = first_valid_prompts(row.get("neighborhood_prompts"), count=3)
    generation_prompts = first_valid_prompts(row.get("generation_prompts"), count=2)
    if len(paraphrases) < 2 or len(neighborhoods) < 3:
        return None

    return CounterFactCase(
        row_idx=row_idx,
        case_id=case_id,
        relation_id=relation_id,
        subject=subject,
        prompt=prompt,
        target_new=target_new,
        target_true=target_true,
        paraphrase_prompts=tuple(paraphrases[:2]),  # type: ignore[arg-type]
        neighborhood_prompts=tuple(neighborhoods[:3]),  # type: ignore[arg-type]
        generation_prompts=tuple(generation_prompts),
    )


def materialize_v4_files(
    *,
    repo_root: Path,
    cases: list[CounterFactCase],
    dataset: str,
    config: str,
    source_split: str,
    host_confirm_path: Path,
    pack_ids: CounterFactPackIds = ACTIVE_PACK_IDS,
) -> None:
    train_path = repo_root / "data" / "train_generators" / f"{pack_ids.train_generator_version}.yaml"
    dev_path = repo_root / "data" / "packs" / f"{pack_ids.visible_pack_id}.yaml"
    summary_path = repo_root / "data" / "packs" / f"{pack_ids.confirmation_pack_id}.summary.yaml"
    hash_path = repo_root / "data" / "packs" / f"{pack_ids.confirmation_pack_id}.hash"
    host_confirm_path.parent.mkdir(parents=True, exist_ok=True)

    provenance = {
        "source_dataset": dataset,
        "source_config": config,
        "source_split": source_split,
        "source_url": f"https://hf.co/datasets/{dataset}",
        "selection_policy": (
            "first deterministic ASCII-valid cases with simple short answers, "
            "preferring one case per relation_id before filling reserves"
        ),
        "scale_label": pack_ids.scale_label,
        "case_count": len(cases),
        "example_count": len(cases) * 3,
        "selected_case_ids": [case.case_id for case in cases],
    }

    write_json(train_path, build_train_generator(cases, provenance=provenance, pack_ids=pack_ids))
    write_json(dev_path, build_visible_pack(cases, provenance=provenance, pack_ids=pack_ids))
    confirmation_pack = build_confirmation_pack(cases, provenance=provenance, pack_ids=pack_ids)
    write_json(host_confirm_path, confirmation_pack)
    pack_hash = sha256_file(host_confirm_path)
    write_json(summary_path, build_confirmation_summary(cases, provenance=provenance, pack_ids=pack_ids))
    write_json(hash_path, build_confirmation_hash(pack_hash, pack_ids=pack_ids))


def build_train_generator(
    cases: list[CounterFactCase],
    *,
    provenance: Mapping[str, Any],
    pack_ids: CounterFactPackIds = ACTIVE_PACK_IDS,
) -> dict[str, Any]:
    episodes = []
    for index, case in enumerate(cases, start=1):
        episodes.append(
            {
                "episode_id": f"counterfact-case-{case.case_id}-rewrite",
                "update_text": statement(case.prompt, case.target_new),
                "target_text": case.target_new,
                "anchor_texts": [
                    statement(case.neighborhood_prompts[0], case.target_true),
                    statement(case.neighborhood_prompts[1], case.target_true),
                    statement(case.neighborhood_prompts[2], case.target_true),
                ],
                "metadata": {
                    **case_metadata(case),
                    "sequence_id": f"counterfact-{case.case_id}",
                    "step_index": 1,
                    "round_index": index,
                    "target_fact_key": f"counterfact:{case.case_id}:rewrite",
                    "target_prompt": case.prompt,
                    "target_prompt_variants": training_target_prompt_variants(case),
                    "retention_prompts": training_anchor_prompts(case),
                    "retention_prompt_variants": training_anchor_prompt_variants(case),
                    "anchor_fact_keys": [
                        f"counterfact:{case.case_id}:neighborhood:{anchor_index}"
                        for anchor_index in range(3)
                    ],
                    "interference_family": f"counterfact:{case.relation_id}",
                },
            }
        )

    return {
        "schema_version": 2,
        "version": pack_ids.train_generator_version,
        "status": pack_ids.train_status,
        "description": (
            f"CounterFact-derived {pack_ids.scale_label} public continual-learning generator for v4. "
            "Each episode applies one requested rewrite; neighborhood anchors "
            "preserve the row's original target_true object."
        ),
        "problem_family": {
            "name": "counterfact_single_fact_rewrites",
            "task_shape": (
                "single-fact knowledge edits with paraphrase generalization and "
                "same-relation neighborhood specificity"
            ),
            "target_answer_shape": "short text object from CounterFact target_new",
        },
        "provenance": dict(provenance),
        "episodes": episodes,
    }


def build_visible_pack(
    cases: list[CounterFactCase],
    *,
    provenance: Mapping[str, Any],
    pack_ids: CounterFactPackIds = ACTIVE_PACK_IDS,
) -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        examples.extend(public_examples_for_case(case, case_index=index))

    return {
        "schema_version": 2,
        "pack_id": pack_ids.visible_pack_id,
        "visibility": "public_visible_dev",
        "status": pack_ids.visible_status,
        "purpose": (
            f"Visible development pack for the v4 CounterFact {pack_ids.scale_label} substrate. "
            "Targets score edited facts under held-out paraphrases; anchors score "
            "same-relation neighborhood specificity against target_true on prompt "
            "surfaces disjoint from training replay prompts."
        ),
        "metadata": {
            "problem_family": "counterfact_single_fact_rewrites",
            "evaluation_contract": {
                "target_probe": "CounterFact rewrite efficacy and paraphrase generalization",
                "anchor_probe": "CounterFact same-relation neighborhood specificity",
                "reported_metric_families": [
                    "target_quality",
                    "interference",
                    "by_probe_family",
                    "retention_by_lag",
                ],
                "scale_label": pack_ids.scale_label,
                "purpose_label": pack_ids.purpose_label,
            },
            "slice_families": [
                "counterfact_rewrite_paraphrase",
                "counterfact_neighborhood_specificity",
                "counterfact_delayed_neighborhood_specificity",
            ],
            "provenance": dict(provenance),
        },
        "examples": examples,
        "notes": [
            "The public pack is derived deterministically from Hugging Face CounterFact rows.",
            "Neighbor anchor targets use requested_rewrite.target_true, following CounterFact specificity construction.",
        ],
    }


def build_confirmation_pack(
    cases: list[CounterFactCase],
    *,
    provenance: Mapping[str, Any],
    pack_ids: CounterFactPackIds = ACTIVE_PACK_IDS,
) -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        examples.extend(locked_examples_for_case(case, case_index=index))
    return {
        "schema_version": 2,
        "pack_id": pack_ids.confirmation_pack_id,
        "visibility": "host_local_locked",
        "metadata": {
            "problem_family": "counterfact_single_fact_rewrites",
            "provenance": dict(provenance),
            "scale_label": pack_ids.scale_label,
            "contract": (
                "Same edited facts as the public train stream, but different "
                "CounterFact prompt surfaces and neighbor anchors."
            ),
        },
        "examples": examples,
    }


def build_confirmation_summary(
    cases: list[CounterFactCase],
    *,
    provenance: Mapping[str, Any],
    pack_ids: CounterFactPackIds = ACTIVE_PACK_IDS,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "pack_id": pack_ids.confirmation_pack_id,
        "visibility": "host_local_locked",
        "artifact_policy": "aggregate_feedback_only",
        "status": pack_ids.confirmation_summary_status,
        "purpose": (
            f"Locked confirmation pack for the v4 CounterFact {pack_ids.scale_label} substrate. "
            "Raw examples remain host-local; public summary exposes only "
            "composition and provenance."
        ),
        "public_composition_summary": {
            "problem_family": "counterfact_single_fact_rewrites",
            "scale_label": pack_ids.scale_label,
            "purpose_label": pack_ids.purpose_label,
            "source_dataset": provenance["source_dataset"],
            "source_config": provenance["source_config"],
            "source_split": provenance["source_split"],
            "case_count": len(cases),
            "example_count": len(cases) * 3,
            "target_invariant": (
                "Every scored target fact is introduced in the public edit stream. "
                "Hiddenness comes from held-out CounterFact paraphrase and prompt surfaces "
                "disjoint from training replay prompts."
            ),
            "slice_families": [
                {
                    "name": "counterfact_rewrite_paraphrase",
                    "count": len(cases),
                },
                {
                    "name": "counterfact_neighborhood_specificity",
                    "count": len(cases),
                },
                {
                    "name": "counterfact_delayed_neighborhood_specificity",
                    "count": len(cases),
                },
            ],
            "exact_items_exposed_in_repo": False,
        },
        "notes": [
            "The underlying manifest is host-local and should not be checked into the repository.",
            "Neighbor anchor targets are the original CounterFact target_true object.",
            "Locked prompts are exact-string disjoint from training and visible-dev prompts.",
            "Agents may receive aggregate confirmation feedback only, not raw locked examples.",
        ],
    }


def build_confirmation_hash(
    pack_hash: str,
    *,
    pack_ids: CounterFactPackIds = ACTIVE_PACK_IDS,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pack_id": pack_ids.confirmation_pack_id,
        "hash_algorithm": "sha256",
        "hash_status": "host_local_registered",
        "pack_hash": pack_hash,
        "notes": [
            "This hash corresponds to the host-local confirmation manifest kept outside the committed repository.",
            "Only the hash and public summary are exposed in the repo; raw confirmation examples remain host-local.",
        ],
    }


def public_examples_for_case(case: CounterFactCase, *, case_index: int) -> list[dict[str, Any]]:
    base = base_example_metadata(case, case_index=case_index)
    return [
        {
            "example_id": f"counterfact-{case.case_id}-rewrite-paraphrase-visible",
            "prompt": case.paraphrase_prompts[0],
            "target": case.target_new,
            "anchor_prompt": case.neighborhood_prompts[0],
            "anchor_target": case.target_true,
            "metadata": {
                **base,
                "probe_family": "counterfact_rewrite_paraphrase",
                "counterfact_probe_kind": "rewrite_paraphrase",
                "delay_since_last_relevant_update": 1,
            },
        },
        {
            "example_id": f"counterfact-{case.case_id}-neighborhood-visible",
            "prompt": case.paraphrase_prompts[0],
            "target": case.target_new,
            "anchor_prompt": case.neighborhood_prompts[1],
            "anchor_target": case.target_true,
            "metadata": {
                **base,
                "probe_family": "counterfact_neighborhood_specificity",
                "counterfact_probe_kind": "neighborhood_specificity",
                "delay_since_last_relevant_update": 0,
            },
        },
        {
            "example_id": f"counterfact-{case.case_id}-delayed-neighborhood-visible",
            "prompt": case.paraphrase_prompts[0],
            "target": case.target_new,
            "anchor_prompt": case.neighborhood_prompts[2],
            "anchor_target": case.target_true,
            "metadata": {
                **base,
                "probe_family": "counterfact_delayed_neighborhood_specificity",
                "counterfact_probe_kind": "delayed_neighborhood_specificity",
                "delay_since_last_relevant_update": 3,
            },
        },
    ]


def locked_examples_for_case(case: CounterFactCase, *, case_index: int) -> list[dict[str, Any]]:
    base = base_example_metadata(case, case_index=case_index)
    return [
        {
            "example_id": f"counterfact-{case.case_id}-rewrite-paraphrase-locked",
            "prompt": case.paraphrase_prompts[1],
            "target": case.target_new,
            "anchor_prompt": locked_anchor_prompt(case.neighborhood_prompts[1]),
            "anchor_target": case.target_true,
            "metadata": {
                **base,
                "probe_family": "counterfact_rewrite_paraphrase",
                "counterfact_probe_kind": "rewrite_paraphrase",
                "delay_since_last_relevant_update": 2,
            },
        },
        {
            "example_id": f"counterfact-{case.case_id}-neighborhood-locked",
            "prompt": case.paraphrase_prompts[1],
            "target": case.target_new,
            "anchor_prompt": locked_anchor_prompt(case.neighborhood_prompts[2]),
            "anchor_target": case.target_true,
            "metadata": {
                **base,
                "probe_family": "counterfact_neighborhood_specificity",
                "counterfact_probe_kind": "neighborhood_specificity",
                "delay_since_last_relevant_update": 0,
            },
        },
        {
            "example_id": f"counterfact-{case.case_id}-delayed-neighborhood-locked",
            "prompt": case.paraphrase_prompts[1],
            "target": case.target_new,
            "anchor_prompt": locked_anchor_prompt(case.neighborhood_prompts[0]),
            "anchor_target": case.target_true,
            "metadata": {
                **base,
                "probe_family": "counterfact_delayed_neighborhood_specificity",
                "counterfact_probe_kind": "delayed_neighborhood_specificity",
                "delay_since_last_relevant_update": 4,
            },
        },
    ]


def training_target_prompt_variants(case: CounterFactCase) -> list[str]:
    held_out = {case.prompt, *case.paraphrase_prompts}
    variants: list[str] = []
    for prompt in case.generation_prompts:
        stripped = prompt.strip()
        if stripped and stripped not in held_out and stripped not in variants:
            variants.append(stripped)
    return variants


def training_anchor_prompts(case: CounterFactCase) -> list[str]:
    return [f"Training neighbor check: {prompt}" for prompt in case.neighborhood_prompts]


def training_anchor_prompt_variants(case: CounterFactCase) -> list[list[str]]:
    return [
        [
            f"Related fact answer: {prompt}",
            f"Answer the nearby CounterFact prompt: {prompt}",
        ]
        for prompt in case.neighborhood_prompts
    ]


def locked_anchor_prompt(prompt: str) -> str:
    return f"Protected neighbor check: {prompt}"


def base_example_metadata(case: CounterFactCase, *, case_index: int) -> dict[str, Any]:
    return {
        **case_metadata(case),
        "sequence_id": f"counterfact-{case.case_id}",
        "case_index": case_index,
    }


def case_metadata(case: CounterFactCase) -> dict[str, Any]:
    return {
        "source_dataset": DEFAULT_DATASET,
        "source_case_id": case.case_id,
        "source_row_idx": case.row_idx,
        "relation_id": case.relation_id,
        "subject": case.subject,
        "target_true": case.target_true,
        "target_new": case.target_new,
    }


def target_string(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return ""
    return string_value(payload.get("str"))


def string_value(value: Any) -> str:
    return clean_space(str(value)) if isinstance(value, str) else ""


def render_counterfact_prompt(prompt_template: str, subject: str) -> str:
    if "{}" in prompt_template:
        return clean_space(prompt_template.replace("{}", subject))
    if subject and subject not in prompt_template:
        return clean_space(f"{prompt_template} {subject}")
    return clean_space(prompt_template)


def statement(prompt: str, answer: str) -> str:
    text = clean_space(f"{prompt.rstrip()} {answer.strip()}")
    if text.endswith((".", "?", "!")):
        return text
    return f"{text}."


def first_valid_prompts(payload: Any, *, count: int) -> list[str]:
    if not isinstance(payload, list):
        return []
    prompts: list[str] = []
    seen: set[str] = set()
    for item in payload:
        prompt = clean_space(str(item)) if isinstance(item, str) else ""
        if prompt in seen or not valid_prompt(prompt):
            continue
        seen.add(prompt)
        prompts.append(prompt)
        if len(prompts) >= count:
            break
    return prompts


def valid_answer(value: str) -> bool:
    return valid_short_text(value, max_words=4, max_chars=48)


def valid_short_text(value: str, *, max_words: int, max_chars: int) -> bool:
    if not value or len(value) > max_chars or not is_ascii(value):
        return False
    if len(value.split()) > max_words:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 '&,.-]*", value))


def valid_prompt(value: str) -> bool:
    if not value or "{}" in value or len(value) > 180 or not is_ascii(value):
        return False
    return any(character.isalpha() for character in value)


def clean_space(value: str) -> str:
    return " ".join(value.replace("\u00a0", " ").split())


def is_ascii(value: str) -> bool:
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
