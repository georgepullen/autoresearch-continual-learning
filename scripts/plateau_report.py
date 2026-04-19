#!/usr/bin/env python3
"""Generate bounded research or diagnosis memos from recent decision history."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-ledger",
        default="experiments/ledgers/runs.jsonl",
        help="Path to the runs ledger containing decision records.",
    )
    parser.add_argument(
        "--memo-ledger",
        default="experiments/ledgers/research_memos.jsonl",
        help="Path to the append-only research memo ledger.",
    )
    parser.add_argument("--plateau-window", type=int, default=5)
    parser.add_argument("--confirm-window", type=int, default=3)
    parser.add_argument("--method-family")
    parser.add_argument("--run-class")
    parser.add_argument("--record-ledger", action="store_true")
    parser.add_argument("--include-integrity-pass", action="store_true")
    args = parser.parse_args()

    decision_records = load_decision_records(resolve_path(args.runs_ledger))
    decision_records = filter_records(
        decision_records,
        method_family=args.method_family,
        run_class=args.run_class,
    )

    memos: list[dict[str, Any]] = []
    plateau = detect_plateau(
        decision_records,
        window=args.plateau_window,
        include_integrity_pass=args.include_integrity_pass,
    )
    if plateau is not None:
        memos.append(plateau)

    confirm_pattern = detect_confirm_regression_pattern(
        decision_records,
        window=args.confirm_window,
        include_integrity_pass=args.include_integrity_pass,
    )
    if confirm_pattern is not None:
        memos.append(confirm_pattern)

    if args.record_ledger:
        memo_ledger = resolve_path(args.memo_ledger)
        for memo in memos:
            append_jsonl(memo_ledger, memo)

    print(json.dumps({"memos": memos}, indent=2, sort_keys=True))
    return 0 if memos else 1


def load_decision_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("record_type") == "decision":
            records.append(payload)
    return sorted(records, key=lambda item: item.get("recorded_at_utc", ""))


def filter_records(
    records: Iterable[dict[str, Any]],
    *,
    method_family: str | None,
    run_class: str | None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for record in records:
        if method_family and record.get("method_family") != method_family:
            continue
        if run_class and record.get("run_class") != run_class:
            continue
        filtered.append(record)
    return filtered


def detect_plateau(
    records: list[dict[str, Any]],
    *,
    window: int,
    include_integrity_pass: bool,
) -> dict[str, Any] | None:
    recent = records[-window:]
    if len(recent) < window:
        return None

    outcomes = [record.get("outcome") for record in recent]
    if "promote" in outcomes:
        return None
    if not all(outcome in {"discard", "needs_human_decision"} for outcome in outcomes):
        return None

    dominant_reasons = most_common(flatten_strings(record.get("reasons", []) for record in recent))
    if not dominant_reasons:
        dominant_reasons = [("no_clear_reason_cluster", len(recent))]

    memo = {
        "record_type": "research_memo",
        "memo_type": "plateau",
        "trigger": "plateau",
        "recorded_at_utc": utc_now(),
        "window_size": window,
        "evidence": {
            "run_ids": [record.get("run_id") for record in recent],
            "outcomes": outcomes,
            "dominant_reasons": [
                {"reason": reason, "count": count} for reason, count in dominant_reasons
            ],
        },
        "hypotheses": plateau_hypotheses(dominant_reasons),
        "recommended_next_action": plateau_next_action(dominant_reasons),
        "falsification_evidence": plateau_falsification(dominant_reasons),
    }
    if include_integrity_pass:
        memo["integrity_pass"] = integrity_pass_advisory(recent)
    return memo


def detect_confirm_regression_pattern(
    records: list[dict[str, Any]],
    *,
    window: int,
    include_integrity_pass: bool,
) -> dict[str, Any] | None:
    recent = records[-window:]
    if len(recent) < window:
        return None

    flagged = [
        record
        for record in recent
        if "confirm_regression_pattern" in record.get("triggers", [])
        or "protected_confirmation_failed" in record.get("reasons", [])
    ]
    if len(flagged) < window:
        return None

    dominant_reasons = most_common(flatten_strings(record.get("reasons", []) for record in flagged))
    memo = {
        "record_type": "research_memo",
        "memo_type": "diagnosis_memo",
        "trigger": "confirm_regression_pattern",
        "recorded_at_utc": utc_now(),
        "window_size": window,
        "evidence": {
            "run_ids": [record.get("run_id") for record in flagged],
            "outcomes": [record.get("outcome") for record in flagged],
            "dominant_reasons": [
                {"reason": reason, "count": count} for reason, count in dominant_reasons
            ],
        },
        "hypotheses": [
            "The visible dev pack is rewarding a pattern that does not survive protected confirmation.",
            "The current hypothesis family may be overfitting a narrow surface rather than improving the real objective.",
            "A pack mismatch or evaluation-scope mismatch may be masquerading as method progress."
        ],
        "recommended_next_action": {
            "action": "Run one bounded diagnosis memo before admitting new external novelty.",
            "details": [
                "Inspect the shared signature across recent confirmation failures.",
                "Check whether the same run class, editable surface, or comparison scope is repeatedly implicated.",
                "Keep the next action inside the current protocol and method family unless the diagnosis falsifies that path."
            ]
        },
        "falsification_evidence": [
            "A subsequent candidate in the same method family passes protected confirmation.",
            "A protocol or pack bug explains the repeated confirmation failures better than overfitting does.",
            "A bounded diagnosis run shows the visible-dev signal was not actually misleading."
        ],
    }
    if include_integrity_pass:
        memo["integrity_pass"] = integrity_pass_advisory(flagged)
    return memo


def plateau_hypotheses(dominant_reasons: list[tuple[str, int]]) -> list[str]:
    joined = ", ".join(reason for reason, _ in dominant_reasons)
    return [
        "The current method family may be saturating under the present editable surface and run-class budget.",
        f"The recent failure cluster is dominated by: {joined}.",
        "The next step should change one bounded assumption rather than widening the whole research space."
    ]


def plateau_next_action(dominant_reasons: list[tuple[str, int]]) -> dict[str, Any]:
    top_reason = dominant_reasons[0][0]
    if "no_required_interference_improvement" in top_reason:
        details = [
            "Keep the same method family and run class.",
            "Change one interference-facing mechanism or loss term only.",
            "Do not broaden the benchmark surface or mutate protocol files."
        ]
    elif "cost_envelope" in top_reason:
        details = [
            "Treat efficiency recovery as the next bounded hypothesis.",
            "Reduce runtime or VRAM overhead before adding more method complexity.",
            "Preserve the same comparison scope while testing the narrower cost fix."
        ]
    else:
        details = [
            "Write one bounded diagnosis memo against the dominant reason cluster.",
            "Admit at most one new mechanism into the next cycle.",
            "Require a falsifiable next experiment before widening the method family."
        ]
    return {
        "action": "Generate one bounded research memo.",
        "details": details,
    }


def plateau_falsification(dominant_reasons: list[tuple[str, int]]) -> list[str]:
    top_reason = dominant_reasons[0][0]
    return [
        f"A subsequent candidate breaks the dominant failure cluster ({top_reason}) in the same run class.",
        "A replay within the same method family shows the apparent stall was just variance.",
        "A protected-confirmation-backed improvement appears without widening the research space."
    ]


def integrity_pass_advisory(records: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = flatten_strings(record.get("reasons", []) for record in records)
    if any("confirmation" in reason for reason in reasons):
        return {
            "status": "promotion_justification_weak",
            "notes": [
                "Recent promising candidates did not survive confirmation cleanly.",
                "Prefer diagnosis of the dev/confirmation gap before importing novelty."
            ],
        }
    if any("sentinel_invalidated" in reason for reason in reasons):
        return {
            "status": "integrity_concern",
            "notes": [
                "Recent history contains integrity-style failures; scientific diagnosis should not outrun protocol cleanup."
            ],
        }
    return {
        "status": "no_integrity_objection",
        "notes": [
            "No obvious protocol-drift pattern appears in the inspected decision window."
        ],
    }


def flatten_strings(groups: Iterable[Iterable[str]]) -> list[str]:
    flattened: list[str] = []
    for group in groups:
        for item in group:
            if isinstance(item, str):
                flattened.append(item)
    return flattened


def most_common(values: list[str], limit: int = 4) -> list[tuple[str, int]]:
    return Counter(values).most_common(limit)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
