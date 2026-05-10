"""Durable branch-aware research state helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from method.selected_stack import load_champion_lane


RESEARCH_STATE_ROOT = Path("experiments/research_state")
STARTUP_SUMMARY_PATH = RESEARCH_STATE_ROOT / "startup_summary.yaml"
CHAMPION_CARD_PATH = RESEARCH_STATE_ROOT / "champion_card.yaml"
BRANCHES_DIR = RESEARCH_STATE_ROOT / "branches"


def load_startup_summary(repo_root: Path) -> dict[str, Any]:
    return load_json(repo_root / STARTUP_SUMMARY_PATH)


def save_startup_summary(repo_root: Path, payload: Mapping[str, Any]) -> None:
    write_json(repo_root / STARTUP_SUMMARY_PATH, payload)


def load_champion_card(repo_root: Path) -> dict[str, Any]:
    return load_json(repo_root / CHAMPION_CARD_PATH)


def save_champion_card(repo_root: Path, payload: Mapping[str, Any]) -> None:
    write_json(repo_root / CHAMPION_CARD_PATH, payload)


def load_branch_cards(repo_root: Path) -> dict[str, dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    for path in sorted((repo_root / BRANCHES_DIR).glob("*.yaml")):
        payload = load_json(path)
        branch_id = payload.get("branch_id")
        if isinstance(branch_id, str) and branch_id:
            cards[branch_id] = payload
    return cards


def active_registry_versions(repo_root: Path) -> dict[str, str]:
    registry = load_json(repo_root / "data" / "registry.yaml")
    active = registry.get("active", {})
    return {
        "train_generator_version": str(active.get("train_generator", "")),
        "development_pack": str(active.get("visible_dev_pack", "")),
        "confirmation_pack_summary": str(active.get("confirmation_pack", "")),
    }


def champion_versions_from_artifact(repo_root: Path, artifact_path: str | None) -> dict[str, str] | None:
    if not artifact_path:
        return None
    path = Path(artifact_path)
    if not path.is_absolute():
        path = repo_root / artifact_path
    if not path.exists():
        return None
    payload = load_json(path)
    data = payload.get("data", {})
    return {
        "train_generator_version": str(data.get("train_generator_version", "")),
        "development_pack": str(data.get("development_pack", "")),
        "confirmation_pack_summary": str(data.get("confirmation_pack_summary", "")),
    }


def champion_lane_context_from_artifact(
    repo_root: Path,
    artifact_path: str | None,
) -> dict[str, str] | None:
    if not artifact_path:
        return None
    path = Path(artifact_path)
    if not path.is_absolute():
        path = repo_root / artifact_path
    if not path.exists():
        return None
    payload = load_json(path)
    method = payload.get("method", {})
    comparison = payload.get("comparison", {})
    if not isinstance(method, Mapping) or not isinstance(comparison, Mapping):
        return None
    return {
        "base_model": str(method.get("base_model", "")),
        "comparison_scope": str(comparison.get("comparison_scope", "")),
    }


def champion_needs_refresh(repo_root: Path, champion: Mapping[str, Any]) -> bool:
    current = champion.get("current_champion")
    artifact_path = current.get("artifact_path") if isinstance(current, Mapping) else None
    champion_versions = champion_versions_from_artifact(repo_root, artifact_path)
    if champion_versions is None:
        return True
    if champion_versions != active_registry_versions(repo_root):
        return True
    try:
        champion_lane = load_champion_lane(repo_root)
    except (FileNotFoundError, ValueError, KeyError):
        return False
    champion_context = champion_lane_context_from_artifact(repo_root, artifact_path)
    if champion_context is None:
        return True
    return (
        champion_context.get("base_model") != champion_lane["model_id"]
        or champion_context.get("comparison_scope") != champion_lane["comparison_scope"]
    )


def choose_next_branch(repo_root: Path, champion: Mapping[str, Any]) -> dict[str, Any]:
    startup_summary = load_startup_summary(repo_root)
    branch_cards = load_branch_cards(repo_root)
    if champion_needs_refresh(repo_root, champion):
        return choose_bootstrap_refresh_branch(startup_summary, branch_cards)

    active_branch_id = startup_summary.get("active_branch_id")
    if isinstance(active_branch_id, str) and active_branch_id in branch_cards:
        active_branch = branch_cards[active_branch_id]
        if active_branch.get("status") == "active":
            return active_branch

    hyper_branch = branch_cards.get("branch_hyper_lora_v0")
    if hyper_branch and int(hyper_branch.get("attempt_count", 0)) == 0:
        return hyper_branch

    candidates = [
        card
        for card in branch_cards.values()
        if card.get("status") == "active" and not card.get("retired_at_utc")
    ]
    if not candidates:
        raise RuntimeError("No active branch cards available.")
    return max(candidates, key=lambda card: float(card.get("branch_score", 0.0)))


def update_champion_card(repo_root: Path, champion: Mapping[str, Any]) -> None:
    current = champion.get("current_champion", {})
    payload = load_champion_card(repo_root)
    payload.update(
        {
            "updated_at_utc": utc_now(),
            "current_run_id": current.get("run_id"),
            "status": champion.get("status"),
            "baseline_acceptance_tier": current.get("baseline_acceptance_tier")
            if isinstance(current, Mapping)
            else payload.get("baseline_acceptance_tier"),
            "method_family": current.get("method_family"),
            "comparison_scope": champion.get("current_champion", {}).get("comparison_scope")
            if isinstance(champion.get("current_champion"), Mapping)
            else payload.get("comparison_scope"),
            "active_registry_versions": active_registry_versions(repo_root),
            "current_champion_versions": champion_versions_from_artifact(
                repo_root,
                current.get("artifact_path") if isinstance(current, Mapping) else None,
            )
            or {},
            "target_champion_lane": load_champion_lane(repo_root),
            "refresh_required": champion_needs_refresh(repo_root, champion),
        }
    )
    if payload["refresh_required"]:
        payload["refresh_reason"] = (
            "Champion data surfaces do not match the active CL substrate in data/registry.yaml."
        )
    else:
        payload["refresh_reason"] = None
    save_champion_card(repo_root, payload)


def update_branch_card(
    repo_root: Path,
    *,
    method_family: str | None,
    run_id: str,
    outcome: str,
    confirmation_status: str | None = None,
    baseline_acceptance_tier: str | None = None,
) -> None:
    if not method_family:
        return
    branch_cards = load_branch_cards(repo_root)
    branch = next(
        (card for card in branch_cards.values() if card.get("method_family") == method_family),
        None,
    )
    if branch is None:
        return

    branch["updated_at_utc"] = utc_now()
    branch["last_run_id"] = run_id
    branch["last_outcome"] = outcome
    branch["attempt_count"] = int(branch.get("attempt_count", 0)) + 1
    if outcome == "promote":
        branch["promotion_count"] = int(branch.get("promotion_count", 0)) + 1
        branch["branch_score"] = float(branch.get("branch_score", 0.0)) + 0.2
    elif outcome in {"discard", "invalid"}:
        branch["branch_score"] = max(0.0, float(branch.get("branch_score", 0.0)) - 0.1)

    if confirmation_status == "pass":
        branch["confirmation_pass_count"] = int(branch.get("confirmation_pass_count", 0)) + 1
    elif confirmation_status == "fail":
        branch["confirmation_fail_count"] = int(branch.get("confirmation_fail_count", 0)) + 1
    if outcome == "promote" and baseline_acceptance_tier is not None:
        branch["last_baseline_acceptance_tier"] = baseline_acceptance_tier

    startup_summary = load_startup_summary(repo_root)
    branch_id = str(branch.get("branch_id", ""))
    if (
        outcome == "promote"
        and branch_id.startswith("branch_baseline_seq_lora_ft")
        and branch.get("target_lane") != "qwen35_surrogate"
    ):
        startup_summary["champion_needs_refresh"] = False
        if baseline_acceptance_tier == "accepted":
            startup_summary["active_branch_id"] = "branch_hyper_lora_v0"
        else:
            startup_summary["active_branch_id"] = branch.get("branch_id")
    elif outcome == "promote" and branch.get("target_lane") == "qwen35_surrogate":
        startup_summary["champion_needs_refresh"] = False
        startup_summary["active_branch_id"] = branch.get("branch_id")
    elif outcome == "promote":
        startup_summary["active_branch_id"] = branch.get("branch_id")
    save_startup_summary(repo_root, startup_summary)

    branch_path = repo_root / BRANCHES_DIR / f"{branch['branch_id']}.yaml"
    write_json(branch_path, branch)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def choose_bootstrap_refresh_branch(
    startup_summary: Mapping[str, Any],
    branch_cards: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    priority_order = startup_summary.get("branch_priority_order", [])
    if isinstance(priority_order, list):
        for branch_id in priority_order:
            if not isinstance(branch_id, str):
                continue
            branch = branch_cards.get(branch_id)
            if branch is None:
                continue
            status = branch.get("status")
            method_family = str(branch.get("method_family", ""))
            if status in {"active", "active_reference_failed_to_seed"} and method_family.startswith(
                "baseline_"
            ):
                return branch
    if "branch_baseline_seq_lora_ft_v1_state_replay" in branch_cards:
        return branch_cards["branch_baseline_seq_lora_ft_v1_state_replay"]
    return branch_cards["branch_baseline_seq_lora_ft_v0"]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
