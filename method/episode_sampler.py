"""Deterministic episode loading and sampling for bounded runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class Episode:
    """One update episode for the bounded continual-learning loop."""

    episode_id: str
    update_text: str
    target_text: str
    anchor_texts: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GeneratorSpec:
    """Loaded training-episode generator specification."""

    schema_version: int
    version: str
    episodes: tuple[Episode, ...]


def load_generator_spec(path: str | Path) -> GeneratorSpec:
    """Load a JSON-shaped training generator specification."""

    payload = json.loads(Path(path).read_text())
    schema_version = int(payload.get("schema_version", 1))
    version = str(payload.get("version", "")).strip()
    if not version:
        raise ValueError("generator spec must include a non-empty version")

    episodes_payload = payload.get("episodes")
    if not isinstance(episodes_payload, list) or not episodes_payload:
        raise ValueError("generator spec must contain a non-empty episodes list")

    episodes = tuple(normalize_episode(item) for item in episodes_payload)
    return GeneratorSpec(
        schema_version=schema_version,
        version=version,
        episodes=episodes,
    )


class DeterministicEpisodeSampler:
    """Stable, seedable episode sampler for serial harness work."""

    def __init__(self, spec: GeneratorSpec, *, seed: int = 0, shuffle: bool = True) -> None:
        self.spec = spec
        self.seed = seed
        self.shuffle = shuffle

    def iter_episodes(self, *, limit: int | None = None, epoch: int = 0) -> Iterator[Episode]:
        indices = list(range(len(self.spec.episodes)))
        if self.shuffle:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(indices)

        if limit is not None:
            indices = indices[:limit]

        for index in indices:
            yield self.spec.episodes[index]


def normalize_episode(payload: Any) -> Episode:
    """Validate and materialize one episode payload."""

    if not isinstance(payload, dict):
        raise ValueError("each episode entry must be a mapping")

    episode_id = str(payload.get("episode_id", "")).strip()
    update_text = str(payload.get("update_text", "")).strip()
    target_text = str(payload.get("target_text", "")).strip()
    anchor_texts = payload.get("anchor_texts", [])
    metadata = payload.get("metadata", {})

    if not episode_id:
        raise ValueError("episode_id must be a non-empty string")
    if not update_text:
        raise ValueError("update_text must be a non-empty string")
    if not target_text:
        raise ValueError("target_text must be a non-empty string")
    if not isinstance(anchor_texts, list):
        raise ValueError("anchor_texts must be a list")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")

    return Episode(
        episode_id=episode_id,
        update_text=update_text,
        target_text=target_text,
        anchor_texts=tuple(str(item) for item in anchor_texts),
        metadata=metadata,
    )
