from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.runner import load_eval_batch
from method import load_generator_spec
from scripts.decide import evaluate_baseline_acceptance_tier, evaluate_bootstrap_seed_gates
from scripts.materialize_counterfact_v4 import (
    build_confirmation_pack,
    build_train_generator,
    build_visible_pack,
    materialize_v4_files,
    select_cases,
)


class CounterFactV4Tests(unittest.TestCase):
    def test_materializer_converts_counterfact_rows_without_extra_dependencies(self) -> None:
        cases = select_cases([counterfact_fixture_row()], count=1)

        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case.case_id, 100)
        self.assertEqual(case.prompt, "The capital of Freedonia is")
        self.assertEqual(case.target_new, "Paris")
        self.assertEqual(case.target_true, "Prague")
        self.assertEqual(case.neighborhood_prompts[0], "The capital of Bohemia is")
        self.assertEqual(case.generation_prompts[0], "Freedonia's capital city is")

    def test_materialized_v4_files_load_through_existing_harness_loaders(self) -> None:
        generator = load_generator_spec(
            REPO_ROOT
            / "data"
            / "train_generators"
            / "cl_seq_train_v4_counterfact_standard.yaml"
        )
        visible = load_eval_batch(
            REPO_ROOT / "data" / "packs" / "cl_dev_visible_v4_counterfact_standard.yaml"
        )
        summary = load_json(
            REPO_ROOT
            / "data"
            / "packs"
            / "cl_confirm_locked_v4_counterfact_standard.summary.yaml"
        )
        registered_hash = load_json(
            REPO_ROOT / "data" / "packs" / "cl_confirm_locked_v4_counterfact_standard.hash"
        )
        smoke_generator = load_generator_spec(
            REPO_ROOT
            / "data"
            / "train_generators"
            / "cl_seq_train_v4_counterfact_smoke.yaml"
        )
        smoke_visible = load_eval_batch(
            REPO_ROOT / "data" / "packs" / "cl_dev_visible_v4_counterfact_smoke.yaml"
        )
        smoke_summary = load_json(
            REPO_ROOT
            / "data"
            / "packs"
            / "cl_confirm_locked_v4_counterfact_smoke.summary.yaml"
        )

        self.assertEqual(generator.version, "cl_seq_train_v4_counterfact_standard")
        self.assertEqual(len(generator.episodes), 96)
        self.assertEqual(visible.pack_id, "cl_dev_visible_v4_counterfact_standard")
        self.assertEqual(len(visible.examples), 288)
        self.assertEqual(summary["public_composition_summary"]["case_count"], 96)
        self.assertEqual(summary["public_composition_summary"]["scale_label"], "standard")
        self.assertEqual(
            registered_hash["pack_id"],
            "cl_confirm_locked_v4_counterfact_standard",
        )
        self.assertRegex(registered_hash["pack_hash"], r"^[a-f0-9]{64}$")
        self.assertEqual(smoke_generator.version, "cl_seq_train_v4_counterfact_smoke")
        self.assertEqual(len(smoke_generator.episodes), 12)
        self.assertEqual(smoke_visible.pack_id, "cl_dev_visible_v4_counterfact_smoke")
        self.assertEqual(len(smoke_visible.examples), 36)
        self.assertEqual(smoke_summary["public_composition_summary"]["scale_label"], "smoke")

    def test_materialized_prompt_surfaces_are_disjoint_across_train_visible_and_locked(self) -> None:
        generator = load_json(
            REPO_ROOT
            / "data"
            / "train_generators"
            / "cl_seq_train_v4_counterfact_standard.yaml"
        )
        visible = load_json(
            REPO_ROOT / "data" / "packs" / "cl_dev_visible_v4_counterfact_standard.yaml"
        )
        host_pack = Path(
            "~/shared/artifacts/autoresearch-continual-learning/protected/"
            "cl_confirm_locked_v4_counterfact_standard.json"
        ).expanduser()
        if not host_pack.exists():
            self.skipTest(f"host-local CounterFact v4 pack is not present: {host_pack}")
        locked = load_json(host_pack)

        train_prompts = replayable_training_prompts(generator)
        visible_prompts = eval_prompts(visible)
        locked_prompts = eval_prompts(locked)

        self.assertFalse(train_prompts & visible_prompts)
        self.assertFalse(train_prompts & locked_prompts)
        self.assertFalse(visible_prompts & locked_prompts)

    def test_built_prompt_surfaces_are_disjoint_for_fixture_case(self) -> None:
        cases = select_cases([counterfact_fixture_row()], count=1)
        generator = build_train_generator(cases, provenance={})
        visible = build_visible_pack(cases, provenance={})
        locked = build_confirmation_pack(cases, provenance={})

        train_prompts = replayable_training_prompts(generator)
        visible_prompts = eval_prompts(visible)
        locked_prompts = eval_prompts(locked)

        self.assertFalse(train_prompts & visible_prompts)
        self.assertFalse(train_prompts & locked_prompts)
        self.assertFalse(visible_prompts & locked_prompts)

    def test_host_local_confirmation_hash_matches_when_pack_is_present(self) -> None:
        for pack_id in (
            "cl_confirm_locked_v4_counterfact_standard",
            "cl_confirm_locked_v4_counterfact",
            "cl_confirm_locked_v4_counterfact_smoke",
        ):
            with self.subTest(pack_id=pack_id):
                host_pack = Path(
                    "~/shared/artifacts/autoresearch-continual-learning/protected/"
                    f"{pack_id}.json"
                ).expanduser()
                if not host_pack.exists():
                    self.skipTest(f"host-local CounterFact v4 pack is not present: {pack_id}")

                registered_hash = load_json(
                    REPO_ROOT / "data" / "packs" / f"{pack_id}.hash"
                )
                actual_hash = hashlib.sha256(host_pack.read_bytes()).hexdigest()

                self.assertEqual(actual_hash, registered_hash["pack_hash"])
                self.assertEqual(load_json(host_pack)["pack_id"], pack_id)

    def test_counterfact_probe_aliases_satisfy_existing_bootstrap_and_acceptance_gates(self) -> None:
        config = {
            "bootstrap_seed_gates": {
                "target_exact_min_abs": 0.35,
                "target_exact_delta_from_base": 0.25,
                "latest_value_probe_families": ["counterfact_rewrite_paraphrase"],
                "latest_value_exact_min": 0.4,
                "anchor_exact_min": 0.25,
                "visible_confirm_gap_max": 0.25,
            },
            "baseline_acceptance_gates": {
                "default_tier": "provisional",
                "accepted_tier": "accepted",
                "confirmation_target_exact_min": 0.75,
                "latest_value_probe_families": ["counterfact_rewrite_paraphrase"],
                "latest_value_target_exact_min": 0.75,
                "anchor_exact_min": 0.4167,
                "neighbor_anchor_probe_families": ["counterfact_neighborhood_specificity"],
                "neighbor_anchor_exact_min": 0.5,
                "delayed_anchor_probe_families": [
                    "counterfact_delayed_neighborhood_specificity"
                ],
                "delayed_anchor_exact_min": 0.25,
                "min_anchor_slice_exact_min": 0.25,
                "require_no_zero_anchor_slice": True,
            },
        }
        artifact = {
            "metrics": {
                "target_quality": {
                    "exact_match": 0.9,
                },
            },
        }
        confirmation = {
            "aggregate_metrics": {
                "target_quality": {
                    "exact_match": 0.9,
                },
                "interference": {
                    "anchor_exact_match": 0.75,
                },
                "by_probe_family": {
                    "counterfact_rewrite_paraphrase": {
                        "target_exact_match": 0.9,
                        "anchor_exact_match": 0.75,
                    },
                    "counterfact_neighborhood_specificity": {
                        "target_exact_match": 0.9,
                        "anchor_exact_match": 0.75,
                    },
                    "counterfact_delayed_neighborhood_specificity": {
                        "target_exact_match": 0.9,
                        "anchor_exact_match": 0.75,
                    },
                },
                "baseline_reference": {
                    "target_quality": {
                        "exact_match": 0.1,
                    },
                    "interference": {
                        "anchor_exact_match": 0.0,
                    },
                },
            },
        }

        seed_eval = evaluate_bootstrap_seed_gates(
            artifact=artifact,
            confirmation_result=confirmation,
            config=config,
        )
        acceptance_eval = evaluate_baseline_acceptance_tier(
            artifact=artifact,
            confirmation_result=confirmation,
            config=config,
        )

        self.assertEqual(seed_eval["outcome"], "pass")
        self.assertEqual(acceptance_eval["outcome"], "pass")
        self.assertEqual(acceptance_eval["tier"], "accepted")

    def test_materializer_writes_public_files_and_host_only_confirmation(self) -> None:
        cases = select_cases([counterfact_fixture_row()], count=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            host_confirm = (
                repo_root / "protected" / "cl_confirm_locked_v4_counterfact_standard.json"
            )
            materialize_v4_files(
                repo_root=repo_root,
                cases=cases,
                dataset="azhx/counterfact",
                config="default",
                source_split="train",
                host_confirm_path=host_confirm,
            )

            self.assertTrue(
                (
                    repo_root
                    / "data"
                    / "train_generators"
                    / "cl_seq_train_v4_counterfact_standard.yaml"
                ).exists()
            )
            self.assertTrue(
                (
                    repo_root
                    / "data"
                    / "packs"
                    / "cl_confirm_locked_v4_counterfact_standard.summary.yaml"
                ).exists()
            )
            self.assertFalse(
                (
                    repo_root
                    / "data"
                    / "packs"
                    / "cl_confirm_locked_v4_counterfact_standard.json"
                ).exists()
            )
            self.assertEqual(
                build_confirmation_pack(cases, provenance={})["pack_id"],
                "cl_confirm_locked_v4_counterfact_standard",
            )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def replayable_training_prompts(generator: dict) -> set[str]:
    prompts: set[str] = set()
    for episode in generator.get("episodes", []):
        metadata = episode.get("metadata", {})
        for field in ("target_prompt", "retention_prompts"):
            value = metadata.get(field)
            if isinstance(value, str) and value.strip():
                prompts.add(value.strip())
            elif isinstance(value, list):
                prompts.update(item.strip() for item in value if isinstance(item, str) and item.strip())
        variants = metadata.get("target_prompt_variants")
        if isinstance(variants, list):
            prompts.update(item.strip() for item in variants if isinstance(item, str) and item.strip())
        retention_variants = metadata.get("retention_prompt_variants")
        if isinstance(retention_variants, list):
            for item in retention_variants:
                if isinstance(item, str) and item.strip():
                    prompts.add(item.strip())
                elif isinstance(item, list):
                    prompts.update(
                        subitem.strip()
                        for subitem in item
                        if isinstance(subitem, str) and subitem.strip()
                    )
    return prompts


def eval_prompts(pack: dict) -> set[str]:
    prompts: set[str] = set()
    for example in pack.get("examples", []):
        for field in ("prompt", "anchor_prompt"):
            value = example.get(field)
            if isinstance(value, str) and value.strip():
                prompts.add(value.strip())
    return prompts


def counterfact_fixture_row() -> dict:
    return {
        "row_idx": 7,
        "row": {
            "case_id": 100,
            "requested_rewrite": {
                "prompt": "The capital of {} is",
                "relation_id": "P36",
                "subject": "Freedonia",
                "target_new": {
                    "id": "Q90",
                    "str": "Paris",
                },
                "target_true": {
                    "id": "Q1085",
                    "str": "Prague",
                },
            },
            "paraphrase_prompts": [
                "Freedonia has its capital in",
                "The seat of government for Freedonia is",
            ],
            "neighborhood_prompts": [
                "The capital of Bohemia is",
                "Moravia has its capital in",
                "The seat of government for Silesia is",
            ],
            "attribute_prompts": [],
            "generation_prompts": [
                "Freedonia's capital city is",
                "Freedonia's seat of government is",
            ],
        },
    }


if __name__ == "__main__":
    unittest.main()
