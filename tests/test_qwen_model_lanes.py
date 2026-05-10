from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from method.capacity import declared_trainable_parameter_count
from method.adapter_surface import get_surface
from method.selected_stack import load_champion_lane, load_default_surrogate_lane
from scripts.research_state import champion_needs_refresh, choose_next_branch
from scripts.run_loop import champion_lane_reference_context, lane_for_branch, surrogate_gate_outcome
from scripts.run_mainline_hyper_lora import mainline_surface_name


class QwenModelLaneTests(unittest.TestCase):
    def test_active_lanes_use_qwen_family_and_hybrid_surface(self) -> None:
        surrogate = load_default_surrogate_lane(REPO_ROOT)
        champion = load_champion_lane(REPO_ROOT)

        self.assertEqual(surrogate["model_id"], "Qwen/Qwen3.5-0.8B-Base")
        self.assertEqual(champion["model_id"], "Qwen/Qwen3.5-4B-Base")
        self.assertEqual(surrogate["surface_name"], "qwen35_top8_hybrid_attention_mlp")
        self.assertEqual(champion["surface_name"], "qwen35_top8_hybrid_attention_mlp")
        self.assertEqual(surrogate["decision_mode"], "surrogate_screen")
        self.assertEqual(champion["decision_mode"], "champion_bootstrap_or_compare")

    def test_qwen_hybrid_surface_targets_linear_and_full_attention_names(self) -> None:
        surface = get_surface("qwen35_top4_hybrid_attention")

        self.assertIn("in_proj_qkv", surface.target_modules)
        self.assertIn("out_proj", surface.target_modules)
        self.assertIn("q_proj", surface.target_modules)
        self.assertEqual(surface.layer_count, 4)

    def test_qwen_wide_surface_targets_suffix_attention_and_mlp_names(self) -> None:
        surface = get_surface("qwen35_top8_hybrid_attention_mlp")

        self.assertIn("in_proj_qkv", surface.target_modules)
        self.assertIn("out_proj", surface.target_modules)
        self.assertIn("gate_proj", surface.target_modules)
        self.assertIn("up_proj", surface.target_modules)
        self.assertIn("down_proj", surface.target_modules)
        self.assertEqual(surface.layer_count, 8)

    def test_legacy_gemma_champion_context_is_stale_under_qwen_lane(self) -> None:
        champion = {
            "current_champion": {
                "artifact_path": "artifacts/runs/bootstrap-refresh-20260509T201943Z/artifact.json",
                "run_id": "bootstrap-refresh-20260509T201943Z",
            }
        }

        self.assertTrue(champion_needs_refresh(REPO_ROOT, champion))

    def test_sealed_workspace_can_use_committed_champion_card_without_artifact(self) -> None:
        champion = {
            "current_champion": {
                "artifact_path": "artifacts/runs/not-present-in-sealed-copy/artifact.json",
                "base_model": "Qwen/Qwen3.5-4B-Base",
                "run_id": "baseline-20260510T133850Z",
            }
        }

        self.assertFalse(champion_needs_refresh(REPO_ROOT, champion))

    def test_surrogate_graduation_uses_bootstrap_ref_when_champion_lane_is_stale(self) -> None:
        champion = {
            "current_champion": {
                "artifact_path": "artifacts/runs/bootstrap-refresh-20260509T201943Z/artifact.json",
                "run_id": "bootstrap-refresh-20260509T201943Z",
            }
        }

        context = champion_lane_reference_context(REPO_ROOT, champion)

        self.assertEqual(context["baseline_ref"], "bootstrap_baseline")
        self.assertEqual(context["parent_champion"], "no_champion_recorded")

    def test_surrogate_gate_passes_only_as_graduation_signal(self) -> None:
        reasons: list[str] = []
        triggers: list[str] = []
        artifact = {
            "metrics": {
                "target_quality": {"exact_match": 0.7},
                "interference": {
                    "anchor_exact_match": 0.5,
                    "joint_success_rate": 0.35,
                },
            }
        }
        lane = load_default_surrogate_lane(REPO_ROOT)

        outcome = surrogate_gate_outcome(
            artifact=artifact,
            model_lane=lane,
            reasons=reasons,
            triggers=triggers,
        )

        self.assertEqual(outcome, "surrogate_pass")
        self.assertIn("champion_lane_run_required", triggers)
        self.assertIn("surrogate_screen_passed_champion_run_required", reasons)

    def test_active_mainline_branch_has_qwen_wide_capacity(self) -> None:
        champion = {
            "current_champion": {
                "artifact_path": "artifacts/runs/baseline-20260510T133850Z/artifact.json",
                "base_model": "Qwen/Qwen3.5-4B-Base",
                "run_id": "baseline-20260510T133850Z",
            }
        }

        branch = choose_next_branch(REPO_ROOT, champion)
        lane = lane_for_branch(branch, bootstrap=False)
        capacity = declared_trainable_parameter_count(
            method_family=str(branch["method_family"]),
            base_model=str(lane["model_id"]),
            surface_name=str(lane["surface_name"]),
        )

        self.assertEqual(branch["method_family"], "hyper_lora_v0")
        self.assertEqual(lane["name"], "qwen35_surrogate_wide")
        self.assertEqual(capacity, 5529352)

    def test_hyper_lora_qwen_wide_champion_capacity_is_registered(self) -> None:
        capacity = declared_trainable_parameter_count(
            method_family="hyper_lora_v0",
            base_model="Qwen/Qwen3.5-4B-Base",
            surface_name="qwen35_top8_hybrid_attention_mlp",
        )

        self.assertEqual(capacity, 13520648)

    def test_mainline_runner_uses_spec_model_lane_surface(self) -> None:
        self.assertEqual(
            mainline_surface_name(
                {
                    "model_lane": {
                        "surface_name": "qwen35_top8_hybrid_attention_mlp",
                    }
                }
            ),
            "qwen35_top8_hybrid_attention_mlp",
        )


if __name__ == "__main__":
    unittest.main()
