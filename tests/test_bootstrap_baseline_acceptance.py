from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.protected_runner import load_confirmation_result
from method.capacity import declared_trainable_parameter_count
from scripts.run_bootstrap_baseline import (
    ReplayRecord,
    policy_for_method_family,
    run_budget_for_class,
    select_balanced_replay_subset,
)
from scripts.decide import (
    decide,
    evaluate_baseline_acceptance_tier,
    evaluate_bootstrap_seed_gates,
)


class BootstrapBaselineAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = cls._load_json(
            "artifacts/runs/bootstrap-refresh-20260420T225651Z/artifact.json"
        )
        cls.spec = cls._load_json(
            "experiments/specs/bootstrap-refresh-20260420T225651Z.yaml"
        )
        cls.config = cls._load_json("protocol/PROMOTION.yaml")
        cls.bootstrap_config = cls._load_json("protocol/BOOTSTRAP.yaml")
        cls.v3_bootstrap_config = json.loads(json.dumps(cls.bootstrap_config))
        cls.v3_bootstrap_config["bootstrap_lane"].update(
            {
                "allowed_run_classes": ["dev_v1"],
                "required_base_model": "google/gemma-3-4b-pt",
                "comparison_scope": "selected_stack_bootstrap_baseline",
                "required_development_pack": "cl_dev_visible_v3_seen_fact_hidden_surface",
                "required_confirmation_pack_summary": "cl_confirm_locked_v2_seen_fact_hidden_surface",
                "required_train_generator_version": "cl_seq_train_v3_seen_fact_hidden_surface",
            }
        )
        cls.confirmation_result = load_confirmation_result(
            REPO_ROOT
            / "experiments"
            / "confirmation"
            / "results"
            / "bootstrap-refresh-20260420T225651Z.json"
        )
        cls.accepted_artifact = cls._load_json(
            "artifacts/runs/baseline-20260509T094353Z/artifact.json"
        )
        cls.accepted_spec = cls._load_json(
            "experiments/specs/baseline-20260509T094353Z.yaml"
        )
        cls.accepted_confirmation_result = load_confirmation_result(
            REPO_ROOT
            / "experiments"
            / "confirmation"
            / "results"
            / "baseline-20260509T094353Z.json"
        )

    @staticmethod
    def _load_json(relative_path: str) -> dict:
        return json.loads((REPO_ROOT / relative_path).read_text())

    def test_bootstrap_seed_promotes_as_provisional(self) -> None:
        decision = decide(
            artifact=self.artifact,
            spec=self.spec,
            champion_artifact=None,
            confirmation_result=self.confirmation_result,
            config=self.config,
            bootstrap_config=self.v3_bootstrap_config,
        )

        self.assertEqual(decision.outcome, "promote")
        self.assertEqual(decision.baseline_acceptance_tier, "provisional")
        self.assertIn("baseline_acceptance_tier_provisional", decision.reasons)
        self.assertIn(
            "accepted_baseline_delayed_anchor_exact_below_floor:0.0000<0.2500",
            decision.reasons,
        )

    def test_baseline_refresh_promotes_with_accepted_tier(self) -> None:
        decision = decide(
            artifact=self.accepted_artifact,
            spec=self.accepted_spec,
            champion_artifact=self.artifact,
            confirmation_result=self.accepted_confirmation_result,
            config=self.config,
            bootstrap_config=self.bootstrap_config,
        )

        self.assertEqual(decision.outcome, "promote")
        self.assertEqual(decision.baseline_acceptance_tier, "accepted")
        self.assertIn("accepted_baseline_gates_passed", decision.reasons)

    def test_confirmation_quality_gate_rejects_low_locked_pack_target(self) -> None:
        low_confirmation_result = json.loads(json.dumps(self.accepted_confirmation_result))
        low_confirmation_result["aggregate_metrics"]["target_quality"]["exact_match"] = 0.1

        decision = decide(
            artifact=self.accepted_artifact,
            spec=self.accepted_spec,
            champion_artifact=self.artifact,
            confirmation_result=low_confirmation_result,
            config=self.config,
            bootstrap_config=self.bootstrap_config,
        )

        self.assertEqual(decision.outcome, "discard")
        self.assertIn(
            "confirmation_target_exact_below_floor:0.1000<0.7500",
            decision.reasons,
        )
        self.assertIn("confirm_regression_pattern", decision.triggers)

    def test_seed_gate_still_passes_without_accepted_baseline_status(self) -> None:
        seed_eval = evaluate_bootstrap_seed_gates(
            artifact=self.artifact,
            confirmation_result=self.confirmation_result,
            config=self.config,
        )

        self.assertEqual(seed_eval["outcome"], "pass")
        self.assertIn("bootstrap_seed_gates_passed", seed_eval["reasons"])

    def test_acceptance_uses_anchor_metrics_for_locality(self) -> None:
        acceptance_eval = evaluate_baseline_acceptance_tier(
            artifact=self.artifact,
            confirmation_result=self.confirmation_result,
            config=self.config,
        )

        self.assertEqual(acceptance_eval["outcome"], "pass")
        self.assertEqual(acceptance_eval["tier"], "provisional")
        self.assertIn(
            "accepted_baseline_delayed_anchor_exact_below_floor:0.0000<0.2500",
            acceptance_eval["reasons"],
        )
        self.assertIn(
            "accepted_baseline_zero_anchor_slice:delayed_retention_after_unrelated_updates",
            acceptance_eval["reasons"],
        )

    def test_hyper_lora_capacity_matches_observed_completed_run(self) -> None:
        artifact = self._load_json(
            "artifacts/runs/mainline-20260421T084903Z/artifact.json"
        )

        declared = declared_trainable_parameter_count(
            method_family="hyper_lora_v0",
            base_model="google/gemma-3-4b-pt",
            surface_name="top4_standard",
        )

        self.assertEqual(
            declared,
            artifact["observed_capacity"]["observed_trainable_parameter_count"],
        )

    def test_rank16_target_replay_baseline_has_declared_policy_and_capacity(self) -> None:
        for method_family in (
            "baseline_seq_lora_ft_v2_rank16_target_replay",
            "baseline_seq_lora_ft_v3_rank16_target_rehearsal",
            "baseline_seq_lora_ft_v4_rank16_final_consolidation",
        ):
            with self.subTest(method_family=method_family):
                policy = policy_for_method_family(method_family)
                declared = declared_trainable_parameter_count(
                    method_family=method_family,
                    base_model="google/gemma-3-4b-pt",
                    surface_name="top4_standard",
                )

                self.assertTrue(policy.state_consistent_replay)
                self.assertFalse(policy.answer_selection)
                self.assertEqual(policy.lora_rank, 16)
                self.assertEqual(policy.target_repeat_count, 4)
                self.assertEqual(policy.anchor_repeat_count, 1)
                self.assertEqual(declared, 1048576)

        v3_policy = policy_for_method_family(
            "baseline_seq_lora_ft_v3_rank16_target_rehearsal"
        )
        self.assertEqual(v3_policy.target_history_replay_episodes, 24)
        self.assertEqual(v3_policy.anchor_history_replay_episodes, 4)
        self.assertFalse(v3_policy.history_lm_replay)

        v4_policy = policy_for_method_family(
            "baseline_seq_lora_ft_v4_rank16_final_consolidation"
        )
        self.assertEqual(v4_policy.target_history_replay_episodes, 8)
        self.assertEqual(v4_policy.anchor_history_replay_episodes, 8)
        self.assertTrue(v4_policy.final_consolidation)
        self.assertEqual(v4_policy.final_target_passes, 2)
        self.assertEqual(v4_policy.final_anchor_passes, 2)
        self.assertFalse(v4_policy.history_lm_replay)

    def test_rank32_mixed_rehearsal_baseline_has_declared_policy_and_capacity(self) -> None:
        for method_family in (
            "baseline_seq_lora_ft_v5_rank32_mixed_rehearsal",
            "baseline_seq_lora_ft_v6_rank32_relation_replay",
        ):
            with self.subTest(method_family=method_family):
                policy = policy_for_method_family(method_family)
                declared = declared_trainable_parameter_count(
                    method_family=method_family,
                    base_model="google/gemma-3-4b-pt",
                    surface_name="top4_standard",
                )

                self.assertTrue(policy.state_consistent_replay)
                self.assertFalse(policy.answer_selection)
                self.assertEqual(policy.lora_rank, 32)
                self.assertEqual(policy.lora_alpha, 64)
                self.assertTrue(policy.accumulate_supervision)
                self.assertEqual(policy.supervision_batch_size, 24)
                self.assertTrue(policy.mixed_final_consolidation)
                self.assertEqual(declared, 2097152)

        v5_policy = policy_for_method_family(
            "baseline_seq_lora_ft_v5_rank32_mixed_rehearsal"
        )
        self.assertEqual(v5_policy.target_repeat_count, 3)
        self.assertEqual(v5_policy.anchor_repeat_count, 2)
        self.assertFalse(v5_policy.relation_aware_replay)

        v6_policy = policy_for_method_family(
            "baseline_seq_lora_ft_v6_rank32_relation_replay"
        )
        self.assertEqual(v6_policy.target_repeat_count, 4)
        self.assertEqual(v6_policy.anchor_repeat_count, 3)
        self.assertEqual(v6_policy.anchor_history_replay_episodes, 16)
        self.assertTrue(v6_policy.relation_aware_replay)

    def test_qwen35_relation_replay_lane_has_declared_policy_and_capacity(self) -> None:
        policy = policy_for_method_family(
            "baseline_seq_lora_ft_v6_rank32_relation_replay_qwen35"
        )

        surrogate_capacity = declared_trainable_parameter_count(
            method_family="baseline_seq_lora_ft_v6_rank32_relation_replay_qwen35",
            base_model="Qwen/Qwen3.5-0.8B-Base",
            surface_name="qwen35_top4_hybrid_attention",
        )
        champion_capacity = declared_trainable_parameter_count(
            method_family="baseline_seq_lora_ft_v6_rank32_relation_replay_qwen35",
            base_model="Qwen/Qwen3.5-4B-Base",
            surface_name="qwen35_top4_hybrid_attention",
        )

        self.assertTrue(policy.state_consistent_replay)
        self.assertEqual(policy.lora_rank, 32)
        self.assertTrue(policy.relation_aware_replay)
        self.assertEqual(surrogate_capacity, 1343488)
        self.assertEqual(champion_capacity, 2457600)

    def test_qwen35_coverage_first_policy_reduces_replay_fanout(self) -> None:
        policy = policy_for_method_family(
            "baseline_seq_lora_ft_v7_qwen35_coverage_first"
        )

        surrogate_capacity = declared_trainable_parameter_count(
            method_family="baseline_seq_lora_ft_v7_qwen35_coverage_first",
            base_model="Qwen/Qwen3.5-0.8B-Base",
            surface_name="qwen35_top4_hybrid_attention",
        )
        champion_capacity = declared_trainable_parameter_count(
            method_family="baseline_seq_lora_ft_v7_qwen35_coverage_first",
            base_model="Qwen/Qwen3.5-4B-Base",
            surface_name="qwen35_top4_hybrid_attention",
        )

        self.assertTrue(policy.state_consistent_replay)
        self.assertFalse(policy.answer_selection)
        self.assertEqual(policy.lora_rank, 32)
        self.assertEqual(policy.episode_passes, 2)
        self.assertEqual(policy.target_repeat_count, 4)
        self.assertEqual(policy.anchor_repeat_count, 1)
        self.assertEqual(policy.target_history_replay_episodes, 3)
        self.assertEqual(policy.anchor_history_replay_episodes, 3)
        self.assertTrue(policy.relation_aware_replay)
        self.assertEqual(policy.supervision_batch_size, 32)
        self.assertEqual(surrogate_capacity, 1343488)
        self.assertEqual(champion_capacity, 2457600)

    def test_qwen35_target_last_coverage_policy_ends_on_targets(self) -> None:
        policy = policy_for_method_family(
            "baseline_seq_lora_ft_v8_qwen35_target_last_coverage"
        )

        surrogate_capacity = declared_trainable_parameter_count(
            method_family="baseline_seq_lora_ft_v8_qwen35_target_last_coverage",
            base_model="Qwen/Qwen3.5-0.8B-Base",
            surface_name="qwen35_top4_hybrid_attention",
        )
        champion_capacity = declared_trainable_parameter_count(
            method_family="baseline_seq_lora_ft_v8_qwen35_target_last_coverage",
            base_model="Qwen/Qwen3.5-4B-Base",
            surface_name="qwen35_top4_hybrid_attention",
        )

        self.assertTrue(policy.state_consistent_replay)
        self.assertFalse(policy.answer_selection)
        self.assertEqual(policy.lora_rank, 32)
        self.assertEqual(policy.episode_passes, 4)
        self.assertEqual(policy.target_repeat_count, 5)
        self.assertEqual(policy.anchor_repeat_count, 1)
        self.assertEqual(policy.final_target_passes, 2)
        self.assertEqual(policy.final_anchor_passes, 1)
        self.assertFalse(policy.mixed_final_consolidation)
        self.assertEqual(policy.anchor_history_replay_episodes, 2)
        self.assertEqual(surrogate_capacity, 1343488)
        self.assertEqual(champion_capacity, 2457600)

    def test_qwen35_wide_suffix_policy_declares_larger_non_shim_capacity(self) -> None:
        policy = policy_for_method_family(
            "baseline_seq_lora_ft_v9_qwen35_wide_suffix_recall"
        )

        surrogate_capacity = declared_trainable_parameter_count(
            method_family="baseline_seq_lora_ft_v9_qwen35_wide_suffix_recall",
            base_model="Qwen/Qwen3.5-0.8B-Base",
            surface_name="qwen35_top8_hybrid_attention_mlp",
        )
        champion_capacity = declared_trainable_parameter_count(
            method_family="baseline_seq_lora_ft_v9_qwen35_wide_suffix_recall",
            base_model="Qwen/Qwen3.5-4B-Base",
            surface_name="qwen35_top8_hybrid_attention_mlp",
        )

        self.assertTrue(policy.state_consistent_replay)
        self.assertFalse(policy.answer_selection)
        self.assertEqual(policy.lora_rank, 32)
        self.assertEqual(policy.episode_passes, 2)
        self.assertEqual(policy.target_repeat_count, 5)
        self.assertEqual(policy.anchor_repeat_count, 1)
        self.assertEqual(policy.final_target_passes, 3)
        self.assertEqual(policy.final_anchor_passes, 2)
        self.assertFalse(policy.mixed_final_consolidation)
        self.assertEqual(policy.supervision_batch_size, 48)
        self.assertEqual(surrogate_capacity, 6225920)
        self.assertEqual(champion_capacity, 13959168)

    def test_qwen35_wide_stable_policy_reduces_lr_and_balances_rehearsal(self) -> None:
        policy = policy_for_method_family(
            "baseline_seq_lora_ft_v10_qwen35_wide_stable_rehearsal"
        )

        surrogate_capacity = declared_trainable_parameter_count(
            method_family="baseline_seq_lora_ft_v10_qwen35_wide_stable_rehearsal",
            base_model="Qwen/Qwen3.5-0.8B-Base",
            surface_name="qwen35_top8_hybrid_attention_mlp",
        )
        champion_capacity = declared_trainable_parameter_count(
            method_family="baseline_seq_lora_ft_v10_qwen35_wide_stable_rehearsal",
            base_model="Qwen/Qwen3.5-4B-Base",
            surface_name="qwen35_top8_hybrid_attention_mlp",
        )

        self.assertTrue(policy.state_consistent_replay)
        self.assertFalse(policy.answer_selection)
        self.assertEqual(policy.lora_rank, 16)
        self.assertEqual(policy.learning_rate, 1e-4)
        self.assertEqual(policy.target_repeat_count, 4)
        self.assertEqual(policy.anchor_repeat_count, 2)
        self.assertEqual(policy.final_target_passes, 3)
        self.assertEqual(policy.final_anchor_passes, 3)
        self.assertTrue(policy.mixed_final_consolidation)
        self.assertEqual(policy.anchor_history_replay_episodes, 6)
        self.assertEqual(surrogate_capacity, 3112960)
        self.assertEqual(champion_capacity, 6979584)

    def test_qwen35_wide_fact_replay_policy_targets_prompt_transfer(self) -> None:
        policy = policy_for_method_family(
            "baseline_seq_lora_ft_v11_qwen35_wide_fact_replay"
        )

        surrogate_capacity = declared_trainable_parameter_count(
            method_family="baseline_seq_lora_ft_v11_qwen35_wide_fact_replay",
            base_model="Qwen/Qwen3.5-0.8B-Base",
            surface_name="qwen35_top8_hybrid_attention_mlp",
        )
        champion_capacity = declared_trainable_parameter_count(
            method_family="baseline_seq_lora_ft_v11_qwen35_wide_fact_replay",
            base_model="Qwen/Qwen3.5-4B-Base",
            surface_name="qwen35_top8_hybrid_attention_mlp",
        )

        self.assertTrue(policy.state_consistent_replay)
        self.assertFalse(policy.answer_selection)
        self.assertEqual(policy.lora_rank, 16)
        self.assertEqual(policy.learning_rate, 2e-4)
        self.assertEqual(policy.episode_passes, 2)
        self.assertTrue(policy.current_lm_replay)
        self.assertFalse(policy.history_lm_replay)
        self.assertEqual(policy.target_repeat_count, 5)
        self.assertEqual(policy.anchor_repeat_count, 1)
        self.assertEqual(policy.final_target_passes, 2)
        self.assertEqual(policy.final_anchor_passes, 1)
        self.assertTrue(policy.mixed_final_consolidation)
        self.assertEqual(surrogate_capacity, 3112960)
        self.assertEqual(champion_capacity, 6979584)

    def test_replay_selector_spreads_recent_and_old_records(self) -> None:
        records = tuple(
            ReplayRecord(
                key=f"alpha-{index}",
                family="alpha",
                kind="anchor",
                lm_text=f"alpha {index}",
                supervision_examples=((f"prompt {index}", f"answer {index}"),),
                updated_at=index,
            )
            for index in range(6)
        )

        selected = select_balanced_replay_subset(list(records), limit=4)

        self.assertEqual(
            [record.updated_at for record in selected],
            [5, 0, 4, 1],
        )

    def test_dev_run_class_uses_frozen_runtime_envelope(self) -> None:
        budget = run_budget_for_class("dev_v1", episode_count=7)

        self.assertEqual(budget.max_steps, 7)
        self.assertEqual(budget.max_runtime_seconds, 3600.0)


if __name__ == "__main__":
    unittest.main()
