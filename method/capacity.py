"""Declared trainable-capacity registry for frozen run specs."""

from __future__ import annotations


_SELECTED_STACK_COUNTS = {
    (
        "baseline_seq_lora_ft_v0",
        "google/gemma-3-4b-pt",
        "top4_standard",
    ): 524288,
    (
        "baseline_seq_lora_ft_v1_state_replay",
        "google/gemma-3-4b-pt",
        "top4_standard",
    ): 524288,
    (
        "baseline_seq_lora_ft_v2_rank16_target_replay",
        "google/gemma-3-4b-pt",
        "top4_standard",
    ): 1048576,
    (
        "baseline_seq_lora_ft_v3_rank16_target_rehearsal",
        "google/gemma-3-4b-pt",
        "top4_standard",
    ): 1048576,
    (
        "baseline_seq_lora_ft_v4_rank16_final_consolidation",
        "google/gemma-3-4b-pt",
        "top4_standard",
    ): 1048576,
    (
        "baseline_seq_lora_ft_v5_rank32_mixed_rehearsal",
        "google/gemma-3-4b-pt",
        "top4_standard",
    ): 2097152,
    (
        "baseline_seq_lora_ft_v6_rank32_relation_replay",
        "google/gemma-3-4b-pt",
        "top4_standard",
    ): 2097152,
    (
        "baseline_seq_lora_ft_v6_rank32_relation_replay_qwen35",
        "Qwen/Qwen3.5-0.8B-Base",
        "qwen35_top4_hybrid_attention",
    ): 1343488,
    (
        "baseline_seq_lora_ft_v6_rank32_relation_replay_qwen35",
        "Qwen/Qwen3.5-4B-Base",
        "qwen35_top4_hybrid_attention",
    ): 2457600,
    (
        "baseline_seq_lora_ft_v7_qwen35_coverage_first",
        "Qwen/Qwen3.5-0.8B-Base",
        "qwen35_top4_hybrid_attention",
    ): 1343488,
    (
        "baseline_seq_lora_ft_v7_qwen35_coverage_first",
        "Qwen/Qwen3.5-4B-Base",
        "qwen35_top4_hybrid_attention",
    ): 2457600,
    (
        "baseline_seq_lora_ft_v8_qwen35_target_last_coverage",
        "Qwen/Qwen3.5-0.8B-Base",
        "qwen35_top4_hybrid_attention",
    ): 1343488,
    (
        "baseline_seq_lora_ft_v8_qwen35_target_last_coverage",
        "Qwen/Qwen3.5-4B-Base",
        "qwen35_top4_hybrid_attention",
    ): 2457600,
    (
        "baseline_seq_lora_ft_v9_qwen35_wide_suffix_recall",
        "Qwen/Qwen3.5-0.8B-Base",
        "qwen35_top8_hybrid_attention_mlp",
    ): 6225920,
    (
        "baseline_seq_lora_ft_v9_qwen35_wide_suffix_recall",
        "Qwen/Qwen3.5-4B-Base",
        "qwen35_top8_hybrid_attention_mlp",
    ): 13959168,
    (
        "baseline_seq_lora_ft_v10_qwen35_wide_stable_rehearsal",
        "Qwen/Qwen3.5-0.8B-Base",
        "qwen35_top8_hybrid_attention_mlp",
    ): 3112960,
    (
        "baseline_seq_lora_ft_v10_qwen35_wide_stable_rehearsal",
        "Qwen/Qwen3.5-4B-Base",
        "qwen35_top8_hybrid_attention_mlp",
    ): 6979584,
    (
        "baseline_seq_lora_ft_v11_qwen35_wide_fact_replay",
        "Qwen/Qwen3.5-0.8B-Base",
        "qwen35_top8_hybrid_attention_mlp",
    ): 3112960,
    (
        "baseline_seq_lora_ft_v11_qwen35_wide_fact_replay",
        "Qwen/Qwen3.5-4B-Base",
        "qwen35_top8_hybrid_attention_mlp",
    ): 6979584,
    (
        "hyper_lora_v0",
        "google/gemma-3-4b-pt",
        "top4_standard",
    ): 1919824,
    (
        "hyper_lora_v0",
        "Qwen/Qwen3.5-0.8B-Base",
        "qwen35_top8_hybrid_attention_mlp",
    ): 5529352,
    (
        "hyper_lora_v0",
        "Qwen/Qwen3.5-4B-Base",
        "qwen35_top8_hybrid_attention_mlp",
    ): 13520648,
}


def declared_trainable_parameter_count(
    *,
    method_family: str,
    base_model: str,
    surface_name: str,
) -> int:
    """Return the pre-registered trainable capacity for one launch-stack method."""

    key = (method_family, base_model, surface_name)
    try:
        return _SELECTED_STACK_COUNTS[key]
    except KeyError as exc:
        known = ", ".join(
            f"{method}/{model}/{surface}"
            for method, model, surface in sorted(_SELECTED_STACK_COUNTS)
        )
        raise KeyError(
            "no declared trainable-capacity record for "
            f"{method_family}/{base_model}/{surface_name}; known records: {known}"
        ) from exc
