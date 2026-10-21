"""Serving checks for the local teacher (osler_v4 Y0): synthetic prompts, rate arithmetic, reasoning fields. No network."""

from __future__ import annotations

from meddecide.teacher.serving import RunStats, reasoning_text, run_stats, synthetic_record


def test_synthetic_record_is_deterministic_and_distinct_per_seed() -> None:
    assert synthetic_record(500, seed=1) == synthetic_record(500, seed=1)
    assert synthetic_record(500, seed=1) != synthetic_record(500, seed=2)


def test_synthetic_record_has_the_requested_length() -> None:
    assert len(synthetic_record(3000, seed=7)) == 3000


def test_run_stats_sums_usage_and_counts_failures_in_the_denominator() -> None:
    usages = [{"prompt_tokens": 100, "completion_tokens": 600}, {"prompt_tokens": 50, "completion_tokens": 0}]
    stats = run_stats(usages, failed=2, wall_s=60.0)
    assert stats.requests == 4
    assert stats.prompt_tokens == 150
    assert stats.completion_tokens == 600
    assert stats.output_tokens_per_s == 10.0
    assert stats.requests_per_hour == 2 * 3600.0 / 60.0


def test_zero_wall_time_gives_zero_rates_not_a_division_error() -> None:
    stats = RunStats(requests=1, failed=0, wall_s=0.0, prompt_tokens=1, completion_tokens=1)
    assert stats.output_tokens_per_s == 0.0
    assert stats.requests_per_hour == 0.0


def test_reasoning_text_reads_either_vllm_field_and_is_empty_when_absent() -> None:
    assert reasoning_text({"content": "yes", "reasoning_content": "step 1"}) == "step 1"
    assert reasoning_text({"content": "yes", "reasoning": "step 2"}) == "step 2"
    assert reasoning_text({"content": "yes", "reasoning_content": None}) == ""
    assert reasoning_text({"content": "yes"}) == ""
