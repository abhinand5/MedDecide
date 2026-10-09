"""Unit tests for the osler_v0 O0 helpers: model-card parsing, length summaries, test denominators."""

from __future__ import annotations

import pytest

from meddecide.bench.denominators import count_test_denominator
from meddecide.eval.model_card import argmax_label, card_blocks, compare_printed
from meddecide.eval.prompt_lengths import summarize_lengths

CARD = """Intro text.

```python
import sys
path = "x"
```

```python
note = "HPI: test"
mj.decide(note, "Is the patient taking X?", ["Yes", "No"], qtype="noul")
# Yes  0.000
# No   1.000
# correct: No
```

```python
mj.decide(None, "Which design?", ["Cohort", "Syndromic"], qtype="choice")
# Cohort       0.064
# Syndromic    0.478
# correct: Cohort
```
"""


def test_card_blocks_reads_printed_values_and_gold_per_block() -> None:
    blocks = card_blocks(CARD)
    assert [b.index for b in blocks] == [0, 1, 2]
    assert blocks[0].printed == {} and blocks[0].correct is None
    assert blocks[1].printed == {"Yes": 0.0, "No": 1.0}
    assert blocks[1].correct == "No"
    assert blocks[2].printed == {"Cohort": 0.064, "Syndromic": 0.478}
    assert blocks[2].correct == "Cohort"


def test_compare_printed_matches_by_prefix_and_reports_worst_gap() -> None:
    comparison = compare_printed({"Yes": 0.0004, "No": 0.9996}, {"Yes": 0.0, "No": 1.0})
    assert comparison.missing == []
    assert comparison.worst_abs_diff == pytest.approx(0.0004)
    assert comparison.diffs["No"]["got"] == pytest.approx(0.9996)


def test_compare_printed_never_skips_a_missing_label() -> None:
    comparison = compare_printed({"Yes": 0.5, "No": 0.5}, {"Maybe": 0.2})
    assert comparison.missing == ["Maybe"]
    assert comparison.worst_abs_diff is None
    assert comparison.diffs["Maybe"]["got"] is None


def test_argmax_label_handles_empty_mapping() -> None:
    assert argmax_label({"a": 0.2, "b": 0.8}) == "b"
    assert argmax_label({}) is None


def test_summarize_lengths_uses_the_recorded_quantile_rule() -> None:
    lengths = list(range(1, 101))  # 1..100, already sorted
    summary = summarize_lengths(lengths)
    assert summary.n == 100
    assert summary.median == 50  # int(median of 1..100 = 50.5)
    assert summary.p90 == 91  # index int(0.90 * 100) = 90 -> value 91
    assert summary.p95 == 96  # index int(0.95 * 100) = 95 -> value 96
    assert summary.max == 100
    assert summary.mean == pytest.approx(50.5)


def test_summarize_lengths_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        summarize_lengths([])


def _screen() -> list[dict[str, object]]:
    return [
        {"template_id": "fresh_kept", "tier": "fresh", "drop": False},
        {"template_id": "fresh_dropped", "tier": "fresh", "drop": True},
        {"template_id": "t1_kept", "tier": "tier1", "drop": False},
    ]


def test_count_test_denominator_separates_kept_dropped_and_tiers() -> None:
    rows = [
        {"split": "test", "template_id": "fresh_kept"},
        {"split": "test", "template_id": "fresh_kept"},
        {"split": "test", "template_id": "fresh_dropped"},
        {"split": "test", "template_id": "t1_kept"},
        {"split": "dev", "template_id": "fresh_kept"},  # not a test row: not counted
    ]
    counts = count_test_denominator(rows, _screen())
    assert counts.raw_test_rows == 4
    assert counts.kept_test_rows == 3
    assert counts.kept_fresh_rows == 2
    assert counts.kept_tier1_rows == 1
    assert counts.dropped_test_rows == 1
    assert counts.dropped_by_template == {"fresh_dropped": 1}


def test_count_test_denominator_refuses_an_unscreened_template() -> None:
    rows = [{"split": "test", "template_id": "not_in_screen"}]
    with pytest.raises(KeyError):
        count_test_denominator(rows, _screen())
