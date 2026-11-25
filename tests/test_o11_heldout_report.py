"""The O11 held-out report (ADVISORY O11; heldout.md rule 3): correctness, the item table, the template macro and the paired
differences, on synthetic predictions with known accuracies (CPU)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("o11_heldout_report", REPO / "scripts/osler/o11_heldout_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(correct_share: float) -> list[dict]:
    """600 prediction rows: 300 choice items (note_lab_range_v1) and 300 score items (policy_triage_v1). The first
    ``correct_share`` of each template puts 0.8 on the gold option; the rest put 0.8 on a wrong option."""
    rows = []
    for template, keys, gold in (("note_lab_range_v1", ["A", "B", "C"], "B"), ("policy_triage_v1", ["1", "2", "3"], "2")):
        right = round(correct_share * 300)
        gold_index = keys.index(gold)
        for i in range(300):
            pick = gold_index if i < right else (gold_index + 1) % len(keys)
            probs = [0.1] * len(keys)
            probs[pick] = 0.8
            rows.append({"item_id": f"{template}-{i:04d}", "template_id": template, "option_keys": keys, "probs": probs,
                         "gold": gold, "status": "scored"})
    return rows


def test_the_answer_is_the_most_probable_option() -> None:
    module = _module()
    assert module.is_correct({"option_keys": ["yes", "no"], "probs": [0.2, 0.8], "gold": "no"})
    assert not module.is_correct({"option_keys": ["yes", "no"], "probs": [0.2, 0.8], "gold": "yes"})


def test_the_item_table_refuses_a_duplicated_row_a_row_that_is_not_scored_and_a_short_file() -> None:
    module = _module()
    rows = _rows(0.5)
    with pytest.raises(ValueError, match="written twice"):
        module.item_table([*rows, rows[0]])
    skipped = dict(rows[0], status="skipped", probs=None)
    with pytest.raises(ValueError, match="not scored"):
        module.item_table([skipped, *rows[1:]])
    with pytest.raises(ValueError, match="600 expected"):
        module.item_table(rows[:10])


def test_the_template_macro_is_the_mean_over_templates_and_the_interval_brackets_it() -> None:
    module = _module()
    summary = module.summarise(module.item_table(_rows(0.5)))
    assert summary["template_macro_accuracy"] == pytest.approx(0.5)
    assert summary["accuracy_overall"] == pytest.approx(0.5)
    assert summary["n_by_template"] == {"note_lab_range_v1": 300, "policy_triage_v1": 300}
    lo, hi = summary["template_macro_ci_95"]
    assert lo <= 0.5 <= hi


def test_the_paired_difference_is_the_difference_of_the_template_macros() -> None:
    module = _module()
    diff = module.paired(module.item_table(_rows(0.9)), module.item_table(_rows(0.5)))
    assert diff["n"] == 600
    assert diff["point"] == pytest.approx(0.4)
    lo, hi = diff["ci_95"]
    assert lo <= 0.4 <= hi


def test_the_interval_is_seeded() -> None:
    module = _module()
    groups = {"a": [1.0, 0.0, 1.0, 1.0], "b": [0.0, 0.0, 1.0, 0.0]}
    assert module.macro_interval(groups) == module.macro_interval(groups)
