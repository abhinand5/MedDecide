"""osler_v4 Gate O4 (meddecide.eval.gate_o4): pairing, the paired statistics, readout restrictions and the verdicts (CPU)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from meddecide.eval.gate_o4 import (
    CheckResult,
    check,
    comparison,
    gate,
    normalise_row,
    option_index,
    pair_rows,
    readout_failures,
)


def options(n: int) -> list[dict[str, Any]]:
    return [{"key": chr(ord("A") + i), "label": f"option {i}"} for i in range(n)]


def item(item_id: str, template: str, n_options: int = 2, gold: str = "A") -> dict[str, Any]:
    return {"item_id": item_id, "template_id": template, "gold": gold, "options": options(n_options)}


def pred(item_id: str, template: str, probs: list[float], gold_key: str = "A") -> dict[str, Any]:
    predicted = chr(ord("A") + probs.index(max(probs)))
    return {"item_id": item_id, "template_id": template, "gold_key": gold_key, "predicted_key": predicted,
            "correct": predicted == gold_key, "probs": probs}


def test_option_index_is_the_position_of_the_gold_key() -> None:
    assert option_index(options(4), "C") == 2
    with pytest.raises(ValueError):
        option_index(options(2), "Z")


def test_pairing_counts_every_unpaired_item_with_its_reason() -> None:
    items = [item("a", "T1"), item("b", "T1", n_options=12), item("c", "T2"), item("d", "T2")]
    model = [pred("a", "T1", [0.9, 0.1]), pred("b", "T1", [0.5, 0.5]), pred("c", "T2", [0.2, 0.8])]
    base = [pred("a", "T1", [0.6, 0.4]), pred("b", "T1", [0.5, 0.5]), pred("d", "T2", [0.3, 0.7])]
    pairing = pair_rows(model, base, items, option_cap=10)
    assert pairing.items == 4
    assert [p[0]["item_id"] for p in pairing.pairs] == ["a"]
    assert pairing.unpaired == {"more than 10 options": 1, "not scored by the baseline": 1,
                                "not scored by the final model": 1}
    assert pairing.unpaired_total == 3
    assert pairing.unpaired_share == pytest.approx(0.75)
    assert pairing.unpaired_templates == {"T1": 1, "T2": 2}


def test_pairing_without_a_cap_keeps_the_twelve_option_item() -> None:
    items = [item("a", "T1"), item("b", "T1", n_options=12)]
    model = [pred("a", "T1", [0.9, 0.1]), pred("b", "T1", [0.5, 0.5] + [0.0] * 10)]
    base = [pred("a", "T1", [0.6, 0.4]), pred("b", "T1", [0.5, 0.5] + [0.0] * 10)]
    assert len(pair_rows(model, base, items).pairs) == 2


def test_a_template_with_no_paired_item_drops_out_and_is_reported() -> None:
    items = [item("a", "T1"), item("b", "T2", n_options=12)]
    model = [pred("a", "T1", [0.9, 0.1]), pred("b", "T2", [0.5, 0.5] + [0.0] * 10)]
    base = [pred("a", "T1", [0.6, 0.4]), pred("b", "T2", [0.5, 0.5] + [0.0] * 10)]
    pairing = pair_rows(model, base, items, option_cap=10)
    assert pairing.paired_templates == frozenset({"T1"})
    assert pairing.dropped_templates == frozenset({"T2"})


def test_comparison_is_the_template_macro_of_the_paired_differences() -> None:
    # T1: the final model minus the baseline correct flags are 0 and +1 (mean +0.5); T2: -1 and -1 (mean -1).
    items = [item("a1", "T1"), item("a2", "T1"), item("b1", "T2"), item("b2", "T2")]
    model = [pred("a1", "T1", [0.9, 0.1]), pred("a2", "T1", [0.9, 0.1]),
             pred("b1", "T2", [0.1, 0.9]), pred("b2", "T2", [0.1, 0.9])]
    base = [pred("a1", "T1", [0.9, 0.1]), pred("a2", "T1", [0.1, 0.9]),
            pred("b1", "T2", [0.9, 0.1]), pred("b2", "T2", [0.9, 0.1])]
    pairing = pair_rows(model, base, items)
    result = comparison(pairing)
    assert result.items == 4
    assert result.templates == ("T1", "T2")
    assert result.accuracy.point == pytest.approx((0.5 + -1.0) / 2)


def test_brier_difference_is_negative_when_the_final_model_puts_more_mass_on_gold() -> None:
    items = [item(f"i{k}", "T1") for k in range(4)]
    model = [pred(f"i{k}", "T1", [0.9, 0.1]) for k in range(4)]
    base = [pred(f"i{k}", "T1", [0.5, 0.5]) for k in range(4)]
    result = comparison(pair_rows(model, base, items))
    # per item: (0.9 - 1)^2 + 0.1^2 = 0.02 for the final model, (0.5 - 1)^2 + 0.5^2 = 0.5 for the baseline
    assert result.brier.point == pytest.approx(0.02 - 0.5)
    assert result.brier.hi < 0


def test_readout_failures_flags_a_constant_answer_and_passes_a_working_cell() -> None:
    rows = [{"template_id": "constant", "gold_key": "A" if k < 50 else "B", "predicted_key": "A", "correct": k < 50,
             "probs": [0.6, 0.4]} for k in range(100)]
    working = []
    for k in range(100):
        gold = "A" if k % 2 == 0 else "B"
        predicted = gold if k % 5 else ("B" if gold == "A" else "A")  # 80 of 100 correct, both answers used
        working.append({"template_id": "working", "gold_key": gold, "predicted_key": predicted,
                        "correct": predicted == gold, "probs": [0.6, 0.4]})
    failures = readout_failures(rows + working)
    assert "constant" in failures
    assert "working" not in failures


def _pairs_with_accuracy(n_per_template: int, model_correct: float, base_correct: float, seed: int = 0,
                         identical: bool = False) -> tuple[list, list, list]:
    """Synthetic paired predictions; ``identical`` makes the baseline's correctness equal the final model's (a zero difference)."""
    rng = random.Random(seed)
    items, model, base = [], [], []
    for t in ("T1", "T2", "T3", "T4"):
        for k in range(n_per_template):
            item_id = f"{t}-{k}"
            items.append(item(item_id, t))
            m_right = rng.random() < model_correct
            b_right = m_right if identical else rng.random() < base_correct
            m_probs = [0.8, 0.2] if m_right else [0.2, 0.8]
            b_probs = [0.6, 0.4] if b_right else [0.4, 0.6]
            model.append(pred(item_id, t, m_probs))
            base.append(pred(item_id, t, b_probs))
    return items, model, base


def test_mr_check_passes_when_the_final_model_is_clearly_better() -> None:
    items, model, base = _pairs_with_accuracy(200, model_correct=0.95, base_correct=0.6)
    result = check("MR", pair_rows(model, base, items), kind="mr_or_mimic")
    assert result.verdict == "PASS"
    assert result.comparison is not None and result.comparison.accuracy.lo > 0


def test_mr_check_fails_when_the_baseline_is_clearly_better() -> None:
    items, model, base = _pairs_with_accuracy(200, model_correct=0.6, base_correct=0.95)
    assert check("MR", pair_rows(model, base, items), kind="mr_or_mimic").verdict == "FAIL"


def test_panel_check_allows_a_zero_difference_and_fails_a_large_deficit() -> None:
    items, model, base = _pairs_with_accuracy(200, model_correct=0.9, base_correct=0.9, identical=True)
    assert check("panel", pair_rows(model, base, items), kind="accuracy").verdict == "PASS"
    items, model, base = _pairs_with_accuracy(200, model_correct=0.6, base_correct=0.95)
    assert check("panel", pair_rows(model, base, items), kind="accuracy").verdict == "FAIL"


def test_a_check_with_more_than_one_percent_unpaired_is_not_measured() -> None:
    items = [item(f"i{k}", "T1", n_options=12 if k < 3 else 2) for k in range(100)]
    model = [pred(f"i{k}", "T1", [0.9, 0.1]) for k in range(100)]
    base = [pred(f"i{k}", "T1", [0.5, 0.5]) for k in range(100)]
    result = check("panel", pair_rows(model, base, items, option_cap=10), kind="accuracy")
    assert result.verdict == "NOT MEASURED"
    assert "above 1 %" in result.reason


def test_a_failing_readout_cell_is_not_measured_without_a_restriction() -> None:
    items, model, base = _pairs_with_accuracy(50, model_correct=0.9, base_correct=0.6)
    result = check("knowledge", pair_rows(model, base, items), kind="accuracy", model_failures=["T2"])
    assert result.verdict == "NOT MEASURED"
    assert "T2" in result.reason


def test_a_restriction_is_labelled_and_needs_a_failing_cell() -> None:
    items, model, base = _pairs_with_accuracy(50, model_correct=0.9, base_correct=0.6)
    pairing = pair_rows(model, base, items)
    restricted = check("MIMIC", pairing, kind="mr_or_mimic", base_failures=["T4"],
                       restriction=frozenset({"T1", "T2", "T3"}), restriction_note="the operator's ruling, STATE §6 question 6")
    assert restricted.comparison is not None and restricted.comparison.templates == ("T1", "T2", "T3")
    assert "restricted to the 3 templates" in restricted.reason
    with pytest.raises(ValueError):
        check("MIMIC", pairing, kind="mr_or_mimic", restriction=frozenset({"T1"}))


def test_gate_is_pass_only_when_every_check_passes() -> None:
    passed = CheckResult("i", "PASS", "")
    failed = CheckResult("ii", "FAIL", "")
    unmeasured = CheckResult("iii", "NOT MEASURED", "")
    assert gate({"i": passed, "ii": passed}) == "PASS"
    assert gate({"i": passed, "ii": failed, "iii": unmeasured}) == "FAIL"
    assert gate({"i": passed, "iii": unmeasured}) == "NOT MEASURED"
    with pytest.raises(ValueError):
        gate({})


def test_normalise_row_takes_the_argmax_and_the_items_gold_and_template() -> None:
    it = item("n1", "T9", n_options=3, gold="B")
    raw = {"item_id": "n1", "probs": [0.2, 0.5, 0.3]}
    row = normalise_row(raw, it)
    assert row["predicted_key"] == "B" and row["correct"] is True and row["gold_key"] == "B"
    assert row["template_id"] == "T9"
    with pytest.raises(ValueError):
        normalise_row({"item_id": "n1", "probs": [0.5, 0.5]}, it)
