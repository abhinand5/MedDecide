"""The O11 self-audit's pure checks (ADVISORY R6; CPU): denominators, alignment, per-template shares and the accuracy tables."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _audit():
    spec = importlib.util.spec_from_file_location("o11_audit", REPO / "scripts/osler/o11_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(item_id: str, *, template: str = "t1", keys=("A", "B", "C"), probs=(0.7, 0.2, 0.1), gold: str = "A",
         benchmark: str = "v0.2", qtype: str = "choice", status: str = "scored") -> dict:
    return {"item_id": item_id, "template_id": template, "option_keys": list(keys), "probs": list(probs), "gold": gold,
            "benchmark": benchmark, "qtype": qtype, "status": status}


def test_the_answer_is_the_most_probable_option() -> None:
    audit = _audit()
    assert audit.is_correct(_row("x", probs=(0.1, 0.2, 0.7), gold="C"))
    assert not audit.is_correct(_row("x", probs=(0.1, 0.2, 0.7), gold="A"))


def test_the_denominator_check_refuses_missing_extra_duplicated_and_unscored_rows() -> None:
    audit = _audit()
    expected = {"a", "b"}
    counts = audit.check_rows([_row("a"), _row("b")], expected, label="both_orders")
    assert counts["rows"] == 2 and counts["unique_ids"] == 2
    with pytest.raises(ValueError, match="missing 1"):
        audit.check_rows([_row("a")], expected, label="both_orders")
    with pytest.raises(ValueError, match="extra 1"):
        audit.check_rows([_row("a"), _row("b"), _row("c")], expected, label="both_orders")
    with pytest.raises(ValueError, match="duplicated 1"):
        audit.check_rows([_row("a"), _row("a"), _row("b")], expected, label="both_orders")
    with pytest.raises(ValueError, match="not scored 1"):
        audit.check_rows([_row("a"), _row("b", status="skipped")], expected, label="both_orders")


def test_probabilities_must_sum_to_one_and_align_with_the_options() -> None:
    audit = _audit()
    with pytest.raises(ValueError, match="probability sums off 1"):
        audit.check_rows([_row("a", probs=(0.7, 0.2, 0.3)), _row("b")], {"a", "b"}, label="both_orders")


def test_the_per_template_shares_flag_a_constant_answer() -> None:
    audit = _audit()
    rows = [_row(f"c{i}", template="constant", probs=(0.9, 0.05, 0.05), gold="A" if i % 2 else "B") for i in range(10)]
    rows += [_row(f"v{i}", template="varied", probs=(0.1, 0.8, 0.1) if i % 2 else (0.1, 0.1, 0.8), gold="B" if i % 2 else "C")
             for i in range(10)]
    table = audit.template_checks(rows)
    assert table["constant"]["top_predicted_share"] == pytest.approx(1.0)
    assert table["constant"]["top_gold_share"] == pytest.approx(0.5)
    assert table["varied"]["top_predicted_share"] == pytest.approx(0.5)
    assert table["constant"]["accuracy"] == pytest.approx(0.5)


def test_accuracy_by_benchmark_is_the_mean_within_each_group() -> None:
    audit = _audit()
    rows = [_row("a", benchmark="v0.2", gold="A"), _row("b", benchmark="v0.2", gold="B"),
            _row("c", benchmark="ext_panel", gold="A")]
    table = audit.accuracy_table(rows, "benchmark")
    assert table["v0.2"]["n"] == 2 and table["v0.2"]["accuracy"] == pytest.approx(0.5)
    assert table["ext_panel"]["accuracy"] == pytest.approx(1.0)
