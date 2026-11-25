"""The readout gate (D12, D21) on synthetic cells: a constant answer, the majority ceiling, below chance, a balanced cell."""

import random

import pytest

from meddecide.eval.readout_gate import cell_gate, gate_by_template


def test_constant_answer_on_a_balanced_template_fails() -> None:
    gold = ["A", "B", "C"] * 10
    predicted = ["A"] * 29 + ["B"]
    correct = [g == p for g, p in zip(gold, predicted, strict=True)]
    cell = cell_gate(gold, predicted, correct, chances=[1 / 3] * 30)
    assert not cell.passes
    assert cell.status.startswith("READOUT_FAIL — constant answer (A) on 0.967 of items")


def test_a_mostly_one_answer_template_is_not_flagged() -> None:
    gold = ["A"] * 27 + ["B"] * 3  # the majority share 0.9 is above the 0.60 ceiling, so the constant answer is correct, not degenerate
    predicted = ["A"] * 30
    correct = [g == p for g, p in zip(gold, predicted, strict=True)]
    cell = cell_gate(gold, predicted, correct, chances=[0.5] * 30)
    assert cell.passes
    assert cell.status == "PASS"


def test_below_chance_fails() -> None:
    gold = ["A", "B"] * 50
    predicted = ["B", "A"] * 50
    correct = [False] * 100
    cell = cell_gate(gold, predicted, correct, chances=[0.5] * 100)
    assert not cell.passes
    assert "below chance" in cell.status


def test_a_balanced_random_cell_passes() -> None:
    rng = random.Random(0)
    gold = [rng.choice("AB") for _ in range(400)]
    predicted = [rng.choice("AB") for _ in range(400)]
    correct = [g == p for g, p in zip(gold, predicted, strict=True)]
    cell = cell_gate(gold, predicted, correct, chances=[0.5] * 400)
    assert cell.passes
    assert 0.4 < cell.accuracy < 0.6


def test_gate_by_template_groups_rows_by_template() -> None:
    rows = [
        {"template_id": "t1", "gold_key": "A", "predicted_key": "A", "correct": True, "probs": [0.9, 0.1]},
        {"template_id": "t1", "gold_key": "B", "predicted_key": "A", "correct": False, "probs": [0.8, 0.2]},
        {"template_id": "t2", "gold_key": "C", "predicted_key": "C", "correct": True, "probs": [0.2, 0.3, 0.5]},
    ]
    gates = gate_by_template(rows)
    assert list(gates) == ["t1", "t2"]
    assert gates["t1"].items == 2
    assert gates["t2"].items == 1
    assert gates["t2"].chance == pytest.approx(1 / 3)


def test_an_empty_or_misaligned_cell_is_refused() -> None:
    with pytest.raises(ValueError):
        cell_gate([], [], [], [])
    with pytest.raises(ValueError):
        cell_gate(["A"], ["A", "B"], [True], [0.5])
