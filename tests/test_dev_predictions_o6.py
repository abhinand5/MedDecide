"""Per-item prediction rows of an O6-O8 arm (the paired-bootstrap input for O9 and O11), on synthetic scored items."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from meddecide.train.osler_arm import dev_prediction_rows


def _scored() -> SimpleNamespace:
    items = [
        SimpleNamespace(item_id="a" * 16, source="pubmed", template_id="t1", qtype="choice"),
        SimpleNamespace(item_id="b" * 16, source="medqa", template_id="t2", qtype="noul"),
    ]
    return SimpleNamespace(
        items=items,
        probs=[np.array([0.1, 0.7, 0.2]), np.array([0.6, 0.4])],
        gold_indices=[1, 1],
    )


def test_rows_follow_input_order_and_flag_correctness() -> None:
    rows = dev_prediction_rows(_scored())
    assert [r["item_id"] for r in rows] == ["a" * 16, "b" * 16]
    assert [r["correct"] for r in rows] == [True, False]
    assert rows[1]["predicted_index"] == 0 and rows[1]["gold_index"] == 1


def test_rows_carry_the_full_distribution_and_identity() -> None:
    row = dev_prediction_rows(_scored())[0]
    assert row["probs"] == [0.1, 0.7, 0.2]
    assert row["n_options"] == 3
    assert (row["source"], row["template_id"], row["qtype"]) == ("pubmed", "t1", "choice")
    assert abs(sum(row["probs"]) - 1.0) < 1e-12
