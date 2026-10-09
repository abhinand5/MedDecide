"""Both-order scoring rows for the Osler arms (ADVISORY O11), with a stub scorer (CPU; no model is loaded)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from meddecide.eval.osler_scoring import score_rows


def _choice_row(item_id: str = "a1b2c3d4e5f60718") -> dict:
    return {
        "item_id": item_id, "benchmark": "v0.2", "set_name": "pubmed", "template_id": "pubmed_example_choice_v1",
        "qtype": "choice", "state": "A patient presents with a headache after a fall.", "question": "Which option applies?",
        "gold": "B", "options": [{"key": "A", "label": "alpha"}, {"key": "B", "label": "beta"},
                                 {"key": "C", "label": "gamma"}],
        "split": "test", "licence": "CC0", "source_url": "https://example.org/record/1", "source_record_id": "rec-1",
        "meta": {"tier": "fresh", "skill": "triage"},
    }


def _noul_row(item_id: str = "b" * 16) -> dict:
    row = _choice_row(item_id)
    row.update({"qtype": "noul", "gold": "yes", "options": [{"key": "yes", "label": "Yes"},
                                                           {"key": "no", "label": "No"}]})
    return row


class _PositionScorer:
    """A stub model whose probabilities depend on the display position only (weights 1, 2, 3 for three options)."""

    def __init__(self) -> None:
        self.calls = 0

    def score_items(self, items, **_kwargs):
        self.calls += 1
        probs = []
        for item in items:
            weights = np.array([1.0, 2.0, 3.0]) if item.n_options == 3 else np.ones(item.n_options)
            probs.append(weights / weights.sum())
        n = len(items)
        return SimpleNamespace(probs=probs, latency_s=[0.01] * n, prompt_tokens=[7] * n)


def test_choice_items_average_both_orders_and_single_order_keeps_the_original() -> None:
    scorer = _PositionScorer()
    both, single = score_rows(scorer, [_choice_row()], rows_per_call=8)
    assert single[0]["probs"] == pytest.approx([1 / 6, 2 / 6, 3 / 6])  # original order
    assert both[0]["probs"] == pytest.approx([2 / 6, 2 / 6, 2 / 6])  # the stub is order-dependent, so the average is uniform
    assert both[0]["extras"]["orders"] == 2 and single[0]["extras"]["orders"] == 1
    assert scorer.calls == 2  # the originals and the reversed choice items
    assert both[0]["status"] == "scored" and both[0]["option_keys"] == ["A", "B", "C"]
    assert both[0]["item_id"] == "a1b2c3d4e5f60718"  # the row's id is the join key


def test_canonical_items_are_scored_once_and_identical_in_both_files() -> None:
    scorer = _PositionScorer()
    both, single = score_rows(scorer, [_noul_row()], rows_per_call=8)
    assert scorer.calls == 1  # no reversed pass for noul
    assert both[0]["probs"] == single[0]["probs"] == pytest.approx([0.5, 0.5])
    assert both[0]["extras"]["orders"] == 1


def test_rows_follow_the_input_order_across_batches() -> None:
    rows = [_choice_row(f"{i:016x}") for i in range(5)] + [_noul_row("f" * 16)]
    both, single = score_rows(_PositionScorer(), rows, rows_per_call=2)
    assert [r["item_id"] for r in both] == [r["item_id"] for r in rows]
    assert [r["item_id"] for r in single] == [r["item_id"] for r in rows]
    assert len(both) == len(single) == 6
