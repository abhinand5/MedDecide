"""Unit test for the subsampling helper of the O4 mix audit (src/meddecide/mix/audit.py)."""

from __future__ import annotations

from meddecide.mix.audit import smallest_hash_rows


def test_smallest_hash_rows_keeps_k_per_template_independent_of_order() -> None:
    rows = [{"item_id": f"id{i:04d}", "template_id": "a" if i % 3 else "b"} for i in range(300)]
    forward = smallest_hash_rows(rows, k=20)
    backward = smallest_hash_rows(list(reversed(rows)), k=20)
    assert {t: [r["item_id"] for r in v] for t, v in forward.items()} == \
        {t: [r["item_id"] for r in v] for t, v in backward.items()}
    assert len(forward["a"]) == 20 and len(forward["b"]) == 20
    assert len({r["item_id"] for r in forward["a"]}) == 20


def test_smallest_hash_rows_returns_fewer_when_a_template_is_small() -> None:
    rows = [{"item_id": f"x{i}", "template_id": "small"} for i in range(3)]
    assert len(smallest_hash_rows(rows, k=10)["small"]) == 3
