"""osler_v5 D56a: the corrected mix drops every row of the named template, nothing else, keeps the order, and refuses other names."""

from __future__ import annotations

import pytest

from meddecide.train.corrected_mix import (
    DROP_ALLOWED,
    drop_template_rows,
    template_counts,
    validate_drop,
)


def _rows() -> list[dict]:
    names = ["fda_route_choice_v1", "mimic_msdrg_v1", "fda_route_choice_v1", "csqa_choice_v1", "fda_route_choice_v1", "snli_nli_choice_v1"]
    return [{"item_id": f"id{i:03d}", "template_id": t} for i, t in enumerate(names)]


def test_every_row_of_the_named_template_is_dropped_and_nothing_else() -> None:
    rows = _rows()
    kept, report = drop_template_rows(rows, ["fda_route_choice_v1"])
    assert report.rows_before == 6 and report.rows_dropped == 3 and report.rows_remaining == 3
    assert report.dropped_by_template == {"fda_route_choice_v1": 3}
    assert all(r["template_id"] != "fda_route_choice_v1" for r in kept)
    assert template_counts(kept) == template_counts([r for r in rows if r["template_id"] != "fda_route_choice_v1"])


def test_the_kept_rows_keep_their_order_and_their_ids() -> None:
    rows = _rows()
    kept, _ = drop_template_rows(rows, ["fda_route_choice_v1"])
    assert [r["item_id"] for r in kept] == ["id001", "id003", "id005"]


def test_only_the_fixed_list_may_be_dropped() -> None:
    assert DROP_ALLOWED == ("fda_route_choice_v1",)
    with pytest.raises(ValueError, match="not allowed to be dropped"):
        validate_drop(["csqa_choice_v1"])
    with pytest.raises(ValueError, match="named twice"):
        validate_drop(["fda_route_choice_v1", "fda_route_choice_v1"])


def test_an_empty_drop_keeps_every_row() -> None:
    rows = _rows()
    kept, report = drop_template_rows(rows, [])
    assert len(kept) == len(rows) and report.rows_dropped == 0
