"""Unit tests for the distill_v0 pool/basket selection (synthetic rows only)."""

from __future__ import annotations

from meddecide.panel.overlap import row_text_key
from meddecide.train.distill_mix import (
    BIGBENCH_PREFIX,
    CT_TEACHER_CAP,
    POOL_ROWS_EXPECTED,
    RULE_LOOKUP_EXPECTED_TOTAL,
    TEACHER_ROWS_EXPECTED,
    basket_shares,
    drop_rule_lookup_rows,
    estimate_tokens,
    is_rule_lookup,
    select_teacher_rows,
)


def row(item_id: str, template: str, state: str = "s", question: str = "q") -> dict[str, object]:
    return {"item_id": item_id, "template_id": template, "state": state, "question": question,
            "options": [{"key": "A", "label": "alpha"}, {"key": "B", "label": "beta"}], "gold": "A"}


def test_rule_lookup_classification() -> None:
    assert is_rule_lookup("fda_class_choice_v1")
    assert is_rule_lookup("pubmed_check_date_noul_v1")
    assert is_rule_lookup("pubmed_pubtype_choice_v1")
    assert not is_rule_lookup("medmcqa_4opt_v1")


def test_rule_lookup_drop_counts_and_reasons() -> None:
    rows = [row("1", "fda_class_choice_v1"), row("2", "fda_class_choice_v1"), row("3", "pubmed_check_x_noul_v1"),
            row("4", "medmcqa_4opt_v1"), row("5", "pubmed_pubtype_choice_v1")]
    kept, report = drop_rule_lookup_rows(rows)
    assert [r["item_id"] for r in kept] == ["4"]
    assert report["reason"] == "rule_lookup_metadata"
    assert report["dropped_total"] == 4
    assert report["dropped_by_template"]["fda_class_choice_v1"] == 2
    assert report["realised"]["pubmed_check_*"] == 1
    assert report["matches_expected"] is False  # the synthetic counts are not the recorded ones
    assert report["expected_total"] == RULE_LOOKUP_EXPECTED_TOTAL
    assert POOL_ROWS_EXPECTED == 1_268_815


def test_teacher_selection_named_and_bigbench() -> None:
    rows = [row("a", "medmcqa_4opt_v1"), row("b", "medmcqa_4opt_v1"), row("c", BIGBENCH_PREFIX + "x_v1"),
            row("d", "mimic_msdrg_v1")]
    indices, report = select_teacher_rows(rows)
    assert [rows[i]["item_id"] for i in indices] == ["a", "b", "c"]
    assert report["per_template"]["medmcqa_4opt_v1"]["available"] == 2
    assert report["per_template"][BIGBENCH_PREFIX + "*"]["selected"] == 1
    assert report["matches_expected"] is False
    assert report["expected_total"] == TEACHER_ROWS_EXPECTED


def test_ct_cap_is_deterministic_and_per_template() -> None:
    rows = [row(f"{i:016x}", "ct_randomised_noul_v1") for i in range(CT_TEACHER_CAP + 5)]
    rows += [row(f"{i + 100000:016x}", "ct_masking_choice_v1") for i in range(3)]
    indices, report = select_teacher_rows(rows)
    assert report["per_template"]["ct_randomised_noul_v1"]["selected"] == CT_TEACHER_CAP
    assert report["per_template"]["ct_masking_choice_v1"]["selected"] == 3
    assert len(indices) == CT_TEACHER_CAP + 3
    again, _ = select_teacher_rows(rows)
    assert again == indices
    other, _ = select_teacher_rows(rows, seed=1)
    ct_pick = {rows[i]["item_id"] for i in indices if str(rows[i]["template_id"]).startswith("ct_")}
    ct_other = {rows[i]["item_id"] for i in other if str(rows[i]["template_id"]).startswith("ct_")}
    assert len(ct_pick) == len(ct_other) == CT_TEACHER_CAP + 3
    assert ct_pick != ct_other  # a different seed takes a different subset of the over-cap template


def test_basket_shares() -> None:
    rows = [row("1", "medmcqa_4opt_v1", state="s1"), row("2", "mimic_msdrg_v1", state="s2"),
            row("3", "mimic_msdrg_v1", state="s3"), row("4", "mr_page_noul_v1", state="s4")]
    shares = basket_shares(rows, {row_text_key(rows[0])})
    assert shares["pool_rows"] == 4
    assert shares["teacher_rows"] == 1
    assert shares["mimic_rows"] == 2
    assert shares["other_gold_rows"] == 1
    assert shares["teacher_share"] == 0.25


def test_estimate_tokens_is_labelled_an_estimate() -> None:
    rows = [row("1", "t", state="a" * 38, question="b" * 19)]
    estimate = estimate_tokens(rows)
    assert estimate["items"] == 1
    assert estimate["estimated_input_tokens"] == int((38 + 19 + 10) / 3.8)
