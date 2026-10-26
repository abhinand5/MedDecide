"""osler_v5 D54 and D52(b): the dev composite S and the positional health check, on synthetic prediction rows."""

from __future__ import annotations

import pytest

from meddecide.eval.composite import (
    accuracy,
    composite,
    mimic_partition,
    mr_partition,
    permutation_for,
    positional_health,
)


def _row(item: str, *, source: str = "medqa", template: str = "medqa_usmle4_v1", correct: bool = True,
         n_options: int = 4, gold_index: int = 0) -> dict:
    return {"item_id": item, "source": source, "template_id": template, "correct": correct, "n_options": n_options,
            "gold_index": gold_index}


def test_accuracy_is_item_pooled_and_an_empty_partition_is_an_error() -> None:
    assert accuracy([_row("a"), _row("b", correct=False), _row("c"), _row("d")]) == 0.75
    with pytest.raises(ValueError, match="empty"):
        accuracy([])


def test_the_mr_and_mimic_partitions_follow_the_source_and_the_template_prefix() -> None:
    rows = [_row("a", source="mr_v1", template="mr_x"), _row("b", source="medqa"),
            _row("c", source="medqa", template="mimic_msdrg_v1"), _row("d", source="mr_pmc", template="mr_y")]
    assert [r["item_id"] for r in mr_partition(rows)] == ["a", "d"]
    assert [r["item_id"] for r in mimic_partition(rows)] == ["c"]


def test_s_is_the_mean_of_mr_mimic_and_cg_micro_with_cg_pooled_over_its_own_items() -> None:
    fast = [_row("m1", source="mr_v1", correct=True), _row("m2", source="mr_v1", correct=False),
            _row("x1", source="medqa", template="mimic_principal_dx_v1", correct=True),
            _row("x2", source="medqa", template="mimic_principal_dx_v1", correct=False),
            _row("x3", source="medqa", template="mimic_principal_dx_v1", correct=False)]
    cg = [_row("c1", correct=True), _row("c2", correct=True), _row("c3", correct=False), _row("c4", correct=True)]
    parts = composite(fast, cg)
    assert parts.mr == 0.5 and parts.mimic == pytest.approx(1 / 3) and parts.cg_micro == 0.75
    assert parts.s == pytest.approx((0.5 + 1 / 3 + 0.75) / 3)
    assert (parts.n_mr, parts.n_mimic, parts.n_cg) == (2, 3, 4)


def test_the_positional_check_compares_two_letter_and_single_letter_gold_on_the_wide_templates_only() -> None:
    wide = "cg_v2_ddxplus_validate_v1"
    rows = [_row(f"w{i}", template=wide, n_options=49, gold_index=30 if i < 4 else 3, correct=(i in (0, 1, 4, 5))) for i in range(8)]
    rows += [_row("n1", n_options=4, gold_index=3, correct=False)]  # a narrow template does not count
    health = positional_health(rows)
    assert health.two_letter_accuracy == 0.5 and health.single_letter_accuracy == 0.5
    assert (health.n_two_letter, health.n_single_letter) == (4, 4)
    assert health.passes  # 0.5 >= 0.5 x 0.5
    collapsed = [_row(f"w{i}", template=wide, n_options=49, gold_index=30 if i < 4 else 3, correct=(i >= 4)) for i in range(8)]
    failing = positional_health(collapsed)
    assert failing.two_letter_accuracy == 0.0 and failing.single_letter_accuracy == 1.0 and not failing.passes


def test_the_positional_check_refuses_a_missing_gold_position_and_a_set_without_wide_templates() -> None:
    with pytest.raises(ValueError, match="no dev template"):
        positional_health([_row("a")])
    with pytest.raises(ValueError, match="no gold position"):
        positional_health([{**_row("w", template="t", n_options=49), "gold_index": None}])


def test_the_permutations_are_seeded_and_are_permutations() -> None:
    assert permutation_for(49, seed=1, index=7) == permutation_for(49, seed=1, index=7)
    assert permutation_for(49, seed=1, index=7) != permutation_for(49, seed=2, index=7)
    assert sorted(permutation_for(49, seed=2, index=7)) == list(range(49))
    assert permutation_for(1, seed=1, index=0) == [0]
