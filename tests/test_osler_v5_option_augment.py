"""osler_v5 D52a: option-count augmentation of closed-list choice rows. Synthetic rows only (no MIMIC text, codes or titles)."""

from __future__ import annotations

import numpy as np
import pytest

from meddecide.train.option_augment import (
    CLOSED_LIST,
    DistractorPool,
    augment_row,
    augment_rows,
    build_pools,
    code_and_title,
    draw_entry,
    expected_two_letter_golds,
    gold_label,
    option_keys,
    validate_templates,
)

PRINCIPAL = "mimic_principal_dx_v1"
DRG = "mimic_msdrg_v1"


def _drg_row(i: int, *, split: str = "train", n_options: int = 4) -> dict:
    """A synthetic DRG row: the gold is the row's own DRG; the offered labels are other synthetic titles."""
    labels = [f"Title {i}"] + [f"Other {i}-{k}" for k in range(1, n_options)]
    return {
        "item_id": f"syn{i:05d}", "template_id": DRG, "qtype": "choice", "split": split, "skill": "s", "source": "syn",
        "state": "state", "question": "question", "meta": {"drg": str(100 + i), "n_options": n_options},
        "options": [{"key": k, "label": label} for k, label in zip("ABCD"[:n_options], labels, strict=True)],
        "gold": "A",
    }


def _principal_row(i: int) -> dict:
    code = f"X{i:03d}"
    return {
        "item_id": f"pri{i:05d}", "template_id": PRINCIPAL, "qtype": "choice", "split": "train", "skill": "s", "source": "syn",
        "state": "state", "question": "question", "meta": {"principal": code},
        "options": [{"key": "A", "label": f"{code} Title {i}"}, {"key": "B", "label": f"Y{i:03d} Other"},
                    {"key": "C", "label": f"Z{i:03d} Other"}, {"key": "D", "label": f"W{i:03d} Other"}],
        "gold": "A",
    }


def _pool(n: int) -> DistractorPool:
    return build_pools([_drg_row(i) for i in range(n)], [DRG])[DRG]


def test_the_closed_list_names_two_templates_each_with_a_reason() -> None:
    assert set(CLOSED_LIST) == {PRINCIPAL, DRG}
    assert all(reason.strip() for reason in CLOSED_LIST.values())


def test_unknown_or_repeated_templates_are_refused() -> None:
    with pytest.raises(ValueError, match="not on the closed list"):
        validate_templates(["csqa_choice_v1"])
    with pytest.raises(ValueError, match="named twice"):
        validate_templates([DRG, DRG])
    assert validate_templates([DRG]) == (DRG,)


def test_option_keys_are_the_readout_codes_in_order() -> None:
    assert option_keys(3) == ["A", "B", "C"]
    assert option_keys(27)[25:27] == ["Z", "AA"]
    assert option_keys(64)[63] == "BL"
    assert len(set(option_keys(64))) == 64
    with pytest.raises(ValueError):
        option_keys(0)


def test_pools_are_built_from_training_rows_only_and_count_distinct_labels() -> None:
    rows = [_drg_row(0), _drg_row(0), _drg_row(1)]
    pool = build_pools(rows, [DRG])[DRG]
    assert len(pool.entries) == 3  # one entry per row: a label is drawn in proportion to its gold frequency
    assert pool.label_set_size == 2
    with pytest.raises(ValueError, match="training rows only"):
        build_pools([_drg_row(2, split="dev")], [DRG])


def test_the_code_and_title_of_a_label_come_from_the_row_metadata() -> None:
    row = _principal_row(7)
    assert code_and_title(PRINCIPAL, row, gold_label(row)) == ("X007", "Title 7")
    drg = _drg_row(3)
    assert code_and_title(DRG, drg, gold_label(drg)) == ("103", "Title 3")


def test_an_augmented_row_keeps_its_gold_has_no_duplicate_codes_or_titles_and_a_count_in_range() -> None:
    pool = _pool(40)
    augmented = 0
    for i in range(300):
        row = _drg_row(i)
        new, outcome = augment_row(row, pool, seed=0, probability=1.0)
        assert outcome == "augmented"
        augmented += 1
        n = len(new["options"])
        assert 8 <= n <= min(64, pool.label_set_size)
        labels = [o["label"] for o in new["options"]]
        assert len(set(labels)) == n
        assert labels.count(gold_label(row)) == 1
        assert gold_label(new) == gold_label(row)
        assert [o["key"] for o in new["options"]] == option_keys(n)
        offered = new["meta"]["offered"]
        assert len(offered) == n
        assert offered[[o["key"] for o in new["options"]].index(new["gold"])] == [str(100 + i), f"Title {i}"]
        assert len({code for code, _ in offered}) == n and len({title for _, title in offered}) == n
        assert new["item_id"] == row["item_id"]
        assert {k: v for k, v in new["meta"].items() if k != "offered"} == row["meta"]
    assert augmented == 300


def test_a_row_with_a_principal_code_keeps_one_gold_among_distinct_codes() -> None:
    pool = build_pools([_principal_row(i) for i in range(30)], [PRINCIPAL])[PRINCIPAL]
    new, outcome = augment_row(_principal_row(0), pool, seed=3, probability=1.0)
    assert outcome == "augmented"
    codes = [code for code, _ in new["meta"]["offered"]]
    assert len(set(codes)) == len(codes) == len(new["options"])
    gold_text = next(o["label"] for o in new["options"] if o["key"] == new["gold"])
    assert gold_text.startswith("X000")


def test_the_gold_position_is_uniform_over_the_offered_options() -> None:
    pool = _pool(8)  # a label set of 8 gives every rebuilt row exactly 8 options
    positions = [0] * 8
    for i in range(800):
        row = _drg_row(i)
        new, outcome = augment_row(row, pool, seed=11, probability=1.0)
        assert outcome == "augmented" and len(new["options"]) == 8
        positions[[o["key"] for o in new["options"]].index(new["gold"])] += 1
    assert sum(positions) == 800
    assert all(60 <= count <= 140 for count in positions), positions  # 100 +- 40 per position


def test_a_pool_too_small_for_the_floor_falls_back_and_the_row_is_unchanged() -> None:
    pool = _pool(5)
    row = _drg_row(1)
    new, outcome = augment_row(row, pool, seed=0, probability=1.0)
    assert outcome == "fallback"
    assert new == row


def test_probability_zero_selects_no_row() -> None:
    pool = _pool(40)
    row = _drg_row(2)
    new, outcome = augment_row(row, pool, seed=0, probability=0.0)
    assert outcome == "not_selected" and new == row


def test_augment_rows_counts_every_pooled_row_and_passes_the_others_through() -> None:
    pool = _pool(40)
    other = {**_drg_row(99), "template_id": "csqa_choice_v1"}
    rows = [_drg_row(i) for i in range(50)] + [other]
    out, report = augment_rows(rows, {DRG: pool}, seed=5, probability=0.5)
    assert len(out) == len(rows)
    assert out[-1] == other
    seen = report.seen[DRG]
    assert seen == 50
    assert seen == report.augmented[DRG] + report.not_selected[DRG] + report.fallback[DRG]
    assert report.added_options[DRG] == sum(len(o["options"]) - 4 for o in out[:50] if len(o["options"]) > 4)
    assert "csqa_choice_v1" not in report.seen


def test_the_draw_is_deterministic_for_a_seed_and_an_item() -> None:
    pool = _pool(40)
    first, _ = augment_row(_drg_row(4), pool, seed=9, probability=1.0)
    second, _ = augment_row(_drg_row(4), pool, seed=9, probability=1.0)
    third, _ = augment_row(_drg_row(4), pool, seed=10, probability=1.0)
    assert first == second
    assert first != third


def test_a_score_row_is_refused() -> None:
    row = {**_drg_row(1), "qtype": "score"}
    with pytest.raises(ValueError, match="choice rows only"):
        augment_row(row, _pool(40), seed=0, probability=1.0)


def test_distractors_are_drawn_in_proportion_to_gold_frequency() -> None:
    heavy = [_drg_row(0)] * 91  # one label that is the gold of 91 training rows
    light = [_drg_row(i) for i in range(1, 9)]  # eight labels that are the gold of one row each
    pool = build_pools(heavy + light, [DRG])[DRG]
    assert len(pool.entries) == 99 and pool.label_set_size == 9
    rng = np.random.default_rng(0)
    draws = [draw_entry(pool, rng)[2] for _ in range(20_000)]
    share = sum(label == gold_label(_drg_row(0)) for label in draws) / len(draws)
    assert share == pytest.approx(91 / 99, abs=0.01)


def test_the_expected_two_letter_golds_follow_the_uniform_count_and_gold_position() -> None:
    expected = expected_two_letter_golds(1000, 1.0)
    assert set(expected) == set(range(26, 64))
    manual_26 = sum(1.0 / c for c in range(27, 65)) / 57 * 1000
    assert expected[26] == pytest.approx(manual_26, rel=1e-9)
    assert expected[63] == pytest.approx(1000 / (57 * 64), rel=1e-9)
    assert expected[26] > expected[40] > expected[63]
    assert expected_two_letter_golds(1000, 0.5)[26] == pytest.approx(manual_26 / 2, rel=1e-9)
