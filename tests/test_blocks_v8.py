"""osler_v1 V8 block runs: the step budget, the pairing, the dev partitions and the keep rule (pure functions, no GPU)."""

from __future__ import annotations

import itertools

import pytest

from meddecide.train.blocks import (
    keep_comparison,
    keep_statistics,
    keep_verdict,
    knowledge_check_passes,
    knowledge_comparison,
    macro_difference,
    micro_difference,
    pair_by_item,
    partition_pairs,
    steps_for_items,
)


def _row(item_id: str, correct: bool, template: str = "t1", source: str = "medqa") -> dict:
    return {"item_id": item_id, "correct": correct, "template_id": template, "source": source}


def test_steps_reach_the_target_on_a_multiple_of_four() -> None:
    assert steps_for_items([6] * 200, 500) == 84  # 84 steps hold 504 items; 83 hold 498 but 83 is not a multiple of 4
    assert steps_for_items([6] * 200, 497) == 84
    assert steps_for_items([6] * 200, 480) == 80  # exactly on target
    assert steps_for_items([6] * 200, 481) == 84


def test_steps_consume_the_batches_lazily() -> None:
    # an infinite iterable is read only as far as the target: 10 steps of 8 hold 80 items, and 10 is not a multiple of 4
    assert steps_for_items(itertools.repeat(8), 80) == 12


def test_the_final_run_steps_are_a_multiple_of_eight() -> None:
    # ADVISORY V9: eight evaluations, so the step count is a multiple of eight; 6 items a step, target 1,000
    assert steps_for_items([6] * 400, 1000, multiple=8) == 168  # 168 steps hold 1,008 items; 167 hold 1,002 but 167 is odd
    assert steps_for_items([6] * 400, 1002, multiple=8) == 168


def test_steps_fail_loudly_when_the_batches_cannot_reach_the_target() -> None:
    with pytest.raises(ValueError):
        steps_for_items([1, 1, 1], 10)
    with pytest.raises(ValueError):
        steps_for_items([6], 0)


def test_pairs_follow_the_candidate_order_and_refuse_mismatches() -> None:
    x = [_row("a", True), _row("b", False)]
    c = [_row("b", True), _row("a", False)]
    pairs = pair_by_item(x, c)
    assert [(p[0]["item_id"], p[1]["item_id"]) for p in pairs] == [("a", "a"), ("b", "b")]
    with pytest.raises(ValueError):
        pair_by_item(x, [_row("a", True)])  # one item missing on the comparison side
    with pytest.raises(ValueError):
        pair_by_item([_row("a", True), _row("a", True)], c)  # a duplicate id


def test_the_dev_parts_are_mr_general_and_knowledge_with_knowledge_inside_general() -> None:
    pairs = pair_by_item(
        [_row("m", True, source="mr_v1"), _row("k1", True, source="medqa"), _row("g", True, source="commonsenseqa"),
         _row("k2", True, source="medmcqa")],
        [_row("m", True, source="mr_v1"), _row("k1", True, source="medqa"), _row("g", True, source="commonsenseqa"),
         _row("k2", True, source="medmcqa")],
    )
    parts = partition_pairs(pairs)
    ids = {name: [p[0]["item_id"] for p in group] for name, group in parts.items()}
    assert ids == {"mr": ["m"], "general": ["k1", "g", "k2"], "knowledge": ["k1", "k2"]}


def test_keep_is_strict_at_every_margin() -> None:
    assert keep_verdict({"mr": 0.001, "general": -0.005, "knowledge": -0.01})["kept"] is True
    assert keep_verdict({"mr": 0.0, "general": -0.005, "knowledge": -0.01})["kept"] is False  # MR must be above 0
    assert keep_verdict({"mr": 0.01, "general": -0.01, "knowledge": -0.01})["kept"] is False  # general above -0.01
    assert keep_verdict({"mr": 0.01, "general": 0.0, "knowledge": -0.02})["kept"] is False  # knowledge above -0.02
    with pytest.raises(ValueError):
        keep_verdict({"mr": 0.1, "general": 0.0})


def test_the_knowledge_check_uses_the_same_margin_strictly() -> None:
    assert knowledge_check_passes(-0.019) is True
    assert knowledge_check_passes(-0.02) is False


def test_macro_and_micro_differ_when_templates_differ_in_size() -> None:
    pairs = pair_by_item(
        [_row(f"a{i}", True, template="small") for i in range(10)] + [_row(f"b{i}", False, template="big") for i in range(90)],
        [_row(f"a{i}", False, template="small") for i in range(10)] + [_row(f"b{i}", False, template="big") for i in range(90)],
    )
    assert macro_difference(pairs).point == pytest.approx(0.5)  # (1.0 + 0.0) / 2 templates
    assert micro_difference(pairs).point == pytest.approx(0.1)  # 10 of 100 items


def _three_parts(mr_correct: bool, other_correct: bool) -> list[dict]:
    """40 MR items, 10 general (not knowledge) and 10 knowledge items, all with the given correctness."""
    rows = [_row(f"m{i}", mr_correct, template=f"t{i % 2}", source="mr_v1") for i in range(40)]
    rows += [_row(f"g{i}", other_correct, template="g1", source="commonsenseqa") for i in range(10)]
    rows += [_row(f"k{i}", other_correct, template="k1", source="medqa") for i in range(10)]
    return rows


def test_identical_predictions_give_zero_and_do_not_keep_the_mr_part() -> None:
    same = _three_parts(mr_correct=True, other_correct=False)
    stats = keep_statistics(pair_by_item(same, same))
    assert stats["mr"].point == pytest.approx(0.0)
    assert stats["mr"].lo == pytest.approx(0.0)
    assert keep_verdict({p: s.lo for p, s in stats.items()})["checks"]["mr"] is False  # 0.0 is not above 0.0


def test_a_clear_gain_on_the_mr_part_passes_its_check() -> None:
    x = _three_parts(mr_correct=True, other_correct=False)
    c = _three_parts(mr_correct=False, other_correct=False)
    stats = keep_statistics(pair_by_item(x, c))
    assert stats["mr"].point == pytest.approx(1.0)
    assert stats["mr"].lo == pytest.approx(1.0)
    assert stats["general"].point == pytest.approx(0.0)
    assert keep_verdict({p: s.lo for p, s in stats.items()})["checks"]["mr"] is True


def test_the_keep_report_names_the_three_parts_and_the_decision() -> None:
    x = _three_parts(mr_correct=True, other_correct=True)
    c = _three_parts(mr_correct=False, other_correct=True)
    report = keep_comparison(x, c, candidate="B1", current="B0")
    assert report["items"] == {"mr": 40, "general": 20, "knowledge": 10}
    assert report["templates"]["mr"] == 2
    assert report["statistics"]["mr"]["ci_lo"] == pytest.approx(1.0)
    assert report["verdict"]["checks"] == {"mr": True, "general": True, "knowledge": True}
    assert report["verdict"]["kept"] is True


def _zs_row(item_id: str, correct: bool, template: str = "k1") -> dict:
    """An O2 prediction row for a two-option item whose gold is ``A``: the argmax is ``A`` when ``correct``."""
    return {"item_id": item_id, "template_id": template, "option_keys": ["A", "B"], "gold": "A", "status": "scored",
            "reason": None, "probs": [0.9, 0.1] if correct else [0.1, 0.9]}


def test_a_failing_zero_shot_cell_gives_not_measured_and_no_accuracy() -> None:
    block = [_row("k0", True, source="medqa"), _row("k1", False, source="medqa")]
    zs = [_zs_row("k0", True), _zs_row("k1", False)]
    report = knowledge_comparison(block, zs, block="B0", comparison="zero-shot",
                                  health={"k1": "READOUT_FAIL — label mass below the gate"})
    assert report["status"].startswith("NOT MEASURED") and "k1" in report["status"]
    assert "statistic" not in report and "passes" not in report


def test_a_passing_gate_gives_the_knowledge_difference_and_its_check() -> None:
    block = [_row(f"k{i}", True, template="k1", source="medqa") for i in range(20)]
    zs = [_zs_row(f"k{i}", i % 2 == 0) for i in range(20)]  # half of the zero-shot answers are correct
    report = knowledge_comparison(block, zs, block="B0", comparison="zero-shot", health={"k1": "PASS"})
    assert report["status"] == "MEASURED"
    assert report["statistic"]["point"] == pytest.approx(0.5)
    assert report["passes"] is True


def test_skipped_zero_shot_items_are_refused_not_dropped() -> None:
    block = [_row("k0", True, source="medqa")]
    zs = [{"item_id": "k0", "template_id": "k1", "option_keys": ["A", "B"], "gold": "A", "status": "skipped",
           "reason": "too long", "probs": None}]
    with pytest.raises(ValueError):
        knowledge_comparison(block, zs, block="B0", comparison="zero-shot", health={"k1": "PASS"})
