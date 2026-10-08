"""student_v1 mix rules: the 8 % share cap is a fixed point, and sampling is deterministic."""

from __future__ import annotations

import pytest

from meddecide.train.mix_v1 import (
    cap_fixed_point,
    cap_items,
    describe,
    max_share,
    stratified_sample,
)


def test_cap_fixed_point_on_a_small_example() -> None:
    counts = {"a": 100, "b": 10, "c": 10}
    cap = cap_fixed_point(counts, share=0.5)
    assert cap == 20
    mix = sum(min(n, cap) for n in counts.values())
    assert int(0.5 * mix) == cap


@pytest.mark.parametrize("counts", [
    {"a": 5000, "b": 3000, "c": 2000, "d": 800},
    {"x": 10**6, "y": 10**6, "z": 1},
    {f"t{i}": 40 * (i + 1) for i in range(25)},
])
def test_capped_mix_has_no_template_above_the_share(counts: dict[str, int]) -> None:
    cap = cap_fixed_point(counts)
    capped = {t: min(n, cap) for t, n in counts.items()}
    assert max_share(capped) <= 0.08 + 1e-12
    assert int(0.08 * sum(capped.values())) == cap  # fixed point


def test_cap_fixed_point_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        cap_fixed_point({})


def test_cap_items_keeps_the_first_in_key_order() -> None:
    items = {"t": [{"id": "c"}, {"id": "a"}, {"id": "b"}]}
    kept = cap_items(items, 2, key=lambda row: row["id"])
    assert [row["id"] for row in kept["t"]] == ["a", "b"]


def test_stratified_sample_is_deterministic_and_bounded() -> None:
    items = {"t1": list(range(10, 0, -1)), "t2": [7, 3]}
    first = stratified_sample(items, 4, key=lambda v: v)
    second = stratified_sample(items, 4, key=lambda v: v)
    assert first == second == {"t1": [1, 2, 3, 4], "t2": [3, 7]}


def test_describe_reports_total_templates_and_share() -> None:
    summary = describe({"a": 3, "b": 1})
    assert summary == {"total": 4, "templates": 2, "max_template_share": 0.75}


def test_balance_classes_keeps_the_smallest_label_count_for_each_label() -> None:
    from meddecide.train.mix_v1 import balance_classes

    rows = [("yes", i) for i in range(2)] + [("no", i) for i in range(7)]
    kept = balance_classes(rows, label=lambda r: r[0], key=lambda r: r[1])
    assert sorted(kept) == [("no", 0), ("no", 1), ("yes", 0), ("yes", 1)]


def test_balanced_sample_caps_each_label_and_keeps_small_labels_whole() -> None:
    from meddecide.train.mix_v1 import balanced_sample

    rows = [("yes", i) for i in range(3)] + [("no", i) for i in range(10)]
    kept = balanced_sample(rows, per_label=4, label=lambda r: r[0], key=lambda r: r[1])
    assert sum(1 for r in kept if r[0] == "yes") == 3
    assert sum(1 for r in kept if r[0] == "no") == 4
