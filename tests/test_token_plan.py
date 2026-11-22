"""Unit tests for the osler_v7 4B token-capped batch plan and the prepared-row cache (no model, no GPU)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from meddecide.train.prepared import PreparedRows, build_prepared, verify_rows
from meddecide.train.token_plan import (
    build_token_plan,
    flagged_batch_share,
    load_plan,
    padded_width,
    plan_statistics,
    save_plan,
)


def test_padded_width_rounds_up_to_the_attention_chunk() -> None:
    assert padded_width(1) == 64
    assert padded_width(64) == 64
    assert padded_width(65) == 128
    with pytest.raises(ValueError):
        padded_width(0)


def test_plan_covers_every_row_once_and_respects_both_caps() -> None:
    lengths = [10, 30, 20, 40, 5] + [200] * 20 + [3_000, 7_000]
    plan = build_token_plan(lengths, batch_size=8, max_tokens=8_192, seed=0)
    assert sorted(index for batch in plan.batches for index in batch) == list(range(len(lengths)))
    assert all(len(batch) <= 8 for batch in plan.batches)
    for batch in plan.batches:
        width = padded_width(max(lengths[index] for index in batch))
        assert width * len(batch) <= 8_192


def test_a_row_longer_than_the_cap_still_gets_its_own_batch() -> None:
    plan = build_token_plan([20_000, 10, 10], batch_size=8, max_tokens=8_192, seed=0)
    longest = [batch for batch in plan.batches if 0 in batch]
    assert len(longest) == 1 and longest[0] == (0,)


def test_short_rows_pack_to_the_item_cap() -> None:
    plan = build_token_plan([100] * 20, batch_size=8, max_tokens=8_192, seed=0)
    assert sorted(len(batch) for batch in plan.batches) == [4, 8, 8]  # the batch order itself is shuffled


def test_plan_is_deterministic_and_its_step_map_cycles() -> None:
    lengths = [50] * 30
    first = build_token_plan(lengths, batch_size=4, max_tokens=1_000, seed=0)
    again = build_token_plan(lengths, batch_size=4, max_tokens=1_000, seed=0)
    other = build_token_plan(lengths, batch_size=4, max_tokens=1_000, seed=1)
    assert first.batches == again.batches
    assert [list(b) for b in first.batches] != [list(b) for b in other.batches]
    assert first.batch(1) == first.batches[0]
    assert first.batch(len(first) + 1) == first.batches[0]
    with pytest.raises(ValueError):
        first.batch(0)


def test_plan_statistics_report_the_realised_composition() -> None:
    plan = build_token_plan([100, 100, 100, 100], batch_size=2, max_tokens=8_192, seed=0)
    stats = plan_statistics(plan, [100, 100, 100, 100])
    assert stats["batches"] == 2 and stats["covered_items"] == 4
    assert stats["mean_items"] == 2.0
    assert stats["mean_real_tokens"] == 200.0
    assert stats["mean_padded_tokens"] == 2 * 128  # the collate rounds a 100-token row to 128
    assert stats["max_padded_tokens"] == 256


def test_flagged_batch_share_counts_steps_with_a_matching_row() -> None:
    plan = build_token_plan([10, 10, 10, 10], batch_size=1, max_tokens=1_000, seed=0)
    assert flagged_batch_share(plan, [True, False, False, False]) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        flagged_batch_share(plan, [True])


def test_plan_round_trips_through_disk(tmp_path) -> None:
    plan = build_token_plan([10, 20, 30, 40, 50], batch_size=2, max_tokens=1_000, seed=3)
    path = save_plan(plan, tmp_path / "plan.npz")
    loaded = load_plan(path)
    assert loaded == plan


class _Row:
    """The subset of ``Item`` the encoders below read."""

    def __init__(self, index: int, template_id: str, n_options: int, length: int, teacher: bool) -> None:
        self.index = index
        self.template_id = template_id
        self.n_options = n_options
        self.length = length
        self.teacher = teacher


def _fake_encoded(item: _Row):
    from meddecide.model.meddecide_model import EncodedItem

    marker = item.length - 1
    return EncodedItem(
        item_id=f"row-{item.index}", qtype="choice", template_id=item.template_id, source="test",
        option_keys=[f"K{i}" for i in range(item.n_options)],
        original_option_keys=[f"orig{i}" for i in range(item.n_options)],
        gold_index=1 % item.n_options, input_ids=list(range(item.length)),
        marker_positions=[marker] * item.n_options, option_end_positions=[marker] * item.n_options,
        answer_position=marker, prompt_tokens=item.length, truncated=False,
        teacher_probs=[1.0 / item.n_options] * item.n_options if item.teacher else None,
    )


def _rows() -> list[_Row]:
    return [
        _Row(0, "t_a", 2, 10, True),
        _Row(1, "t_b", 3, 20, False),
        _Row(2, "t_a", 2, 5, False),
    ]


def test_prepared_cache_round_trips_every_field(tmp_path) -> None:
    rows = _rows()
    prepared = build_prepared(len(rows), lambda i: _fake_encoded(rows[i]), option_codes=["K0", "K1", "K2"],
                              progress_every=0)
    assert len(prepared) == 3
    assert prepared.lengths.tolist() == [10, 20, 5]
    assert prepared.labelled().tolist() == [True, False, False]
    path = prepared.save(tmp_path / "prepared.npz")
    loaded = PreparedRows.load(path)
    for index in range(3):
        a, b = prepared.encoded(index), loaded.encoded(index)
        assert a.input_ids == b.input_ids and a.marker_positions == b.marker_positions
        assert a.gold_index == b.gold_index and a.template_id == b.template_id and a.item_id == b.item_id
        assert a.teacher_probs == b.teacher_probs
        assert a.option_keys == b.option_keys  # rebuilt from the cache's option codes, as the encoder wrote them
    assert loaded.templates == ["t_a", "t_b"]
    assert loaded.option_codes == ["K0", "K1", "K2"]


def test_verify_rows_detects_a_changed_encoding() -> None:
    rows = _rows()
    prepared = build_prepared(len(rows), lambda i: _fake_encoded(rows[i]), option_codes=["K0", "K1", "K2"],
                              progress_every=0)
    live = [_fake_encoded(item) for item in rows]
    assert verify_rows(prepared, live, [0, 1, 2]) == {"checked": 3, "mismatched": [], "identical": True}
    live[1] = dataclasses.replace(live[1], input_ids=[1, 2, 3])
    assert verify_rows(prepared, live, [0, 1, 2])["mismatched"] == [1]


def test_prepared_cache_rejects_a_different_layout(tmp_path) -> None:
    rows = _rows()
    prepared = build_prepared(len(rows), lambda i: _fake_encoded(rows[i]), option_codes=["K0"], progress_every=0)
    path = tmp_path / "prepared.npz"
    prepared.save(path)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload["meta"] = np.array('{"version": 999}')
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="cache version"):
        PreparedRows.load(path)
