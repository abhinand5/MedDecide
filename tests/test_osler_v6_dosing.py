"""Unit tests for the osler_v6 D57 dosing sampler (``meddecide.train.dosing``) and its config guards.

CPU-only: a fake encoder reports a fixed token count per item, so the quota and the token-cap behaviour are exercised
without a model. The pre-registered properties are: each batch holds at least the labelled quota, the labelled anchors
are drawn without replacement per epoch, the stream is a pure function of ``(seed, epoch, items)``, and no fill item is
silently dropped (the soft-cap path trains a long fill item instead of deferring it forever).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from meddecide.train.config import StudentConfig
from meddecide.train.dosing import DosingStats, dosed_batches, labelled_positions


def make_items(n_labelled: int, n_fill: int, *, tokens: int = 10) -> list[SimpleNamespace]:
    """Items with a teacher distribution (``L*``) and without (``F*``); the fake encoder reads ``n_tokens``."""
    items = []
    for i in range(n_labelled):
        items.append(SimpleNamespace(item_id=f"L{i}", teacher_probs=[0.5, 0.5], n_options=2, n_tokens=tokens))
    for i in range(n_fill):
        items.append(SimpleNamespace(item_id=f"F{i}", teacher_probs=None, n_options=2, n_tokens=tokens))
    return items


class FakeEncoder:
    """A tokenizer stand-in: the token count is a local function of the item."""

    def __init__(self) -> None:
        self.permutations = 0

    def encode_item(self, item, permutation=None, max_prompt_tokens: int = 8192):
        if permutation is not None:
            self.permutations += 1
        n_tokens = int(min(item.n_tokens, max_prompt_tokens))
        return SimpleNamespace(item_id=item.item_id, n_tokens=n_tokens, truncated=item.n_tokens > max_prompt_tokens)


def run_batches(items, *, batch_size=8, min_labelled=3, max_batch_tokens=10_000, epoch=0, seed=0, stats=None):
    model = FakeEncoder()
    batches = list(dosed_batches(
        model, items, batch_size=batch_size, min_labelled=min_labelled,
        max_batch_tokens=max_batch_tokens, max_prompt_tokens=8192,
        augment_options=False, epoch=epoch, seed=seed, stats=stats,
    ))
    return batches


def test_labelled_positions_reports_the_teacher_items() -> None:
    items = make_items(3, 4)
    assert labelled_positions(items) == [0, 1, 2]


def test_every_batch_holds_the_quota_and_labelled_are_drawn_without_replacement() -> None:
    items = make_items(30, 100)
    batches = run_batches(items)
    assert len(batches) == 10
    seen: list[str] = []
    for _, batch in batches:
        labelled = [encoded.item_id for encoded in batch if encoded.item_id.startswith("L")]
        assert len(labelled) == 3
        seen.extend(labelled)
    assert sorted(seen) == sorted(f"L{i}" for i in range(30))  # every labelled item exactly once per epoch


def test_fill_items_are_not_dropped() -> None:
    items = make_items(12, 20)  # 4 batches x 5 fill slots for 20 fill items: the epoch consumes every one once
    batches = run_batches(items)
    assert len(batches) == 4
    fill = [encoded.item_id for _, batch in batches for encoded in batch if encoded.item_id.startswith("F")]
    assert len(fill) == 20
    assert set(fill) == {f"F{i}" for i in range(20)}
    assert len(fill) == len(set(fill))  # without replacement within the epoch


def test_no_item_is_repeated_within_a_batch() -> None:
    items = make_items(30, 100)
    for _, batch in run_batches(items):
        ids = [encoded.item_id for encoded in batch]
        assert len(ids) == len(set(ids))


def test_stream_is_deterministic_in_seed_and_epoch() -> None:
    items = make_items(30, 100)
    first = [[e.item_id for e in b] for _, b in run_batches(items, seed=7, epoch=2)]
    again = [[e.item_id for e in b] for _, b in run_batches(items, seed=7, epoch=2)]
    other = [[e.item_id for e in b] for _, b in run_batches(items, seed=8, epoch=2)]
    assert first == again
    assert first != other


def test_batches_respect_the_token_cap_and_fill_until_it_binds() -> None:
    items = make_items(30, 100, tokens=100)
    batches = run_batches(items, max_batch_tokens=500)  # the quota is 300 tokens; two more items reach 500
    for _, batch in batches:
        assert len(batch) == 5
        assert sum(encoded.n_tokens for encoded in batch) == 500


def test_first_fill_item_that_does_not_fit_is_trained_under_the_soft_cap() -> None:
    items = make_items(30, 100, tokens=100)
    for item in items:
        if item.item_id.startswith("F"):
            item.n_tokens = 5_000
    stats = DosingStats(min_labelled=3)
    batches = run_batches(items, max_batch_tokens=500, stats=stats)
    for _, batch in batches:
        assert len(batch) == 4  # quota + the long fill item that cannot wait
        assert sum(encoded.n_tokens for encoded in batch) == 5_300
    assert stats.soft_cap_batches == stats.batches == 10
    assert stats.batches_meeting_minimum == 10


def test_partial_final_batch_is_counted_honestly() -> None:
    items = make_items(7, 20)  # quotas 3, 3, then 1 at the epoch's end
    stats = DosingStats(min_labelled=3)
    batches = run_batches(items, stats=stats)
    assert [len([e for e in batch if e.item_id.startswith("L")]) for _, batch in batches] == [3, 3, 1]
    assert stats.batches == 3
    assert stats.batches_meeting_minimum == 2
    assert stats.to_dict()["share_batches_meeting_minimum"] == pytest.approx(2 / 3)
    assert stats.to_dict()["labelled_histogram"] == {"1": 1, "3": 2}


def test_label_only_batch_when_the_quota_alone_exceeds_the_cap() -> None:
    items = make_items(9, 20, tokens=100)
    stats = DosingStats(min_labelled=3)
    batches = run_batches(items, max_batch_tokens=200, stats=stats)  # 3 x 100 > 200: no fill is added
    for _, batch in batches:
        assert len(batch) == 3
    assert stats.labelled_only_batches == 3


def test_augmentation_permutes_every_encoded_item() -> None:
    items = make_items(3, 10)
    model = FakeEncoder()
    list(dosed_batches(
        model, items, batch_size=8, min_labelled=3, max_batch_tokens=10_000, max_prompt_tokens=8192,
        augment_options=True, epoch=0, seed=0,
    ))
    assert model.permutations == 8  # one batch: 3 labelled + 5 fill


def test_the_sampler_refuses_an_empty_stratum() -> None:
    with pytest.raises(ValueError, match="at least one item with a teacher"):
        run_batches(make_items(0, 5))
    with pytest.raises(ValueError, match="unlabelled items"):
        run_batches(make_items(3, 0))
    with pytest.raises(ValueError, match="min_labelled"):
        run_batches(make_items(3, 5), batch_size=8, min_labelled=9)


def test_config_guards_the_quota_and_the_probe() -> None:
    with pytest.raises(ValueError, match="must not exceed batch_size"):
        StudentConfig.from_dict({"distill_min_labelled_per_batch": 9, "distill_weight": 1.0})
    with pytest.raises(ValueError, match="needs a positive distill_weight"):
        StudentConfig.from_dict({"distill_min_labelled_per_batch": 3})
    with pytest.raises(ValueError, match="distill_probe"):
        StudentConfig.from_dict({"distill_min_labelled_per_batch": 3, "distill_weight": 1.0, "distill_probe": True,
                                 "kl_to_base": 1.0, "readout": "option_code"})
    cfg = StudentConfig.from_dict({"distill_min_labelled_per_batch": 3, "distill_weight": 1.0, "distill_probe": True})
    assert cfg.distill_min_labelled_per_batch == 3 and cfg.distill_probe
