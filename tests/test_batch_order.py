"""Arm C batch order: a permutation of the epoch, length-bucketed batches, no length trend, arms A/B streams."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from meddecide.eval.paired import spearman
from meddecide.train import data as arms_data
from meddecide.train.batch_order import (
    BATCH_STREAM,
    CHUNK_STREAM,
    length_bucketed_order,
    option_permutation,
    permute_options,
)
from meddecide.train.unsloth_rows import to_unsloth_rows

ROOT = Path(__file__).resolve().parents[1]
DATA_PY = ROOT / "src" / "meddecide" / "train" / "data.py"


def _batch_spreads(lengths: list[int], order: list[int], batch_size: int) -> np.ndarray:
    batches = [order[k : k + batch_size] for k in range(0, len(order) - batch_size + 1, batch_size)]
    return np.array([max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches])


def test_order_is_a_permutation_of_the_epoch():
    lengths = np.random.default_rng(1).integers(1, 8192, size=1_000).tolist()
    order = length_bucketed_order(lengths, batch_size=8, chunk_factor=100, seed=0)
    assert sorted(order) == list(range(1_000))


def test_batches_hold_similar_lengths():
    lengths = np.random.default_rng(2).integers(1, 8192, size=40_000).tolist()
    order = length_bucketed_order(lengths, batch_size=8, chunk_factor=100, seed=0)
    bucketed = _batch_spreads(lengths, order, 8).mean()
    shuffled = np.random.default_rng(3).permutation(40_000).tolist()
    random_spread = _batch_spreads(lengths, shuffled, 8).mean()
    assert bucketed < 0.2 * random_spread


def test_file_sorted_by_length_gives_no_batch_trend():
    # the file itself runs shortest to longest: the chunk visiting order must hide that. The size is the
    # real mix's (482,889 items = 604 chunks): with few chunks, a random chunk order alone gives a
    # Spearman of order 1 / sqrt(chunks), which would make the 0.1 bound a coin toss
    lengths = list(range(482_889))
    order = length_bucketed_order(lengths, batch_size=8, chunk_factor=100, seed=0)
    batch_means = [float(np.mean([lengths[i] for i in order[k : k + 8]])) for k in range(0, len(order), 8)]
    rho = spearman(list(range(len(batch_means))), batch_means)
    assert abs(rho) < 0.1


def test_seed_and_epoch_select_the_order():
    lengths = np.random.default_rng(4).integers(1, 8192, size=5_000).tolist()
    first = length_bucketed_order(lengths, batch_size=8, chunk_factor=100, seed=0)
    assert first == length_bucketed_order(lengths, batch_size=8, chunk_factor=100, seed=0)
    assert first != length_bucketed_order(lengths, batch_size=8, chunk_factor=100, seed=1)
    assert first != length_bucketed_order(lengths, batch_size=8, chunk_factor=100, seed=0, epoch=1)


def test_rejects_non_positive_sizes():
    with pytest.raises(ValueError):
        length_bucketed_order([1, 2], batch_size=0, chunk_factor=100, seed=0)


def test_stream_constants_are_the_arms_a_and_b_ones():
    assert (CHUNK_STREAM, BATCH_STREAM) == (3131, 7717)
    source = DATA_PY.read_text(encoding="utf-8")
    assert "[int(seed), 3131, int(epoch)]" in source
    assert "[int(seed), 7717, int(epoch), int(chunk_index)]" in source
    assert "chunk_size = max(int(batch_size), int(chunk_factor) * int(batch_size))" in source


def test_option_permutation_is_the_arms_a_and_b_one():
    for n in range(1, 9):
        for index in range(0, 60, 7):
            ours = option_permutation(n, seed=0, epoch=0, index=index)
            assert ours == arms_data.option_permutation(n, seed=0, epoch=0, index=index)


def test_permute_options_keeps_content_and_relabels_by_position():
    rows = [
        {"qtype": "choice", "options": [{"key": "A", "label": "alpha"}, {"key": "B", "label": "beta"},
                                        {"key": "C", "label": "gamma"}], "gold": "B"},
        {"qtype": "score", "options": [{"key": "1", "label": "low"}, {"key": "2", "label": "mid"},
                                       {"key": "3", "label": "high"}], "gold": "3"},
        {"qtype": "noul", "options": [{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}],
         "gold": "yes"},
    ]
    out, changed = permute_options(rows, seed=0, epoch=0)
    choice = out[0]
    assert [o["key"] for o in choice["options"]] == ["A", "B", "C"]
    assert {o["key"]: o["label"] for o in choice["options"]}[choice["gold"]] == "beta"
    score = out[1]
    assert [o["key"] for o in score["options"]] == ["1", "2", "3"]
    assert {o["key"]: o["label"] for o in score["options"]}[score["gold"]] == "high"
    assert out[2] == rows[2]
    expected = sum(option_permutation(n, seed=0, epoch=0, index=i) != list(range(n))
                   for i, n in [(0, 3), (1, 3)])
    assert changed == expected


def test_permute_options_removes_a_gold_position_bias():
    # a file whose gold is always the first option: after augmentation each position holds about a quarter
    rows = [{"qtype": "choice", "options": [{"key": k, "label": f"x{k}"} for k in "ABCD"], "gold": "A"}
            for _ in range(4_000)]
    out, _ = permute_options(rows, seed=0, epoch=0)
    counts = {k: sum(1 for r in out if r["gold"] == k) for k in "ABCD"}
    assert all(900 <= c <= 1_100 for c in counts.values()), counts


def test_converter_reads_the_permuted_rows_with_the_gold_on_its_content():
    row = {"item_id": "t1", "template_id": "t", "qtype": "choice", "state": "s", "question": "q",
           "options": [{"key": "A", "label": "alpha"}, {"key": "B", "label": "beta"},
                       {"key": "C", "label": "gamma"}], "gold": "C"}
    out, _ = permute_options([row], seed=0, epoch=0)
    converted, refused = to_unsloth_rows(out)
    assert refused == {}
    question = converted[0]["questions"]["q"]
    gold_key = converted[0]["gold"]["q"]["label"]
    assert question["criteria"][gold_key] == "gamma"
    assert list(question["criteria"]) == ["A", "B", "C"]
