"""Hand-computed tests for the harness plumbing: transforms, flip rate, summarise().

These run on synthetic ``Item`` objects and hand-built ``Prediction`` objects — no model, no
GPU, no network.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from meddecide.bench.schema import Item, Option, QuestionType, Split, Tier, make_item
from meddecide.eval.harness import Prediction, summarise
from meddecide.eval.probes import (
    FILLER_OPTIONS,
    NONE_OF_THE_ABOVE_LABEL,
    detect_abstention,
    none_of_the_above,
    scale_candidates,
    shuffle_options,
)


def _item(*, qtype=QuestionType.CHOICE, n_options=4, split=Split.TEST, seed=1) -> Item:
    if qtype is QuestionType.NOUL:
        options = [Option(key="yes", label="Yes"), Option(key="no", label="No")]
    elif qtype is QuestionType.SCORE:
        options = [Option(key=str(i + 1), label=f"Level {i + 1}") for i in range(n_options)]
    else:
        options = [Option(key=chr(ord("A") + i), label=f"Option {i + 1}") for i in range(n_options)]
    return make_item(
        tier=Tier.FRESH,
        source="synthetic",
        source_record_id="r1",
        source_url="https://example.org/r1",
        source_license="synthetic",
        record_date=date(2026, 1, 1),
        split=split,
        template_id="t1",
        skill="test",
        qtype=qtype,
        state="state text",
        question="question text?",
        options=options,
        gold=options[1].key,
        option_order_seed=seed,
    )


def _prediction(item: Item, probs, *, model="m", latency=0.01) -> Prediction:
    probs = list(probs)
    argmax = int(np.argmax(probs))
    return Prediction(
        item_id=item.item_id,
        model_id=model,
        source=item.source,
        template_id=item.template_id,
        split=str(item.split),
        qtype=str(item.qtype),
        option_keys=item.option_keys,
        option_probs=probs,
        label_mass=0.5,
        argmax_index=argmax,
        gold_key=item.gold,
        correct=argmax == item.gold_index,
        expected_level=None,
        latency_s=latency,
        variant="space",
        prompt_tokens=10,
    )


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------
def test_shuffle_preserves_gold_content_and_is_deterministic() -> None:
    item = _item()
    gold_label = item.options[item.gold_index].label
    a, rec_a = shuffle_options(item, seed=0)
    b, rec_b = shuffle_options(item, seed=0)
    assert [o.label for o in a.options] == [o.label for o in b.options]
    assert rec_a.details["perm"] == rec_b.details["perm"]
    assert a.options[a.gold_index].label == gold_label
    assert a.item_id != item.item_id
    assert a.item_id == b.item_id
    # re-keyed contiguously in display order
    assert a.option_keys == [chr(ord("A") + i) for i in range(item.n_options)]


def test_shuffle_of_score_item_keeps_numeric_keys_ordered() -> None:
    item = _item(qtype=QuestionType.SCORE, n_options=4)
    shuffled, _ = shuffle_options(item, seed=3)
    assert shuffled.option_keys == ["1", "2", "3", "4"]
    assert shuffled.gold == shuffled.options[shuffled.gold_index].key


def test_scale_candidates_adds_fillers_and_never_moves_gold() -> None:
    item = _item(n_options=4)
    gold_label = item.options[item.gold_index].label
    scaled, rec = scale_candidates(item, target_n=16, seed=0)
    assert scaled is not None and rec is not None
    assert scaled.n_options == 16
    assert scaled.options[scaled.gold_index].label == gold_label
    assert rec.details["n_filler"] == 12
    for label, _desc in FILLER_OPTIONS:
        if label in rec.details["fillers"]:
            assert label != gold_label
    # deterministic
    again, rec2 = scale_candidates(item, target_n=16, seed=0)
    assert again is not None and rec2 is not None
    assert [o.label for o in again.options] == [o.label for o in scaled.options]
    assert again.item_id == scaled.item_id


def test_scale_candidates_refuses_impossible_targets() -> None:
    assert scale_candidates(_item(n_options=4), target_n=2, seed=0) == (None, None)
    assert scale_candidates(_item(n_options=4), target_n=64, seed=0) == (None, None)
    assert scale_candidates(_item(qtype=QuestionType.NOUL), target_n=8, seed=0) == (None, None)
    assert scale_candidates(_item(qtype=QuestionType.SCORE, n_options=4), target_n=8, seed=0) == (None, None)


def test_scale_candidates_target_equal_to_current_is_identity() -> None:
    item = _item(n_options=4)
    scaled, rec = scale_candidates(item, target_n=4, seed=0)
    assert scaled is not None and rec is not None
    assert scaled.item_id == item.item_id
    assert rec.details["n_filler"] == 0


def test_none_of_the_above_removes_gold_and_adds_option() -> None:
    item = _item(n_options=4)
    gold_label = item.options[item.gold_index].label
    nota, rec = none_of_the_above(item, seed=0)
    assert nota.n_options == item.n_options  # one removed, one added
    labels = [o.label for o in nota.options]
    assert gold_label not in labels
    assert NONE_OF_THE_ABOVE_LABEL in labels
    assert nota.options[nota.gold_index].label == NONE_OF_THE_ABOVE_LABEL
    assert rec.details["removed_label"] == gold_label
    again, _ = none_of_the_above(item, seed=0)
    assert [o.label for o in again.options] == labels


def test_detect_abstention_hand_computed() -> None:
    report = detect_abstention(
        nota_probabilities={"i1": [0.1, 0.2, 0.7], "i2": [0.5, 0.3, 0.2]},
        nota_gold_key={"i1": "C", "i2": "C"},
        nota_option_keys={"i1": ["A", "B", "C"], "i2": ["A", "B", "C"]},
    )
    assert report["n_abstention_items"] == 2
    assert report["detection_rate"] == pytest.approx(0.5)
    # gold key need not be last
    report = detect_abstention(
        nota_probabilities={"i1": [0.7, 0.2, 0.1]},
        nota_gold_key={"i1": "A"},
        nota_option_keys={"i1": ["A", "B", "C"]},
    )
    assert report["detection_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------
def test_summarise_hand_computed_group() -> None:
    items = [_item(seed=i) for i in range(4)]
    # gold is index 1 ("B"); predict correctly on 3 of 4, with one confident error
    predictions = [
        _prediction(items[0], [0.1, 0.8, 0.05, 0.05]),
        _prediction(items[1], [0.1, 0.7, 0.1, 0.1]),
        _prediction(items[2], [0.1, 0.6, 0.2, 0.1]),
        _prediction(items[3], [0.6, 0.3, 0.05, 0.05]),
    ]
    summary = summarise(predictions, n_resamples=200, seed=0)
    (entry,) = summary["groups"].values()
    assert entry["n"] == 4
    assert entry["n_correct"] == 3
    assert entry["accuracy"] == pytest.approx(0.75)
    assert entry["majority_baseline"] == pytest.approx(1.0)  # every gold is "B"
    assert entry["qtype"] == "choice"
    # Brier: rows 1-3 put 0.8/0.7/0.6 on gold, row 4 puts 0.3
    expected_brier = np.mean(
        [
            (0.8 - 1) ** 2 + 0.1**2 + 0.05**2 + 0.05**2,
            (0.7 - 1) ** 2 + 0.1**2 + 0.1**2 + 0.1**2,
            (0.6 - 1) ** 2 + 0.1**2 + 0.2**2 + 0.1**2,
            (0.3 - 1) ** 2 + 0.6**2 + 0.05**2 + 0.05**2,
        ]
    )
    assert entry["brier"] == pytest.approx(expected_brier, abs=1e-9)
    assert entry["accuracy_ci95"][0] <= entry["accuracy"] <= entry["accuracy_ci95"][1]
    assert summary["n_predictions"] == 4


def test_summarise_score_group_reports_level_error() -> None:
    items = [_item(qtype=QuestionType.SCORE, n_options=4, seed=i) for i in range(2)]
    # gold is level "2"; predictions of 1.0 and 3.0 -> mean abs error 1.0
    p1 = _prediction(items[0], [1.0, 0.0, 0.0, 0.0])
    p1.expected_level = 1.0
    p2 = _prediction(items[1], [0.0, 0.0, 0.0, 1.0])
    p2.expected_level = 3.0
    summary = summarise([p1, p2], n_resamples=100)
    (entry,) = summary["groups"].values()
    assert entry["qtype"] == "score"
    assert entry["mean_gold_level"] == pytest.approx(2.0)
    assert entry["mean_expected_level"] == pytest.approx(2.0)
    assert entry["mean_level_abs_error"] == pytest.approx(1.0)


def test_summarise_groups_by_template_and_split() -> None:
    a = [_prediction(_item(seed=i), [0.2, 0.8, 0.0, 0.0]) for i in range(2)]
    b = [_prediction(_item(split=Split.DEV, seed=i), [0.2, 0.8, 0.0, 0.0]) for i in range(3)]
    summary = summarise([*a, *b], n_resamples=100)
    assert len(summary["groups"]) == 2
    sizes = sorted(v["n"] for v in summary["groups"].values())
    assert sizes == [2, 3]
    for key in summary["groups"]:
        assert "|test" in key or "|dev" in key
