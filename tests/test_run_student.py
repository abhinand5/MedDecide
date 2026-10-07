"""Unit tests for the S8 student evaluation path (``scripts/bench/run_student.py``).

Every test here is CPU-only and needs no trained checkpoint: the acceptance-relevant behaviour
is factored into pure functions and exercised with hand-built inputs.

Covered (the S8 acceptance list):

* the prediction row carries every column the reports need, and ``predlog`` deduplicates it by
  ``(run_id, item_id)``;
* a constant-answer cell is flagged by the constant-answer check and a below-chance cell by the
  accuracy-CI check — and the two applicable D12 checks agree with
  :func:`meddecide.eval.health.evaluate_cell` on the same inputs;
* the two letter-readout D12 checks are recorded as ``NOT APPLICABLE`` with their reason and are
  never reported as passed;
* the ``score`` expected-level error is computed correctly on hand-built cases;
* the temperature fit refuses a test split, and reports instead of guessing when a qtype has too
  few items or the search hits a bound;
* the additional option-permutation check and the selective-accuracy metric;
* ``decisions_from_scored`` produces exactly what ``MedDecideModel.predict`` produces.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from meddecide.bench.schema import Item, Option, QuestionType, Tier, make_item
from meddecide.eval.harness import Prediction
from meddecide.eval.health import evaluate_cell
from meddecide.eval.predlog import read_prediction_log
from meddecide.model.meddecide_model import Decision, MedDecideModel, ScoredItems

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "run_student.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_student", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


rs = _load_runner()

REQUIRED_ROW_COLUMNS = (
    "item_id",
    "model_id",
    "run_id",
    "source",
    "template_id",
    "split",
    "qtype",
    "option_keys",
    "option_probs",
    "argmax_key",
    "gold_key",
    "correct",
    "expected_level",
    "latency_s",
    "prompt_tokens",
)


# --------------------------------------------------------------------------- helpers
def _item(*, qtype: QuestionType = QuestionType.CHOICE, n_options: int = 4, seed: int = 0,
          split: str = "test", template_id: str = "synthetic_choice_v1") -> Item:
    if qtype is QuestionType.NOUL:
        options = [Option(key="yes", label="Yes"), Option(key="no", label="No")]
        gold = "yes" if seed % 2 == 0 else "no"
    elif qtype is QuestionType.SCORE:
        options = [Option(key=str(i + 1), label=f"Level {i + 1}") for i in range(n_options)]
        gold = str(1 + seed % n_options)
    else:
        options = [
            Option(key=chr(ord("A") + i), label=f"Candidate answer {i + 1}")
            for i in range(n_options)
        ]
        gold = options[seed % n_options].key
    return make_item(
        tier=Tier.FRESH,
        source="synthetic_s8",
        source_record_id=f"r{seed}-{n_options}",
        source_url="https://example.org/synthetic",
        source_license="synthetic",
        record_date=date(2026, 1, 1),
        split=split,
        template_id=template_id,
        skill="synthetic",
        qtype=qtype,
        state="A 61-year-old patient with hypertension presents with headache.",
        question="Which of the following is the most appropriate next step?",
        options=options,
        gold=gold,
        option_order_seed=seed,
    )


def _decision(
    item: Item,
    probs: list[float],
    *,
    gold_key: str | None = None,
    latency_s: float = 0.01,
    prompt_tokens: int = 100,
) -> Decision:
    keys = [o.key for o in item.options]
    gold = gold_key or item.gold
    gold_index = keys.index(gold)
    argmax = int(np.argmax(probs))
    return Decision(
        item_id=item.item_id,
        qtype=str(item.qtype),
        template_id=item.template_id,
        source=item.source,
        option_keys=keys,
        original_option_keys=keys,
        probs=probs,
        argmax_index=argmax,
        gold_key=gold,
        correct=argmax == gold_index,
        expected_level=(
            float(sum((i + 1) * p for i, p in enumerate(probs)))
            if item.qtype is QuestionType.SCORE
            else None
        ),
        n_options=len(keys),
        prompt_tokens=prompt_tokens,
        latency_s=latency_s,
        temperature=1.0,
    )


def _prediction(item: Item, probs: list[float], **kwargs) -> Prediction:
    return rs.decision_to_prediction(
        _decision(item, probs, **kwargs),
        model_id="meddecide-0.8b-lora-pointer",
        run_id="run-1",
        split=str(item.split),
    )


def _gate(**overrides):
    """A healthy default cell (half right, varied readout); overrides make it fail."""
    n = overrides.pop("n", 40)
    gold = [chr(ord("A") + i % 4) for i in range(n)]
    predicted = [g if i % 2 == 0 else chr(ord("A") + (i + 1) % 4)
                 for i, g in enumerate(gold)]
    kwargs = {
        "correct": [p == g for p, g in zip(predicted, gold, strict=True)],
        "predicted_labels": predicted,
        "gold_labels": gold,
        "n_options": 4,
        "seed": 0,
        "n_resamples": 500,
    }
    kwargs.update(overrides)
    return rs.pointer_head_gate(**kwargs)


# --------------------------------------------------------------------------- (a) prediction rows
def test_prediction_row_has_every_column_the_reports_need(tmp_path) -> None:
    item = _item(seed=1)
    pred = _prediction(item, [0.1, 0.6, 0.2, 0.1])
    row = pred.to_json()
    for column in REQUIRED_ROW_COLUMNS:
        assert column in row, column
    assert row["run_id"] == "run-1"
    assert row["model_id"] == "meddecide-0.8b-lora-pointer"
    assert row["split"] == "test"
    assert row["qtype"] == "choice"
    assert row["argmax_key"] == row["option_keys"][pred.argmax_index]
    assert row["correct"] is (row["argmax_key"] == row["gold_key"])
    assert row["expected_level"] is None
    assert row["variant"] == rs.POINTER_VARIANT
    # the letter-readout-only diagnostics are marked, not silently filled with a measurement
    assert row["transform"]["readout"] == rs.POINTER_VARIANT
    assert sum(row["option_probs"]) == pytest.approx(1.0, abs=1e-6)

    # `predlog` must be able to read the file and dedupe by (run_id, item_id), last row wins
    path = tmp_path / "preds.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    rerun = dict(row)
    rerun["run_id"] = "run-2"
    rerun["correct"] = not row["correct"]
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rerun) + "\n")
    log = read_prediction_log(path)
    assert log.n_raw == 2
    assert log.n_unique == 2  # two runs, one item: two rows, not one
    assert log.n_distinct_items == 1
    assert log.run_ids == ["run-1", "run-2"]
    assert {r["run_id"]: r["correct"] for r in log.rows} == {"run-1": row["correct"],
                                                            "run-2": rerun["correct"]}
    assert log.rows_for_run("run-2")[0]["correct"] == rerun["correct"]
    # a repeated row inside one run collapses (the resumed-run case)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    log2 = read_prediction_log(path)
    assert log2.n_raw == 3 and log2.n_unique == 2 and log2.n_duplicate_rows == 1


def test_score_prediction_carries_its_expected_level() -> None:
    item = _item(qtype=QuestionType.SCORE, n_options=3, seed=0)
    pred = _prediction(item, [0.2, 0.3, 0.5])
    assert pred.expected_level == pytest.approx(2.3)
    assert pred.to_json()["expected_level"] == pytest.approx(2.3)


# --------------------------------------------------------------------------- (b) the two applicable D12 checks
def test_constant_answer_cell_is_flagged() -> None:
    # every item answered "A" on a template whose gold is balanced over 4 classes
    gold = [chr(ord("A") + i % 4) for i in range(40)]
    gate = _gate(correct=[g == "A" for g in gold], predicted_labels=["A"] * 40, gold_labels=gold)
    assert gate["status"] == "READOUT_FAIL — constant_answer"
    assert gate["checks"]["constant_answer"]["status"] == "FAIL"
    assert gate["checks"]["constant_answer"]["modal_share"] == pytest.approx(1.0)
    assert "degenerate" in gate["checks"]["constant_answer"]["failure"]
    # the accuracy itself is still measured and reported beside the failure
    assert gate["checks"]["accuracy_ci_vs_chance"]["status"] == "PASS"
    assert gate["checks"]["accuracy_ci_vs_chance"]["accuracy"] == pytest.approx(0.25)


def test_below_chance_cell_is_flagged_by_the_ci_check() -> None:
    # a varied readout (so the constant-answer check stays quiet) that is never right
    gate = _gate(correct=[False] * 40)
    assert gate["status"] == "READOUT_FAIL — accuracy_ci_vs_chance"
    assert gate["checks"]["accuracy_ci_vs_chance"]["status"] == "FAIL"
    assert "below-chance" in gate["checks"]["accuracy_ci_vs_chance"]["failure"]
    assert gate["checks"]["constant_answer"]["status"] == "PASS"


def test_healthy_cell_passes_and_a_cell_at_chance_does_not_trip_the_ci() -> None:
    gold = [chr(ord("A") + i % 4) for i in range(40)]
    predicted = [g if i % 2 == 0 else chr(ord("A") + (i + 1) % 4)
                 for i, g in enumerate(gold)]
    gate = _gate(correct=[p == g for p, g in zip(predicted, gold, strict=True)],
                 predicted_labels=predicted, gold_labels=gold)
    assert gate["status"] == "PASS", gate["failures"]


def test_applicable_checks_agree_with_the_library_gate() -> None:
    """The two D12 checks reproduce `health.evaluate_cell`'s verdicts on the same inputs."""
    cases = {
        "passing": ([True, False, True, False] * 10,
                    [chr(ord("A") + i % 4) for i in range(40)],
                    [chr(ord("A") + i % 4) for i in range(40)]),
        "constant": ([g == "A" for g in [chr(ord("A") + i % 4) for i in range(40)]],
                     ["A"] * 40,
                     [chr(ord("A") + i % 4) for i in range(40)]),
        "below_chance": ([False] * 40,
                         [chr(ord("A") + i % 4) for i in range(40)],
                         [chr(ord("A") + i % 4) for i in range(40)]),
        "imbalanced_majority": ([True] * 30 + [False] * 10,
                                ["A"] * 40,
                                ["A"] * 30 + ["B"] * 10),
    }
    for name, (correct, predicted, gold) in cases.items():
        mine = _gate(correct=correct, predicted_labels=predicted, gold_labels=gold)
        theirs = evaluate_cell(
            model_id="m",
            template_id="t",
            # label mass is not a measurement for a pointer head; a passing value is passed so
            # that the library's mass check does not add a failure unrelated to the comparison
            label_masses=[1.0] * len(correct),
            correct=correct,
            greedy_matches=None,
            n_options=4,
            n_resamples=500,
            seed=0,
            predicted_labels=predicted,
            majority_share=Counter(gold).most_common(1)[0][1] / len(gold),
        )
        assert mine["checks"]["accuracy_ci_vs_chance"]["status"] == (
            "FAIL" if any("below-chance" in f for f in theirs.failures) else "PASS"
        ), name
        assert mine["checks"]["constant_answer"]["status"] == (
            "FAIL" if any("degenerate" in f for f in theirs.failures) else "PASS"
        ), name


# --------------------------------------------------------------------------- (c) inapplicable checks
def test_letter_readout_checks_are_not_applicable_and_never_reported_as_passed() -> None:
    gate = _gate()
    assert gate["status"] == "PASS"
    assert gate["applied_checks"] == ["accuracy_ci_vs_chance", "constant_answer"]
    assert gate["inapplicable_checks"] == ["median_label_mass", "greedy_agreement"]
    assert gate["inapplicable_reason"] == rs.NO_LETTER_READOUT_REASON
    for name in ("median_label_mass", "greedy_agreement"):
        check = gate["checks"][name]
        assert check["applied"] is False
        assert check["status"] == rs.NO_LETTER_READOUT_REASON
        assert check["status"].startswith("NOT APPLICABLE —")
        assert check["reason"] == rs.NO_LETTER_READOUT_REASON
        assert "pointer head reads the option states directly" in check["reason"]
        assert check["status"] != "PASS"
        assert name not in gate["applied_checks"]
        # the note the cell carries says the same thing
        assert any(check["reason"] in note for note in gate["notes"])
    # the two inapplicable checks never contribute a failure, and the reason is on the cell
    assert gate["failures"] == []
    assert len(gate["notes"]) == len([n for n in gate["notes"] if n]) == 3


def test_inapplicable_checks_stay_inapplicable_in_a_failing_cell() -> None:
    gate = _gate(correct=[False] * 40)
    assert gate["status"].startswith("READOUT_FAIL")
    assert gate["checks"]["median_label_mass"]["status"] == rs.NO_LETTER_READOUT_REASON
    assert gate["checks"]["greedy_agreement"]["applied"] is False


# --------------------------------------------------------------------------- (d) score expected-level error
def test_score_expected_level_error_hand_computed() -> None:
    first = _item(qtype=QuestionType.SCORE, n_options=2, seed=0)
    second = _item(qtype=QuestionType.SCORE, n_options=3, seed=1, template_id="synthetic_score_v2",
                   split="dev")
    third = _item(qtype=QuestionType.SCORE, n_options=3, seed=2, template_id="synthetic_score_v3",
                  split="dev")
    predictions = [
        _prediction(first, [0.5, 0.5], gold_key="1"),   # expected level 1.5, gold 1 -> 0.5
        _prediction(second, [0.2, 0.3, 0.5], gold_key="3"),  # 2.3 vs 3 -> 0.7
        _prediction(third, [0.0, 1.0, 0.0], gold_key="2"),   # 2.0 vs 2 -> 0.0
    ]
    out = rs.score_expected_level_error(predictions)
    assert out["n"] == 3
    assert out["mean_abs_error"] == pytest.approx((0.5 + 0.7 + 0.0) / 3)
    assert out["median_abs_error"] == pytest.approx(0.5)
    assert out["mean_expected_level"] == pytest.approx((1.5 + 2.3 + 2.0) / 3)
    assert out["mean_gold_level"] == pytest.approx(2.0)


def test_score_expected_level_error_is_absent_from_choice_cells() -> None:
    choice = _item(seed=0)
    pred = _prediction(choice, [0.25, 0.25, 0.25, 0.25])
    assert "score" not in rs.summarise_cell(
        model_id="m", tier="tier1", template_id=choice.template_id, split="test",
        predictions=[pred], template_total=1, template_available=1, sampled=False,
        strict_ids=None, seed=0, shuffle_flip=None, shuffle_n=0, shuffle_error=None,
        temperature={}, uncalibrated=None, wall_seconds=0.0,
    )


# --------------------------------------------------------------------------- (e) temperature fitting
def _logits_case(n_high: int, n_low: int) -> tuple[list[np.ndarray], list[int]]:
    logits = [np.asarray([1.0, 0.0]) for _ in range(n_high + n_low)]
    gold = [0] * n_high + [1] * n_low
    return logits, gold


def test_temperature_fit_refuses_a_test_split(tmp_path) -> None:
    logits, gold = _logits_case(36, 24)
    target = tmp_path / "never-written.json"
    with pytest.raises(ValueError, match="dev-only"):
        rs.fit_temperatures(
            split="test", logits_by_qtype={"choice": logits}, gold_by_qtype={"choice": gold}, seed=0
        )
    with pytest.raises(ValueError, match="dev-only"):
        rs.write_temperature_fit(
            target, {"per_qtype": {}}, checkpoint=tmp_path / "cp",
            split="test", model_id="m", sampled=False,
        )
    assert not target.exists()


def test_temperature_fit_reports_too_few_items_and_bound_hits() -> None:
    # 3 items: below the minimum, so it must not be presented as a fitted temperature
    few_logits, few_gold = _logits_case(2, 1)
    fit = rs.fit_temperatures(
        split="dev", logits_by_qtype={"score": few_logits}, gold_by_qtype={"score": few_gold},
        seed=0, min_items=50,
    )
    entry = fit["per_qtype"]["score"]
    assert entry["fitted"] is False
    assert entry["status"].startswith("NOT FITTED — too few dev items")
    assert entry["n_items"] == 3
    assert fit["n_qtype_fitted"] == 0 and fit["n_qtype_not_fitted"] == 1

    # separable logits: the objective becomes numerically flat at zero (the model is saturated),
    # so the returned temperature is arbitrary and must not be reported as fitted
    sep_logits = [np.asarray([5.0, 0.0]) for _ in range(60)]
    sep_gold = [0] * 60
    degenerate = rs.fit_temperatures(
        split="dev", logits_by_qtype={"choice": sep_logits}, gold_by_qtype={"choice": sep_gold},
        seed=0, min_items=50,
    )["per_qtype"]["choice"]
    assert degenerate["fitted"] is False
    assert degenerate["status"].startswith("NOT FITTED — the fit reached a numerically zero NLL")

    # an optimum outside the search box: the fitted value lands on the upper bound
    flat_logits = [np.asarray([3.0, 0.0]) for _ in range(60)]
    flat_gold = [1] * 60  # gold is the *low*-logit option, so the fit wants to flatten hard
    bound = rs.fit_temperatures(
        split="dev", logits_by_qtype={"choice": flat_logits}, gold_by_qtype={"choice": flat_gold},
        seed=0, min_items=50,
    )["per_qtype"]["choice"]
    assert bound["fitted"] is False
    assert bound["status"].startswith("NOT FITTED — the search landed on the upper bound")

    # an interior optimum: a real fit
    logits, gold = _logits_case(36, 24)
    good = rs.fit_temperatures(
        split="dev", logits_by_qtype={"choice": logits}, gold_by_qtype={"choice": gold},
        seed=0, min_items=50,
    )["per_qtype"]["choice"]
    assert good["fitted"] is True
    assert good["status"] == "FITTED"
    assert rs.TEMPERATURE_BOUNDS[0] < good["temperature"] < rs.TEMPERATURE_BOUNDS[1]
    assert good["nll_after"] <= good["nll_before"]

    # an empty qtype is not fitted either
    empty = rs.fit_temperatures(split="dev", logits_by_qtype={}, gold_by_qtype={}, seed=0)
    assert empty["per_qtype"] == {}
    assert empty["n_qtype_fitted"] == 0


def test_temperature_file_only_applies_fitted_entries(tmp_path) -> None:
    payload = {
        "split": "dev",
        "per_qtype": {
            "choice": {"temperature": 1.5, "n_items": 100, "fitted": True, "status": "FITTED"},
            "score": {"temperature": 4.0, "n_items": 3, "fitted": False,
                      "status": "NOT FITTED — too few dev items"},
        },
    }
    path = tmp_path / "temperature.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    applied, provenance = rs.load_temperature_file(path)
    assert applied == {"choice": 1.5}
    assert provenance["skipped_not_fitted"] == {"score": "NOT FITTED — too few dev items"}


# --------------------------------------------------------------------------- additional check + selectivity
def test_option_permutation_check_is_additional_and_labelled() -> None:
    gate = _gate(shuffle_flip=0.9, shuffle_n=100)
    check = gate["checks"]["option_permutation_consistency"]
    assert check["kind"] == "additional"
    assert check["status"] == "FAIL"
    assert check["threshold"] == rs.FLIP_CHECK_THRESHOLD
    assert "NOT part of D12" in check["label"]
    # the D12 status is decided by the two applicable D12 checks only
    assert gate["status"] == "PASS"
    assert gate["failures"] == []
    assert any("additional check FAILED" in note for note in gate["notes"])
    assert check["failure"] is not None and "flip rate" in check["failure"]

    ok = _gate(shuffle_flip=0.1, shuffle_n=100)["checks"]["option_permutation_consistency"]
    assert ok["status"] == "PASS"
    missing = _gate()["checks"]["option_permutation_consistency"]
    assert missing["status"].startswith("NOT MEASURED —")


def test_selective_accuracy_hand_computed() -> None:
    confidence = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    correct = [True, True, True, False, True, False, False, True, False, False]
    out = rs.selective_accuracy(confidence, correct, fractions=(0.5, 1.0))
    assert out["0.50"]["n"] == 5
    assert out["0.50"]["accuracy"] == pytest.approx(4 / 5)
    assert out["0.50"]["min_confidence"] == pytest.approx(0.5)
    assert out["1.00"]["n"] == 10
    assert out["1.00"]["accuracy"] == pytest.approx(5 / 10)
    with pytest.raises(ValueError):
        rs.selective_accuracy(confidence, correct[:-1])
    with pytest.raises(ValueError):
        rs.selective_accuracy([], [])


# --------------------------------------------------------------------------- model-API equivalence
class _StubModel:
    """A duck-typed stand-in: `predict` needs only `calibration`, `score_items` and the real
    `apply_temperature` (which reads nothing but its arguments and `calibration`)."""

    # the real method, unbound: it touches only `self.calibration` and the ScoredItems passed in
    apply_temperature = MedDecideModel.apply_temperature

    def __init__(self, scored: ScoredItems, calibration: dict[str, float] | None = None):
        self.scored = scored
        self.calibration = calibration or {}

    def score_items(self, items, **_kwargs) -> ScoredItems:
        assert [i.item_id for i in items] == [i.item_id for i in self.scored.items]
        return self.scored


def _probs_for(n_options: int) -> np.ndarray:
    """A deterministic, valid distribution over ``n_options`` options (unsorted argmax)."""
    probs = np.full(n_options, 0.1, dtype=np.float64)
    probs[1 % n_options] = 1.0 - 0.1 * (n_options - 1)
    return probs


def _scored_for(items: list[Item]) -> ScoredItems:
    probs = [_probs_for(item.n_options) for item in items]
    logits = [np.log(p) for p in probs]
    return ScoredItems(
        items=list(items),
        option_keys=[[o.key for o in item.options] for item in items],
        original_option_keys=[[o.key for o in item.options] for item in items],
        logits=logits,
        probs=probs,
        gold_indices=[item.gold_index for item in items],
        prompt_tokens=[10 * (i + 1) for i in range(len(items))],
        latency_s=[0.01 * (i + 1) for i in range(len(items))],
        wall_clock_s=0.1,
        batch_sizes=[len(items)],
    )


def test_decisions_from_scored_matches_predict() -> None:
    items = [
        _item(seed=0),
        _item(qtype=QuestionType.NOUL, seed=1, template_id="synthetic_noul_v1"),
        _item(qtype=QuestionType.SCORE, n_options=3, seed=2, template_id="synthetic_score_v1"),
    ]
    scored = _scored_for(items)
    for temperature in ({}, {"score": 2.0}, None):
        stub = _StubModel(scored, calibration={"choice": 1.5})
        mine = rs.decisions_from_scored(stub, scored, temperature=temperature)
        theirs = MedDecideModel.predict(stub, items, temperature=temperature)
        assert [d.to_json() for d in mine] == [d.to_json() for d in theirs]
    # calibration=None means "use the model's stored calibration", exactly as predict does
    stub = _StubModel(scored, calibration={"choice": 1.5})
    assert rs.decisions_from_scored(stub, scored, temperature=None)[0].temperature == 1.5


def test_brier_groups_mixed_option_counts() -> None:
    two = _item(qtype=QuestionType.SCORE, n_options=2, seed=0)
    three = _item(qtype=QuestionType.SCORE, n_options=3, seed=1, template_id="synthetic_score_v2")
    preds = [_prediction(two, [1.0, 0.0]), _prediction(three, [0.0, 0.0, 1.0])]
    # item 1: perfectly confident and right -> 0.0 on 2 columns;
    # item 2: gold is level 2, the readout says level 3 with probability 1 -> 2.0 on 3 columns;
    # the weighted mean over the two groups is therefore exactly 1.0
    assert rs.brier_from_predictions(preds) == pytest.approx(1.0)


def test_no_letter_readout_reason_is_the_required_string() -> None:
    assert rs.NO_LETTER_READOUT_REASON == (
        "NOT APPLICABLE — pointer head reads the option states directly; no letter readout "
        "exists to validate"
    )
