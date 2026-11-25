"""Unit tests for the distill_v0 hosted d1-ultra teacher client (pure functions, synthetic rows only)."""

from __future__ import annotations

import json
import math

import pytest

from meddecide.distill.d1_teacher import (
    MODEL,
    QUESTION_ID,
    CostLedger,
    TeacherAnswerError,
    TeacherRequest,
    file_sha256,
    freeze_row,
    hosted_guard,
    is_correct,
    load_frozen,
    parse_response,
    predicted_key,
    request_for_item,
)


def item(qtype: str = "choice", **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "item_id": "0123456789abcdef",
        "template_id": "toy_v1",
        "qtype": qtype,
        "state": "state text",
        "question": "which one?",
        "gold": "A",
        "options": [
            {"key": "A", "label": "alpha"},
            {"key": "B", "label": "beta"},
            {"key": "C", "label": "gamma"},
        ],
    }
    base.update(overrides)
    return base


def noul_item(**overrides: object) -> dict[str, object]:
    base = item("noul", options=[{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}], gold="yes")
    base.update(overrides)
    return base


def request_for(**overrides: object) -> TeacherRequest:
    request, reason = request_for_item(item(**overrides))
    assert reason is None and request is not None
    return request


def test_choice_request_shape() -> None:
    request = request_for()
    body = request.body
    assert body["model"] == MODEL
    assert body["state"] == ["state text"]
    spec = body["questions"][QUESTION_ID]
    assert spec == {"type": "choice", "instructions": "which one?", "criteria": {"A": "alpha", "B": "beta", "C": "gamma"}}
    assert request.option_keys == ("A", "B", "C")
    assert request.item_id == "0123456789abcdef"
    assert request.template_id == "toy_v1"
    assert json.loads(request.body_bytes()) == body


def test_noul_request_has_no_criteria() -> None:
    request, reason = request_for_item(noul_item())
    assert reason is None and request is not None
    spec = request.body["questions"][QUESTION_ID]
    assert spec == {"type": "noul", "instructions": "which one?"}
    assert request.option_keys == ("yes", "no")


def test_score_request_levels_are_ordered() -> None:
    row = item("score", options=[{"key": str(i), "label": label} for i, label in enumerate(["low", "mid", "high"])])
    request, reason = request_for_item(row)
    assert reason is None and request is not None
    assert request.body["questions"][QUESTION_ID] == {
        "type": "score", "instructions": "which one?", "criteria": ["low", "mid", "high"]}


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"state": "   "}, "empty_state"),
        ({"question": ""}, "empty_question"),
        ({"options": [{"key": "A", "label": "alpha"}]}, "option_count"),
        ({"options": [{"key": "A", "label": "a"}, {"key": "A", "label": "b"}]}, "repeated_key"),
        ({"options": [{"key": "A", "label": " "}, {"key": "B", "label": "b"}]}, "empty_label"),
        ({"qtype": "freeform"}, "qtype"),
        ({"qtype": "noul", "options": [{"key": "true", "label": "Yes"}, {"key": "false", "label": "No"}]}, "noul_keys"),
    ],
)
def test_unrequestable_rows_have_a_reason(overrides: dict[str, object], reason: str) -> None:
    request, got = request_for_item(item(**overrides))
    assert request is None and got == reason


def choice_answer(**overrides: object) -> dict[str, object]:
    answer: dict[str, object] = {
        "answers": {QUESTION_ID: {"type": "choice", "probabilities": {"A": 0.2, "B": 0.5, "C": 0.3}}}
    }
    answer.update(overrides)
    return answer


def test_parse_choice_aligns_to_the_request_order() -> None:
    request = TeacherRequest("id", "t", "choice", ("C", "A", "B"), {})
    assert parse_response(choice_answer(), request) == [0.3, 0.2, 0.5]


def test_parse_noul_maps_by_key() -> None:
    payload = {"answers": {QUESTION_ID: {"type": "noul", "noul": 0.8}}}
    assert parse_response(payload, TeacherRequest("id", "t", "noul", ("yes", "no"), {})) == [0.8, pytest.approx(0.2)]
    assert parse_response(payload, TeacherRequest("id", "t", "noul", ("no", "yes"), {})) == [pytest.approx(0.2), 0.8]


def test_parse_score_maps_level_indices() -> None:
    payload = {"answers": {QUESTION_ID: {"type": "score", "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}}}
    assert parse_response(payload, TeacherRequest("id", "t", "score", ("0", "1", "2"), {})) == [0.1, 0.2, 0.7]


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({}, "no_answers"),
        ({"answers": {}}, "missing_question"),
        ({"answers": {QUESTION_ID: {"probabilities": {"A": 1.0}}}}, "key_set"),
        ({"answers": {QUESTION_ID: {"probabilities": {"A": 0.5, "B": 0.4, "C": 0.05}}}}, "prob_sum"),
        ({"answers": {QUESTION_ID: {"probabilities": {"A": 1.1, "B": -0.2, "C": 0.1}}}}, "negative"),
        ({"answers": {QUESTION_ID: {"probabilities": {"A": math.nan, "B": 0.5, "C": 0.5}}}}, "non_finite"),
        ({"answers": {QUESTION_ID: {"refusal": "no"}}}, "refusal"),
        ({"answers": {QUESTION_ID: {"type": "noul", "probabilities": {"A": 1.0, "B": 0.0, "C": 0.0}}}}, "qtype_mismatch"),
        ({"answers": {QUESTION_ID: {"probabilities": "nope"}}}, "no_probabilities"),
    ],
)
def test_parse_validation_reasons(payload: dict[str, object], reason: str) -> None:
    with pytest.raises(TeacherAnswerError) as excinfo:
        parse_response(payload, TeacherRequest("id", "t", "choice", ("A", "B", "C"), {}))
    assert excinfo.value.reason == reason


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ({"type": "noul"}, "no_noul"),
        ({"type": "noul", "noul": math.inf}, "non_finite"),
        ({"type": "noul", "noul": 1.5}, "range"),
        ({"type": "choice", "noul": 0.5}, "qtype_mismatch"),
    ],
)
def test_parse_noul_validation(answer: dict[str, object], reason: str) -> None:
    with pytest.raises(TeacherAnswerError) as excinfo:
        parse_response({"answers": {QUESTION_ID: answer}}, TeacherRequest("id", "t", "noul", ("yes", "no"), {}))
    assert excinfo.value.reason == reason


def test_score_key_set_is_positional() -> None:
    payload = {"answers": {QUESTION_ID: {"type": "score", "probabilities": {"0": 0.5, "2": 0.5}}}}
    with pytest.raises(TeacherAnswerError) as excinfo:
        parse_response(payload, TeacherRequest("id", "t", "score", ("0", "1", "2"), {}))
    assert excinfo.value.reason == "key_set"


def test_predicted_key_and_correctness() -> None:
    request = TeacherRequest("id", "t", "choice", ("A", "B", "C"), {})
    assert predicted_key(request, [0.1, 0.7, 0.2]) == "B"
    assert predicted_key(request, [0.4, 0.4, 0.2]) == "A"  # a tie goes to the earlier option
    assert is_correct(item(), [0.6, 0.3, 0.1]) is True
    assert is_correct(item(), [0.2, 0.6, 0.2]) is False


def test_score_correctness_is_the_modal_level() -> None:
    row = item("score", gold="2", options=[{"key": "0", "label": "low"}, {"key": "1", "label": "mid"},
                                          {"key": "2", "label": "high"}])
    assert is_correct(row, [0.1, 0.2, 0.7]) is True
    assert is_correct(row, [0.1, 0.7, 0.2]) is False


def test_freeze_row_has_the_recorded_schema_and_no_text() -> None:
    request = request_for()
    row = freeze_row(request, [0.2, 0.5, 0.3], cost=0.0000042, input_tokens=68)
    assert set(row) == {"item_id", "template_id", "qtype", "option_keys", "teacher_probs", "teacher", "cost",
                        "input_tokens"}
    assert row["teacher"] == MODEL
    assert row["option_keys"] == ["A", "B", "C"]
    assert row["teacher_probs"] == [0.2, 0.5, 0.3]
    text = json.dumps(row).lower()
    for forbidden in ("state text", "which one", "alpha", "beta", "gamma", "gold"):
        assert forbidden not in text
    with pytest.raises(ValueError):
        freeze_row(request, [1.0], cost=0.0, input_tokens=1)


def test_freeze_row_with_text_key_is_still_text_free() -> None:
    request = request_for()
    row = freeze_row(request, [0.2, 0.5, 0.3], cost=0.0, input_tokens=1, text_key="0123456789abcdef")
    assert row["text_key"] == "0123456789abcdef"
    assert "text_key" not in freeze_row(request, [0.2, 0.5, 0.3], cost=0.0, input_tokens=1)


def test_ledger_books_charges_and_stops_at_the_cap() -> None:
    ledger = CostLedger(cap_usd=0.5)
    assert not ledger.exhausted
    ledger.charge(0.2, 100)
    ledger.charge(0.1, 50)
    assert ledger.to_dict()["spent_usd"] == pytest.approx(0.3)
    assert ledger.remaining_usd == pytest.approx(0.2)
    ledger.note_failure()
    ledger.charge(0.25, 10)
    assert ledger.exhausted
    assert ledger.to_dict()["calls"] == 3
    assert ledger.to_dict()["failures"] == 1
    with pytest.raises(ValueError):
        ledger.charge(-0.1, 1)


def test_hosted_guard_refuses_mimic_by_licence_source_and_metadata() -> None:
    public = item()
    assert hosted_guard(public) == (True, None)
    licensed = item(source_license="PhysioNet Credentialed Health Data License 1.5.0")
    assert hosted_guard(licensed) == (False, "physionet_licence")
    assert hosted_guard(item(source="mimic_iv")) == (False, "mimic_source")
    assert hosted_guard(item(set_name="mimic_suite")) == (False, "mimic_source")
    assert hosted_guard(item(meta={"note": "from MIMIC-IV"})) == (False, "mimic_source")


def test_hosted_guard_allows_protected_under_operator_authorization() -> None:
    row = item(source="mimic_iv", source_license="PhysioNet Credentialed Health Data License 1.5.0")
    assert hosted_guard(row) == (False, "physionet_licence")
    assert hosted_guard(row, allow_protected=True) == (True, None)


def test_ledger_seed_is_never_negative_and_adds_to_the_spend() -> None:
    ledger = CostLedger(cap_usd=1.0)
    ledger.seed(0.25, 1000, 10)
    ledger.charge(0.1, 100)
    assert ledger.to_dict()["spent_usd"] == pytest.approx(0.35)
    assert ledger.to_dict()["calls"] == 11
    assert ledger.to_dict()["input_tokens"] == 1100
    with pytest.raises(ValueError):
        ledger.seed(-1.0, 0, 0)


def test_file_hash_and_frozen_roundtrip(tmp_path) -> None:
    path = tmp_path / "labels.jsonl"
    rows = [freeze_row(request_for(), [0.2, 0.5, 0.3], cost=0.0, input_tokens=1)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert file_sha256(path) == __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    loaded = load_frozen(path)
    assert loaded["0123456789abcdef"]["teacher_probs"] == [0.2, 0.5, 0.3]
    assert load_frozen(tmp_path / "missing.jsonl") == {}
