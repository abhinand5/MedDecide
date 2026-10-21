"""Y5 soft labels (osler_v4): request and answer mapping, parity counts, exclusion reasons and the anchor-pool order."""

from __future__ import annotations

import random

import pytest

from meddecide.distill import soft_labels as sl


def _choice_row(**overrides):
    row = {"item_id": "c1", "qtype": "choice", "template_id": "broad_x_v1", "state": "A patient has pain.",
           "question": "Which is the cause?",
           "options": [{"key": "A", "label": "Gallstones"}, {"key": "B", "label": "Appendicitis"},
                       {"key": "C", "label": "Pneumonia"}],
           "gold": "B", "source": "x", "source_record_id": "r1"}
    return {**row, **overrides}


def _noul_row(**overrides):
    row = {"item_id": "n1", "qtype": "noul", "template_id": "ct_randomised_noul_v1", "state": "Trial design text.",
           "question": "Is the allocation randomised?",
           "options": [{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}], "gold": "yes"}
    return {**row, **overrides}


def _score_row(**overrides):
    row = {"item_id": "s1", "qtype": "score", "template_id": "ct_enrol_score_v1", "state": "Enrolment record.",
           "question": "How large is the study?",
           "options": [{"key": "0", "label": "fewer than 5"}, {"key": "1", "label": "50 to 199"},
                       {"key": "2", "label": "200 to 999"}],
           "gold": "1"}
    return {**row, **overrides}


def test_choice_request_is_keyed_by_label_in_row_order():
    request, reason = sl.request_for_row(_choice_row())
    assert reason is None
    assert request == {"state": "A patient has pain.", "questions": {"q": {
        "type": "choice", "instructions": "Which is the cause?",
        "criteria": {"Gallstones": None, "Appendicitis": None, "Pneumonia": None}}}}


def test_noul_request_is_the_native_question_with_yes_and_no_criteria():
    request, reason = sl.request_for_row(_noul_row())
    assert reason is None
    assert request["questions"]["q"] == {"type": "noul", "instructions": "Is the allocation randomised?",
                                         "criteria": {"true": "Yes", "false": "No"}}


def test_score_request_lists_the_levels_lowest_first():
    request, reason = sl.request_for_row(_score_row())
    assert reason is None
    assert request["questions"]["q"]["criteria"] == ["fewer than 5", "50 to 199", "200 to 999"]
    assert request["questions"]["q"]["type"] == "score"


@pytest.mark.parametrize(("row", "reason"), [
    (_choice_row(state="  "), "empty_state"),
    (_choice_row(question=""), "empty_question"),
    (_choice_row(options=[]), "option_count"),
    (_choice_row(options=[{"key": "A", "label": "Same"}, {"key": "B", "label": "Same"}]), "repeated_label"),
    (_choice_row(options=[{"key": "A", "label": " "}, {"key": "B", "label": "Other"}]), "empty_label"),
    (_noul_row(options=[{"key": "Y", "label": "Yes"}, {"key": "N", "label": "No"}]), "noul_keys"),
    (_score_row(options=[{"key": "0", "label": "only one"}]), "score_levels"),
    (_choice_row(qtype="rank"), "qtype"),
])
def test_rows_that_cannot_be_sent_exactly_are_skipped_with_a_reason(row, reason):
    assert sl.request_for_row(row) == (None, reason)


def test_choice_probabilities_follow_the_row_order_not_the_answer_order():
    answer = {"type": "choice", "probabilities": {"Pneumonia": 0.2, "Gallstones": 0.5, "Appendicitis": 0.3}}
    assert sl.teacher_probabilities(_choice_row(), answer) == [0.5, 0.3, 0.2]


def test_noul_probability_is_p_yes_aligned_to_the_yes_and_no_keys():
    assert sl.teacher_probabilities(_noul_row(), {"type": "noul", "noul": 0.8}) == pytest.approx([0.8, 0.2])


def test_score_probabilities_are_keyed_by_position():
    answer = {"type": "score", "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}, "score": 1.5}
    assert sl.teacher_probabilities(_score_row(), answer) == pytest.approx([0.1, 0.3, 0.6])


def test_a_teacher_answer_that_is_not_a_distribution_raises():
    answer = {"type": "choice", "probabilities": {"Gallstones": 0.9, "Appendicitis": 0.9, "Pneumonia": 0.1}}
    with pytest.raises(ValueError, match="distribution"):
        sl.teacher_probabilities(_choice_row(), answer)


def test_compare_scorings_counts_argmax_agreement_and_differences():
    reference = [[0.6, 0.4], [0.2, 0.8], [0.5, 0.5]]
    same = sl.compare_scorings(reference, [list(p) for p in reference])
    assert same == {"rows": 3, "argmax_agree": 3, "max_abs_diff": 0.0, "mean_abs_diff": 0.0}
    flipped = sl.compare_scorings(reference, [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]])
    assert flipped["argmax_agree"] == 2
    assert flipped["max_abs_diff"] == pytest.approx(0.5)


def test_argmax_tie_goes_to_the_earlier_option():
    assert sl.argmax_index([0.5, 0.5]) == 0


def test_accuracy_by_qtype_states_its_denominators():
    scored = [(_choice_row(), [0.1, 0.8, 0.1]), (_choice_row(gold="A"), [0.1, 0.8, 0.1]),
              (_noul_row(), [0.3, 0.7])]
    report = sl.accuracy_by_qtype(scored)
    assert report["all"] == {"items": 3, "correct": 1, "accuracy": pytest.approx(1 / 3)}
    assert report["choice"] == {"items": 2, "correct": 1, "accuracy": 0.5}
    assert report["noul"] == {"items": 1, "correct": 0, "accuracy": 0.0}


def test_overlap_reason_distinguishes_text_record_and_both():
    row = _choice_row()
    text_key, record_key = sl.overlap_keys(row)
    assert record_key == ("x", "r1")
    assert sl.overlap_reason(row, {text_key}, set()) == "text"
    assert sl.overlap_reason(row, set(), {record_key}) == "record"
    assert sl.overlap_reason(row, {text_key}, {record_key}) == "text and record"
    assert sl.overlap_reason(row, set(), set()) is None


def test_a_row_without_a_source_record_has_no_record_key():
    row = _choice_row(source="", source_record_id="")
    assert sl.overlap_keys(row)[1] is None


def test_pool_order_is_stable_and_independent_of_input_order():
    ids = [f"item-{i:05d}" for i in range(200)]
    forward = sorted(ids, key=sl.pool_order_key)
    backward = sorted(random.Random(0).sample(ids, len(ids)), key=sl.pool_order_key)
    assert forward == backward
    assert forward != ids  # a shuffle, not the input order
    assert sl.pool_order_key("item-00001") == sl.pool_order_key("item-00001")


def test_soft_label_row_keeps_the_row_and_adds_the_teacher_distribution():
    row = _choice_row()
    out = sl.soft_label_row(row, (0.5, 0.3, 0.2))
    assert out["teacher_probs"] == [0.5, 0.3, 0.2]
    assert out["teacher"] == sl.TEACHER
    assert {k: v for k, v in out.items() if k not in {"teacher_probs", "teacher"}} == row
