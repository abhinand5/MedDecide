"""Minimal-pair helpers (osler_v4 Y3): sentence numbering, pure deletion, section removal, the teacher's answer parsers, the
string rules and the pair pipeline with a fake teacher. Synthetic notes only."""

from __future__ import annotations

import random
import re

from meddecide.mr import mimic_pairs as mp

NOTE = """Chief Complaint:
Cough and fever. She also reports a dry mouth.
History of Present Illness:
Pneumonia was suspected on admission. Treated with antibiotics.
Discharge Diagnosis:
Pneumonia, resolved.
"""

PAIR_NOTE = """Chief Complaint:
Cough. Pneumonia suspected on arrival. Also a dry mouth.
History of Present Illness:
Patient with cough for two days. Her pneumonia was treated with antibiotics.
Brief Hospital Course:
Stable on discharge. Walking unaided. Diet tolerated. Pain controlled. Vital signs normal. Discharged home.
"""
OTHERS = (("R05", "Cough"), ("R13", "Dysphagia"))
TARGET = ("J18.9", "Pneumonia, unspecified organism")


def test_sentences_are_numbered_across_lines_and_headers_are_not_numbered() -> None:
    lines, refs = mp.parse_note(NOTE)
    assert [r.text for r in refs] == ["Cough and fever.", "She also reports a dry mouth.",
                                      "Pneumonia was suspected on admission.", "Treated with antibiotics.",
                                      "Pneumonia, resolved."]
    view = mp.numbered_view(lines).splitlines()
    assert view[0] == "Chief Complaint:"
    assert view[1] == "[1] Cough and fever."
    assert "Discharge Diagnosis:" in view


def test_deleting_one_sentence_keeps_every_other_sentence_byte_identical() -> None:
    lines, _ = mp.parse_note(NOTE)
    edited = mp.delete_sentences(lines, [2])
    assert "She also reports a dry mouth." not in edited
    assert "Cough and fever." in edited
    assert mp.check_pure_deletion(NOTE, edited)


def test_a_section_whose_sentences_are_all_deleted_loses_its_header() -> None:
    lines, _ = mp.parse_note(NOTE)
    edited = mp.delete_sentences(lines, [3, 4])
    assert "History of Present Illness:" not in edited
    assert "Chief Complaint:" in edited
    assert "Discharge Diagnosis:" in edited


def test_a_partly_deleted_section_keeps_its_header() -> None:
    lines, _ = mp.parse_note(NOTE)
    edited = mp.delete_sentences(lines, [4])
    assert "History of Present Illness:" in edited
    assert "Pneumonia was suspected on admission." in edited


def test_a_leading_sentence_deleted_from_a_line_leaves_the_rest_without_a_stray_space() -> None:
    lines, _ = mp.parse_note(NOTE)
    edited = mp.delete_sentences(lines, [1])
    assert "Cough and fever." not in edited
    assert "She also reports a dry mouth." in edited
    assert mp.check_pure_deletion(NOTE, edited)


def test_a_rewritten_sentence_fails_the_pure_deletion_check() -> None:
    assert not mp.check_pure_deletion(NOTE, NOTE.replace("Treated with antibiotics.", "Treated with fluids."))
    assert mp.check_pure_deletion(NOTE, NOTE.replace("Treated with antibiotics.", ""))


def test_parse_selection_accepts_a_json_list_and_rejects_anything_else() -> None:
    assert mp.parse_selection("[3, 1]", 5) == [1, 3]
    assert mp.parse_selection("Selected: [2]", 5) == [2]
    assert mp.parse_selection("[]", 5) == []
    assert mp.parse_selection("[6]", 5) is None
    assert mp.parse_selection("[1, 1]", 5) is None
    assert mp.parse_selection("[true]", 5) is None
    assert mp.parse_selection('["1"]', 5) is None
    assert mp.parse_selection("no list here", 5) is None


def test_parse_yes_no_reads_the_first_word_only() -> None:
    assert mp.parse_yes_no("Yes.") == "yes"
    assert mp.parse_yes_no("No, it does not.") == "no"
    assert mp.parse_yes_no("Maybe") is None
    assert mp.parse_yes_no("") is None


def test_keywords_come_from_the_title_and_the_family_terms_minus_stop_words() -> None:
    assert mp.condition_keywords("Pneumonia, unspecified organism") == ("pneumonia",)
    assert mp.condition_keywords("Hyponatremia", ("hyponatraemia",)) == ("hyponatraemia", "hyponatremia")


def test_a_keyword_left_in_the_edit_is_reported() -> None:
    keywords = mp.condition_keywords("Pneumonia, unspecified organism")
    assert mp.leaked_keywords("The pneumonia resolved.", keywords) == ["pneumonia"]
    assert mp.leaked_keywords("The infection resolved.", keywords) == []


def test_the_selection_share_is_taken_over_all_numbered_sentences() -> None:
    _, refs = mp.parse_note(NOTE)
    assert mp.selection_share(refs, [1, 2]) == 2 / 5
    assert mp.other_codes(["A1", "B2", "A1", "C3"], "B2") == ["A1", "C3"]


def _fake_teacher(select_word: str | None = None, verify_mode: str = "", select_answer: str | None = None):
    """A fake teacher. Selection returns the numbers of the sentences containing ``select_word`` (or ``select_answer``).
    Verification answers from the record: "yes" or "no" when ``verify_mode`` is set; otherwise yes exactly when the first
    word of the question's title occurs in the record."""
    def chat(messages: list[dict[str, str]], kind: str) -> str:
        if kind == "select":
            if select_answer is not None:
                return select_answer
            numbers = [m.group(1) for m in re.finditer(r"^\[(\d+)\] (.*)$", messages[1]["content"], flags=re.M)
                       if select_word and select_word.lower() in m.group(2).lower()]
            return "[" + ", ".join(numbers) + "]"
        if verify_mode in ("yes", "no"):
            return verify_mode
        record, question = messages[1]["content"].split("\n\nDoes this record document ", 1)
        word = re.split(r"[ ,(]", question.strip())[0].lower()
        return "yes" if word in record.lower() else "no"
    return chat


def test_a_pair_is_kept_when_the_edit_removes_the_condition_and_the_checks_pass() -> None:
    attempt = mp.build_pair(PAIR_NOTE, *TARGET, ("pneumonia",), OTHERS, _fake_teacher("pneumonia"), random.Random(0))
    assert attempt.outcome == "kept"
    assert "pneumonia" not in attempt.edited.lower()
    assert "Stable on discharge." in attempt.edited
    assert attempt.original.lower().count("pneumonia") == 2
    assert (attempt.others_checked, attempt.others_skipped) == (1, 1)


def test_a_note_that_does_not_document_the_condition_is_dropped_at_the_original_check() -> None:
    attempt = mp.build_pair(PAIR_NOTE, *TARGET, ("pneumonia",), OTHERS, _fake_teacher("pneumonia", "no"), random.Random(0))
    assert attempt.outcome == "original not read as documented"


def test_an_edit_that_still_reads_as_documented_is_dropped() -> None:
    attempt = mp.build_pair(PAIR_NOTE, *TARGET, ("pneumonia",), OTHERS, _fake_teacher("pneumonia", "yes"), random.Random(0))
    assert attempt.outcome == "edit still read as documented"


def test_an_answer_that_is_not_a_list_is_asked_again_and_then_dropped() -> None:
    attempt = mp.build_pair(PAIR_NOTE, *TARGET, ("pneumonia",), OTHERS,
                            _fake_teacher(select_answer="not a list"), random.Random(0))
    assert attempt.outcome == "selection not a valid list"
    assert attempt.calls == 2


def test_more_than_thirty_percent_selected_is_dropped() -> None:
    attempt = mp.build_pair(PAIR_NOTE, *TARGET, ("pneumonia",), OTHERS,
                            _fake_teacher(select_answer="[1, 2, 3, 4]"), random.Random(0))
    assert attempt.outcome == "more than 30 % of sentences selected"


def test_a_keyword_left_in_the_edit_is_a_string_rule_failure() -> None:
    attempt = mp.build_pair(PAIR_NOTE, *TARGET, ("pneumonia",), OTHERS, _fake_teacher(select_answer="[2]"),
                            random.Random(0))
    assert attempt.outcome.startswith("string rule")


def test_an_other_coded_diagnosis_lost_by_the_edit_drops_the_pair() -> None:
    # sentence 7 is the only one that mentions walking, so deleting it loses that other code (3 of 11 sentences selected)
    attempt = mp.build_pair(PAIR_NOTE, *TARGET, ("pneumonia",), (("Z99", "Walking"),),
                            _fake_teacher(select_answer="[2, 5, 7]"), random.Random(0))
    assert attempt.outcome == "other coded diagnosis lost by the edit"


def test_a_short_note_is_dropped_before_any_call() -> None:
    calls: list[str] = []

    def chat(messages: list[dict[str, str]], kind: str) -> str:
        calls.append(kind)
        return "[]"

    attempt = mp.build_pair("One sentence only.", *TARGET, ("pneumonia",), OTHERS, chat, random.Random(0))
    assert attempt.outcome == "too few sentences"
    assert calls == []
