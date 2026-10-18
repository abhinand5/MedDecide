"""osler_v3 X3 generators, pure parts (CPU): claim grounding, restyled pages and the substrate split and passage windows."""

from __future__ import annotations

from collections import Counter

import pytest

from meddecide.mix import clinclaim as cc
from meddecide.mix import restyle as rs
from meddecide.mix import substrate as sub
from meddecide.mr.realtext import PAGE_LETTER

# ---- claims: draws, prompts and parsing --------------------------------------------------------------------------------

def test_draws_are_deterministic_and_only_contradicted_items_have_an_applicable_edit_kind() -> None:
    passage = "The patient has fever. Blood pressure was 120/80 on the left arm."
    assert cc.draw("doc-1", 0, 0, passage) == cc.draw("doc-1", 0, 0, passage)
    seen = set()
    for i in range(300):
        label, kind = cc.draw("doc-x", i, 0, passage)
        assert label in cc.LABELS
        assert (kind is None) == (label != "contradicted")
        if kind is not None:
            assert kind in cc.applicable_kinds(passage)
        seen.add(label)
    assert seen == set(cc.LABELS)


def test_applicable_kinds_follow_what_the_passage_contains() -> None:
    assert cc.applicable_kinds("No numbers, no sides, no drugs here.") == ("negation",)
    kinds = cc.applicable_kinds("Aspirin 81 mg was given on the left side before discharge.")
    assert set(kinds) == {"number", "laterality", "negation", "temporal", "drug"}


@pytest.mark.parametrize(("reply", "want"), [
    ("SUPPORTED", "supported"), ("CONTRADICTED.", "contradicted"), ("NOT_MENTIONED", "not_mentioned"),
    ("NOT MENTIONED", "not_mentioned"), ("SUPPORTED or CONTRADICTED", None), ("", None), ("I am not sure", None),
])
def test_verdict_parser(reply: str, want: str | None) -> None:
    assert cc.parse_verdict(reply) == want


def test_clean_claim_takes_the_first_line_without_label_or_quotes() -> None:
    assert cc.clean_claim('Claim: "The patient is afebrile."\nSecond line') == "The patient is afebrile."
    assert cc.clean_claim("\n\n  ") == ""


PASSAGE = ("The patient was admitted with chest pain on 3 March. Troponin was 7 ng/mL. "
           "He has no fever. The left knee is swollen. Aspirin was started before discharge. "
           "Surgery was performed after the admission.")


@pytest.mark.parametrize(("kind", "claim", "ok"), [
    ("number", "Troponin was 9 ng/mL on arrival.", True),           # 9 is not in the passage
    ("number", "Troponin was 7 ng/mL.", False),                      # every number occurs in the passage
    ("laterality", "The right knee is swollen.", True),
    ("laterality", "The left knee is swollen.", False),
    ("negation", "He has a fever.", True),                           # claim affirms what the passage negates
    ("negation", "He has no fever.", False),
    ("temporal", "Surgery was performed before the admission.", True),
    ("temporal", "Surgery was performed after the admission.", False),
    ("drug", "Warfarin was started before discharge.", True),
    ("drug", "Aspirin was started before discharge.", False),
])
def test_edit_rules_find_the_changed_element(kind: str, claim: str, ok: bool) -> None:
    assert cc.edit_ok(kind, claim, PASSAGE) is ok


def test_copy_share_flags_a_verbatim_run() -> None:
    # the claim has nine words; seven of them run together in the first passage sentence
    assert cc.copy_share("The patient was admitted with chest pain and nausea.", PASSAGE) == pytest.approx(7 / 9)
    verbatim = "The patient was admitted with chest pain on 3 March"
    assert cc.copy_share(verbatim, PASSAGE) == pytest.approx(1.0)
    paraphrase = "A chest complaint led to hospital entry in early March."
    assert cc.copy_share(paraphrase, PASSAGE) < 0.5


def test_keep_requires_the_verifier_to_agree_and_the_rule_to_hold() -> None:
    assert cc.keep("supported", None, "The patient is afebrile.", PASSAGE, "supported") == (True, "kept")
    assert cc.keep("contradicted", "drug", "Warfarin was started before discharge.", PASSAGE, "contradicted") == (True, "kept")
    assert cc.keep("contradicted", "drug", "Aspirin was started before discharge.", PASSAGE, "contradicted") == (
        False, "edit_rule_failed")
    assert cc.keep("supported", None, "He has no fever.", PASSAGE, "contradicted") == (False, "verifier_disagrees")
    assert cc.keep("supported", None, "x", PASSAGE, None) == (False, "verifier_no_single_class")


def test_option_order_is_a_stable_permutation_and_the_gold_key_names_the_label() -> None:
    order = cc.option_order("rec-1")
    assert sorted(order) == sorted(cc.LABELS) and order == cc.option_order("rec-1")
    row = cc.build_row("rec-1", PASSAGE, "He has a fever.", "contradicted", {"source": "multicare"}, licence="CC BY 4.0",
                       record_url="https://example.org/x", split="train")
    gold = next(o for o in row["options"] if o["key"] == row["gold"])
    assert gold["label"] == cc.LABEL_TEXT["contradicted"]
    assert row["template_id"] == cc.TEMPLATE and row["meta"]["label"] == "contradicted"


# ---- restyled pages -----------------------------------------------------------------------------------------------------

def test_restyle_draws_only_narrative_classes_and_writer_names_the_type() -> None:
    for i in range(50):
        assert rs.draw_class("doc", i) in rs.RESTYLE_CLASSES
    messages = rs.writer_messages("A 53-year-old woman with pneumonia.", "discharge_summary")
    assert "Discharge summary" in messages[1]["content"]
    with pytest.raises(ValueError):
        rs.writer_messages("x", "bill")


def test_classifier_letters_and_parse() -> None:
    assert rs.parse_letter("C") == "C"
    assert rs.parse_letter("c. Progress note") == "C"
    assert rs.parse_letter("The answer is C") is None
    assert rs.parse_letter("") is None
    listing = rs.classifier_messages("page")[1]["content"]
    assert f"{PAGE_LETTER['discharge_summary']}. Discharge summary" in listing


def test_page_checks_and_keep() -> None:
    page = " ".join(["word"] * 150)
    assert rs.page_check(page) == "kept"
    assert rs.page_check("too short") == "page_too_short"
    assert rs.page_check(" ".join(["word"] * 150) + " As an AI I cannot") == "refusal_text"
    assert rs.keep("admission", page, PAGE_LETTER["admission"]) == (True, "kept")
    assert rs.keep("admission", page, PAGE_LETTER["progress_note"]) == (False, "verifier_disagrees")
    assert rs.keep("admission", page, None) == (False, "verifier_no_letter")


def test_clean_page_drops_fences_and_a_preface() -> None:
    assert rs.clean_page("Here is the page:\n```\nAdmission note\nText\n```") == "Admission note\nText"


def test_restyle_row_gold_is_the_drawn_class_letter() -> None:
    row = rs.build_row("r1", " ".join(["word"] * 80), "nutrition", {"source": "mtsamples"}, licence="unrecorded",
                       record_url="https://example.org/r", split="dev")
    assert row["gold"] == PAGE_LETTER["nutrition"] and len(row["options"]) == 12


# ---- substrate: splits and passage windows ----------------------------------------------------------------------------

def test_split_of_is_stable_and_roughly_80_10_10() -> None:
    keys = [f"doc-{i}" for i in range(4000)]
    got = [sub.split_of(k) for k in keys]
    assert got == [sub.split_of(k) for k in keys]
    for name, want in (("train", 0.8), ("dev", 0.1), ("test", 0.1)):
        assert abs(got.count(name) / len(keys) - want) < 0.03


def test_paragraphs_split_on_blank_lines_then_on_newlines() -> None:
    assert sub.paragraphs("A one.\n\nB two.\n\n\nC three.") == ["A one.", "B two.", "C three."]
    assert sub.paragraphs("line one\nline two") == ["line one", "line two"]
    assert sub.paragraphs("  ") == []


def test_passage_is_a_window_of_whole_paragraphs_within_the_cap() -> None:
    text = "\n\n".join(f"Paragraph {i} with some words." for i in range(10))
    for key in ("a", "b", "c", "d"):
        got = sub.passage(text, key)
        assert got is not None and len(got) <= sub.PASSAGE_MAX_CHARS
        for para in got.split("\n\n"):
            assert para.startswith("Paragraph ") and para.endswith("words.")
    assert sub.passage("x" * (sub.PASSAGE_MAX_CHARS + 10), "k") is None
    assert sub.passage("   ", "k") is None


def test_claim_balance_gives_equal_labels_and_keeps_the_kinds_as_drawn() -> None:
    rows = []
    for i in range(40):
        rows.append({"key": f"s{i}", "label": "supported", "kind": None})
        rows.append({"key": f"n{i}", "label": "not_mentioned", "kind": None})
    for i, kind in enumerate(cc.EDIT_KINDS * 12):
        rows.append({"key": f"c{i}", "label": "contradicted", "kind": kind})
    kept, report = cc.balance_claims(rows, salt="t")
    labels = Counter(r["label"] for r in kept)
    assert labels["supported"] == labels["not_mentioned"] == labels["contradicted"] == report["per_label_target"] == 40
    assert set(Counter(r["kind"] for r in kept if r["label"] == "contradicted")) == set(cc.EDIT_KINDS)
    assert cc.balance_claims(rows, salt="t")[0] == kept  # deterministic
