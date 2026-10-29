"""Tests for the grounded generator's pure code (facts, render, verify, manifest). Synthetic fixtures only."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from meddecide.synth.facts import CaseFacts, Fact, TemplateSpec, fact_list_id
from meddecide.synth.manifest import MANIFEST_SCHEMA, build_manifest, summarize, write_manifest
from meddecide.synth.render import (
    LabelLeakError,
    StyleSpec,
    render_request,
    render_system_prompt,
    render_user_prompt,
)
from meddecide.synth.verify import (
    REASON_BANNED_LABEL,
    REASON_BANNED_META,
    REASON_EMPTY,
    REASON_LENGTH,
    REASON_MISSING_FACT,
    Verdict,
    contains_phrase,
    tokenize,
    verify_text,
)

LABEL = "pneumonia"


def _facts() -> tuple[Fact, ...]:
    return (
        Fact("demographic", "age", "67 years"),
        Fact("finding", "cough", "productive cough for 5 days"),
        Fact("lab", "crp", "CRP 84 mg/L"),
        Fact("medication", "antibiotic", "amoxicillin 500 mg three times daily"),
    )


def _case(label: str = LABEL, facts: tuple[Fact, ...] | None = None) -> CaseFacts:
    return CaseFacts(
        family="toy_chest", facts=facts if facts is not None else _facts(), label=label
    )


STYLE = StyleSpec(
    name="plain_note",
    register="outpatient clinical note",
    instructions=("Use short sentences.", "Avoid headings."),
)

GOOD_TEXT = (
    "A 67 years old patient attends with a productive cough for 5 days. CRP 84 mg/L on arrival. "
    "Started on amoxicillin 500 mg three times daily and reviewed in clinic next week."
)


# ---- stable ids ----------------------------------------------------------------------------------


def test_case_id_is_stable_and_short() -> None:
    first = _case().case_id
    second = _case().case_id
    assert first == second
    assert re.fullmatch(r"[0-9a-f]{16}", first)


def test_case_id_ignores_fact_order() -> None:
    reversed_facts = tuple(reversed(_facts()))
    assert _case().case_id == _case(facts=reversed_facts).case_id


def test_case_id_changes_with_facts_and_family() -> None:
    base = _case().case_id
    changed_value = (*_facts()[:-1], Fact("medication", "antibiotic", "amoxicillin 250 mg"))
    assert _case(facts=changed_value).case_id != base
    other_family = CaseFacts(family="other_chest", facts=_facts(), label=LABEL)
    assert other_family.case_id != base


def test_case_id_does_not_depend_on_unicode_composition() -> None:
    composed = (Fact("finding", "note", "café visit"),)
    decomposed = (Fact("finding", "note", "café visit"),)
    assert fact_list_id("toy_chest", composed) == fact_list_id("toy_chest", decomposed)


def test_case_rejects_duplicates_and_empty_inputs() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        CaseFacts(family="toy_chest", facts=(_facts()[0], _facts()[0]), label=LABEL)
    with pytest.raises(ValueError, match="at least one fact"):
        CaseFacts(family="toy_chest", facts=(), label=LABEL)
    with pytest.raises(ValueError, match="unknown fact kind"):
        Fact("rumour", "x", "y")
    with pytest.raises(ValueError, match="non-empty"):
        Fact("lab", "x", "   ")


def test_template_options_must_offer_the_label() -> None:
    template = TemplateSpec(
        family="toy_chest",
        label_set=(LABEL, "asthma"),
        option_builder=lambda facts: ("asthma", "bronchitis"),
    )
    with pytest.raises(ValueError, match="not among the offered options"):
        template.options_for(_case())
    ok = TemplateSpec(
        family="toy_chest",
        label_set=(LABEL, "asthma"),
        option_builder=lambda facts: (LABEL, "asthma"),
    )
    assert ok.options_for(_case()) == (LABEL, "asthma")


# ---- render --------------------------------------------------------------------------------------


def test_render_excludes_label_from_system_and_user_prompt() -> None:
    case = _case()
    user = render_user_prompt(case, target_words=180)
    system = render_system_prompt(STYLE)
    assert LABEL not in user.lower()
    assert LABEL not in system.lower()
    assert LABEL not in render_request(case, target_words=180, style=STYLE).lower()


def test_render_lists_only_the_facts_and_target_length() -> None:
    user = render_user_prompt(_case(), target_words=180)
    assert "Write about 180 words." in user
    for fact in _facts():
        assert f"{fact.kind} | {fact.key}: {fact.value}" in user
    assert user.count("\n- ") == len(_facts())


def test_render_refuses_when_a_fact_carries_the_label() -> None:
    leaky = (*_facts(), Fact("finding", "impression", "findings consistent with pneumonia"))
    with pytest.raises(LabelLeakError) as info:
        render_user_prompt(_case(facts=leaky), target_words=180)
    # The error names an index, never the label value.
    assert LABEL not in str(info.value).lower()


def test_render_checks_extra_forbidden_labels_in_system_prompt() -> None:
    style_with_label = StyleSpec(name="x", register="pneumonia clinic note")
    with pytest.raises(LabelLeakError):
        render_system_prompt(style_with_label, forbidden=("pneumonia",))


def test_render_checks_extra_forbidden_labels_in_user_prompt() -> None:
    leaky = (*_facts(), Fact("finding", "impression", "bronchitis suspected"))
    with pytest.raises(LabelLeakError):
        render_request(
            _case(label="asthma", facts=leaky),
            target_words=120,
            style=STYLE,
            forbidden=("bronchitis",),
        )
    clean = render_request(
        _case(label="asthma"), target_words=120, style=STYLE, forbidden=("bronchitis",)
    )
    assert "asthma" not in clean.lower()
    assert "bronchitis" not in clean.lower()


def test_render_rejects_out_of_range_target() -> None:
    with pytest.raises(ValueError, match="target_words"):
        render_user_prompt(_case(), target_words=5)


# ---- verify: fact matching -----------------------------------------------------------------------


def test_tokenize_folds_case_punctuation_and_unicode() -> None:
    assert tokenize("CRP 84 mg/L.") == ("crp", "84", "mg", "l")
    assert tokenize("Café") == tokenize("café")


def test_contains_phrase_respects_token_boundaries() -> None:
    assert contains_phrase("dose of 15 mg", "5 mg") is False
    assert contains_phrase("dose of 5 mg", "5 mg") is True
    assert contains_phrase("CRP 84 mg/l", "crp 84 MG/L") is True
    assert contains_phrase("text", "") is False


def test_fact_matching_is_case_and_punctuation_insensitive() -> None:
    text = (
        "Seen today, a 67 years old man. CRP 84 mg / L. productive COUGH for 5 days, "
        "amoxicillin 500 mg three times daily."
    )
    verdict = verify_text(text, _case())
    assert verdict.missing_facts == ()
    assert verdict.kept


def test_missing_fact_is_a_reason_with_the_value_listed() -> None:
    text = GOOD_TEXT.replace("CRP 84 mg/L", "inflammatory markers were raised")
    verdict = verify_text(text, _case())
    assert not verdict.kept
    assert verdict.reasons == (REASON_MISSING_FACT,)
    assert verdict.missing_facts == ("CRP 84 mg/L",)
    assert verdict.primary_reason == REASON_MISSING_FACT


def test_partial_number_does_not_satisfy_a_fact() -> None:
    text = GOOD_TEXT.replace("for 5 days", "for 15 days")
    verdict = verify_text(text, _case())
    assert verdict.missing_facts == ("productive cough for 5 days",)


def test_good_text_is_kept_with_no_reasons() -> None:
    verdict = verify_text(GOOD_TEXT, _case(), word_range=(20, 80))
    assert verdict.kept
    assert verdict.reasons == ()
    assert verdict.primary_reason is None
    assert verdict.banned_hits == ()
    assert verdict.case_id == _case().case_id


# ---- verify: banned tokens and reasons -----------------------------------------------------------


def test_label_in_text_is_rejected_as_banned_label() -> None:
    text = GOOD_TEXT + " Overall this looks like pneumonia."
    verdict = verify_text(text, _case())
    assert REASON_BANNED_LABEL in verdict.reasons
    assert verdict.primary_reason == REASON_BANNED_LABEL


def test_meta_token_is_rejected_and_hit_recorded() -> None:
    text = GOOD_TEXT + " The correct answer is not stated."
    verdict = verify_text(text, _case())
    assert verdict.reasons == (REASON_BANNED_META,)
    assert verdict.banned_hits == ("answer", "correct answer")


def test_word_boundaries_apply_to_banned_phrases() -> None:
    # "answers" and "labelled" are different tokens from "answer" and "label".
    text = GOOD_TEXT + " The patient answers questions and the lesion was labelled on imaging."
    assert verify_text(text, _case()).kept


def test_length_reason_only_when_range_given_and_violated() -> None:
    assert verify_text(GOOD_TEXT, _case()).kept
    verdict = verify_text(GOOD_TEXT, _case(), word_range=(100, 120))
    assert verdict.reasons == (REASON_LENGTH,)
    assert verdict.word_count == len(GOOD_TEXT.split())


def test_empty_text_has_priority_and_lists_every_missing_fact() -> None:
    verdict = verify_text("   ", _case(), word_range=(10, 20))
    assert verdict.reasons == (REASON_EMPTY, REASON_MISSING_FACT, REASON_LENGTH)
    assert verdict.primary_reason == REASON_EMPTY
    assert len(verdict.missing_facts) == len(_facts())


def test_multiple_reasons_keep_fixed_priority_order() -> None:
    text = "Diagnosis of pneumonia. As an AI I cannot say more."
    verdict = verify_text(text, _case(), word_range=(100, 200))
    assert verdict.reasons == (
        REASON_BANNED_LABEL,
        REASON_BANNED_META,
        REASON_MISSING_FACT,
        REASON_LENGTH,
    )
    assert verdict.primary_reason == REASON_BANNED_LABEL


# ---- manifest ------------------------------------------------------------------------------------

SECRET_VALUE = "amoxicillin 500 mg three times daily"


def _verdicts() -> list[Verdict]:
    case = _case()
    other = CaseFacts(
        family="toy_other", facts=(Fact("finding", "rash", "rash on forearm"),), label="asthma"
    )
    return [
        verify_text(GOOD_TEXT, case),
        verify_text("short", case),
        verify_text(GOOD_TEXT.replace(SECRET_VALUE, "dose unknown"), case),
        verify_text(GOOD_TEXT + " This is pneumonia.", case),
        verify_text("rash on forearm noted, unclear cause", other),
    ]


def test_summarize_counts_and_keep_rates_per_family() -> None:
    summaries = summarize(_verdicts())
    chest = summaries["toy_chest"]
    assert chest.generated == 4
    assert chest.kept == 1
    assert chest.rejected == 3
    assert chest.keep_rate == pytest.approx(0.25)
    assert sum(chest.rejected_by_primary_reason.values()) == chest.rejected
    assert chest.rejected_by_primary_reason[REASON_BANNED_LABEL] == 1
    assert chest.rejected_by_any_reason[REASON_MISSING_FACT] == 2
    other = summaries["toy_other"]
    assert other.generated == 1
    assert other.kept == 1
    assert other.keep_rate == 1.0


def test_family_with_no_items_has_no_keep_rate() -> None:
    summaries = summarize([])
    assert summaries == {}
    manifest = build_manifest([], run_name="empty")
    assert manifest["totals"] == {"generated": 0, "kept": 0, "rejected": 0, "keep_rate": None}


def test_manifest_contents_and_totals_close() -> None:
    manifest = build_manifest(
        _verdicts(), run_name="synth_test", config={"seed": 7, "model": "fixture"}
    )
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["item_text_included"] is False
    assert manifest["config"] == {"seed": 7, "model": "fixture"}
    totals = manifest["totals"]
    assert totals["generated"] == 5
    assert totals["kept"] == 2
    assert totals["rejected"] == 3
    assert totals["kept"] + totals["rejected"] == totals["generated"]
    family_sum = sum(f["generated"] for f in manifest["families"].values())
    assert family_sum == totals["generated"]


def test_manifest_never_contains_item_text_or_fact_values(tmp_path: Path) -> None:
    manifest = build_manifest(_verdicts(), run_name="synth_test")
    path = write_manifest(tmp_path / "manifest.json", manifest)
    written = path.read_text(encoding="utf-8")
    assert json.loads(written) == json.loads(json.dumps(manifest))
    for leaked in (
        SECRET_VALUE,
        "CRP 84 mg/L",
        "rash on forearm",
        "productive cough",
        "pneumonia",
        "asthma",
    ):
        assert leaked.lower() not in written.lower()


def test_write_manifest_refuses_text_flag(tmp_path: Path) -> None:
    manifest = build_manifest(_verdicts(), run_name="synth_test")
    manifest["item_text_included"] = True
    with pytest.raises(ValueError, match="item_text_included"):
        write_manifest(tmp_path / "bad.json", manifest)


# ---- binary families: label_match="sentence" -----------------------------------------------------

_YN_SENTENCES = ("the answer is yes", "the answer is no")


def _yn_spec(mode: str) -> TemplateSpec:
    return TemplateSpec(
        family="toy_yesno",
        label_set=("yes", "no"),
        option_builder=lambda facts: ("yes", "no"),
        label_match=mode,
        label_sentences=_YN_SENTENCES if mode == "sentence" else (),
    )


def _yn_case(facts: tuple[Fact, ...] | None = None) -> CaseFacts:
    return CaseFacts(
        family="toy_yesno",
        facts=facts if facts is not None else (Fact("finding", "fever", "no fever"),),
        label="no",
    )


_YN_TEXT = "The patient reports no fever and has been well since Monday."


def test_label_match_defaults_to_token_and_validates() -> None:
    assert _yn_spec("token").label_match == "token"
    with pytest.raises(ValueError, match="label_match"):
        TemplateSpec(family="f", label_set=("a", "b"), option_builder=lambda f: ("a", "b"), label_match="x")
    with pytest.raises(ValueError, match="at least one label sentence"):
        TemplateSpec(family="f", label_set=("a", "b"), option_builder=lambda f: ("a", "b"), label_match="sentence")
    with pytest.raises(ValueError, match="two words"):
        TemplateSpec(
            family="f",
            label_set=("a", "b"),
            option_builder=lambda f: ("a", "b"),
            label_match="sentence",
            label_sentences=("yes",),
        )


def test_no_fever_passes_under_sentence_and_fails_under_token() -> None:
    case = _yn_case()
    assert verify_text(_YN_TEXT, case, spec=_yn_spec("sentence")).kept
    token_verdict = verify_text(_YN_TEXT, case, spec=_yn_spec("token"))
    assert token_verdict.reasons == (REASON_BANNED_LABEL,)
    # No spec is the same as the token rule.
    assert verify_text(_YN_TEXT, case).reasons == (REASON_BANNED_LABEL,)


def test_label_sentence_in_text_is_rejected_under_sentence() -> None:
    case = _yn_case()
    for sentence in ("The answer is no.", "So THE ANSWER IS YES, clearly."):
        verdict = verify_text(_YN_TEXT + " " + sentence, case, spec=_yn_spec("sentence"))
        assert REASON_BANNED_LABEL in verdict.reasons


def test_sentence_rule_rejects_spec_for_other_family() -> None:
    with pytest.raises(ValueError, match="does not match template"):
        verify_text(_YN_TEXT, _case(), spec=_yn_spec("sentence"))


def test_render_allows_bare_label_in_fact_under_sentence_only() -> None:
    case = _yn_case()
    prompt = render_user_prompt(case, target_words=100, spec=_yn_spec("sentence"))
    assert "no fever" in prompt
    with pytest.raises(LabelLeakError):
        render_user_prompt(case, target_words=100, spec=_yn_spec("token"))


def test_render_refuses_fact_with_label_sentence_under_sentence() -> None:
    leaky = _yn_case((Fact("finding", "note", "chart says the answer is no"),))
    with pytest.raises(LabelLeakError) as info:
        render_user_prompt(leaky, target_words=100, spec=_yn_spec("sentence"))
    assert "answer is no" not in str(info.value).lower()
    with pytest.raises(LabelLeakError):
        render_request(leaky, target_words=100, style=STYLE, spec=_yn_spec("sentence"))


def test_render_request_sentence_mode_passes_system_prompt_with_bare_word() -> None:
    # The fixed system preamble contains the word "no"; the token rule would refuse every request.
    case = _yn_case()
    with pytest.raises(LabelLeakError):
        render_request(case, target_words=100, style=STYLE, spec=_yn_spec("token"))
    assert "no fever" in render_request(case, target_words=100, style=STYLE, spec=_yn_spec("sentence"))
