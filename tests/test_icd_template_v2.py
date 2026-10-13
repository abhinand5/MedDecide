"""The v2 ICD template (osler_v1 V1): twins flip, the question names the target condition, the distractors are never a held-out
family and never share a content word with the target, and both twins of a pair share their distractors."""

from __future__ import annotations

from meddecide.mr import icd
from meddecide.mr.generators import (
    HELD_OUT_ICD_FAMILIES,
    _content_words,
    mr_icd_support_template_v2,
)


def _pairs(n: int):
    for k in range(n):
        items = mr_icd_support_template_v2(f"icd-v2-test-{k:07d}")
        base = next(i for i in items if i.role == "base")
        twin = next(i for i in items if i.role == "twin")
        yield base, twin


def test_twins_flip_and_the_question_names_the_target_condition() -> None:
    for base, twin in _pairs(120):
        family = icd.FAMILY_BY_KEY[base.meta["family"]]
        assert base.gold != twin.gold
        assert family.condition in base.question and family.condition in twin.question


def test_the_target_condition_is_stated_once_and_the_page_has_three_diagnosis_lines() -> None:
    for base, twin in _pairs(120):
        family = icd.FAMILY_BY_KEY[base.meta["family"]]
        for item in (base, twin):
            assert item.state.lower().count(family.condition.lower()) == 1
            assert len(item.state.split("\n")) == 5  # head, target, two distractors, plan


def test_distractors_are_other_families_that_are_not_held_out_and_share_no_word_with_the_target() -> None:
    for base, twin in _pairs(200):
        target = icd.FAMILY_BY_KEY[base.meta["family"]]
        distractors = base.meta["distractor_families"]
        assert len(distractors) == 2 and len(set(distractors)) == 2
        assert base.meta["family"] not in distractors
        assert not set(distractors) & HELD_OUT_ICD_FAMILIES
        for key in distractors:
            assert not _content_words(icd.FAMILY_BY_KEY[key].condition) & _content_words(target.condition)
        assert base.meta["distractor_families"] == twin.meta["distractor_families"]
        # the distractor facts are stored with the item, so the teacher variant can write them (V4)
        assert [d["family"] for d in base.meta["distractors"]] == distractors
        assert base.meta["distractors"] == twin.meta["distractors"]


def test_the_held_out_family_never_appears_as_a_distractor_or_in_training_items() -> None:
    for k in range(300):
        for item in mr_icd_support_template_v2(f"held-out-scan-{k:07d}"):
            if item.meta["family"] in HELD_OUT_ICD_FAMILIES:
                continue
            assert "heart failure" not in item.state.lower()
