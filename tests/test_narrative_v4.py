"""Narrative pages from Synthea encounters (osler_v1 ADVISORY V4, synthea_narrative_page_v1): the fact cleaning, the page-class
rule and the item builder, without a network call."""

from __future__ import annotations

from meddecide.gen.core import split_of
from meddecide.mr.generators import PAGE_LETTER
from meddecide.mr.narrative import (
    clean_fact_text,
    narrative_item,
    narrative_page_class,
    narrative_request,
)
from meddecide.mr.pages import PAGE_CLASSES
from meddecide.mr.synthea import NARRATIVE_PAGES, SyntheaEncounter


def _encounter(encounter_class: str = "emergency", conditions: tuple[str, ...] = ("Fracture of bone (disorder)",)
               ) -> SyntheaEncounter:
    return SyntheaEncounter(encounter_id="enc-1", patient_id="pat-1", encounter_class=encounter_class,
                            description="Emergency room admission (procedure) (finding)", reason="Fall (event)",
                            age=19, sex="male", observations=(), meds=(), charges=(), conditions=conditions)


def test_snomed_semantic_tags_are_removed_and_plain_text_is_kept() -> None:
    assert clean_fact_text("Fracture of bone (disorder)") == "Fracture of bone"
    assert clean_fact_text("chronic pain (finding)") == "chronic pain"
    assert clean_fact_text("Chronic pain") == "Chronic pain"
    assert clean_fact_text("Type 2 diabetes (2 of 3)") == "Type 2 diabetes (2 of 3)"  # not a semantic tag


def test_the_teacher_request_carries_cleaned_conditions() -> None:
    req = narrative_request(_encounter(), "discharge_summary")
    assert req.kind == "page" and req.page_class == "discharge_summary" and req.source == "synthea"
    assert req.facts["conditions"] == ["Fracture of bone"]
    assert req.setting == "outpatient"  # an emergency encounter
    assert narrative_request(_encounter("inpatient"), "admission").setting == "inpatient"


def test_radiology_is_drawn_only_when_the_encounter_has_an_imaging_procedure() -> None:
    for k in range(300):
        assert narrative_page_class(f"enc-{k}", has_imaging=False) != "radiology"
    drawn = {narrative_page_class(f"enc-{k}", has_imaging=True) for k in range(300)}
    assert "radiology" in drawn and drawn <= set(NARRATIVE_PAGES)


def test_the_item_asks_for_the_page_category_with_the_drawn_class_as_gold_and_the_patient_split() -> None:
    enc = _encounter()
    item = narrative_item(enc, "consult_note", "Consultation note text.", split_of("synthea-pat-1"))
    assert item.generator == "synthea_narrative_page_v1" and item.qtype == "choice" and item.role == "base"
    assert item.gold == PAGE_LETTER["consult_note"]
    assert len(item.options) == len(PAGE_CLASSES) == 12
    assert item.split == split_of("synthea-pat-1")
    assert item.meta["page_class"] == "consult_note" and item.meta["source"] == "synthea_teacher_prose"
