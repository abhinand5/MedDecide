"""Narrative record pages written by the teacher from Synthea encounters (osler_v1 ADVISORY V4, ``synthea_narrative_page_v1``).

One encounter gives one request: the facts come from ``synthea.narrative_facts`` (synthetic, so a hosted teacher may write
the page, D30). The page class is drawn from ``synthea.NARRATIVE_PAGES`` with radiology only when the encounter has an
imaging procedure. The item asks which page category the page is; its gold is the class the code asked for, and the
verifier (``teacher_prose.verify`` on kind ``page``) keeps the page only when the teacher wrote that kind of page and
stated the listed conditions. Items are split by patient, as every other Synthea item (V5a).
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from meddecide.gen.core import GeneratedItem, make_item, rng_for, split_of
from meddecide.mr.generators import PAGE_LETTER
from meddecide.mr.pages import PAGE_CLASSES
from meddecide.mr.synthea import NARRATIVE_PAGES, SyntheaEncounter, build_encounter, narrative_facts
from meddecide.mr.teacher_prose import ProseRequest, page_facts

# the three questions of the tabular page category, so every page-category item asks the same way
PAGE_QUESTIONS = ("Which type of medical-record page is this?", "Classify this page of the medical record.",
                  "What category does this record page belong to?")
IMAGING = re.compile(r"x-ray|radiograph|imaging|\bct\b|\bmri\b|ultrasound|scan|echocardiogra", re.IGNORECASE)


def narrative_page_class(encounter_id: str, has_imaging: bool) -> str:
    """The page class for an encounter: a deterministic draw from the narrative pages (radiology needs an imaging procedure)."""
    pages = [p for p in NARRATIVE_PAGES if has_imaging or p != "radiology"]
    return rng_for("synthea_narrative_page_v1", encounter_id, "class").choice(pages)


SEMANTIC_TAG = re.compile(r"\s*\((disorder|finding|procedure|event|situation|observable entity|substance|organism|"
                          r"body structure|qualifier value|product|regime/therapy)\)\s*$", re.IGNORECASE)


def clean_fact_text(text: str) -> str:
    """A Synthea descriptor without its SNOMED semantic tag: "Fracture of bone (disorder)" -> "Fracture of bone"."""
    return SEMANTIC_TAG.sub("", text).strip()


def narrative_facts_clean(enc: SyntheaEncounter) -> dict[str, object]:
    facts = dict(narrative_facts(enc))
    facts["conditions"] = [clean_fact_text(c) for c in facts["conditions"]]
    facts["encounter"] = clean_fact_text(str(facts["encounter"]))
    facts["reason"] = clean_fact_text(str(facts["reason"]))
    facts["medications"] = [clean_fact_text(m) for m in facts["medications"]]
    return facts


def narrative_request(enc: SyntheaEncounter, page_class: str) -> ProseRequest:
    setting = "inpatient" if enc.encounter_class == "inpatient" else "outpatient"
    return ProseRequest(kind="page", page_class=page_class, facts=page_facts(narrative_facts_clean(enc)), setting=setting,
                        source="synthea")


def narrative_item(enc: SyntheaEncounter, page_class: str, text: str, split: str) -> GeneratedItem:
    """The page-category item for one verified narrative page (generator ``synthea_narrative_page_v1``)."""
    question = rng_for("synthea_narrative_page_v1", enc.encounter_id, "question").choice(PAGE_QUESTIONS)
    options = [(PAGE_LETTER[k], label) for k, label in PAGE_CLASSES]
    item = make_item("synthea_narrative_page_v1", "MR_page_category", f"synthea-narr:{enc.encounter_id}", "base",
                     "choice", text, question, options, PAGE_LETTER[page_class],
                     {"page_class": page_class, "source": "synthea_teacher_prose", "encounter_class": enc.encounter_class})
    return replace(item, split=split)


@dataclass(frozen=True)
class EncounterRecord:
    encounter: SyntheaEncounter
    has_imaging: bool
    split: str


def load_encounters(data_dir: str | Path) -> dict[str, EncounterRecord]:
    """The extracted Synthea encounters (V5a), built with the V5a builder, keyed by encounter id."""
    import pandas as pd

    root = Path(data_dir)
    patients = {r["Id"]: r for r in pd.read_parquet(root / "patients.parquet").to_dict("records")}
    encounters = pd.read_parquet(root / "encounters.parquet").to_dict("records")
    grouped: dict[str, dict[str, list[dict]]] = {}
    for table in ("observations", "medications", "procedures", "conditions"):
        bucket: dict[str, list[dict]] = {}
        for r in pd.read_parquet(root / f"{table}.parquet").to_dict("records"):
            bucket.setdefault(r["ENCOUNTER"], []).append(r)
        grouped[table] = bucket
    counts: dict[str, int] = {}
    out: dict[str, EncounterRecord] = {}
    for e in encounters:
        patient = patients.get(e["PATIENT"])
        if patient is None:
            continue
        enc = build_encounter(e, patient, grouped["observations"].get(e["Id"], []),
                              grouped["medications"].get(e["Id"], []), grouped["procedures"].get(e["Id"], []),
                              grouped["conditions"].get(e["Id"], []), counts)
        procedures = [str(p.get("DESCRIPTION", "")) for p in grouped["procedures"].get(e["Id"], [])]
        has_imaging = any(IMAGING.search(d) for d in procedures)
        out[str(e["Id"])] = EncounterRecord(enc, has_imaging, split_of(f"synthea-{enc.patient_id}"))
    return out


def sample_encounters(records: Mapping[str, EncounterRecord], n: int, *, seed: int = 0) -> list[str]:
    """A deterministic sample of encounter ids (for the pilot)."""
    ids = sorted(records)
    rng = random.Random(seed)
    return sorted(rng.sample(ids, min(n, len(ids))))
