"""Teacher-written variants of the osler_v1 generated decisions (ADVISORY V4; D27).

The decision is the generator's: the question, the options, the gold and the meta come from the code that drew the facts
(``mr_icd_support_template_v1`` and ``indicator_pair``). Only the state text changes: a teacher writes a record page from
the code-built facts, and ``meddecide.mr.teacher_prose`` keeps the page only when a second call extracts the same facts.
Nothing here reads a teacher answer, a teacher option distribution or a teacher judgement.

A page that contains an ICD-10-CM code string is dropped (ADVISORY V4, failure mode 8.1): the teacher is not allowed to
write the coding decision into the record.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping
from typing import Any

from meddecide.gen.core import GeneratedItem, make_item
from meddecide.mr import icd
from meddecide.mr.criteria import ChunkFacts, supports
from meddecide.mr.generators import YN
from meddecide.mr.teacher_prose import ProseRequest, chunk_facts, diagnosis_facts

# ADVISORY V4: a page class from this set; outpatient pages only as progress or consultation notes
ICD_PAGE_CLASSES: dict[str, tuple[str, ...]] = {
    "inpatient": ("discharge_summary", "admission", "progress_note", "consult_note"),
    "outpatient": ("progress_note", "consult_note"),
}
# the indicator pages are written as progress notes (ADVISORY V4 mr_indicator_prose_v1)
INDICATOR_PAGE_CLASS = "progress_note"

# the ADVISORY's code-string regex: a letter other than U, a digit, a digit or A/B, then optional dot and characters
CODE_STRING = re.compile(r"[A-TV-Z][0-9][0-9AB]\.?[0-9A-TV-Z]{0,4}")


def has_code_string(text: str) -> bool:
    """True when the text contains something shaped like an ICD-10-CM code (ADVISORY V4 regex)."""
    return CODE_STRING.search(text) is not None


def icd_page_class(rng: random.Random, setting: str) -> str:
    if setting not in ICD_PAGE_CLASSES:
        raise ValueError(f"unknown setting {setting!r}")
    return rng.choice(ICD_PAGE_CLASSES[setting])


def dx_from_meta(meta: Mapping[str, Any]) -> icd.Diagnosis:
    """The diagnosis a template item was drawn from, rebuilt from its meta (the draw is not repeated)."""
    return icd.Diagnosis(
        family=str(meta["family"]),
        attributes={str(k): str(v) for k, v in dict(meta["attributes"]).items()},
        certainty=meta["certainty"],
        subject=meta["subject"],
        status=meta["status"],
    )


def icd_request(dx: icd.Diagnosis, setting: str, page_class: str) -> ProseRequest:
    return ProseRequest(kind="diagnosis", page_class=page_class, facts=diagnosis_facts(dx), setting=setting,
                        source="generated")


def indicator_request(facts: ChunkFacts) -> ProseRequest:
    return ProseRequest(kind="chunk", page_class=INDICATOR_PAGE_CLASS, facts=chunk_facts(facts), setting="inpatient",
                        source="generated")


def icd_prose_item(template: GeneratedItem, pid: str, text: str, page_class: str) -> GeneratedItem:
    """The template item's decision with the teacher's page as its state (generator ``mr_icd_support_prose_v1``)."""
    meta = dict(template.meta)
    meta.update({"source": "teacher_prose", "page_class": page_class, "template_item": template.item_id})
    return make_item("mr_icd_support_prose_v1", "MR_icd_support", pid, template.role, template.qtype, text,
                     template.question, list(template.options), template.gold, meta)


def indicator_prose_item(pid: str, role: str, facts: ChunkFacts, indicator: str, trap: str | None, question: str,
                         text: str) -> GeneratedItem:
    """A yes/no evidence item whose gold is ``criteria.supports`` on the facts (generator ``mr_indicator_prose_v1``)."""
    gold = "yes" if supports(indicator, facts) else "no"
    meta = {"indicator": indicator, "trap": trap, "page": INDICATOR_PAGE_CLASS, "page_class": INDICATOR_PAGE_CLASS,
            "source": "teacher_prose"}
    return make_item("mr_indicator_prose_v1", "MR_indicator_evidence", pid, role, "noul", text, question, YN, gold,
                     meta)
