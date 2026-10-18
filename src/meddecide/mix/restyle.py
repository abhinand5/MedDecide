"""EHR-style restyling of real case text into a code-chosen page class (osler_v3 X3; D43; ADVISORY X3 step 2): restyle_page_category_v1.

The code draws one narrative page class (discharge summary, admission, progress note, consult, emergency department, nutrition,
radiology). A local Gemma rewrites a real case as a page of that type, keeping its clinical facts. An independent verification
call (a different system prompt, temperature 0) classifies the page into the twelve page classes; an item is kept only if that
class equals the drawn one. The label is the code's draw, never the writer's or the verifier's.

Pure functions only; the generation and the assembly are in scripts/osler_v3/x3_generate.py and x3_assemble.py.
"""

from __future__ import annotations

import re
from typing import Any

from meddecide.mix.rows import make_row
from meddecide.mr.pages import PAGE_CLASSES
from meddecide.mr.realtext import PAGE_LETTER
from meddecide.utils.hashing import stable_hash

TEMPLATE = "restyle_page_category_v1"
SKILL = "ehr_page_category"
RESTYLE_CLASSES = ("discharge_summary", "admission", "progress_note", "consult_note", "emergency_department", "nutrition",
                   "radiology")
PAGE_LABEL = dict(PAGE_CLASSES)
MIN_WORDS, MAX_WORDS = 60, 600
CLASS_CAP = 0.25  # ADVISORY X3: no class above 25 % of the items
SYSTEM_WRITE = (
    "You rewrite a real clinical case as one medical-record page of a stated type. You keep every clinical fact of the case "
    "(diagnoses, findings, numbers, medications, dates) and add no facts. You reply with the page text only."
)
SYSTEM_CLASSIFY = (
    "You classify one medical-record page into one category from a fixed list. You reply with the category letter only."
)
_PREFACE = re.compile(r"^(here is|here's|sure|certainly|below is|rewritten|the page)\b.*:?\s*$", re.IGNORECASE)
_REFUSAL = re.compile(r"\b(i cannot|i can't|as an ai|i am unable)\b", re.IGNORECASE)


def draw_class(doc_key: str, index: int, seed: int = 0) -> str:
    """The narrative page class drawn by code for one item (stable hash)."""
    return RESTYLE_CLASSES[int(stable_hash({"restyle": doc_key, "index": index, "seed": seed}, length=8), 16)
                           % len(RESTYLE_CLASSES)]


def writer_messages(case_text: str, page_class: str) -> list[dict[str, str]]:
    if page_class not in RESTYLE_CLASSES:
        raise ValueError(f"{page_class!r} is not a narrative page class")
    user = (f"PAGE TYPE: {PAGE_LABEL[page_class]}\n\nCASE:\n{case_text}\n\n"
            f"Write the page now, between 120 and 400 words. Keep the clinical facts of the case.")
    return [{"role": "system", "content": SYSTEM_WRITE}, {"role": "user", "content": user}]


def classifier_messages(page_text: str) -> list[dict[str, str]]:
    listing = "\n".join(f"{PAGE_LETTER[key]}. {label}" for key, label in PAGE_CLASSES)
    user = f"CATEGORIES:\n{listing}\n\nPAGE:\n{page_text}\n\nReply with the letter of the category only."
    return [{"role": "system", "content": SYSTEM_CLASSIFY}, {"role": "user", "content": user}]


def clean_page(text: str) -> str:
    """The reply without code fences and without a one-line preface such as 'Here is the page:'."""
    lines = [ln.rstrip() for ln in str(text).strip().splitlines() if not ln.strip().startswith("```")]
    while lines and (not lines[0].strip() or _PREFACE.match(lines[0].strip())):
        lines.pop(0)
    return "\n".join(lines).strip()


def parse_letter(text: str) -> str | None:
    """The category letter a classifier reply starts with (for example 'C' or 'C.'), or None."""
    match = re.match(r"\s*\(?([A-L])(?![A-Za-z])", str(text).upper())
    return match.group(1) if match else None


def page_check(page_text: str) -> str:
    """'kept' when the page is a usable page; otherwise the reason it is not."""
    words = len(page_text.split())
    if words < MIN_WORDS:
        return "page_too_short"
    if words > MAX_WORDS:
        return "page_too_long"
    if _REFUSAL.search(page_text):
        return "refusal_text"
    return "kept"


def keep(page_class: str, page_text: str, verified_letter: str | None) -> tuple[bool, str]:
    """Whether a restyled page is kept, and the reason (every drop has one)."""
    reason = page_check(page_text)
    if reason != "kept":
        return False, reason
    if verified_letter is None:
        return False, "verifier_no_letter"
    if verified_letter != PAGE_LETTER[page_class]:
        return False, "verifier_disagrees"
    return True, "kept"


def build_row(record_id: str, page_text: str, page_class: str, meta: dict[str, Any], *, licence: str, record_url: str,
              split: str) -> dict[str, Any]:
    options = [(PAGE_LETTER[key], label) for key, label in PAGE_CLASSES]
    return make_row(source=str(meta["source"]), template_id=TEMPLATE, skill=SKILL, qtype="choice", state=page_text,
                    question="Which type of medical-record page is this?", options=options, gold=PAGE_LETTER[page_class],
                    record_id=record_id, record_url=record_url, licence=licence, record_date="", split=split,
                    tier="osler_v3_source", meta=dict(meta, page_class=page_class))
