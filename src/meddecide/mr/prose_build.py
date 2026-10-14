"""Per-patient-id processing for the teacher-prose builds (osler_v1 ADVISORY V4).

Each function takes one patient id and a chat client, draws the decision with the code that owns it, asks the teacher for
the page(s), verifies them, and returns an ``Outcome``. The build script (``scripts/osler_v1/v4_teacher_build.py``) only
schedules these calls and writes their results. Teacher text is returned to the caller and never logged here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from meddecide.gen.core import GeneratedItem, rng_for
from meddecide.mr.generators import (
    HELD_OUT_INDICATORS,
    indicator_pair,
    is_held_out,
    mr_icd_support_template_v2,
)
from meddecide.mr.narrative import (
    EncounterRecord,
    narrative_item,
    narrative_page_class,
    narrative_request,
)
from meddecide.mr.prose_items import (
    dx_from_dict,
    dx_from_meta,
    has_code_string,
    icd_multi_request,
    icd_page_class,
    icd_prose_item,
    indicator_prose_item,
    indicator_request,
)
from meddecide.mr.teacher_prose import ProseResult, generate

PROSE_PREFIX = {"icd": "mr_icd_support_prose_v1", "indicator": "mr_indicator_prose_v1",
                "narrative": "synthea_narrative_page_v1"}


class ChatLike(Protocol):
    local: bool

    def chat(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int) -> str: ...


@dataclass
class Outcome:
    pid: str
    verdict: str                                    # accepted | rejected | skipped | error
    reasons: list[str] = field(default_factory=list)
    items: list[GeneratedItem] = field(default_factory=list)
    calls: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


def bucket(reason: str) -> str:
    """A coarse, count-friendly name for one verification problem (the text of the problem is not kept)."""
    if reason.startswith("attribute "):
        return "attribute"
    if reason.startswith("missing or changed observation"):
        return "observation_value"
    if reason.startswith("indicator-relevant observation count"):
        return "observation_count"
    if reason.startswith("malformed observation"):
        return "observation_malformed"
    if reason.startswith("condition not reported"):
        return "condition_not_reported"
    for key in ("certainty", "subject", "status", "page_type", "medications"):
        if reason.startswith(key):
            return key
    return reason.split(":")[0][:40]


def _rejected(pid: str, result: ProseResult) -> Outcome:
    return Outcome(pid, "rejected", [bucket(r) for r in result.reasons])


def process_icd(pid: str, client: ChatLike) -> Outcome:
    """One base and twin of ``mr_icd_support_template_v2``, each written as a page of the same class and setting that
    documents the target diagnosis and the two distractor diagnoses of the template."""
    base, twin = mr_icd_support_template_v2(pid)
    if is_held_out(base.meta):  # heart_failure is held out of every training source (ADVISORY §3.3)
        return Outcome(pid, "skipped", ["held-out family"])
    setting = str(base.meta["setting"])
    page_class = icd_page_class(rng_for("mr_icd_support_prose_v1", pid, "page"), setting)
    distractors = [dx_from_dict(d) for d in base.meta["distractors"]]
    pages: list[tuple[GeneratedItem, str]] = []
    for template in (base, twin):
        result = generate(client, icd_multi_request(dx_from_meta(template.meta), distractors, setting, page_class))
        if not result.ok:
            return _rejected(pid, result)
        if has_code_string(result.text or ""):
            return Outcome(pid, "rejected", ["icd_code_string"])
        pages.append((template, result.text or ""))
    return Outcome(pid, "accepted", [], [icd_prose_item(t, pid, text, page_class) for t, text in pages])


def process_indicator(pid: str, client: ChatLike) -> Outcome:
    """One base and twin of ``mr_indicator_evidence_v1``, each written as a progress note from its facts."""
    pair = indicator_pair(pid)
    if pair.indicator in HELD_OUT_INDICATORS:  # aki_creatinine_rise is held out of every training source (§3.3)
        return Outcome(pid, "skipped", ["held-out indicator"])
    pages: list[tuple[str, Any, str]] = []
    for role, facts in pair.order:
        result = generate(client, indicator_request(facts))
        if not result.ok:
            return _rejected(pid, result)
        if has_code_string(result.text or ""):
            return Outcome(pid, "rejected", ["icd_code_string"])
        pages.append((role, facts, result.text or ""))
    items = [indicator_prose_item(pid, role, facts, pair.indicator, pair.trap, pair.question, text)
             for role, facts, text in pages]
    return Outcome(pid, "accepted", [], items)


def process_narrative(pid: str, client: ChatLike, *, store: Mapping[str, EncounterRecord]) -> Outcome:
    """One Synthea encounter: one teacher page of the class the code draws, kept when it verifies (ADVISORY V4)."""
    enc_id = pid.split(":", 1)[1]
    rec = store[enc_id]
    page_class = narrative_page_class(enc_id, rec.has_imaging)
    result = generate(client, narrative_request(rec.encounter, page_class))
    if not result.ok:
        return _rejected(pid, result)
    if has_code_string(result.text or ""):
        return Outcome(pid, "rejected", ["icd_code_string"])
    return Outcome(pid, "accepted", [], [narrative_item(rec.encounter, page_class, result.text or "", rec.split)])


PROCESS: dict[str, Callable[[str, ChatLike], Outcome]] = {"icd": process_icd, "indicator": process_indicator}
