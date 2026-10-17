"""Whole synthetic charts for chart-level audit decisions, written by a local teacher (osler_v2 W3b; operator 2026-10-17).

A chart is one Synthea encounter rendered as 7 to 14 pages: code-rendered pages (laboratory, flowsheet, MAR, bill), one
**planted evidence page** for a clinical indicator, and narrative pages (admission note, progress notes, optionally a
consult or nutrition note, and a discharge summary) written by the teacher from the encounter's synthetic facts.
Labels are computed by code only (D20, D27):

- the indicator label reads the code-rendered pages' facts (``criteria.supports`` over all of them); narrative pages are
  written without any vital sign or laboratory value and are checked for that by re-extraction, so prose cannot change
  the label;
- the evidence-page label is the one page whose facts support the indicator (asked only when exactly one does);
- the ICD label is ``icd.supported_code`` of the final diagnosis written into the discharge summary, checked by the
  diagnosis extraction.

The twin chart is the same chart with only the planted page changed (one trap or one value across the threshold), so
the narrative pages are generated once per pair. Only synthetic facts are sent, so any teacher may be used; the local
Gemma 4 server is the intended one (TEACHER_LOCAL=1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from meddecide.gen.core import GeneratedItem, make_item, rng_for
from meddecide.mr import criteria, icd
from meddecide.mr.criteria import ChunkFacts
from meddecide.mr.generators import (
    CASES,
    HELD_OUT_ICD_FAMILIES,
    HELD_OUT_INDICATORS,
    LETTERS,
    _indicator_facts,
)
from meddecide.mr.noise import dress_page
from meddecide.mr.pages import (
    PageContext,
    make_context,
    render_flowsheet_page,
    render_lab_page,
    render_mar_page,
)
from meddecide.mr.synthea import (
    SyntheaEncounter,
    narrative_facts,
    page_facts,
    render_page,
    renderable_pages,
)
from meddecide.mr.teacher_prose import (
    OBS_NAMES,
    ChatClient,
    ProseRequest,
    _json,
    _obs_name,
    diagnosis_facts,
    extract_messages,
    generate,
    verify,
)

NO_NUMBERS = ("Do not state any vital sign or laboratory value (no numbers for temperature, heart rate, blood pressure, "
              "respiratory rate, oxygen saturation or any lab test); refer to results only in words such as 'labs "
              "reviewed'. Do not mention medication administration times or doses.")
ICD_FAMILIES = tuple(f.key for f in icd.FAMILIES if f.key not in HELD_OUT_ICD_FAMILIES)


@dataclass(frozen=True)
class ChartPage:
    page_class: str
    text: str
    facts: ChunkFacts | None          # code pages only; narrative pages carry no measured facts
    source: str                       # "code" or "teacher"


@dataclass(frozen=True)
class ChartResult:
    items: list[GeneratedItem]
    reasons: tuple[str, ...]          # why a chart or an item was dropped (empty when everything was kept)


def _render_evidence(page: str, facts: ChunkFacts, rng: random.Random, ctx: PageContext) -> str:
    if page == "lab":
        body = render_lab_page(ctx, facts, rng)
    elif page == "flowsheet":
        body = render_flowsheet_page(ctx, facts, rng)
    else:
        body = render_mar_page(ctx, facts, rng)
    return dress_page(body, rng, fax_p=0.0)


def measurement_free(client: ChatClient, text: str) -> list[str]:
    """Indicator-relevant measurements a narrative page states (must be none)."""
    req = ProseRequest("chunk", "progress_note", {"observations": [], "medications": []})
    raw = client.chat(extract_messages(req, text), temperature=0.0, max_tokens=1200)
    got = _json(raw) or {}
    return [f"narrative page states {(_obs_name(o.get('name')))}" for o in got.get("observations") or []
            if _obs_name(o.get("name")) in OBS_NAMES]


def _final_dx(enc: SyntheaEncounter, rng: random.Random) -> icd.Diagnosis:
    conditions = " ".join(enc.conditions).lower()
    fams = [f for f in ICD_FAMILIES if icd.FAMILY_BY_KEY[f].condition.split()[0] not in conditions]
    fam = icd.FAMILY_BY_KEY[rng.choice(fams or list(ICD_FAMILIES))]
    attrs = {k: rng.choice(v) for k, v in fam.attributes.items()}
    certainty = rng.choice(("confirmed", "confirmed", "probable", "suspected", "ruled_out"))
    return icd.Diagnosis(fam.key, attrs, certainty)


def _narrative(client: ChatClient, enc: SyntheaEncounter, page_class: str, dx: icd.Diagnosis | None,
               rng: random.Random, ctx: PageContext) -> tuple[str | None, list[str]]:
    facts = {k: v for k, v in narrative_facts(enc).items() if k != "measurements"}
    facts.update({"patient_name": ctx.patient, "mrn": ctx.mrn, "facility": ctx.facility,
                  "admission_date": ctx.admit_date})
    extra = NO_NUMBERS
    if dx is not None:
        facts["final_diagnosis"] = diagnosis_facts(dx)
        extra += (" State the final diagnosis exactly as final_diagnosis gives it (its certainty, attributes and that it "
                  "is the patient's current problem).")
    else:
        extra += " Do not name a final diagnosis."
    req = ProseRequest("page", page_class, facts, setting="inpatient",
                       style={"extra": extra, "length": rng.choice(("about 200 words", "about 300 words",
                                                                    "about 400 words"))},
                       source="synthea-chart")
    res = generate(client, req)
    problems = list(res.reasons)
    if res.text is None:
        return None, problems or ["no text"]
    problems += measurement_free(client, res.text)
    if dx is not None:
        dreq = ProseRequest("diagnosis", page_class, diagnosis_facts(dx))
        problems += verify(dreq, _json(client.chat(extract_messages(dreq, res.text), temperature=0.0, max_tokens=600)))
    return (res.text if not problems else None), problems


def build_chart(client: ChatClient, enc: SyntheaEncounter, pid: str) -> ChartResult:
    """One chart pair (base and twin) and its items. Narrative pages are written once and shared by both charts."""
    rng = rng_for("chart_v1", pid)
    ctx = make_context(rng)
    # an indicator the encounter's own pages do not already support, so the planted page decides the label
    own = [page_facts(enc, p) for p in renderable_pages(enc) if p != "emergency_department"]
    keys = [k for k in sorted(CASES) if k not in HELD_OUT_INDICATORS
            and not any(criteria.supports(k, f) for f in own)]
    if not keys:
        return ChartResult([], ("every indicator already supported by the encounter's pages",))
    key = rng.choice(keys)
    case = CASES[key]
    trap = rng.choice(case.traps)
    seed = rng.randrange(10**9)
    sup = _indicator_facts(key, case, random.Random(seed), True, None)
    non = (_indicator_facts(key, case, random.Random(seed), False, None) if trap in ("value_cross", "drug_swap")
           else _indicator_facts(key, case, random.Random(seed), True, trap))

    code_pages: list[ChartPage] = []
    for page in renderable_pages(enc):
        if page == "emergency_department":
            continue
        prng = rng_for("chart_v1", pid, page)
        code_pages.append(ChartPage(page, render_page(enc, page, prng, ctx), page_facts(enc, page), "code"))

    dx = _final_dx(enc, rng)
    plan = ["admission"] + ["progress_note"] * rng.randint(1, 3)
    if rng.random() < 0.3:
        plan.append("consult_note")
    if rng.random() < 0.15:
        plan.append("nutrition")
    narrative: list[ChartPage] = []
    for cls in plan:
        text, problems = _narrative(client, enc, cls, None, rng, ctx)
        if text is None:
            return ChartResult([], tuple(f"{cls}: {p}" for p in problems))
        narrative.append(ChartPage(cls, text, None, "teacher"))
    ds_text, problems = _narrative(client, enc, "discharge_summary", dx, rng, ctx)
    if ds_text is None:
        return ChartResult([], tuple(f"discharge_summary: {p}" for p in problems))
    discharge = ChartPage("discharge_summary", ds_text, None, "teacher")

    ev_class = {"lab": "laboratory", "flowsheet": "flowsheet", "mar": "mar"}[case.page]
    ev_seed = rng.randrange(10**9)
    middle = narrative[1:] + code_pages
    rng.shuffle(middle)
    pos = rng.randint(0, len(middle))

    def chart(evidence: ChunkFacts) -> tuple[str, list[ChartPage]]:
        ev = ChartPage(ev_class, _render_evidence(case.page, evidence, random.Random(ev_seed), ctx), evidence, "code")
        pages = [narrative[0], *middle[:pos], ev, *middle[pos:], discharge]
        n = len(pages)
        text = "\n\n".join(f"=== Page {i} of {n} ===\n{p.text}" for i, p in enumerate(pages, 1))
        return text, pages

    def support(pages: list[ChartPage]) -> list[int]:
        return [i for i, p in enumerate(pages, 1) if p.facts is not None and criteria.supports(key, p.facts)]

    def union(pages: list[ChartPage]) -> ChunkFacts:
        obs = tuple(o for p in pages if p.facts for o in p.facts.observations)
        meds = tuple(m for p in pages if p.facts for m in p.facts.meds)
        return ChunkFacts(obs, meds)

    items: list[GeneratedItem] = []
    reasons: list[str] = []
    charts = [("base", *chart(sup)), ("twin", *chart(non))]
    if rng.random() < 0.5:
        charts = [("base", charts[1][1], charts[1][2]), ("twin", charts[0][1], charts[0][2])]
    golds = {role: criteria.supports(key, union(pages)) for role, _, pages in charts}
    ind = criteria.INDICATOR_BY_KEY[key]
    meta = {"indicator": key, "trap": trap, "evidence_page_class": ev_class, "source": "synthea-chart",
            "pages": len(charts[0][2])}
    if golds["base"] != golds["twin"]:
        for role, text, _ in charts:
            items.append(make_item("chart_indicator_evidence_v1", "MR_chart_audit", pid, role, "noul", text,
                                   f"Does this medical record document {ind.description} during the current admission?",
                                   [("yes", "Yes"), ("no", "No")], "yes" if golds[role] else "no", dict(meta)))
    else:
        reasons.append("indicator twin did not flip (another page supports it)")
    for role, text, pages in charts:
        hits = support(pages)
        if len(hits) == 1 and len(pages) + 1 <= len(LETTERS):
            options = [(LETTERS[i - 1], f"Page {i}") for i in range(1, len(pages) + 1)]
            options.append((LETTERS[len(pages)], "No page documents it"))
            items.append(make_item("chart_evidence_page_v1", "MR_chart_audit", pid, role, "choice", text,
                                   f"Which page documents {ind.description} during the current admission?",
                                   options, LETTERS[hits[0] - 1], {**meta, "page": hits[0]}))
    fam = icd.FAMILY_BY_KEY[dx.family]
    codes = sorted(fam.codes)
    options = [(LETTERS[i], f"{c} - {fam.codes[c]}") for i, c in enumerate(codes)]
    none_key = LETTERS[len(codes)]
    options.append((none_key, "None of these codes is supported by the documentation"))
    gold_code = icd.supported_code(dx, "inpatient")
    items.append(make_item("chart_icd_support_v1", "MR_chart_audit", pid, "base", "choice", charts[0][1],
                           "Which ICD-10-CM code does this inpatient record support for the final diagnosis, as the most "
                           "specific code?", options, none_key if gold_code is None else LETTERS[codes.index(gold_code)],
                           {"family": dx.family, "code": gold_code, "certainty": dx.certainty, "source": "synthea-chart"}))
    return ChartResult(items, tuple(reasons))


def chart_items_for(client: ChatClient, encounters: list[SyntheaEncounter], pid_prefix: str = "chart"
                    ) -> tuple[list[GeneratedItem], dict[str, int]]:
    """Convenience loop for tests and small runs; the build script parallelises ``build_chart`` itself."""
    out: list[GeneratedItem] = []
    counts: dict[str, int] = {}
    for enc in encounters:
        res = build_chart(client, enc, f"{pid_prefix}-{enc.patient_id}:{enc.encounter_id}")
        out += res.items
        for r in res.reasons or (("kept",) if res.items else ()):
            counts[r.split(":")[0]] = counts.get(r.split(":")[0], 0) + 1
    return out, counts


