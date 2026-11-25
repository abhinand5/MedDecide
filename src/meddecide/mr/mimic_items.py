"""Structured-gold MIMIC and MedNLI templates (osler_v4 Y2; ADVISORY Y2; D27, D46). Pure builders.

Every gold label comes from a database field or from code: a coded family, a DRG title, an APR level, an ED triage field,
a disposition field, or a dataset label. No model writes a label. Each builder returns the rows kept, the held-out rows,
and a count for every drop reason and every shortage (no silent drops). Rows use ``meddecide.mix.rows.make_row``.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from meddecide.mix.rows import make_row
from meddecide.mr import icd, mimic
from meddecide.mr.generators import HELD_OUT_ICD_FAMILIES
from meddecide.utils.hashing import stable_hash

LICENCE = "PhysioNet Credentialed Health Data License 1.5.0"
URL_NOTE = "https://physionet.org/content/mimic-iv-note/2.2/"
URL_HOSP = "https://physionet.org/content/mimiciv/3.1/"
URL_ED = "https://physionet.org/content/mimic-iv-ed/2.2/"
URL_MEDNLI = "https://physionet.org/content/mednli/1.0.0/"
HELD_OUT_FAMILIES = HELD_OUT_ICD_FAMILIES | {"aki"}          # heart failure and AKI (ADVISORY §3.4)
PROMPT_CAP = 8192
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
YES_NO = (("yes", "Yes"), ("no", "No"))
# how each curated family is named in a note (regex, case-insensitive); used to pick hard negatives
ICD_CONDITION = {
    "respiratory_failure": r"respiratory failure",
    "ckd": r"chronic kidney disease|\bCKD\b",
    "type2_diabetes": r"type 2 diabetes|type ii diabetes|diabetes mellitus",
    "sepsis_organism": r"\bsepsis\b|\bseptic\b",
    "severe_sepsis": r"severe sepsis|sepsis with (acute )?organ",
    "heart_failure": r"heart failure|\bCHF\b",
    "malnutrition": r"malnutrition",
    "pneumonia_organism": r"pneumonia",
    "knee_oa": r"osteoarthritis of the knee|knee osteoarthritis|knee oa",
    "hyponatremia": r"hypo-?natr[ae]mia",
    "metabolic_encephalopathy": r"encephalopathy",
    "aki": r"acute kidney|\bAKI\b|kidney injury",
}
DRG_MARKERS = (
    (re.compile(r"\s+(?:WITH|W) MCC$"), "MCC"),
    (re.compile(r"\s+(?:WITH|W) CC$"), "CC"),
    (re.compile(r"\s+(?:WITHOUT|W/O) CC/MCC$"), "NONE"),
)
STOP_TITLE = frozenset({"WITH", "WITHOUT", "AND", "OR", "OF", "THE", "W", "W/O", "CC", "MCC", "IN", "TO", "FOR", "ON"})
ED_DISPOSITION = {"ADMITTED": "admitted", "HOME": "home"}


@dataclass
class Built:
    """One template's output for one split: kept rows, held-out rows, drop reasons, and shortages."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    heldout: list[dict[str, Any]] = field(default_factory=list)
    drops: Counter[str] = field(default_factory=Counter)
    shortage: Counter[str] = field(default_factory=Counter)
    truncated: int = 0


@dataclass
class Ctx:
    """Read-only inputs. ``notes`` maps note_id to the discharge text; ``count`` is the pplx token counter."""

    notes: Mapping[str, str]
    icd_titles: Mapping[str, str]        # undotted ICD-10-CM code -> long title
    pcs_titles: Mapping[str, str]        # undotted ICD-10-PCS code -> long title
    count: Callable[[str], int]
    seed: int = 0
    token_of: Mapping[str, int] = field(default_factory=dict)   # note_id -> pplx token count of the note (Y1)

    def note_state(self, note_id: str, drop_sections: Sequence[str] = ()) -> tuple[str, bool]:
        """The discharge note as a prompt state: whole when it fits the cap, else the section-priority cut (ADVISORY §3.8)."""
        text = self.notes[note_id].replace("\r\n", "\n").replace("\r", "\n")
        if not drop_sections and self.token_of.get(note_id, PROMPT_CAP + 1) <= PROMPT_CAP:
            return text, False                  # the stored token count says the whole note fits: no tokenising
        if drop_sections:
            kept = {k: v for k, v in mimic.sections(text).items() if k not in drop_sections}
            text = "\n\n".join(v if k == "preamble" else f"{mimic.SECTION_TITLES.get(k, k.replace('_', ' ').capitalize())}:"
                               f"\n{v}" for k, v in kept.items())
        if self.count(text) <= PROMPT_CAP:
            return text, False
        return mimic.prioritised_text(mimic.sections(text), PROMPT_CAP, self.count)


def dotted(code: str) -> str:
    return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"


def stable_order(items: Iterable[Any], seed: int, tag: str, key: Callable[[Any], Any]) -> list[Any]:
    """A deterministic shuffle: sorted by a stable hash of (seed, template tag, item key)."""
    rows = list(items)
    rows.sort(key=lambda r: stable_hash({"seed": seed, "tag": tag, "key": str(key(r))}, length=16))
    return rows


def mimic_row(template: str, skill: str, qtype: str, state: str, question: str, options: Sequence[tuple[str, str]],
              gold: str, record_id: str, split: str, url: str, meta: Mapping[str, Any]) -> dict[str, Any]:
    return make_row(source="mimic_iv", template_id=template, skill=skill, qtype=qtype, state=state, question=question,
                    options=list(options), gold=gold, record_id=record_id, record_url=url, licence=LICENCE,
                    record_date="", split=split, tier="mimic", meta=dict(meta))


def admission_in_test_scope(split: str, late: bool) -> bool:
    """The test split keeps only late-anchor patients (ADVISORY Y2 shared rules); train and dev keep every patient."""
    return split != "test" or bool(late)


# ---- ICD-10-CM curated families ----------------------------------------------------------------------

def family_codes(dx_rows: Iterable[tuple[int, str]]) -> dict[int, dict[str, list[str]]]:
    """hadm_id -> family key -> the admission's coded dotted codes of that curated family (ICD-10 rows)."""
    out: dict[int, dict[str, list[str]]] = {}
    for hadm, code in dx_rows:
        dotted_code = mimic.UNDOTTED_TO_CODE.get(str(code))
        if dotted_code is None:
            continue
        out.setdefault(int(hadm), {}).setdefault(mimic.CODE_TO_FAMILY[dotted_code], []).append(dotted_code)
    return out


def icd_support(ctx: Ctx, split: str, quota: int, admissions: Sequence[tuple[int, str, bool]],
                fam_of: Mapping[int, Mapping[str, list[str]]]) -> Built:
    """mimic_icd_support_v1 (choice). For each family: positives are admissions whose coder assigned exactly one code of the
    family; gold is that code. Negatives (one per positive) have no code of the family; half are hard (the condition's name
    occurs in the note). Gold "none" for a negative. ``admissions`` are (hadm_id, note_id, late) in pool order."""
    out = Built()
    families = list(icd.FAMILIES)
    per_family = max(1, quota // len(families) // 2)
    for fam in families:
        codes = sorted(fam.codes)
        options = [(LETTERS[i], f"{c} - {fam.codes[c]}") for i, c in enumerate(codes)]
        none_key = LETTERS[len(codes)]
        options.append((none_key, "None of these codes is supported by the documentation"))
        held = fam.key in HELD_OUT_FAMILIES
        if held and split != "test":
            out.drops[f"held-out family {fam.key} outside the test split"] += 1
            continue
        pattern = re.compile(ICD_CONDITION[fam.key], re.IGNORECASE)
        positives: list[tuple[int, str, str]] = []
        hard: list[tuple[int, str, str]] = []
        easy: list[tuple[int, str, str]] = []
        for hadm, note_id, late in admissions:
            if not admission_in_test_scope(split, late):
                continue
            coded = fam_of.get(hadm, {}).get(fam.key, [])
            if len(set(coded)) > 1:
                out.drops[f"several codes of family {fam.key}"] += 1
                continue
            if coded:
                if len(positives) < per_family:
                    positives.append((hadm, note_id, coded[0]))
                continue
            if len(hard) < per_family // 2 + per_family % 2 or len(easy) < per_family // 2:
                text = ctx.notes[note_id]
                if pattern.search(text):
                    if len(hard) < per_family // 2 + per_family % 2:
                        hard.append((hadm, note_id, ""))
                elif len(easy) < per_family // 2:
                    easy.append((hadm, note_id, ""))
            if len(positives) >= per_family and len(hard) + len(easy) >= per_family:
                break
        negatives = (hard + easy)[: len(positives)]
        if len(negatives) < len(positives):
            out.shortage[f"icd_support {fam.key}: negatives"] += len(positives) - len(negatives)
        for hadm, note_id, code in positives:
            state, cut = ctx.note_state(note_id)
            out.truncated += int(cut)
            gold = next(k for k, lab in options if lab.startswith(code + " "))
            row = mimic_row("mimic_icd_support_v1", "mimic_icd_support", "choice", state,
                            "Which ICD-10-CM code does this documentation support, as the most specific code?", options,
                            gold, f"hadm:{hadm}:{fam.key}:pos", split, URL_NOTE,
                            {"family": fam.key, "role": "positive", "code": code, "cut": cut})
            (out.heldout if held else out.rows).append(row)
        for hadm, note_id, _ in negatives:
            state, cut = ctx.note_state(note_id)
            out.truncated += int(cut)
            row = mimic_row("mimic_icd_support_v1", "mimic_icd_support", "choice", state,
                            "Which ICD-10-CM code does this documentation support, as the most specific code?", options,
                            none_key, f"hadm:{hadm}:{fam.key}:neg", split, URL_NOTE,
                            {"family": fam.key, "role": "negative", "code": None, "cut": cut,
                             "hard": bool(re.search(ICD_CONDITION[fam.key], ctx.notes[note_id], re.I))})
            (out.heldout if held else out.rows).append(row)
    return out


def icd_specificity(ctx: Ctx, split: str, quota: int, admissions: Sequence[tuple[int, str, bool, list[str]]],
                    billable_by_prefix: Mapping[str, list[str]]) -> Built:
    """mimic_icd_specificity_v1 (choice, 3 to 6 options). Each admission contributes one 3-character category in which it
    coded exactly one code; options are that code plus billable siblings of the category (from the ICD-10-CM table)."""
    out = Built()
    for hadm, note_id, late, codes in admissions:
        if len(out.rows) + len(out.heldout) >= quota:
            break
        if not admission_in_test_scope(split, late):
            continue
        by_cat: dict[str, list[str]] = {}
        for code in codes:
            by_cat.setdefault(code[:3], []).append(code)
        singles = [cat for cat, members in by_cat.items() if len(set(members)) == 1]
        if not singles:
            out.drops["no category with exactly one coded code"] += 1
            continue
        cat = stable_order(singles, ctx.seed, f"icd_specificity:{hadm}", lambda c: c)[0]
        code = by_cat[cat][0]
        siblings = [c for c in billable_by_prefix.get(cat, []) if c != code]
        if len(siblings) < 2:
            out.drops["fewer than two billable siblings"] += 1
            continue
        picks = stable_order(siblings, ctx.seed, f"icd_specificity_pick:{hadm}", lambda c: c)[:5]
        candidates = [code, *picks]
        candidates = stable_order(candidates, ctx.seed, f"icd_specificity_order:{hadm}", lambda c: c)
        options = [(LETTERS[i], f"{dotted(c)} - {ctx.icd_titles[c]}") for i, c in enumerate(candidates)]
        gold = LETTERS[candidates.index(code)]
        state, cut = ctx.note_state(note_id)
        out.truncated += int(cut)
        family = mimic.CODE_TO_FAMILY.get(dotted(code))
        held = family in HELD_OUT_FAMILIES or code[:3] in ("I50", "N17")
        row = mimic_row("mimic_icd_specificity_v1", "mimic_icd_specificity", "choice", state,
                        f"Which billable ICD-10-CM code in category {dotted(cat)} did the coder assign to this admission?",
                        options, gold, f"hadm:{hadm}:{cat}", split, URL_NOTE,
                        {"category": cat, "code": dotted(code), "n_options": len(options), "cut": cut})
        if held and split != "test":
            out.drops["held-out target outside the test split"] += 1
            continue
        (out.heldout if held else out.rows).append(row)
        if len(out.rows) >= quota:
            break
    return out


def principal_dx(ctx: Ctx, split: str, quota: int, admissions: Sequence[tuple[int, str, bool, str, list[str]]]) -> Built:
    """mimic_principal_dx_v1 (choice, 4 to 6 options). Gold is the code of seq_num 1; the note has its Discharge Diagnosis
    section removed. Options are the principal plus three to five secondary codes, the same chapter first."""
    out = Built()
    for hadm, note_id, late, principal, codes in admissions:
        if len(out.rows) >= quota:
            break
        if not admission_in_test_scope(split, late):
            continue
        secondary = [c for c in dict.fromkeys(codes) if c != principal and c in ctx.icd_titles]
        if len(secondary) < 3:
            out.drops["fewer than three secondary codes"] += 1
            continue
        same = [c for c in secondary if c[:1] == principal[:1]]
        other = [c for c in secondary if c[:1] != principal[:1]]
        picks = (stable_order(same, ctx.seed, f"principal:{hadm}", lambda c: c)
                 + stable_order(other, ctx.seed, f"principal_o:{hadm}", lambda c: c))[:5]
        candidates = stable_order([principal, *picks], ctx.seed, f"principal_order:{hadm}", lambda c: c)
        options = [(LETTERS[i], f"{dotted(c)} - {ctx.icd_titles[c]}") for i, c in enumerate(candidates)]
        gold = LETTERS[candidates.index(principal)]
        state, cut = ctx.note_state(note_id, drop_sections=("discharge_diagnosis",))
        out.truncated += int(cut)
        held = mimic.CODE_TO_FAMILY.get(dotted(principal)) in HELD_OUT_FAMILIES or principal[:3] in ("I50", "N17")
        if held and split != "test":
            out.drops["held-out principal code outside the test split"] += 1
            continue
        row = mimic_row("mimic_principal_dx_v1", "mimic_principal_dx", "choice", state,
                        "Which of this admission's coded diagnoses was sequenced first as the principal diagnosis?",
                        options, gold, f"hadm:{hadm}:principal", split, URL_NOTE,
                        {"principal": dotted(principal), "n_options": len(options),
                         "same_chapter_options": sum(1 for c in candidates if c[:1] == principal[:1]), "cut": cut})
        (out.heldout if held else out.rows).append(row)
    return out


# ---- MS-DRG and APR-DRG (CMS titles are public domain; APR descriptions are never read) ----------------

def drg_split(title: str) -> tuple[str, str | None]:
    """(base title, severity level) for a MS-DRG title. Level "MCC", "CC" or "NONE" when the title carries a three-way split
    marker; None for a title without one (a two-way 'CC/MCC' split is left out, as ADVISORY Y2 requires)."""
    upper = " ".join(title.upper().split())
    for pattern, level in DRG_MARKERS:
        match = pattern.search(upper)
        if match:
            return upper[: match.start()].strip(), level
    return upper, None


def _title_tokens(title: str) -> set[str]:
    return {w for w in re.findall(r"[A-Z0-9]+", title.upper()) if w not in STOP_TITLE and len(w) > 1}


def msdrg(ctx: Ctx, split: str, quota: int, admissions: Sequence[tuple[int, str, bool, str, str]],
          drg_titles: Mapping[str, str]) -> Built:
    """mimic_msdrg_v1 (choice, 4 to 6 options). Gold is the admission's MS-DRG (HCFA row). Options: the severity siblings
    (same base title, other markers) and the most similar DRG titles in the data (MDC is not in the data: approximated by
    title tokens, a recorded deviation)."""
    out = Built()
    bases: dict[str, list[str]] = {}
    for code, title in drg_titles.items():
        bases.setdefault(drg_split(title)[0], []).append(code)
    ranked: dict[str, list[str]] = {}    # per DRG: the other DRGs by title similarity (computed once per code)

    def similar_codes(code: str) -> list[str]:
        if code not in ranked:
            tokens = _title_tokens(drg_titles[code])
            scored = []
            for other, other_title in drg_titles.items():
                if other == code:
                    continue
                other_tokens = _title_tokens(other_title)
                union = other_tokens | tokens
                score = len(other_tokens & tokens) / len(union) if union else 0.0
                if score > 0:
                    scored.append((-score, other))
            ranked[code] = [c for _, c in sorted(scored)]
        return ranked[code]

    for hadm, note_id, late, code, title in admissions:
        if len(out.rows) >= quota:
            break
        if not admission_in_test_scope(split, late):
            continue
        base, _ = drg_split(title)
        siblings = [c for c in bases.get(base, []) if c != code][:2]
        distractors = [c for c in similar_codes(code) if c not in siblings][: max(0, 5 - len(siblings))]
        candidates = [code, *siblings, *distractors]
        if len(candidates) < 4:
            out.drops["fewer than four DRG options"] += 1
            continue
        candidates = candidates[:6]
        candidates = stable_order(candidates, ctx.seed, f"msdrg_order:{hadm}", lambda c: c)
        options = [(LETTERS[i], f"{c} - {drg_titles[c]}") for i, c in enumerate(candidates)]
        gold = LETTERS[candidates.index(code)]
        state, cut = ctx.note_state(note_id)
        out.truncated += int(cut)
        out.rows.append(mimic_row("mimic_msdrg_v1", "mimic_msdrg", "choice", state,
                                  "Which MS-DRG was assigned to this admission?", options, gold,
                                  f"hadm:{hadm}:msdrg", split, URL_HOSP,
                                  {"drg": code, "n_siblings": len(siblings), "n_options": len(options), "cut": cut}))
    return out


def drg_cc_level(ctx: Ctx, split: str, quota: int, admissions: Sequence[tuple[int, str, bool, str, str]]) -> Built:
    """mimic_drg_cc_level_v1 (choice: MCC, CC, neither). The label is parsed by code from the MS-DRG title; only titles with a
    three-way marker are used. Classes are balanced to the quota."""
    out = Built()
    keys = [("A", "MCC"), ("B", "CC"), ("C", "neither (without CC/MCC)")]
    per_class = max(1, quota // 3)
    taken: Counter[str] = Counter()
    gold_of = {"MCC": "A", "CC": "B", "NONE": "C"}
    for hadm, note_id, late, code, title in admissions:
        if not admission_in_test_scope(split, late):
            continue
        _, level = drg_split(title)
        if level is None:
            out.drops["title without a three-way marker"] += 1
            continue
        if taken[level] >= per_class:
            continue
        taken[level] += 1
        state, cut = ctx.note_state(note_id)
        out.truncated += int(cut)
        out.rows.append(mimic_row("mimic_drg_cc_level_v1", "mimic_drg_cc_level", "choice", state,
                                  "Which severity level (MCC, CC or neither) is in this admission's MS-DRG title?",
                                  keys, gold_of[level], f"hadm:{hadm}:cc", split, URL_HOSP,
                                  {"drg": code, "level": level, "cut": cut}))
        if all(taken[k] >= per_class for k in ("MCC", "CC", "NONE")):
            break
    for level, n in (("MCC", taken["MCC"]), ("CC", taken["CC"]), ("NONE", taken["NONE"])):
        if n < per_class:
            out.shortage[f"drg_cc_level {level}"] += per_class - n
    return out


def apr_level(ctx: Ctx, split: str, quota: int, admissions: Sequence[tuple[int, str, bool, str, int]], skill: str,
              template: str) -> Built:
    """mimic_apr_severity_v1 and mimic_apr_mortality_v1 (score 1 to 4, number only). The label is the APR-DRG level field."""
    out = Built()
    keys = [(str(i), str(i)) for i in range(1, 5)]
    for hadm, note_id, late, _, level in admissions:
        if len(out.rows) >= quota:
            break
        if not admission_in_test_scope(split, late):
            continue
        state, cut = ctx.note_state(note_id)
        out.truncated += int(cut)
        question = ("What is the APR-DRG severity of illness level for this admission (1 to 4)?" if skill.endswith("severity")
                    else "What is the APR-DRG risk of mortality level for this admission (1 to 4)?")
        out.rows.append(mimic_row(template, skill, "score", state, question, keys, str(level), f"hadm:{hadm}:{skill}",
                                  split, URL_HOSP, {"level": level, "cut": cut}))
    return out


# ---- ICD-10-PCS (noul) -------------------------------------------------------------------------------

def pcs_support(ctx: Ctx, split: str, quota: int, admissions: Sequence[tuple[int, str, bool, str, list[str]]],
                index: Mapping[tuple[str, str], list[str]]) -> Built:
    """mimic_pcs_support_v1 (noul). Positive: one PCS code the coder assigned. Negative: a PCS code of the same body system
    (characters 1 and 2) coded in another admission with the same principal-diagnosis category, and not coded here."""
    out = Built()
    for hadm, note_id, late, principal, codes in admissions:
        if len(out.rows) >= quota:
            break
        if not admission_in_test_scope(split, late):
            continue
        if not codes:
            out.drops["no PCS code"] += 1
            continue
        code = stable_order(codes, ctx.seed, f"pcs_pos:{hadm}", lambda c: c)[0]
        category = principal[:3]
        negatives = [c for c in index.get((category, code[:2]), []) if c not in codes]
        if not negatives:
            out.drops["no same-body-system negative in the same category"] += 1
            continue
        neg = stable_order(negatives, ctx.seed, f"pcs_neg:{hadm}", lambda c: c)[0]
        state, cut = ctx.note_state(note_id)
        out.truncated += int(cut)
        for role, c, gold in (("positive", code, "yes"), ("negative", neg, "no")):
            out.rows.append(mimic_row(
                "mimic_pcs_support_v1", "mimic_pcs_support", "noul", state,
                f"Was ICD-10-PCS procedure {c} ({ctx.pcs_titles.get(c, '')}) coded for this admission?", YES_NO, gold,
                f"hadm:{hadm}:pcs:{role}", split, URL_HOSP, {"role": role, "code": c, "cut": cut}))
    return out


def pcs_index(admissions: Iterable[tuple[int, str, bool, str, list[str]]], split: str) -> dict[tuple[str, str], list[str]]:
    """(principal category, body system) -> the PCS codes coded in admissions of that category (same split)."""
    index: dict[tuple[str, str], set[str]] = {}
    for _, _, late, principal, codes in admissions:
        if not admission_in_test_scope(split, late):
            continue
        for code in codes:
            index.setdefault((principal[:3], code[:2]), set()).add(code)
    return {k: sorted(v) for k, v in index.items()}


# ---- radiology exam (choice; the exam-name header is masked by code) ----------------------------------

MODALITY_RULES = (
    ("CT", re.compile(r"\bCT\b|\bCTA\b|COMPUTED TOMOGRAPHY", re.I)),
    ("MR", re.compile(r"\bMRI?\b|\bMRA\b|MAGNETIC RESONANCE", re.I)),
    ("US", re.compile(r"ULTRASOUND|\bUS\b|SONOG|DOPPLER", re.I)),
    ("NM", re.compile(r"\bPET\b|NUCLEAR|SCINT", re.I)),
    ("MAM", re.compile(r"MAMMO", re.I)),
    ("XR", re.compile(r"\bX-?RAY\b|RADIOGRAPH|\bXR\b|\bCHEST\b|\bPA\b|\bAP\b|\bLATERAL\b|\bKUB\b|\bFLUORO", re.I)),
)


def modality(exam: str) -> str | None:
    """The modality class of an exam name, or None when no rule matches (such exams are dropped)."""
    for name, pattern in MODALITY_RULES:
        if pattern.search(exam):
            return name
    return None


def rad_exam(ctx: Ctx, split: str, quota: int, reports: Sequence[tuple[str, int, str, str, bool]],
             names_by_modality: Mapping[str, list[str]]) -> Built:
    """mimic_rad_exam_v1 (choice, 4 to 6 options). Input rows are (note_id, hadm, text, exam_name, late). The exam name is
    masked in the report; options are exams of the same modality from the data."""
    out = Built()
    for note_id, _hadm, text, exam, late in reports:
        if len(out.rows) >= quota:
            break
        if not admission_in_test_scope(split, late):
            continue
        mod = modality(exam)
        if mod is None:
            out.drops["exam modality not recognised"] += 1
            continue
        pool = [n for n in names_by_modality.get(mod, []) if n.upper() != exam.upper()]
        if len(pool) < 3:
            out.drops["fewer than three same-modality exams"] += 1
            continue
        masked, hits = re.subn(re.escape(exam), "[exam name masked]", text, flags=re.IGNORECASE)
        if hits == 0:
            out.drops["exam name not in the report text"] += 1
            continue
        distractors = stable_order(pool, ctx.seed, f"rad:{note_id}", lambda n: n)[:4]
        candidates = stable_order([exam, *distractors], ctx.seed, f"rad_order:{note_id}", lambda n: n)
        options = [(LETTERS[i], c) for i, c in enumerate(candidates)]
        gold = LETTERS[candidates.index(exam)]
        cut = False
        if ctx.count(masked) > PROMPT_CAP:
            masked, cut = mimic.prioritised_text(mimic.sections(masked), PROMPT_CAP, ctx.count)
            out.truncated += 1
        out.rows.append(mimic_row("mimic_rad_exam_v1", "mimic_rad_exam", "choice", masked,
                                  "Which examination was performed for this report?", options, gold,
                                  f"note:{note_id}", split, URL_NOTE, {"modality": mod, "n_options": len(options), "cut": cut}))
    return out


# ---- ED triage (MIMIC-IV-ED 2.2) ---------------------------------------------------------------------

def _num(value: Any, unit: str) -> str:
    """A triage value as written: numbers rounded to one decimal; text (MIMIC-IV-ED pain scores can be text) as it is."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == "":
        return "not recorded"
    try:
        shown = f"{round(float(value), 1):g}"
    except (TypeError, ValueError):
        shown = str(value).strip()
    return f"{shown} {unit}".strip()


def triage_page(row: Mapping[str, Any]) -> str:
    """The triage record rendered by code from the triage fields (no model text)."""
    chief = str(row.get("chiefcomplaint") or "").strip() or "not recorded"
    return (f"Emergency department triage record.\nChief complaint: {chief}.\n"
            f"Temperature: {_num(row.get('temperature'), 'F')}. Heart rate: {_num(row.get('heartrate'), 'bpm')}. "
            f"Respiratory rate: {_num(row.get('resprate'), '/min')}. Oxygen saturation: {_num(row.get('o2sat'), '%')}. "
            f"Blood pressure: {_num(row.get('sbp'), 'mmHg')} systolic, {_num(row.get('dbp'), 'mmHg')} diastolic. "
            f"Pain score: {_num(row.get('pain'), '')}.")


def ed_acuity(ctx: Ctx, split: str, quota: int, triage: Sequence[Mapping[str, Any]], late_of: Mapping[int, bool]) -> Built:
    """mimic_ed_acuity_v1 (score 1 to 5). Gold is the triage acuity field. ``triage`` rows carry subject_id and stay_id."""
    out = Built()
    keys = [(str(i), str(i)) for i in range(1, 6)]
    for row in triage:
        if len(out.rows) >= quota:
            break
        if not admission_in_test_scope(split, late_of.get(int(row["subject_id"]), False)):
            continue
        acuity = row.get("acuity")
        if acuity is None or (isinstance(acuity, float) and math.isnan(acuity)) or not 1 <= float(acuity) <= 5:
            out.drops["acuity missing or outside 1-5"] += 1
            continue
        out.rows.append(mimic_row("mimic_ed_acuity_v1", "mimic_ed_acuity", "score", triage_page(row),
                                  "What is the triage acuity level for this patient (1 = most urgent, 5 = least urgent)?",
                                  keys, str(int(float(acuity))), f"stay:{int(row['stay_id'])}:acuity", split, URL_ED,
                                  {"acuity": int(float(acuity))}))
    return out


def ed_disposition(ctx: Ctx, split: str, quota: int, stays: Sequence[Mapping[str, Any]], late_of: Mapping[int, bool],
                   diagnoses: Mapping[int, list[tuple[str, int, str]]]) -> Built:
    """mimic_ed_disposition_v1 (choice: admitted, home, other). Gold is the edstays disposition field, grouped by code.
    The state is the triage record plus the ED diagnoses (code, version and title as recorded in the ED tables)."""
    out = Built()
    keys = [("A", "admitted"), ("B", "home"), ("C", "other")]
    gold_of = {"admitted": "A", "home": "B", "other": "C"}
    for stay in stays:
        if len(out.rows) >= quota:
            break
        if not admission_in_test_scope(split, late_of.get(int(stay["subject_id"]), False)):
            continue
        disposition = str(stay.get("disposition") or "").strip().upper()
        if not disposition:
            out.drops["disposition missing"] += 1
            continue
        group = ED_DISPOSITION.get(disposition, "other")
        dx_lines = [f"{code} (ICD-10 version {version}): {title}" for code, version, title in diagnoses.get(int(stay["stay_id"]), [])]
        state = triage_page(stay) + "\n\nEmergency department diagnoses:\n" + ("\n".join(dx_lines) or "none recorded")
        out.rows.append(mimic_row("mimic_ed_disposition_v1", "mimic_ed_disposition", "choice", state,
                                  "What was the disposition of this emergency department visit?", keys, gold_of[group],
                                  f"stay:{int(stay['stay_id'])}:disposition", split, URL_ED, {"group": group}))
    return out


# ---- MedNLI (dataset labels; the test split is an additional row, not in training) ---------------------

MEDNLI_KEYS = (("A", "entailment"), ("B", "contradiction"), ("C", "neutral"))


def mednli(split: str, pairs: Iterable[Mapping[str, Any]]) -> Built:
    """mednli_v1 (choice). The premise is the state; the hypothesis is in the question; gold is the file's gold_label."""
    out = Built()
    gold_of = {label: key for key, label in MEDNLI_KEYS}
    for pair in pairs:
        label = str(pair.get("gold_label", ""))
        if label not in gold_of:
            out.drops[f"gold_label not in the three classes: {label!r}"] += 1
            continue
        state = str(pair["sentence1"]).strip()
        question = f"Hypothesis: {str(pair['sentence2']).strip()} Given the premise, is the hypothesis an entailment, a contradiction, or neutral?"
        out.rows.append(mimic_row("mednli_v1", "mednli", "choice", state, question, MEDNLI_KEYS, gold_of[label],
                                  f"pair:{pair['pairID']}", split, URL_MEDNLI, {"label": label}))
    return out


# ---- laboratory validation (noul; the page is rendered by code from labevents) -------------------------

# analyte -> (accepted MIMIC dictionary labels, the unit of the criterion, the units accepted for it)
ANALYTES: dict[str, tuple[tuple[str, ...], str, frozenset[str]]] = {
    "sodium": (("Sodium", "Sodium, Whole Blood"), "mmol/L", frozenset({"mmol/L", "mEq/L"})),
    "potassium": (("Potassium", "Potassium, Whole Blood"), "mmol/L", frozenset({"mmol/L", "mEq/L"})),
    "hemoglobin": (("Hemoglobin",), "g/dL", frozenset({"g/dL"})),
    "platelets": (("Platelet Count",), "K/uL", frozenset({"K/uL", "K/µL", "10^3/uL"})),
    "wbc": (("White Blood Cells",), "K/uL", frozenset({"K/uL", "K/µL", "10^3/uL"})),
    "glucose": (("Glucose", "Glucose, Whole Blood"), "mg/dL", frozenset({"mg/dL"})),
    "magnesium": (("Magnesium",), "mg/dL", frozenset({"mg/dL"})),
    "lactate": (("Lactate",), "mmol/L", frozenset({"mmol/L"})),
}
# condition -> (analyte, predicate(value, sex) -> criterion met, criterion text, ICD-10-CM prefixes (undotted) of the coded
# condition). The criteria are the ADVISORY Y2 list; AKI is excluded (ADVISORY §3.4).
LAB_CONDITIONS: tuple[tuple[str, str, Callable[[float, str], bool], str, tuple[str, ...]], ...] = (
    ("hyponatremia", "sodium", lambda v, s: v < 135, "sodium below 135 mmol/L", ("E871",)),
    ("hypokalemia", "potassium", lambda v, s: v < 3.5, "potassium below 3.5 mmol/L", ("E876",)),
    ("hyperkalemia", "potassium", lambda v, s: v > 5.0, "potassium above 5.0 mmol/L", ("E875",)),
    ("anemia", "hemoglobin", lambda v, s: v < (13 if s == "M" else 12), "haemoglobin below 13 g/dL (male) or 12 g/dL (female)",
     ("D64",)),
    ("thrombocytopenia", "platelets", lambda v, s: v < 150, "platelet count below 150 K/µL", ("D696",)),
    ("leukocytosis", "wbc", lambda v, s: v > 11, "white cell count above 11 K/µL", ("D7282",)),
    ("hypoglycemia", "glucose", lambda v, s: v < 70, "glucose below 70 mg/dL", ("E162",)),
    ("hypomagnesemia", "magnesium", lambda v, s: v < 1.7, "magnesium below 1.7 mg/dL", ("E8342",)),
    ("lactic_acidosis", "lactate", lambda v, s: v > 2.0, "lactate above 2.0 mmol/L", ("E872",)),
)
LAB_PAGE_ROWS = 20


def lab_page(values: Sequence[tuple[str, str, float, str, float | None, float | None]]) -> tuple[str, int]:
    """The lab page for one admission and one analyte, rendered by code: values in time order (capped), with the dictionary
    name, the value, the unit and the reference range when the source has one. Returns (text, rows not shown)."""
    ordered = sorted(values, key=lambda v: v[0])
    shown = ordered[:LAB_PAGE_ROWS]
    lines = ["Laboratory results from this admission (in time order):"]
    for _, label, value, unit, lo, hi in shown:
        ref = f" (reference {lo:g}-{hi:g} {unit})" if lo is not None and hi is not None and not pd.isna(lo) and not pd.isna(hi) else ""
        lines.append(f"- {label}: {value:g} {unit}{ref}")
    return "\n".join(lines), max(0, len(ordered) - LAB_PAGE_ROWS)


def lab_validation(ctx: Ctx, split: str, quota: int, pools: Mapping[str, list[tuple[int, list[tuple]]]],
                   coded: Mapping[int, set[str]], sex: Mapping[int, str]) -> Built:
    """mimic_lab_validation_v1 (noul). For each condition: positives have at least one value meeting the criterion in the
    admission window, negatives have measured values none of which meets it. Half of each class is coded for the condition
    (ICD-10-CM prefix), half is not. ``pools[analyte]`` holds (hadm, [(time, label, value, unit, lo, hi), ...]) for the split."""
    out = Built()
    per_condition = max(2, quota // len(LAB_CONDITIONS))
    for name, analyte, predicate, criterion, prefixes in LAB_CONDITIONS:
        pos_coded: list[int] = []
        pos_plain: list[int] = []
        neg_coded: list[int] = []
        neg_plain: list[int] = []
        for hadm, values in stable_order(pools.get(analyte, []), ctx.seed, f"lab:{name}:{split}", lambda p: p[0]):
            s = sex.get(hadm, "")
            met = any(predicate(float(v[2]), s) for v in values)
            is_coded = any(c.startswith(p) for c in coded.get(hadm, set()) for p in prefixes)
            bucket = {(True, True): pos_coded, (True, False): pos_plain,
                      (False, True): neg_coded, (False, False): neg_plain}[(met, is_coded)]
            if len(bucket) < per_condition // 4:
                bucket.append(hadm)
        pools_by_hadm = dict(pools.get(analyte, []))
        for label_met, coded_list, plain_list in ((True, pos_coded, pos_plain), (False, neg_coded, neg_plain)):
            for want, members in ((True, coded_list), (False, plain_list)):
                if len(members) < per_condition // 4:
                    out.shortage[f"lab {name} met={label_met} coded={want}"] += per_condition // 4 - len(members)
                for hadm in members:
                    values = pools_by_hadm[hadm]
                    page, hidden = lab_page(values)
                    question = (f"Does any laboratory value in this admission meet the criterion for {name.replace('_', ' ')} "
                                f"({criterion})?")
                    out.truncated += int(hidden > 0)
                    out.rows.append(mimic_row("mimic_lab_validation_v1", "mimic_lab_validation", "noul", page, question,
                                              YES_NO, "yes" if label_met else "no", f"hadm:{hadm}:lab:{name}", split,
                                              URL_HOSP, {"condition": name, "analyte": analyte, "met": label_met,
                                                         "coded": want, "rows_not_shown": hidden}))
    return out


# ---- eMAR administration (noul) -----------------------------------------------------------------------

def med_given(ctx: Ctx, split: str, quota: int, admissions: Sequence[tuple[int, str, bool, str]],
              administered: Mapping[int, set[str]], ordered: Mapping[int, set[str]],
              category_drugs: Mapping[str, list[str]]) -> Built:
    """mimic_med_given_v1 (noul): was this drug administered during the stay (an eMAR "Administered" event)? For each
    admission with a discharge note: one positive (an administered drug) and one negative. The negative is a drug ordered for
    this admission and never administered; if there is none, a drug administered in another admission of the same principal
    category (not administered here)."""
    out = Built()
    for hadm, note_id, late, principal in admissions:
        if len(out.rows) >= quota:
            break
        if not admission_in_test_scope(split, late):
            continue
        given = administered.get(hadm, set())
        if not given:
            out.drops["no administered drug"] += 1
            continue
        pos = stable_order(sorted(given), ctx.seed, f"med_pos:{hadm}", lambda d: d)[0]
        never = sorted(ordered.get(hadm, set()) - given)
        if never:
            neg, source = stable_order(never, ctx.seed, f"med_neg:{hadm}", lambda d: d)[0], "ordered, never administered"
        else:
            others = [d for d in category_drugs.get(principal[:3], []) if d not in given]
            if not others:
                out.drops["no negative drug"] += 1
                continue
            neg, source = stable_order(others, ctx.seed, f"med_neg2:{hadm}", lambda d: d)[0], "administered in the same category"
        state, cut = ctx.note_state(note_id)
        out.truncated += int(cut)
        for role, drug, gold in (("positive", pos, "yes"), ("negative", neg, "no")):
            out.rows.append(mimic_row("mimic_med_given_v1", "mimic_med_given", "noul", state,
                                      f"Was {drug} administered to this patient during the hospital stay?", YES_NO, gold,
                                      f"hadm:{hadm}:med:{role}", split, URL_HOSP,
                                      {"role": role, "drug": drug, "negative_source": source if role == "negative" else None,
                                       "cut": cut}))
    return out


def dedupe_across_splits(rows_by_split: Mapping[str, list[dict[str, Any]]]) -> tuple[dict[str, list[dict[str, Any]]],
                                                                                     dict[str, int]]:
    """Leakage rule (ADVISORY Y2): an item in dev or test whose state text already occurs in an earlier split (train, then
    dev) is removed and counted. Nothing is removed within a split. Returns (kept rows by split, removed counts by split)."""
    seen: set[str] = set()
    kept: dict[str, list[dict[str, Any]]] = {}
    removed: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        rows = list(rows_by_split.get(split, []))
        keep: list[dict[str, Any]] = []
        digests: list[str] = []
        for row in rows:
            digest = hashlib.sha256(row["state"].encode("utf-8")).hexdigest()
            if split != "train" and digest in seen:
                continue
            keep.append(row)
            digests.append(digest)
        seen.update(digests)
        kept[split] = keep
        removed[split] = len(rows) - len(keep)
    return kept, removed
