"""Real clinical text with structured labels, no credential needed (osler_v1; operator decisions 2026-10-13).

- **MTSamples** (`harishnair04/mtsamples`; operator accepted the licence uncertainty): transcribed sample reports whose
  ``medical_specialty`` field is often a note type -> page category. One transcription can appear under several
  specialties; duplicates are merged and a conflict between mapped classes drops the text (counted).
- **CodiEsp** (`Biomedical-TeMU/CodiEsp_corpus`, CC BY 4.0): 1,000 clinical cases with expert ICD-10 diagnosis codes
  (CIE-10-ES = ICD-10-CM) and machine-translated English text -> ICD-10-CM code support. Codes absent from the
  FY order file are dropped (counted); a category with more than one coded code is ambiguous and dropped.
- **PMC-Patients** (`zhengyun21/PMC-Patients`, CC BY-NC-SA 4.0 overall; the operator chose to use all of it): patient
  summaries from published case reports with citation-derived similar-patient links -> "are these two patients
  similar?" decisions. The ``file_path`` prefix (``comm/`` or ``noncomm/``) is kept as the licence subset of each row.

Labels come only from the sources' structured fields.
"""

from __future__ import annotations

import ast
import csv
import random
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from meddecide.gen.core import GeneratedItem, make_item, rng_for
from meddecide.mr.icd import OrderRow
from meddecide.mr.pages import PAGE_CLASSES
from meddecide.utils.hashing import normalize_text, stable_hash

PAGE_LETTER = {k: "ABCDEFGHIJKL"[i] for i, (k, _) in enumerate(PAGE_CLASSES)}
LETTERS = "ABCDEFGHIJKLMNOP"

# ---- MTSamples ----------------------------------------------------------------------------------

MTS_SPECIALTY: dict[str, str] = {
    "discharge summary": "discharge_summary",
    "emergency room reports": "emergency_department",
    "soap / chart / progress notes": "progress_note",
    "radiology": "radiology",
    "diets and nutritions": "nutrition",
    "lab medicine - pathology": "laboratory",
}


def mtsamples_page_class(specialty: str, sample_name: str) -> tuple[str | None, str]:
    spec = (specialty or "").strip().lower()
    name = (sample_name or "").strip().lower()
    if spec in MTS_SPECIALTY:
        return MTS_SPECIALTY[spec], f"specialty {spec}"
    if spec == "consult - history and phy.":
        if "consult" in name:
            return "consult_note", "consult by sample name"
        if "h&p" in name or "history and physical" in name or "admission" in name:
            return "admission", "H&P by sample name"
        return None, "consult/H&P not distinguishable by name"
    return None, f"specialty not a note type: {spec}"


@dataclass(frozen=True)
class TextPage:
    source: str
    doc_id: str
    page_class: str
    text: str
    meta: dict[str, object]


def read_mtsamples(csv_path: str | Path, counts: dict[str, int] | None = None) -> list[TextPage]:
    counts = counts if counts is not None else {}
    by_text: dict[str, set[str]] = {}
    texts: dict[str, str] = {}
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            text = (row.get("transcription") or "").strip()
            if len(text) < 200:
                counts["dropped: transcription missing or < 200 chars"] = counts.get(
                    "dropped: transcription missing or < 200 chars", 0) + 1
                continue
            key = stable_hash(normalize_text(text), length=16)
            texts[key] = text
            page, reason = mtsamples_page_class(row.get("medical_specialty", ""), row.get("sample_name", ""))
            counts[("kept: " if page else "unmapped: ") + reason] = counts.get(("kept: " if page else "unmapped: ") + reason, 0) + 1
            by_text.setdefault(key, set())
            if page:
                by_text[key].add(page)
    out = []
    for key, classes in by_text.items():
        if len(classes) == 1:
            out.append(TextPage("mtsamples", key, next(iter(classes)), texts[key], {}))
        elif len(classes) > 1:
            counts["dropped: conflicting classes for one transcription"] = counts.get(
                "dropped: conflicting classes for one transcription", 0) + 1
    return out


def page_item(page: TextPage, template: str) -> GeneratedItem:
    rng = rng_for(template, page.doc_id)
    options = [(PAGE_LETTER[k], label) for k, label in PAGE_CLASSES]
    return make_item(template, "MR_page_category", f"{page.source}:{page.doc_id}", "base", "choice", page.text,
                     rng.choice(("Which type of medical-record page is this?",
                                 "Classify this page of the medical record.")),
                     options, PAGE_LETTER[page.page_class], {"page_class": page.page_class, "source": page.source})


# ---- CodiEsp ------------------------------------------------------------------------------------

def read_codiesp_split(root: str | Path, split: str) -> list[tuple[str, str, list[str]]]:
    """[(case id, English text, [dotted upper-case ICD-10 codes])] for one split (train or dev; test has no codes)."""
    base = Path(root) / split
    codes: dict[str, list[str]] = {}
    with open(base / f"{split}D.tsv", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                codes.setdefault(parts[0], []).append(parts[1].strip().upper())
    out = []
    for case_id, cs in sorted(codes.items()):
        path = base / "text_files_en" / f"{case_id}.txt"
        if path.exists():
            out.append((case_id, path.read_text(encoding="utf-8"), cs))
    return out


def _category(code: str) -> str:
    return code.split(".")[0]


def codiesp_items(case_id: str, text: str, codes: Iterable[str], order: Mapping[str, OrderRow],
                  counts: dict[str, int] | None = None, max_options: int = 9) -> list[GeneratedItem]:
    """One choice item per unambiguous coded billable code: options = the billable codes of its category (at most
    ``max_options``, gold always included) + "none"; template ``codiesp_icd_support_v1``. The item's hashed split is
    replaced by the builder with the corpus split (CodiEsp train -> train, dev -> test; its test set has no codes)."""
    counts = counts if counts is not None else {}
    coded = sorted({c for c in codes if c in order and order[c].billable})
    counts["dropped: code not billable in the order file"] = counts.get(
        "dropped: code not billable in the order file", 0) + len(set(codes)) - len(coded)
    by_cat: dict[str, list[str]] = {}
    for c in coded:
        by_cat.setdefault(_category(c), []).append(c)
    billable_by_cat: dict[str, list[str]] = {}
    for code, row in order.items():
        if row.billable:
            billable_by_cat.setdefault(_category(code), []).append(code)
    out = []
    for cat, cs in sorted(by_cat.items()):
        if len(cs) > 1:
            counts["dropped: several coded codes in one category"] = counts.get(
                "dropped: several coded codes in one category", 0) + 1
            continue
        gold = cs[0]
        siblings = sorted(billable_by_cat.get(cat, []))
        if len(siblings) < 2:
            counts["dropped: category has a single billable code"] = counts.get(
                "dropped: category has a single billable code", 0) + 1
            continue
        rng = rng_for("codiesp_icd_support_v1", case_id, gold)
        others = [c for c in siblings if c != gold]
        chosen = sorted([*rng.sample(others, min(max_options - 1, len(others))), gold])
        options = [(LETTERS[i], f"{c} - {order[c].long}") for i, c in enumerate(chosen)]
        options.append((LETTERS[len(chosen)], "None of these codes is supported by the documentation"))
        out.append(make_item("codiesp_icd_support_v1", "MR_icd_support", f"codiesp:{case_id}:{cat}", "base", "choice",
                             text, "Which ICD-10-CM code does this documentation support, as the most specific code?",
                             options, LETTERS[chosen.index(gold)], {"code": gold, "source": "codiesp"}))
    return out


# ---- PMC-Patients -------------------------------------------------------------------------------

@dataclass(frozen=True)
class PmcPatient:
    uid: str
    text: str
    similar: dict[str, int]
    licence_subset: str          # "comm" or "noncomm" (from file_path)


def read_pmc_patients(csv_path: str | Path, limit: int | None = None) -> Iterator[PmcPatient]:
    csv.field_size_limit(1 << 26)
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if limit is not None and i >= limit:
                return
            try:
                similar = {str(k): int(v) for k, v in ast.literal_eval(row.get("similar_patients") or "{}").items()}
            except (ValueError, SyntaxError):
                similar = {}
            yield PmcPatient(row["patient_uid"], row["patient"], similar,
                             (row.get("file_path") or "").split("/", 1)[0])


def pmc_similar_items(patients: list[PmcPatient], seed: int = 0, counts: dict[str, int] | None = None
                      ) -> list[GeneratedItem]:
    """For each patient with a similar-patient link: one positive pair (a linked patient) and one negative pair (a
    random unlinked patient); template ``pmc_similar_patient_v1``. Pairs are keyed by the first patient, so the split
    is by that patient; the builder must keep both patients of a test pair out of training (``pair_uids`` in meta)."""
    counts = counts if counts is not None else {}
    by_uid = {p.uid: p for p in patients}
    uids = sorted(by_uid)
    rng = random.Random(seed)
    out = []
    for p in patients:
        linked = [u for u in sorted(p.similar) if u in by_uid]
        if not linked:
            counts["skipped: no linked patient in the file"] = counts.get("skipped: no linked patient in the file", 0) + 1
            continue
        pos = rng.choice(linked)
        neg = rng.choice(uids)
        tries = 0
        while (neg == p.uid or neg in p.similar or p.uid in by_uid[neg].similar) and tries < 20:
            neg, tries = rng.choice(uids), tries + 1
        if tries >= 20:
            continue
        for role, other, gold in (("base", pos, "yes"), ("twin", neg, "no")):
            state = f"Patient A:\n{p.text}\n\nPatient B:\n{by_uid[other].text}"
            out.append(make_item("pmc_similar_patient_v1", "clinical_similarity", f"pmc:{p.uid}", role, "noul", state,
                                 "Do these two published case summaries describe clinically similar patients?",
                                 [("yes", "Yes"), ("no", "No")], gold,
                                 {"pair_uids": [p.uid, other], "licence_subset": p.licence_subset,
                                  "source": "pmc_patients"}))
    return out
