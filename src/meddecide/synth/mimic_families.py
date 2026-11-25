"""MIMIC-grounded families for synth_v1: case facts built by code from structured MIMIC fields.

Five osler_v4 structured-label templates are used (the label is one field with three to six options). For each
training-split admission (or ED stay) the builder reads structured fields only: coded diagnoses and their titles,
the MS-DRG code and title (used for the gold and for the offered options, never as a fact), the APR-DRG severity
level, and ED triage vitals and disposition. The facts are the clinical findings the label depends on, each with its
exact value. The gold label is the structured field itself. Nothing in this module reads a note, calls a model or
returns a value to a caller other than a ``CaseFacts`` or a count.

Exclusions (decided here, counted by reason in the builder output):

- records outside the training split (``not_train_split``);
- records that carry a held-out condition: the heart-failure family, AKI (N17), CHF (I50), and ICD-9 428 and 584
  (``held_out_condition``), so held-out content never enters a training fact;
- records without the fields a template needs (reason codes below).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from meddecide.mr import mimic
from meddecide.mr.mimic_items import HELD_OUT_FAMILIES, dotted, drg_split
from meddecide.synth.facts import CaseFacts, Fact, TemplateSpec
from meddecide.utils.hashing import stable_hash

TRAIN = "train"
TEMPLATES: tuple[str, ...] = (
    "mimic_icd_specificity_v1",
    "mimic_msdrg_v1",
    "mimic_drg_cc_level_v1",
    "mimic_apr_severity_v1",
    "mimic_ed_disposition_v1",
)
# Templates from the osler_v4 set that are considered and not used, with the reason code (see mimic_families.md).
EXCLUDED_TEMPLATES: dict[str, str] = {
    "mimic_rad_exam_v1": "near_variant_option_names",
}
# Templates kept out of the training mix, with the reason code. The items stay generated and counted; the mix
# builder must skip them. icd_specificity: the gold option's title is stated in the facts (leakage probe 0.983 vs 0.173).
MIX_EXCLUSIONS: dict[str, str] = {"mimic_icd_specificity_v1": "answer_in_facts"}


def in_training_mix(template: str) -> bool:
    """False for templates listed in MIX_EXCLUSIONS (reason recorded there)."""
    return template not in MIX_EXCLUSIONS


HELD_OUT_PREFIXES: tuple[str, ...] = ("I50", "N17")
HELD_OUT_ICD9_PREFIXES: tuple[str, ...] = ("428", "584")
HELD_OUT_TEXT = re.compile(r"heart failure|acute kidney|kidney injury|\bCHF\b", re.IGNORECASE)
MAX_OTHER_DIAGNOSES = 3
BALANCED = frozenset({"mimic_drg_cc_level_v1", "mimic_apr_severity_v1", "mimic_ed_disposition_v1"})

# DRG CC/MCC level: the label words and the option text shown to the probe (the item carries the label words only).
CC_KEYS: tuple[str, ...] = ("MCC", "CC", "neither")
CC_DISPLAY: tuple[str, ...] = ("MCC", "CC", "neither (without CC/MCC)")
ED_KEYS: tuple[str, ...] = ("admitted", "home", "other")
ED_DISPOSITION = {"ADMITTED": "admitted", "HOME": "home"}


def family_of(template: str) -> str:
    """Family name used in CaseFacts and manifests, e.g. ``synth_mimic_icd_specificity_v1``."""
    if template not in TEMPLATES:
        raise ValueError(f"not a grounded template: {template!r}")
    return f"synth_{template}"


def is_held_out_icd10(code: str) -> bool:
    """True for an undotted ICD-10-CM code in the heart-failure family or the AKI code (CHF, AKI prefixes)."""
    if code[:3] in HELD_OUT_PREFIXES:
        return True
    curated = mimic.UNDOTTED_TO_CODE.get(code)
    return curated is not None and mimic.CODE_TO_FAMILY.get(curated) in HELD_OUT_FAMILIES


def is_held_out_ed(code: str, version: int, title: str) -> bool:
    """ED diagnosis held-out test: ICD-10 by code, ICD-9 by prefix, and any version by the recorded title."""
    if HELD_OUT_TEXT.search(title or ""):
        return True
    if int(version) == 10:
        return is_held_out_icd10(code)
    return code[:3] in HELD_OUT_ICD9_PREFIXES


@dataclass(frozen=True)
class GroundedCase:
    """One item before generation: the facts, the offered option keys (the label is one of them), and the option text.

    ``display`` is used only by the leakage probe and by the gitignored item file, never by a committed manifest.
    """

    template: str
    case: CaseFacts
    keys: tuple[str, ...]
    display: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.display) or not 3 <= len(self.keys) <= 6:
            raise ValueError("a grounded case needs three to six options with matching display text")
        if self.case.label not in self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError("the gold label must be one of the unique option keys")
        if self.case.family != family_of(self.template):
            raise ValueError("case family does not match the template")

    @property
    def spec(self) -> TemplateSpec:
        keys = self.keys
        return TemplateSpec(
            family=self.case.family,
            label_set=keys,
            option_builder=lambda _facts: keys,
            label_match="token",
        )


@dataclass
class Built:
    """Output of one builder: the cases, and a count per drop reason (no silent drops)."""

    cases: list[GroundedCase] = field(default_factory=list)
    drops: Counter[str] = field(default_factory=Counter)


def _order_key(seed: int, tag: str, value: Any) -> str:
    return stable_hash({"seed": seed, "tag": tag, "value": str(value)}, length=16)


def _codes(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("icd10_codes")
    if raw is None:
        return []
    return [str(c) for c in raw if str(c).strip()]


def _usable(row: Mapping[str, Any], drops: Counter[str], *, need_principal: bool,
            titles: Mapping[str, str]) -> bool:
    """Common gate for admissions: training split, no held-out condition, ICD-10 coding, a principal title if needed."""
    if row.get("split") != TRAIN:
        drops["not_train_split"] += 1
        return False
    codes = _codes(row)
    if any(is_held_out_icd10(c) for c in codes):
        drops["held_out_condition"] += 1
        return False
    if row.get("coding") != "icd10":
        drops["non_icd10_coding"] += 1
        return False
    if need_principal:
        principal = row.get("principal_icd10")
        if not principal or principal not in titles:
            drops["no_principal_title"] += 1
            return False
    return True


def _diagnosis_facts(row: Mapping[str, Any], titles: Mapping[str, str], drops: Counter[str],
                     primary: str | None, primary_key: str = "principal diagnosis") -> tuple[Fact, ...] | None:
    """The primary diagnosis and up to three other coded diagnoses as facts (exact titles). None when no fact is possible."""
    facts: list[Fact] = []
    seen: set[str] = set()
    if primary is not None:
        facts.append(Fact("condition", primary_key, titles[primary]))
        seen.add(titles[primary])
    for code in _codes(row):
        if len(facts) >= MAX_OTHER_DIAGNOSES + (1 if primary is not None else 0):
            break
        title = titles.get(code)
        if not title or title in seen:
            continue
        seen.add(title)
        facts.append(Fact("condition", "other coded diagnosis", title))
    if not facts:
        drops["no_coded_diagnosis_title"] += 1
        return None
    return tuple(facts)


def _drg_title_table(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """MS-DRG code -> title from training-split admissions only."""
    table: dict[str, str] = {}
    for row in rows:
        if row.get("split") != TRAIN:
            continue
        code, title = row.get("ms_drg_code"), row.get("ms_drg_title")
        if code and title:
            table.setdefault(str(code), str(title))
    return table


def icd_specificity_cases(rows: Sequence[Mapping[str, Any]], titles: Mapping[str, str],
                          billable_by_category: Mapping[str, Sequence[str]], *, seed: int) -> Built:
    """mimic_icd_specificity_v1: which billable code of a category was coded. Options: the coded code and up to five
    billable siblings of its three-character category (three to six options). Facts: the coded diagnoses' titles."""
    out = Built()
    template = "mimic_icd_specificity_v1"
    for row in rows:
        if not _usable(row, out.drops, need_principal=False, titles=titles):
            continue
        hadm = row.get("hadm_id")
        codes = _codes(row)
        by_cat: dict[str, list[str]] = {}
        for code in codes:
            by_cat.setdefault(code[:3], []).append(code)
        singles = [cat for cat, members in by_cat.items() if len(set(members)) == 1]
        if not singles:
            out.drops["no_single_coded_category"] += 1
            continue
        cat = min(singles, key=lambda c: _order_key(seed, f"cat:{hadm}", c))
        code = by_cat[cat][0]
        if code not in titles:
            out.drops["no_title_for_code"] += 1
            continue
        siblings = [c for c in billable_by_category.get(cat, []) if c != code and c in titles]
        if len(siblings) < 2:
            out.drops["fewer_than_two_billable_siblings"] += 1
            continue
        picks = sorted(siblings, key=lambda c: _order_key(seed, f"pick:{hadm}", c))[:5]
        offered = sorted([code, *picks], key=lambda c: _order_key(seed, f"order:{hadm}", c))
        facts = _diagnosis_facts(row, titles, out.drops, primary=code, primary_key="coded diagnosis")
        if facts is None:
            continue
        keys = tuple(dotted(c) for c in offered)
        display = tuple(f"{dotted(c)} - {titles[c]}" for c in offered)
        case = CaseFacts(family=family_of(template), facts=facts, label=dotted(code))
        out.cases.append(GroundedCase(template, case, keys, display))
    return out


def _similar_codes(code: str, drg_titles: Mapping[str, str], cache: dict[str, list[str]]) -> list[str]:
    if code not in cache:
        tokens = {w for w in re.findall(r"[A-Z0-9]+", drg_titles[code].upper()) if len(w) > 1}
        scored = []
        for other, title in drg_titles.items():
            if other == code:
                continue
            other_tokens = {w for w in re.findall(r"[A-Z0-9]+", title.upper()) if len(w) > 1}
            union = other_tokens | tokens
            score = len(other_tokens & tokens) / len(union) if union else 0.0
            if score > 0:
                scored.append((-score, other))
        cache[code] = [c for _, c in sorted(scored)]
    return cache[code]


def msdrg_cases(rows: Sequence[Mapping[str, Any]], titles: Mapping[str, str], drg_titles: Mapping[str, str], *,
                seed: int) -> Built:
    """mimic_msdrg_v1: which MS-DRG was assigned. Options: the DRG, up to two severity siblings (same base title) and the
    most similar DRG titles of the training split, four to six in all. Facts: the principal and other coded diagnoses."""
    out = Built()
    template = "mimic_msdrg_v1"
    bases: dict[str, list[str]] = {}
    for code, title in drg_titles.items():
        bases.setdefault(drg_split(title)[0], []).append(code)
    cache: dict[str, list[str]] = {}
    for row in rows:
        if not _usable(row, out.drops, need_principal=True, titles=titles):
            continue
        drg = row.get("ms_drg_code")
        if not drg or str(drg) not in drg_titles:
            out.drops["drg_without_training_title"] += 1
            continue
        drg = str(drg)
        hadm = row.get("hadm_id")
        base = drg_split(drg_titles[drg])[0]
        siblings = [c for c in bases.get(base, []) if c != drg][:2]
        distractors = [c for c in _similar_codes(drg, drg_titles, cache) if c not in siblings][: max(0, 5 - len(siblings))]
        candidates = [drg, *siblings, *distractors][:6]
        if len(candidates) < 4:
            out.drops["fewer_than_four_drg_options"] += 1
            continue
        offered = sorted(candidates, key=lambda c: _order_key(seed, f"drg:{hadm}", c))
        facts = _diagnosis_facts(row, titles, out.drops, primary=row["principal_icd10"])
        if facts is None:
            continue
        display = tuple(f"{c} - {drg_titles[c]}" for c in offered)
        case = CaseFacts(family=family_of(template), facts=facts, label=drg)
        out.cases.append(GroundedCase(template, case, tuple(offered), display))
    return out


def drg_cc_cases(rows: Sequence[Mapping[str, Any]], titles: Mapping[str, str]) -> Built:
    """mimic_drg_cc_level_v1: is the MS-DRG title's severity marker MCC, CC or neither. The label is parsed by code from
    the DRG title (titles without a three-way marker are dropped); the title itself is never a fact."""
    out = Built()
    template = "mimic_drg_cc_level_v1"
    level_key = {"MCC": "MCC", "CC": "CC", "NONE": "neither"}
    for row in rows:
        if not _usable(row, out.drops, need_principal=True, titles=titles):
            continue
        title = row.get("ms_drg_title")
        if not title:
            out.drops["no_drg_title"] += 1
            continue
        _, level = drg_split(str(title))
        if level is None:
            out.drops["title_without_three_way_marker"] += 1
            continue
        facts = _diagnosis_facts(row, titles, out.drops, primary=row["principal_icd10"])
        if facts is None:
            continue
        case = CaseFacts(family=family_of(template), facts=facts, label=level_key[level])
        out.cases.append(GroundedCase(template, case, CC_KEYS, CC_DISPLAY))
    return out


def apr_cases(rows: Sequence[Mapping[str, Any]], titles: Mapping[str, str]) -> Built:
    """mimic_apr_severity_v1: APR-DRG severity of illness, level 1 to 4 (the label is the APR field). Facts: diagnoses."""
    out = Built()
    template = "mimic_apr_severity_v1"
    for row in rows:
        if not _usable(row, out.drops, need_principal=True, titles=titles):
            continue
        level = row.get("apr_severity")
        if level is None or (isinstance(level, float) and math.isnan(level)) or int(level) not in (1, 2, 3, 4):
            out.drops["apr_severity_missing_or_outside_1_to_4"] += 1
            continue
        facts = _diagnosis_facts(row, titles, out.drops, primary=row["principal_icd10"])
        if facts is None:
            continue
        keys = ("1", "2", "3", "4")
        case = CaseFacts(family=family_of(template), facts=facts, label=str(int(level)))
        out.cases.append(GroundedCase(template, case, keys, keys))
    return out


def _vital(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == "":
        return None
    try:
        return f"{round(float(value), 1):g}"
    except (TypeError, ValueError):
        return str(value).strip()


def ed_facts(stay: Mapping[str, Any], dx: Sequence[tuple[str, int, str]]) -> tuple[Fact, ...]:
    """Facts of one ED stay from its triage fields and its ED diagnoses. Missing vitals are left out, not written."""
    facts: list[Fact] = []
    chief = str(stay.get("chiefcomplaint") or "").strip()
    if chief:
        facts.append(Fact("finding", "chief complaint", chief))
    for key, label, unit in (("temperature", "temperature", "F"), ("heartrate", "heart rate", "bpm"),
                             ("resprate", "respiratory rate", "breaths per minute"), ("o2sat", "SpO2", "%")):
        shown = _vital(stay.get(key))
        if shown is not None:
            facts.append(Fact("finding", label, f"{label} {shown} {unit}" if unit != "%" else f"{label} {shown}%"))
    sbp, dbp = _vital(stay.get("sbp")), _vital(stay.get("dbp"))
    if sbp is not None and dbp is not None:
        facts.append(Fact("finding", "blood pressure", f"blood pressure {sbp}/{dbp} mmHg"))
    pain = _vital(stay.get("pain"))
    if pain is not None:
        facts.append(Fact("finding", "pain score", f"pain score {pain}"))
    seen: set[str] = set()
    for _code, _version, title in dx:
        if len(seen) >= MAX_OTHER_DIAGNOSES:
            break
        title = (title or "").strip()
        if title and title not in seen:
            seen.add(title)
            facts.append(Fact("condition", "emergency department diagnosis", title))
    return tuple(facts)


def ed_disposition_cases(stays: Sequence[Mapping[str, Any]], dx_by_stay: Mapping[int, Sequence[tuple[str, int, str]]]
                         ) -> Built:
    """mimic_ed_disposition_v1: admitted, home or other, from the ED disposition field. Each stay carries its ``split``
    (the patient-level split of its subject). Facts: triage findings and the ED diagnoses' recorded titles."""
    out = Built()
    template = "mimic_ed_disposition_v1"
    for stay in stays:
        if stay.get("split") != TRAIN:
            out.drops["not_train_split"] += 1
            continue
        dx = list(dx_by_stay.get(int(stay["stay_id"]), []))
        if any(is_held_out_ed(code, version, title) for code, version, title in dx):
            out.drops["held_out_condition"] += 1
            continue
        disposition = str(stay.get("disposition") or "").strip().upper()
        if not disposition:
            out.drops["disposition_missing"] += 1
            continue
        try:
            facts = ed_facts(stay, dx)
            case = CaseFacts(family=family_of(template), facts=facts,
                             label=ED_DISPOSITION.get(disposition, "other"))
        except ValueError:
            out.drops["fact_value_without_token"] += 1
            continue
        out.cases.append(GroundedCase(template, case, ED_KEYS, ED_KEYS))
    return out


def select(cases: Sequence[GroundedCase], template: str, n_total: int, *, seed: int) -> tuple[list[GroundedCase], Counter[str]]:
    """Deterministic selection of up to ``n_total`` cases. Duplicate fact lists are dropped (counted). For the balanced
    templates each label gets an equal share, and any shortage of a label is counted (never silently refilled)."""
    shortage: Counter[str] = Counter()
    unique: dict[str, GroundedCase] = {}
    duplicates = 0
    for case in sorted(cases, key=lambda g: _order_key(seed, f"select:{template}", g.case.case_id)):
        if case.case.case_id in unique:
            duplicates += 1
            continue
        unique[case.case.case_id] = case
    ordered = list(unique.values())
    if template not in BALANCED:
        chosen = ordered[:n_total]
    else:
        keys = ordered[0].keys if ordered else ()
        cap = n_total // len(keys) if keys else 0
        taken: Counter[str] = Counter()
        chosen = []
        for case in ordered:
            if taken[case.case.label] < cap:
                taken[case.case.label] += 1
                chosen.append(case)
        for key in keys:
            if taken[key] < cap:
                shortage[f"label {key} below share"] += cap - taken[key]
    shortage["duplicate facts (not selected)"] += duplicates
    return chosen, shortage


def label_counts(cases: Iterable[GroundedCase]) -> dict[str, int]:
    counts: Counter[str] = Counter(c.case.label for c in cases)
    return dict(sorted(counts.items()))

