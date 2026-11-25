"""Training-only templates for student_v1 (V4): structured fields from pre-window records.

Every builder takes raw source records dated before the v0.2 window (2026-03-01) and returns
``(items, drops)``. Items carry ``split`` = ``train`` or ``dev`` by the record-hash rule the
benchmark builders use (same salts, same 20 % dev fraction), so a record keeps one split across
templates. Nothing here is benchmark data, and no builder can emit a held-out template (D14).

Drops are counted by reason, never silent: every record either yields an item or appears in
``drops`` under a named reason.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

from meddecide.bench.fresh.clinicaltrials import LICENSE as CT_LICENSE
from meddecide.bench.fresh.clinicaltrials import _module, _record_date, _state_text
from meddecide.bench.fresh.openfda import _effective_date
from meddecide.bench.fresh.openfda import _state_text as _label_state_text
from meddecide.bench.fresh.pubmed import PUBTYPE_LABELS, PubmedRecord, pubtype_label
from meddecide.bench.schema import Item, QuestionType, Split, Tier, make_item, split_by_record_hash

HELD_OUT_TEMPLATES = frozenset({
    "ct_phase_choice_v1",
    "fda_boxed_warning_noul_v1",
    "pubmed_humans_noul_v1",
    "ct_arm_role_noul_v1",
})
WINDOW_START = date(2026, 3, 1)
MIN_STATE_CHARS = 80


@dataclass(frozen=True)
class TemplateSpec:
    """What a training template asks, and which source field its gold comes from."""

    template_id: str
    source: str
    qtype: QuestionType
    skill: str
    question: str
    source_field: str


SPECS = {
    "ct_masking_choice_v1": TemplateSpec(
        "ct_masking_choice_v1", "clinicaltrials", QuestionType.CHOICE, "trial_design",
        "What is the masking (blinding) of this study?",
        "designModule.designInfo.maskingInfo.masking",
    ),
    "ct_eligibility_sex_choice_v1": TemplateSpec(
        "ct_eligibility_sex_choice_v1", "clinicaltrials", QuestionType.CHOICE, "eligibility",
        "Which sexes are eligible to enrol in this study?",
        "eligibilityModule.sex",
    ),
    "ct_enrollment_band_score_v1": TemplateSpec(
        "ct_enrollment_band_score_v1", "clinicaltrials", QuestionType.SCORE, "trial_design",
        "How large is the planned enrollment of this study (level 1 smallest, 5 largest)?",
        "designModule.enrollmentInfo.count",
    ),
    "ct_interventional_noul_v1": TemplateSpec(
        "ct_interventional_noul_v1", "clinicaltrials", QuestionType.NOUL, "trial_design",
        "Do the investigators assign participants to an intervention (an interventional study)?",
        "designModule.studyType",
    ),
    "fda_application_family_choice_v1": TemplateSpec(
        "fda_application_family_choice_v1", "openfda", QuestionType.CHOICE, "label_reading",
        "Which approval pathway does the application behind this drug label belong to?",
        "openfda.application_number",
    ),
    "fda_product_type_noul_v1": TemplateSpec(
        "fda_product_type_noul_v1", "openfda", QuestionType.NOUL, "label_reading",
        "Is this a prescription drug label (rather than an over-the-counter label)?",
        "openfda.product_type",
    ),
    "pubmed_check_female_noul_v1": TemplateSpec(
        "pubmed_check_female_noul_v1", "pubmed", QuestionType.NOUL, "record_indexing",
        "Does this record carry the MeSH check tag Female?",
        "MeshHeadingList DescriptorName 'Female'",
    ),
    "pubmed_check_male_noul_v1": TemplateSpec(
        "pubmed_check_male_noul_v1", "pubmed", QuestionType.NOUL, "record_indexing",
        "Does this record carry the MeSH check tag Male?",
        "MeshHeadingList DescriptorName 'Male'",
    ),
    "pubmed_check_adult_noul_v1": TemplateSpec(
        "pubmed_check_adult_noul_v1", "pubmed", QuestionType.NOUL, "record_indexing",
        "Does this record carry the MeSH check tag Adult?",
        "MeshHeadingList DescriptorName 'Adult'",
    ),
    "pubmed_check_child_noul_v1": TemplateSpec(
        "pubmed_check_child_noul_v1", "pubmed", QuestionType.NOUL, "record_indexing",
        "Does this record carry the MeSH check tag Child?",
        "MeshHeadingList DescriptorName 'Child'",
    ),
    "pubmed_pubtype_choice_v1": TemplateSpec(
        "pubmed_pubtype_choice_v1", "pubmed", QuestionType.CHOICE, "record_indexing",
        "Which publication type best describes this record?",
        "PublicationTypeList",
    ),
}

CT_MASKING = {"NONE": "None (open label)", "SINGLE": "Single", "DOUBLE": "Double",
              "TRIPLE": "Triple", "QUADRUPLE": "Quadruple"}
CT_SEX = {"ALL": "All sexes", "FEMALE": "Female only", "MALE": "Male only"}
ENROLLMENT_BANDS = [(0, 49, "fewer than 50"), (50, 199, "50 to 199"), (200, 999, "200 to 999"),
                    (1000, 4999, "1,000 to 4,999"), (5000, 10**9, "5,000 or more")]
FDA_PATHWAYS = {"NDA": "NDA (new drug application)", "ANDA": "ANDA (generic)",
                "BLA": "BLA (biologic licence)"}
PUBMED_CHECK_TAGS = {"pubmed_check_female_noul_v1": "Female", "pubmed_check_male_noul_v1": "Male",
                     "pubmed_check_adult_noul_v1": "Adult", "pubmed_check_child_noul_v1": "Child"}


def _split(record_id: str, salt: str) -> Split:
    """Train or dev by the record-hash rule (the benchmark's dev fraction, train otherwise)."""
    base = split_by_record_hash(record_id, dev_fraction=0.2, salt=salt)
    return Split.DEV if base is Split.DEV else Split.TRAIN


def _guard(template_id: str) -> TemplateSpec:
    if template_id in HELD_OUT_TEMPLATES:
        raise ValueError(f"{template_id} is held out (D14) and cannot be built for training")
    return SPECS[template_id]


def _item(spec: TemplateSpec, *, source_record_id: str, source_url: str, license_: str,
          record_date: date, split: Split, state: str, options: list[dict[str, str]],
          gold: str, meta: dict[str, Any]) -> Item:
    if record_date >= WINDOW_START:
        raise ValueError(f"{spec.template_id}: record {source_record_id} is dated {record_date}, "
                         "inside or after the v0.2 window")
    return make_item(
        tier=Tier.FRESH, source=spec.source, source_record_id=source_record_id,
        source_url=source_url, source_license=license_, record_date=record_date, split=split,
        template_id=spec.template_id, skill=spec.skill, qtype=spec.qtype, state=state,
        question=spec.question, options=options, gold=gold, option_order_seed=0,
        meta={**meta, "source_field": spec.source_field, "prewindow": True, "training_only": True},
    )


def _yes_no(flag: bool) -> tuple[list[dict[str, str]], str]:
    options = [{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}]
    return options, "yes" if flag else "no"


def build_ct_items(
    studies: Iterable[dict[str, Any]], template_ids: Iterable[str]
) -> tuple[dict[str, list[Item]], dict[str, dict[str, int]]]:
    """ClinicalTrials.gov templates from raw study records (pre-window by first-posted date)."""
    wanted = [_guard(t) for t in template_ids]
    items: dict[str, list[Item]] = {s.template_id: [] for s in wanted}
    drops: dict[str, dict[str, int]] = {s.template_id: {} for s in wanted}

    def drop(template_id: str, reason: str) -> None:
        drops[template_id][reason] = drops[template_id].get(reason, 0) + 1

    for study in studies:
        nct = _module(study, "identificationModule").get("nctId")
        record_date = _record_date(study)
        if not nct:
            for spec in wanted:
                drop(spec.template_id, "missing_nct_id")
            continue
        if record_date is None:
            for spec in wanted:
                drop(spec.template_id, "missing_first_posted_date")
            continue
        state = _state_text(study)
        design = _module(study, "designModule")
        split = _split(str(nct), "clinicaltrials")
        url = f"https://clinicaltrials.gov/study/{nct}"
        base_meta = {"nct_id": str(nct), "first_posted": record_date.isoformat()}
        for spec in wanted:
            meta = dict(base_meta)
            if len(state) < MIN_STATE_CHARS:
                drop(spec.template_id, "state_too_short")
                continue
            tid = spec.template_id
            if tid == "ct_masking_choice_v1":
                masking = (design.get("designInfo") or {}).get("maskingInfo", {}).get("masking")
                if masking not in CT_MASKING:
                    drop(tid, f"masking_missing_or_unknown:{masking}")
                    continue
                keys = list(CT_MASKING)
                options = [{"key": chr(ord("A") + i), "label": CT_MASKING[k]} for i, k in enumerate(keys)]
                gold = chr(ord("A") + keys.index(masking))
            elif tid == "ct_eligibility_sex_choice_v1":
                sex = _module(study, "eligibilityModule").get("sex")
                if sex not in CT_SEX:
                    drop(tid, f"sex_missing_or_unknown:{sex}")
                    continue
                keys = list(CT_SEX)
                options = [{"key": chr(ord("A") + i), "label": CT_SEX[k]} for i, k in enumerate(keys)]
                gold = chr(ord("A") + keys.index(sex))
            elif tid == "ct_enrollment_band_score_v1":
                count = (design.get("enrollmentInfo") or {}).get("count")
                if not isinstance(count, int) or count < 0:
                    drop(tid, "enrollment_missing")
                    continue
                band = next(i for i, (lo, hi, _) in enumerate(ENROLLMENT_BANDS) if lo <= count <= hi)
                options = [{"key": str(i + 1), "label": label} for i, (_, _, label) in enumerate(ENROLLMENT_BANDS)]
                gold = str(band + 1)
                meta = {**meta, "enrollment_count": count}
            elif tid == "ct_interventional_noul_v1":
                study_type = design.get("studyType")
                if study_type not in ("INTERVENTIONAL", "OBSERVATIONAL"):
                    drop(tid, f"study_type_not_yes_no:{study_type}")
                    continue
                options, gold = _yes_no(study_type == "INTERVENTIONAL")
            else:  # pragma: no cover - guarded by SPECS
                raise AssertionError(tid)
            items[tid].append(_item(
                SPECS[tid], source_record_id=str(nct), source_url=url, license_=CT_LICENSE,
                record_date=record_date, split=split, state=state, options=options, gold=gold,
                meta=meta,
            ))
    return items, drops


def build_openfda_items(
    labels: Iterable[dict[str, Any]], template_ids: Iterable[str]
) -> tuple[dict[str, list[Item]], dict[str, dict[str, int]]]:
    """openFDA label templates (pre-window by effective date; set_id is the record id)."""
    wanted = [_guard(t) for t in template_ids]
    items: dict[str, list[Item]] = {s.template_id: [] for s in wanted}
    drops: dict[str, dict[str, int]] = {s.template_id: {} for s in wanted}

    def drop(template_id: str, reason: str) -> None:
        drops[template_id][reason] = drops[template_id].get(reason, 0) + 1

    for label in labels:
        set_id = label.get("set_id")
        record_date = _effective_date(label)
        if not set_id:
            for spec in wanted:
                drop(spec.template_id, "missing_set_id")
            continue
        if record_date is None:
            for spec in wanted:
                drop(spec.template_id, "missing_effective_date")
            continue
        openfda = label.get("openfda") or {}
        split = _split(str(set_id), "openfda")
        url = f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?setid={set_id}"
        for spec in wanted:
            tid = spec.template_id
            if tid == "fda_application_family_choice_v1":
                number = str((openfda.get("application_number") or [""])[0]).upper()
                family = next((f for f in FDA_PATHWAYS if number.startswith(f)), None)
                if family is None:
                    drop(tid, "application_family_not_nda_anda_bla")
                    continue
                state = _label_state_text(label, exclude=())
                if len(state) < MIN_STATE_CHARS:
                    drop(tid, "state_too_short")
                    continue
                keys = list(FDA_PATHWAYS)
                options = [{"key": chr(ord("A") + i), "label": FDA_PATHWAYS[k]} for i, k in enumerate(keys)]
                gold = chr(ord("A") + keys.index(family))
            elif tid == "fda_product_type_noul_v1":
                # openFDA writes "HUMAN PRESCRIPTION DRUG" / "HUMAN OTC DRUG"; a " LABEL" suffix is
                # accepted too, so both spellings map the same way
                product = str((openfda.get("product_type") or [""])[0]).upper().replace(" LABEL", "")
                if product not in ("HUMAN PRESCRIPTION DRUG", "HUMAN OTC DRUG"):
                    drop(tid, "product_type_not_rx_or_otc")
                    continue
                state = _label_state_text(label, exclude=())
                if len(state) < MIN_STATE_CHARS:
                    drop(tid, "state_too_short")
                    continue
                options, gold = _yes_no(product == "HUMAN PRESCRIPTION DRUG")
            else:  # pragma: no cover - guarded by SPECS
                raise AssertionError(tid)
            items[tid].append(_item(
                SPECS[tid], source_record_id=str(set_id), source_url=url,
                license_="public-domain (openFDA label data)", record_date=record_date, split=split,
                state=state, options=options, gold=gold, meta={"set_id": str(set_id)},
            ))
    return items, drops


def build_pubmed_items(
    records: Iterable[PubmedRecord], template_ids: Iterable[str],
    *, age_tags: dict[str, list[str]] | None = None,
) -> tuple[dict[str, list[Item]], dict[str, dict[str, int]]]:
    """PubMed templates from parsed records (pre-window by Entrez date, the builder's rule).

    ``age_tags`` maps a PMID to its age-group check tags; the fresh builder reads only the
    humans/animals/sex tags, so the age tags are passed in by the caller (see
    ``scripts/student/v1_pubmed_extract.py``).
    """
    wanted = [_guard(t) for t in template_ids]
    items: dict[str, list[Item]] = {s.template_id: [] for s in wanted}
    drops: dict[str, dict[str, int]] = {s.template_id: {} for s in wanted}

    def drop(template_id: str, reason: str) -> None:
        drops[template_id][reason] = drops[template_id].get(reason, 0) + 1

    for record in records:
        split = _split(record.pmid, "pubmed")
        url = f"https://pubmed.ncbi.nlm.nih.gov/{record.pmid}/"
        state = record.state_text
        ages = set((age_tags or {}).get(record.pmid, []))
        for spec in wanted:
            tid = spec.template_id
            if len(state) < MIN_STATE_CHARS:
                drop(tid, "state_too_short")
                continue
            if tid == "pubmed_pubtype_choice_v1":
                label = pubtype_label(record.pub_types)
                if label is None:
                    drop(tid, "no_pubtype_label")
                    continue
                options = [{"key": chr(ord("A") + i), "label": name} for i, name in enumerate(PUBTYPE_LABELS)]
                gold = chr(ord("A") + PUBTYPE_LABELS.index(label))
            else:
                tag = PUBMED_CHECK_TAGS[tid]
                present = tag in record.check_tags or tag in ages
                options, gold = _yes_no(present)
            items[tid].append(_item(
                spec, source_record_id=record.pmid, source_url=url,
                license_="NLM-public-domain", record_date=record.entrez_date, split=split,
                state=state, options=options, gold=gold,
                meta={"pmid": record.pmid, "journal": record.journal},
            ))
    return items, drops


BUILDERS: dict[str, Callable[..., Any]] = {
    "clinicaltrials": build_ct_items,
    "openfda": build_openfda_items,
    "pubmed": build_pubmed_items,
}
