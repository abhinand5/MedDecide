"""Fresh-tier ClinicalTrials.gov builder (API v2, filtered on **first-posted** date).

Five templates, each with gold taken from one structured field of the study record:

* ``ct_phase_choice_v1`` — study phase (the documented CT.gov phase vocabulary).
* ``ct_randomised_noul_v1`` — is allocation randomised? (``Randomized`` vs anything else)
* ``ct_primary_purpose_choice_v1`` — primary purpose.
* ``ct_intervention_type_choice_v1`` — intervention type.
* ``ct_healthy_volunteers_noul_v1`` — accepts healthy volunteers?

The state is the brief summary + detailed description + condition, with the title and every
field the question asks about removed. Records are fetched by
``AREA[StudyFirstPostDate]RANGE[start, end]``, which is what makes them *fresh*; a study that
merely received an update inside the window is not selected.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any

from meddecide.bench.schema import QuestionType, Tier, make_item, split_by_record_hash

SOURCE = "clinicaltrials"
API = "https://clinicaltrials.gov/api/v2/studies"
LICENSE = "public-domain"  # ClinicalTrials.gov terms: data is public domain
# No `fields` projection: API v2 rejects the short field names with HTTP 400 (verified
# 2026-10-05) and a projection would silently change which fields exist. The full study
# JSON is fetched instead; only the modules the builder needs are read.
FIELDS = None

PHASE_LABELS = ["Early Phase 1", "Phase 1", "Phase 1/Phase 2", "Phase 2", "Phase 2/Phase 3",
                "Phase 3", "Phase 4"]
PURPOSE_LABELS = ["Treatment", "Prevention", "Diagnostic", "Supportive Care", "Screening",
                  "Health Services Research", "Basic Science", "Device Feasibility", "Other"]
INTERVENTION_LABELS = ["Drug", "Biological", "Device", "Procedure", "Behavioral", "Genetic",
                       "Dietary Supplement", "Combination Product", "Diagnostic Test",
                       "Radiation", "Other"]


def fetch_studies(
    *,
    start: date,
    end: date,
    page_size: int = 1000,
    max_records: int | None = None,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Fetch every study first posted inside ``[start, end]`` (paginated, no date drift)."""
    import httpx

    params = {
        # first-posted (not last-updated): a study updated inside the window is NOT fresh
        "filter.advanced": f"AREA[StudyFirstPostDate]RANGE[{start.isoformat()}, {end.isoformat()}]",
        "pageSize": str(page_size),
        "countTotal": "true",
    }
    if FIELDS:
        params["fields"] = FIELDS
    studies: list[dict[str, Any]] = []
    own_client = client is None
    http = client or httpx.Client(timeout=120.0)
    try:
        token: str | None = None
        while True:
            query = dict(params)
            if token:
                query["pageToken"] = token
            # The API returns 429 when a window is paged quickly (a 7-month window is ~40
            # pages). Retry with exponential backoff instead of failing the whole build, and
            # keep a small delay between pages.
            response = None
            for attempt in range(6):
                response = http.get(API, params=query)
                if response.status_code != 429:
                    break
                wait = 2.0 * (2**attempt)
                time.sleep(wait)
            assert response is not None
            response.raise_for_status()
            payload = response.json()
            studies.extend(payload.get("studies", []))
            token = payload.get("nextPageToken")
            if not token or (max_records is not None and len(studies) >= max_records):
                break
            time.sleep(0.2)
    finally:
        if own_client:
            http.close()
    return studies


def _is_true(value: Any) -> bool:
    """Interpret a CT.gov boolean-ish field: booleans, "Yes"/"True"/"yes"/"true" are true."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("yes", "true", "y", "1")


def _phase_label(raw: Any) -> str | None:
    """``PHASE1_PHASE2`` → ``Phase 1/Phase 2``; ``NA``/empty → ``None``."""
    if not raw:
        return None
    text = str(raw).strip().upper()
    if text in ("NA", "N/A", "NONE"):
        return None
    parts = [p for p in text.split("_") if p.startswith("PHASE")]
    if not parts:
        return None
    names = []
    for part in parts:
        number = part.removeprefix("PHASE").strip()
        if number == "1" and "EARLY" in text:
            names.append("Early Phase 1")
        else:
            names.append(f"Phase {number}")
    return "/".join(dict.fromkeys(names))


def _purpose_label(raw: Any) -> str | None:
    """``SUPPORTIVE_CARE`` → ``Supportive Care``; ``NA``/empty → ``None``."""
    if not raw:
        return None
    text = str(raw).strip().upper()
    if text in ("NA", "N/A", "NONE"):
        return None
    label = " ".join(w.capitalize() for w in text.split("_"))
    return label if label in PURPOSE_LABELS else None


def _intervention_label(raw: Any) -> str | None:
    """``DEVICE`` → ``Device``; ``DIETARY_SUPPLEMENT`` → ``Dietary Supplement``."""
    if not raw:
        return None
    text = str(raw).strip().upper()
    if text in ("NA", "N/A", "NONE"):
        return None
    words = text.split("_")
    return " ".join(w.capitalize() for w in words)


def _module(study: dict[str, Any], name: str) -> dict[str, Any]:
    return (study.get("protocolSection") or {}).get(name) or {}


def _state_text(study: dict[str, Any]) -> str:
    """Brief summary + detailed description + condition — never the title, never the answer."""
    ident = _module(study, "identificationModule")
    desc = _module(study, "descriptionModule")
    cond = _module(study, "conditionsModule")
    conditions = ", ".join(cond.get("conditions") or [])
    parts = []
    if conditions:
        parts.append(f"Conditions: {conditions}")
    if desc.get("briefSummary"):
        parts.append(str(desc["briefSummary"]).strip())
    if desc.get("detailedDescription"):
        parts.append(str(desc["detailedDescription"]).strip())
    # the title is deliberately excluded: trial titles routinely state the phase and design
    del ident
    return "\n\n".join(parts).strip()


def _record_date(study: dict[str, Any]) -> date | None:
    raw = _module(study, "statusModule").get("studyFirstPostDateStruct", {}).get("date")
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw))
    except ValueError:
        return None


def build_items(
    studies: list[dict[str, Any]],
    *,
    cfg: dict[str, Any],
    window_start: date,
    window_end: date,
) -> tuple[list[Any], dict[str, int], list[str]]:
    """Turn CT.gov study records into fresh items; returns ``(items, drops, notes)``."""
    seed = int(cfg.get("seed", 0))
    drops: dict[str, int] = {}
    notes: list[str] = []

    def drop(reason: str, n: int = 1) -> None:
        drops[reason] = drops.get(reason, 0) + n

    items: list[Any] = []

    for study in studies:
        ident = _module(study, "identificationModule")
        nct = ident.get("nctId")
        record_date = _record_date(study)
        if not nct:
            drop("missing_nct_id")
            continue
        if record_date is None:
            drop("missing_first_posted_date")
            continue
        if not (window_start <= record_date <= window_end):
            drop("first_posted_outside_window")
            continue
        state = _state_text(study)
        if len(state) < 80:
            drop("state_too_short")
            continue

        split = split_by_record_hash(str(nct), dev_fraction=0.2, salt="clinicaltrials")
        url = f"https://clinicaltrials.gov/study/{nct}"
        base_meta = {
            "nct_id": str(nct),
            "first_posted": record_date.isoformat(),
            "lead_sponsor": _module(study, "sponsorCollaboratorsModule").get("leadSponsor", {}).get("name"),
            "enrollment": _module(study, "designModule").get("enrollmentInfo", {}).get("count"),
        }

        # --- phase ---
        # CT.gov returns phases as upper-snake enums ("PHASE2", "PHASE1_PHASE2") and "NA"
        # for studies without a phase; studyType is needed to tell "not applicable" from
        # "not reported".
        phases_raw = _module(study, "designModule").get("phases") or []
        phases = [_phase_label(p) for p in phases_raw]
        phases = [p for p in phases if p]
        study_type = _module(study, "designModule").get("studyType")
        phase = next((p for p in phases if p in PHASE_LABELS), None)
        if phase is None:
            if study_type == "OBSERVATIONAL":
                drop("observational_study_has_no_phase")
            elif phases_raw in ([], ["NA"]):
                drop("phase_not_applicable_or_missing")
            else:
                drop("phase_not_in_vocabulary")
        else:
            items.append(
                make_item(
                    tier=Tier.FRESH, source=SOURCE, source_record_id=str(nct), source_url=url,
                    source_license=LICENSE, record_date=record_date, split=split,
                    template_id="ct_phase_choice_v1", skill="trial_design",
                    qtype=QuestionType.CHOICE, state=state,
                    question="What is the phase of this study?",
                    options=[{"key": chr(ord("A") + i), "label": t} for i, t in enumerate(PHASE_LABELS)],
                    gold=chr(ord("A") + PHASE_LABELS.index(phase)),
                    option_order_seed=seed,
                    meta={**base_meta, "source_field": "designModule.phases", "phases": phases},
                )
            )

        # --- allocation randomised? ---
        # Values in the wild: RANDOMIZED, NON_RANDOMIZED, NA, absent. NA means the field is
        # not applicable (an observational or single-arm study), which is a real "No" for
        # this question, so it is kept and recorded rather than dropped.
        allocation = _module(study, "designModule").get("designInfo", {}).get("allocation")
        if not allocation:
            drop("missing_allocation")
        else:
            items.append(
                make_item(
                    tier=Tier.FRESH, source=SOURCE, source_record_id=str(nct), source_url=url,
                    source_license=LICENSE, record_date=record_date, split=split,
                    template_id="ct_randomised_noul_v1", skill="trial_design",
                    qtype=QuestionType.NOUL, state=state,
                    question="Is the allocation of participants to arms randomised?",
                    options=[{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}],
                    gold="yes" if allocation == "RANDOMIZED" else "no",
                    option_order_seed=seed,
                    meta={**base_meta, "source_field": "designModule.designInfo.allocation",
                          "allocation": allocation, "study_type": study_type},
                )
            )

        # --- primary purpose ---
        raw_purpose = _module(study, "designModule").get("designInfo", {}).get("primaryPurpose")
        purpose = _purpose_label(raw_purpose)
        if not raw_purpose:
            drop("missing_primary_purpose")
        elif purpose is None:
            # "NA" is the source's way of saying the field does not apply here
            drop("primary_purpose_not_applicable")
        else:
            items.append(
                make_item(
                    tier=Tier.FRESH, source=SOURCE, source_record_id=str(nct), source_url=url,
                    source_license=LICENSE, record_date=record_date, split=split,
                    template_id="ct_primary_purpose_choice_v1", skill="trial_design",
                    qtype=QuestionType.CHOICE, state=state,
                    question="What is the primary purpose of this study?",
                    options=[{"key": chr(ord("A") + i), "label": t} for i, t in enumerate(PURPOSE_LABELS)],
                    gold=chr(ord("A") + PURPOSE_LABELS.index(purpose)),
                    option_order_seed=seed,
                    meta={**base_meta, "source_field": "designModule.designInfo.primaryPurpose",
                          "raw_primary_purpose": raw_purpose},
                )
            )

        # --- intervention type ---
        # The interventions themselves live in armsInterventionsModule.interventions, and
        # their type is an upper-snake enum ("DEVICE", "DRUG", "OTHER", …).
        raw_types = [
            arm.get("type") for arm in (_module(study, "armsInterventionsModule").get("interventions") or [])
        ]
        types = [_intervention_label(t) for t in raw_types]
        types = [t for t in types if t]
        primary_type = types[0] if types else None
        if primary_type is None:
            drop("missing_intervention_type")
        elif primary_type not in INTERVENTION_LABELS:
            drop("intervention_type_not_in_vocabulary")
        else:
            items.append(
                make_item(
                    tier=Tier.FRESH, source=SOURCE, source_record_id=str(nct), source_url=url,
                    source_license=LICENSE, record_date=record_date, split=split,
                    template_id="ct_intervention_type_choice_v1", skill="trial_design",
                    qtype=QuestionType.CHOICE, state=state,
                    question="What type of intervention does this study test?",
                    options=[{"key": chr(ord("A") + i), "label": t}
                             for i, t in enumerate(INTERVENTION_LABELS)],
                    gold=chr(ord("A") + INTERVENTION_LABELS.index(primary_type)),
                    option_order_seed=seed,
                    meta={**base_meta, "source_field": "armsInterventionsModule.interventions[].type",
                          "intervention_types": types, "raw_intervention_types": raw_types},
                )
            )

        # --- accepts healthy volunteers? ---
        # The field is eligibilityModule.healthyVolunteers. Its value is a **JSON boolean**
        # (True/False) in the current API: measured over 2,000 studies in the v0.1 window,
        # {"False": 1461, "True": 506, "None": 33}. bench_v0 compared `str(value).lower()`
        # against "yes", so every record became "no" — which is how this template reached a
        # single 100 %-negative gold class and was dropped as degenerate in the first v0.1
        # build. Boolean and "Yes"/"No" string shapes are both handled here.
        accepts = _module(study, "eligibilityModule").get("healthyVolunteers")
        if accepts is None:
            drop("missing_healthy_volunteers_field")
        else:
            items.append(
                make_item(
                    tier=Tier.FRESH, source=SOURCE, source_record_id=str(nct), source_url=url,
                    source_license=LICENSE, record_date=record_date, split=split,
                    template_id="ct_healthy_volunteers_noul_v1", skill="trial_design",
                    qtype=QuestionType.NOUL,
                    # the eligibility text itself is excluded: it states the answer
                    state=state,
                    question="Does this study accept healthy volunteers?",
                    options=[{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}],
                    gold="yes" if _is_true(accepts) else "no",
                    option_order_seed=seed,
                    meta={**base_meta, "source_field": "eligibilityModule.healthyVolunteers",
                          "healthy_volunteers": accepts},
                )
            )

    notes.append(
        "state = conditions + brief summary + detailed description; the brief title is excluded "
        "because trial titles routinely state phase and design"
    )
    notes.append("eligibility criteria text is never included in the state")
    return items, drops, notes
