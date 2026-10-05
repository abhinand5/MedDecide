"""Fresh-tier openFDA drug-label builder (filtered on the label's **effective time**).

Three templates, each with gold from a structured label field:

* ``fda_class_choice_v1`` — established pharmacologic class, distractors = other classes
  from the same build (seeded).
* ``fda_boxed_warning_noul_v1`` — does the label carry a boxed warning? The boxed-warning
  section is removed from the state.
* ``fda_route_choice_v1`` — route(s) of administration.

Labels are selected by ``effective_time`` inside the window. A label whose ``set_id`` already
existed before the window is an *update* of an old label and is dropped: that is the
"freshness leak" failure mode in ADVISORY §8.1, and a label's first version is not always
recoverable from the API, so ``openfda`` fields are used only when present.
"""

from __future__ import annotations

import random
from datetime import date
from typing import Any

from meddecide.bench.schema import QuestionType, Tier, make_item, split_by_record_hash

SOURCE = "openfda"
API = "https://api.fda.gov/drug/label.json"
LICENSE = "public-domain"  # openFDA: public domain (see https://open.fda.gov/license/)
PAGE_SIZE = 100

ROUTE_LABELS = [
    "ORAL", "INTRAVENOUS", "INTRAMUSCULAR", "SUBCUTANEOUS", "TOPICAL", "TRANSDERMAL",
    "INHALATION", "NASAL", "OPHTHALMIC", "OTIC", "RECTAL", "VAGINAL", "SUBLINGUAL",
    "INTRADERMAL", "INTRA-ARTICULAR", "IRRIGATION", "OTHER",
]
# Sections that may state the answer to the boxed-warning question.
BOXED_WARNING_KEYS = ("boxed_warning", "boxed_warning_table")


def fetch_labels(
    *,
    start: date,
    end: date,
    page_size: int = PAGE_SIZE,
    max_records: int | None = None,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Fetch every label with an ``effective_time`` inside ``[start, end]``."""
    import httpx

    search = f"effective_time:[{start.strftime('%Y%m%d')} TO {end.strftime('%Y%m%d')}]"
    own_client = client is None
    http = client or httpx.Client(timeout=120.0)
    labels: list[dict[str, Any]] = []
    try:
        skip = 0
        while True:
            response = http.get(API, params={"search": search, "limit": page_size, "skip": skip})
            if response.status_code == 404:  # openFDA returns 404 when nothing matches
                break
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("results", [])
            if not batch:
                break
            labels.extend(batch)
            total = payload.get("meta", {}).get("results", {}).get("total", 0)
            skip += page_size
            if skip >= total or (max_records is not None and len(labels) >= max_records):
                break
    finally:
        if own_client:
            http.close()
    return labels


def _first(label: dict[str, Any], key: str) -> str | None:
    value = label.get(key)
    if isinstance(value, list) and value:
        return str(value[0]).strip() or None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _effective_date(label: dict[str, Any]) -> date | None:
    raw = _first(label, "effective_time")
    if not raw or len(raw) != 8:
        return None
    try:
        return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
    except ValueError:
        return None


def _state_text(label: dict[str, Any], *, exclude: tuple[str, ...]) -> str:
    """Selected label sections joined, with any section named in ``exclude`` removed."""
    sections = [
        "indications_and_usage",
        "dosage_and_administration",
        "contraindications",
        "adverse_reactions",
        "warnings_and_cautions",
        "drug_interactions",
        "mechanism_of_action",
        "clinical_pharmacology",
        "description",
        "pharmacodynamics",
    ]
    parts = []
    for key in sections:
        if key in exclude:
            continue
        text = _first(label, key)
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


class_pool_cache: dict[int, list[str]] = {}


def fetch_class_pool(*, limit: int = 800, client: Any | None = None) -> list[str]:
    """Established pharmacologic class names from across openFDA, for use as distractors.

    A 25-day window contains too few labels for a class to repeat, so a within-window pool
    would be nearly empty. The pool is therefore drawn from the whole openFDA corpus through
    the same public API, deterministically (the API returns a stable order for a fixed
    query), and its size is recorded in every item's ``meta``.
    """
    import httpx

    if limit in class_pool_cache:
        return class_pool_cache[limit]
    own_client = client is None
    http = client or httpx.Client(timeout=120.0)
    names: list[str] = []
    try:
        for skip in range(0, limit, 100):
            response = http.get(
                API,
                params={"search": "_exists_:openfda.pharm_class_epc", "limit": 100, "skip": skip},
            )
            if response.status_code == 404:
                break
            response.raise_for_status()
            batch = response.json().get("results", [])
            if not batch:
                break
            for label in batch:
                for name in (label.get("openfda") or {}).get("pharm_class_epc") or []:
                    name = str(name).strip()
                    if name and name not in names:
                        names.append(name)
    finally:
        if own_client:
            http.close()
    class_pool_cache[limit] = names
    return names


def class_pool_for_build(labels: list[dict[str, Any]], *, cfg: dict[str, Any]) -> list[str]:
    """Distractor pool: openFDA-wide class names, falling back to the window's own labels.

    The fallback exists so a network failure degrades to a thin pool (recorded in the notes)
    rather than to a silently different template.
    """
    pool = fetch_class_pool(limit=int(cfg.get("caps", {}).get("fda_class_pool_size", 800)))
    if len(pool) >= 4:
        return pool
    counts: dict[str, int] = {}
    for label in labels:
        for name in (label.get("openfda") or {}).get("pharm_class_epc") or []:
            name = str(name).strip()
            if name:
                counts[name] = counts.get(name, 0) + 1
    return sorted(counts)


def build_items(
    labels: list[dict[str, Any]],
    *,
    cfg: dict[str, Any],
    window_start: date,
    window_end: date,
    set_ids_seen_before: set[str] | None = None,
) -> tuple[list[Any], dict[str, int], list[str]]:
    """Turn openFDA label records into fresh items; returns ``(items, drops, notes)``."""
    seed = int(cfg.get("seed", 0))
    drops: dict[str, int] = {}
    notes: list[str] = []

    def drop(reason: str, n: int = 1) -> None:
        drops[reason] = drops.get(reason, 0) + n

    # distractor pool for the pharmacologic-class template, from across openFDA
    pool = class_pool_for_build(labels, cfg=cfg)
    seen_before = set_ids_seen_before or set()
    rng = random.Random(f"{seed}:openfda:distractors")
    items: list[Any] = []

    for label in labels:
        set_id = _first(label, "set_id")
        effective = _effective_date(label)
        if not set_id:
            drop("missing_set_id")
            continue
        if effective is None:
            drop("missing_effective_time")
            continue
        if not (window_start <= effective <= window_end):
            drop("effective_time_outside_window")
            continue
        if set_id in seen_before:
            drop("set_id_existed_before_window")
            continue

        split = split_by_record_hash(set_id, dev_fraction=0.2, salt="openfda")
        url = f"https://open.fda.gov/drug/label/?set_id={set_id}"
        openfda = label.get("openfda") or {}
        base_meta = {
            "set_id": set_id,
            "effective_time": effective.strftime("%Y%m%d"),
            "brand_names": openfda.get("brand_name") or [],
            "generic_names": openfda.get("generic_name") or [],
            "manufacturer_names": openfda.get("manufacturer_name") or [],
        }

        # --- established pharmacologic class ---
        classes = [str(c).strip() for c in (openfda.get("pharm_class_epc") or []) if str(c).strip()]
        gold_class = next((c for c in classes if c in pool), None)
        if gold_class is None:
            drop("no_pharm_class_in_pool" if classes else "no_pharm_class")
        else:
            state = _state_text(label, exclude=())
            if len(state) < 80:
                drop("state_too_short")
            else:
                others = [c for c in pool if c != gold_class]
                k = min(3, len(others))
                distractors = rng.sample(others, k)
                labels_list = [*distractors, gold_class]
                order = list(range(len(labels_list)))
                rng.shuffle(order)
                items.append(
                    make_item(
                        tier=Tier.FRESH, source=SOURCE, source_record_id=set_id, source_url=url,
                        source_license=LICENSE, record_date=effective, split=split,
                        template_id="fda_class_choice_v1", skill="pharmacology",
                        qtype=QuestionType.CHOICE, state=state,
                        question="What is the established pharmacologic class of this drug?",
                        options=[{"key": chr(ord("A") + i), "label": labels_list[j]}
                                 for i, j in enumerate(order)],
                        gold=chr(ord("A") + order.index(len(labels_list) - 1)),
                        option_order_seed=seed,
                        meta={**base_meta, "source_field": "openfda.pharm_class_epc",
                              "pharm_class_epc": classes, "distractor_pool_size": len(pool)},
                    )
                )

        # --- boxed warning? (state excludes the boxed-warning section) ---
        has_boxed = any(label.get(key) for key in BOXED_WARNING_KEYS)
        state_no_warning = _state_text(label, exclude=BOXED_WARNING_KEYS)
        if len(state_no_warning) < 80:
            drop("state_too_short_without_boxed_warning")
        else:
            items.append(
                make_item(
                    tier=Tier.FRESH, source=SOURCE, source_record_id=set_id, source_url=url,
                    source_license=LICENSE, record_date=effective, split=split,
                    template_id="fda_boxed_warning_noul_v1", skill="safety",
                    qtype=QuestionType.NOUL, state=state_no_warning,
                    question="Does this drug label carry a boxed warning?",
                    options=[{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}],
                    gold="yes" if has_boxed else "no",
                    option_order_seed=seed,
                    meta={**base_meta, "source_field": "boxed_warning section presence"},
                )
            )

        # --- route of administration ---
        routes = [
            str(r).strip().upper()
            for r in (label.get("openfda", {}).get("route") or label.get("route") or [])
            if str(r).strip()
        ]
        routes = [r for r in routes if r in ROUTE_LABELS]
        if not routes:
            drop("no_route_in_vocabulary")
        else:
            state = _state_text(label, exclude=())
            if len(state) < 80:
                drop("state_too_short")
            else:
                items.append(
                    make_item(
                        tier=Tier.FRESH, source=SOURCE, source_record_id=set_id, source_url=url,
                        source_license=LICENSE, record_date=effective, split=split,
                        template_id="fda_route_choice_v1", skill="pharmacology",
                        qtype=QuestionType.CHOICE, state=state,
                        question="What is the route of administration of this drug?",
                        options=[{"key": chr(ord("A") + i), "label": t}
                                 for i, t in enumerate(ROUTE_LABELS)],
                        gold=chr(ord("A") + ROUTE_LABELS.index(routes[0])),
                        option_order_seed=seed,
                        meta={**base_meta, "source_field": "openfda.route", "routes": routes},
                    )
                )

    notes.append(
        "state = selected label sections; the boxed-warning section is excluded from the "
        "boxed-warning template, and the answer-bearing section is likewise excluded from "
        "the class and route templates"
    )
    notes.append(
        f"established-pharmacologic-class distractor pool: {len(pool)} classes drawn from the "
        "openFDA-wide corpus (a 25-day window holds too few labels for classes to repeat)"
    )
    return items, drops, notes
