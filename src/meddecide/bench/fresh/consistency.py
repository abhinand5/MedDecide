"""Record-claim consistency templates (D15 family, loop task S2).

A **record-claim consistency** item asks whether a record supports a *stated* fact, rather
than asking the model to infer a hidden field. Gold is known by construction from a structured
field, and — this is the design constraint — the swapped (unsupported) value must also appear
in the state, in a **different role**, so that a string-presence rule cannot solve the item.

Three templates, two sources, all built for both windows (fresh >= 2026-03-01 and pre-window
< 2026-03-01, the latter for training only):

===========================  ======  ====================================================================
template                     qtype   role-binding design
===========================  ======  ====================================================================
``ct_arm_role_noul_v1``      noul    the arms are listed with their interventions and the *arm types removed*;
                                     the claim names an intervention, the answer is which role its arm has
``ct_claim_set_choice_v1``   choice  four stated fields, exactly one swapped; for the intervention-type
                                     field the swapped value is another intervention's type in the same
                                     record, so the value is present in the state in a different role
``fda_route_claim_noul_v1``  noul    the claim names a route of administration; the unsupported route is one
                                     that also appears verbatim in the label text
===========================  ======  ====================================================================

Every builder returns ``(items, drops, notes)``; every drop has a counted reason. Nothing here
is model-produced: the gold is a lookup of a structured source field.
"""

from __future__ import annotations

import random
import re
from datetime import date
from typing import Any

from meddecide.bench.fresh.clinicaltrials import (
    PHASE_LABELS,
    PURPOSE_LABELS,
    _is_true,
    _module,
    _phase_label,
    _purpose_label,
    _record_date,
    _state_text,
)
from meddecide.bench.schema import QuestionType, Tier, make_item, split_by_record_hash

CT_SOURCE = "clinicaltrials"
CT_URL = "https://clinicaltrials.gov/study/{nct}"
FDA_SOURCE = "openfda"
FDA_URL = "https://open.fda.gov/drug/label/?set_id={set_id}"
CT_LICENSE = "public-domain"
FDA_LICENSE = "public-domain"

## Roles an arm can have. Only EXPERIMENTAL is the "experimental" role; everything else is a
## comparator role for the purpose of a claim about the experimental arm.
EXPERIMENTAL_ARM_TYPE = "EXPERIMENTAL"
ARM_TYPE_LABELS = {
    "EXPERIMENTAL": "experimental",
    "ACTIVE_COMPARATOR": "active comparator",
    "PLACEBO_COMPARATOR": "placebo comparator",
    "SHAM_COMPARATOR": "sham comparator",
    "NO_INTERVENTION": "no-intervention",
    "OTHER": "other",
}
ROUTE_WORDS = [
    "ORAL", "INTRAVENOUS", "INTRAMUSCULAR", "SUBCUTANEOUS", "TOPICAL", "TRANSDERMAL",
    "INHALATION", "NASAL", "OPHTHALMIC", "OTIC", "RECTAL", "VAGINAL", "SUBLINGUAL",
    "INTRADERMAL", "INTRA-ARTICULAR", "IRRIGATION",
]
# Words that name an arm's role rather than its content: excluded from the arms block, because
# keeping them would let a string rule read the answer.
ROLE_WORD_RE = re.compile(
    r"\b(experimental|placebo|comparator|sham|control|active comparator|no intervention)\b",
    re.IGNORECASE,
)


def _confined_names(arm: dict[str, Any], arms: list[dict[str, Any]]) -> list[str]:
    """Interventions this arm alone lists, excluding names that are themselves role words.

    A name in two arms makes the claim true of either arm, so it cannot carry a consistency
    question; a name like "Placebo" would let a string rule answer it.
    """
    occurrence: dict[str, int] = {}
    for other in arms:
        for name in dict.fromkeys(other["interventions"]):
            occurrence[name] = occurrence.get(name, 0) + 1
    return [
        name for name in arm["interventions"]
        if occurrence[name] == 1 and not ROLE_WORD_RE.search(name)
    ]


def _route_stem_in(upper_state: str, route: str) -> bool:
    """True when the route word occurs in the text as a word or as its stem.

    ``INTRAVENOUS`` matches "intravenously"; ``ORAL`` matches "orally"; it does not match a
    different word that merely starts the same way (``OTIC`` does not match "otic" inside
    another word because the match is anchored on a word boundary).
    """
    return re.search(rf"\b{re.escape(route)}\w*", upper_state) is not None


def _intervention_name(formatted: str) -> str:
    """``"Drug: Aspirin"`` -> ``"Aspirin"``; keeps the name, drops the type prefix."""
    text = str(formatted).strip()
    if ":" in text:
        head, _, tail = text.partition(":")
        if head.strip().lower() in {
            "drug", "biological", "device", "procedure", "behavioral", "genetic",
            "dietary supplement", "combination product", "diagnostic test", "radiation", "other",
        }:
            return tail.strip()
    return text


def _arm_rows(study: dict[str, Any]) -> list[dict[str, Any]]:
    """Arms as ``[{type, interventions, index}]``, in record order, with names only."""
    groups = _module(study, "armsInterventionsModule").get("armGroups") or []
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        names = [
            _intervention_name(name)
            for name in (group.get("interventionNames") or [])
            if str(name).strip()
        ]
        rows.append(
            {
                "index": index,
                "type": str(group.get("type") or "").strip().upper(),
                "interventions": [n for n in dict.fromkeys(names) if n],
            }
        )
    return rows


def _arms_block(arms: list[dict[str, Any]]) -> str:
    """Render the arms with their interventions and **no** type, label, title or description.

    An arm with no intervention names is dropped from the block rather than rendered empty: an
    empty arm is a role cue by elimination.
    """
    lines = []
    for arm in arms:
        if not arm["interventions"]:
            continue
        names = ", ".join(arm["interventions"])
        lines.append(f"Arm {arm['index'] + 1}: {names}")
    return "\n".join(lines)


def build_ct_arm_role_items(
    studies: list[dict[str, Any]],
    *,
    cfg: dict[str, Any],
    window_start: date,
    window_end: date,
    template_id: str = "ct_arm_role_noul_v1",
) -> tuple[list[Any], dict[str, int], list[str]]:
    """Claim: ``<intervention> is given in the experimental arm of this study.``

    Supported when the intervention is confined to the arm whose structured type is
    ``EXPERIMENTAL``; unsupported when it is confined to a comparator arm. "Confined" is
    enforced in both directions: an intervention that appears in more than one arm (background
    therapy, a shared placebo) is never used, because the claim would then be true regardless
    of which arm it names.
    """
    seed = int(cfg.get("seed", 0))
    drops: dict[str, int] = {}
    notes: list[str] = []

    def drop(reason: str, n: int = 1) -> None:
        drops[reason] = drops.get(reason, 0) + n

    items: list[Any] = []
    supported_flags: list[bool] = []
    for study in studies:
        nct = _module(study, "identificationModule").get("nctId")
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
        arms = _arm_rows(study)
        experimental = [a for a in arms if a["type"] == EXPERIMENTAL_ARM_TYPE and a["interventions"]]
        comparators = [a for a in arms if a["type"] and a["type"] != EXPERIMENTAL_ARM_TYPE
                       and a["interventions"]]
        if not experimental:
            drop("no_experimental_arm_with_interventions")
            continue
        if not comparators:
            drop("no_comparator_arm_with_interventions")
            continue
        # A name that *is* a role word ("Placebo") would let a string rule answer the
        # question, so such names are excluded from the candidate pools. The arms that carry
        # them stay in the state: removing an arm would remove a role cue by elimination.
        supported_pool = _confined_names(experimental[0], arms)
        comparator_pool = [n for arm in comparators for n in _confined_names(arm, arms)]
        if not supported_pool:
            drop("no_intervention_confined_to_the_experimental_arm")
            continue
        if not comparator_pool:
            drop("no_intervention_confined_to_a_comparator_arm")
            continue

        state_block = _arms_block(arms)
        base_state = _state_text(study)
        if len(base_state) < 80:
            drop("state_too_short")
            continue
        state = (
            f"{base_state}\n\nArms and the interventions assigned to them:\n{state_block}"
        )

        # Alternate supported / unsupported so the template is balanced by construction, and
        # record the flag so the balance can be checked on the built file.
        want_supported = len(supported_flags) % 2 == 0
        pool = supported_pool if want_supported else comparator_pool
        rng = random.Random(f"{seed}:{nct}:arm_role")
        claimed = pool[rng.randrange(len(pool))]
        supported_flags.append(want_supported)
        split = split_by_record_hash(str(nct), dev_fraction=0.2, salt="clinicaltrials")
        items.append(
            make_item(
                tier=Tier.FRESH,
                source=CT_SOURCE,
                source_record_id=str(nct),
                source_url=CT_URL.format(nct=nct),
                source_license=CT_LICENSE,
                record_date=record_date,
                split=split,
                template_id=template_id,
                skill="consistency",
                qtype=QuestionType.NOUL,
                state=state,
                question=(
                    f"Does this record support the claim that {claimed} is given in the "
                    "experimental arm of this study?"
                ),
                options=[{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}],
                gold="yes" if want_supported else "no",
                option_order_seed=seed,
                meta={
                    "source_field": "armsInterventionsModule.armGroups[].type",
                    "claimed_intervention": claimed,
                    "supported": want_supported,
                    "role_rule": (
                        "intervention confined to the arm typed EXPERIMENTAL"
                        if want_supported
                        else "intervention confined to a comparator arm"
                    ),
                    "arm_types": [a["type"] for a in arms],
                    "n_arms": len(arms),
                    "first_posted": record_date.isoformat(),
                },
            )
        )

    n_supported = sum(1 for flag in supported_flags if flag)
    notes.append(
        f"claim = '<intervention> is given in the experimental arm'; {n_supported} supported of "
        f"{len(supported_flags)}; the arms block lists every arm with its interventions and no "
        "arm type, label, title or description"
    )
    notes.append(
        "an intervention listed in more than one arm is never used: the claim would be true of "
        "either arm, so it is not a consistency question"
    )
    return items, drops, notes


def build_ct_claim_set_items(
    studies: list[dict[str, Any]],
    *,
    cfg: dict[str, Any],
    window_start: date,
    window_end: date,
    template_id: str = "ct_claim_set_choice_v1",
    n_fields: int = 4,
) -> tuple[list[Any], dict[str, int], list[str]]:
    """Multi-field variant: four stated fields, exactly one of them unsupported.

    The fields come from the same structured vocabulary the trial-design templates use (phase,
    allocation, primary purpose, intervention type, healthy volunteers). The swapped value is
    another value of that field's vocabulary — for the intervention-type field it is
    preferably the type of a *different intervention in the same record*, which puts the
    claimed value in the state in a different role and defeats a string-presence rule.
    """
    seed = int(cfg.get("seed", 0))
    drops: dict[str, int] = {}
    notes: list[str] = []

    def drop(reason: str, n: int = 1) -> None:
        drops[reason] = drops.get(reason, 0) + n

    items: list[Any] = []
    swapped_counter: dict[str, int] = {}
    for study in studies:
        nct = _module(study, "identificationModule").get("nctId")
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

        # ---- gather up to n_fields true field values -------------------------
        design = _module(study, "designModule")
        fields: list[dict[str, Any]] = []
        phases = [p for p in (_phase_label(raw) for raw in (design.get("phases") or [])) if p]
        if phases:
            fields.append({"name": "phase", "value": phases[0], "vocabulary": PHASE_LABELS,
                           "source_field": "designModule.phases"})
        allocation = design.get("designInfo", {}).get("allocation")
        if allocation:
            fields.append({
                "name": "allocation",
                "value": "Randomised" if allocation == "RANDOMIZED" else "Not randomised",
                "vocabulary": ["Randomised", "Not randomised"],
                "source_field": "designModule.designInfo.allocation",
            })
        purpose = _purpose_label(design.get("designInfo", {}).get("primaryPurpose"))
        if purpose:
            fields.append({"name": "primary purpose", "value": purpose,
                           "vocabulary": PURPOSE_LABELS,
                           "source_field": "designModule.designInfo.primaryPurpose"})
        interventions = _module(study, "armsInterventionsModule").get("interventions") or []
        types = [
            str(i.get("type") or "").strip().upper().replace("_", " ").title()
            for i in interventions
        ]
        types = [t for t in dict.fromkeys(types) if t]
        if types:
            fields.append({"name": "intervention type", "value": types[0], "vocabulary": types,
                           "source_field": "armsInterventionsModule.interventions[].type",
                           "other_record_values": types[1:]})
        healthy = _module(study, "eligibilityModule").get("healthyVolunteers")
        if healthy is not None:
            fields.append({"name": "accepts healthy volunteers",
                           "value": "Yes" if _is_true(healthy) else "No",
                           "vocabulary": ["Yes", "No"],
                           "source_field": "eligibilityModule.healthyVolunteers"})
        if len(fields) < n_fields:
            drop("fewer_than_n_fields_available")
            continue

        rng = random.Random(f"{seed}:{nct}:claim_set")
        chosen = fields[:n_fields]
        # the swapped field rotates, so no single field carries the answer
        swapped_index = len(swapped_counter) % n_fields
        swapped_counter[chosen[swapped_index]["name"]] = (
            swapped_counter.get(chosen[swapped_index]["name"], 0) + 1
        )
        swapped = chosen[swapped_index]
        # a near-miss swap: prefer another value the record itself carries elsewhere
        candidates = [
            value for value in swapped.get("other_record_values", [])
            if value != swapped["value"]
        ]
        candidates += [value for value in swapped["vocabulary"] if value != swapped["value"]]
        candidates = [c for c in candidates if c != swapped["value"]]
        if not candidates:
            drop("no_alternative_value_for_the_swapped_field")
            continue
        swap_value = candidates[rng.randrange(len(candidates))]

        stated = []
        for index, field in enumerate(chosen):
            value = swap_value if index == swapped_index else field["value"]
            stated.append((field["name"], value))
        rng.shuffle(stated)
        options = [
            {"key": chr(ord("A") + i), "label": f"{name}: {value}"}
            for i, (name, value) in enumerate(stated)
        ]
        gold_label = f"{swapped['name']}: {swap_value}"
        gold = next(o["key"] for o in options if o["label"] == gold_label)
        split = split_by_record_hash(str(nct), dev_fraction=0.2, salt="clinicaltrials")
        items.append(
            make_item(
                tier=Tier.FRESH,
                source=CT_SOURCE,
                source_record_id=str(nct),
                source_url=CT_URL.format(nct=nct),
                source_license=CT_LICENSE,
                record_date=record_date,
                split=split,
                template_id=template_id,
                skill="consistency",
                qtype=QuestionType.CHOICE,
                state=(
                    f"{state}\n\nStated fields for this record:\n"
                    + "\n".join(f"- {name}: {value}" for name, value in stated)
                ),
                question="Which one of these stated fields is not supported by the record?",
                options=options,
                gold=gold,
                option_order_seed=seed,
                meta={
                    "source_field": swapped["source_field"],
                    "swapped_field": swapped["name"],
                    "true_value": swapped["value"],
                    "stated_value": swap_value,
                    "stated_fields": [{"name": n, "value": v} for n, v in stated],
                    "swap_is_another_value_of_this_record": swap_value
                    in (swapped.get("other_record_values") or []),
                    "first_posted": record_date.isoformat(),
                },
            )
        )

    notes.append(
        f"multi-field variant: {n_fields} stated fields, exactly one swapped; the swapped field "
        f"rotates (counts {dict(sorted(swapped_counter.items()))})"
    )
    notes.append(
        "for the intervention-type field the swapped value is preferably another intervention's "
        "type in the same record, so the claimed value is in the state in a different role"
    )
    return items, drops, notes


def build_fda_route_claim_items(
    labels: list[dict[str, Any]],
    *,
    cfg: dict[str, Any],
    window_start: date,
    window_end: date,
    template_id: str = "fda_route_claim_noul_v1",
    set_ids_seen_before: set[str] | None = None,
) -> tuple[list[Any], dict[str, int], list[str]]:
    """Claim: ``the route of administration of this drug is <ROUTE>.``

    Gold is ``openfda.route``. An **unsupported** claim names a route that also occurs verbatim
    in the label text but is not in the structured field — so the claim's value is present in
    the state in a different role, and "supported iff the value appears" cannot be right more
    than half the time by construction. A label with no such alternative route is dropped with
    a counted reason rather than paired with an unseen word.
    """
    from meddecide.bench.fresh.openfda import _effective_date, _first
    from meddecide.bench.fresh.openfda import _state_text as fda_state

    seed = int(cfg.get("seed", 0))
    drops: dict[str, int] = {}
    notes: list[str] = []

    def drop(reason: str, n: int = 1) -> None:
        drops[reason] = drops.get(reason, 0) + n

    seen_before = set_ids_seen_before or set()
    items: list[Any] = []
    n_supported = 0
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
        routes = [
            str(r).strip().upper()
            for r in (label.get("openfda", {}).get("route") or label.get("route") or [])
            if str(r).strip()
        ]
        routes = [r for r in routes if r in ROUTE_WORDS]
        if not routes:
            drop("no_route_in_vocabulary")
            continue
        state = fda_state(label, exclude=())
        if len(state) < 80:
            drop("state_too_short")
            continue
        upper_state = state.upper()
        truth = routes[0]
        rng = random.Random(f"{seed}:{set_id}:route_claim")
        # A stem match, deliberately generous: a label that says "given intravenously" is
        # treated as containing the route word INTRAVENOUS. The string-presence baseline is
        # then measured with the same generous rule, so the template is not credited with a
        # defence it does not have.
        alternatives = [
            word for word in ROUTE_WORDS
            if word not in routes and _route_stem_in(upper_state, word)
        ]
        want_supported = (n_supported * 2) <= len(items)
        if want_supported:
            claimed = truth
        else:
            if not alternatives:
                drop("no_alternative_route_word_in_the_state_text")
                continue
            claimed = alternatives[rng.randrange(len(alternatives))]
        n_supported += int(want_supported)
        split = split_by_record_hash(set_id, dev_fraction=0.2, salt="openfda")
        items.append(
            make_item(
                tier=Tier.FRESH,
                source=FDA_SOURCE,
                source_record_id=set_id,
                source_url=FDA_URL.format(set_id=set_id),
                source_license=FDA_LICENSE,
                record_date=effective,
                split=split,
                template_id=template_id,
                skill="consistency",
                qtype=QuestionType.NOUL,
                state=state,
                question=(
                    f"Does this label support the claim that the route of administration of "
                    f"this drug is {claimed}?"
                ),
                options=[{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}],
                gold="yes" if want_supported else "no",
                option_order_seed=seed,
                meta={
                    "source_field": "openfda.route",
                    "claimed_route": claimed,
                    "supported": want_supported,
                    "routes": routes,
                    "alternative_route_words_in_state": alternatives,
                    "role_rule": (
                        "claimed route is the structured route value"
                        if want_supported
                        else "claimed route is a route word that occurs in the label text but "
                             "is not the structured route value"
                    ),
                    "effective_time": effective.strftime("%Y%m%d"),
                },
            )
        )

    notes.append(
        f"claim = 'the route of administration is <ROUTE>'; {n_supported} supported of "
        f"{len(items)}; an unsupported claim's route word always occurs verbatim in the label "
        "text, so a string-presence rule cannot separate the classes"
    )
    return items, drops, notes
