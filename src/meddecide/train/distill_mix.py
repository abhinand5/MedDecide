"""distill_v0 D2: the distilled training pool and the teacher basket (GOAL §3.3, §3.5).

The pool is the corrected mix (D56a, ``fda_route_choice_v1`` dropped whole) minus the literal
rule-lookup templates: the FDA field lookups and the PubMed publication-type / check family. Those
rows are metadata lookups, not decisions; they are dropped with counts and reasons, never silently.

The teacher basket T is the pre-registered template list of GOAL §3.3: knowledge, general reasoning
and replay, clinical real text, and the nine clinical-trial templates capped at 10,000 rows each
(drawn by stable-hash order inside each template, so the cap is deterministic). Everything else in
the pool is gold-only G.

Pure functions only; the heavy composition and the API run are ``scripts/distill_v0/d2_basket.py``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from meddecide.utils.hashing import stable_hash

RULE_LOOKUP_TEMPLATES: tuple[str, ...] = (
    "fda_route_claim_noul_v1",
    "fda_class_choice_v1",
    "fda_class_choice_v2",
    "fda_application_family_choice_v1",
    "pubmed_pubtype_choice_v1",
)
RULE_LOOKUP_PREFIX = "pubmed_check_"
RULE_LOOKUP_REASON = "rule_lookup_metadata"
RULE_LOOKUP_EXPECTED: dict[str, int] = {
    "fda_route_claim_noul_v1": 16_612,
    "fda_class_choice_v1": 13_452,
    "fda_class_choice_v2": 9_799,
    "fda_application_family_choice_v1": 18_628,
    "pubmed_pubtype_choice_v1": 42_370,
    RULE_LOOKUP_PREFIX + "*": 86_762,
}
RULE_LOOKUP_EXPECTED_TOTAL = 187_623
POOL_ROWS_EXPECTED = 1_268_815

TEACHER_TEMPLATES: dict[str, int] = {
    # knowledge
    "medmcqa_4opt_v1": 42_370,
    "medqa_usmle4_v1": 9_084,
    "mednli_v1": 10_626,
    # general reasoning / replay
    "broad_aqua_rat_v1": 72_469,
    "snli_nli_choice_v1": 61_680,
    "winogrande_blank_choice_v1": 40_398,
    "hellaswag_ending_choice_v1": 39_905,
    "boolq_passage_noul_v1": 9_427,
    "csqa_choice_v1": 10_765,
    "qasc_science_choice_v1": 9_064,
    "arc_science_choice_v1": 3_370,
    "broad_social_iqa_v1": 24_782,
    "broad_cosmos_qa_v1": 18_727,
    "broad_strategyqa_v1": 1_179,
    "broad_quartz_v1": 1_987,
    "broad_math_qa_v1": 521,
    # clinical real text
    "pmc_similar_patient_v1": 52_549,
    "pubhealth_claim_true_noul_v1": 8_300,
}
BIGBENCH_PREFIX = "broad_bigbench_"
BIGBENCH_EXPECTED = 1_286
CT_PREFIX = "ct_"
CT_TEACHER_CAP = 10_000
CT_TEMPLATES_EXPECTED = 9
CT_ROWS_EXPECTED = 90_000
TEACHER_ROWS_EXPECTED = 508_516
MIMIC_TEMPLATE_PREFIX = "mimic_"


def is_rule_lookup(template_id: str) -> bool:
    return template_id in RULE_LOOKUP_TEMPLATES or template_id.startswith(RULE_LOOKUP_PREFIX)


def drop_rule_lookup_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """The pool rows (every rule-lookup row dropped whole), with the counts and reasons of the drop."""
    kept: list[Mapping[str, Any]] = []
    dropped: Counter[str] = Counter()
    for row in rows:
        template = str(row["template_id"])
        if is_rule_lookup(template):
            dropped[template] += 1
        else:
            kept.append(row)
    named = {name: dropped.get(name, 0) for name in RULE_LOOKUP_TEMPLATES}
    prefixed = sum(count for template, count in dropped.items() if template.startswith(RULE_LOOKUP_PREFIX))
    expected = {**{name: RULE_LOOKUP_EXPECTED[name] for name in RULE_LOOKUP_TEMPLATES},
                RULE_LOOKUP_PREFIX + "*": RULE_LOOKUP_EXPECTED[RULE_LOOKUP_PREFIX + "*"]}
    realised = {**named, RULE_LOOKUP_PREFIX + "*": prefixed}
    report = {
        "reason": RULE_LOOKUP_REASON,
        "rows_before": len(rows),
        "dropped_total": sum(dropped.values()),
        "rows_remaining": len(kept),
        "dropped_by_template": dict(sorted(dropped.items())),
        "expected": expected,
        "realised": realised,
        "matches_expected": realised == expected and sum(dropped.values()) == RULE_LOOKUP_EXPECTED_TOTAL,
        "expected_total": RULE_LOOKUP_EXPECTED_TOTAL,
        "templates_dropped": len(dropped),
    }
    return kept, report


def is_teacher_template(template_id: str) -> bool:
    return template_id in TEACHER_TEMPLATES or template_id.startswith(BIGBENCH_PREFIX) or template_id.startswith(CT_PREFIX)


def teacher_priority(template_id: str) -> int:
    """ADVISORY §1.2 priority: P0 knowledge/general/replay, P1 clinical real text, P2 clinical trials."""
    if template_id.startswith(CT_PREFIX):
        return 2
    if template_id in {"pmc_similar_patient_v1", "pubhealth_claim_true_noul_v1"}:
        return 1
    return 0


def select_teacher_rows(rows: Sequence[Mapping[str, Any]], seed: int = 0) -> tuple[list[int], dict[str, Any]]:
    """The T basket as row indices: every listed template, the ``broad_bigbench_*`` rows, and 10,000 rows per CT template.

    The CT cap is applied inside each template by ascending stable hash of ``{seed, tag, item_id}``; it
    is deterministic and independent of the pool order. Indices are returned in label order
    (ADVISORY §1.2 priority P0/P1/P2, then item id, then pool position) so the run is reproducible.
    Indices rather than rows because Winogrande twins share an item id: a text key, not an id, is the
    unique key of a row (see ``basket_shares``).
    """
    named: Counter[str] = Counter()
    bigbench: list[int] = []
    ct_rows: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        template = str(row["template_id"])
        if template in TEACHER_TEMPLATES:
            named[template] += 1
        elif template.startswith(BIGBENCH_PREFIX):
            bigbench.append(index)
        elif template.startswith(CT_PREFIX):
            ct_rows.setdefault(template, []).append(index)
    selected: list[int] = [index for index, row in enumerate(rows)
                           if str(row["template_id"]) in TEACHER_TEMPLATES or str(row["template_id"]).startswith(BIGBENCH_PREFIX)]
    ct_selected: dict[str, int] = {}
    for template in sorted(ct_rows):
        ranked = sorted(ct_rows[template], key=lambda index: stable_hash(
            {"seed": seed, "tag": "d2_ct_cap", "item_id": str(rows[index]["item_id"])}, length=16))
        kept = ranked[:CT_TEACHER_CAP]
        ct_selected[template] = len(kept)
        selected.extend(kept)
    selected.sort(key=lambda index: (teacher_priority(str(rows[index]["template_id"])),
                                     str(rows[index]["item_id"]), index))
    per_template: dict[str, dict[str, Any]] = {}
    for template in sorted(TEACHER_TEMPLATES):
        per_template[template] = {"available": named.get(template, 0), "selected": named.get(template, 0),
                                  "expected": TEACHER_TEMPLATES[template]}
    per_template[BIGBENCH_PREFIX + "*"] = {"available": len(bigbench), "selected": len(bigbench),
                                           "expected": BIGBENCH_EXPECTED}
    for template in sorted(ct_rows):
        per_template[template] = {"available": len(ct_rows[template]), "selected": ct_selected[template],
                                  "expected": CT_TEACHER_CAP}
    missing = [template for template in TEACHER_TEMPLATES if named.get(template, 0) == 0]
    counts_match = (all(named.get(template, 0) == expected for template, expected in TEACHER_TEMPLATES.items())
                    and len(bigbench) == BIGBENCH_EXPECTED and len(ct_rows) == CT_TEMPLATES_EXPECTED
                    and sum(ct_selected.values()) == CT_ROWS_EXPECTED)
    report = {
        "selected_total": len(selected),
        "expected_total": TEACHER_ROWS_EXPECTED,
        "per_template": per_template,
        "missing_templates": missing,
        "ct_templates": len(ct_rows),
        "ct_cap": CT_TEACHER_CAP,
        "ct_selected": sum(ct_selected.values()),
        "matches_expected": counts_match and len(selected) == TEACHER_ROWS_EXPECTED,
        "counts_match_expected": counts_match,
    }
    return selected, report


def basket_shares(pool: Sequence[Mapping[str, Any]], teacher_keys: set[str]) -> dict[str, Any]:
    """Counts and shares of the pool split, keyed by the normalised text key (Winogrande twins share an item id)."""
    from meddecide.panel.overlap import row_text_key

    teacher = 0
    mimic = 0
    other = 0
    total = 0
    for row in pool:
        total += 1
        if row_text_key(row) in teacher_keys:
            teacher += 1
        elif str(row["template_id"]).startswith(MIMIC_TEMPLATE_PREFIX):
            mimic += 1
        else:
            other += 1
    return {
        "pool_rows": total,
        "teacher_rows": teacher,
        "teacher_share": teacher / total if total else None,
        "mimic_rows": mimic,
        "mimic_share": mimic / total if total else None,
        "other_gold_rows": other,
        "other_gold_share": other / total if total else None,
    }


def estimate_tokens(rows: Iterable[Mapping[str, Any]], chars_per_token: float = 3.8) -> dict[str, Any]:
    """A pre-run cost estimate from the row text lengths (the realised usage is the ledger's, not this)."""
    chars = 0
    items = 0
    for row in rows:
        items += 1
        chars += len(str(row.get("state") or "")) + len(str(row.get("question") or ""))
        chars += sum(len(str(option.get("label") or "")) for option in row.get("options") or [])
    tokens = chars / chars_per_token
    return {"items": items, "characters": chars, "chars_per_token": chars_per_token,
            "estimated_input_tokens": int(tokens),
            "estimated_cost_usd": round(tokens * 0.042 / 1_000_000, 4)}
