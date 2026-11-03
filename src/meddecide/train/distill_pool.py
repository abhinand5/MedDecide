"""distill_v0: the training pool (D2/D4) and the teacher-label attachment.

The pool is osler_v4's final mix (Y8 composer; original composition hash checked), with
``fda_route_choice_v1`` dropped whole (D56a; corrected hash checked) and the literal rule-lookup
templates dropped (GOAL §3.3). The training basket is keyed by the normalised text key because
Winogrande twins share an item id while their state text differs (5,232 ids / 27,340 pool rows).

``attach_teacher_probs`` adds the frozen d1-ultra distribution to a row only when the row's template
is on the teacher list, the text key has a label, the template matches and the option keys are in the
same order; it renormalises the distribution to sum to 1 (the API's tolerance is 1e-3, the item
schema's is 1e-5) and reports how many rows needed it and the largest deviation.
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from meddecide.panel.overlap import row_text_key
from meddecide.train.corrected_mix import (
    CORRECTED_COMPOSITION_SHA256,
    DROP_ALLOWED,
    drop_template_rows,
)
from meddecide.train.distill_mix import (
    POOL_ROWS_EXPECTED,
    drop_rule_lookup_rows,
    is_teacher_template,
)
from meddecide.train.final_mix import compose_final_mix, composition_sha256

REPO = Path(__file__).resolve().parents[3]
ORIGINAL_COMPOSITION = "9d25e30f53d7823d051b9c71334c248660318c86ce904c26f855b9a9dbc618b8"


def _y8() -> Any:
    spec = importlib.util.spec_from_file_location("y8_final_mix", REPO / "scripts" / "osler_v4" / "y8_final_mix.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compose_pool() -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """The D2/D4 pool (corrected mix minus the rule-lookup templates), its text keys, and the composition record."""
    y8 = _y8()
    decisions = {name: y8.entering(name) for name in ("B1", "B2", "B3", "B4")}
    family = {name: y8.add_files(name) for name in y8.FAMILY if decisions[name][0]}
    other = y8.add_files("B4") if decisions["B4"][0] else []
    rows, composition = compose_final_mix(y8.BASE, family, other, y8.BROAD)
    if composition["composition_sha256"] != ORIGINAL_COMPOSITION:
        raise SystemExit(f"composed hash {composition['composition_sha256']} is not osler_v4's {ORIGINAL_COMPOSITION}")
    corrected, drop = drop_template_rows(rows, DROP_ALLOWED)
    corrected_hash = composition_sha256(str(row["item_id"]) for row in corrected)
    if corrected_hash != CORRECTED_COMPOSITION_SHA256:
        raise SystemExit(f"corrected hash {corrected_hash} is not the recorded {CORRECTED_COMPOSITION_SHA256}")
    pool, lookup = drop_rule_lookup_rows(corrected)
    keys = [row_text_key(row) for row in pool]
    id_counts = Counter(str(row["item_id"]) for row in pool)
    duplicated = {item_id: count for item_id, count in id_counts.items() if count > 1}
    key_counts = Counter(keys)
    duplicate_keys = {key for key, count in key_counts.items() if count > 1}
    groups: dict[str, list[dict[str, Any]]] = {}
    for key, row in zip(keys, pool, strict=True):
        if key in duplicate_keys:
            groups.setdefault(key, []).append(row)
    conflicting_gold = sum(1 for members in groups.values() if len({str(row["gold"]) for row in members}) > 1)
    conflicting_options = sum(1 for members in groups.values()
                              if len({tuple(str(option["key"]) for option in row["options"]) for row in members}) > 1)
    info = {
        "original_composition_sha256": composition["composition_sha256"],
        "fda_route_choice_drop": drop.to_dict(),
        "corrected_composition_sha256": corrected_hash,
        "rule_lookup_drop": lookup,
        "pool_rows": len(pool),
        "pool_rows_expected": POOL_ROWS_EXPECTED,
        "pool_matches_expected": len(pool) == POOL_ROWS_EXPECTED,
        "pool_item_id_sha256": composition_sha256(str(row["item_id"]) for row in pool),
        "pool_text_sha256": composition_sha256(keys),
        "text_keys_unique": not duplicate_keys,
        "text_key_duplicates": {
            "keys": len(duplicate_keys), "rows": sum(key_counts[key] for key in duplicate_keys),
            "extra_rows": sum(key_counts[key] - 1 for key in duplicate_keys),
            "templates": dict(sorted(Counter(str(row["template_id"]) for members in groups.values()
                                             for row in members).items())),
            "conflicting_gold_keys": conflicting_gold,
            "conflicting_option_keys": conflicting_options,
            "note": "identical state, question and option labels share a text key; one teacher distribution is "
                    "frozen per unique text and attached to every row with that text"},
        "item_id_duplicates": {
            "ids": len(duplicated), "rows": sum(duplicated.values()),
            "templates": dict(sorted(Counter(str(row["template_id"]) for row in pool
                                             if str(row["item_id"]) in duplicated).items())),
            "note": "Winogrande twins share an item id; the pool is keyed by text_key"},
        "family_kept_rows": composition["family_kept"],
        "replay_rows": composition["replay_rows"],
        "entering_blocks": {name: {"entering": decisions[name][0], "reason": decisions[name][1]} for name in decisions},
    }
    return pool, keys, info


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    """The frozen D2 label rows keyed by text key (the file's rows are unique by construction)."""
    labels: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["text_key"])
            if key in labels:
                raise ValueError(f"a text key appears twice in the frozen label file: {key}")
            labels[key] = row
    return labels


def attach_teacher_probs(rows: Sequence[dict[str, Any]], keys: Sequence[str],
                         labels: Mapping[str, Mapping[str, Any]], *,
                         model_id: str) -> dict[str, Any]:
    """Add the teacher distribution to every teacher-template row whose text key has a label (in place)."""
    attached = 0
    missing = 0
    template_mismatch = 0
    order_mismatch = 0
    renormalised = 0
    max_deviation = 0.0
    for row, key in zip(rows, keys, strict=True):
        template = str(row["template_id"])
        if not is_teacher_template(template):
            continue
        label = labels.get(key)
        if label is None:
            missing += 1
            continue
        if str(label["template_id"]) != template:
            template_mismatch += 1
            continue
        option_keys = [str(option["key"]) for option in row["options"]]
        if list(label["option_keys"]) != option_keys:
            order_mismatch += 1
            continue
        probs = [float(value) for value in label["teacher_probs"]]
        total = sum(probs)
        max_deviation = max(max_deviation, abs(total - 1.0))
        if abs(total - 1.0) > 1e-9:
            renormalised += 1
            probs = [value / total for value in probs]
        row["teacher_probs"] = probs
        row["teacher"] = str(label.get("teacher") or model_id)
        attached += 1
    return {"attached": attached, "missing_label": missing, "template_mismatch": template_mismatch,
            "option_order_mismatch": order_mismatch, "renormalised": renormalised,
            "max_abs_sum_deviation": max_deviation, "model_id": model_id}
