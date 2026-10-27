"""osler_v5 D56a (ADVISORY §8.2): build the corrected mix on the CPU and report it. The osler_v4 final mix is composed in memory with the
Y8 composer (its composition hash must equal 9d25e30f…), then every row of the templates in DROP_ALLOWED is dropped whole. The script
reports the rows dropped and remaining, the new composition hash, the difference from C048's 155 rows, and checks that no other
template's row count changed. Nothing is written to the mix; the report goes to outputs/osler_v5/Z0/corrected_mix.json.

Usage (CPU):
    uv run --frozen python scripts/osler_v5/build_corrected_mix.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from meddecide.train.corrected_mix import DROP_ALLOWED, drop_template_rows, template_counts
from meddecide.train.final_mix import compose_final_mix, composition_sha256
from meddecide.utils.io import write_json

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v5" / "Z0" / "corrected_mix.json"
ORIGINAL_HASH = "9d25e30f53d7823d051b9c71334c248660318c86ce904c26f855b9a9dbc618b8"
C048_ROWS = 155  # the fda_route_choice_v1 rows with 17 or 18 options (C048)


def _y8() -> Any:
    spec = importlib.util.spec_from_file_location("y8_final_mix", REPO / "scripts" / "osler_v4" / "y8_final_mix.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    y8 = _y8()
    decisions = {name: y8.entering(name) for name in ("B1", "B2", "B3", "B4")}
    family = {name: y8.add_files(name) for name in y8.FAMILY if decisions[name][0]}
    other = y8.add_files("B4") if decisions["B4"][0] else []
    rows, composition = compose_final_mix(y8.BASE, family, other, y8.BROAD)
    if composition["composition_sha256"] != ORIGINAL_HASH:
        raise SystemExit(f"the composed mix hash is {composition['composition_sha256']}, not osler_v4's {ORIGINAL_HASH}")
    before = template_counts(rows)
    kept, report = drop_template_rows(rows, DROP_ALLOWED)
    after = template_counts(kept)
    changed = {t: (before[t], after.get(t, 0)) for t in before if before[t] != after.get(t, 0)}
    other_changed = {t: v for t, v in changed.items() if t not in DROP_ALLOWED}
    new_hash = composition_sha256(str(row["item_id"]) for row in kept)
    result: dict[str, Any] = {
        "original_hash": ORIGINAL_HASH, "dropped_templates": list(DROP_ALLOWED), "rows_before": report.rows_before,
        "rows_dropped": report.rows_dropped, "rows_remaining": report.rows_remaining, "dropped_by_template": report.dropped_by_template,
        "composition_sha256_corrected": new_hash, "difference_from_C048_rows": report.rows_dropped - C048_ROWS,
        "other_templates_changed": other_changed, "templates_before": len(before), "templates_after": len(after),
        "replay_rows": composition["replay_rows"], "replay_share_after": composition["replay_rows"] / report.rows_remaining,
        "family_kept": composition["family_kept"], "family_share_after": composition["family_kept"] / report.rows_remaining,
        "family_cap": 0.35, "family_cap_exceeded_by_rows": composition["family_kept"] - int(0.35 * report.rows_remaining),
        "largest_template_after": max(after, key=lambda t: after[t]), "largest_template_share_after": max(after.values()) / report.rows_remaining,
        "note": "Nothing else in the mix changes (ADVISORY §8.2): the family rows and the replay top-up are kept as composed; the "
                "family share after the drop is reported against the 35 % cap, not re-sampled.",
    }
    write_json(OUT, result)
    print(json.dumps(result, indent=1, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
