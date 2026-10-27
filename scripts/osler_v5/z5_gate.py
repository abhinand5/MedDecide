"""osler_v5 Z5 gate (D49 = Gate O4, unchanged by D55): the four checks of the final model against zero-shot pplx-decider-flash-v1.2 and
against MedDecider-4B, reusing the osler_v4 gate code (scripts/osler_v4/y9_gate.py: its item loader, its baseline loaders and its check
runner; meddecide.eval.gate_o4 for the bootstrap and the verdict). Nothing in the rules is changed here.

Two modes:
- ``--validate``: the osler_v4 final's own Y9 predictions are the "final". The wiring must reproduce osler_v4's published Gate O4 figures
  (C220, C221): against zero-shot pplx, MR +0.1387, MIMIC +0.1654, panel -0.0421 (FAIL), knowledge -0.0067; against MedDecider-4B, MR
  +0.2449, MIMIC +0.1651 on eight templates, panel NOT MEASURED (4.87 % unpaired). Writes outputs/osler_v5/Z5/validate_gate.json.
- default: the osler_v5 final's predictions under outputs/osler_v5/Z5/<prefix>_<set>/predictions.jsonl (written by z5_score.py) are the
  "final". Writes outputs/osler_v5/Z5/gate.json.

Usage (CPU):
    uv run --frozen python scripts/osler_v5/z5_gate.py --validate
    uv run --frozen python scripts/osler_v5/z5_gate.py --prefix v5blend
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

from meddecide.utils.io import iter_jsonl, write_json

REPO = Path(__file__).resolve().parents[2]
Z5 = REPO / "outputs" / "osler_v5" / "Z5"
Y9_PATH = REPO / "scripts" / "osler_v4" / "y9_gate.py"
Y9_FINAL = REPO / "outputs" / "osler_v4" / "Y9" / "baselines"
GATE_SETS = ("mr_verdict", "mimic_suite", "panel", "knowledge")
SETS = ("mr_verdict", "mr_reported", "mr_heldout_test", "mimic_suite", "mimic_heldout", "panel", "knowledge", "mednli_test", "pairs_test")
BASELINES = ("pplx_flash", "meddecider_4b")


def load_y9() -> Any:
    spec = importlib.util.spec_from_file_location("y9_gate", Y9_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def final_rows(y9: Any, items: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    """The final's normalised rows on one set, from a prediction file keyed by the recomputed item id (as the Y9 runners write it)."""
    by_id = {str(i["item_id"]): i for i in items}
    rows: list[dict[str, Any]] = []
    skipped = 0
    for raw in iter_jsonl(path):
        item = by_id.get(str(raw["item_id"]))
        if item is None:
            raise ValueError(f"{path}: a prediction for an item outside the set")
        if raw.get("status", "scored") != "scored":
            skipped += 1
            continue
        rows.append(y9.normalise_row(raw, item))
    if len(rows) + skipped != len(items):
        raise ValueError(f"{path}: {len(rows)} scored and {skipped} skipped rows for {len(items)} items")
    return {"rows": rows, "path": str(path.relative_to(REPO)), "skipped_by_runner": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--validate", action="store_true", help="use the osler_v4 final's Y9 predictions as the final")
    parser.add_argument("--prefix", default="v5blend", help="the osler_v5 final's run prefix under outputs/osler_v5/Z5")
    args = parser.parse_args()
    y9 = load_y9()
    items_by_set = {name: y9.load_items(name) for name in SETS}
    final_by_set: dict[str, Any] = {}
    for name in SETS:
        if args.validate:
            path = Y9_FINAL / f"final_{name}" / "predictions.jsonl"
        else:
            path = Z5 / f"{args.prefix}_{name}" / "predictions.jsonl"
        if path.exists():
            final_by_set[name] = final_rows(y9, items_by_set[name], path)
    results: dict[str, Any] = {}
    for base in BASELINES:
        base_by_set = {name: y9.load_scored(base, name, items_by_set[name]) for name in SETS}
        results[base] = y9.run_checks("final", base, final_by_set, base_by_set, items_by_set)
    out = {"mode": "validate" if args.validate else "final", "prefix": None if args.validate else args.prefix,
           "sets_scored": sorted(final_by_set), "results": results}
    write_json(Z5 / ("validate_gate.json" if args.validate else "gate.json"), out)
    for base in BASELINES:
        verdict = results[base]["verdict"]
        checks = results[base]["checks"]
        print(f"{base}: verdict {verdict}", flush=True)
        for key, check in checks.items():
            print(f"  ({key}) {check['check']}: {check['verdict']}  acc {check['accuracy']}  brier {check['brier']}  "
                  f"reason {check['reason'] if check['verdict'] != 'PASS' else ''}", flush=True)


if __name__ == "__main__":
    main()
