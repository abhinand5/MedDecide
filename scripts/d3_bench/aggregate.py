"""d3_bench steps 3-4 (CPU): aggregates of d3-ultra and the comparison models on the same seeded subsamples.

For each set (knowledge, panel, mr_verdict) and model, the rows of the subsample's items are put in the gate_o4 common shape
(argmax over the item's options in the set file's order; the set file is the authority for gold and template), then
meddecide.bench.quick_bench.aggregate gives item micro accuracy, template macro accuracy and the D12/D21 readout gate per
template cell. A comparison model's row counts only when its source file holds the item (unpaired items are counted).
Comparisons are subsample-to-subsample only: d3-ultra minus the model, paired on items, template-macro accuracy with the
gate_o4 bootstrap (1,000 resamples, seed 0, within templates); not a gate.

Sources (all gitignored, read only):
- d3-ultra: outputs/d3_bench/predictions/<set>.jsonl (this task)
- d1-ultra teacher: outputs/distill_v0/D1/labels.jsonl (liquid/d1-ultra-20261015; teacher_probs over option_keys)
- zero-shot pplx-decider-flash-v1.2: outputs/osler_v4/Y9/baselines/pplx_<set>/predictions.jsonl
- zero-shot Clef-Flash: outputs/osler_v5/Z10/clef_flash/<set>.jsonl

Usage:
    uv run --frozen python scripts/d3_bench/aggregate.py --out outputs/d3_bench/aggregate.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from meddecide.bench.quick_bench import aggregate
from meddecide.eval.gate_o4 import comparison, normalise_row, pair_rows
from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.io import iter_jsonl, read_json, write_json

REPO = Path(__file__).resolve().parents[2]
MAIN = REPO if (REPO / "data" / "bench").exists() else REPO.parents[2]  # a git worktree shares data/ and outputs/
Y9 = MAIN / "data" / "bench" / "osler_v4" / "y9"
SETS = ("knowledge", "panel", "mr_verdict")
MODELS = {
    "d1_ultra": "d1-ultra teacher (liquid/d1-ultra-20261015, hosted, distill_v0 D1)",
    "pplx_flash": "zero-shot pplx-decider-flash-v1.2 (osler_v4 Y9)",
    "clef_flash": "zero-shot Clef-Flash (osler_v5 Z10)",
}


def source_path(model: str, set_name: str, d3_dir: Path) -> Path:
    if model == "d3_ultra":
        return d3_dir / f"{set_name}.jsonl"
    if model == "d1_ultra":
        return MAIN / "outputs" / "distill_v0" / "D1" / "labels.jsonl"
    if model == "pplx_flash":
        return MAIN / "outputs" / "osler_v4" / "Y9" / "baselines" / f"pplx_{set_name}" / "predictions.jsonl"
    if model == "clef_flash":
        return MAIN / "outputs" / "osler_v5" / "Z10" / "clef_flash" / f"{set_name}.jsonl"
    raise ValueError(model)


def probs_in_item_order(raw: dict[str, Any], item: dict[str, Any]) -> list[float]:
    keys = [str(o["key"]) for o in item["options"]]
    if "teacher_probs" in raw:
        given = [str(k) for k in raw["option_keys"]]
        if sorted(given) != sorted(keys):
            raise ValueError(f"{item['item_id']}: teacher keys {given} are not the item's {keys}")
        by_key = dict(zip(given, raw["teacher_probs"], strict=True))
        return [float(by_key[k]) for k in keys]
    if raw.get("option_keys") is not None and [str(k) for k in raw["option_keys"]] != keys:
        raise ValueError(f"{item['item_id']}: row option keys are not in the item's order")
    return [float(p) for p in raw["probs"]]


def model_rows(model: str, set_name: str, items: list[dict[str, Any]], d3_dir: Path) -> dict[str, Any]:
    """The subsample's rows of one model, keyed by the set file's id; counts of what was not usable."""
    by_raw = {str(i["item_id"]): i for i in items}
    by_computed = {str(i["computed_id"]): i for i in items}
    rows: dict[str, dict[str, Any]] = {}
    skipped = 0
    key_disagreements = 0
    path = source_path(model, set_name, d3_dir)
    if not path.exists():
        return {"path": str(path), "exists": False, "rows": [], "skipped": 0, "missing": len(items)}
    for raw in iter_jsonl(path):
        item = by_raw.get(str(raw["item_id"])) or by_computed.get(str(raw["item_id"]))
        if item is None:
            continue  # outside the subsample (or another set, for the D1 label file)
        if raw.get("status", "scored") != "scored" or (raw.get("probs") is None and "teacher_probs" not in raw):
            skipped += 1
            continue
        row = normalise_row({"probs": probs_in_item_order(raw, item)}, {**item, "item_id": item["item_id"]})
        if raw.get("predicted_key") is not None and str(raw["predicted_key"]) != row["predicted_key"]:
            key_disagreements += 1
        if row["item_id"] in rows:
            raise ValueError(f"{model} {set_name}: duplicate row for {row['item_id']}")
        rows[row["item_id"]] = row
    return {"path": str(path.relative_to(MAIN)) if path.is_relative_to(MAIN) else str(path), "exists": True,
            "rows": list(rows.values()), "skipped": skipped, "missing": len(items) - len(rows) - skipped,
            "predicted_key_disagreements": key_disagreements}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subsamples", type=Path, default=REPO / "outputs" / "d3_bench" / "subsamples.json")
    parser.add_argument("--d3", type=Path, default=REPO / "outputs" / "d3_bench" / "predictions")
    parser.add_argument("--out", type=Path, default=REPO / "outputs" / "d3_bench" / "aggregate.json")
    args = parser.parse_args()
    plan = read_json(args.subsamples)
    result: dict[str, Any] = {"subsamples_ids_sha256": {s: plan["sets"][s]["ids_sha256"] for s in SETS}, "sets": {}}
    for set_name in SETS:
        wanted = set(plan["sets"][set_name]["item_ids"])
        items = []
        for row in iter_jsonl(Y9 / f"{set_name}.jsonl"):
            if str(row["item_id"]) in wanted:
                items.append({**row, "item_id": str(row["item_id"]), "computed_id": mix_row_to_item(row).item_id})
        if len(items) != len(wanted):
            raise SystemExit(f"{set_name}: {len(items)} items found for {len(wanted)} ids")
        entry: dict[str, Any] = {"k": len(items), "models": {}, "comparisons": {}}
        loaded = {m: model_rows(m, set_name, items, args.d3) for m in ("d3_ultra", *MODELS)}
        for model, data in loaded.items():
            agg = aggregate(data["rows"])
            entry["models"][model] = {**{k: v for k, v in data.items() if k != "rows"}, **agg,
                                      "coverage": len(data["rows"]) / len(items)}
        d3 = loaded["d3_ultra"]["rows"]
        for model, label in MODELS.items():
            other = loaded[model]["rows"]
            if not other or not d3:
                missing = "d3-ultra" if not d3 else label
                entry["comparisons"][model] = {"label": label, "verdict": "NOT MEASURED",
                                               "reason": f"no rows for {missing} on this subsample"}
                continue
            pairing = pair_rows(d3, other, items)
            paired = comparison(pairing)
            entry["comparisons"][model] = {
                "label": label, "scope": "subsample-to-subsample only (not a gate)",
                "paired_items": paired.items, "unpaired": pairing.unpaired,
                "d3_minus_model_template_macro": {"point": paired.accuracy.point, "lower": paired.accuracy.lo,
                                                  "upper": paired.accuracy.hi},
                "d3_minus_model_brier": {"point": paired.brier.point, "lower": paired.brier.lo, "upper": paired.brier.hi},
            }
        result["sets"][set_name] = entry
        for model in ("d3_ultra", *MODELS):
            m = entry["models"][model]
            micro = "-" if m.get("accuracy_micro") is None else f"{m['accuracy_micro']:.4f}"
            macro = "-" if m.get("accuracy_template_macro") is None else f"{m['accuracy_template_macro']:.4f}"
            print(f"{set_name:10s} {model:11s} items={m['items']:5d} micro={micro} macro={macro} gate={m['gate']} "
                  f"skipped={m.get('skipped')} missing={m.get('missing')}", flush=True)
    write_json(args.out, result)


if __name__ == "__main__":
    main()
