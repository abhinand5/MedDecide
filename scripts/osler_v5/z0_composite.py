"""osler_v5 Z0 step 11 (D54; ADVISORY Z0.6): the dev composite S, its parts and the positional health (original order) for the models whose
dev predictions exist, from those predictions only (additional rows; they calibrate expectations and choose nothing). Counts only.

Reused baseline prediction files are checked against the frozen item files by item id before use (osler_v4's rule). A model whose
MR partition has no prediction file is NOT MEASURED for S; nothing is scored here.

Usage (CPU):
    uv run --frozen python scripts/osler_v5/z0_composite.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meddecide.eval.composite import accuracy, composite, positional_health
from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.io import write_json

REPO = Path(__file__).resolve().parents[2]
FASTDEV = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"
CG = REPO / "data" / "bench" / "osler_v4" / "cg_v2.jsonl"
OUT = REPO / "outputs" / "osler_v5" / "Z0" / "composite.json"
PREDICTIONS = {
    "osler_v4_final": {
        "fast": REPO / "outputs/osler_v4/Y8/dev_predictions.jsonl",
        "cg": REPO / "outputs/osler_v4/Y8/cg_v2_predictions.jsonl",
    },
    "A0": {
        "fast": REPO / "outputs/osler_v4/Y6/A0/fastdev_v4_predictions.jsonl",
        "cg": REPO / "outputs/osler_v4/Y6/A0/cg_v2_predictions.jsonl",
    },
    "zero_shot_pplx_flash": {  # the MR partition of fast dev v4 is not in any existing file (NOT MEASURED)
        "fast_mimic": REPO / "outputs/osler_v4/Y4/baselines/fastdev_mimic_flash/predictions.jsonl",
        "cg": REPO / "outputs/osler_v4/Y4/baselines/cg_v2_flash/predictions.jsonl",
    },
}


def item_index(path: Path) -> dict[str, tuple[int, int]]:
    """id -> (n_options, gold position) from the frozen item file, under the id scheme the prediction files use (mix_row_to_item)."""
    index: dict[str, tuple[int, int]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                item = mix_row_to_item(json.loads(line))
                index[item.item_id] = (item.n_options, item.gold_index)
    return index


def read_rows(path: Path, index: dict[str, tuple[int, int]]) -> list[dict[str, Any]]:
    """The rows of a prediction file, with n_options and gold_index from the frozen item file (the baseline files carry their own key
    scheme, which is not always the readout codes). The probabilities are in the item's option order (no permutation at evaluation)."""
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    for row in rows:
        n_options, gold_index = index[str(row["item_id"])]
        if n_options != len(row["probs"]):
            raise ValueError(f"{path.name}: {n_options} options in the item file, {len(row['probs'])} probabilities")
        row["n_options"], row["gold_index"] = n_options, gold_index
    return rows


CG_FILES = {files["cg"] for files in PREDICTIONS.values()}


def main() -> None:
    fast_index, cg_index = item_index(FASTDEV), item_index(CG)
    fast_ids, cg_ids = set(fast_index), set(cg_index)
    result: dict[str, Any] = {"item_files": {"fastdev_v4": {"path": str(FASTDEV.relative_to(REPO)), "items": len(fast_ids)},
                                             "cg_v2": {"path": str(CG.relative_to(REPO)), "items": len(cg_ids)}},
                              "models": {}}
    rows_cache: dict[str, list[dict[str, Any]]] = {}

    def load(path: Path) -> list[dict[str, Any]]:
        key = str(path)
        if key not in rows_cache:
            index = cg_index if "cg" in path.name or path == PREDICTIONS["osler_v4_final"]["cg"] or path in CG_FILES else fast_index
            rows_cache[key] = read_rows(path, index)
        return rows_cache[key]

    for name, files in PREDICTIONS.items():
        entry: dict[str, Any] = {"files": {k: str(v.relative_to(REPO)) for k, v in files.items()}}
        cg_rows = load(files["cg"])
        cg_set = {str(r["item_id"]) for r in cg_rows}
        entry["cg_rows"] = len(cg_rows)
        entry["cg_items_missing_from_file"] = len(cg_ids - cg_set)
        entry["cg_items_not_in_item_file"] = len(cg_set - cg_ids)
        if "fast" in files:
            fast_rows = load(files["fast"])
            fast_set = {str(r["item_id"]) for r in fast_rows}
            if fast_set != fast_ids or len(fast_rows) != len(fast_ids):
                raise SystemExit(f"{name}: the fast dev v4 prediction ids differ from the frozen item file")
            if cg_set != cg_ids or len(cg_rows) != len(cg_ids):
                raise SystemExit(f"{name}: the CG v2 prediction ids differ from the frozen item file")
            parts = composite(fast_rows, cg_rows)
            entry["composite"] = parts.to_dict()
            entry["positional_health_original_order"] = positional_health(cg_rows).to_dict()
        else:
            mimic_rows = load(files["fast_mimic"])
            entry["mimic_rows"] = len(mimic_rows)
            entry["mimic_accuracy"] = accuracy(mimic_rows)
            entry["cg_micro_on_file_rows"] = accuracy(cg_rows)
            entry["composite"] = "NOT MEASURED — the MR partition of fast dev v4 has no prediction file for this model"
            entry["positional_health_original_order"] = positional_health(cg_rows).to_dict()
        result["models"][name] = entry

    # the paired CG v2 micro of the two models on identical items (the zero-shot file has 11 items fewer)
    final_cg = {str(r["item_id"]): r for r in load(PREDICTIONS["osler_v4_final"]["cg"])}
    flash_cg = {str(r["item_id"]): r for r in load(PREDICTIONS["zero_shot_pplx_flash"]["cg"])}
    common = sorted(set(final_cg) & set(flash_cg))
    result["cg_micro_identical_items"] = {
        "items": len(common),
        "osler_v4_final": accuracy([final_cg[i] for i in common]),
        "zero_shot_pplx_flash": accuracy([flash_cg[i] for i in common]),
        "note": "additional; the alpha = 0 reference of D54 is scored on GPU in Z1 with the reference path",
    }
    write_json(OUT, result)
    for name, entry in result["models"].items():
        print(name, json.dumps({k: v for k, v in entry.items() if k != "files"}, sort_keys=True), flush=True)
    print("cg_micro_identical_items", json.dumps(result["cg_micro_identical_items"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
