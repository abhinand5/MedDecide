#!/usr/bin/env python
"""Score Unsloth raw answers in the main environment and select a checkpoint (arm C, V8).

For each raw answers file (one per checkpoint, written by ``v8_unsloth_predict.py``) this converts the
answers to run_student-format rows (``meddecide.eval.unsloth_predictions``), then computes:

  * the pooled-class macro accuracy (the selection metric of arms A and B: per-class recall over the
    canonical option letters, pooled across templates; canonical letters from ``canonicalise_options``),
  * micro accuracy and the multi-class Brier score.

Selection rule (the S9 rule, unchanged): highest pooled-class macro accuracy; ties (|Δ| <= 1e-12) broken
by lower Brier, then by the earlier step. Writes ``selection.json`` and a per-step table. Nothing here
uses a test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.bench.schema import Item  # noqa: E402
from meddecide.eval.metrics import macro_accuracy  # noqa: E402
from meddecide.eval.paired import item_brier  # noqa: E402
from meddecide.eval.readout import canonicalise_options  # noqa: E402
from meddecide.eval.unsloth_predictions import option_probabilities  # noqa: E402


def load_items(path: Path) -> dict[str, Item]:
    rows = {}
    for line in path.open(encoding="utf-8"):
        item = Item.model_validate_json(line)
        rows[item.item_id] = item
    return rows


def score_file(raw_path: Path, items: dict[str, Item]) -> dict:
    pooled_pred, pooled_gold, correct, briers = [], [], [], []
    for line in raw_path.open(encoding="utf-8"):
        raw = json.loads(line)
        item = items[raw["item_id"]]
        canonical, _original = canonicalise_options(item)
        keys = [o.key for o in canonical.options]
        probs = option_probabilities(item.qtype.value, raw["probabilities"], [str(o.key) for o in item.options])
        gold_letter = canonical.gold
        pred_letter = keys[int(np.argmax(probs))]
        pooled_pred.append(pred_letter)
        pooled_gold.append(gold_letter)
        correct.append(pred_letter == gold_letter)
        briers.append(item_brier(probs, keys.index(gold_letter)))
    n = len(correct)
    return {
        "n": n,
        "micro_accuracy": float(np.mean(correct)) if n else None,
        "pooled_class_macro": macro_accuracy(pooled_pred, pooled_gold) if n else None,
        "micro_brier": float(np.mean(briers)) if n else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="folder of step_<n>.jsonl raw answers")
    parser.add_argument("--items", type=Path, required=True, help="the dev items (MedDecide JSONL)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    items = load_items(args.items)
    table = []
    for path in sorted(args.raw_dir.glob("step_*.jsonl"), key=lambda p: int(p.stem.split("_")[1])):
        step = int(path.stem.split("_")[1])
        metrics = score_file(path, items)
        table.append({"step": step, **metrics})
    if not table:
        raise SystemExit("no raw answer files found")
    best = None
    for row in table:
        if best is None:
            best = row
            continue
        delta = row["pooled_class_macro"] - best["pooled_class_macro"]
        tied = abs(delta) <= 1e-12
        if delta > 1e-12 or (tied and row["micro_brier"] < best["micro_brier"] - 1e-12):
            best = row  # a tie keeps the earlier step unless the Brier score is lower
    result = {
        "kind": "arm_c_selection",
        "rule": "highest pooled-class macro accuracy on the V4 dev set; ties (|dmacro| <= 1e-12) broken by "
                "lower Brier, then by the earlier step",
        "dev_items": len(items),
        "selected_step": best["step"],
        "selected": best,
        "table": table,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_step": best["step"], "pooled_class_macro": best["pooled_class_macro"],
                      "steps_scored": len(table)}))


if __name__ == "__main__":
    main()
