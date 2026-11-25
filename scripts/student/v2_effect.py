#!/usr/bin/env python
"""V2: the padding bug's effect on student_v0 step 1,000, test split (v0.2).

Four scorings of the same test items, all with the checkpoint's stored calibration:

  old        student_v0 run: original code, batch 16              outputs/student_v0/S9_run3
  fixed16    fixed code (position ids, chunk-rounded rows), batch 16   outputs/student_v1/V2/fixed_b16
  fixed1     fixed code, batch 1, rows rounded to the chunk          outputs/student_v1/V2/fixed_b1
  unpadded1  batch 1, no padding at all: the V2 evaluation path       outputs/student_v1/V2/unpadded_b1

Comparisons, always as A - B:
  unpadded1 - old     the adopted path against the original batched path (headline)
  fixed16 - old       the code fix alone, at equal batching
  unpadded1 - fixed16 the adopted path against batched fixed-code scoring
  fixed1 - unpadded1  the chunk rounding alone, at batch 1

Pure metric code is in ``meddecide.eval.paired``; this script only reads, joins and writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.paired import (  # noqa: E402
    item_brier,
    paired_difference,
    stratified_macro_difference,
)

PREDS = "preds_meddecide-0p8b-lora-pointer__test.jsonl"
SCORINGS = {
    "old": ROOT / "outputs" / "student_v0" / "S9_run3",
    "fixed16": ROOT / "outputs" / "student_v1" / "V2" / "fixed_b16",
    "fixed1": ROOT / "outputs" / "student_v1" / "V2" / "fixed_b1",
    "unpadded1": ROOT / "outputs" / "student_v1" / "V2" / "unpadded_b1",
}
COMPARISONS = [("unpadded1", "old"), ("fixed16", "old"), ("unpadded1", "fixed16"), ("fixed1", "unpadded1")]
N_RESAMPLES = 2000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["item_id"] in rows:
                raise ValueError(f"{path}: duplicate item_id {row['item_id']}")
            rows[row["item_id"]] = row
    return rows


def per_item(rows: dict[str, dict[str, Any]], order: list[str]) -> dict[str, Any]:
    """Arrays aligned to ``order``: correctness, Brier, argmax, probabilities, template."""
    correct, brier, argmax, probs, template = [], [], [], [], []
    for item_id in order:
        row = rows[item_id]
        keys = row["option_keys"]
        gold_index = keys.index(row["gold_key"])
        recomputed = keys[int(np.argmax(row["option_probs"]))]
        if recomputed != row["argmax_key"]:
            raise ValueError(f"{item_id}: stored argmax_key disagrees with option_probs")
        correct.append(1.0 if row["correct"] else 0.0)
        brier.append(item_brier(row["option_probs"], gold_index))
        argmax.append(row["argmax_key"])
        probs.append(np.asarray(row["option_probs"], dtype=np.float64))
        template.append(row["template_id"])
    return {"correct": correct, "brier": brier, "argmax": argmax, "probs": probs, "template": template}


def summarise_scoring(data: dict[str, Any]) -> dict[str, Any]:
    by_template: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[int]] = defaultdict(list)
    for index, template in enumerate(data["template"]):
        groups[template].append(index)
    for template in sorted(groups):
        idx = groups[template]
        by_template[template] = {
            "n": len(idx),
            "n_correct": int(sum(data["correct"][i] for i in idx)),
            "accuracy": float(np.mean([data["correct"][i] for i in idx])),
            "brier": float(np.mean([data["brier"][i] for i in idx])),
        }
    n = len(data["correct"])
    return {
        "n": n,
        "n_correct": int(sum(data["correct"])),
        "micro_accuracy": float(np.mean(data["correct"])),
        "macro_accuracy": float(np.mean([v["accuracy"] for v in by_template.values()])),
        "micro_brier": float(np.mean(data["brier"])),
        "by_template": by_template,
    }


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Paired A - B comparison: overall, per template, and prediction changes."""
    template = a["template"]
    assert template == b["template"], "scorings must cover the same items in the same order"
    groups = defaultdict(list)
    for index, name in enumerate(template):
        groups[name].append(index)

    def pick(data: dict[str, Any], key: str, idx: list[int]) -> list[float]:
        return [data[key][i] for i in idx]

    per_template = {}
    acc_groups_a, acc_groups_b, brier_groups_a, brier_groups_b = {}, {}, {}, {}
    for name in sorted(groups):
        idx = groups[name]
        acc_a, acc_b = pick(a, "correct", idx), pick(b, "correct", idx)
        br_a, br_b = pick(a, "brier", idx), pick(b, "brier", idx)
        acc_groups_a[name], acc_groups_b[name] = acc_a, acc_b
        brier_groups_a[name], brier_groups_b[name] = br_a, br_b
        acc_diff = paired_difference(acc_a, acc_b, n_resamples=N_RESAMPLES)
        brier_diff = paired_difference(br_a, br_b, n_resamples=N_RESAMPLES)
        max_dp = max(
            float(np.max(np.abs(a["probs"][i] - b["probs"][i]))) for i in idx
        )
        per_template[name] = {
            "n": len(idx),
            "accuracy_diff": acc_diff.as_dict(),
            "brier_diff": brier_diff.as_dict(),
            "argmax_changed": int(sum(a["argmax"][i] != b["argmax"][i] for i in idx)),
            "correct_to_wrong": int(sum(1 for i in idx if a["correct"][i] == 0 and b["correct"][i] == 1)),
            "wrong_to_correct": int(sum(1 for i in idx if a["correct"][i] == 1 and b["correct"][i] == 0)),
            "max_abs_prob_diff": max_dp,
        }
    all_idx = list(range(len(template)))
    overall_acc = paired_difference(
        pick(a, "correct", all_idx), pick(b, "correct", all_idx), n_resamples=N_RESAMPLES
    )
    overall_brier = paired_difference(
        pick(a, "brier", all_idx), pick(b, "brier", all_idx), n_resamples=N_RESAMPLES
    )
    macro_acc = stratified_macro_difference(acc_groups_a, acc_groups_b, n_resamples=N_RESAMPLES)
    macro_brier = stratified_macro_difference(brier_groups_a, brier_groups_b, n_resamples=N_RESAMPLES)
    worst = max(per_template, key=lambda k: abs(per_template[k]["accuracy_diff"]["point"]))
    return {
        "n": len(template),
        "micro_accuracy_diff": overall_acc.as_dict(),
        "macro_accuracy_diff": macro_acc.as_dict(),
        "micro_brier_diff": overall_brier.as_dict(),
        "macro_brier_diff": macro_brier.as_dict(),
        "argmax_changed_total": int(sum(per_template[k]["argmax_changed"] for k in per_template)),
        "max_abs_template_accuracy_diff": {
            "template_id": worst,
            "point": per_template[worst]["accuracy_diff"]["point"],
        },
        "per_template": per_template,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = {name: load_rows(folder / PREDS) for name, folder in SCORINGS.items()}
    sizes = {name: len(r) for name, r in rows.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"scorings have different row counts: {sizes}")
    order = sorted(rows["old"])
    for name, r in rows.items():
        if set(r) != set(order):
            raise ValueError(f"{name} covers different item ids than old")
        for item_id in order:
            if r[item_id]["template_id"] != rows["old"][item_id]["template_id"]:
                raise ValueError(f"{name}: template differs for {item_id}")
            if r[item_id]["option_keys"] != rows["old"][item_id]["option_keys"]:
                raise ValueError(f"{name}: option keys differ for {item_id}")

    data = {name: per_item(r, order) for name, r in rows.items()}
    scorings = {name: summarise_scoring(d) for name, d in data.items()}
    comparisons = {f"{a}-{b}": compare(data[a], data[b]) for a, b in COMPARISONS}
    report = {
        "task": "V2",
        "kind": "padding_effect",
        "split": "test",
        "dataset": "data/bench/v0.2",
        "checkpoint": "outputs/student_v0/S9_run3/best (= checkpoints/step_1000)",
        "dtype": "bfloat16",
        "calibration": "stored checkpoint calibration (identical for all scorings)",
        "n_resamples": N_RESAMPLES,
        "seed": 0,
        "inputs": {
            name: {
                "path": str(folder.relative_to(ROOT) / PREDS),
                "sha256": sha256(folder / PREDS),
                "rows": sizes[name],
            }
            for name, folder in SCORINGS.items()
        },
        "scorings": scorings,
        "comparisons": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, s in scorings.items():
        print(f"{name:8s} n={s['n']} micro_acc={s['micro_accuracy']:.4f} "
              f"macro_acc={s['macro_accuracy']:.4f} micro_brier={s['micro_brier']:.4f}")
    for key, c in comparisons.items():
        acc = c["micro_accuracy_diff"]
        print(f"{key:14s} micro_acc_diff={acc['point']:+.5f} [{acc['ci_lo']:+.5f}, {acc['ci_hi']:+.5f}] "
              f"argmax_changed={c['argmax_changed_total']} "
              f"max_template={c['max_abs_template_accuracy_diff']['template_id']}:"
              f"{c['max_abs_template_accuracy_diff']['point']:+.4f}")


if __name__ == "__main__":
    main()
