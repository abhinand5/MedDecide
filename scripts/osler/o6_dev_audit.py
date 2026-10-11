"""Self-audit of an O6-O8 arm on the dev split (ADVISORY R6; CPU only; reads dev predictions, runs no model).

Re-derives the arm's full-dev headline numbers from its ``dev_predictions.jsonl`` in this fresh process and compares them with
``arm_result.json``; checks the denominators; checks for a constant answer; reports accuracy by source and question type; and
checks the generated items' minimal-pair twins (both twins right, the gold flips within a pair, the same letter for both).
Writes ``outputs/osler_v0/O6/<arm>/dev_audit.json`` (gitignored); the committed record is the CLAIMS rows.

Usage:
    uv run --frozen python scripts/osler/o6_dev_audit.py --run-dir outputs/osler_v0/O6/arm_L
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from meddecide.eval.metrics import template_macro_accuracy
from meddecide.eval.paired import item_brier
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.io import write_json

REPO = Path(__file__).resolve().parents[2]
REFERENCE_CONFIG = REPO / "configs/osler_v0/arm_L.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO / args.run_dir
    rows = [json.loads(line) for line in (run_dir / "dev_predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    result = json.loads((run_dir / "arm_result.json").read_text(encoding="utf-8"))
    fd = result["full_dev_metrics"]
    config = load_config(REFERENCE_CONFIG)
    dev_rows = [json.loads(line) for line in (REPO / config.dev_path).read_text(encoding="utf-8").splitlines()
                if line.strip()]

    correct = np.array([float(r["correct"]) for r in rows])
    templates = np.array([r["template_id"] for r in rows], dtype=object)
    per_letter: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        per_letter[r["gold_key"]].append(float(r["correct"]))
    recomputed = {
        "n": len(rows),
        "n_correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "template_macro_accuracy": float(template_macro_accuracy(correct, templates)),
        "macro_accuracy": float(np.mean([np.mean(v) for v in per_letter.values()])),
        "brier": float(np.mean([item_brier(r["probs"], r["gold_index"]) for r in rows])),
    }
    arm = {k: (fd[k] if k != "n_correct" else fd["n_correct"]) for k in recomputed}
    predicted = collections.Counter(r["predicted_key"] for r in rows)
    gold_items = collections.Counter(r["gold_key"] for r in rows)

    by_source: dict[str, list[float]] = collections.defaultdict(list)
    by_qtype: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(float(r["correct"]))
        by_qtype[r["qtype"]].append(float(r["correct"]))

    # minimal-pair twins of the generated items (ADVISORY O3): both twins right, the gold flips, the same letter for both
    meta: dict[str, dict] = {}
    gold: dict[str, str] = {}
    for row in dev_rows:
        if row.get("source") != "generated":
            continue
        item = mix_row_to_item(row)
        meta[item.item_id] = row["meta"]
        gold[item.item_id] = row["gold"]
    predictions = {r["item_id"]: r for r in rows if r["source"] == "generated"}
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for item_id, m in meta.items():
        groups[m["pair_id"]].append(item_id)
    pairs = [ids for ids in groups.values() if len(ids) == 2 and all(i in predictions for i in ids)]
    twins = {
        "generated_items_with_predictions": sum(1 for i in meta if i in predictions),
        "pairs": len(pairs),
        "both_correct": sum(1 for a, b in pairs if predictions[a]["correct"] and predictions[b]["correct"]),
        "both_wrong": sum(1 for a, b in pairs if not predictions[a]["correct"] and not predictions[b]["correct"]),
        "gold_differs_within_pair": sum(1 for a, b in pairs if gold[a] != gold[b]),
        "same_letter_for_both_twins": sum(1 for a, b in pairs
                                          if predictions[a]["predicted_key"] == predictions[b]["predicted_key"]),
    }

    audit = {
        "run_dir": str(run_dir.relative_to(REPO)),
        "denominators": {"dev_predictions_rows": len(rows), "unique_items": len({r["item_id"] for r in rows}),
                         "arm_full_dev_items": result["full_dev_items"], "dev_split_rows": len(dev_rows)},
        "recomputed": recomputed,
        "arm_result": arm,
        "matches": {k: bool(abs(recomputed[k] - arm[k]) < 1e-9) for k in recomputed},
        "majority_baseline": float(max(gold_items.values()) / len(rows)),
        "most_frequent_predicted_share": max(predicted.values()) / len(rows),
        "accuracy_by_source": {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in by_source.items()},
        "accuracy_by_qtype": {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in by_qtype.items()},
        "generated_twins": twins,
        "tier1_trajectory": [{"step": t["step"], **{k: round(v, 4) for k, v in t["accuracy_by_source"].items()}}
                             for t in result["tier1_trajectory"]],
    }
    write_json(run_dir / "dev_audit.json", audit)
    print(json.dumps({k: audit[k] for k in ("denominators", "matches", "majority_baseline",
                                            "most_frequent_predicted_share", "generated_twins")}, indent=1))
    print("recomputed:", {k: round(v, 6) if isinstance(v, float) else v for k, v in recomputed.items()})
    print("accuracy by source (n >= 200):",
          {k: (v["n"], round(v["accuracy"], 4)) for k, v in audit["accuracy_by_source"].items() if v["n"] >= 200})
    print("accuracy by qtype:", {k: (v["n"], round(v["accuracy"], 4)) for k, v in audit["accuracy_by_qtype"].items()})
    print("tier-1 trajectory:", audit["tier1_trajectory"])


if __name__ == "__main__":
    main()
