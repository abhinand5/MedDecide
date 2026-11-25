"""Y8 self-audit (R6): re-derives the final run's figures from its logs and predictions, in a fresh process (CPU).

Compared with the values the run recorded: each evaluation's accuracy and item count; the CG v2 and MIMIC partition counts at each
evaluation; the logged steps (non-finite values); the evaluation steps against the grid; the selected step (the trainer's rule: the
highest template macro accuracy on fast dev v4, a tie to the lower Brier; configs/osler_v3/block_arm.yaml sets the metric); the CG v2
and fast dev v4 figures at the selected checkpoint, from the predictions the run wrote (accuracy, template macro accuracy, the
majority baseline, and the Brier score as the sum over options averaged over items); the replay's batches and realised items.
The prediction item ids are the trainer's (mix_row_to_item), compared with the source rows' ids in file order.

Reported, not compared as equal: the in-training evaluation at the selected step against the reloaded checkpoint's scoring. The
reloaded scores come from another batch composition, and bfloat16 scores can move a few decisions (CLAIMS C114, C116, C119). The
end-of-run predictions (dev_predictions.jsonl, cg_v2_predictions.jsonl) are scored by Trainer.score at temperature 1, the same path as
the in-training evaluation, so their Brier scores are uncalibrated (STATE §7 deviation item 1; C191). The calibrated figures are the
labelled re-scores in outputs/osler_v4/Y8/calibrated, not these.

The readout gate (D12; meddecide.eval.readout_gate) is applied per template to the selected checkpoint's CG v2 and fast dev v4 rows.

Writes outputs/osler_v4/Y8/rederive.json (counts and figures only; no item text). Exits 1 when a compared value differs.
Usage (CPU): CUDA_VISIBLE_DEVICES= uv run --frozen python scripts/osler_v4/y8_rederive.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.eval.metrics import template_macro_accuracy
from meddecide.eval.readout_gate import gate_by_template
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.trainer import evaluation_grid
from meddecide.utils.io import write_json

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v4" / "Y8"
CG_FILE = REPO / "data" / "bench" / "osler_v4" / "cg_v2.jsonl"
DEV_FILE = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"
STEPS = 63_304
EVALUATIONS = 10
DEV_ITEMS = 17_532  # fast dev v4 (Y4 manifest)
CG_ITEMS = 8_307  # CG v2 (Y4 manifest)
MIMIC_ITEMS = 5_562  # the fast dev v4 MIMIC partition (Y4 manifest)
REALISED_ITEMS = 242_255  # the Y8 replay count recorded in CLAIMS C166
TOLERANCE = 1e-6


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def brier(rows: list[dict[str, Any]]) -> float:
    """Mean over items of the sum over options of (p_k - y_k)^2 (meddecide.eval.metrics.brier_score)."""
    total = 0.0
    for row in rows:
        gold = int(row["gold_index"])
        total += sum((float(p) - (1.0 if k == gold else 0.0)) ** 2 for k, p in enumerate(row["probs"]))
    return total / len(rows)


def figures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [bool(r["correct"]) for r in rows]
    label, count = Counter(str(r["gold_key"]) for r in rows).most_common(1)[0]
    return {
        "n": len(rows),
        "n_correct": sum(correct),
        "accuracy": sum(correct) / len(rows),
        "template_macro_accuracy": template_macro_accuracy([float(c) for c in correct], [str(r["template_id"]) for r in rows]),
        "majority_baseline": count / len(rows),
        "majority_label": label,
        "brier": brier(rows),
    }


def readout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gates = gate_by_template(rows)
    return {"cells": len(gates), "failing": {t: g.status for t, g in gates.items() if not g.passes}}


def select(evals: list[dict[str, Any]]) -> int:
    """The trainer's rule (trainer.py, the selection block): the highest template macro accuracy; a tie goes to the lower Brier."""
    best_key, best_step = None, None
    for e in evals:
        key = (e["template_macro_accuracy"], -e["brier"])
        if best_key is None or key > best_key:
            best_key, best_step = key, e["step"]
    return best_step


def main() -> None:
    train_rows = read_jsonl(OUT / "logs" / "train.jsonl")
    steps = [r for r in train_rows if r.get("event") == "step"]
    evals = [{"step": r["step"], **r["metrics"]} for r in train_rows if r.get("event") == "eval"]
    partitions = read_jsonl(OUT / "logs" / "partitions.jsonl")
    grid = list(evaluation_grid(STEPS, EVALUATIONS))
    eval_steps = [e["step"] for e in evals]
    checks: list[dict[str, Any]] = []
    info: dict[str, Any] = {"logged_steps": len(steps), "last_logged_step": max(r["step"] for r in steps),
                            "evaluation_steps_so_far": eval_steps}

    def compare(name: str, recomputed: Any, recorded: Any) -> None:
        if isinstance(recorded, bool) or not isinstance(recorded, (int, float)):
            ok = recomputed == recorded
        else:
            ok = abs(float(recomputed) - float(recorded)) <= TOLERANCE
        checks.append({"check": name, "recomputed": recomputed, "recorded": recorded, "ok": bool(ok)})

    for e in evals:
        compare(f"evaluation {e['step']}: accuracy = n_correct / n", e["n_correct"] / e["n"], e["accuracy"])
        compare(f"evaluation {e['step']}: fast dev v4 items", e["n"], DEV_ITEMS)
    for p in partitions:
        compare(f"partition {p['step']}: CG v2 items", p["cg_v2"]["n"], CG_ITEMS)
        compare(f"partition {p['step']}: MIMIC partition items", p["mimic_partition"]["n"], MIMIC_ITEMS)
        compare(f"partition {p['step']}: CG v2 accuracy = n_correct / n",
                p["cg_v2"]["n_correct"] / p["cg_v2"]["n"], p["cg_v2"]["accuracy"])
        compare(f"partition {p['step']}: MIMIC accuracy = n_correct / n",
                p["mimic_partition"]["n_correct"] / p["mimic_partition"]["n"], p["mimic_partition"]["accuracy"])
    compare("partition steps = evaluation steps", [p["step"] for p in partitions], eval_steps)
    compare("evaluation steps are the first of the grid", eval_steps, grid[: len(eval_steps)])
    nonfinite = [r["step"] for r in steps if not all(math.isfinite(float(r[k])) for k in ("loss", "ce", "brier", "grad_norm"))]
    compare("logged steps with a non-finite value", len(nonfinite), 0)
    if evals:
        info["selected_among_evaluations_so_far"] = select(evals)

    cg_path, dev_path, final_path = OUT / "cg_v2_predictions.jsonl", OUT / "dev_predictions.jsonl", OUT / "final_result.json"
    cg_fig: dict[str, Any] = {}
    dev_fig: dict[str, Any] = {}
    if cg_path.exists():
        cg_rows = read_jsonl(cg_path)
        cg_fig = figures(cg_rows)
        compare("CG v2 predictions: items", len(cg_rows), CG_ITEMS)
        # the scorer writes the trainer's item ids (mix_row_to_item), in the source file's order; the source file's own item_id
        # is not the scoring path's id (the first run compared those and found no overlap; kept as rederive_first_run.json)
        compare("CG v2 predictions: item ids are the trainer's ids of the CG v2 rows, in file order",
                [str(r["item_id"]) for r in cg_rows] == [mix_row_to_item(row).item_id for row in read_jsonl(CG_FILE)], True)
        compare("CG v2 majority baseline (definition check, first partition)", cg_fig["majority_baseline"],
                partitions[0]["cg_v2"]["majority_baseline"])
        info["cg_v2_at_selected"] = cg_fig
        info["cg_v2_readout"] = readout(cg_rows)
    if dev_path.exists():
        dev_rows = read_jsonl(dev_path)
        dev_fig = figures(dev_rows)
        compare("fast dev v4 predictions: items", len(dev_rows), DEV_ITEMS)
        compare("fast dev v4 predictions: item ids are the trainer's ids of the fast dev v4 rows, in file order",
                [str(r["item_id"]) for r in dev_rows] == [mix_row_to_item(row).item_id for row in read_jsonl(DEV_FILE)], True)
        compare("fast dev v4 majority baseline (definition check, last evaluation)", dev_fig["majority_baseline"],
                evals[-1]["majority_baseline"])
        info["dev_at_selected"] = dev_fig
        info["dev_readout"] = readout(dev_rows)
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        compare("steps taken", final["steps"], STEPS)
        compare("evaluations: all ten on the grid", eval_steps, grid)
        compare("selected step: the trainer's rule over the ten evaluations", select(evals), final["best_step"])
        for key in ("n_correct", "accuracy", "template_macro_accuracy", "brier"):
            compare(f"CG v2 at the selected checkpoint: {key}", cg_fig[key], final["cg_v2_at_selected"][key])
            compare(f"fast dev v4 at the selected checkpoint: {key}", dev_fig[key], final["full_dev_metrics"][key])
        replay = json.loads((OUT / "replays" / "final_mix_replay.json").read_text(encoding="utf-8"))
        compare("replay batches = steps", len(replay["sizes"]), STEPS)
        compare("realised items = the replay's batch sizes", sum(replay["sizes"]), replay["realised_items"])
        compare("realised items = the run's record", sum(replay["sizes"]), final["realised_items"])
        compare("realised items = CLAIMS C166", sum(replay["sizes"]), REALISED_ITEMS)
        compare("recorded logged-step validation: mismatches", final["validation"]["logged_steps"]["mismatches"], 0)
        compare("recorded validation: evaluation steps as planned", final["validation"]["evaluation_steps_as_planned"], True)
        best = final["best_step"]
        in_training = next(e for e in evals if e["step"] == best)
        in_training_cg = next(p for p in partitions if p["step"] == best)["cg_v2"]
        info["in_training_vs_reloaded_at_selected"] = {
            "step": best,
            "fast_dev_n_correct": {"in_training": in_training["n_correct"], "reloaded": dev_fig["n_correct"]},
            "fast_dev_template_macro": {"in_training": in_training["template_macro_accuracy"],
                                        "reloaded": dev_fig["template_macro_accuracy"]},
            "cg_v2_n_correct": {"in_training": in_training_cg["n_correct"], "reloaded": cg_fig["n_correct"]},
            "fast_dev_brier": {"in_training_T1": in_training["brier"], "reloaded_T1": dev_fig["brier"]},
        }

    mismatches = [c for c in checks if not c["ok"]]
    result = {"compared": len(checks), "mismatches": mismatches, "all_ok": not mismatches, "checks": checks, **info}
    write_json(OUT / "rederive.json", result)
    print(f"Y8 re-derivation: {len(checks)} values compared, {len(mismatches)} mismatches", flush=True)
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
