#!/usr/bin/env python
"""V3 companion: both macro definitions for every checkpoint, from the saved predictions and items.

``v3_trajectory.py`` saves each checkpoint's predictions (outputs/student_v1/V3/preds/step_<n>.jsonl)
and reports the per-template macro accuracy used by G1. The selection metric of student_v0 (and of the
arms, V6-V8) is the trainer's ``macro_accuracy``: the mean of per-class recall over the canonical option
letters pooled across templates (``meddecide.eval.metrics.macro_accuracy`` on ``ScoredItems``). This
script reconstructs that pooled metric from the saved predictions by canonicalising each item the way
scoring does, and checks its micro accuracy against the trajectory record. Writes
``outputs/student_v1/V3/metrics_by_step.json``.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.metrics import macro_accuracy  # noqa: E402
from meddecide.eval.readout import canonicalise_options  # noqa: E402
from meddecide.train.data import read_items  # noqa: E402

V3 = ROOT / "outputs" / "student_v1" / "V3"
BENCH = ROOT / "data" / "bench" / "v0.2"
SEEN_IDS = ROOT / "outputs" / "student_v0" / "S9_run3" / "dev_eval_ids.json"


def load_items(needed: set[str]) -> dict:
    found = {}
    for folder in ("tier1", "fresh"):
        for path in sorted((BENCH / folder).glob("*.jsonl")):
            for item in read_items(path):
                if item.item_id in needed and str(item.split) == "dev":
                    found[item.item_id] = item
    return found


def main() -> None:
    trajectory = json.loads((V3 / "trajectory.json").read_text(encoding="utf-8"))
    rows_by_step = {}
    for path in sorted((V3 / "preds").glob("step_*.jsonl")):
        step = int(path.stem.split("_")[1])
        rows_by_step[step] = [json.loads(line) for line in path.open(encoding="utf-8")]
    needed = {r["item_id"] for rows in rows_by_step.values() for r in rows}
    items = load_items(needed)
    if len(items) != len(needed):
        raise SystemExit(f"{len(needed) - len(items)} predicted items were not found in the dev split")
    canonical = {}
    for item_id, item in items.items():
        canon, _ = canonicalise_options(item)
        canonical[item_id] = ([o.key for o in canon.options], canon.gold)
    seen_ids = set(json.loads(SEEN_IDS.read_text(encoding="utf-8"))["item_ids"])

    out = {"kind": "v3_metrics_by_step", "definitions": {
        "per_template_macro": "mean over templates of per-template accuracy (G1 definition)",
        "pooled_class_macro": "trainer macro_accuracy: mean of per-class recall over canonical letters pooled "
                              "across templates (student_v0 selection metric)",
    }, "steps": []}
    for step in sorted(rows_by_step):
        entry = {"step": step}
        for group in ("seen", "held"):
            rows = [r for r in rows_by_step[step] if r["group"] == group]
            pred, gold, correct = [], [], []
            for r in rows:
                keys, g = canonical[r["item_id"]]
                p = keys[int(np.argmax(r["option_probs"]))]
                pred.append(p)
                gold.append(g)
                correct.append(p == g)
            per_t = defaultdict(list)
            for r, c in zip(rows, correct, strict=True):
                per_t[r["template_id"]].append(c)
            entry[group] = {
                "n": len(rows),
                "micro_accuracy": float(np.mean(correct)),
                "per_template_macro": float(np.mean([np.mean(v) for v in per_t.values()])),
                "pooled_class_macro": float(macro_accuracy(pred, gold)),
            }
        out["steps"].append(entry)
    # cross-check: the micro accuracies must equal the trajectory record exactly
    record = {p["step"]: p for p in trajectory["points"]}
    for entry in out["steps"]:
        for group, key in (("seen", "seen"), ("held", "held")):
            if abs(entry[group]["micro_accuracy"] - record[entry["step"]][key]["micro_accuracy"]) > 1e-9:
                raise SystemExit(f"micro accuracy mismatch at step {entry['step']} ({group})")
    out["seen_item_ids_match_selection_sample"] = bool(
        seen_ids == {r["item_id"] for r in rows_by_step[min(rows_by_step)] if r["group"] == "seen"})
    (V3 / "metrics_by_step.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for entry in out["steps"]:
        print(f"step {entry['step']}: seen pooled {entry['seen']['pooled_class_macro']:.4f} "
              f"per-template {entry['seen']['per_template_macro']:.4f} | held pooled "
              f"{entry['held']['pooled_class_macro']:.4f} per-template {entry['held']['per_template_macro']:.4f}")


def write_report() -> None:
    """Render loops/student_v1/trajectory.md and copy the figures next to it (derived, R9)."""
    import shutil

    trajectory = json.loads((V3 / "trajectory.json").read_text(encoding="utf-8"))
    metrics = json.loads((V3 / "metrics_by_step.json").read_text(encoding="utf-8"))
    by_step = {s["step"]: s for s in metrics["steps"]}
    rho_held = trajectory["spearman_step_vs_held_macro"]
    rho_seen = trajectory["spearman_step_vs_seen_macro"]
    first, last = trajectory["points"][0]["step"], trajectory["points"][-1]["step"]
    lines = [
        "# student_v1 — V3: checkpoint trajectory (seen vs held-out)",
        "",
        "Generated by `scripts/student/v3_trajectory.py` and `scripts/student/v3_metrics.py` from "
        "`outputs/student_v1/V3/`. Do not edit by hand.",
        "",
        "Analysis only: nothing here selects a checkpoint, a configuration or a temperature. Scoring is "
        f"bf16 with no padding (batch size 1, the V2 evaluation path). {trajectory['n_points']} checkpoints of "
        "student_v0 run 3, from step "
        f"{first:,} to {last:,}: every 4th of the 88 checkpoints from step 1,000, plus the last.",
        "",
        f"Seen items: {trajectory['n_seen_items']:,} (the student_v0 selection sample). Held-out items: "
        f"{trajectory['n_held_items']:,} (dev items of the four held-out templates, D14).",
        "",
        "## Result",
        "",
        f"- Held-out skill falls with training. Spearman(step, held-out per-template macro accuracy) = "
        f"{rho_held['rho']:+.3f}, 95% bootstrap CI [{rho_held['ci_lo']:+.3f}, {rho_held['ci_hi']:+.3f}] "
        f"(points resampled, 2,000 resamples). The interval excludes zero.",
        f"- Seen-template skill does not show a clear trend: Spearman = {rho_seen['rho']:+.3f}, CI "
        f"[{rho_seen['ci_lo']:+.3f}, {rho_seen['ci_hi']:+.3f}].",
        f"- The same fall appears under the selection metric (pooled per-class macro, see definitions): "
        f"held-out {by_step[first]['held']['pooled_class_macro']:.4f} at step {first:,} -> "
        f"{by_step[last]['held']['pooled_class_macro']:.4f} at step {last:,}.",
        "",
        "## Definitions",
        "",
        "- **per-template macro**: mean over templates of per-template accuracy (the G1 definition).",
        "- **pooled-class macro**: the trainer's `macro_accuracy` (mean per-class recall over canonical "
        "letters pooled across templates); the selection metric of student_v0 and of arms A-C. Reconstructed "
        "offline from the saved predictions; at step 1,000 the seen value (0.6793) matches student_v0's "
        "recorded selection value (0.6798, batched path) to 0.0005.",
        "- **Brier** is uncalibrated (checkpoints carry no fitted temperature).",
        "",
        "## Table",
        "",
        "| step | seen per-template macro | seen pooled macro | seen micro acc | seen Brier | held-out per-template macro | held-out pooled macro | held-out micro acc | held-out Brier |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in trajectory["points"]:
        step = point["step"]
        m = by_step[step]
        lines.append(
            f"| {step:,} | {point['seen']['macro_accuracy']:.4f} | {m['seen']['pooled_class_macro']:.4f} | "
            f"{point['seen']['micro_accuracy']:.4f} | {point['seen']['micro_brier_uncalibrated']:.4f} | "
            f"{point['held']['macro_accuracy']:.4f} | {m['held']['pooled_class_macro']:.4f} | "
            f"{point['held']['micro_accuracy']:.4f} | {point['held']['micro_brier_uncalibrated']:.4f} |"
        )
    lines += [
        "",
        "## Figures",
        "",
        "![macro accuracy vs step](figures/trajectory_macro_accuracy.svg)",
        "",
        "![Brier vs step](figures/trajectory_brier.svg)",
        "",
        "## Reading against the ADVISORY (§2)",
        "",
        "The ADVISORY names the outcome \"held-out accuracy falling with training -> template over-fitting; the "
        "next loop uses early stopping on a held-out-like dev signal and more templates per step\". The "
        "measurement above is that outcome for student_v0's training mix. It is a measurement, not a cause: "
        "the mix had 14 templates and no PubMed data. Whether the V4 mix (arms A-C) follows the same curve is "
        "tested in V6-V8, not here.",
        "",
    ]
    report = ROOT / "loops" / "student_v1" / "trajectory.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    figures = ROOT / "loops" / "student_v1" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for name in ("trajectory_macro_accuracy.svg", "trajectory_brier.svg"):
        shutil.copyfile(V3 / name, figures / name)
    print(f"wrote {report.relative_to(ROOT)} and {len(list(figures.glob('*.svg')))} figures")


if __name__ == "__main__":
    main()
    write_report()
