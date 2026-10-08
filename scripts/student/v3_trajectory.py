#!/usr/bin/env python
"""V3: does held-out skill rise or fall with training? Analysis only, nothing is selected here.

Scores 22 checkpoints of student_v0 run 3 (step 1,000 .. the last, step 44,000) with the V2 fixed
path, bf16 (the precision of every reported student_v0 number), on two groups of v0.2 dev items:

  seen  the 2,019 seen-template dev items that student_v0 used for checkpoint selection
        (outputs/student_v0/S9_run3/dev_eval_ids.json, read as stored)
  held  every v0.2 dev item of the four held-out templates (D14): 2,574 items

Checkpoint rule: every 4th of the 88 checkpoints starting at step 1,000 (indices 1, 5, ..., 81),
plus the last checkpoint (44,000). That is 22 points and includes both required steps.

Metrics are computed from the raw probabilities. Checkpoints carry no fitted temperature, so
Brier is uncalibrated and is labelled that way. Accuracy is unaffected by temperature.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.paired import bootstrap_spearman, item_brier, spearman  # noqa: E402
from meddecide.model.meddecide_model import MedDecideModel  # noqa: E402
from meddecide.train.data import read_items  # noqa: E402

HELD_OUT = (
    "ct_phase_choice_v1",
    "fda_boxed_warning_noul_v1",
    "pubmed_humans_noul_v1",
    "ct_arm_role_noul_v1",
)
RUN_DIR = ROOT / "outputs" / "student_v0" / "S9_run3"
BENCH = ROOT / "data" / "bench" / "v0.2"
OUT_DIR = ROOT / "outputs" / "student_v1" / "V3"
REPORT = ROOT / "loops" / "student_v1" / "trajectory.md"


def selected_checkpoints() -> list[Path]:
    steps = sorted(
        (int(m.group(1)), p)
        for p in (RUN_DIR / "checkpoints").iterdir()
        if (m := re.fullmatch(r"step_(\d+)", p.name))
    )
    if len(steps) != 88:
        raise SystemExit(f"expected 88 checkpoints, found {len(steps)}")
    indices = sorted(set(range(1, 82, 4)) | {len(steps) - 1})
    return [steps[i][1] for i in indices]


def load_groups() -> tuple[list, list[str], list, list[str]]:
    seen_ids = json.loads((RUN_DIR / "dev_eval_ids.json").read_text(encoding="utf-8"))["item_ids"]
    dev = {}
    for directory in ("tier1", "fresh"):
        for path in sorted((BENCH / directory).glob("*.jsonl")):
            for item in read_items(path):
                if str(item.split) == "dev":
                    dev[item.item_id] = item
    seen = [dev[i] for i in seen_ids]
    held = [i for i in dev.values() if str(i.template_id) in HELD_OUT]
    held.sort(key=lambda i: i.item_id)
    return seen, [i.item_id for i in seen], held, [i.item_id for i in held]


def group_metrics(rows: list[dict]) -> dict:
    by_template: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_template[row["template_id"]].append(row)
    per_template = {
        t: {
            "n": len(v),
            "accuracy": float(np.mean([r["correct"] for r in v])),
            "brier_uncalibrated": float(np.mean([r["brier"] for r in v])),
        }
        for t, v in sorted(by_template.items())
    }
    return {
        "n": len(rows),
        "micro_accuracy": float(np.mean([r["correct"] for r in rows])),
        "macro_accuracy": float(np.mean([v["accuracy"] for v in per_template.values()])),
        "micro_brier_uncalibrated": float(np.mean([r["brier"] for r in rows])),
        "per_template": per_template,
    }


def score_checkpoint(path: Path, items: list, group: str, batch_size: int) -> list[dict]:
    model = MedDecideModel.load(path, device="cuda:0", dtype="bfloat16")
    scored = model.score_items(items, batch_size=batch_size, max_batch_tokens=32768,
                               round_to_chunk=False)
    rows = []
    for item, probs, gold in zip(items, scored.probs, scored.gold_indices, strict=True):
        rows.append({
            "item_id": item.item_id,
            "group": group,
            "template_id": str(item.template_id),
            "option_probs": [float(p) for p in probs],
            "gold_index": int(gold),
            "correct": int(int(np.argmax(probs)) == int(gold)),
            "brier": item_brier(probs, int(gold)),
        })
    del model
    return rows


def write_svg(path: Path, steps: list[int], series: dict[str, list[float]], title: str) -> None:
    width, height, pad = 760, 380, 60
    xs = np.array(steps, dtype=float)
    lo = min(min(v) for v in series.values())
    hi = max(max(v) for v in series.values())
    span = (hi - lo) or 1.0

    def sx(x: float) -> float:
        return pad + (x - xs.min()) / ((xs.max() - xs.min()) or 1.0) * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - lo) / span * (height - 2 * pad)

    colours = {"seen (macro acc)": "#1f6f8b", "held-out (macro acc)": "#c0392b"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'font-family="sans-serif" font-size="12">',
             f'<rect width="{width}" height="{height}" fill="white"/>',
             f'<text x="{pad}" y="24" font-size="14" font-weight="bold">{title}</text>',
             f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#333"/>',
             f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#333"/>']
    for step in steps:
        x = sx(step)
        parts.append(f'<text x="{x:.1f}" y="{height - pad + 16}" text-anchor="middle">{step // 1000}k</text>')
    for name, values in series.items():
        colour = colours.get(name, "#555")
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(steps, values, strict=True))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2"/>')
        for x, y in zip(steps, values, strict=True):
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{colour}"/>')
        parts.append(f'<text x="{width - pad - 4}" y="{pad + 16 * (1 + list(series).index(name))}" '
                     f'text-anchor="end" fill="{colour}">{name}</text>')
    parts.append(f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle">training step</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="1 = the V2 evaluation path (no padding)")
    parser.add_argument("--limit-checkpoints", type=int, default=None, help="debug: first N only")
    args = parser.parse_args()

    started = time.time()
    seen, seen_ids, held, held_ids = load_groups()
    checkpoints = selected_checkpoints()[: args.limit_checkpoints]
    preds_dir = OUT_DIR / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    points = []
    for path in checkpoints:
        step = int(path.name.split("_")[1])
        t0 = time.time()
        rows = score_checkpoint(path, seen, "seen", args.batch_size)
        rows += score_checkpoint(path, held, "held", args.batch_size)
        with (preds_dir / f"step_{step}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        seen_m = group_metrics([r for r in rows if r["group"] == "seen"])
        held_m = group_metrics([r for r in rows if r["group"] == "held"])
        points.append({"step": step, "seen": seen_m, "held": held_m,
                       "wall_clock_s": round(time.time() - t0, 1)})
        print(f"step {step}: seen macro {seen_m['macro_accuracy']:.4f} held macro "
              f"{held_m['macro_accuracy']:.4f} ({points[-1]['wall_clock_s']} s)", flush=True)

    steps = [p["step"] for p in points]
    seen_macro = [p["seen"]["macro_accuracy"] for p in points]
    held_macro = [p["held"]["macro_accuracy"] for p in points]
    held_brier = [p["held"]["micro_brier_uncalibrated"] for p in points]
    seen_brier = [p["seen"]["micro_brier_uncalibrated"] for p in points]
    rho_held = spearman(steps, held_macro)
    rho_held_ci = bootstrap_spearman(steps, held_macro, n_resamples=2000, seed=0)
    rho_seen = spearman(steps, seen_macro)
    rho_seen_ci = bootstrap_spearman(steps, seen_macro, n_resamples=2000, seed=0)
    result = {
        "kind": "checkpoint_trajectory",
        "task": "V3",
        "checkpoint_rule": "every 4th of 88 from step 1000 (indices 1,5,...,81) plus the last (44000)",
        "n_points": len(points),
        "dtype": "bfloat16",
        "batch_size": args.batch_size,
        "round_to_chunk": False,
        "seen_item_ids_source": "outputs/student_v0/S9_run3/dev_eval_ids.json",
        "n_seen_items": len(seen_ids),
        "n_held_items": len(held_ids),
        "held_templates": list(HELD_OUT),
        "spearman_step_vs_held_macro": {"rho": rho_held, "ci_lo": rho_held_ci[1], "ci_hi": rho_held_ci[2]},
        "spearman_step_vs_seen_macro": {"rho": rho_seen, "ci_lo": rho_seen_ci[1], "ci_hi": rho_seen_ci[2]},
        "points": points,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True, check=False).stdout.strip(),
        "wall_clock_s": round(time.time() - started, 1),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "trajectory.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_svg(OUT_DIR / "trajectory_macro_accuracy.svg", steps,
              {"seen (macro acc)": seen_macro, "held-out (macro acc)": held_macro},
              "student_v0 run 3: macro accuracy vs training step (dev)")
    write_svg(OUT_DIR / "trajectory_brier.svg", steps,
              {"seen (micro Brier, uncalibrated)": seen_brier,
               "held-out (micro Brier, uncalibrated)": held_brier},
              "student_v0 run 3: Brier vs training step (dev, uncalibrated)")
    print(json.dumps({"rho_held": rho_held, "ci": rho_held_ci[1:], "rho_seen": rho_seen}))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| step | seen macro acc | held-out macro acc | held-out micro acc | seen Brier (uncal.) | held-out Brier (uncal.) |",
             "|---:|---:|---:|---:|---:|---:|"]
    for p in points:
        lines.append(f"| {p['step']} | {p['seen']['macro_accuracy']:.4f} | {p['held']['macro_accuracy']:.4f} | "
                     f"{p['held']['micro_accuracy']:.4f} | {p['seen']['micro_brier_uncalibrated']:.4f} | "
                     f"{p['held']['micro_brier_uncalibrated']:.4f} |")
    REPORT.write_text("\n".join(["# student_v1 — V3 checkpoint trajectory", "", *lines, ""]) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
