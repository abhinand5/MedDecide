#!/usr/bin/env python
"""Render loops/student_v1/arm_training.md from the arm training artifacts (derived, R9).

For each arm that has finished training (arm A: outputs/student_v1/V6/arm_a; arm B: outputs/student_v1/V7/arm_b;
arm C: outputs/student_v1/V8), the report gives the dev trajectory (every 500 steps), the selected checkpoint,
the fitted temperatures, the throughput and wall-clock, and the per-step grad-norm tail with the divergence
tripwires of ADVISORY V6-V8 (dev macro < 0.50 on two consecutive evals; pre-clip grad norm > 5,000).
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "loops" / "student_v1" / "arm_training.md"
ARMS = {
    "A (pointer head)": ROOT / "outputs" / "student_v1" / "V6" / "arm_a",
    "B (letter readout)": ROOT / "outputs" / "student_v1" / "V7" / "arm_b",
}


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p * (len(ordered) - 1)))]


def section(name: str, base: Path) -> list[str]:
    if not (base / "logs" / "dev_evals.jsonl").exists():
        return [f"## {name}", "", "NOT MEASURED — training has not finished.", ""]
    evals = [json.loads(line) for line in (base / "logs" / "dev_evals.jsonl").open(encoding="utf-8")]
    dev = [e for e in evals if e.get("event") == "dev_eval"]
    steps = [json.loads(line) for line in (base / "logs" / "train_steps.jsonl").open(encoding="utf-8")]
    steps = [s for s in steps if s.get("event") == "step"]
    grad = [s["grad_norm"] for s in steps if s.get("grad_norm") is not None]
    run = json.loads((base / "run.json").read_text(encoding="utf-8")) if (base / "run.json").exists() else {}
    best = json.loads((base / "best.json").read_text(encoding="utf-8")) if (base / "best.json").exists() else {}
    temps = json.loads((base / "temperature.json").read_text(encoding="utf-8")) if (base / "temperature.json").exists() else {}
    fitted = {k: v.get("temperature") for k, v in (temps.get("fits") or {}).items()}
    run_training = run.get("training", {})
    throughput = run_training.get("throughput", {})
    longest_below = max((len(list(g)) for k, g in itertools.groupby(
        [e["metrics"]["macro_accuracy"] < 0.50 for e in dev]) if k), default=0)
    trip_steps = [s["step"] for s in steps if s.get("grad_norm") is not None and s["grad_norm"] > 5000]
    first_trip = min(trip_steps) if trip_steps else None
    eligible = [e for e in dev if first_trip is None or e["step"] <= first_trip]
    rule_best = None
    for e in eligible:
        m = e["metrics"]
        if rule_best is None or m["macro_accuracy"] > rule_best["metrics"]["macro_accuracy"] + 1e-12 or (
                abs(m["macro_accuracy"] - rule_best["metrics"]["macro_accuracy"]) <= 1e-12
                and m["brier"] < rule_best["metrics"]["brier"] - 1e-12):
            rule_best = e
    lines = [
        f"## {name}", "",
        f"- dev evaluations: {len(dev)} (every 500 steps, {dev[0]['n_items'] if dev else 0} items).",
        "- tripwire (pre-clip grad norm > 5,000): " + (
            f"tripped at step {first_trip:,} ({len(trip_steps)} step(s)); the run is recorded as diverged and "
            f"its verdict checkpoint is the best saved at or before the trip: step "
            f"{rule_best['step'] if rule_best else 'none'} (dev pooled macro "
            f"{rule_best['metrics']['macro_accuracy']:.4f}, accuracy {rule_best['metrics']['accuracy']:.4f}). "
            f"Unrestricted best (additional, not the verdict): step {best.get('step')}."
            if first_trip is not None else
            f"not tripped; selected checkpoint: step {best.get('step')} (rule: highest dev pooled-class macro, "
            f"ties by Brier, then earlier step)."),
        "- fitted temperatures (dev, per qtype): " + ", ".join(f"{k} {v:.4f}" for k, v in sorted(fitted.items())
                                                             if v is not None),
        f"- training: {len(steps)} steps logged; wall {run.get('wall_clock_s', float('nan')):.0f} s; "
        f"items/s {throughput.get('items_per_s', float('nan')):.2f}; tokens/s "
        f"{throughput.get('tokens_per_s', float('nan')):.0f}; stop: {run.get('exit')}.",
        f"- grad norm (pre-clip, every step): p50 {percentile(grad, 0.5):.3f}, p95 {percentile(grad, 0.95):.3f}, "
        f"max {max(grad):.3f}.",
        f"- divergence tripwires: longest run of consecutive dev macro < 0.50: {longest_below} (tripwire: 2); "
        f"max pre-clip grad norm {max(grad):.1f} (tripwire: 5,000); "
        f"grad tripwire {'TRIPPED' if first_trip is not None else 'not tripped'}.",
        "",
        "| step | dev acc | dev pooled macro | dev Brier | best so far |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for e in dev:
        m = e["metrics"]
        lines.append(f"| {e['step']} | {m['accuracy']:.4f} | {m['macro_accuracy']:.4f} | "
                     f"{m['brier']:.4f} | {'yes' if e.get('selected') else ''} |")
    lines.append("")
    return lines


def main() -> None:
    lines = ["# student_v1 — arm training trajectories (V6-V8)", "",
             "Generated by `scripts/student/v6_report.py` from the arm training folders. Do not edit by hand.", ""]
    for name, base in ARMS.items():
        lines += section(name, base)
    lines += ["## Arm C (Clef-style head via Unsloth)", "",
              "NOT MEASURED — training and selection are scheduled after arm B (see STATE).", ""]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
