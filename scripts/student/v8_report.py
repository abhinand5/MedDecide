#!/usr/bin/env python
"""Render loops/student_v1/arm_c_training.md from the arm C (V8) artifacts (derived, R9; do not edit by hand).

Training summary from outputs/student_v1/V8/arm_c: training.json and logs/train_steps.jsonl (steps, the batch
order and option-order records, throughput, the pre-clip grad-norm tail, and the divergence tripwire of ADVISORY
V6-V8: pre-clip grad norm > 5,000). When outputs/student_v1/V8/selection.json exists, the dev trajectory and the
selection are added: the rule is the S9 rule (highest dev pooled-class macro; ties by Brier, then the earlier
step), and if the tripwire tripped the eligible checkpoints are those saved at or before the first trip. When
outputs/student_v1/V8/final/finalize.json exists, the calibration is added.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "student_v1" / "V8"
OUT = ROOT / "loops" / "student_v1" / "arm_c_training.md"
TRIPWIRE = 5000.0


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p * (len(ordered) - 1)))]


def training_lines() -> list[str]:
    training = json.loads((BASE / "arm_c" / "training.json").read_text(encoding="utf-8"))
    steps = [json.loads(line) for line in (BASE / "arm_c" / "logs" / "train_steps.jsonl").open(encoding="utf-8")]
    steps = [s for s in steps if "grad_norm" in s and "step" in s]
    grad = [float(s["grad_norm"]) for s in steps]
    trips = [int(s["step"]) for s in steps if float(s["grad_norm"]) > TRIPWIRE]
    first_trip = min(trips) if trips else None
    order = training.get("batch_order", {})
    option = training.get("option_order", {})
    lines = [
        "## Training (arm C, Unsloth decision head)", "",
        f"- steps logged: {len(steps)} (steps {min(int(s['step']) for s in steps):,} to "
        f"{max(int(s['step']) for s in steps):,}); max steps {training.get('max_steps'):,}; "
        f"warmup {training.get('warmup_steps')} steps; head learning rate {training.get('head_learning_rate')}; "
        f"training wall-clock {training.get('train_seconds', float('nan')):.0f} s; mean training loss "
        f"{training.get('train_loss', float('nan')):.4f}.",
        f"- batch order (D15): {order.get('rule')}; seed {order.get('seed')}, epoch {order.get('epoch')}; "
        f"items used {order.get('items_used'):,}, distinct {order.get('distinct_items_used'):,}; "
        f"sha256 of the used indices `{order.get('sha256_used_indices')}`.",
        f"- option order (D16): {option.get('rule')}; items whose order changed "
        f"{option.get('items_changed'):,}.",
        f"- pre-clip grad norm, every step: p50 {percentile(grad, 0.5):.3f}, p95 {percentile(grad, 0.95):.3f}, "
        f"max {max(grad):.3f}.",
        "- divergence tripwire (pre-clip grad norm > 5,000): " + (
            f"TRIPPED at step {first_trip:,} ({len(trips)} step(s))." if trips else
            f"not tripped ({len(steps):,} steps, largest pre-clip norm {max(grad):.1f})."),
        "",
    ]
    return lines


def selection_lines() -> list[str]:
    path = BASE / "selection.json"
    if not path.exists():
        return ["## Dev trajectory and selection", "",
                "NOT MEASURED — dev predictions and selection have not finished.", ""]
    selection = json.loads(path.read_text(encoding="utf-8"))
    table = sorted(selection["table"], key=lambda r: r["step"])
    steps = [json.loads(line) for line in (BASE / "arm_c" / "logs" / "train_steps.jsonl").open(encoding="utf-8")]
    trips = [int(s["step"]) for s in steps if "grad_norm" in s and float(s["grad_norm"]) > TRIPWIRE]
    first_trip = min(trips) if trips else None
    eligible = [r for r in table if first_trip is None or r["step"] <= first_trip]
    rule_best = None
    for row in eligible:
        if rule_best is None or row["pooled_class_macro"] > rule_best["pooled_class_macro"] + 1e-12 or (
                abs(row["pooled_class_macro"] - rule_best["pooled_class_macro"]) <= 1e-12
                and row["micro_brier"] < rule_best["micro_brier"] - 1e-12):
            rule_best = row
    lines = [
        "## Dev trajectory and selection", "",
        f"- dev items: {selection.get('dev_items'):,}; checkpoints scored: {len(table)}; selection (unrestricted, "
        f"S9 rule): step {selection['selected_step']:,}; selection restricted by the tripwire: step "
        f"{rule_best['step'] if rule_best else 'none'}.",
        "",
        "| step | dev acc | dev pooled macro | dev Brier | selected (rule) |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for row in table:
        mark = "yes" if rule_best is not None and row["step"] == rule_best["step"] else ""
        lines.append(f"| {row['step']} | {row['micro_accuracy']:.4f} | {row['pooled_class_macro']:.4f} | "
                     f"{row['micro_brier']:.4f} | {mark} |")
    lines.append("")
    final = BASE / "final" / "finalize.json"
    if final.exists():
        report = json.loads(final.read_text(encoding="utf-8"))
        lines += ["## Calibration and test predictions", "",
                  f"- checkpoint {report.get('checkpoint')}; calibration on seen dev (Unsloth `calibrate`): "
                  f"{json.dumps(report.get('calibration'), default=str)[:400]}; test items predicted "
                  f"{report.get('test_items_predicted')}, refused {report.get('test_items_refused')}.", ""]
    return lines


def main() -> None:
    lines = ["# student_v1 — arm C training and selection (V8)", "",
             "Generated by `scripts/student/v8_report.py` from `outputs/student_v1/V8`. Do not edit by hand.", ""]
    lines += training_lines()
    lines += selection_lines()
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
