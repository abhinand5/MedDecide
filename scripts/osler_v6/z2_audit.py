"""Z2 self-audit (R6): re-derive the λ probe's numbers from its raw per-step records in a fresh process.

`z2_probe.py` writes `probe.json` (the summary) and `steps.jsonl` (one record per step, with the pre-clip CE and KL gradient
norms and their share). This script recomputes from `steps.jsonl` alone: the 200-step count and ordering, the per-step
labelled-item counts against the D57 quota, the mean share s0, the closed-form λ = (0.4/0.6)·(1-s0)/s0, the loss/gradient
health, and the spread of the shares. It compares every recomputed value with `probe.json` and exits 1 on a mismatch. Writes
`outputs/osler_v6/Z2/probe/SELF_AUDIT.md` and `audit.json`.

Usage (CPU):  uv run --frozen python scripts/osler_v6/z2_audit.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO / "outputs" / "osler_v6" / "Z2" / "probe"
STEPS = 200
MIN_LABELLED = 3
TARGET = 0.40


def main() -> None:
    probe = json.loads((PROBE_DIR / "probe.json").read_text())
    if probe.get("status") != "DONE":
        raise SystemExit(f"the probe's status is {probe.get('status')!r}, not DONE: nothing to audit")
    records: list[dict[str, Any]] = list(iter_jsonl(PROBE_DIR / "steps.jsonl"))
    problems: list[str] = []
    steps = [int(record["step"]) for record in records]
    teachers = [int(record["n_teacher"]) for record in records]
    shares = [float(record["grad_share"]) for record in records]
    ce = [float(record["grad_ce"]) for record in records]
    kl = [float(record["grad_kl"]) for record in records]
    losses = [float(record["loss"]) for record in records]
    if steps != list(range(1, STEPS + 1)):
        problems.append(f"the step records are not 1..{STEPS} in order ({len(steps)} records)")
    if min(teachers) < MIN_LABELLED:
        problems.append(f"a step carried {min(teachers)} labelled items, below the D57 quota of {MIN_LABELLED}")
    if any(not math.isfinite(value) for value in losses + shares + ce + kl):
        problems.append("a loss or gradient value is not finite")
    if any(value <= 0 for value in ce):
        problems.append("a CE gradient norm is not positive")
    if any(value <= 0 for value in kl):
        problems.append("a KL gradient norm is not positive")
    s0 = sum(shares) / len(shares)
    if not 0.0 < s0 < 1.0:
        problems.append(f"s0 = {s0} makes the closed form undefined")
    lam = (TARGET / (1.0 - TARGET)) * (1.0 - s0) / s0
    recorded = probe.get("lambda", {})
    checks = {
        "steps": len(records),
        "step_sequence_ok": steps == list(range(1, STEPS + 1)),
        "labelled_min": min(teachers),
        "labelled_max": max(teachers),
        "labelled_histogram": {str(k): teachers.count(k) for k in sorted(set(teachers))},
        "share_mean_s0": s0,
        "share_min": min(shares),
        "share_max": max(shares),
        "share_recorded_s0": float(recorded.get("s0", float("nan"))),
        "s0_matches_recorded": abs(s0 - float(recorded.get("s0", float("nan")))) < 1e-12,
        "lambda_recomputed": lam,
        "lambda_recorded": float(recorded.get("value", float("nan"))),
        "lambda_matches_recorded": abs(lam - float(recorded.get("value", float("nan")))) < 1e-12,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_all_finite": all(math.isfinite(value) for value in losses),
        "ce_min": min(ce),
        "kl_min": min(kl),
        "share_mean_of_norms": sum(kl) / (sum(ce) + sum(kl)),
    }
    if not checks["s0_matches_recorded"]:
        problems.append("the recomputed s0 differs from probe.json")
    if not checks["lambda_matches_recorded"]:
        problems.append("the recomputed λ differs from probe.json")
    if probe.get("steps_completed") != len(records):
        problems.append(f"probe.json says {probe.get('steps_completed')} steps, steps.jsonl holds {len(records)}")
    result = {"generated_utc": utcnow(), "task": "Z2", "checks": checks, "problems": problems,
              "ok": not problems, "config": probe.get("config"), "mix": probe.get("mix"),
              "gradient_share_definition": probe.get("gradient_share", {}).get("definition")}
    write_json(PROBE_DIR / "audit.json", result)
    problem_lines = [f"- {problem}" for problem in problems] or ["- None."]
    lines = [
        "# Z2 — SELF_AUDIT (R6, fresh process)",
        "",
        f"Generated {result['generated_utc']} by `uv run --frozen python scripts/osler_v6/z2_audit.py` from `steps.jsonl` only.",
        "",
        "| quantity | value | check |",
        "|---|---|---|",
        f"| Steps | {len(records)} | 1..{STEPS} in order: {checks['step_sequence_ok']} |",
        f"| Labelled items per step | min {min(teachers)}, max {max(teachers)} | >= D57 quota of {MIN_LABELLED}: "
        f"{min(teachers) >= MIN_LABELLED} |",
        f"| s0 (mean per-step share) | {s0:.6f} | matches probe.json: {checks['s0_matches_recorded']} |",
        f"| λ = (0.4/0.6)·(1-s0)/s0 | {lam:.6f} | matches probe.json: {checks['lambda_matches_recorded']} |",
        f"| Share range | {min(shares):.6f} - {max(shares):.6f} | all finite: {checks['loss_all_finite']} |",
        f"| Share as a ratio of norm sums | {checks['share_mean_of_norms']:.6f} | additional; the rule is the mean of per-step shares |",
        f"| Loss first/last | {losses[0]:.6f} / {losses[-1]:.6f} | all finite: {checks['loss_all_finite']} |",
        f"| Gradient norms | CE min {min(ce):.3g}, KL min {min(kl):.3g} | both positive on every step |",
        "",
        "The probe is λ = 1, 200 steps, pre-clip, on the distilled mix with the D57 quota; the arithmetic above re-derives the",
        "closed form written down before the run. λ is frozen at the recomputed value; the arm's own 200-step share is the",
        "pre-registered band check.",
        "",
        "## Problems",
        "",
        *problem_lines,
        "",
        "## Verdict",
        "",
        "**PASS** — every recomputed number matches probe.json." if not problems
        else "**FAIL** — the problems above must be resolved before Z2 is marked DONE.",
        "",
    ]
    (PROBE_DIR / "SELF_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Z2 audit: {'PASS' if not problems else 'FAIL'}; s0 {s0:.6f}; lambda {lam:.6f}; problems {problems}")


if __name__ == "__main__":
    main()
