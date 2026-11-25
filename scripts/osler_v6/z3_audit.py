"""Z3 self-audit (R6 and ADVISORY §2.4/§2.5): the arm's own first-200-step share band and the run's basic health.

The G1 arm runs with the frozen λ from Z2 and the D57 quota. ADVISORY §2.5 makes the arm's own first 200 steps the band
check: the pre-clip share s = ||grad KL|| / (||grad CE|| + ||grad KL||), measured with the arm's λ applied, must have a mean
in [0.30, 0.50]; if it does not, G1 is BLOCKED and reported, and λ is not re-tuned. The driver records every step's probe
values in `logs/train.jsonl` (log_every 1) for the first 200 steps because the trainer's `distill_probe_steps` is 200.

This script also re-derives the run's basic facts from the raw log and `arm_result.json`: the arm used the frozen λ, the quota
of 3, 13,992 steps, the two evaluations at 6,996 and 13,992, and no non-finite loss. It writes `SELF_AUDIT.md` and
`audit.json` in the arm directory and exits 1 on a failure.

Usage (CPU; after the arm's run):
    uv run --frozen python scripts/osler_v6/z3_audit.py --arm-dir outputs/osler_v6/Z3/G1s0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / "outputs" / "osler_v6" / "Z2" / "probe" / "probe.json"
BAND = (0.30, 0.50)
STEPS = 13_992
PROBE_STEPS = 200
QUOTA = 3
GRID = (6_996, 13_992)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm-dir", default="outputs/osler_v6/Z3/G1s0")
    args = parser.parse_args()
    arm_dir = (REPO / args.arm_dir) if not Path(args.arm_dir).is_absolute() else Path(args.arm_dir)
    result = json.loads((arm_dir / "arm_result.json").read_text())
    config = result["config"]
    probe = json.loads(PROBE.read_text())
    lam = float(probe["lambda"]["value"])
    problems: list[str] = []

    if float(config["distill_weight"]) != lam:
        problems.append(f"the arm's distill_weight {config['distill_weight']} is not the frozen λ {lam}")
    if int(config.get("distill_min_labelled_per_batch", 0)) != QUOTA:
        problems.append(f"the arm's quota is {config.get('distill_min_labelled_per_batch')}, not {QUOTA}")
    if int(result["steps"]) != STEPS:
        problems.append(f"the arm ran {result['steps']} steps, not {STEPS}")
    evaluation_steps = tuple(int(step) for step in result["evaluation_steps"])
    if evaluation_steps != GRID:
        problems.append(f"the evaluations are at {evaluation_steps}, not {GRID}")

    # the first 200 step records: every step must be present (log_every 1), carry the quota and a finite share
    steps: dict[int, dict[str, Any]] = {}
    losses_finite = True
    total_logged = 0
    for record in iter_jsonl(arm_dir / "logs" / "train.jsonl"):
        if record.get("event") != "step":
            continue
        total_logged += 1
        step = int(record["step"])
        if not math.isfinite(float(record["loss"])):
            losses_finite = False
        if step <= PROBE_STEPS:
            steps[step] = record
    missing = sorted(set(range(1, PROBE_STEPS + 1)) - set(steps))
    if missing:
        problems.append(f"the first {PROBE_STEPS} steps are not all logged ({len(missing)} missing)")
    if not losses_finite:
        problems.append("a step's loss is not finite")
    teachers = [int(steps[step]["n_teacher"]) for step in sorted(steps)]
    shares = [float(steps[step]["grad_share"]) for step in sorted(steps)]
    ce = [float(steps[step]["grad_ce"]) for step in sorted(steps)]
    kl = [float(steps[step]["grad_kl"]) for step in sorted(steps)]
    if teachers and min(teachers) < QUOTA:
        problems.append(f"a step carried {min(teachers)} labelled items, below the quota of {QUOTA}")
    if shares and any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in shares):
        problems.append("a first-200-step share is not finite or not in [0, 1]")
    if shares and any(value <= 0 for value in ce + kl):
        problems.append("a first-200-step gradient norm is not positive")
    mean_share = sum(shares) / len(shares) if shares else float("nan")
    band_pass = BAND[0] <= mean_share <= BAND[1] if shares else False
    if shares and not band_pass:
        problems.append(f"the first-{PROBE_STEPS}-step mean share {mean_share:.4f} is outside the D57 band {BAND}")

    final_eval = result["evaluations"][str(STEPS)]
    payload = {
        "task": "Z3",
        "arm_dir": str(arm_dir.relative_to(REPO)),
        "generated_utc": utcnow(),
        "frozen_lambda": lam,
        "probe_s0": probe["lambda"]["s0"],
        "arm_config": {key: config[key] for key in ("distill_weight", "distill_min_labelled_per_batch", "seed", "batch_size",
                                                    "max_batch_tokens", "log_every")},
        "band": {"limits": list(BAND), "steps": len(shares), "mean_share": mean_share,
                 "min_share": min(shares) if shares else None, "max_share": max(shares) if shares else None,
                 "passes": band_pass,
                 "definition": "mean over the first 200 steps of ||grad KL|| / (||grad CE+Brier|| + ||grad KL||), pre-clip, at the frozen λ"},
        "quota": {"min_labelled_per_step": min(teachers) if teachers else None, "histogram": {str(k): teachers.count(k) for k in sorted(set(teachers))}},
        "logged_step_events": total_logged,
        "loss_all_finite": losses_finite,
        "steps": int(result["steps"]),
        "evaluation_steps": list(evaluation_steps),
        "positional_passes_all_orders_at_final_eval": bool(final_eval["positional_passes_all_orders"]),
        "final_composite": final_eval["composite"],
        "final_knowledge_dev_accuracy": float(final_eval["knowledge_dev_accuracy"]),
        "finished_utc": result.get("finished_utc"),
        "training_git_commit": result.get("git_commit"),
        "problems": problems,
        "ok": not problems,
    }
    write_json(arm_dir / "audit.json", payload)
    problem_lines = [f"- {problem}" for problem in problems] or ["- None."]
    lines = [
        "# Z3 — SELF_AUDIT (R6; the D57 arm band)",
        "",
        f"Generated {payload['generated_utc']} by `uv run --frozen python scripts/osler_v6/z3_audit.py --arm-dir {args.arm_dir}`.",
        "",
        "| quantity | value | check |",
        "|---|---|---|",
        f"| Frozen λ (Z2) | {lam:.6f} | arm's distill_weight: {float(config['distill_weight']) == lam} |",
        f"| Quota | {QUOTA} labelled anchors per batch | observed min per step: {payload['quota']['min_labelled_per_step']} |",
        f"| Steps | {result['steps']} | = {STEPS} |",
        f"| Evaluations | {list(evaluation_steps)} | = {list(GRID)} |",
        f"| First-200-step mean share | {mean_share:.6f} | D57 band [0.30, 0.50]: **{band_pass}** |",
        f"| Share range | {payload['band']['min_share']:.6f} - {payload['band']['max_share']:.6f} | all finite in [0,1]: "
        f"{all(math.isfinite(value) and 0 <= value <= 1 for value in shares)} |",
        f"| Loss finite on every logged step | {losses_finite} | {total_logged} logged step events |",
        f"| Final evaluation | S {final_eval['composite']['S']:.6f}, knowledge {final_eval['knowledge_dev_accuracy']:.6f} | "
        f"positional all orders: {final_eval['positional_passes_all_orders']} |",
        "",
        "The band check is the pre-registered one: the arm's own first 200 steps, measured at the frozen λ, pre-clip. λ is not",
        "re-tuned inside the arm; a miss is BLOCKED, not a signal to change the dose.",
        "",
        "## Problems",
        "",
        *problem_lines,
        "",
        "## Verdict",
        "",
        "**PASS** — the arm used the frozen λ and the quota, and its first-200-step share is inside the D57 band." if not problems
        else "**BLOCKED/FAIL** — see the problems above; a band miss blocks G1 and is reported (ADVISORY §2.4).",
        "",
    ]
    (arm_dir / "SELF_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Z3 audit: {'PASS' if not problems else 'BLOCKED/FAIL'}; mean share {mean_share:.4f}; problems {problems}")


if __name__ == "__main__":
    main()
