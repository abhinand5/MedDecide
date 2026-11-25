#!/usr/bin/env python
"""osler_v7 T8 self-audit (R6, CPU): re-derive the 9B arm's step log, dose and dev evaluations from the raw files.

The in-run evaluator wrote empty composite partitions because the converted dev file predated the id/origin fix; its
prediction files are intact and keyed by the raw T2 dev item ids. This audit re-aggregates those files against the T2
origin map, re-derives the step log of the final contiguous run (the file also holds rows from the failed attempts),
checks the checkpoints, and reports the original-order positional health. All-orders positional health is measured
separately in T9.

Run (CPU): ``uv run --frozen python scripts/osler_v7/t8_audit.py``
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.composite import composite, positional_health  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

OUT = ROOT / "outputs" / "osler_v7" / "T8"
DEV = ROOT / "data" / "bench" / "osler_v7" / "dev_format_matched.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def final_run(rows: list[dict[str, Any]], steps: int) -> list[dict[str, Any]]:
    start = max(i for i, row in enumerate(rows) if int(row.get("step", -1)) == 1)
    return [row for row in rows[start:] if "step" in row]


def main() -> None:
    arm = OUT / "arm"
    checks: list[dict[str, Any]] = []
    problems: list[str] = []

    def check(name: str, recorded: Any, recomputed: Any) -> None:
        match = recorded == recomputed
        checks.append({"check": name, "recorded": recorded, "recomputed": recomputed, "match": match})
        if not match:
            problems.append(f"{name}: recorded {recorded!r} != recomputed {recomputed!r}")

    result = json.loads((arm / "arm_result.json").read_text(encoding="utf-8"))
    steps = int(result["steps"])
    check("arm.steps", 13_992, steps)
    log = final_run(read_jsonl(arm / "logs" / "train.jsonl"), steps)
    check("log.final_run_rows", steps, len(log))
    check("log.first_last", (1, steps), (int(log[0]["step"]), int(log[-1]["step"])))
    check("log.contiguous", True, [int(row["step"]) for row in log] == list(range(1, steps + 1)))
    check("log.losses_finite", True, all(math.isfinite(float(row["ce"])) for row in log))
    first = [row for row in log if int(row["step"]) <= 200]
    share = sum(float(row["share"]) for row in first) / len(first)
    kl_active = sum(1 for row in log if float(row["kl_base"]) > 0)
    dose = {
        "first200_mean_grad_share": round(share, 6),
        "run_kl_base_active_steps": kl_active,
        "run_kl_base_active_share": round(kl_active / len(log), 6),
        "run_mean_kl_base": round(sum(float(row["kl_base"]) for row in log) / len(log), 6),
        "first200_mean_kl_teacher": round(sum(float(row["kl_teacher"]) for row in first) / len(first), 6),
    }
    check("checkpoints.eval_steps", True,
          all((arm / "checkpoints" / f"step_{step:06d}.pt").is_file() for step in (6_996, 13_992)))

    dev = read_jsonl(DEV)
    origin = {str(row["item_id"]): str(row["dev_origin"]) for row in dev}
    evaluations: dict[str, Any] = {}
    for step in (6_996, 13_992):
        rows = read_jsonl(arm / "evals" / f"step_{step:06d}" / "dev.jsonl")
        check(f"eval{step}.rows", len(dev), len(rows))
        check(f"eval{step}.ids_matched", True, all(str(row["item_id"]) in origin for row in rows))
        fast = [row for row in rows if origin[str(row["item_id"])] == "fastdev_v4"]
        cg = [row for row in rows if origin[str(row["item_id"])] == "cg_v2"]
        know = [row for row in rows if origin[str(row["item_id"])] == "knowledge_dev_v2"]
        parts = composite(fast, cg).to_dict()
        positional = positional_health([row for row in rows if str(row["template_id"]).startswith("cg_v2_ddxplus")]).to_dict()
        evaluations[str(step)] = {
            "S": round(parts["S"], 8), "mr": round(parts["mr"], 8), "mimic": round(parts["mimic"], 8),
            "cg_micro": round(parts["cg_micro"], 8),
            "knowledge_dev_accuracy": round(sum(bool(row["correct"]) for row in know) / len(know), 8),
            "positional_original": {key: (round(value, 8) if isinstance(value, float) else value)
                                    for key, value in positional.items()},
        }
        check(f"eval{step}.partition_sizes", (14_597, 7_508, 829), (len(fast), len(cg), len(know)))
    audit = {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t8_audit.py",
        "checks": checks,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
        "evaluations_recomputed": evaluations,
        "dose": dose,
        "notes": [
            "the in-run evaluator recorded empty composite partitions because the converted dev rows then lacked "
            "dev_origin/raw ids; the prediction files are intact and were re-aggregated here against the T2 origin map",
            "all-orders positional health is measured in T9 (ADVISORY §9)",
        ],
        "source_files": {"arm_result": "outputs/osler_v7/T8/arm/arm_result.json",
                         "train_log": "outputs/osler_v7/T8/arm/logs/train.jsonl",
                         "evals": "outputs/osler_v7/T8/arm/evals/step_{006996,013992}/dev.jsonl",
                         "checkpoints": "outputs/osler_v7/T8/arm/checkpoints/"},
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# T8 SELF_AUDIT — 9B arm on d3-flash (osler_v7)",
        "",
        f"Generated {audit['generated_utc']} at `{audit['git_commit']}` in a fresh process; command `{audit['command']}`.",
        "",
        f"**Status: {audit['status']}** — {len(checks)} checks, {len(problems)} problems.",
        "",
        "## Dev evaluations (re-aggregated from the raw prediction files)",
        "",
        "| step | S | MR | MIMIC | CG v2 micro | knowledge | positional (original) |",
        "|---|---|---|---|---|---|---|",
    ]
    for step, row in evaluations.items():
        pos = row["positional_original"]
        lines.append(f"| {step} | {row['S']:.6f} | {row['mr']:.6f} | {row['mimic']:.6f} | {row['cg_micro']:.6f} "
                     f"| {row['knowledge_dev_accuracy']:.6f} | two {pos['two_letter_accuracy']:.6f} / single "
                     f"{pos['single_letter_accuracy']:.6f}, {'PASS' if pos['passes'] else 'FAIL'} |")
    lines.extend([
        "",
        "## Dose",
        "",
        f"- first-200 mean gradient distill share: {dose['first200_mean_grad_share']}",
        f"- KL-to-base active steps: {dose['run_kl_base_active_steps']} ({dose['run_kl_base_active_share']} of the run), "
        f"mean KL {dose['run_mean_kl_base']}",
        "",
        "## Problems",
        "",
    ])
    lines.extend([f"- {problem}" for problem in problems] or ["None."])
    lines.extend(["", "## Notes", ""])
    lines.extend([f"- {note}" for note in audit["notes"]])
    (OUT / "SELF_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"T8 audit {audit['status']}: {len(checks)} checks, {len(problems)} problems")
    print(json.dumps(evaluations, indent=1))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
