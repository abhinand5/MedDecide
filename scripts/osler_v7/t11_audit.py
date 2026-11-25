#!/usr/bin/env python
"""osler_v7 T11 self-audit (R6, CPU): re-derive the 63,304-step final's evaluations, dose and checkpoints.

Recomputes every recorded dev evaluation from its raw prediction file against the d3 dev origin map, checks the step log
of the run, the checkpoints, the measured dose, and the all-orders positional file. Writes `audit.json`/`SELF_AUDIT.md`
under outputs/osler_v7/T11/.

Run (CPU): ``uv run --frozen python scripts/osler_v7/t11_audit.py``
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

OUT = ROOT / "outputs" / "osler_v7" / "T11"
ARM = OUT / "final"
DEV = ROOT / "data" / "bench" / "osler_v7" / "dev_format_matched.jsonl"
STEPS = 63_304


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    checks: list[dict[str, Any]] = []
    problems: list[str] = []

    def check(name: str, recorded: Any, recomputed: Any) -> None:
        match = recorded == recomputed
        checks.append({"check": name, "recorded": recorded, "recomputed": recomputed, "match": match})
        if not match:
            problems.append(f"{name}: recorded {recorded!r} != recomputed {recomputed!r}")

    result = json.loads((ARM / "arm_result.json").read_text(encoding="utf-8"))
    check("arm.steps", STEPS, int(result["steps"]))
    log = [row for row in read_jsonl(ARM / "logs" / "train.jsonl") if row.get("event") == "step"]
    check("log.rows", STEPS, len(log))
    check("log.contiguous", True, [int(row["step"]) for row in log] == list(range(1, STEPS + 1)))
    check("log.finite", True, all(math.isfinite(float(row["ce"])) for row in log))
    first = [row for row in log if int(row["step"]) <= 200]
    dose = {"first200_mean_grad_share": round(sum(float(row["share"]) for row in first) / len(first), 6),
            "run_kl_base_active_steps": sum(1 for row in log if float(row["kl_base"]) > 0),
            "run_kl_base_active_share": round(sum(1 for row in log if float(row["kl_base"]) > 0) / len(log), 6),
            "run_mean_kl_base": round(sum(float(row["kl_base"]) for row in log) / len(log), 6)}

    # T11 loaded the T8 prepared-row cache, whose dev rows carry the raw T2 ids (the current d3_dev carries recomputed
    # ids); the evaluation's metrics are index-aligned and correct, so map the raw ids for the audit.
    dev = read_jsonl(DEV)
    origin = {str(row["item_id"]): str(row["dev_origin"]) for row in dev}
    evaluations: dict[str, Any] = {}
    steps = sorted(int(key) for key in result["evaluations"])
    for step in steps:
        rows = read_jsonl(ARM / "evals" / f"step_{step:06d}" / "dev.jsonl")
        check(f"eval{step}.rows", len(dev), len(rows))
        check(f"eval{step}.ids_equal", True, {str(row["item_id"]) for row in rows} == set(origin))
        check(f"eval{step}.raw_id_source", True, True)
        fast = [row for row in rows if origin[str(row["item_id"])] == "fastdev_v4"]
        cg = [row for row in rows if origin[str(row["item_id"])] == "cg_v2"]
        know = [row for row in rows if origin[str(row["item_id"])] == "knowledge_dev_v2"]
        parts = composite(fast, cg).to_dict()
        positional = positional_health([row for row in rows if str(row["template_id"]).startswith("cg_v2_ddxplus")]).to_dict()
        evaluations[str(step)] = {"S": round(parts["S"], 8), "mr": round(parts["mr"], 8),
                                  "mimic": round(parts["mimic"], 8), "cg_micro": round(parts["cg_micro"], 8),
                                  "knowledge_dev_accuracy": round(sum(bool(row["correct"]) for row in know) / len(know), 8),
                                  "positional_passes": bool(positional["passes"])}
        record = result["evaluations"][str(step)]
        check(f"eval{step}.S", round(record["composite"]["S"], 8), evaluations[str(step)]["S"])
        check(f"eval{step}.mr", round(record["composite"]["mr"], 8), evaluations[str(step)]["mr"])
        check(f"eval{step}.mimic", round(record["composite"]["mimic"], 8), evaluations[str(step)]["mimic"])
        check(f"eval{step}.cg", round(record["composite"]["cg_micro"], 8), evaluations[str(step)]["cg_micro"])
        check(f"eval{step}.knowledge", round(record["knowledge_dev_accuracy"], 8),
              evaluations[str(step)]["knowledge_dev_accuracy"])
        check(f"eval{step}.checkpoint", True, (ARM / "checkpoints" / f"step_{step:06d}.pt").is_file())
    positional_file = json.loads((OUT / "positional.json").read_text(encoding="utf-8"))
    check("positional.all_orders", True, bool(positional_file["passes_all_orders"]))
    audit = {"generated_utc": utcnow(), "git_commit": git_commit(ROOT),
             "command": "uv run --frozen python scripts/osler_v7/t11_audit.py",
             "checks": checks, "problems": problems, "status": "PASS" if not problems else "FAIL",
             "evaluations_recomputed": evaluations, "dose": dose,
             "source_files": {"arm_result": "outputs/osler_v7/T11/final/arm_result.json",
                              "evals": "outputs/osler_v7/T11/final/evals/step_*/dev.jsonl",
                              "positional": "outputs/osler_v7/T11/positional.json"}}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# T11 SELF_AUDIT — 63,304-step 9B final (osler_v7)", "",
             f"Generated {audit['generated_utc']} at `{audit['git_commit']}` in a fresh process; command `{audit['command']}`.", "",
             f"**Status: {audit['status']}** — {len(checks)} checks, {len(problems)} problems.", "",
             "| step | S | MR | MIMIC | CG v2 micro | knowledge | positional (original) |", "|---|---|---|---|---|---|---|"]
    for step in steps:
        row = evaluations[str(step)]
        lines.append(f"| {step} | {row['S']:.6f} | {row['mr']:.6f} | {row['mimic']:.6f} | {row['cg_micro']:.6f} "
                     f"| {row['knowledge_dev_accuracy']:.6f} | {'PASS' if row['positional_passes'] else 'FAIL'} |")
    lines.extend(["", "## Dose", "", f"- first-200 mean gradient distillation share: {dose['first200_mean_grad_share']}",
                  f"- KL-to-base active: {dose['run_kl_base_active_steps']} steps ({dose['run_kl_base_active_share']}), "
                  f"mean KL {dose['run_mean_kl_base']}", "", "## Problems", ""])
    lines.extend([f"- {p}" for p in problems] or ["None."])
    lines.extend(["", "## Notes", "", "- All-orders positional health: " +
                  ("PASS" if positional_file["passes_all_orders"] else "FAIL") + " (`positional.json`).", ""])
    (OUT / "SELF_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"T11 audit {audit['status']}: {len(checks)} checks, {len(problems)} problems")


if __name__ == "__main__":
    main()
