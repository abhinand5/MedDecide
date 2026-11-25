#!/usr/bin/env python
"""osler_v7 T6 self-audit (R6, CPU): re-derive the arm's health and dev evaluations from the raw logs.

Fresh process. Checks the step log (contiguous steps to the budget, finite losses and gradient norms, the D57 teacher
quota and the distillation gradient share over the first 200 steps), the KL-to-base coverage of the v7_c pool
(active-step share, anchors per step), the evaluation records (recomputing the S composite, knowledge accuracy and
positional results from the per-evaluation prediction files), and the checkpoints. Writes `audit.json` and
`SELF_AUDIT.md`.

Run (CPU; after the arm): ``uv run --frozen python scripts/osler_v7/t6_audit.py``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.composite import composite  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

OUT = ROOT / "outputs" / "osler_v7" / "T6"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, default=OUT / "arm")
    parser.add_argument("--steps", type=int, default=13_992)
    args = parser.parse_args()
    checks: list[dict[str, Any]] = []
    problems: list[str] = []

    def check(name: str, recorded: Any, recomputed: Any) -> None:
        match = recorded == recomputed
        checks.append({"check": name, "recorded": recorded, "recomputed": recomputed, "match": match})
        if not match:
            problems.append(f"{name}: recorded {recorded!r} != recomputed {recomputed!r}")

    raw_log = read_jsonl(args.arm / "logs" / "train.jsonl")
    log = [row for row in raw_log if row.get("event") == "step"]
    steps = [int(row["step"]) for row in log]
    check("log.rows", args.steps, len(log))
    check("log.first_last", (1, args.steps), (steps[0], steps[-1]))
    check("log.contiguous", True, steps == list(range(1, args.steps + 1)))
    check("log.losses_finite", True, all(math.isfinite(float(row["loss"])) for row in log))
    check("log.grads_finite", True, all(math.isfinite(float(row["grad_norm"])) for row in log))
    first = [row for row in log if int(row["step"]) <= 200]
    check("first200.rows", 200, len(first))
    check("first200.teacher_quota", True, all(int(row["n_teacher"]) >= 3 for row in first))
    share = sum(float(row["grad_share"]) for row in first) / len(first)
    kl_active = sum(1 for row in first if float(row["kl"]) > 0)

    result = json.loads((args.arm / "arm_result.json").read_text(encoding="utf-8"))
    check("arm_result.steps", args.steps, int(result["steps"]))
    evaluations = {int(key): value for key, value in result["evaluations"].items()}
    check("evaluations.keys", [6_996, args.steps], sorted(evaluations))
    for step, record in evaluations.items():
        folder = args.arm / "evals" / f"step_{step:06d}"
        fast = read_jsonl(folder / "fast_dev.jsonl")
        cg = read_jsonl(folder / "cg_v2.jsonl")
        know = read_jsonl(folder / "knowledge_dev.jsonl")
        parts = composite(fast, cg)
        check(f"eval{step}.S", record["composite"]["S"], parts.to_dict()["S"])
        check(f"eval{step}.mr", record["composite"]["mr"], parts.to_dict()["mr"])
        check(f"eval{step}.mimic", record["composite"]["mimic"], parts.to_dict()["mimic"])
        check(f"eval{step}.cg_micro", record["composite"]["cg_micro"], parts.to_dict()["cg_micro"])
        knowledge = sum(bool(row["correct"]) for row in know) / len(know)
        check(f"eval{step}.knowledge", record["knowledge_dev_accuracy"], knowledge)
        check(f"eval{step}.positional", record["positional_passes_all_orders"],
              bool(record["positional_original"]["passes"] and all(
                  record["positional_permuted"][seed]["passes"] for seed in ("1", "2"))))
        check(f"eval{step}.checkpoint", True, (args.arm / "checkpoints" / f"step_{step:06d}").is_dir())
        check(f"eval{step}.files", True, all(path.is_file() for path in
                                              (folder / "fast_dev.jsonl", folder / "cg_v2.jsonl",
                                               folder / "knowledge_dev.jsonl", folder / "positional_seed1.jsonl",
                                               folder / "positional_seed2.jsonl")))
    check("checkpoints.final", True, (args.arm / "checkpoints" / f"final_step_{args.steps:06d}").is_dir())
    audit = {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t6_audit.py",
        "checks": checks,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
        "dose": {"first200_mean_distill_share": round(share, 6),
                 "first200_kl_active_steps": kl_active,
                 "run_kl_active_steps": sum(1 for row in log if float(row["kl"]) > 0),
                 "run_kl_active_share": round(sum(1 for row in log if float(row["kl"]) > 0) / len(log), 6),
                 "run_mean_kl": round(sum(float(row["kl"]) for row in log) / len(log), 6),
                 "run_mean_n_anchor": round(sum(int(row["n_anchor"]) for row in log) / len(log), 6)},
    }
    (args.arm.parent / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# T6 SELF_AUDIT — 4B arm (osler_v7)",
        "",
        f"Generated {audit['generated_utc']} at `{audit['git_commit']}` in a fresh process; command `{audit['command']}`.",
        "",
        f"**Status: {audit['status']}** — {len(checks)} checks, {len(problems)} problems.",
        "",
        f"First-200-step distillation share {share:.6f}; KL-to-base active on {kl_active}/200 steps "
        f"({sum(1 for row in log if float(row['kl']) > 0)}/{len(log)} over the run, mean KL "
        f"{audit['dose']['run_mean_kl']:.6f}).",
        "",
        "| check | recorded | re-derived | match |",
        "|---|---|---|---|",
    ]
    for row in checks:
        lines.append(
            f"| {row['check']} | `{row['recorded']}` | `{row['recomputed']}` | {'yes' if row['match'] else '**NO**'} |"
        )
    lines.extend(["", "## Problems", ""])
    lines.extend([f"- {problem}" for problem in problems] or ["None."])
    lines.extend(["", "## Notes", "",
                  "- The v7 ADVISORY does not pre-register a distillation share band or a KL active-share target; the",
                  "  measured values are reported, and the v6 D57 0.30-0.50 band is shown only as context.",
                  ""])
    (args.arm.parent / "SELF_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"T6 audit {audit['status']}: {len(checks)} checks, {len(problems)} problems")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
