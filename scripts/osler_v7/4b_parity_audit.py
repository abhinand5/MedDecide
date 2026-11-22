#!/usr/bin/env python
"""osler_v7 4B full run, Step 1: a fresh-process re-derivation of the parity numbers (R6 self-audit).

Every check below recomputes a headline number from the raw ``logs/train.jsonl`` records with its own code (no helper is
imported from the report script), then compares it with ``parity.json`` and with what the batch plan predicts. It writes
``SELF_AUDIT.md`` next to the report and exits non-zero if any check fails.

Usage::

    uv run python scripts/osler_v7/4b_parity_audit.py \
        --run old=outputs/osler_v7/4B_FULL/parity/old \
        --run token=outputs/osler_v7/4B_FULL/parity/token \
        --parity outputs/osler_v7/4B_FULL/parity/parity.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.train.token_plan import build_token_plan

REPO = Path(__file__).resolve().parents[2]
CAP = 8_192


def resolve(path: str | Path) -> Path:
    return Path(path) if Path(path).is_absolute() else REPO / Path(path)


def records_of(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = (run_dir / "logs" / "train.jsonl").read_text(encoding="utf-8")
    for line in text.split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "step":
            rows.append(record)
    return rows


def load_plan_rows(path: Path, steps: int) -> list[int]:
    """The item counts of the plan's first ``steps`` batches (the step -> batch map is the plan's order)."""
    with np.load(path, allow_pickle=False) as data:
        offsets = data["offsets"]
    return [int(offsets[i + 1] - offsets[i]) for i in range(min(steps, len(offsets) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--parity", default="outputs/osler_v7/4B_FULL/parity/parity.json")
    parser.add_argument("--plan", default="outputs/osler_v7/4B_FULL/plan.npz")
    parser.add_argument("--out", default="outputs/osler_v7/4B_FULL/parity/SELF_AUDIT.md")
    args = parser.parse_args()
    parity = json.loads(resolve(args.parity).read_text())
    plan_path = resolve(args.plan)
    lines = ["# 4B full run - Step 1 parity self-audit", ""]
    failures = 0
    checks = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal failures, checks
        checks += 1
        if not ok:
            failures += 1
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] **{name}** - {detail}")

    for item in args.run:
        name, _, path = item.partition("=")
        run_dir = resolve(path)
        records = records_of(run_dir)
        summary = parity["runs"][name]
        steps = len(records)
        items = [int(record["n_items"]) for record in records]
        real = [int(record["real_tokens"]) for record in records]
        padded = [int(record["padded_tokens"]) for record in records]
        kl = [float(record.get("kl", 0.0)) for record in records]
        teacher_rows = [int(record.get("n_teacher", 0)) for record in records]
        ce = [float(record["ce"]) for record in records]
        elapsed = [float(record["elapsed_s"]) for record in records]
        lines.append(f"## `{name}` ({summary['batching']} batching, quota {summary['quota']})")
        lines.append("")
        check(f"{name}: step count", steps == summary["steps_logged"] == summary["steps_requested"],
              f"{steps} records, report {summary['steps_logged']}, request {summary['steps_requested']}")
        check(f"{name}: step numbers are 1..n", [int(r["step"]) for r in records] == list(range(1, steps + 1)),
              f"first {records[0]['step']}, last {records[-1]['step']}")
        check(f"{name}: items per step mean", math.isclose(sum(items) / steps, summary["items_per_step_mean"], rel_tol=0, abs_tol=1e-9),
              f"{sum(items) / steps:.6f} vs {summary['items_per_step_mean']:.6f}")
        check(f"{name}: real tokens per step mean",
              math.isclose(math.fsum(real) / steps, summary["real_tokens_per_step_mean"], abs_tol=1e-6),
              f"{math.fsum(real) / steps:.3f} vs {summary['real_tokens_per_step_mean']:.3f}")
        check(f"{name}: padded tokens per step mean",
              math.isclose(math.fsum(padded) / steps, summary["padded_tokens_per_step_mean"], abs_tol=1e-6),
              f"{math.fsum(padded) / steps:.3f} vs {summary['padded_tokens_per_step_mean']:.3f}")
        active = sum(1 for value in kl if value > 1e-12)
        check(f"{name}: KL-active count", active == summary["kl_active_steps"] and active / steps == summary["kl_active_share"],
              f"{active}/{steps} = {active / steps:.6f} vs {summary['kl_active_share']:.6f}")
        active_kl = [value for value in kl if value > 1e-12]
        check(f"{name}: mean KL on active steps",
              not active_kl or math.isclose(math.fsum(active_kl) / len(active_kl), summary["kl_mean_active_steps"], abs_tol=1e-9),
              f"{math.fsum(active_kl) / len(active_kl) if active_kl else float('nan'):.9f}")
        t_active = sum(1 for count in teacher_rows if count > 0)
        check(f"{name}: distillation-active count", t_active == summary["teacher_active_steps"],
              f"{t_active}/{steps} = {t_active / steps:.4%}")
        check(f"{name}: mean CE", math.isclose(math.fsum(ce) / steps, summary["ce_mean"], abs_tol=1e-9),
              f"{math.fsum(ce) / steps:.9f}")
        check(f"{name}: mean seconds per step", math.isclose(math.fsum(elapsed) / steps, summary["seconds_per_step_mean"], abs_tol=1e-6),
              f"{math.fsum(elapsed) / steps:.3f}s")
        # the token plan caps *padded* tokens, so only a single row longer than the cap may exceed it. The old batching caps
        # the sum of *real* tokens instead, so its padded count runs over the cap by design (recorded, not a failure).
        over = [(index, n, p) for index, (n, p) in enumerate(zip(items, padded, strict=True)) if p > CAP]
        if summary["batching"] == "token":
            allowed = all(n == 1 for _, n, _ in over)
            check(f"{name}: padded-token cap", allowed,
                  f"{len(over)} steps over {CAP} padded tokens, all single-row: {allowed}")
        else:
            check(f"{name}: old batching caps real tokens, not padding", True,
                  f"{len(over)}/{steps} steps exceed {CAP} padded tokens (multi-row batches; "
                  f"max {max(padded)}), which is why the plan exists")
        if summary["batching"] == "old" and summary["quota"]:
            check(f"{name}: quota realised", all(count == summary["quota"] for count in teacher_rows),
                  f"n_teacher values {sorted(set(teacher_rows))} (quota {summary['quota']})")
        if summary["batching"] == "token":
            plan_counts = load_plan_rows(plan_path, steps)
            check(f"{name}: items follow the plan", plan_counts == items,
                  f"first {items[:5]} vs plan {plan_counts[:5]}; totals {sum(items)} vs {sum(plan_counts)}")
            dose = summary.get("plan_dose") or {}
            expected = float(dose.get("steps_with_v7_c_share", float("nan")))
            measured = active / steps
            sd = math.sqrt(max(expected * (1 - expected), 1e-9) / steps)
            check(f"{name}: KL share matches the plan's dose", abs(measured - expected) <= 4 * sd + 0.01,
                  f"measured {measured:.4%}, plan {expected:.4%}, 4 sd = {4 * sd:.4%}")
            expected_teacher = float(dose.get("steps_with_teacher_share", float("nan")))
            sd_teacher = math.sqrt(max(expected_teacher * (1 - expected_teacher), 1e-9) / steps)
            check(f"{name}: distillation share matches the plan's dose", abs(t_active / steps - expected_teacher) <= 4 * sd_teacher + 0.01,
                  f"measured {t_active / steps:.4%}, plan {expected_teacher:.4%}")
        lines.append("")

    # the port is the 9B rule: with no padding quantum (chunk=1) the plan must equal t8_run_arm.TokenBatches batch for batch
    import importlib.util
    import random

    spec = importlib.util.spec_from_file_location("t8_run_arm", str(REPO / "scripts" / "osler_v7" / "t8_run_arm.py"))
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load scripts/osler_v7/t8_run_arm.py for the equivalence check")
    t8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t8)
    rng = random.Random(0)
    same = True
    for _ in range(3):
        lengths = [rng.choice([50, 200, 800, 1500, 4000, 9000, 16000]) + rng.randint(0, 60) for _ in range(400)]
        nine = t8.TokenBatches(lengths, batch=8, max_tokens=8192, seed=0)
        plan = build_token_plan(lengths, batch_size=8, max_tokens=8192, seed=0, chunk=1)
        same = same and [list(batch) for batch in nine.batches] == [list(batch) for batch in plan.batches]
    check("port: the plan equals the 9B TokenBatches at chunk=1", same,
          "three 400-row synthetic mixes, batch 8 / 8,192 tokens, seed 0, batch for batch")

    mechanism = parity["mechanism"]
    check("mechanism: T6 anchors per step equal 3 x the v7_c share of the labelled pool",
          math.isclose(mechanism["t6_mean_anchor_rows"], mechanism["mean_anchor_rows_implied"], abs_tol=0.01),
          f"{mechanism['t6_mean_anchor_rows']:.4f} vs {mechanism['mean_anchor_rows_implied']:.4f}")
    check("mechanism: the three-anchor prediction matches T6's measured KL share",
          math.isclose(mechanism["expected_share_from_three_anchors"], mechanism["t6_kl_active_share"], abs_tol=0.02),
          f"{mechanism['expected_share_from_three_anchors']:.4%} vs {mechanism['t6_kl_active_share']:.4%}")
    check("mechanism: the T6 quota filled every step", math.isclose(mechanism["t6_mean_teacher_rows"], 3.0, abs_tol=1e-9),
          f"mean teacher rows {mechanism['t6_mean_teacher_rows']:.4f}")
    lines.append("")
    lines.append(f"**{checks - failures}/{checks} checks pass.**")
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
