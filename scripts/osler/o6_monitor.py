"""Print a summary of a training run's log while it runs (read-only; ADVISORY §8.3; no GPU).

Shows every dev evaluation (the selection statistic first), the tripwire count, the best evaluation so far by the template
macro, and the first and last 20 step records side by side (a drift check on loss, CE, Brier, batch accuracy and grad norm).

Usage:
    uv run --frozen python scripts/osler/o6_monitor.py --run-dir outputs/osler_v0/O6/arm_L
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from meddecide.train.monitor import (
    eval_rows,
    multi_item_steps,
    step_rows,
    tripwire_count,
    window_summary,
)

REPO = Path(__file__).resolve().parents[2]
WINDOW = 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, help="the arm's output directory (holds logs/train.jsonl)")
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO / args.run_dir
    log = run_dir / "logs" / "train.jsonl"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    evals = eval_rows(rows)
    steps = step_rows(rows)
    print(f"run: {run_dir.relative_to(REPO) if run_dir.is_relative_to(REPO) else run_dir}")
    print(f"last step record: {steps[-1]['step'] if steps else 'none'}; tripwire events: {tripwire_count(rows)}")
    print("")
    print("| step | template macro (selection) | per-letter macro (diagnostic) | accuracy | Brier | n |")
    print("|---|---|---|---|---|---|")
    for e in evals:
        print(f"| {e['step']} | {e['template_macro_accuracy']:.4f} | {e['macro_accuracy']:.4f} | {e['accuracy']:.4f} | "
              f"{e['brier']:.4f} | {e['n']} |")
    if evals:
        best = max(evals, key=lambda e: (e["template_macro_accuracy"], -e["brier"]))
        print(f"\nbest evaluation so far by the selection rule (template macro, Brier tie-break): step {best['step']} "
              f"({best['template_macro_accuracy']:.4f})")
    if len(steps) >= 2 * WINDOW:
        print("")
        print("| window | steps | loss | CE | Brier | batch accuracy | grad norm p50 | p95 | max |")
        print("|---|---|---|---|---|---|---|---|---|")
        multi = multi_item_steps(steps)
        windows = [("first", steps[:WINDOW]), ("last", steps[-WINDOW:])]
        if len(multi) >= 2 * WINDOW:
            windows += [("first multi-item", multi[:WINDOW]), ("last multi-item", multi[-WINDOW:])]
        for name, window in windows:
            s = window_summary(window)
            print(f"| {name} {s['records']} | {s['first_step']}-{s['last_step']} | {s['loss']:.4f} | {s['ce']:.4f} | "
                  f"{s['brier']:.4f} | {s['batch_accuracy']:.3f} | {s['grad_norm_p50']:.2f} | {s['grad_norm_p95']:.2f} | "
                  f"{s['grad_norm_max']:.2f} |")
        singles = [r for r in steps if int(r["n_items"]) == 1]
        print(f"\nsteps with one item: {len(singles)} of {len(steps)} logged; the multi-item windows hold steps with at "
              "least 4 items (the token budget closes batches early on long records)")


if __name__ == "__main__":
    main()
