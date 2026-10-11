"""Read-only summaries of a training run's log (``logs/train.jsonl``) for the O6-O8 and O10 monitoring (ADVISORY §8.3).

Pure functions over the log's rows: the dev evaluations (the selection statistic is ``template_macro_accuracy``; the per-letter
``macro_accuracy`` is a diagnostic), the tripwire count, and the first and last windows of step records (loss, CE, Brier,
batch accuracy, grad-norm quantiles). Nothing here writes to a run directory.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from typing import Any


def eval_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per dev evaluation: step and the metrics the run selects and reports on."""
    out = []
    for row in rows:
        if row.get("event") != "eval":
            continue
        m = row["metrics"]
        out.append({"step": int(row["step"]), "template_macro_accuracy": float(m["template_macro_accuracy"]),
                    "macro_accuracy": float(m["macro_accuracy"]), "accuracy": float(m["accuracy"]),
                    "brier": float(m["brier"]), "n": int(m["n"])})
    return out


def tripwire_count(rows: Iterable[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("event") == "tripwire")


def window_summary(steps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Means over one window of step records (logged every ``log_every`` steps) and the grad-norm quantiles."""
    if not steps:
        raise ValueError("window_summary needs at least one step record")
    grad = sorted(float(r["grad_norm"]) for r in steps)
    return {
        "records": len(steps),
        "first_step": int(steps[0]["step"]),
        "last_step": int(steps[-1]["step"]),
        "loss": statistics.fmean(float(r["loss"]) for r in steps),
        "ce": statistics.fmean(float(r["ce"]) for r in steps),
        "brier": statistics.fmean(float(r["brier"]) for r in steps),
        "batch_accuracy": statistics.fmean(float(r["accuracy"]) for r in steps),
        "grad_norm_p50": grad[len(grad) // 2],
        "grad_norm_p95": grad[min(len(grad) - 1, int(0.95 * len(grad)))],
        "grad_norm_max": grad[-1],
    }


def step_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("event") == "step"]


def multi_item_steps(steps: Iterable[dict[str, Any]], *, min_items: int = 4) -> list[dict[str, Any]]:
    """The step records whose batch holds at least ``min_items`` items.

    The token budget closes batches early on long records, so 1-item batches (mostly openFDA records, many with near-zero
    loss) would otherwise dominate a window's means. Drift checks read the multi-item windows as well.
    """
    return [row for row in steps if int(row["n_items"]) >= min_items]
