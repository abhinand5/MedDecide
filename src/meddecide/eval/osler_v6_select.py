"""Checkpoint selection for the osler_v6 final (ADVISORY §2.7, as amended by A1 on 2026-11-10).

**Amendment A1 (operator, 2026-11-10; supersedes the best-S rule for Z6):** the final checkpoint is the one at step
**63,304** and no other. It is chosen only if that evaluation passes positional health in all three orders; if it fails,
the final is BLOCKED and reported, and no other checkpoint is substituted. The intermediate evaluations are reported as a
dev trajectory, never as candidates. The reason is that the cosine schedule spans 63,304 steps: intermediate S values are
not comparable to a finished run's.

`select_final_step` is the rule for Z6. `select_by_s` is the retired best-S rule, kept with its tests for the loop's record;
the Z6 driver does not call it. The calibration of the chosen checkpoint is always on dev fast dev v4 only.

This module holds the pure rules; `scripts/osler_v6/z6_select.py` reads the final's evaluations, copies (and calibrates) the
chosen checkpoint and writes the record.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Selection:
    """The chosen evaluation step, its S and the eligibility table (every evaluation, in step order)."""

    chosen_step: int | None
    chosen_s: float | None
    outcome: str
    eligible: tuple[int, ...]
    table: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"chosen_step": self.chosen_step, "chosen_s": self.chosen_s, "outcome": self.outcome,
                "eligible_steps": list(self.eligible), "evaluations": list(self.table)}


FINAL_STEP = 63_304  # ADVISORY §2.7 Amendment A1: the final checkpoint is the final at 63,304 only


def evaluation_table(evaluations: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Every evaluation, in step order, with its S, composite parts, positional pass and knowledge accuracy (no drops)."""
    table = []
    for key in sorted(evaluations, key=lambda k: int(k)):
        step = int(key)
        record = evaluations[key]
        parts = record["composite"]
        table.append({"step": step, "S": float(parts["S"]), "mr": float(parts["mr"]), "mimic": float(parts["mimic"]),
                      "cg_micro": float(parts["cg_micro"]),
                      "positional_passes_all_orders": bool(record["positional_passes_all_orders"]),
                      "knowledge_dev_accuracy": float(record.get("knowledge_dev_accuracy", float("nan")))})
    return tuple(table)


def failed_positional_orders(record: Mapping[str, Any]) -> list[str]:
    """The option orders whose positional-health check fails (the original and each seeded permutation)."""
    failed = []
    original = record.get("positional_original") or {}
    if original and not bool(original.get("passes")):
        failed.append("original")
    for seed, detail in sorted((record.get("positional_permuted") or {}).items()):
        if not bool(detail.get("passes")):
            failed.append(f"seed{seed}")
    return failed


def select_final_step(evaluations: Mapping[str, Any], *, step: int = FINAL_STEP) -> Selection:
    """Apply ADVISORY §2.7 Amendment A1: the evaluation at ``step`` only, and only if it passes positional health.

    Every evaluation stays in the returned table (the dev trajectory); the final step is the only candidate. A missing or
    failing final evaluation returns a BLOCKED Selection with ``chosen_step`` None — never an earlier checkpoint.
    """
    if not evaluations:
        raise ValueError("no evaluations to select from")
    table = evaluation_table(evaluations)
    record = next((evaluations[key] for key in evaluations if int(key) == step), None)
    if record is None:
        return Selection(None, None, f"BLOCKED — the final evaluation at step {step:,} is not present",
                         (), table)
    if not bool(record["positional_passes_all_orders"]):
        failed = failed_positional_orders(record)
        return Selection(None, None, f"BLOCKED — the step-{step:,} evaluation fails positional health"
                         + (f" ({', '.join(failed)})" if failed else ""), (), table)
    s = float(record["composite"]["S"])
    return Selection(step, s, f"chosen step {step:,}: the final checkpoint (A1), positional health passes in all three orders",
                     (step,), table)


def select_by_s(evaluations: Mapping[str, Any]) -> Selection:
    """The retired best-S rule (kept for the loop's record; the Z6 driver does not call it under A1)."""
    if not evaluations:
        raise ValueError("no evaluations to select from")
    table = []
    eligible: list[tuple[int, float]] = []
    for key in sorted(evaluations, key=lambda k: int(k)):
        step = int(key)
        record = evaluations[key]
        parts = record["composite"]
        s = float(parts["S"])
        passes = bool(record["positional_passes_all_orders"])
        table.append({"step": step, "S": s, "mr": float(parts["mr"]), "mimic": float(parts["mimic"]),
                      "cg_micro": float(parts["cg_micro"]), "positional_passes_all_orders": passes,
                      "knowledge_dev_accuracy": float(record.get("knowledge_dev_accuracy", float("nan")))})
        if passes:
            eligible.append((step, s))
    if not eligible:
        return Selection(None, None, "BLOCKED — no evaluation passes positional health in all three orders",
                         (), tuple(table))
    best_s = max(s for _, s in eligible)
    best_steps = sorted(step for step, s in eligible if s == best_s)
    if len(best_steps) > 1:
        raise ValueError(f"S is tied between the evaluations {best_steps}: the pre-registration names no tie-break")
    return Selection(best_steps[0], best_s,
                     f"chosen step {best_steps[0]}: the highest S ({best_s:.6f}) among the {len(eligible)} positional-health passes",
                     tuple(step for step, _ in eligible), tuple(table))
