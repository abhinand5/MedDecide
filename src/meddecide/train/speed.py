"""Batching speed recipe for osler_v1 (ADVISORY V0 step 4; D26).

Pure functions that turn the per-step records of a short training run into the numbers the recipe is judged on, and
apply the pre-registered choice rule. The GPU driver is ``scripts/osler_v1/v0_speed.py``; everything here is testable
without a GPU.

The rule, fixed before any measurement: among the configurations that finish the target number of steps and whose peak
reserved GPU memory is at most ``MEMORY_CAP_GB``, choose the one with the highest wall-clock items per second over the
steady-state window (every step after the first ``WARMUP_STEPS``). Wall-clock includes batch encoding, so it is the
time a real run pays, not only the optimiser-step time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

MEMORY_CAP_GB = 85.0
WARMUP_STEPS = 10
LOSS_WINDOW = 50


def summarise(records: Sequence[Mapping[str, Any]], *, warmup: int = WARMUP_STEPS,
              loss_window: int = LOSS_WINDOW) -> dict[str, Any]:
    """Throughput, padding efficiency, peak memory and the loss over the last ``loss_window`` steps of one run.

    Each record is one optimiser step with keys ``step`` (1-based), ``n_items``, ``real_tokens``, ``padded_tokens``,
    ``loss``, ``elapsed_s`` (the step's compute time), ``t_end_s`` (seconds since the run started, at the end of the step)
    and ``peak_reserved_gb`` (peak reserved GPU memory during the step). Raises unless the steps are 1..n with no gaps.
    """
    if not records:
        raise ValueError("no step records")
    ordered = sorted(records, key=lambda r: int(r["step"]))
    steps = [int(r["step"]) for r in ordered]
    if steps != list(range(1, len(steps) + 1)):
        raise ValueError("step records must be 1..n with no gaps or repeats")
    n = len(ordered)
    if n <= warmup:
        raise ValueError(f"{n} steps is not more than the {warmup}-step warm-up")
    window = ordered[warmup:]
    wall = float(ordered[-1]["t_end_s"]) - float(ordered[warmup - 1]["t_end_s"])
    if wall <= 0:
        raise ValueError("the steady-state window has no elapsed time")
    items = sum(int(r["n_items"]) for r in window)
    real = sum(int(r["real_tokens"]) for r in window)
    padded = sum(int(r["padded_tokens"]) for r in window)
    compute = sum(float(r["elapsed_s"]) for r in window)
    tail = [float(r["loss"]) for r in ordered if int(r["step"]) > n - loss_window]
    return {
        "n_steps": n,
        "warmup_steps_excluded": warmup,
        "window_steps": len(window),
        "window_items": items,
        "window_real_tokens": real,
        "window_padded_tokens": padded,
        "window_wall_s": round(wall, 3),
        "window_compute_s": round(compute, 3),
        "items_per_s": items / wall,
        "real_tokens_per_s": real / wall,
        "compute_items_per_s": items / compute if compute > 0 else None,
        "items_per_batch": items / len(window),
        "padding_efficiency": real / padded if padded else None,
        "loss_last_window_mean": sum(tail) / len(tail),
        "loss_last_window_steps": len(tail),
        "peak_reserved_gb": max(float(r["peak_reserved_gb"]) for r in ordered),
    }


def choose_config(summaries: Mapping[str, Mapping[str, Any]], *, target_steps: int,
                  cap_gb: float = MEMORY_CAP_GB) -> tuple[str | None, str]:
    """The configuration to use, and the reason, under the pre-registered rule.

    A configuration is eligible when it did not run out of memory, finished ``target_steps`` steps and its peak reserved
    memory is at most ``cap_gb``. The highest ``items_per_s`` wins; a tie goes to the name that sorts first, so the choice
    is deterministic. Returns ``(None, reason)`` when no configuration is eligible.
    """
    eligible: list[tuple[float, str]] = []
    rejected: list[str] = []
    for name in sorted(summaries):
        s = summaries[name]
        if s.get("oom"):
            rejected.append(f"{name}: out of memory at step {s.get('oom_at_step')}")
        elif int(s.get("n_steps", 0)) != target_steps:
            rejected.append(f"{name}: finished {s.get('n_steps')} of {target_steps} steps")
        elif float(s["peak_reserved_gb"]) > cap_gb:
            rejected.append(f"{name}: peak reserved {float(s['peak_reserved_gb']):.1f} GB above the {cap_gb:g} GB cap")
        else:
            eligible.append((float(s["items_per_s"]), name))
    if not eligible:
        return None, "no configuration is eligible; " + "; ".join(rejected)
    best_rate = max(rate for rate, _ in eligible)
    chosen = min(name for rate, name in eligible if rate == best_rate)
    reason = f"highest steady-state items/s among the eligible ({best_rate:.3f} items/s)"
    if rejected:
        reason += "; not eligible: " + "; ".join(rejected)
    return chosen, reason
