"""osler_v3 helpers: the KL-to-base anchor rule (D41, ADVISORY §3.2) and the timing summary of a training run.

Pure functions; the scripts do the reading and writing. The domain prefixes are fixed for the whole loop and are the same in
every run. An anchor is a training item whose template starts with none of them: the KL-to-base penalty applies to anchors
only, so the domain templates may move while the general skill stays near the base.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

# ADVISORY osler_v3 §3.2, fixed before any osler_v3 run; a new template family must take one of these or be an anchor
DOMAIN_PREFIXES: tuple[str, ...] = (
    "mr_",
    "synthea_",
    "chart_",
    "codiesp_",
    "mtsamples_",
    "mimic_",
    "ehr_rel_",
    "pmc_",
    "ragtruth_",
    "drugprot_",
    "multicare_",
    "clinclaim_",
    "restyle_",
)


def is_anchor(template_id: str, prefixes: Sequence[str] = DOMAIN_PREFIXES) -> bool:
    """True when the item is a KL-to-base anchor: its template starts with none of ``prefixes``.

    With no prefixes every item is an anchor, which is the trainer's rule (``kl_exclude_template_prefixes`` empty).
    """
    return not str(template_id).startswith(tuple(prefixes))


def anchor_share(template_ids: Iterable[str], prefixes: Sequence[str] = DOMAIN_PREFIXES) -> dict[str, Any]:
    """Counts of anchor and domain items, and the anchor share over all items (the denominator is stated)."""
    total = 0
    anchors = 0
    for template_id in template_ids:
        total += 1
        anchors += int(is_anchor(template_id, prefixes))
    if total == 0:
        raise ValueError("no items to count")
    return {"items": total, "anchors": anchors, "domain": total - anchors, "anchor_share": anchors / total}


def timing_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Step time, anchor share, peak memory and the KL value of a run, from its per-step records (``StepRecord.to_dict``).

    The first step carries the CUDA warm-up, so the step-time means leave it out; the median is over all steps. The KL mean
    is over the steps that had at least one anchor. Non-finite losses are counted, not dropped.
    """
    if len(records) < 2:
        raise ValueError("a timing summary needs at least two steps (the first is warm-up)")
    elapsed = np.array([float(r["elapsed_s"]) for r in records])
    steady = elapsed[1:]
    items = sum(int(r["n_items"]) for r in records)
    anchors = sum(int(r["n_anchor"]) for r in records)
    kl_steps = [float(r["kl"]) for r in records if int(r["n_anchor"]) > 0]
    losses = [float(r["loss"]) for r in records]
    return {
        "steps": len(records),
        "s_per_step_mean_excl_first": float(steady.mean()),
        "s_per_step_median": float(np.median(elapsed)),
        "s_per_step_first": float(elapsed[0]),
        "items": items,
        "anchor_items": anchors,
        "anchor_share": anchors / items if items else math.nan,
        "gpu_peak_gb_max": max(float(r["gpu_peak_gb"]) for r in records),
        "kl_mean_over_anchor_steps": float(np.mean(kl_steps)) if kl_steps else math.nan,
        "kl_steps_with_anchors": len(kl_steps),
        "nonfinite_losses": sum(1 for x in losses if not math.isfinite(x)),
    }
