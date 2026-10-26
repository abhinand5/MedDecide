"""osler_v5 Z4 blend rule (ADVISORY Z4; pre-registered). Pure functions over the dev statistics of each alpha on the selected checkpoint.

Retention: for each of the MR and MIMIC dev template macros, the gain over alpha = 0 at alpha must be at least 80 % of the gain at alpha = 1.
An alpha qualifies if it passes the retention rule, positional health in all three orders, and CG v2 micro >= the CG v2 micro at alpha = 1.
The smallest qualifying alpha is chosen. If none qualifies, alpha = 1 is kept and the blend is reported as not used (ADVISORY Z4, outcome 3).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

RETENTION = 0.8


@dataclass(frozen=True)
class AlphaStats:
    alpha: float
    mr_macro: float
    mimic_macro: float
    cg_micro: float
    positional_passes: bool


def choose_alpha(stats: Sequence[AlphaStats]) -> dict[str, Any]:
    """The pre-registered choice over the alphas in ``stats``; alpha 0 and alpha 1 must both be present."""
    by_alpha = {round(s.alpha, 12): s for s in stats}
    if 0.0 not in by_alpha or 1.0 not in by_alpha:
        raise ValueError("the reference (alpha 0) and the full model (alpha 1) must be among the statistics")
    if len(by_alpha) != len(stats):
        raise ValueError("an alpha appears twice")
    ref, full = by_alpha[0.0], by_alpha[1.0]
    gain_mr = full.mr_macro - ref.mr_macro
    gain_mimic = full.mimic_macro - ref.mimic_macro
    rows: dict[str, Any] = {}
    qualifying: list[float] = []
    for alpha in sorted(a for a in by_alpha if a > 0.0):
        s = by_alpha[alpha]
        retention_mr = (s.mr_macro - ref.mr_macro) >= RETENTION * gain_mr
        retention_mimic = (s.mimic_macro - ref.mimic_macro) >= RETENTION * gain_mimic
        cg_ok = s.cg_micro >= full.cg_micro
        qualifies = retention_mr and retention_mimic and s.positional_passes and cg_ok
        rows[f"{alpha:g}"] = {"qualifies": qualifies, "retention_mr": retention_mr, "retention_mimic": retention_mimic,
                              "positional_passes": s.positional_passes, "cg_micro_at_least_full": cg_ok,
                              "mr_macro": s.mr_macro, "mimic_macro": s.mimic_macro, "cg_micro": s.cg_micro}
        if qualifies:
            qualifying.append(alpha)
    report: dict[str, Any] = {"gain_over_reference_at_1": {"mr_macro": gain_mr, "mimic_macro": gain_mimic},
                              "retention": RETENTION, "alphas": rows}
    below_one = [alpha for alpha in qualifying if alpha < 1.0]
    if below_one:
        report.update({"outcome": "chosen", "chosen_alpha": min(below_one), "blend_used": True})
    else:  # alpha 1 is the only qualifier at most: it is kept, and the blend is reported as not used (outcome 3)
        report.update({"outcome": "no alpha below 1 qualifies (ADVISORY Z4, outcome 3): alpha 1 is kept; the blend is not used",
                       "chosen_alpha": 1.0, "blend_used": False})
    return report
