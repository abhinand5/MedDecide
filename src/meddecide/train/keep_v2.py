"""osler_v4 keep rule v2 (D47; ADVISORY Y7): a block is kept iff, against the last kept block, five lower bounds are above margins.

The bounds are paired bootstrap lower bounds within templates (meddecide.train.blocks, 1,000 resamples, seed 0):

- own: the block's new templates on fast dev, the template macro of the paired differences over those templates; lower bound above +0.02;
- mr: the MR partition (osler_v3's), the template macro; above -0.015;
- general: every item that is not MR, with the new templates removed, the template macro; above -0.01;
- knowledge: the MedQA and MedMCQA items, with the new templates removed, the micro accuracy; above -0.02;
- cg_v2: the CG v2 items, the micro accuracy; above -0.01.

The v1 margins of meddecide.train.blocks (osler_v1 to osler_v3) are not changed: they are the record of those loops.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

KEEP_V2_MARGINS: dict[str, float] = {"own": 0.02, "mr": -0.015, "general": -0.01, "knowledge": -0.02, "cg_v2": -0.01}


def keep_v2_verdict(lower_bounds: Mapping[str, float]) -> dict[str, Any]:
    """Each check (its lower bound strictly above its margin) and the decision: kept iff every check passes."""
    missing = set(KEEP_V2_MARGINS) - set(lower_bounds)
    if missing:
        raise ValueError(f"no lower bound for {sorted(missing)}")
    checks = {part: float(lower_bounds[part]) > margin for part, margin in KEEP_V2_MARGINS.items()}
    return {"rule": "keep rule v2 (D47)", "margins": dict(KEEP_V2_MARGINS),
            "lower_bounds": {part: float(lower_bounds[part]) for part in KEEP_V2_MARGINS},
            "checks": checks, "kept": all(checks.values())}
