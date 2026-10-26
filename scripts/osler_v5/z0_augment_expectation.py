"""osler_v5 Z0 step 4 (ADVISORY Z0.3; D52a): the expected two-letter golds of the closed-list augmentation, from the Y8 replay's realised items
(counts only). For each run length, the eligible realised items are the closed-list rows of the batches the run draws; at probability p
the expected number of training items whose gold lands on option position r (26..63) is p x eligible x g(r), where g(r) is the
uniform-count, uniform-gold probability of meddecide.train.option_augment.expected_two_letter_golds.

The per-row criterion of ADVISORY Z0.3 is read as the mean over the two-letter rows 26..63 (the rows the augmentation can reach with
counts up to 64). The minimum row (63) is also reported; it cannot reach 20 at any p in (0, 1] at arm length, and the report says so.

Usage (CPU):
    uv run --frozen python scripts/osler_v5/z0_augment_expectation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from meddecide.train.option_augment import CLOSED_LIST, expected_two_letter_golds
from meddecide.utils.io import write_json

REPO = Path(__file__).resolve().parents[2]
STEPS_NPZ = REPO / "outputs" / "osler_v5" / "Z0" / "replay" / "steps.npz"
STATS = REPO / "outputs" / "osler_v5" / "Z0" / "mix_template_stats.json"
OUT = REPO / "outputs" / "osler_v5" / "Z0" / "augment_expectation.json"
TARGET_PER_ROW = 20.0


def main() -> None:
    steps = np.load(STEPS_NPZ)
    closed_per_step = steps["closed_items"].astype(np.int64)
    items_per_step = steps["n_items"].astype(np.int64)
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    label_sets = {t: stats["templates"][t]["distinct_gold_labels"] for t in CLOSED_LIST}
    report: dict[str, object] = {"target_per_row": TARGET_PER_ROW, "closed_templates": sorted(CLOSED_LIST),
                                 "label_set_sizes": label_sets, "runs": {}}
    for horizon in (13_992, 63_304):
        eligible = int(closed_per_step[:horizon].sum())
        realised = int(items_per_step[:horizon].sum())
        p = 1.0
        expected = expected_two_letter_golds(eligible, p, label_set_size=min(label_sets.values()))
        per_row = list(expected.values())
        mean_row = float(np.mean(per_row))
        report["runs"][str(horizon)] = {
            "steps": horizon, "realised_items": realised, "eligible_closed_items": eligible,
            "eligible_share": eligible / realised, "probability": p,
            "expected_per_row_mean_26_63": mean_row, "expected_per_row_min_26_63": float(min(per_row)),
            "expected_per_row_max_26_63": float(max(per_row)),
            "meets_target_mean_at_p1": mean_row >= TARGET_PER_ROW,
            "smallest_p_meeting_target_mean": (TARGET_PER_ROW / mean_row if mean_row > 0 else None),
        }
    # the share of trained items with more than 26 options (count uniform on 8..64: 38 of the 57 counts are above 26)
    report["share_of_trained_items_over_26_options"] = {
        str(h): report["runs"][str(h)]["eligible_share"] * (38 / 57) for h in (13_992, 63_304)
    }
    write_json(OUT, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
