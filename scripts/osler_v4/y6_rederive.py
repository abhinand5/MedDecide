"""Y6 self-audit (R6): re-derives the Stage A choice's point estimates from the raw predictions, in a fresh process (CPU).

For each arm: the CG v2 accuracy (all 13 templates; the restriction of STATE §7 is the full set when no cell fails); the MR
difference against A0 as the template macro of the paired correctness differences (the mean over MR templates of the mean
over that template's items); the knowledge difference against A0 as the micro mean over the knowledge items; and the item counts.
These must equal the point estimates of `choice.json` (the bootstrap intervals are not re-derived here).

Usage (CPU; after `y6_stage_a.py choose`):
    uv run --frozen python scripts/osler_v4/y6_rederive.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from meddecide.train.blocks import KNOWLEDGE_SOURCES, MR_SOURCES
from meddecide.utils.io import iter_jsonl

OUT = Path("outputs/osler_v4/Y6")
ARMS = ("A0", "A1", "A3", "A4", "A5")


def load(arm: str, name: str) -> dict[str, dict[str, object]]:
    return {str(r["item_id"]): r for r in iter_jsonl(OUT / arm / f"{name}_predictions.jsonl")}


def main() -> None:
    choice = json.loads((OUT / "choice.json").read_text(encoding="utf-8"))
    ref_dev = load("A0", "fastdev_v4")
    ref_cg = load("A0", "cg_v2")
    for arm in ARMS:
        dev = load(arm, "fastdev_v4")
        cg = load(arm, "cg_v2")
        cg_acc = sum(bool(r["correct"]) for r in cg.values()) / len(cg)
        mr_by_template: dict[str, list[int]] = defaultdict(list)
        knowledge: list[int] = []
        for item_id, row in dev.items():
            if item_id not in ref_dev:
                continue
            diff = int(bool(row["correct"])) - int(bool(ref_dev[item_id]["correct"]))
            source = str(row["source"])
            if source in MR_SOURCES:
                mr_by_template[str(row["template_id"])].append(diff)
            if source in KNOWLEDGE_SOURCES:
                knowledge.append(diff)
        mr_point = sum(sum(v) / len(v) for v in mr_by_template.values()) / len(mr_by_template) if mr_by_template else float("nan")
        knowledge_point = sum(knowledge) / len(knowledge) if knowledge else float("nan")
        mr_items = sum(len(v) for v in mr_by_template.values())
        cg_common = sum(1 for r in cg.values() if str(r["template_id"]) in set(choice["cg_v2_templates_compared"]))
        recorded = choice["arms"][arm]
        print(f"{arm}: CG v2 accuracy (compared templates, {cg_common} items) {round(sum(bool(r['correct']) for r in cg.values() if str(r['template_id']) in set(choice['cg_v2_templates_compared'])) / cg_common, 4)}; "
              f"MR template-macro difference vs A0 {round(mr_point, 4) if arm != 'A0' else 'reference'} over {mr_items} items; "
              f"knowledge micro difference vs A0 {round(knowledge_point, 4) if arm != 'A0' else 'reference'} over {len(knowledge)} items; "
              f"recorded CG {round(recorded['cg_v2_accuracy_on_compared'], 4)}; cg full-set accuracy {round(cg_acc, 4)}")
        if arm != "A0":
            print(f"   recorded MR point {round(recorded['mr_macro_difference']['point'], 4)} ({recorded['mr_macro_difference']['n']} items), "
                  f"knowledge point {round(recorded['knowledge_micro_difference']['point'], 4)} ({recorded['knowledge_micro_difference']['n']} items)")
    _ = ref_cg  # the reference's CG rows are read to show the file is complete
    for arm in ARMS:
        for name in ("fastdev_v4", "cg_v2"):
            rows = list(iter_jsonl(OUT / arm / f"{name}_predictions.jsonl"))
            off = sum(abs(sum(r["probs"]) - 1.0) > 1e-3 for r in rows)
            bad = sum(not all(math.isfinite(x) and x >= -1e-9 for x in r["probs"]) for r in rows)
            worst = max(abs(sum(r["probs"]) - 1.0) for r in rows)
            print(f"{arm} {name}: rows {len(rows)}; probability vectors off sum-to-one (over 1e-3) {off}; not finite or "
                  f"negative {bad}; worst |sum - 1| {worst:.1e}")


if __name__ == "__main__":
    main()
