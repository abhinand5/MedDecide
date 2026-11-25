"""V8 diagnostic (additional; never a keep or gate input): the MR dev part of B3 against B1, template by template.

The keep rule compares template macros over the MR dev part (ADVISORY V8). This script shows which templates carry that
difference, and the difference without pmc_similar_patient_v1, a template that only B3 trains on (CLAIMS V150, V151). The
template macro of the seven templates must equal the keep rule's point estimate (checked, not assumed).

Writes outputs/osler_v1/V8/diagnostics/mr_templates_b3_vs_b1.json (aggregates only).

Usage (CPU; no GPU):
    source scripts/pod_env.sh && uv run --frozen python -I scripts/osler_v1/v8_mr_templates.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from meddecide.train.blocks import MR_SOURCES
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
V8 = REPO / "outputs" / "osler_v1" / "V8"
KEEP = V8 / "keep" / "B3_vs_B1.json"
OUT = V8 / "diagnostics" / "mr_templates_b3_vs_b1.json"
PMC = "pmc_similar_patient_v1"


def read_mr(block: str) -> dict[str, dict]:
    """The MR dev rows of one block's fast dev predictions, by item id."""
    with (V8 / block / "dev_predictions.jsonl").open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {row["item_id"]: row for row in rows if row["source"] in MR_SOURCES}


def main() -> None:
    b3, b1 = read_mr("B3"), read_mr("B1")
    if set(b3) != set(b1):
        raise SystemExit("B3 and B1 do not hold the same MR dev items")
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # items, B3 correct, B1 correct
    for item_id, row in b3.items():
        slot = counts[row["template_id"]]
        slot[0] += 1
        slot[1] += int(bool(row["correct"]))
        slot[2] += int(bool(b1[item_id]["correct"]))
    templates = {}
    for name, (n, c3, c1) in sorted(counts.items()):
        templates[name] = {"items": n, "b3_accuracy": round(c3 / n, 4), "b1_accuracy": round(c1 / n, 4),
                           "difference": round(c3 / n - c1 / n, 4)}
    diffs = [counts[name][1] / counts[name][0] - counts[name][2] / counts[name][0] for name in counts]
    macro = sum(diffs) / len(diffs)
    keep_point = json.loads(KEEP.read_text(encoding="utf-8"))["statistics"]["mr"]["point"]
    if abs(macro - keep_point) > 1e-9:
        raise SystemExit(f"the template macro {macro} does not equal the keep rule's point {keep_point}")
    without = [d for name, d in zip(counts, diffs, strict=True) if name != PMC]
    record = {
        "kind": "osler_v1_V8_mr_templates", "written_utc": utcnow(), "git_commit": git_commit(REPO),
        "mr_items": len(b3), "templates": templates,
        "template_macro_difference": round(macro, 4), "keep_rule_point": round(keep_point, 4),
        "template_macro_difference_without_pmc": round(sum(without) / len(without), 4),
        "templates_without_pmc": len(without),
    }
    write_json(OUT, record)
    print(f"MR dev items {len(b3)}; template macro difference {record['template_macro_difference']}; "
          f"without {PMC} {record['template_macro_difference_without_pmc']} over {len(without)} templates; wrote {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
