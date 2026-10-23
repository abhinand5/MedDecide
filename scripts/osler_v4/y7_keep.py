"""Y7 (osler_v4 ADVISORY Y7; D47): the keep decision of one Stage B block against the last kept block (CPU).

The five lower bounds of keep rule v2 (meddecide.train.keep_v2): own (the block's new templates on fast dev v4), MR (osler_v3's MR
partition), general (every other item, the new templates removed), knowledge (MedQA and MedMCQA, the new templates removed) and CG v2.
The paired bootstrap within templates (meddecide.train.blocks: 1,000 resamples, seed 0) gives each lower bound. As in the Stage A
choice, a readout cell that fails the gate (meddecide.eval.readout_gate) is not a result, so each comparison uses the templates
that pass for both the candidate and the base; the excluded templates are written out.

The new templates are the template ids of the block's added files (read from the block's manifest).

Writes outputs/osler_v4/Y7/keep_<arm>.json. Usage (CPU; after the arm is scored):
    uv run --frozen python scripts/osler_v4/y7_keep.py --arm B1 --base-dir outputs/osler_v4/Y6/A0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from meddecide.eval.readout_gate import gate_by_template
from meddecide.train.blocks import KNOWLEDGE_SOURCES, MR_SOURCES, macro_difference, micro_difference, pair_by_item
from meddecide.train.keep_v2 import keep_v2_verdict
from meddecide.utils.io import iter_jsonl, read_json, write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v4" / "Y7"


def passing(rows: list[dict[str, Any]]) -> set[str]:
    return {t for t, gate in gate_by_template(rows).items() if gate.passes}


def failing(rows: list[dict[str, Any]]) -> list[str]:
    return sorted(t for t, gate in gate_by_template(rows).items() if not gate.passes)


def summary(difference: Any) -> dict[str, float]:
    return {"point": round(difference.point, 6), "lower": round(difference.lo, 6), "upper": round(difference.hi, 6)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", required=True, help="the candidate block, e.g. B1")
    parser.add_argument("--base-dir", required=True, type=Path, help="the last kept block's scored directory (predictions)")
    args = parser.parse_args()
    manifest = read_json(OUT / f"block_{args.arm}.json")
    new_templates: set[str] = set()
    for path in manifest["add"]:
        p = Path(path) if Path(path).is_absolute() else REPO / path
        for row in iter_jsonl(p):
            new_templates.add(str(row["template_id"]))
    cand_dir = OUT / args.arm
    cand_dev = list(iter_jsonl(cand_dir / "fastdev_v4_predictions.jsonl"))
    base_dev = list(iter_jsonl(args.base_dir / "fastdev_v4_predictions.jsonl"))
    cand_cg = list(iter_jsonl(cand_dir / "cg_v2_predictions.jsonl"))
    base_cg = list(iter_jsonl(args.base_dir / "cg_v2_predictions.jsonl"))
    cand_pass = passing(cand_dev) & passing(base_dev)
    cg_pass = passing(cand_cg) & passing(base_cg)
    dev_pairs = pair_by_item(cand_dev, base_dev)
    cg_pairs = pair_by_item(cand_cg, base_cg)
    own_pairs = [(x, c) for x, c in dev_pairs if str(x["template_id"]) in new_templates and str(x["template_id"]) in cand_pass]
    rest = [(x, c) for x, c in dev_pairs if str(x["template_id"]) not in new_templates and str(x["template_id"]) in cand_pass]
    mr_pairs = [(x, c) for x, c in rest if str(x["source"]) in MR_SOURCES]
    general_pairs = [(x, c) for x, c in rest if str(x["source"]) not in MR_SOURCES]
    knowledge_pairs = [(x, c) for x, c in general_pairs if str(x["source"]) in KNOWLEDGE_SOURCES]
    cg_pairs_pass = [(x, c) for x, c in cg_pairs if str(x["template_id"]) in cg_pass]
    own = macro_difference(own_pairs)
    mr = macro_difference(mr_pairs)
    general = macro_difference(general_pairs)
    knowledge = micro_difference(knowledge_pairs)
    cg = micro_difference(cg_pairs_pass)
    verdict = keep_v2_verdict({"own": own.lo, "mr": mr.lo, "general": general.lo, "knowledge": knowledge.lo, "cg_v2": cg.lo})
    result: dict[str, Any] = {
        "block": args.arm, "generated_utc": utcnow(), "base_dir": str(args.base_dir), "candidate_dir": str(cand_dir),
        "new_templates": sorted(new_templates),
        "statistics": {"own": summary(own), "mr": summary(mr), "general": summary(general), "knowledge": summary(knowledge),
                       "cg_v2": summary(cg)},
        "items": {"own": len(own_pairs), "mr": len(mr_pairs), "general": len(general_pairs), "knowledge": len(knowledge_pairs),
                  "cg_v2": len(cg_pairs_pass)},
        "failing_cells": {"candidate_fastdev_v4": failing(cand_dev), "base_fastdev_v4": failing(base_dev),
                          "candidate_cg_v2": failing(cand_cg), "base_cg_v2": failing(base_cg)},
        "excluded_new_templates": sorted(new_templates - cand_pass), "verdict": verdict,
    }
    write_json(OUT / f"keep_{args.arm}.json", result)
    print(f"{args.arm}: kept {verdict['kept']}; " + ", ".join(
        f"{k} {v['point']:+.4f} (lower {v['lower']:+.4f})" for k, v in result["statistics"].items()), flush=True)


if __name__ == "__main__":
    main()
