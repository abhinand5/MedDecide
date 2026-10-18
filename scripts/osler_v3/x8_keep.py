"""osler_v3 X8 keep rule (ADVISORY X8; D34) for one Stage B block against the last kept block. Dev only.

The rule is meddecide.train.blocks.keep_comparison on the fast dev v3 predictions: MR (template macro, MR_SOURCES), general
(template macro) and knowledge (micro accuracy, medqa and medmcqa); the margins are KEEP_MARGINS; paired bootstrap within templates,
1,000 resamples, seed 0. Pre-registered D-X8-1: the items of the new sources (ragtruth, drugprot, synthea_chart) are removed from
both sides before the rule, so they cannot move the general or knowledge bounds; they are reported beside the rule (labelled).
Also reported beside the rule, outside it: the clinical-general (CG) accuracy of both sides and each side's difference from
zero-shot pplx-decider-flash-v1.2 on the same CG items (ADVISORY X8: report the CG partition for every block).

Writes outputs/osler_v3/X8/keep_<candidate>.json (aggregates only; the decision is in "decision"). It prints KEPT or NOT KEPT.

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x8_keep.py --candidate B1 \
            --candidate-dir outputs/osler_v3/stage_b/B1 --current B0 --current-dir outputs/osler_v3/stage_a/A0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from meddecide.eval.o2_metrics import score_row
from meddecide.train.blocks import keep_comparison, macro_difference, pair_by_item
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v3" / "X8"
ZERO_SHOT_CG = REPO / "outputs" / "osler_v3" / "X6" / "zeroshot_pplx_cg" / "pplx-decider-flash-v1.2" / "predictions.jsonl"
NEW_SOURCES = frozenset({"ragtruth", "drugprot", "synthea_chart"})  # D-X8-1 (pre-registered; see STATE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def without(rows: list[dict[str, Any]], sources: frozenset[str]) -> list[dict[str, Any]]:
    return [r for r in rows if r["source"] not in sources]


def only(rows: list[dict[str, Any]], sources: frozenset[str]) -> list[dict[str, Any]]:
    return [r for r in rows if r["source"] in sources]


def cg_report(candidate_dir: Path, current_dir: Path) -> dict[str, Any]:
    """CG accuracy of each side on the items the zero-shot pplx reference scored (both sides, the same items)."""
    if not ZERO_SHOT_CG.exists():
        return {"status": "NOT MEASURED — the zero-shot pplx CG predictions are absent"}
    zero = {str(r["item_id"]): bool(score_row(r).correct) for r in read_jsonl(ZERO_SHOT_CG) if r.get("status") == "scored"}
    out: dict[str, Any] = {"zero_shot_pplx_items": len(zero)}
    for name, folder in (("candidate", candidate_dir), ("current", current_dir)):
        rows = [r for r in read_jsonl(folder / "cg_predictions.jsonl") if str(r["item_id"]) in zero]
        if not rows:
            out[name] = {"status": "NOT MEASURED — no CG predictions on the paired items"}
            continue
        acc = sum(bool(r["correct"]) for r in rows) / len(rows)
        zacc = sum(zero[str(r["item_id"])] for r in rows) / len(rows)
        out[name] = {"items": len(rows), "accuracy": round(acc, 4), "zero_shot_pplx_accuracy": round(zacc, 4),
                     "difference_from_zero_shot_pplx": round(acc - zacc, 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidate", required=True, help="the block being judged, e.g. B1")
    ap.add_argument("--candidate-dir", required=True, type=Path)
    ap.add_argument("--current", required=True, help="the last kept block, e.g. B0")
    ap.add_argument("--current-dir", required=True, type=Path)
    args = ap.parse_args()
    cand_dir = args.candidate_dir if args.candidate_dir.is_absolute() else REPO / args.candidate_dir
    cur_dir = args.current_dir if args.current_dir.is_absolute() else REPO / args.current_dir
    cand = read_jsonl(cand_dir / "dev_predictions.jsonl")
    cur = read_jsonl(cur_dir / "dev_predictions.jsonl")
    rule = keep_comparison(without(cand, NEW_SOURCES), without(cur, NEW_SOURCES), candidate=args.candidate, current=args.current)
    beside = pair_by_item(only(cand, NEW_SOURCES), only(cur, NEW_SOURCES))
    report: dict[str, Any] = {
        "built_utc": utcnow(), "git_commit": git_commit(REPO), "command": " ".join(sys.argv),
        "candidate": args.candidate, "current": args.current, "candidate_dir": str(cand_dir.relative_to(REPO)),
        "current_dir": str(cur_dir.relative_to(REPO)), "new_sources_excluded_from_rule": sorted(NEW_SOURCES),
        "rule": rule, "decision": rule["verdict"]["kept"],
        "beside_rule_new_sources": {"items": len(beside), "templates": len({str(x["template_id"]) for x, _ in beside}),
                                    "macro_difference": macro_difference(beside).as_dict() if beside else None},
        "beside_rule_cg": cg_report(cand_dir, cur_dir),
    }
    write_json(OUT / f"keep_{args.candidate}.json", report)
    print(f"{args.candidate} vs {args.current}: lower bounds {rule['verdict']['lower_bounds']}", flush=True)
    print("KEPT" if report["decision"] else "NOT KEPT", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
