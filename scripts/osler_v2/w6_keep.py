"""osler_v2 W6: the base choice (D39), the keep rule (D34) and the knowledge checks. Dev only; no test set is read.

  --base            C0p minus C0q on fast dev v2: pplx is chosen iff the lower bounds of the paired differences are > -0.01 on MR,
                    > -0.01 on general and > -0.02 on knowledge, AND the MR or general point difference is > 0. Otherwise Qwen.
  --candidate X --current C
                    the keep rule for X against the last kept block C (margins: meddecide.train.blocks.KEEP_MARGINS).
  --knowledge-check B --zs {qwen,pplx}
                    B's knowledge items minus the zero-shot base on the same items (lower bound above -0.02), after the D12 gate.

Paired bootstrap within templates, 1,000 resamples, seed 0 (meddecide.train.blocks). The items with duplicate option labels (12)
cannot be scored by the pplx reference; they are excluded from every knowledge comparison, on both sides, and counted.

Writes outputs/osler_v2/W6/keep/*.json (aggregates only).

Usage (CPU): source scripts/pod_env.sh && uv run --frozen python scripts/osler_v2/w6_keep.py --base
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from meddecide.eval.o2_metrics import cell_health
from meddecide.train.blocks import (
    KNOWLEDGE_SOURCES,
    keep_comparison,
    keep_statistics,
    knowledge_comparison,
    pair_by_item,
)
from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
W6 = REPO / "outputs" / "osler_v2" / "W6"
KEEP = W6 / "keep"
FAST_DEV = REPO / "data" / "bench" / "osler_v2" / "fastdev_v2.jsonl"
KNOWLEDGE_O2 = REPO / "data" / "bench" / "osler_v2" / "knowledge_dev_v2.jsonl"
ZS = {"qwen": REPO / "outputs" / "osler_v2" / "W5" / "zeroshot_knowledge" / "qwen35-4b" / "predictions.jsonl",
      "pplx": REPO / "outputs" / "osler_v2" / "W5" / "zeroshot_knowledge" / "pplx-decider-flash-v1.2" / "predictions.jsonl"}
ZS_LABEL = {"qwen": "zero-shot Qwen3.5-4B", "pplx": "zero-shot pplx-decider-flash-v1.2"}
BASE_MARGINS = {"mr": -0.01, "general": -0.01, "knowledge": -0.02}  # D39


def dev_rows(name: str) -> list[dict[str, Any]]:
    return list(iter_jsonl(W6 / name / "dev_predictions.jsonl"))


def trainer_ids() -> dict[str, str]:
    """O2 knowledge item id -> the trainer's item id, through the fast dev v2 row that carries it (record id)."""
    mapping: dict[str, str] = {}
    for raw in iter_jsonl(FAST_DEV):
        if raw.get("tier") == "knowledge_dev_v2":
            mapping[raw["source_record_id"]] = mix_row_to_item(raw).item_id
    return mapping


def duplicate_label_items() -> set[str]:
    """O2 knowledge items whose option labels repeat (the pplx reference cannot score them): excluded on both sides."""
    out = set()
    for row in iter_jsonl(KNOWLEDGE_O2):
        labels = [o["label"] for o in row["options"]]
        if len(set(labels)) != len(labels):
            out.add(row["item_id"])
    return out


def base_choice() -> dict[str, Any]:
    qwen = dev_rows("C0_qwen")
    pplx = dev_rows("C0_pplx")
    pairs = pair_by_item(pplx, qwen)  # C0p minus C0q
    stats = keep_statistics(pairs)
    lows = {p: float(s.lo) for p, s in stats.items()}
    points = {p: float(s.point) if hasattr(s, "point") else None for p, s in stats.items()}
    checks = {p: lows[p] > BASE_MARGINS[p] for p in BASE_MARGINS}
    point_positive = (points["mr"] or 0) > 0 or (points["general"] or 0) > 0
    chosen = "pplx" if all(checks.values()) and point_positive else "qwen"
    return {"kind": "osler_v2_W6_base_choice", "written_utc": utcnow(), "git_commit": git_commit(REPO),
            "comparison": "C0p (pplx-decider-flash) minus C0q (Qwen3.5-4B), fast dev v2, paired bootstrap, 1,000, seed 0",
            "statistics": {p: s.as_dict() for p, s in stats.items()}, "lower_bounds": lows, "point": points,
            "margins": BASE_MARGINS, "checks": checks, "point_difference_positive_on_mr_or_general": point_positive,
            "chosen_base": chosen, "rule": "D39 (fixed before any osler_v2 result)"}


def keep_report(candidate: str, current: str) -> dict[str, Any]:
    return {"kind": "osler_v2_W6_keep_rule", "written_utc": utcnow(), "git_commit": git_commit(REPO),
            **keep_comparison(dev_rows(candidate), dev_rows(current), candidate=candidate, current=current)}


def knowledge_report(block: str, zs: str) -> dict[str, Any]:
    excluded = duplicate_label_items()
    ids = trainer_ids()
    block_rows = [r for r in dev_rows(block) if r["source"] in KNOWLEDGE_SOURCES]
    zs_rows = []
    for r in iter_jsonl(ZS[zs]):
        if r["item_id"] in excluded:
            continue
        if r["status"] != "scored":
            raise ValueError(f"{zs}: a knowledge item is {r['status']} ({r.get('reason')}) outside the excluded set")
        zs_rows.append({**r, "item_id": ids[r["item_id"]]})
    excluded_trainer = {ids[i] for i in excluded if i in ids}
    block_rows = [r for r in block_rows if r["item_id"] not in excluded_trainer]
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in zs_rows:
        by_template[r["template_id"]].append(r)
    health = {t: cell_health(ZS_LABEL[zs], t, rows, zero_shot=True)["status"] for t, rows in by_template.items()}
    return {"kind": "osler_v2_W6_knowledge_check", "written_utc": utcnow(), "git_commit": git_commit(REPO),
            "excluded_duplicate_label_items": len(excluded), "block_knowledge_items": len(block_rows),
            "zero_shot_items": len(zs_rows),
            **knowledge_comparison(block_rows, zs_rows, block=block, comparison=ZS_LABEL[zs], health=health)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--candidate")
    ap.add_argument("--current")
    ap.add_argument("--knowledge-check")
    ap.add_argument("--zs", choices=["qwen", "pplx"], default="qwen")
    args = ap.parse_args()
    if args.base:
        report, path = base_choice(), KEEP / "base_choice.json"
    elif args.candidate and args.current:
        report, path = keep_report(args.candidate, args.current), KEEP / f"{args.candidate}_vs_{args.current}.json"
    elif args.knowledge_check:
        report = knowledge_report(args.knowledge_check, args.zs)
        path = KEEP / f"knowledge_{args.knowledge_check}_vs_{args.zs}.json"
    else:
        raise SystemExit("give --base, --candidate with --current, or --knowledge-check")
    write_json(path, report)
    print(f"wrote {path.relative_to(REPO)}", flush=True)
    print(json.dumps({k: report[k] for k in report if k in ("chosen_base", "verdict", "checks", "lower_bounds")},
                     default=str)[:800], flush=True)


if __name__ == "__main__":
    main()
