"""Y6 (osler_v4 ADVISORY Y6): the readout gate (D12, D21) of one Stage A arm, template by template, on fast dev v4 and CG v2 (CPU).

For each set it writes every cell's figures, its partition (MR, knowledge or general, from the item source as the keep rule's
partition_pairs assigns it) and its status to outputs/osler_v4/Y6/<arm>/readout_gate.json, and it prints the failing cells. A
failing cell is not a result: the Stage A choice then uses only the templates that pass (STATE §7, the Y6 deviation).

Usage (CPU; after the arm's predictions exist):
    uv run --frozen python scripts/osler_v4/y6_readout_gate.py --arm A0
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from meddecide.eval.readout_gate import gate_by_template
from meddecide.train.blocks import KNOWLEDGE_SOURCES, MR_SOURCES
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import utcnow

OUT = Path("outputs/osler_v4/Y6")
SETS = {"fastdev_v4": "fastdev_v4_predictions.jsonl", "cg_v2": "cg_v2_predictions.jsonl"}


def partition(source: str) -> str:
    if source in MR_SOURCES:
        return "mr"
    return "knowledge" if source in KNOWLEDGE_SOURCES else "general"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    report: dict[str, object] = {"arm": args.arm, "generated_utc": utcnow(), "sets": {}}
    sets: dict[str, object] = {}
    for name, file in SETS.items():
        rows = list(iter_jsonl(OUT / args.arm / file))
        gates = gate_by_template(rows)
        parts: dict[str, str] = {}
        for row in rows:
            parts.setdefault(str(row["template_id"]), partition(str(row["source"])))
        cells = {t: {"partition": parts[t], "items": g.items, "accuracy": round(g.accuracy, 4), "status": g.status}
                 for t, g in gates.items()}
        failing = sorted(t for t, g in gates.items() if not g.passes)
        sets[name] = {"items": len(rows), "templates": len(gates), "failing": failing,
                      "partitions": dict(sorted(Counter(parts.values()).items())), "cells": cells}
        print(f"{args.arm} {name}: items {len(rows)}; templates {len(gates)}; failing cells {len(failing)}"
              + (f" ({', '.join(failing)})" if failing else ""), flush=True)
    report["sets"] = sets
    write_json(OUT / args.arm / "readout_gate.json", report)


if __name__ == "__main__":
    main()
