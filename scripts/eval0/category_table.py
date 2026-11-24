#!/usr/bin/env python
"""eval0 — render per-category accuracy tables from outputs/eval0/leaderboard/leaderboard.json (CPU, no recomputation)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SHORT = {
    "Osler-9B final (63,304 steps)": "Osler-9B final",
    "Osler-9B arm (13,992 steps)": "Osler-9B arm",
    "Osler-4B arm (13,992 steps)": "Osler-4B arm",
    "Osler-4B v4 final": "Osler-4B v4 final",
    "Osler-4B v2": "Osler-4B v2",
    "pplx-decider-flash-v1.2 (zero-shot, 4B base)": "pplx-flash (4B base)",
    "d3-flash (zero-shot, 9B base)": "d3-flash (9B base)",
    "pplx-decider-v1.2-27b (zero-shot, ceiling)": "pplx-27B",
    "MedDecider-4B (lion-ai)": "MedDecider-4B",
    "MedDecider-9B (lion-ai)": "MedDecider-9B",
    "JEV-9B": "JEV-9B",
}
SETS = [("panel", "by_source", "External panel, by source"),
        ("knowledge", "by_source", "Knowledge test (frozen), by source"),
        ("mr_verdict", "by_source", "MR verdict (in-domain), by source"),
        ("mr_verdict", "by_template", "MR verdict (in-domain), by template"),
        ("mimic_suite", "by_template", "MIMIC suite (in-domain), by template")]


def cell(entry: dict | None) -> str:
    if not entry:
        return "—"
    return f"{entry['accuracy']:.3f} (n={entry['n']})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=Path("outputs/eval0/leaderboard/leaderboard.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    board = json.loads(args.board.read_text(encoding="utf-8"))
    models = list(board["per_category"])
    lines = ["# Per-category accuracy (eval0 snapshot, 2026-11-21)", "",
             "Recomputed from prediction files by `scripts/eval0/leaderboard.py`. `—` = no predictions for that set. "
             "Counts are the items the model scored in the category. Categories are as labelled in the frozen files.", ""]
    for set_name, level, title in SETS:
        cats: list[str] = []
        for m in models:
            for c in board["per_category"][m].get(set_name, {}).get(level, {}):
                if c not in cats:
                    cats.append(c)
        lines += [f"## {title}", "", "| category | " + " | ".join(SHORT.get(m, m) for m in models) + " |",
                  "|---|" + "---|" * len(models)]
        for c in cats:
            row = []
            for m in models:
                entry = board["per_category"][m].get(set_name, {}).get(level, {}).get(c)
                row.append(cell(entry))
            lines.append(f"| {c} | " + " | ".join(row) + " |")
        lines.append("")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
