"""O11 constant-answer investigation (ADVISORY §8.3; CPU; reads the calibrated audit, the dev predictions and the training mix).

For each template where an Osler model's calibrated two-order predictions put more than a threshold share on one option, this
records what the dev predictions show for the same template (the in-distribution check) and the training labels of the template
(whether the model was trained on it at all). A collapse that also shows on dev, on a template with no training rows, is a learned
or transfer behaviour; a collapse absent from dev would point at the readout. Writes outputs/osler_v0/O11/constant_check.json
(gitignored); the CLAIMS rows cite its contents.

Usage:
    uv run --frozen python scripts/osler/o11_constant_check.py --name osler_0p8b --dev-root outputs/osler_v0/O10/osler_0p8b
"""

from __future__ import annotations

import argparse
import collections
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from meddecide.utils.io import iter_jsonl, read_json, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
CAL = REPO / "outputs/osler_v0/O11/calibrated"
TRAIN = REPO / "data/train/osler_v0/train.jsonl"
THRESHOLD = 0.8


def dominant_templates(audit: Mapping[str, Any], threshold: float) -> dict[str, dict[str, Any]]:
    """The templates whose most frequent predicted option holds more than the threshold share of their items."""
    return {t: v for t, v in audit["by_template_both_orders"].items() if v["top_predicted_share"] > threshold}


def dev_share(dev_rows: Sequence[Mapping[str, Any]], template: str) -> dict[str, Any] | None:
    rows = [r for r in dev_rows if r["template_id"] == template]
    if not rows:
        return None
    top_key, top_n = collections.Counter(r["predicted_key"] for r in rows).most_common(1)[0]
    return {"n": len(rows), "accuracy": sum(float(r["correct"]) for r in rows) / len(rows),
            "top_predicted_key": top_key, "top_predicted_share": top_n / len(rows)}


def train_labels(rows: Iterable[Mapping[str, Any]], templates: set[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, collections.Counter] = {t: collections.Counter() for t in templates}
    for row in rows:
        if row["template_id"] in templates:
            counts[row["template_id"]][row["gold"]] += 1
    return {t: dict(c) for t, c in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="the calibrated O11 output name, e.g. osler_0p8b")
    parser.add_argument("--dev-root", type=Path, required=True,
                        help="the arm or O10 output directory with dev_predictions.jsonl (for example outputs/osler_v0/O10/osler_0p8b)")
    args = parser.parse_args()
    audit = read_json(CAL / args.name / "audit.json")
    dominant = dominant_templates(audit, THRESHOLD)
    dev_rows = list(iter_jsonl(REPO / args.dev_root / "dev_predictions.jsonl"))
    train = train_labels(iter_jsonl(TRAIN), set(dominant))
    record: dict[str, Any] = {
        "kind": "osler_v0_O11_constant_check",
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "command": f"scripts/osler/o11_constant_check.py --name {args.name} --dev-root {args.dev_root}",
        "model": args.name,
        "threshold": THRESHOLD,
        "templates": len(audit["by_template_both_orders"]),
        "dominant": {},
    }
    for template, v in sorted(dominant.items(), key=lambda kv: -kv[1]["top_predicted_share"]):
        record["dominant"][template] = {
            "test": {"n": v["n"], "accuracy": v["accuracy"], "top_predicted_key": v["top_predicted_key"],
                     "top_predicted_share": v["top_predicted_share"], "top_gold_share": v["top_gold_share"]},
            "dev": dev_share(dev_rows, template),
            "train_gold_counts": train.get(template, {}),
        }
    write_json(CAL / args.name / "constant_check.json", record)
    for template, d in record["dominant"].items():
        print(template, "test", d["test"]["n"], round(d["test"]["top_predicted_share"], 3), d["test"]["top_predicted_key"],
              "| dev", d["dev"] and (d["dev"]["n"], round(d["dev"]["top_predicted_share"], 3)),
              "| train rows", sum(d["train_gold_counts"].values()))


if __name__ == "__main__":
    main()
