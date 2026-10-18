"""osler_v3 X4 screen diagnostic (ADVISORY X4, section 7.4): why the Naive Bayes screen is low on the choice templates.

The O3 screen (meddecide.gen.screen.NaiveBayes, gold_label) scores a choice item by its gold option's TEXT. On chart_icd_support_v1
and chart_evidence_page_v1 the offered options differ from item to item (2-10 codes; 5-12 page names), so two things can keep the
screen low, and this script measures both on dev, for the record (the verdicts come from the label-tracking check, not from here):
  - seen: the share of dev items whose gold option text occurs as a gold option text in the train split at all;
  - outside: the share of dev items whose NB prediction is not one of the item's offered options (it cannot be right).
It also reports the screen's macro accuracy and, for comparison, the chance level of a uniform guess over the offered options.

Writes outputs/osler_v3/X4/screen_diagnostic.json (aggregates only). Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x4_screen_diagnostic.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.gen.screen import NaiveBayes, gold_label, macro_accuracy
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
CHARTS = REPO / "data" / "gen" / "osler_v3" / "charts"
OUT = REPO / "outputs" / "osler_v3" / "X4" / "screen_diagnostic.json"
TEMPLATES = ("chart_icd_support_v1", "chart_evidence_page_v1", "chart_indicator_evidence_v1")


def rows_of(split: str) -> list[dict[str, Any]]:
    with (CHARTS / f"{split}.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    rows = {s: rows_of(s) for s in ("train", "dev")}
    report: dict[str, Any] = {"built_utc": utcnow(), "git_commit": git_commit(REPO), "templates": {}}
    for template in TEMPLATES:
        train = [r for r in rows["train"] if r["generator"] == template]
        dev = [r for r in rows["dev"] if r["generator"] == template]
        seen_texts = {gold_label(r) for r in train}
        nb = NaiveBayes().fit(train)
        preds = [(gold_label(r), nb.predict(r)) for r in dev]
        offered = [{o["label"] for o in r["options"]} for r in dev]
        outside = sum(1 for (_, p), off in zip(preds, offered, strict=True) if p not in off)
        chance = sum(1.0 / len(off) for off in offered) / len(offered)
        top_n = Counter(p for _, p in preds).most_common(1)[0][1]
        states = Counter(str(r["state"]) for r in dev)
        report["templates"][template] = {
            "train_items": len(train), "dev_items": len(dev),
            "dev_nb_distinct_predictions": len({p for _, p in preds}),
            "dev_nb_most_frequent_prediction_share": round(top_n / len(dev), 4),
            "dev_items_sharing_their_state_with_another_dev_item": sum(states[str(r["state"])] > 1 for r in dev),
            "dev_gold_majority_share": round(Counter(g for g, _ in preds).most_common(1)[0][1] / len(dev), 4),
            "dev_gold_text_seen_in_train": sum(gold_label(r) in seen_texts for r in dev),
            "dev_nb_prediction_outside_offered_options": outside,
            "dev_nb_macro_accuracy": None if macro_accuracy(preds) is None else round(macro_accuracy(preds), 4),
            "dev_uniform_guess_over_offered_options_mean": round(chance, 4),
            "dev_offered_options_distribution": dict(sorted(Counter(len(off) for off in offered).items())),
        }
        print(f"{template}: dev {len(dev)}, gold text seen in train {report['templates'][template]['dev_gold_text_seen_in_train']}, "
              f"NB outside offered {outside}, NB macro {report['templates'][template]['dev_nb_macro_accuracy']}, "
              f"uniform over offered {round(chance, 4)}", flush=True)
    write_json(OUT, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
