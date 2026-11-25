"""osler_v4 Y7 (ADVISORY Y7, B4 pre-check; D35): the shortcut screens of osler_v3's X1 real-source templates (CPU).

For each template of the X1 train file: the naive-Bayes macro accuracy on its fast dev v4 items (the naive Bayes is fit on the template's
own train rows), the string-presence macro (choice items; the training majority label otherwise), and the share of dev items whose gold
label occurs in the record text. The D35 limit is 0.90 for the two macro screens. Prints counts and figures only; no item text is printed.

Usage (CPU):
    uv run --frozen python scripts/osler_v4/y7_x1_screen.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from meddecide.gen.screen import (
    NaiveBayes,
    gold_in_state,
    gold_label,
    macro_accuracy,
    string_presence_predict,
)

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "data" / "train" / "osler_v3" / "x1_sources_train.jsonl"
DEV = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    train: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows(TRAIN):
        train[str(row["template_id"])].append(row)
    dev: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows(DEV):
        if str(row["template_id"]) in train:
            dev[str(row["template_id"])].append(row)
    for template in sorted(train):
        items = dev[template]
        nb = NaiveBayes().fit(train[template])
        nb_macro = macro_accuracy([(gold_label(r), nb.predict(r)) for r in items])
        majority = collections.Counter(gold_label(r) for r in train[template]).most_common(1)[0][0]
        choice = [r for r in items if r["qtype"] != "noul"]
        sp_macro = macro_accuracy([(gold_label(r), string_presence_predict(r, majority)) for r in choice]) if choice else None
        in_state = sum(1 for r in items if gold_in_state(r))
        sp_text = "NOT APPLICABLE (yes/no items only)" if sp_macro is None else f"{sp_macro:.4f}"
        print(f"{template}: train {len(train[template])}, dev {len(items)}, NB macro dev {nb_macro:.4f}, "
              f"string-presence macro dev {sp_text}, gold in record text {in_state} of {len(items)}", flush=True)


if __name__ == "__main__":
    main()
