"""O4 audit: per-template bag-of-words learnability and gold visibility in training mix v2 (additional; not a gate).

For each template, a deterministic sample of up to 4,000 training rows (smallest stable hash of the item id) trains a
bag-of-words Naive Bayes, which is scored on the template's dev rows. Reported beside it: the majority-class macro on the
same dev rows (the reference for that label set), the gold-in-state rate on dev for choice and score items, and the
template's train and dev counts. Nothing here selects or gates anything (R4): it describes what the data look like.

Outputs (gitignored): outputs/osler_v0/O4/audit_mix.json.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o4_audit_mix.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from meddecide.gen.screen import NaiveBayes, gold_label, macro_accuracy, screen_generator
from meddecide.mix.audit import smallest_hash_rows
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "data/train/osler_v0/train.jsonl"
DEV = REPO / "data/train/osler_v0/dev.jsonl"
OUT = REPO / "outputs/osler_v0/O4/audit_mix.json"
SAMPLE = 4000


def stream(path: Path, counts: Counter | None = None) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if counts is not None:
                counts[row["template_id"]] += 1
            yield row


def with_pair(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "pair_id": row["item_id"]}


def main() -> None:
    train_counts: Counter = Counter()
    sample = smallest_hash_rows(stream(TRAIN, train_counts), SAMPLE)
    dev_by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stream(DEV):
        dev_by_template[row["template_id"]].append(with_pair(row))
    templates: dict[str, Any] = {}
    for template in sorted(set(sample) & set(dev_by_template)):
        train_rows = [with_pair(r) for r in sample[template]]
        dev_rows = dev_by_template[template]
        rep = screen_generator(template, False, dev_rows, train_rows,
                               f"smallest-hash sample of the train split (up to {SAMPLE} rows)")
        majority = Counter(gold_label(r) for r in train_rows).most_common(1)[0][0]
        majority_macro = macro_accuracy([(gold_label(r), majority) for r in dev_rows])
        nb = NaiveBayes().fit(train_rows)
        predictions = [nb.predict(r) for r in dev_rows]
        nb_accuracy = sum(gold_label(r) == p for r, p in zip(dev_rows, predictions, strict=True)) / len(dev_rows)
        chance_accuracy = sum(1 / len(r["options"]) for r in dev_rows) / len(dev_rows)
        phrase_dev = [r for r in dev_rows if r["qtype"] != "noul"]
        templates[template] = {
            "qtype": Counter(r["qtype"] for r in dev_rows).most_common(1)[0][0],
            "n_train_rows": train_counts[template],
            "n_train_sampled": len(train_rows),
            "n_dev": len(dev_rows),
            "gold_in_state_dev_hits": rep.gold_in_state_hits,
            "gold_in_state_dev_applicable": len(phrase_dev),
            "naive_bayes_macro_dev": rep.naive_bayes_macro,
            "naive_bayes_accuracy_dev": round(nb_accuracy, 4),
            "chance_accuracy_dev": round(chance_accuracy, 4),
            "majority_macro_dev": None if majority_macro is None else round(majority_macro, 4),
            "naive_bayes_minus_majority": None if rep.naive_bayes_macro is None or majority_macro is None
            else round(rep.naive_bayes_macro - majority_macro, 4),
            "screen_status": "PASS" if rep.passes else "FAIL",
            "screen_failures": rep.failures,
        }
        print(f"{template:36s} n_train={train_counts[template]:6d} n_dev={len(dev_rows):5d} "
              f"nb={rep.naive_bayes_macro} majority={templates[template]['majority_macro_dev']} "
              f"gold_in_state={rep.gold_in_state_hits}", flush=True)
    write_json(OUT, {
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "method": f"Naive Bayes over record text and question tokens, trained on up to {SAMPLE} training rows per "
                  "template (smallest stable hash of the item id), scored on the template's dev rows; majority-class "
                  "macro on the same dev rows; gold-in-state on dev for choice and score items",
        "train_rows_total": sum(train_counts.values()),
        "templates": templates,
    })


if __name__ == "__main__":
    main()
