"""osler_v3 X1 checks (ADVISORY X1 step 4; GOAL "Do not"): leakage against every protected set, overlap between the X1 splits,
and the O3 Naive Bayes screen on dev per template.

The protected sets are meddecide.mix.protected.load_protected (osler_v0's training sets plus the MR suite v1 test and held-out
files, the knowledge dev v2, the fast dev v2 and the EHR-Rel test). A hit is a failure, not a filter: the script reports the
counts and exits non-zero if any training row hits a protected set. Nothing is removed silently.

Writes outputs/osler_v3/X1/checks.json (counts and screen numbers; no item text).
Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x1_checks.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.gen.screen import NaiveBayes, gold_label, macro_accuracy
from meddecide.mix.leakage import row_text_key
from meddecide.mix.protected import hit_total, leakage_counts, load_protected, read_jsonl
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "data" / "train" / "osler_v3" / "x1_sources_train.jsonl"
DEV = REPO / "data" / "bench" / "osler_v3" / "x1_dev.jsonl"
TEST = REPO / "data" / "bench" / "osler_v3" / "x1_test.jsonl"
OUT = REPO / "outputs" / "osler_v3" / "X1" / "checks.json"
SCREEN_THRESHOLD = 0.90  # the D35 vocabulary-shortcut threshold, used as a screen here (a flag is a reason to look)


def within_x1(train: list[dict[str, Any]], other: list[dict[str, Any]]) -> dict[str, int]:
    """Overlap between the training rows and the dev or test rows: the same text, the same PMID or the same RAGTruth context."""
    text_train = {row_text_key(r) for r in train}
    text_other = {row_text_key(r) for r in other}
    pmid_train = {str(r["meta"]["pmid"]) for r in train if r["meta"].get("pmid")}
    pmid_other = {str(r["meta"]["pmid"]) for r in other if r["meta"].get("pmid")}
    ctx_train = {str(r["source_record_id"]).split(":")[0] for r in train if r["source"] == "ragtruth"}
    ctx_other = {str(r["source_record_id"]).split(":")[0] for r in other if r["source"] == "ragtruth"}
    return {"text_keys_shared": len(text_train & text_other), "pmids_shared": len(pmid_train & pmid_other),
            "ragtruth_contexts_shared": len(ctx_train & ctx_other)}


def screen(template: str, train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> dict[str, Any]:
    """The O3 Naive Bayes screen on dev: a Naive Bayes fitted on the template's train rows, macro accuracy on its dev rows."""
    nb = NaiveBayes().fit(train)
    pairs = [(gold_label(r), nb.predict(r)) for r in dev]
    macro = macro_accuracy(pairs)
    majority_label = Counter(gold_label(r) for r in train).most_common(1)[0][0]
    majority_dev = sum(1 for r in dev if gold_label(r) == majority_label) / len(dev)
    chance = sum(1 / len(r["options"]) for r in dev) / len(dev)
    return {"template": template, "train_rows": len(train), "dev_rows": len(dev),
            "naive_bayes_macro": None if macro is None else round(macro, 4),
            "majority_dev_accuracy": round(majority_dev, 4), "chance": round(chance, 4),
            "flag_at_or_above_0_90": bool(macro is not None and macro >= SCREEN_THRESHOLD)}


def main() -> int:
    text, bare, qualified = load_protected(REPO)
    train, dev, test = read_jsonl(TRAIN), read_jsonl(DEV), read_jsonl(TEST)
    report: dict[str, Any] = {"checked_utc": utcnow(), "git_commit": git_commit(REPO),
                              "protected_sets": {"text": {k: len(v) for k, v in text.items()},
                                                 "bare_ids": {k: len(v) for k, v in bare.items()}},
                              "leakage": {}, "within_x1": {}, "screens": []}
    for name, rows in (("train", train), ("dev", dev), ("test", test)):
        report["leakage"][name] = leakage_counts(rows, text, bare, qualified)
    report["within_x1"] = {"train_vs_dev": within_x1(train, dev), "train_vs_test": within_x1(train, test)}
    for template in sorted({r["template_id"] for r in train}):
        report["screens"].append(screen(template, [r for r in train if r["template_id"] == template],
                                        [r for r in dev if r["template_id"] == template]))
    write_json(OUT, report)
    for name in ("train", "dev", "test"):
        print(f"leakage {name}: {json.dumps(report['leakage'][name], sort_keys=True)}", flush=True)
    print(f"within X1: {report['within_x1']}", flush=True)
    for row in report["screens"]:
        print(f"screen {row['template']}: NB macro {row['naive_bayes_macro']}, majority {row['majority_dev_accuracy']}, "
              f"chance {row['chance']}, flag {row['flag_at_or_above_0_90']}", flush=True)
    if hit_total(report["leakage"]["train"]):
        print("LEAKAGE: training rows hit a protected set; the build must be fixed before any run", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
