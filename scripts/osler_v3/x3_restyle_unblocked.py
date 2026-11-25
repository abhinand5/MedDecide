"""osler_v3 X3 restyle, unblocked by the operator's ruling of 2026-10-19 (D-X3-16): the verified restyled pages become rows.

The ruling: under D44(b) the genre vocabulary of the requested page type is the labelled fact. All kept (verified) pages are used,
not class-balanced. The genre vocabulary is declared here, before the check runs, as three parts (nothing is tuned afterwards):
  (a) the words of the page type's label (meddecide.mr.pages.PAGE_CLASSES), which name the type itself;
  (b) the words of heading lines (a short line that ends with a colon) that recur in at least 10 % of that class's kept pages;
  (c) the two words the operator named for radiology ("impression", "demonstrates").
A label-tracking phrase is flagged only when it contains none of these words (the osler_v2 phrase test and thresholds, through the
assembler's phrase_check). Every flagged phrase that remains is counted and reported; nothing is dropped silently.

Protected-paragraph matches are dropped as in the assembler (paragraphs of 60 characters or more that occur in a protected set),
and the count is recorded. Writes: data/train/osler_v3/x3_restyle_train.jsonl (rows), data/bench/osler_v3/x3_dev.jsonl and
x3_test.jsonl (the fast dev v3 and new test pools read these), outputs/osler_v3/X3/restyle_unblocked.json (aggregates only).

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x3_restyle_unblocked.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.gen.screen import NaiveBayes, gold_label, macro_accuracy
from meddecide.mix import restyle as rs
from meddecide.mix.protected import (
    hit_total,
    leakage_counts,
    load_protected,
    paragraph_key,
    protected_paragraph_keys,
)
from meddecide.mix.substrate import paragraphs
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "data" / "gen" / "osler_v3"
OUT = REPO / "outputs" / "osler_v3" / "X3" / "restyle_unblocked.json"
ROWS = {
    "train": REPO / "data" / "train" / "osler_v3" / "x3_restyle_train.jsonl",
    "dev": REPO / "data" / "bench" / "osler_v3" / "x3_dev.jsonl",
    "test": REPO / "data" / "bench" / "osler_v3" / "x3_test.jsonl",
}
SPLITS = ("train", "dev", "test")
HEADING_SHARE = 0.10  # (b): a heading recurring in at least 10 % of the class's pages
OPERATOR_WORDS = {"impression", "demonstrates"}  # (c): named by the operator for radiology
SCREEN_THRESHOLD = 0.90
CLASS_CAP = 0.25
HEADING = re.compile(r"^\s*([A-Za-z][A-Za-z /&()-]{0,60}):\s*$", re.MULTILINE)


def _assembler() -> Any:
    spec = importlib.util.spec_from_file_location("x3_assemble", REPO / "scripts" / "osler_v3" / "x3_assemble.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_candidates(split: str) -> list[dict[str, Any]]:
    with (GEN / f"x3_restyle_{split}.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def genre_words(pages_by_class: dict[str, list[str]]) -> dict[str, set[str]]:
    """Per class: (a) label words, (b) recurring heading words, (c) the operator's named words for radiology."""
    out: dict[str, set[str]] = {}
    for cls, texts in pages_by_class.items():
        words = set(re.findall(r"[a-z]+", rs.PAGE_LABEL[cls].lower()))
        counts: Counter[str] = Counter()
        for text in texts:
            heads = {m.group(1).strip().lower() for m in HEADING.finditer(text)}
            counts.update(heads)
        for head, n in counts.items():
            if n >= HEADING_SHARE * len(texts):
                words.update(re.findall(r"[a-z]+", head))
        if cls == "radiology":
            words |= OPERATOR_WORDS
        out[cls] = words
    return out


def main() -> int:
    asm = _assembler()
    protected = protected_paragraph_keys(REPO)
    text_keys, bare, qualified = load_protected(REPO)
    kept: dict[str, list[dict[str, Any]]] = {}
    dropped_protected = Counter()
    for split in SPLITS:
        rows = [r for r in read_candidates(split) if r.get("kept") is True]
        keep = []
        for r in rows:
            body = r.get("page") or ""
            if any(len(p) >= 60 and paragraph_key(p) in protected for p in paragraphs(body)):
                dropped_protected[split] += 1
                continue
            keep.append(r)
        kept[split] = keep
    built: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        built[split] = asm.page_rows(kept[split])
    counts = {s: len(built[s]) for s in SPLITS}
    all_rows = [r for s in SPLITS for r in built[s]]
    shares: dict[str, dict[str, float]] = {}
    for s in SPLITS:
        c = Counter(r["meta"]["page_class"] for r in built[s])
        shares[s] = {cls: round(n / len(built[s]), 4) for cls, n in sorted(c.items())}
    largest = max(max(v.values()) for v in shares.values() if v)
    pages_by_class: dict[str, list[str]] = defaultdict(list)
    for r in built["train"]:
        pages_by_class[r["meta"]["page_class"]].append(r["state"])
    genre = genre_words(pages_by_class)
    exempt_all = set().union(*genre.values())
    page_texts = dict(pages_by_class)
    phrases_all = asm.phrase_check(page_texts, set())
    phrases_exempt = asm.phrase_check(page_texts, exempt_all)
    nb = NaiveBayes().fit(built["train"])
    dev_pairs = [(gold_label(r), nb.predict(r)) for r in built["dev"]]
    macro = macro_accuracy(dev_pairs)
    majority = Counter(gold_label(r) for r in built["train"]).most_common(1)[0][0]
    leak = {s: leakage_counts(built[s], text_keys, bare, qualified) for s in SPLITS}
    paragraph_hits = sum(1 for r in all_rows for p in paragraphs(r["state"])
                         if len(p) >= 60 and paragraph_key(p) in protected)
    for s in SPLITS:
        write_jsonl(ROWS[s], built[s])
    report: dict[str, Any] = {
        "built_utc": utcnow(), "git_commit": git_commit(REPO),
        "ruling": "operator, 2026-10-19 (D-X3-16): genre vocabulary is the labelled fact; all verified pages used, not balanced",
        "rows": counts, "dropped_protected_paragraph": dict(dropped_protected),
        "class_shares": shares, "largest_class_share": round(largest, 4), "class_cap": CLASS_CAP,
        "class_cap_holds": largest <= CLASS_CAP + 1e-12,
        "genre_words_per_class": {cls: sorted(words) for cls, words in genre.items()},
        "phrase_check_without_exemption_flagged": phrases_all["flagged_count"],
        "phrase_check_with_genre_exemption_flagged": phrases_exempt["flagged_count"],
        "phrase_check_flagged_groups_listed": dict(Counter(e["label"] for e in phrases_exempt["flagged"])),
        "phrase_check_flagged_examples_are_not_recorded": True,
        "nb_screen_dev": {"train_rows": len(built["train"]), "dev_rows": len(built["dev"]),
                          "naive_bayes_macro": None if macro is None else round(macro, 4),
                          "majority_dev_accuracy": round(sum(gold_label(r) == majority for r in built["dev"]) / len(built["dev"]), 4),
                          "passes_0_90": bool(macro is not None and macro < SCREEN_THRESHOLD)},
        "leakage_hits": {s: hit_total(leak[s]) for s in SPLITS}, "paragraph_hits_in_protected_states": paragraph_hits,
        "files": {s: str(ROWS[s].relative_to(REPO)) for s in SPLITS},
    }
    write_json(OUT, report)
    print({k: report[k] for k in ("rows", "largest_class_share", "phrase_check_with_genre_exemption_flagged",
                                  "nb_screen_dev", "leakage_hits", "paragraph_hits_in_protected_states")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
