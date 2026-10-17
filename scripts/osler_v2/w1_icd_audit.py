"""W1 step 1 (osler_v2 ADVISORY W1): the too-good ICD template, mr_icd_support_template_v2 (1.000 on 554 test items in
osler_v1 V10).

(a) A rule-level probe answers each test item from its rendered text with no model: it finds the sentences that name a
    candidate condition, judges each sentence affirmed or not from negation, uncertainty and family or history words, and
    picks the most specific code whose label words the affirmed sentence contains. Accuracy is reported as measured.
(b) An exact and near-duplicate overlap check: test states against every row of osler_v1 block B1 (the training block of
    the same template family). Near-duplicate = Jaccard ≥ 0.9 over word 8-grams of the normalised state.

Neither step changes the template. Writes outputs/osler_v2/W1/icd_audit.json (counts and rates only).

Usage: uv run --frozen python scripts/osler_v2/w1_icd_audit.py
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
TEST = REPO / "data" / "bench" / "osler_mr_v0" / "mr_eval.jsonl"
B1 = REPO / "data" / "train" / "osler_v1" / "block_B1.jsonl"
OUT = REPO / "outputs" / "osler_v2" / "W1" / "icd_audit.json"
TEMPLATE = "mr_icd_support_template_v2"
NEAR_DUP = 0.9
NGRAM = 8

NEGATION = ("ruled out", "rule out", "no evidence", "negative for", "without", "no ", "not ", "denies", "absence",
            "resolved", "excluded", "no sign")
UNCERTAIN = ("possible", "suspected", "probable", "likely", "questionable", "may ", "concern for", "cannot")
HISTORY = ("history of", "family", "mother", "father", "sibling", "brother", "sister", "prior", "previous", "past ")
STOP = {"of", "the", "and", "with", "without", "in", "to", "a", "an", "by", "for", "due", "on", "unspecified", "or",
        "whether", "other", "specified", "type"}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def condition_name(label: str) -> str:
    """'J96.01 - Acute respiratory failure with hypoxia' -> 'Acute respiratory failure with hypoxia'."""
    return label.split(" - ", 1)[1] if " - " in label else label


def is_none_option(label: str) -> bool:
    return label.startswith("None of these")


def sentences(state: str) -> list[str]:
    return [s.strip() for s in re.split(r"[\n.]", state) if s.strip()]


def affirmed(sentence: str) -> bool:
    low = sentence.lower()
    return not any(cue in low for cue in NEGATION + UNCERTAIN + HISTORY)


def probe_answer(item: dict[str, Any]) -> str:
    """Return the option key the rule probe chooses (or the 'None' key)."""
    options = item["options"]
    none_key = next((o["key"] for o in options if is_none_option(o["label"])), None)
    candidates = [o for o in options if not is_none_option(o["label"])]
    if not candidates:
        return none_key
    name_words = {c["key"]: [w for w in words(condition_name(c["label"])) if w not in STOP] for c in candidates}
    # words shared by every candidate name carry no information; with one candidate, all of its words count
    common = set.intersection(*(set(v) for v in name_words.values())) if len(candidates) > 1 else set()
    best_key, best_rank = None, (0, 0.0)
    for sent in sentences(item["state"]):
        if not affirmed(sent):
            continue
        sent_words = set(words(sent))
        for c in candidates:
            distinctive = [w for w in name_words[c["key"]] if w not in common]
            if not distinctive or not all(w in sent_words for w in distinctive):
                continue
            label_words = set(name_words[c["key"]])
            # the more specific code wins; among equally specific codes, the closest wording wins
            rank = (len(distinctive), len(label_words & sent_words) / len(label_words | sent_words))
            if rank > best_rank:
                best_key, best_rank = c["key"], rank
    return best_key if best_key is not None else none_key


def normalised(state: str) -> str:
    return " ".join(words(state))


def shingles(text: str) -> set[int]:
    toks = text.split()
    grams = [" ".join(toks[i:i + NGRAM]) for i in range(max(1, len(toks) - NGRAM + 1))]
    return {int(hashlib.blake2b(g.encode(), digest_size=8).hexdigest(), 16) for g in grams}


def load_items(split: str) -> list[dict[str, Any]]:
    items = []
    with TEST.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["template_id"] == TEMPLATE and row["set_name"] == "mr_test" and row["split"] == split:
                items.append(row)
    return items


def overlap(test_items: list[dict[str, Any]]) -> dict[str, Any]:
    test_norm = {it["item_id"]: normalised(it["state"]) for it in test_items}
    exact_set = {v: k for k, v in test_norm.items()}
    test_sh = {k: shingles(v) for k, v in test_norm.items()}
    index: dict[int, list[str]] = defaultdict(list)
    for k, sh in test_sh.items():
        for h in sh:
            index[h].append(k)
    scanned = icd_rows = exact_hits = 0
    near_pairs: set[str] = set()
    exact_ids: set[str] = set()
    with B1.open(encoding="utf-8") as fh:
        for line in fh:
            scanned += 1
            row = json.loads(line)
            if row.get("template_id") != TEMPLATE:
                continue
            icd_rows += 1
            norm = normalised(row["state"])
            if norm in exact_set:
                exact_hits += 1
                exact_ids.add(exact_set[norm])
            sh = shingles(norm)
            cands: dict[str, int] = defaultdict(int)
            for h in sh:
                for k in index.get(h, ()):
                    cands[k] += 1
            for k, shared in cands.items():
                union = len(sh | test_sh[k])
                if union and shared / union >= NEAR_DUP:
                    near_pairs.add(k)
    return {"b1_rows_scanned": scanned, "b1_icd_rows": icd_rows, "exact_normalised_state_matches": exact_hits,
            "test_items_with_exact_match": len(exact_ids), "test_items_with_near_duplicate": len(near_pairs),
            "jaccard_threshold": NEAR_DUP, "ngram": NGRAM}


def main() -> None:
    test = load_items("test")
    correct = 0
    by_gold_none = {"gold_none": [0, 0], "gold_code": [0, 0]}
    for it in test:
        pred = probe_answer(it)
        ok = pred == it["gold"]
        correct += ok
        none_key = next(o["key"] for o in it["options"] if is_none_option(o["label"]))
        bucket = "gold_none" if it["gold"] == none_key else "gold_code"
        by_gold_none[bucket][0] += ok
        by_gold_none[bucket][1] += 1
    result = {
        "template": TEMPLATE,
        "n_test": len(test),
        "probe_accuracy": round(correct / len(test), 4) if test else None,
        "probe_correct": correct,
        "probe_by_gold": {k: {"correct": v[0], "n": v[1]} for k, v in by_gold_none.items()},
        "overlap": overlap(test),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "overlap"}), flush=True)
    print(json.dumps(result["overlap"]))


if __name__ == "__main__":
    main()
