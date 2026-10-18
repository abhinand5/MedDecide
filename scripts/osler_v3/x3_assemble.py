"""osler_v3 X3 assembly (ADVISORY X3 steps 1-3; D43, D44(b)): keep the verified candidates, balance them, check them, write the rows.

Input: the candidate files written by x3_generate.py (data/gen/osler_v3/x3_claims_<split>.jsonl and x3_restyle_<split>.jsonl).
Every candidate carries its verdict and its keep flag; only kept candidates enter the rows. Balancing is by a stable-hash order:
- claims: contradicted items are balanced to the smallest edit kind; then the three labels to the smallest label;
- restyled pages: the twelve-class rows are balanced to the smallest narrative class (no class above 25 %, ADVISORY).

Checks, all reported (none filters silently):
- leakage against every protected set (meddecide.mix.protected) and paragraph-level against the protected states (must be 0);
- the O3 Naive Bayes screen on dev per template (a macro accuracy at or above 0.90 is a defect for that template, ADVISORY §7.4);
- the label-tracking phrase check (D44(b)): n-grams that track the label, excluding the labelled fact (the edit word for a
  contradicted number/laterality/negation/drug/temporal item; the page class name for a restyled page). A phrase that is not the
  labelled fact blocks its own template only.

Writes data/train/osler_v3/x3_train.jsonl, data/bench/osler_v3/x3_dev.jsonl, x3_test.jsonl (rows; gitignored),
outputs/osler_v3/X3/build.json (counts, rates, checks), and outputs/osler_v3/X3/readthrough_*.json (samples for the spot read;
gitignored, never pasted into committed files).

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x3_assemble.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.gen.screen import NaiveBayes, gold_in_state, gold_label, macro_accuracy
from meddecide.mix import clinclaim as cc
from meddecide.mix import restyle as rs
from meddecide.mix.clinclaim import balance_claims
from meddecide.mix.protected import (
    hit_total,
    leakage_counts,
    load_protected,
    paragraph_key,
    protected_paragraph_keys,
)
from meddecide.mix.substrate import paragraphs
from meddecide.mr.realtext import PAGE_LETTER
from meddecide.utils.hashing import stable_hash
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "data" / "gen" / "osler_v3"
OUT = REPO / "outputs" / "osler_v3" / "X3"
TRAIN = REPO / "data" / "train" / "osler_v3" / "x3_train.jsonl"
DEV = REPO / "data" / "bench" / "osler_v3" / "x3_dev.jsonl"
TEST = REPO / "data" / "bench" / "osler_v3" / "x3_test.jsonl"
SCREEN_THRESHOLD = 0.90
PHRASE_MIN_ITEMS = 20  # a phrase must occur this often to be checked
PHRASE_RATIO = 5.0  # a phrase is label-tracking when its share in one label is this many times its share in every other label
PHRASE_SHARE = 0.05
READ_N = 30
SPLITS = ("train", "dev", "test")
# the claim template is blocked after its one allowed prompt fix (D-X3-1): its not-mentioned claims carry a label-tracking
# phrase ("family", "family members") in 39 % of items against 0 % of the other labels (D44(b)); no claim rows are written
BLOCKED = {"clinclaim_grounding_v1": "label-tracking phrase 'family' / 'family members' in 39 % of not-mentioned items and 0 % of "
                                    "the others, after the one allowed prompt fix (D-X3-1; D44(b))"}


def read_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def balance_pages(rows: list[dict[str, Any]], salt: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_class[r["page_class"]].append(r)
    per = min((len(v) for v in by_class.values()), default=0)
    kept = [r for cls in sorted(by_class) for r in
            sorted(by_class[cls], key=lambda x: stable_hash({"salt": salt, "key": x["key"]}, length=16))[:per]]
    report = {"before": dict(Counter(r["page_class"] for r in rows)), "per_class_target": per, "after": len(kept),
              "largest_class_share": round(max(Counter(r["page_class"] for r in kept).values()) / len(kept), 4) if kept else None}
    return kept, report


def claim_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        meta = {"source": r["source"], "doc_id": r["doc_id"], "label": r["label"], "kind": r["kind"],
                "copy_share": r["copy_share"]}
        rows.append(cc.build_row(r["key"], r["passage"], r["claim"], r["label"], meta, licence=r["licence"],
                                 record_url=r["record_url"], split=r["split"]))
    return rows


def page_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [rs.build_row(r["key"], r["page"], r["page_class"], {"source": r["source"], "doc_id": r["doc_id"]},
                         licence=r["licence"], record_url=r["record_url"], split=r["split"]) for r in records]


def agreement(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per edit kind: candidates, verifier agreement with the drawn label, edit-rule pass and keep (ADVISORY §7.3)."""
    out: dict[str, Any] = {}
    for kind in cc.EDIT_KINDS:
        rs_ = [r for r in records if r["label"] == "contradicted" and r.get("kind") == kind]
        out[kind] = {"candidates": len(rs_), "verifier_agrees": sum(r.get("verdict") == "contradicted" for r in rs_),
                     "kept": sum(bool(r["kept"]) for r in rs_)}
    for label in cc.LABELS:
        rs_ = [r for r in records if r["label"] == label]
        out[label] = {"candidates": len(rs_), "verifier_agrees": sum(r.get("verdict") == label for r in rs_),
                      "kept": sum(bool(r["kept"]) for r in rs_)}
    return out


def phrase_check(texts_by_label: dict[str, list[str]], exempt: set[str]) -> dict[str, Any]:
    """Word 1-3-grams that track the label: share in one label at least PHRASE_RATIO times its share in every other label."""
    doc_grams: dict[str, list[set[str]]] = {}
    for label, texts in texts_by_label.items():
        doc_grams[label] = []
        for text in texts:
            words = re.findall(r"[a-z]+", text.lower())
            grams = set()
            for n in (1, 2, 3):
                grams.update(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
            doc_grams[label].append(grams)
    totals = {label: len(v) for label, v in doc_grams.items()}
    counts: dict[str, Counter[str]] = {label: Counter() for label in doc_grams}
    for label, sets in doc_grams.items():
        for grams in sets:
            counts[label].update(grams)
    flagged = []
    vocabulary = set().union(*(set(c) for c in counts.values())) if counts else set()
    for gram in vocabulary:
        if gram in exempt or any(w in exempt for w in gram.split()):
            continue
        shares = {label: counts[label][gram] / totals[label] for label in totals if totals[label]}
        if not shares or sum(counts[label][gram] for label in totals) < PHRASE_MIN_ITEMS:
            continue
        for label, share in shares.items():
            others = max((s for other, s in shares.items() if other != label), default=0.0)
            if share >= PHRASE_SHARE and share >= PHRASE_RATIO * others:
                flagged.append({"phrase": gram, "label": label, "share": round(share, 4), "others_max_share": round(others, 4),
                                "items": counts[label][gram]})
    flagged.sort(key=lambda f: (-f["share"], f["phrase"]))
    return {"flagged": flagged[:50], "flagged_count": len(flagged)}


def screen(rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], template: str) -> dict[str, Any]:
    train = [r for r in rows if r["template_id"] == template]
    dev = [r for r in dev_rows if r["template_id"] == template]
    if not train or not dev:
        return {"template": template, "status": "NOT MEASURED — no dev or train rows"}
    nb = NaiveBayes().fit(train)
    macro = macro_accuracy([(gold_label(r), nb.predict(r)) for r in dev])
    majority = Counter(gold_label(r) for r in train).most_common(1)[0][0]
    return {"template": template, "train_rows": len(train), "dev_rows": len(dev),
            "naive_bayes_macro": None if macro is None else round(macro, 4),
            "majority_dev_accuracy": round(sum(gold_label(r) == majority for r in dev) / len(dev), 4),
            "flag_at_or_above_0_90": bool(macro is not None and macro >= SCREEN_THRESHOLD)}


def main() -> int:
    text, bare, qualified = load_protected(REPO)
    protected_paragraphs = protected_paragraph_keys(REPO)
    report: dict[str, Any] = {"built_utc": utcnow(), "git_commit": git_commit(REPO), "command": "scripts/osler_v3/x3_assemble.py",
                              "candidates": {}, "agreement": {}, "balance": {}, "rows": {}, "checks": {}}
    rows_by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    kept_records: dict[str, dict[str, list[dict[str, Any]]]] = {"claims": {}, "restyle": {}}
    for task, name in (("claims", "x3_claims"), ("restyle", "x3_restyle")):
        if cc.TEMPLATE in BLOCKED:
            report["blocked"] = BLOCKED
            if task == "claims":
                for split in SPLITS:
                    report["candidates"][f"{task}/{split}"] = {"written": len(read_candidates(GEN / f"{name}_{split}.jsonl")),
                                                               "rows_written": 0, "status": "BLOCKED"}
                continue
        all_records: list[dict[str, Any]] = []
        for split in SPLITS:
            records = read_candidates(GEN / f"{name}_{split}.jsonl")
            all_records.extend(records)
            reasons = Counter(r.get("reason", "missing") for r in records)
            report["candidates"][f"{task}/{split}"] = {"written": len(records), "kept": sum(bool(r.get("kept")) for r in records),
                                                       "reasons": dict(reasons)}
        report["agreement"][task] = agreement(all_records) if task == "claims" else {
            "candidates": len(all_records),
            "verifier_agrees": sum(r.get("verified_letter") == PAGE_LETTER[r["page_class"]] for r in all_records),
            "kept": sum(bool(r["kept"]) for r in all_records)}
        for split in SPLITS:
            kept = [r for r in all_records if r["split"] == split and r.get("kept")]
            kept = [r for r in kept if not any(paragraph_key(p) in protected_paragraphs for p in paragraphs(r.get("passage") or r.get("page") or "")
                                              if len(p) >= 60)]
            if task == "claims":
                balanced, bal = balance_claims(kept, salt=f"{task}:{split}")
                built = claim_rows(balanced)
            else:
                balanced, bal = balance_pages(kept, salt=f"{task}:{split}")
                built = page_rows(balanced)
            report["balance"][f"{task}/{split}"] = bal
            kept_records[task][split] = balanced
            rows_by_split[split].extend(built)
            report["rows"][f"{task}/{split}"] = len(built)
    all_rows = [r for s in SPLITS for r in rows_by_split[s]]
    leak = leakage_counts(all_rows, text, bare, qualified)
    report["checks"]["leakage"] = {s: leakage_counts(rows_by_split[s], text, bare, qualified) for s in SPLITS}
    report["checks"]["paragraph_hits_in_protected_states"] = sum(
        1 for r in all_rows for p in paragraphs(r["state"]) if len(p) >= 60 and paragraph_key(p) in protected_paragraphs)
    report["checks"]["screens"] = [screen(rows_by_split["train"], rows_by_split["dev"], t) for t in (cc.TEMPLATE, rs.TEMPLATE)
                                   if t not in BLOCKED]
    report["checks"]["gold_in_state"] = {
        s: sum(1 for r in rows_by_split[s] if r["template_id"] == rs.TEMPLATE and gold_in_state(r)) for s in SPLITS}
    if cc.TEMPLATE not in BLOCKED:
        claim_texts = {lab: [r["question"] for r in rows_by_split["train"] if r["template_id"] == cc.TEMPLATE
                             and r["meta"]["label"] == lab] for lab in cc.LABELS}
        exempt = {"contradicted", "supported", "not", "mentioned", "passage", "claim"}
        report["checks"]["phrase_claims"] = phrase_check(claim_texts, exempt)
    page_texts = {cls: [r["state"] for r in rows_by_split["train"] if r["template_id"] == rs.TEMPLATE
                        and r["meta"]["page_class"] == cls] for cls in rs.RESTYLE_CLASSES}
    page_exempt = {w for label in rs.PAGE_LABEL.values() for w in re.findall(r"[a-z]+", label.lower())}
    report["checks"]["phrase_pages"] = phrase_check(page_texts, page_exempt)
    report["checks"]["leakage_all_rows"] = leak
    for split in SPLITS:
        rows_by_split[split] = [r for r in rows_by_split[split] if r["template_id"] not in BLOCKED]
    write_jsonl(TRAIN, rows_by_split["train"])
    write_jsonl(DEV, rows_by_split["dev"])
    write_jsonl(TEST, rows_by_split["test"])
    report["files"] = {"train": str(TRAIN.relative_to(REPO)), "dev": str(DEV.relative_to(REPO)), "test": str(TEST.relative_to(REPO))}
    write_json(OUT / "build.json", report)
    rng = random.Random(0)
    all_candidates = {task: [r for name in (f"x3_{task}",) for s in SPLITS for r in read_candidates(GEN / f"{name}_{s}.jsonl")]
                      for task in ("claims", "restyle")}
    rejected = [r for r in all_candidates["claims"] if not r.get("kept")]
    write_json(OUT / "readthrough_rejected_claims.json", [{k: r.get(k) for k in ("key", "label", "kind", "reason", "claim",
                                                                             "verdict", "passage")}
                                                         for r in rng.sample(rejected, min(READ_N, len(rejected)))])
    for task in ("claims", "restyle"):
        allk = [r for s in SPLITS for r in kept_records[task].get(s, [])]
        sample = rng.sample(allk, min(READ_N, len(allk)))
        write_json(OUT / f"readthrough_kept_{task}.json", [{k: r.get(k) for k in ("key", "label", "kind", "page_class", "claim",
                                                                             "page", "passage")} for r in sample])
    if hit_total(leak) or report["checks"]["paragraph_hits_in_protected_states"]:
        print("LEAKAGE: the rows hit a protected set; fix before any run", file=sys.stderr)
        return 1
    print(f"x3: rows train {len(rows_by_split['train'])}, dev {len(rows_by_split['dev'])}, test {len(rows_by_split['test'])}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
