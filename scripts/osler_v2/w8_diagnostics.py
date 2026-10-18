"""osler_v2 W8 diagnostics (aggregates only; ADVISORY §7 defects and the real-text question).

1. Per-template accuracy of Osler-4B on the MR verdict set against MedDecider-4B and zero-shot pplx (paired items).
2. CodiEsp leakage check: exact and near-duplicate (Jaccard >= 0.9 over word 8-grams) matches between the MR test CodiEsp items
   and the CodiEsp training rows of the final mix C1 (the mix the final model trained on).
3. C1's dev accuracy on the CodiEsp carve-out (held out of training) against the CodiEsp test accuracy.
4. The too-good flag: mr_icd_support_template_v2's accuracy on its test items and its dev items (C1 and the final model).

Writes outputs/osler_v2/W8/diagnostics.json. No item text is written.

Usage: uv run --frozen python scripts/osler_v2/w8_diagnostics.py
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.utils.io import iter_jsonl, write_json

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data" / "bench" / "osler_v1" / "v10_items.jsonl"
MANIFEST = REPO / "data" / "bench" / "osler_mr_v1" / "manifest.json"
OSLER = REPO / "outputs" / "osler_v2" / "W8" / "osler_4b" / "both_orders.jsonl"
MEDDECIDER = REPO / "outputs" / "osler_v1" / "V6" / "baselines" / "meddecider-4b" / "predictions.jsonl"
PPLX = REPO / "outputs" / "osler_v2" / "W5" / "pplx_test" / "mr" / "pplx-decider-flash-v1.2" / "predictions.jsonl"
C1 = REPO / "data" / "train" / "osler_v2" / "block_C1.jsonl"
C1_DEV = REPO / "outputs" / "osler_v2" / "W6" / "C1_pplx" / "dev_predictions.jsonl"
FINAL_DEV = REPO / "outputs" / "osler_v2" / "W7" / "osler_4b" / "dev_predictions.jsonl"
OUT = REPO / "outputs" / "osler_v2" / "W8" / "diagnostics.json"
CODIESP = "codiesp_icd_support_v1"
TOO_GOOD = "mr_icd_support_template_v2"


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def shingles(text: str, n: int = 8) -> set[int]:
    toks = text.split()
    grams = [" ".join(toks[i:i + n]) for i in range(max(1, len(toks) - n + 1))]
    return {int(hashlib.blake2b(g.encode(), digest_size=8).hexdigest(), 16) for g in grams}


def correct(row: dict[str, Any]) -> bool:
    keys = row.get("option_keys") or row.get("original_keys")
    probs = row["probs"]
    return keys[max(range(len(keys)), key=lambda i: probs[i])] == row["gold"]


def per_template(verdict: set[str]) -> dict[str, Any]:
    template_of = {}
    for row in iter_jsonl(ITEMS):
        if row["v10_scope"] == "mr_test" and row["template_id"] in verdict:
            template_of[row["item_id"]] = row["template_id"]
    osler = {r["item_id"]: correct(r) for r in iter_jsonl(OSLER) if r["item_id"] in template_of and r["status"] == "scored"}
    md = {r["item_id"]: correct(r) for r in iter_jsonl(MEDDECIDER) if r["item_id"] in template_of and r["status"] == "scored"}
    pp = {r["item_id"]: correct(r) for r in iter_jsonl(PPLX) if r["item_id"] in template_of and r["status"] == "scored"}
    out: dict[str, Any] = {}
    for item_id, t in template_of.items():
        if item_id not in osler:
            continue
        c = out.setdefault(t, Counter())
        c["n"] += 1
        c["osler"] += osler[item_id]
        if item_id in md:
            c["n_md"] += 1
            c["osler_md"] += osler[item_id]
            c["md"] += md[item_id]
        if item_id in pp:
            c["n_pplx"] += 1
            c["osler_pplx"] += osler[item_id]
            c["pplx"] += pp[item_id]
    def rate(a: int, b: int) -> float | None:
        return round(a / b, 4) if b else None
    return {t: {"n": c["n"], "osler": rate(c["osler"], c["n"]), "n_meddecider": c["n_md"],
                "osler_vs_meddecider_items": rate(c["osler_md"], c["n_md"]), "meddecider": rate(c["md"], c["n_md"]),
                "n_pplx": c["n_pplx"], "osler_vs_pplx_items": rate(c["osler_pplx"], c["n_pplx"]),
                "pplx": rate(c["pplx"], c["n_pplx"])} for t, c in sorted(out.items())}


def codiesp_overlap() -> dict[str, Any]:
    test = {}
    for row in iter_jsonl(ITEMS):
        if row["v10_scope"] == "mr_test" and row["template_id"] == CODIESP:
            test[row["item_id"]] = norm(row["state"])
    exact_states = set(test.values())
    test_shingles = {k: shingles(v) for k, v in test.items()}
    index: dict[int, list[str]] = defaultdict(list)
    for k, sh in test_shingles.items():
        for h in sh:
            index[h].append(k)
    train_rows = exact = 0
    near: set[str] = set()
    with C1.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["template_id"] != CODIESP:
                continue
            train_rows += 1
            text = norm(row["state"])
            exact += text in exact_states
            sh = shingles(text)
            cands: dict[str, int] = defaultdict(int)
            for h in sh:
                for k in index.get(h, ()):
                    cands[k] += 1
            for k, shared in cands.items():
                if shared / len(sh | test_shingles[k]) >= 0.9:
                    near.add(k)
    return {"test_items": len(test), "c1_codiesp_train_rows": train_rows, "exact_row_matches": exact,
            "test_items_with_near_duplicate": len(near), "jaccard_threshold": 0.9, "ngram_words": 8}


def dev_accuracy(path: Path, templates: set[str]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for row in iter_jsonl(path):
        if row["template_id"] in templates:
            counts[row["template_id"]] += 1
            hits[row["template_id"]] += bool(row["correct"]) if isinstance(row["correct"], bool) else str(row["correct"]) == "True"
    return {t: {"n": counts[t], "accuracy": round(hits[t] / counts[t], 4)} for t in sorted(counts)}


def main() -> None:
    verdict = set(json.loads(MANIFEST.read_text(encoding="utf-8"))["verdict_templates"])
    report = {
        "per_template_mr_verdict": per_template(verdict),
        "codiesp_overlap_with_C1": codiesp_overlap(),
        "dev_accuracy_C1": dev_accuracy(C1_DEV, {CODIESP, TOO_GOOD, "mr_indicator_evidence_v1", "pmc_similar_patient_v1",
                                                 "mtsamples_page_category_v1"}),
        "dev_accuracy_final": dev_accuracy(FINAL_DEV, {CODIESP, TOO_GOOD, "mr_indicator_evidence_v1",
                                                       "pmc_similar_patient_v1", "mtsamples_page_category_v1"}),
    }
    write_json(OUT, report)
    print(json.dumps(report["codiesp_overlap_with_C1"]))
    print(json.dumps({k: v for k, v in report["per_template_mr_verdict"].items()})[:1200])
    print(json.dumps(report["dev_accuracy_final"]))


if __name__ == "__main__":
    main()
