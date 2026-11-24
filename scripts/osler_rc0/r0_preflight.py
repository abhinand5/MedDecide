#!/usr/bin/env python
"""osler_rc0 R0 — preflight: SHA-256 hashes, pplx snapshot equivalence, medconceptsqa overlap, prediction inventory.

CPU only; writes aggregate-only JSON under ``outputs/osler_rc0/R0/`` (gitignored). No raw item text is written.
Run one step at a time::

    source scripts/pod_env.sh && uv run --frozen python scripts/osler_rc0/r0_preflight.py --step hashes
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_rc0/r0_preflight.py --step pplx
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_rc0/r0_preflight.py --step overlap
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_rc0/r0_preflight.py --step inventory
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from array import array
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs" / "osler_rc0" / "R0"
TOKEN = re.compile(r"[a-z0-9]+")

SETS = ["panel", "knowledge", "confirm_general", "mr_verdict", "mimic_suite",
        "mednli_test", "mimic_heldout", "mr_heldout_test", "pairs_test"]

HASH_FILES = {
    "panel": "data/bench/osler_v4/y9/panel.jsonl",
    "knowledge": "data/bench/osler_v4/y9/knowledge.jsonl",
    "confirm_general": "data/bench/osler_v7/confirm/general.jsonl",
    "mr_verdict": "data/bench/osler_v4/y9/mr_verdict.jsonl",
    "mimic_suite": "data/bench/osler_v4/y9/mimic_suite.jsonl",
    "mednli_test": "data/bench/osler_v4/y9/mednli_test.jsonl",
    "mimic_heldout": "data/bench/osler_v4/y9/mimic_heldout.jsonl",
    "mr_heldout_test": "data/bench/osler_v4/y9/mr_heldout_test.jsonl",
    "pairs_test": "data/bench/osler_v4/y9/pairs_test.jsonl",
    "dev_format_matched": "data/bench/osler_v7/dev_format_matched.jsonl",
    "mix_v7": "data/train/osler_v7/mix_v7.jsonl",
    "4b_prepared": "outputs/osler_v7/4B_FULL/prepared.npz",
    "4b_plan": "outputs/osler_v7/4B_FULL/plan.npz",
    "9b_final_checkpoint": "outputs/osler_v7/T11/final/checkpoints/step_063304.pt",
}

# model -> (id scheme, {set: relative or absolute path}), verified against disk 2026-11-24 in ADVISORY §2
Y9 = "outputs/osler_v4/Y9/baselines"
Z10_QS9 = "outputs/osler_v5/Z10/zeroshot/qwen35_9b"
Z10_QS4 = "outputs/osler_v5/Z10/zeroshot/qwen35_4b"
CLEF = "outputs/osler_v5/Z10/clef_flash"
D1 = "/workspace/MedDecide-d1-2b/outputs/d1_bench"

INVENTORY: dict[str, tuple[str, dict[str, str]]] = {
    "Osler-9B-rc0": ("recomp", {
        "panel": "outputs/eval0/panel/9b_final_panel/predictions.jsonl",
        "knowledge": "outputs/osler_v7/T10/9b_final_knowledge/predictions.jsonl",
        "confirm_general": "outputs/osler_v7/T10/9b_final_general/predictions.jsonl",
        "mr_verdict": "outputs/osler_v7/T10/9b_final_mr_verdict/predictions.jsonl",
        "mimic_suite": "outputs/osler_v7/T10/9b_final_mimic_suite/predictions.jsonl",
    }),
    "d3-flash": ("recomp", {
        "panel": "outputs/eval0/panel/9b_base_panel/predictions.jsonl",
        "knowledge": "outputs/osler_v7/T10/9b_base_knowledge/predictions.jsonl",
        "confirm_general": "outputs/osler_v7/T10/9b_base_general/predictions.jsonl",
        "mr_verdict": "outputs/osler_v7/T10/9b_base_mr_verdict/predictions.jsonl",
        "mimic_suite": "outputs/osler_v7/T10/9b_base_mimic_suite/predictions.jsonl",
    }),
    "pplx-flash": ("recomp", {
        **{s: f"{Y9}/pplx_{s}/predictions.jsonl" for s in
           ["panel", "knowledge", "mr_verdict", "mednli_test", "mr_heldout_test", "pairs_test"]},
        "mimic_suite": "outputs/osler_v7/T10/baselines/4b_base_mimic_suite/predictions.jsonl",
        "confirm_general": "outputs/osler_v7/T10/baselines/4b_base_general/predictions.jsonl",
    }),
    "MedDecider-9B": ("recomp", {
        **{s: f"{Y9}/md9b_{s}/predictions.jsonl" for s in
           ["panel", "knowledge", "mr_verdict", "mednli_test", "mimic_heldout", "mr_heldout_test", "pairs_test"]},
        "mimic_suite": "outputs/osler_v4/Y4/baselines/suite_meddecider9b/predictions.jsonl",
    }),
    "MedDecider-4B": ("recomp", {
        **{s: f"{Y9}/md4b_{s}/predictions.jsonl" for s in
           ["panel", "knowledge", "mr_verdict", "mednli_test", "mr_heldout_test", "pairs_test"]},
        "mimic_suite": "outputs/osler_v4/Y4/baselines/suite_meddecider4b/predictions.jsonl",
        "mimic_heldout": "outputs/osler_v4/Y4/baselines/heldout_meddecider4b/predictions.jsonl",
    }),
    "Clef-Flash": ("raw", {s: f"{CLEF}/{s}.jsonl" for s in
                           ["panel", "knowledge", "mr_verdict", "mimic_suite", "mednli_test",
                            "mimic_heldout", "mr_heldout_test"]}),
    "JEV-9B": ("raw", {s: f"{Y9}/jev9b_{s}/jev-9b/predictions.jsonl" for s in SETS}),
    "d1-2B": ("raw", {s: f"{D1}/{s}/predictions.jsonl" for s in
                      ["panel", "knowledge", "mr_verdict", "mimic_suite", "mednli_test"]}),
    "Qwen3.5-9B": ("raw", {s: f"{Z10_QS9}/{s}.jsonl" for s in
                           ["panel", "knowledge", "mr_verdict", "mimic_suite", "mednli_test",
                            "mimic_heldout", "mr_heldout_test"]}),
    "Qwen3.5-4B": ("raw", {s: f"{Z10_QS4}/{s}.jsonl" for s in
                           ["panel", "knowledge", "mr_verdict", "mimic_suite", "mednli_test",
                            "mimic_heldout", "mr_heldout_test"]}),
    "pplx-27B": ("recomp", {s: f"{Y9}/pplx27b_{s}/predictions.jsonl" for s in SETS}),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(name: str, payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


def step_hashes() -> None:
    out = {"root": str(ROOT), "files": {}}
    for key, rel in HASH_FILES.items():
        path = ROOT / rel
        if not path.exists():
            out["files"][key] = {"path": rel, "exists": False}
            continue
        out["files"][key] = {"path": rel, "exists": True, "bytes": path.stat().st_size,
                             "sha256": sha256_file(path)}
        print(f"hashed {key} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
    write_json("hashes.json", out)


def step_pplx() -> None:
    base = Path("/workspace/.hf_home/hub/models--perplexity-ai--pplx-decider-flash-v1.2/snapshots")
    snaps = ["6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d", "9ff7cf9838e59df461c276b7699d56b244d2af58"]
    files: dict[str, dict[str, str]] = {}
    for snap in snaps:
        d = base / snap
        files[snap] = {str(p.relative_to(d)): sha256_file(p) for p in sorted(d.rglob("*"))
                       if p.is_file() and "__pycache__" not in p.parts and p.name != "README.md"}
    a, b = (files[s] for s in snaps)
    differing = sorted(set(a) ^ set(b) | {k for k in set(a) & set(b) if a[k] != b[k]})
    out = {"snapshots": snaps, "file_counts": {s: len(files[s]) for s in snaps},
           "identical": not differing, "differing_files": differing,
           "files": {s: dict(sorted(files[s].items())) for s in snaps}}
    write_json("pplx_snapshot_check.json", out)


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


def ngram_hashes(text: str, n: int = 13) -> list[int]:
    toks = TOKEN.findall(text.lower())
    return [hash(tuple(toks[i:i + n])) & 0xFFFFFFFFFFFFFFFF for i in range(len(toks) - n + 1)]


def step_overlap() -> None:
    """medconceptsqa panel rows vs every mix_v7 row: exact normalised hash, exact question, 13-gram containment."""
    import numpy as np

    panel = [r for r in read_jsonl(ROOT / HASH_FILES["panel"]) if r.get("source") == "medconceptsqa_all"]
    mix_path = ROOT / HASH_FILES["mix_v7"]
    mix_q: set[str] = set()
    mix_sq: set[str] = set()
    hashes = array("Q")
    rows = 0
    with mix_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            q = normalized(str(row.get("question", "")))
            sq = normalized(f"{row.get('state', '')} {row.get('question', '')}")
            mix_q.add(q)
            mix_sq.add(sq)
            hashes.extend(ngram_hashes(f"{row.get('state', '')} {row.get('question', '')}"))
            rows += 1
            if rows % 200_000 == 0:
                print(f"  mix rows {rows:,} ngrams {len(hashes):,}", flush=True)
    arr = np.frombuffer(hashes, dtype=np.uint64) if len(hashes) else np.zeros(0, dtype=np.uint64)
    del hashes
    uniq = np.unique(arr)
    del arr
    print(f"mix: {rows:,} rows, {len(uniq):,} distinct 13-grams", flush=True)

    per_row = []
    exact_sq = 0
    exact_q = 0
    any_hit = 0
    all_hit = 0
    for row in panel:
        q = normalized(str(row.get("question", "")))
        sq = normalized(f"{row.get('state', '')} {row.get('question', '')}")
        qh = ngram_hashes(sq)
        if qh:
            query = np.asarray(qh, dtype=np.uint64)
            positions = np.searchsorted(uniq, query)
            positions = np.minimum(positions, len(uniq) - 1)
            hits = int((uniq[positions] == query).sum())
        else:
            hits = 0
        exact_sq += int(sq in mix_sq)
        exact_q += int(q in mix_q)
        any_hit += int(hits > 0)
        all_hit += int(bool(qh) and hits == len(qh))
        per_row.append({"hits": hits, "total": len(qh), "containment": (hits / len(qh)) if qh else 0.0})
    containments = sorted(r["containment"] for r in per_row)
    n = len(per_row)
    out = {
        "panel_rows_medconceptsqa": n,
        "mix_rows": rows,
        "mix_distinct_13grams_state_question": len(uniq),
        "definition": ("13-grams over tokenised (lowercased alphanumeric) state+question only; option labels excluded "
                       "to avoid common-label false positives. containment = matched 13-grams / item 13-grams; "
                       "all_13grams_present is the token-substring-level near-duplicate criterion."),
        "exact_normalized_state_question_hash_matches": exact_sq,
        "exact_normalized_question_text_matches": exact_q,
        "items_with_any_13gram_hit": any_hit,
        "items_with_all_13grams_present": all_hit,
        "containment_mean": round(sum(containments) / n, 6) if n else 0.0,
        "containment_p95": round(containments[int(0.95 * (n - 1))], 6) if n else 0.0,
        "containment_max": round(containments[-1], 6) if n else 0.0,
        "items_containment_ge_0_5": sum(1 for v in containments if v >= 0.5),
    }
    write_json("medconceptsqa_overlap.json", out)


def step_overlap_context() -> None:
    """Interpretation context for the exact-question matches: the medconceptsqa ``question`` field is one generic string."""
    panel = [r for r in read_jsonl(ROOT / HASH_FILES["panel"]) if r.get("source") == "medconceptsqa_all"]
    questions = {normalized(str(r.get("question", ""))) for r in panel}
    states = {normalized(str(r.get("state", ""))) for r in panel}
    out = {
        "panel_rows_medconceptsqa": len(panel),
        "distinct_normalized_question_strings": len(questions),
        "distinct_normalized_state_strings": len(states),
        "question_strings": sorted(questions) if len(questions) <= 5 else "more than 5",
        "interpretation": ("the panel's ``question`` field is a generic multiple-choice prompt shared by every "
                           "medconceptsqa item, so an exact question-text match in the mix is not item-level overlap; "
                           "the state+question hash and the 13-gram containment are the item-level checks"),
        "state_question_exact_matches": json.loads((OUT / "medconceptsqa_overlap.json").read_text())[
            "exact_normalized_state_question_hash_matches"],
        "state_question_13gram_items_with_any_hit": json.loads((OUT / "medconceptsqa_overlap.json").read_text())[
            "items_with_any_13gram_hit"],
    }
    write_json("medconceptsqa_overlap_context.json", out)


def _detect_ids(pred_ids: list[str], raw: set[str], recomp: set[str]) -> tuple[str, int, int]:
    raw_n = sum(1 for i in pred_ids if i in raw)
    recomp_n = sum(1 for i in pred_ids if i in recomp)
    scheme = "raw" if raw_n >= recomp_n else "recomp"
    return scheme, raw_n, recomp_n


def step_inventory() -> None:
    frozen = {s: read_jsonl(ROOT / HASH_FILES[s]) for s in SETS}
    raw = {s: {str(r["item_id"]) for r in rows} for s, rows in frozen.items()}
    from meddecide.train.mix_items import mix_row_to_item
    recomp = {s: {mix_row_to_item(r).item_id for r in rows} for s, rows in frozen.items()}

    out: dict[str, Any] = {"sets": SETS, "set_sizes": {s: len(frozen[s]) for s in SETS}, "models": {}}
    for model, (declared, paths) in INVENTORY.items():
        entry: dict[str, Any] = {"declared_scheme": declared, "sets": {}}
        for s in SETS:
            rel = paths.get(s)
            if rel is None:
                entry["sets"][s] = {"status": "MISSING"}
                continue
            path = Path(rel) if Path(rel).is_absolute() else ROOT / rel
            if not path.exists():
                entry["sets"][s] = {"status": "MISSING", "path": rel, "exists": False}
                continue
            rows = read_jsonl(path)
            ids = [str(r.get("item_id", "")) for r in rows]
            scheme, raw_n, recomp_n = _detect_ids(ids, raw[s], recomp[s])
            scored = sum(1 for r in rows if r.get("status", "scored") == "scored")
            with_probs = sum(1 for r in rows if isinstance(r.get("probs"), list))
            entry["sets"][s] = {
                "status": "PRESENT", "path": rel, "rows": len(rows),
                "unique_ids": len(set(ids)), "duplicate_ids": len(ids) - len(set(ids)),
                "scored": scored, "skipped": len(rows) - scored,
                "probs_rows": with_probs, "has_probs": with_probs > 0,
                "detected_scheme": scheme, "raw_id_matches": raw_n, "recomp_id_matches": recomp_n,
                "coverage": round(scored / len(frozen[s]), 6),
            }
        out["models"][model] = entry
    write_json("inventory.json", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True,
                        choices=["hashes", "pplx", "overlap", "overlap-context", "inventory"])
    args = parser.parse_args()
    {"hashes": step_hashes, "pplx": step_pplx, "overlap": step_overlap,
     "overlap-context": step_overlap_context, "inventory": step_inventory}[args.step]()


if __name__ == "__main__":
    main()
