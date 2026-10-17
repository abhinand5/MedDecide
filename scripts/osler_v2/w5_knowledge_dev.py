"""W5 step 1 (osler_v2 ADVISORY §3.7, W5): knowledge dev v2, 4,000 items from the official train splits.

1,000 MedQA rows (GBaker/MedQA-USMLE-4-options, train) and 3,000 MedMCQA rows (openlifescienceai/medmcqa, train). A row is
eligible only if its question text appears in no osler_v1 training block (B0-B3, any template) and in no test set (the MedQA
test file and the MedMCQA validation split, which is the v0.2 MedMCQA test set). Sampling is seeded (0) over rows sorted by
a stable hash, so the set does not depend on file order. Every drop is counted by reason.

Writes data/bench/osler_v2/knowledge_dev_v2.jsonl (gitignored; item text) and
outputs/osler_v2/W5/knowledge_dev_v2.json (counts and hashes only).

Usage: source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v2/w5_knowledge_dev.py > outputs/osler_v2/W5/logs/knowledge_dev_v2.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from meddecide.utils.hashing import file_sha256, stable_hash
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
BLOCKS = [REPO / "data" / "train" / "osler_v1" / f"block_{b}.jsonl" for b in ("B0", "B1", "B2", "B3")]
OUT_ROWS = REPO / "data" / "bench" / "osler_v2" / "knowledge_dev_v2.jsonl"
OUT_MANIFEST = REPO / "outputs" / "osler_v2" / "W5" / "knowledge_dev_v2.json"
TEMPLATES = {"medqa_usmle4_v1", "medmcqa_4opt_v1"}
N_MEDQA, N_MEDMCQA = 1000, 3000
SEED = 0
QUESTION = "Which of the following is the best answer?"
LETTERS = "ABCDE"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def block_states() -> tuple[set[str], Counter[str]]:
    states: set[str] = set()
    per_block: Counter[str] = Counter()
    for path in BLOCKS:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row["template_id"] in TEMPLATES:
                    states.add(norm(row["state"]))
                    per_block[path.stem] += 1
    return states, per_block


def read_medqa_train() -> tuple[list[dict[str, Any]], list[str], set[str]]:
    path = Path(hf_hub_download("GBaker/MedQA-USMLE-4-options", "phrases_no_exclude_train.jsonl", repo_type="dataset"))
    test = Path(hf_hub_download("GBaker/MedQA-USMLE-4-options", "phrases_no_exclude_test.jsonl", repo_type="dataset"))
    with test.open(encoding="utf-8") as fh:
        test_states = {norm(json.loads(line)["question"]) for line in fh}
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    return rows, [str(path.name), str(test.name)], test_states


def read_medmcqa_train() -> tuple[list[dict[str, Any]], list[str], set[str]]:
    base = "openlifescienceai/medmcqa"
    train = Path(hf_hub_download(base, "data/train-00000-of-00001.parquet", repo_type="dataset"))
    valid = Path(hf_hub_download(base, "data/validation-00000-of-00001.parquet", repo_type="dataset"))
    valid_states = {norm(r["question"]) for r in pq.read_table(valid).to_pylist()}
    rows = pq.read_table(train).to_pylist()
    return rows, [train.name, valid.name], valid_states


def medqa_row(rec: dict[str, Any]) -> dict[str, Any] | None:
    options = rec.get("options") or {}
    answer_idx = str(rec.get("answer_idx", "")).strip()
    if not isinstance(options, dict) or len(options) < 2 or answer_idx not in options:
        return None
    keys = sorted(options)
    labels = [str(options[k]) for k in keys]
    letters = list(LETTERS[:len(keys)])
    return {"options": [{"key": k, "label": lab} for k, lab in zip(letters, labels, strict=True)],
            "gold": letters[keys.index(answer_idx)], "source": "medqa", "template_id": "medqa_usmle4_v1",
            "state": str(rec["question"]).strip(), "record": stable_hash({"q": str(rec["question"])}, length=20)}


def medmcqa_row(rec: dict[str, Any]) -> dict[str, Any] | None:
    labels = [rec.get("opa"), rec.get("opb"), rec.get("opc"), rec.get("opd")]
    cop = rec.get("cop")
    if cop not in (0, 1, 2, 3) or any(not str(x or "").strip() for x in labels):
        return None
    return {"options": [{"key": k, "label": str(lab).strip()} for k, lab in zip("ABCD", labels, strict=True)],
            "gold": "ABCD"[int(cop)], "source": "medmcqa", "template_id": "medmcqa_4opt_v1",
            "state": str(rec["question"]).strip(), "record": str(rec.get("id"))}


def pick(rows: list[dict[str, Any]], convert, n: int, excluded: set[str], test_states: set[str],
         counts: Counter[str]) -> list[dict[str, Any]]:
    eligible = []
    for rec in rows:
        key = norm(str(rec.get("question", "")))
        if key in excluded:
            counts["in_block"] += 1
            continue
        if key in test_states:
            counts["in_test_set"] += 1
            continue
        item = convert(rec)
        if item is None:
            counts["unparseable"] += 1
            continue
        eligible.append(item)
    eligible.sort(key=lambda it: stable_hash({"item": it["record"], "template": it["template_id"]}, length=16))
    rng = random.Random(SEED)
    if len(eligible) < n:
        raise SystemExit(f"only {len(eligible)} eligible rows for {n} items")
    return rng.sample(eligible, n)


def main() -> None:
    states, per_block = block_states()
    print("block rows with knowledge templates:", dict(per_block), "distinct states:", len(states), flush=True)
    medqa_rows, medqa_files, medqa_test = read_medqa_train()
    medmcqa_rows, medmcqa_files, medmcqa_test = read_medmcqa_train()
    counts_q: Counter[str] = Counter()
    counts_m: Counter[str] = Counter()
    chosen_q = pick(medqa_rows, medqa_row, N_MEDQA, states, medqa_test, counts_q)
    chosen_m = pick(medmcqa_rows, medmcqa_row, N_MEDMCQA, states, medmcqa_test, counts_m)
    rows = []
    for item in chosen_q + chosen_m:
        rows.append({"item_id": stable_hash({"kdev2": item["record"], "t": item["template_id"]}, length=16),
                     "benchmark": "mix_v2", "set_name": "knowledge_dev_v2", "template_id": item["template_id"],
                     "qtype": "choice", "state": item["state"], "question": QUESTION, "options": item["options"],
                     "gold": item["gold"], "split": "dev", "held_out": False, "source": item["source"]})
    # the leakage guarantee, checked on the output itself
    leaked = sum(norm(r["state"]) in states for r in rows)
    in_test = sum(norm(r["state"]) in (medqa_test | medmcqa_test) for r in rows)
    OUT_ROWS.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_ROWS, rows)
    report = {
        "written_utc": utcnow(), "git_commit": git_commit(REPO), "seed": SEED,
        "rule": "official train splits; excluded: any osler_v1 block row with a knowledge template, and any test or validation question",
        "sources": {"medqa": {"files": medqa_files, "train_rows": len(medqa_rows), "drops": dict(counts_q),
                              "kept": N_MEDQA},
                    "medmcqa": {"files": medmcqa_files, "train_rows": len(medmcqa_rows), "drops": dict(counts_m),
                                "kept": N_MEDMCQA}},
        "block_rows_with_knowledge_templates": dict(per_block),
        "items": len(rows), "items_matching_block_states": leaked, "items_matching_test_states": in_test,
        "test_state_sets": {"medqa_test_questions": len(medqa_test),
                            "medmcqa_validation_questions": len(medmcqa_test)},
        "rows_sha256": file_sha256(OUT_ROWS),
    }
    write_json(OUT_MANIFEST, report)
    print(json.dumps({k: report[k] for k in ("items", "items_matching_block_states")}), flush=True)


if __name__ == "__main__":
    main()
