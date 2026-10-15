"""V8 diagnostic (additional; never a gate or keep input): how far word overlap alone separates the yes and no answers of
pmc_similar_patient_v1 on the fast dev items. Threshold-free (an AUC on the 400 dev items), and it fits nothing.

The dev items are the fast dev set; the B3 predictions give the same template's dev accuracy. Writes
outputs/osler_v1/V8/diagnostics/pmc_similar_overlap.json (aggregates only).

Usage (CPU; no GPU):
    source scripts/pod_env.sh && uv run --frozen python -I scripts/osler_v1/v8_pmc_overlap.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
FAST_DEV = REPO / "data" / "bench" / "osler_mr_v0" / "fastdev_mix.jsonl"
PREDICTIONS = REPO / "outputs" / "osler_v1" / "V8" / "B3" / "dev_predictions.jsonl"
OUT = REPO / "outputs" / "osler_v1" / "V8" / "diagnostics" / "pmc_similar_overlap.json"
TEMPLATE = "pmc_similar_patient_v1"


def patient_texts(state: str) -> dict[str, str]:
    """The two vignettes of a state ("Patient A:" and "Patient B:" sections), keyed by letter."""
    pieces = re.split(r"Patient ([AB]):", state)
    return {pieces[k]: pieces[k + 1] for k in range(1, len(pieces) - 1, 2)}


def word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def jaccard(a: str, b: str) -> float:
    """Word-set overlap of two texts (intersection over union); 0.0 when both are empty."""
    wa, wb = word_set(a), word_set(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def auc(positives: list[float], negatives: list[float]) -> float:
    """The probability that a random positive scores above a random negative (ties count one half)."""
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def main() -> None:
    items = []
    with FAST_DEV.open(encoding="utf-8") as fh:
        for line in fh:
            if f'"{TEMPLATE}"' in line:
                items.append(json.loads(line))
    if not items:
        raise SystemExit(f"no fast dev items for {TEMPLATE}")
    scores: dict[str, list[float]] = {"yes": [], "no": []}
    unparsed = 0
    for item in items:
        texts = patient_texts(item["state"])
        if set(texts) != {"A", "B"}:
            unparsed += 1
            continue
        scores[item["gold"]].append(jaccard(texts["A"], texts["B"]))
    with PREDICTIONS.open(encoding="utf-8") as fh:
        correct = [json.loads(line) for line in fh if f'"{TEMPLATE}"' in line]
    record = {
        "kind": "osler_v1_V8_pmc_similar_overlap", "written_utc": utcnow(), "git_commit": git_commit(REPO),
        "template": TEMPLATE, "dev_items": len(items), "unparsed": unparsed,
        "gold_counts": {k: len(v) for k, v in scores.items()},
        "b3_dev_items": len(correct), "b3_dev_correct": sum(bool(r["correct"]) for r in correct),
        "jaccard_mean": {k: round(sum(v) / len(v), 4) for k, v in scores.items() if v},
        "jaccard_auc_for_yes": round(auc(scores["yes"], scores["no"]), 4),
    }
    write_json(OUT, record)
    print(f"{TEMPLATE}: {record['dev_items']} dev items, AUC for yes {record['jaccard_auc_for_yes']}; wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
