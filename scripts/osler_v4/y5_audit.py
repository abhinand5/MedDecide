"""Y5 self-audit (osler_v4 R6): the headline numbers of the 27B soft labels, re-derived in a fresh process from the anchor records,
the pool, and the joined soft-label file. Counts and aggregates only: no item text or identifiers are printed or committed.

Checks: the scored records are the stable-order prefix of the pool (pool positions 0..n-1); every distribution is finite,
non-negative and sums to 1; the saturation of the distributions (mean maximum probability, share above 0.99) per qtype; the
argmax key against the gold key and the distribution of answers against the gold distribution per qtype (a constant-answer
check on the distillation targets, diagnostic); the joined file's rows and option counts; and the overlap of the joined rows
with every benchmark row (text or source record), which must be zero. Writes outputs/osler_v4/Y5/audit.json.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.distill import soft_labels as sl
from meddecide.utils.io import iter_jsonl, write_json

POOL = Path("data/train/osler_v4/y5_anchor_pool.jsonl")
ANCHOR = Path("outputs/osler_v4/Y5/anchor_probs.jsonl")
SOFT = Path("data/train/osler_v4/y5_anchor_soft.jsonl")
BENCH = Path("data/bench")
OUT = Path("outputs/osler_v4/Y5/audit.json")
SATURATED = 0.99


def anchor_report() -> dict[str, Any]:
    indices: list[int] = []
    statuses: Counter[str] = Counter()
    bad = 0
    saturated: Counter[str] = Counter()
    maxima: dict[str, list[float]] = defaultdict(list)
    for rec in iter_jsonl(ANCHOR):
        statuses[str(rec["status"])] += 1
        indices.append(int(rec["pool_index"]))
        if rec["status"] != "scored":
            continue
        probs = rec["probs"]
        if any(not math.isfinite(p) or p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-5:
            bad += 1
        top = max(probs)
        maxima[str(rec["qtype"])].append(top)
        saturated[str(rec["qtype"])] += int(top > SATURATED)
    prefix = sorted(indices) == list(range(len(indices)))
    return {
        "records": len(indices), "statuses": dict(statuses), "scored_positions_are_prefix_0_to_n_minus_1": prefix,
        "invalid_distributions": bad,
        "per_qtype": {q: {"scored": len(v), "mean_max_prob": round(sum(v) / len(v), 4),
                          "share_max_above_0_99": round(saturated[q] / len(v), 4)} for q, v in sorted(maxima.items())},
    }


def answer_report() -> dict[str, Any]:
    """The joined rows: argmax against gold and the answer distribution against the gold distribution (diagnostic)."""
    rows = 0
    options_match = 0
    dist_bad = 0
    correct: Counter[str] = Counter()
    total: Counter[str] = Counter()
    predicted: dict[str, Counter[str]] = defaultdict(Counter)
    gold: dict[str, Counter[str]] = defaultdict(Counter)
    for row in iter_jsonl(SOFT):
        rows += 1
        probs = row["teacher_probs"]
        if len(probs) == len(row["options"]):
            options_match += 1
        if abs(sum(probs) - 1.0) > 1e-5:
            dist_bad += 1
        qtype = str(row["qtype"])
        total[qtype] += 1
        correct[qtype] += int(sl.argmax_correct(row, probs))
        predicted[qtype][str(row["options"][sl.argmax_index(probs)]["key"])] += 1
        gold[qtype][str(row["gold"])] += 1
    return {
        "soft_rows": rows, "option_count_matches": options_match, "distribution_sum_violations": dist_bad,
        "per_qtype_argmax_accuracy_on_train_gold": {q: {"items": total[q], "accuracy": round(correct[q] / total[q], 4)}
                                                    for q in sorted(total)},
        "per_qtype_top_predicted_keys": {q: predicted[q].most_common(4) for q in sorted(predicted)},
        "per_qtype_top_gold_keys": {q: gold[q].most_common(4) for q in sorted(gold)},
    }


def overlap_report() -> dict[str, Any]:
    text_keys: set[str] = set()
    record_keys: set[sl.RecordKey] = set()
    for path in sorted(BENCH.rglob("*.jsonl")):
        for row in iter_jsonl(path):
            text, record = sl.overlap_keys(row)
            text_keys.add(text)
            if record is not None:
                record_keys.add(record)
    overlapping = sum(1 for row in iter_jsonl(SOFT) if sl.overlap_reason(row, text_keys, record_keys) is not None)
    return {"benchmark_text_keys": len(text_keys), "benchmark_record_keys": len(record_keys),
            "soft_rows_overlapping_a_benchmark_row": overlapping}


def main() -> None:
    report = {"anchor": anchor_report(), "answers": answer_report(), "overlap": overlap_report()}
    write_json(OUT, report)
    print(report["anchor"]["statuses"], report["anchor"]["scored_positions_are_prefix_0_to_n_minus_1"],
          report["overlap"]["soft_rows_overlapping_a_benchmark_row"])


if __name__ == "__main__":
    main()
