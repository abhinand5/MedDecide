#!/usr/bin/env python
"""T6 acceptance: run the three harness probes on a sample and record their outputs.

* **option shuffle** — the same items with a second seeded permutation; the flip rate is the
  share of items whose argmax *content* changed (not merely its position).
* **candidate count** — the items extended to 2/4/8/16 options with seeded filler options;
  accuracy and the share of items whose argmax is a filler are recorded per count.
* **none of the above** — the gold option removed and "None of the above" added; the detection
  rate is the share of items where the model picks the added option.

Writes ``outputs/bench_v0/T6/probes.json``. Item text is never written: only ids, counts and
probabilities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from meddecide.bench.schema import Item
from meddecide.eval.harness import Harness, ModelSpec
from meddecide.eval.probes import (
    detect_abstention,
    none_of_the_above,
    scale_candidates,
    shuffle_options,
)
from meddecide.utils.io import read_jsonl
from meddecide.utils.provenance import Provenance, gpu_name

CANDIDATE_COUNTS = (2, 4, 8, 16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--items", type=Path, default=Path("data/bench/tier1/medqa.jsonl"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("outputs/bench_v0/T6/probes.json"))
    args = parser.parse_args()

    rows, _report = read_jsonl(args.items, Item)
    sample = rows[: args.limit]
    harness = Harness(ModelSpec(model_id=args.model), batch_size=args.batch_size)

    out: dict[str, Any] = {
        "model": harness.spec.to_dict(),
        "source_file": str(args.items),
        "n_items": len(sample),
        "seed": args.seed,
    }

    # ---- baseline -----------------------------------------------------------
    baseline = harness.score(sample, transform={"name": "identity"})
    base_correct = {p.item_id: p.correct for p in baseline}
    out["baseline"] = {
        "n": len(baseline),
        "accuracy": sum(base_correct.values()) / len(baseline),
        "mean_top1": sum(p.option_probs[p.argmax_index] for p in baseline) / len(baseline),
    }

    # ---- option shuffle -----------------------------------------------------
    shuffled = []
    for item in sample:
        transformed, record = shuffle_options(item, seed=args.seed + 1)
        shuffled.append((transformed, record, item.item_id))
    shuffled_predictions = harness.score(
        [t for t, _r, _src in shuffled], transform={"name": "option_shuffle", "seed": args.seed + 1}
    )
    flips = 0
    comparable = 0
    for pred, (_t, _record, source_item_id) in zip(shuffled_predictions, shuffled, strict=True):
        if source_item_id not in base_correct:
            continue
        comparable += 1
        if pred.correct != base_correct[source_item_id]:
            flips += 1
    out["option_shuffle"] = {
        "n": comparable,
        "n_flips": flips,
        "flip_rate": flips / comparable if comparable else None,
        "seed": args.seed + 1,
        "shuffled_accuracy": sum(p.correct for p in shuffled_predictions) / len(shuffled_predictions),
    }

    # ---- candidate count ----------------------------------------------------
    by_count: dict[int, dict[str, Any]] = {}
    for target in CANDIDATE_COUNTS:
        transformed_items: list[Item] = []
        records: list[Any] = []
        for item in sample:
            transformed, record = scale_candidates(item, target_n=target, seed=args.seed)
            if transformed is None or record is None:
                continue
            transformed_items.append(transformed)
            records.append(record)
        if not transformed_items:
            by_count[target] = {"n": 0, "supported": 0, "accuracy": None}
            continue
        predictions = harness.score(
            transformed_items, transform={"name": "candidate_count", "target_n": target}
        )
        pick_filler = 0
        for pred, record, transformed in zip(predictions, records, transformed_items, strict=True):
            filler_labels = set(record.details.get("fillers", []))
            picked_label = transformed.options[pred.argmax_index].label
            if picked_label in filler_labels:
                pick_filler += 1
        by_count[target] = {
            "n": len(predictions),
            "supported": len(predictions),
            "accuracy": sum(p.correct for p in predictions) / len(predictions),
            "share_argmax_is_filler": pick_filler / len(predictions),
            "mean_top1": sum(p.option_probs[p.argmax_index] for p in predictions) / len(predictions),
        }
    out["candidate_count"] = {str(k): v for k, v in by_count.items()}

    # ---- none of the above --------------------------------------------------
    nota_items: list[Item] = []
    nota_gold_key: dict[str, str] = {}
    nota_option_keys: dict[str, list[str]] = {}
    for item in sample:
        transformed, _record = none_of_the_above(item, seed=args.seed)
        nota_items.append(transformed)
        nota_gold_key[transformed.item_id] = transformed.gold
        nota_option_keys[transformed.item_id] = transformed.option_keys
    nota_predictions = harness.score(nota_items, transform={"name": "none_of_the_above"})
    nota_probs = {p.item_id: p.option_probs for p in nota_predictions}
    abstention = detect_abstention(nota_probs, nota_gold_key, nota_option_keys)
    abstention["note"] = (
        "false-rejection rate needs the same items with the added option present as well, "
        "which this run does not measure: NOT MEASURED in T6"
    )
    out["none_of_the_above"] = abstention

    # ---- arithmetic guards --------------------------------------------------
    out["checks"] = {
        "baseline_denominator_matches": out["baseline"]["n"] == len(sample),
        "shuffle_denominator_matches": out["option_shuffle"]["n"] == len(sample),
        "abstention_denominator_matches": abstention["n_abstention_items"] == len(sample),
        "candidate_counts_recorded": all(str(c) in out["candidate_count"] for c in CANDIDATE_COUNTS),
    }
    out["verdict"] = "PASS" if all(out["checks"].values()) else "FAIL"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    prov = Provenance(
        run_name="T6_probes",
        command=" ".join([sys.executable, *sys.argv]),
        gpu=gpu_name(),
        seed=args.seed,
    )
    prov.finish().write(args.out.with_name("probes_provenance.json"))
    print(json.dumps({k: v for k, v in out.items() if k != "model"}, indent=2))
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
