#!/usr/bin/env python
"""F2: readout-health check for `noul` and `score` items on real models.

Runs the F2 readout (every question type rendered with letter labels and read from option-letter
tokens) over a seeded sample of `noul` and `score` items, and reports, per (model, qtype):

* median label mass over the sampled items;
* greedy agreement — on up to ``--greedy-items`` of them, the readout argmax option equals the
  option the model generates greedily from the same prompt;
* accuracy with its bootstrap CI, for context.

Acceptance (ADVISORY F2): on 200 `noul` and 200 `score` items for Qwen3.5-0.8B and Qwen3.5-4B,
median label mass >= 0.9 and greedy agreement >= 0.9.

Writes ``outputs/bench_v0_fix0/F2/readout_health.json`` (no item text).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.bench.schema import Item, QuestionType
from meddecide.eval.harness import Harness, ModelSpec, greedy_first_token
from meddecide.eval.health import evaluate_cell
from meddecide.eval.readout import canonicalise_options, render_prompt
from meddecide.utils.io import read_jsonl
from meddecide.utils.provenance import Provenance, gpu_name, utcnow


def _load_items(files: list[Path], qtype: QuestionType, limit: int, seed: int) -> list[Item]:
    rows: list[Item] = []
    for path in files:
        loaded, report = read_jsonl(path, Item)
        report.check_closes()
        rows.extend(i for i in loaded if i.qtype is qtype and str(i.split) == "test")
    rng = random.Random(f"{seed}:{qtype.value}")
    rng.shuffle(rows)
    return rows[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        default=["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"])
    parser.add_argument("--tier1", type=Path, default=Path("data/bench/v0.1/tier1"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--greedy-items", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/bench_v0_fix0/F2/readout_health.json"))
    args = parser.parse_args()

    files = sorted(args.tier1.glob("*.jsonl"))
    sample = {
        "noul": _load_items(files, QuestionType.NOUL, args.limit, args.seed),
        "score": _load_items(files, QuestionType.SCORE, args.limit, args.seed),
    }
    print(json.dumps({k: len(v) for k, v in sample.items()}), flush=True)

    report: dict[str, Any] = {
        "generated_at_utc": utcnow(),
        "seed": args.seed,
        "limit_per_qtype": args.limit,
        "greedy_items": args.greedy_items,
        "sample_sources": {
            k: dict(sorted(Counter(i.source for i in v).items())) for k, v in sample.items()
        },
        "models": {},
    }

    for model_id in args.models:
        spec = ModelSpec(model_id=model_id)
        harness = Harness(spec, batch_size=args.batch_size)
        model_entry: dict[str, Any] = {
            "model": spec.to_dict(),
            "variant_detection": harness.variant_detection,
            "qtypes": {},
        }
        for qtype_name, items in sample.items():
            if not items:
                model_entry["qtypes"][qtype_name] = {"status": "NOT MEASURED — no items"}
                continue
            # canonical items are what the harness renders; do them once here so the greedy
            # sample uses exactly the same prompts as the scored run
            canonical = [canonicalise_options(i)[0] for i in items]
            predictions = harness.score(items)
            prompts = [render_prompt(i, harness.tokenizer, harness.variant) for i in canonical]
            option_ids = [[harness._key_token(k) for k in i.option_keys] for i in canonical]

            greedy_indices = random.Random(f"{args.seed}:greedy").sample(
                range(len(canonical)), min(args.greedy_items, len(canonical))
            )
            greedy_predictions = greedy_first_token(
                harness, [prompts[i] for i in greedy_indices], [option_ids[i] for i in greedy_indices]
            )
            greedy_matches = [
                greedy_predictions[n] == predictions[i].argmax_index
                for n, i in enumerate(greedy_indices)
            ]

            n_options = max(len(p.option_keys) for p in predictions)
            health = evaluate_cell(
                model_id=model_id,
                template_id=f"qtype:{qtype_name}",
                label_masses=[p.label_mass for p in predictions],
                correct=[p.correct for p in predictions],
                greedy_matches=greedy_matches,
                n_options=n_options,
                seed=args.seed,
            )
            entry = health.to_dict()
            entry["n_with_options"] = n_options
            entry["greedy_not_an_option"] = sum(1 for g in greedy_predictions if g < 0)
            entry["accuracy_by_source"] = {}
            by_source: dict[str, list[Any]] = defaultdict(list)
            for pred in predictions:
                by_source[pred.source].append(pred)
            for source, preds in sorted(by_source.items()):
                entry["accuracy_by_source"][source] = {
                    "n": len(preds),
                    "accuracy": sum(1 for p in preds if p.correct) / len(preds),
                    "median_label_mass": float(np.median([p.label_mass for p in preds])),
                }
            model_entry["qtypes"][qtype_name] = entry
            print(
                f"{model_id} {qtype_name}: mass={health.median_label_mass:.4f} "
                f"greedy={health.greedy_agreement} acc={health.accuracy:.4f} "
                f"status={health.status}",
                flush=True,
            )
        report["models"][model_id] = model_entry

    # Acceptance is computed over the models this invocation actually ran; a model that was not
    # run is recorded as NOT MEASURED, not as a failure (R3). The F2 acceptance pair
    # (Qwen3.5-0.8B and Qwen3.5-4B) is verified by the SELF_AUDIT from the two per-model runs.
    checks: dict[str, Any] = {}
    for model_id, entry in sorted(report["models"].items()):
        for qtype_name in ("noul", "score"):
            q = entry["qtypes"].get(qtype_name, {})
            checks[f"{model_id}|{qtype_name}"] = {
                "median_label_mass": q.get("median_label_mass"),
                "mass_ge_0_9": (q.get("median_label_mass") or 0) >= 0.9,
                "greedy_agreement": q.get("greedy_agreement"),
                "greedy_ge_0_9": (q.get("greedy_agreement") or 0) >= 0.9,
                "gate_status": q.get("status"),
            }
    for model_id in ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"):
        if model_id not in report["models"]:
            checks[f"{model_id}|*"] = "NOT MEASURED — model not run in this invocation"
    report["acceptance_checks"] = checks
    ran = [k for k, v in checks.items() if isinstance(v, dict)]
    report["verdict"] = (
        "PASS"
        if ran and all(v.get("mass_ge_0_9") and v.get("greedy_ge_0_9") for k, v in checks.items()
                       if isinstance(v, dict))
        else ("NOT MEASURED — no model run" if not ran else "FAIL")
    )
    report["acceptance_pair_note"] = (
        "ADVISORY F2 requires >=0.9 mass and >=0.9 greedy agreement on 200 noul and 200 score "
        "items for BOTH Qwen3.5-0.8B and Qwen3.5-4B. This invocation's verdict covers the models "
        "listed in acceptance_checks with a dict value; run the other model to complete the pair."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    prov = Provenance(
        run_name="F2_readout_health",
        command=" ".join([sys.executable, *sys.argv]),
        gpu=gpu_name(),
        seed=args.seed,
        config={"limit": args.limit, "greedy_items": args.greedy_items, "models": args.models},
    )
    prov.finish().write(args.out.with_name("readout_health_provenance.json"))
    print(json.dumps({"verdict": report["verdict"], "checks": checks}, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
