#!/usr/bin/env python
"""F6: ladder baselines on v0.1 (tier-1 test + fresh test, kept templates) with the D12 health gate.

One model per invocation so a long run is resumable: predictions stream to
``outputs/bench_v0_fix0/F6/preds_<model>.jsonl`` and a per-model summary is written to
``outputs/bench_v0_fix0/F6/model_<model>.json``. `scripts/bench/report_baselines_v0_1.py` then
aggregates every model file into ``results.json`` and the committed `baselines_v0_1.md`.

What this fixes relative to bench_v0's T9:

* **the F2 readout** (`noul`/`score` items go through the validated letter path);
* **`coverage` means what the F6 spec says** — the share of *this template's* test items the model
  could take, not the share of the whole source (v0 divided by the source's item count, so a
  template with 94 openFDA items reported coverage 0.514);
* **class-balanced templates**, so the majority baseline beside each accuracy is meaningful;
* **the health gate** (D12): every cell carries median label mass, greedy agreement on a seeded
  sample, and an accuracy CI; a cell that fails is reported as `READOUT_FAIL — <check>`;
* **the option-shuffle flip rate** — every model, up to `--shuffle-items` items per template;
* **the strict slice** — fresh results are reported whole and restricted to items dated on/after
  `2026-09-10`.

Predictions and item text stay in the gitignored `outputs/`; only aggregates are committed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.bench.schema import Item, QuestionType
from meddecide.eval.harness import Harness, ModelSpec, Prediction, greedy_first_token
from meddecide.eval.health import evaluate_cell
from meddecide.eval.metrics import bootstrap_ci, ece, macro_accuracy
from meddecide.eval.predlog import read_prediction_log
from meddecide.eval.probes import shuffle_options
from meddecide.eval.readout import canonicalise_options, render_prompt
from meddecide.utils.io import read_jsonl
from meddecide.utils.provenance import Provenance, gpu_name, utcnow

LADDER = [
    "LiquidAI/LFM2.5-350M",
    "Qwen/Qwen3.5-0.8B-Base",
    "Qwen/Qwen3.5-0.8B",
    "google/medgemma-1.5-4b-it",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
]


def slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace(".", "p")


# Batch size per model. The 4B/9B models OOM on the widest `choice` templates when sharing the
# GPU with another job, and F7 hit the same wall; a smaller batch is the cheap insurance, and the
# batch size is recorded in each model's summary so a reader can see which run used what.
BATCH_BY_MODEL = {
    "Qwen/Qwen3.5-9B": 2,
    "Qwen/Qwen3.5-4B": 3,
    "google/medgemma-1.5-4b-it": 4,
}


def load_benchmark(tier1: Path, fresh: Path, keep: set[str] | None) -> dict[str, list[Item]]:
    """All test items, grouped by tier: {tier: [Item, ...]}, filtered to kept templates."""
    out: dict[str, list[Item]] = {}
    for tier, directory in (("tier1", tier1), ("fresh", fresh)):
        rows: list[Item] = []
        for path in sorted(directory.glob("*.jsonl")):
            loaded, report = read_jsonl(path, Item)
            report.check_closes()
            rows.extend(i for i in loaded if str(i.split) == "test")
        if keep is not None:
            rows = [i for i in rows if i.template_id in keep]
        out[tier] = rows
    return out


def per_template_totals(items: list[Item]) -> dict[str, int]:
    counts: Counter[str] = Counter(i.template_id for i in items)
    return dict(counts)


def summarise_cell(
    *,
    model_id: str,
    tier: str,
    template_id: str,
    predictions: list[Prediction],
    items_by_id: dict[str, Item],
    template_total: int,
    greedy_matches: list[bool] | None,
    shuffle_flip: float | None,
    strict_ids: set[str] | None,
    seed: int,
) -> dict[str, Any]:
    """One (model, tier, template) cell, with the health gate applied."""
    n = len(predictions)
    if n == 0:
        return {
            "model_id": model_id, "tier": tier, "template_id": template_id, "n": 0,
            "template_total": template_total, "coverage": 0.0,
            "status": "NOT MEASURED — model could not take any item of this template",
        }
    qtype = str(predictions[0].qtype)
    n_options = max(len(p.option_keys) for p in predictions)
    chance = 0.5 if qtype == QuestionType.NOUL.value else 1.0 / n_options
    correct = [p.correct for p in predictions]
    accuracy = sum(correct) / n
    _point, lo, hi = bootstrap_ci([1.0 if c else 0.0 for c in correct], n_resamples=1000, seed=seed)
    gold = [p.gold_key for p in predictions]
    majority_label, majority_count = Counter(gold).most_common(1)[0]
    gold_probs = np.asarray(
        [p.option_probs[p.option_keys.index(p.gold_key)] for p in predictions], dtype=np.float64
    )
    calibration_error = ece(
        np.asarray([p.option_probs for p in predictions], dtype=np.float64),
        [p.option_keys.index(p.gold_key) for p in predictions],
    )
    health = evaluate_cell(
        model_id=model_id,
        template_id=f"{tier}:{template_id}",
        label_masses=[p.label_mass for p in predictions],
        correct=correct,
        greedy_matches=greedy_matches,
        n_options=n_options,
        seed=seed,
        # constant-answer check (S1): the readout's argmax key per item, against the
        # template's own majority share
        predicted_labels=[p.option_keys[p.argmax_index] for p in predictions],
        majority_share=majority_count / n,
    )
    cell: dict[str, Any] = {
        "model_id": model_id,
        "tier": tier,
        "template_id": template_id,
        "qtype": qtype,
        "source": predictions[0].source,
        "n": n,
        "template_total": template_total,
        "coverage": n / template_total if template_total else None,
        "n_options": n_options,
        "chance": chance,
        "accuracy": accuracy,
        "accuracy_ci95": [lo, hi],
        "n_correct": sum(correct),
        "majority_label": majority_label,
        "majority_share": majority_count / n,
        "macro_accuracy": macro_accuracy(
            [p.option_keys[p.argmax_index] for p in predictions], gold
        ),
        "mean_gold_prob": float(gold_probs.mean()),
        "ece": calibration_error,
        "median_label_mass": health.median_label_mass,
        "label_mass_min": float(np.min([p.label_mass for p in predictions])),
        "greedy_agreement": health.greedy_agreement,
        "greedy_sample_size": health.greedy_sample_size,
        "modal_option": health.modal_option,
        "modal_share": health.modal_share,
        "shuffle_flip_rate": shuffle_flip,
        "p50_seconds_per_item": float(np.median([p.latency_s for p in predictions])),
        "gate_status": health.status,
        "gate_failures": health.failures,
        "gate_notes": health.notes,
        "variant": predictions[0].variant,
        "prompt_tokens_p50": float(np.median([p.prompt_tokens for p in predictions])),
    }
    if strict_ids is not None:
        strict = [p for p in predictions if p.item_id in strict_ids]
        if strict:
            s_correct = sum(1 for p in strict if p.correct)
            cell["strict_slice"] = {
                "n": len(strict),
                "n_correct": s_correct,
                "accuracy": s_correct / len(strict),
                "median_label_mass": float(np.median([p.label_mass for p in strict])),
            }
        else:
            cell["strict_slice"] = {"n": 0, "status": "NOT MEASURED — no strict-slice items"}
    return cell


def run_model(args: argparse.Namespace) -> int:
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = None
    if args.keep_screen and args.keep_screen.exists():
        screen = json.loads(args.keep_screen.read_text())
        keep = {t["template_id"] for t in screen["templates"] if not t["drop"]}
    benchmark = load_benchmark(args.tier1, args.fresh, keep)
    strict_path = args.tier1.parent / "fresh" / "acceptance.json"
    strict_ids: set[str] | None = None
    if strict_path.exists():
        strict_start = json.loads(strict_path.read_text()).get("strict_slice_start")
        if strict_start:
            strict_ids = set()
            for item in benchmark["fresh"]:
                if str(item.record_date) >= str(strict_start):
                    strict_ids.add(item.item_id)

    spec = ModelSpec(model_id=args.model, revision=args.revision)
    run_id = args.run_id or f"{slug(args.model)}:{utcnow()}"
    batch_size = args.batch_size if args.batch_size != 8 else BATCH_BY_MODEL.get(
        args.model, args.batch_size
    )
    harness = Harness(
        spec, batch_size=batch_size, max_batch_tokens=args.max_batch_tokens, run_id=run_id
    )
    # per-template runs must not overwrite each other: with --only-template the template
    # name is part of the file name (the default, whole-benchmark behaviour is unchanged)
    suffix = f"__{args.only_template}" if args.only_template else ""
    preds_path = out_dir / f"preds_{slug(args.model)}{suffix}.jsonl"
    model_report: dict[str, Any] = {
        "model_id": args.model,
        "run_id": run_id,
        "variant_detection": harness.variant_detection,
        "generated_at_utc": utcnow(),
        "batch_size": batch_size,
        "batch_size_requested": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "gpu": gpu_name(),
        "tiers": {},
        "cells": [],
    }
    started = time.time()

    for tier, items in benchmark.items():
        if not items:
            continue
        totals = per_template_totals(items)
        by_template: dict[str, list[Item]] = defaultdict(list)
        for item in items:
            by_template[item.template_id].append(item)
        tier_cells: list[dict[str, Any]] = []
        for template_id, template_items in sorted(by_template.items()):
            if args.only_template and template_id != args.only_template:
                continue
            t0 = time.time()
            predictions = harness.score(template_items)
            # coverage: items the model could actually take, of this template's items
            if len(predictions) < len(template_items):
                print(f"  WARNING {template_id}: scored {len(predictions)}/{len(template_items)}",
                      flush=True)
            greedy_matches: list[bool] | None = None
            if args.greedy_items > 0 and predictions:
                idx = random.Random(f"{args.seed}:greedy:{model_id(template_id)}").sample(
                    range(len(predictions)), min(args.greedy_items, len(predictions))
                )
                canonical = [canonicalise_options(template_items[i])[0] for i in idx]
                prompts = [render_prompt(c, harness.tokenizer, harness.variant) for c in canonical]
                option_ids = [[harness._key_token(k) for k in c.option_keys] for c in canonical]
                picks = greedy_first_token(harness, prompts, option_ids)
                greedy_matches = [
                    picks[k] == predictions[i].argmax_index for k, i in enumerate(idx)
                ]
            flip = option_shuffle_flip(
                harness, template_items, predictions, args.shuffle_items, args.seed
            )
            cell = summarise_cell(
                model_id=args.model, tier=tier, template_id=template_id,
                predictions=predictions, items_by_id={i.item_id: i for i in template_items},
                template_total=totals.get(template_id, len(template_items)),
                greedy_matches=greedy_matches, shuffle_flip=flip,
                strict_ids=strict_ids if tier == "fresh" else None, seed=args.seed,
            )
            cell["wall_seconds"] = time.time() - t0
            tier_cells.append(cell)
            print(f"{args.model} {tier}/{template_id}: n={cell['n']}/{cell['template_total']} "
                  f"acc={cell['accuracy']:.4f} mass={cell['median_label_mass']:.4f} "
                  f"greedy={cell['greedy_agreement']} flip={cell['shuffle_flip_rate']} "
                  f"[{cell['gate_status']}]", flush=True)
            if args.write_predictions:
                # append: the run is resumable per model, and `Prediction.to_json` is the same
                # serialisation the v0 harness wrote (so the report scripts can read both)
                preds_path.parent.mkdir(parents=True, exist_ok=True)
                with preds_path.open("a", encoding="utf-8") as fh:
                    for p in predictions:
                        fh.write(json.dumps(p.to_json(), ensure_ascii=False) + "\n")
                model_report["preds_path"] = str(preds_path)
        model_report["tiers"][tier] = {
            "n_items": len(items),
            "n_scored": sum(c["n"] for c in tier_cells),
            "n_cells": len(tier_cells),
            "coverage": sum(c["n"] for c in tier_cells) / len(items) if items else None,
            "n_gate_pass": sum(1 for c in tier_cells if c.get("gate_status") == "PASS"),
            "n_gate_fail": sum(
                1 for c in tier_cells if str(c.get("gate_status", "")).startswith("READOUT_FAIL")
            ),
        }
        model_report["cells"].extend(tier_cells)

    model_report["wall_seconds"] = time.time() - started
    model_report["n_prediction_rows"] = sum(c["n"] for c in model_report["cells"])
    if model_report.get("preds_path"):
        # append-only file: say what is in it, not just how many rows it has (S1/P7)
        model_report["prediction_log"] = read_prediction_log(Path(model_report["preds_path"])).report()
    out_path = out_dir / f"model_{slug(args.model)}{suffix}.json"
    out_path.write_text(json.dumps(model_report, indent=2) + "\n")
    prov = Provenance(
        run_name=f"F6_baseline_{slug(args.model)}",
        command=" ".join([sys.executable, *sys.argv]),
        gpu=gpu_name(),
        seed=args.seed,
        config={"model": args.model, "batch_size": args.batch_size,
                "greedy_items": args.greedy_items, "shuffle_items": args.shuffle_items},
    )
    prov.finish().write(out_dir / f"model_{slug(args.model)}{suffix}_provenance.json")
    tiers = model_report["tiers"]
    print(json.dumps(tiers, indent=2))
    print(f"wall: {model_report['wall_seconds']:.1f}s; wrote {out_path}")
    return 0


def model_id(template_id: str) -> str:
    return template_id


def option_shuffle_flip(
    harness: Harness,
    items: list[Item],
    predictions: list[Prediction],
    n_items: int,
    seed: int,
) -> float | None:
    """Share of sampled items whose chosen *original* option changes when options are reordered.

    A model that is reading the options should usually keep choosing the same option content; a
    model that is guessing from position will flip often. Options are shuffled deterministically
    and the chosen content is mapped back through the original keys, so this measures content
    stability rather than key stability.
    """
    if n_items <= 0 or len(items) < 2:
        return None
    rng = random.Random(f"{seed}:shuffle:items")
    idx = rng.sample(range(len(items)), min(n_items, len(items)))
    transformed = []
    perms: list[list[int]] = []
    original_pick_positions: list[int] = []
    for i in idx:
        item = items[i]
        shuffled, record = shuffle_options(item, seed=seed, salt=item.item_id)
        transformed.append(shuffled)
        perms.append([int(x) for x in record.details["perm"]])
        original_pick_positions.append(predictions[i].argmax_index)
    shuffled_preds = harness.score(transformed)
    if len(shuffled_preds) != len(transformed):
        return None
    flips = 0
    for n, pred in enumerate(shuffled_preds):
        # `Prediction` carries keys, not labels. The shuffled item is re-keyed in display order,
        # so display position k holds original option `perm[k]`; mapping the model's chosen
        # display position back through the permutation gives the original option it chose. A
        # flip means the model's choice of *content* changed when the order did.
        pick_position = pred.argmax_index
        original_position = perms[n][pick_position]
        flips += int(original_position != original_pick_positions[n])
    return flips / len(shuffled_preds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tier1", type=Path, default=Path("data/bench/v0.1/tier1"))
    parser.add_argument("--fresh", type=Path, default=Path("data/bench/v0.1/fresh"))
    parser.add_argument("--keep-screen", type=Path, default=Path("data/bench/v0.1/screen.json"))
    parser.add_argument("--only-template", default=None, help="debug: run one template only")
    parser.add_argument("--out", type=Path, default=Path("outputs/bench_v0_fix0/F6"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--greedy-items", type=int, default=50)
    parser.add_argument("--shuffle-items", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None,
                        help="stamp on every prediction row; defaults to <model>:<utc>")
    parser.add_argument("--write-predictions", action="store_true", default=True)
    args = parser.parse_args()
    return run_model(args)


if __name__ == "__main__":
    raise SystemExit(main())
