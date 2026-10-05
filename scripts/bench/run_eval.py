#!/usr/bin/env python
"""Run the T6 eval harness on a model and a set of bench files.

Usage:
    uv run python scripts/bench/run_eval.py --model Qwen/Qwen3.5-0.8B \
        --items data/bench/tier1/medqa.jsonl --limit 50 --task T6 --tag e2e50

Writes, under ``outputs/bench_v0/<task>/``:
    preds/<tag>__<model-slug>__<source>.jsonl   per-item predictions (never committed)
    summaries/<tag>__<model-slug>.json          per (source, template, split) metrics
    configs/<tag>__<model-slug>.json            the exact config + label-token check used
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from meddecide.bench.schema import Item
from meddecide.eval.harness import Harness, ModelSpec, summarise, write_predictions, write_summary
from meddecide.utils.io import read_jsonl
from meddecide.utils.provenance import Provenance, gpu_name, utcnow


def _slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--items", nargs="+", required=True, help="bench JSONL files")
    parser.add_argument("--limit", type=int, default=None, help="max items per file")
    parser.add_argument("--task", default="T6")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-options", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=32768,
                        help="cap on total prompt tokens per batch (memory safety)")
    parser.add_argument("--max-prompt-tokens", type=int, default=16384,
                        help="truncate prompts longer than this")
    parser.add_argument("--debug-batches", action="store_true")
    parser.add_argument("--splits", nargs="*", default=None, help="restrict to these splits")
    parser.add_argument("--config", type=Path, default=Path("configs/bench_v0.yaml"))
    args = parser.parse_args()

    out_root = Path(f"outputs/bench_v0/{args.task}")
    for sub in ("preds", "summaries", "configs", "logs"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    items: list[Item] = []
    per_file: dict[str, int] = {}
    for path_str in args.items:
        path = Path(path_str)
        rows, report = read_jsonl(path, Item)
        report.check_closes()
        if args.splits:
            rows = [r for r in rows if str(r.split) in set(args.splits)]
        if args.limit is not None:
            rows = rows[: args.limit]
        items.extend(rows)
        per_file[path.name] = len(rows)
    if not items:
        raise SystemExit("no items to score")

    spec = ModelSpec(model_id=args.model, revision=args.revision, max_options=args.max_options)
    t_start = utcnow()
    harness = Harness(
        spec,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        debug=args.debug_batches,
    )
    predictions = harness.score(items)
    summary = summarise(predictions, seed=args.seed)

    slug = _slug(args.model)
    preds_path = write_predictions(
        out_root / "preds" / f"{args.tag}__{slug}.jsonl", predictions
    )
    summary["run"] = {
        "tag": args.tag,
        "task": args.task,
        "model": spec.to_dict(),
        "started_at_utc": t_start,
        "finished_at_utc": utcnow(),
        "items_per_file": per_file,
        "n_items": len(items),
        "n_predictions": len(predictions),
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "seed": args.seed,
        "splits": args.splits,
        "label_token_check": harness.label_check.to_dict(),
        "variant_detection": harness.variant_detection,
        "predictions_path": str(preds_path),
    }
    summary_path = write_summary(out_root / "summaries" / f"{args.tag}__{slug}.json", summary)
    config_path = out_root / "configs" / f"{args.tag}__{slug}.json"
    config_path.write_text(
        json.dumps(
            {
                "model": spec.to_dict(),
                "items": args.items,
                "limit": args.limit,
                "splits": args.splits,
                "batch_size": args.batch_size,
                "max_batch_tokens": args.max_batch_tokens,
                "max_prompt_tokens": args.max_prompt_tokens,
                "seed": args.seed,
                "max_options": args.max_options,
                "bench_config": str(args.config),
                "bench_config_sha256": __import__("hashlib").sha256(
                    Path(args.config).read_bytes()
                ).hexdigest()
                if Path(args.config).is_file()
                else None,
            },
            indent=2,
        )
        + "\n"
    )

    prov = Provenance(
        run_name=f"{args.task}_{args.tag}",
        command=" ".join([sys.executable, *sys.argv]),
        gpu=gpu_name(),
        seed=args.seed,
        config={"items": args.items, "limit": args.limit, "batch_size": args.batch_size},
    )
    try:
        from huggingface_hub import HfApi

        if args.revision is None:
            spec.revision = HfApi().model_info(args.model).sha
        prov.models.append({"id": args.model, "revision": spec.revision})
    except Exception as exc:  # pragma: no cover - network dependent
        prov.notes = f"revision lookup failed: {type(exc).__name__}"
    prov.finish().write(out_root / "logs" / f"{args.tag}__{slug}__provenance.json")

    print(json.dumps({"predictions": str(preds_path), "summary": str(summary_path)}, indent=2))
    print(json.dumps(summary["groups"], indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
