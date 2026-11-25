#!/usr/bin/env python
"""F5 (T4): tier-1 contamination probe — Min-K%-Prob memorisation score with a matched control.

For every ladder model and every tier-1 test source, on a fixed sample of up to 500 items:

* **score** — Min-K% Prob (K=20 %) of the item's question text under the model: the mean
  log-probability of the least likely 20 % of its tokens. Higher (less negative) means the model
  found the exact wording easy to predict, which is what memorisation looks like.
* **control** — the same measurement on the *same* item text with its sentences deterministically
  reordered (same tokens, different order). Reordering destroys verbatim sequence memorisation
  while leaving vocabulary and length unchanged, so the **gap** (score - control) is the probe.

Every text is scored with its own leading token dropped (the first token of a text has no
predictor). Items whose text cannot be reordered (fewer than two sentences) have no control and
are excluded from the gap, with the count reported — never scored against themselves.

**These are probes, not proofs.** A high gap is consistent with memorisation but also with the
original ordering simply being more predictable prose; a low gap does not prove a model never saw
the data. The report says so, and no tier-1 accuracy in this loop is adjusted by these numbers.

Writes `outputs/bench_v0_fix0/F5/contamination.json` and the committed
`loops/bench_v0_fix0/contamination.md`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from meddecide.bench.schema import Item
from meddecide.eval.probes import min_k_prob, reorder_control
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


def load_hf(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


def token_logprobs(model, tokenizer, device: str, text: str) -> list[float]:
    """Per-token log-probabilities of ``text``; the first token has no predictor and is dropped."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 2:
        return []
    inputs = torch.tensor([ids], device=device)
    with torch.inference_mode():
        logits = model(inputs).logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    # position i predicts token i+1
    return [
        float(logprobs[i, ids[i + 1]]) for i in range(len(ids) - 1)
    ]


def sample_items(directory: Path, per_source: int, seed: int) -> dict[str, list[Item]]:
    by_source: dict[str, list[Item]] = defaultdict(list)
    for path in sorted(directory.glob("*.jsonl")):
        rows, report = read_jsonl(path, Item)
        report.check_closes()
        by_source[path.stem].extend(i for i in rows if str(i.split) == "test")
    out: dict[str, list[Item]] = {}
    for source, rows in sorted(by_source.items()):
        rows = sorted(rows, key=lambda r: r.item_id)  # deterministic before sampling
        rng = random.Random(f"{seed}:{source}")
        out[source] = rows if len(rows) <= per_source else rng.sample(rows, per_source)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier1", type=Path, default=Path("data/bench/v0.1/tier1"))
    parser.add_argument("--models", nargs="+", default=LADDER)
    parser.add_argument("--per-source", type=int, default=500)
    parser.add_argument("--k-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="skip items whose text exceeds this many tokens (counted, not hidden)")
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/bench_v0_fix0/F5/contamination.json"))
    parser.add_argument("--report", type=Path,
                        default=Path("loops/bench_v0_fix0/contamination.md"))
    args = parser.parse_args()

    sample = sample_items(args.tier1, args.per_source, args.seed)
    report: dict[str, Any] = {
        "generated_at_utc": utcnow(),
        "probe": (
            f"Min-K% Prob, K={args.k_fraction}, over the item's state text; control = deterministic "
            "reorder of the same text (sentences, else clauses, else lines)"
        ),
        "caveat": (
            "Probes, not proofs: a gap is consistent with memorisation but also with the original "
            "ordering being more predictable prose, and a small gap does not prove a model never "
            "saw the data. No tier-1 accuracy in this loop is adjusted by these numbers."
        ),
        "k_fraction": args.k_fraction,
        "seed": args.seed,
        "per_source_cap": args.per_source,
        "max_tokens": args.max_tokens,
        "sample": {s: len(rows) for s, rows in sample.items()},
        "models": {},
    }

    for model_id in args.models:
        model, tokenizer, device = load_hf(model_id)
        entry: dict[str, Any] = {"model_id": model_id, "sources": {}}
        for source, rows in sample.items():
            scores: list[float] = []
            controls: list[float] = []
            no_control = 0
            too_long = 0
            control_kinds: dict[str, int] = {}
            for item in rows:
                text = item.state.strip()
                if len(tokenizer.encode(text, add_special_tokens=False)) > args.max_tokens:
                    too_long += 1
                    continue
                control_text, record = reorder_control(text, seed=args.seed, salt=item.item_id)
                lp = token_logprobs(model, tokenizer, device, text)
                if not lp:
                    continue
                scores.append(min_k_prob(lp, args.k_fraction))
                kind = str(record.details.get("kind"))
                control_kinds[kind] = control_kinds.get(kind, 0) + 1
                if not record.details.get("applied"):
                    no_control += 1
                    continue
                lp_c = token_logprobs(model, tokenizer, device, control_text)
                if lp_c:
                    controls.append(min_k_prob(lp_c, args.k_fraction))
            n = len(scores)
            entry["sources"][source] = {
                "n_scored": n,
                "n_with_control": len(controls),
                "n_without_control": no_control,
                "n_excluded_too_long": too_long,
                "control_kinds": dict(sorted(control_kinds.items())),
                "mean_score": float(np.mean(scores)) if scores else None,
                "mean_control": float(np.mean(controls)) if controls else None,
                "mean_gap": float(np.mean(scores) - np.mean(controls)) if controls else None,
                "median_score": float(np.median(scores)) if scores else None,
                "median_control": float(np.median(controls)) if controls else None,
                "median_gap": (
                    float(np.median(scores) - np.median(controls)) if controls else None
                ),
            }
            src = entry["sources"][source]
            score_txt = "n/a" if src["mean_score"] is None else f"{src['mean_score']:.4f}"
            control_txt = "n/a" if src["mean_control"] is None else f"{src['mean_control']:.4f}"
            gap_txt = "n/a" if src["mean_gap"] is None else f"{src['mean_gap']:+.4f}"
            print(f"{model_id} {source}: n={n} ctrl={src['n_with_control']} kinds={control_kinds} "
                  f"score={score_txt} control={control_txt} gap={gap_txt}", flush=True)
        all_gaps = [s["mean_gap"] for s in entry["sources"].values() if s["mean_gap"] is not None]
        entry["mean_gap_across_sources"] = float(np.mean(all_gaps)) if all_gaps else None
        report["models"][model_id] = entry
        del model
        torch.cuda.empty_cache()

    gaps = {
        model_id: entry["mean_gap_across_sources"]
        for model_id, entry in report["models"].items()
    }
    report["summary"] = {
        "n_models": len(report["models"]),
        "mean_gap_by_model": gaps,
        "max_gap_model": max(gaps, key=lambda k: gaps[k] or -99) if gaps else None,
        "max_gap": max((g for g in gaps.values() if g is not None), default=None),
        "verdict": (
            "NOT MEASURED — no model produced a control-comparable gap"
            if not any(g is not None for g in gaps.values())
            else "measured (see caveat: probes, not proofs)"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Tier-1 contamination probe (F5 / T4)",
        "",
        f"Generated: `{report['generated_at_utc']}`  ",
        f"Probe: {report['probe']}  ",
        f"Sample: up to {args.per_source} tier-1 **test** items per source, seed {args.seed}.",
        "",
        "> **Probes, not proofs.** A score gap between the original text and a sentence-reordered",
        "> control is *consistent with* verbatim memorisation, but the original ordering can simply",
        "> be more predictable prose. A small gap does not prove a model never saw the data. No",
        "> tier-1 accuracy in this loop is adjusted by these numbers.",
        "",
        "Min-K% Prob is the mean log-probability of the least likely 20 % of the text's tokens under",
        "the model, so **less negative = easier to predict**. The control reorders the *same* text at",
        "the finest available granularity - sentences, else clauses, else lines - keeping every",
        "token; the granularity used per source is in `control_kinds`. Items with only one sentence,",
        "clause and line have no control and are excluded from the gap (counted in the JSON).",
        "",
        "## Mean Min-K% score, control, and gap by model and source",
        "",
        "| model | source | n scored | score | control | gap |",
        "|---|---|---|---|---|---|",
    ]
    for model_id, entry in report["models"].items():
        for source, values in entry["sources"].items():
            gap = values["mean_gap"]
            control = values["mean_control"]
            control_txt = "NOT MEASURED" if control is None else f"{control:.4f}"
            gap_txt = "NOT MEASURED" if gap is None else f"{gap:+.4f}"
            lines.append(
                f"| `{model_id}` | {source} | {values['n_scored']} | "
                f"{values['mean_score']:.4f} | {control_txt} | {gap_txt} |"
            )
    lines += ["", "## Summary", "", "| model | mean gap across sources |", "|---|---|"]
    for model_id, gap in gaps.items():
        gap_txt = "NOT MEASURED" if gap is None else f"{gap:+.4f}"
        lines.append(f"| `{model_id}` | {gap_txt} |")
    lines += [
        "",
        "## How to read this",
        "",
        "* A **negative** gap means the reordered control was *easier* to predict than the original",
        "  ordering; that is the expected direction for prose whose original order is informative.",
        "* The probe is informative only in comparison: a model whose gap is much larger than the",
        "  others' on a given source is the one a reviewer should suspect of having seen that source.",
        "* Tier-1 numbers in `FINDINGS.md` are reported as they are measured, with this caveat beside",
        "  them; the contamination-resistant claim rests on **tier 2**, not on this probe.",
        "",
    ]
    args.report.write_text("\n".join(lines) + "\n")
    prov = Provenance(
        run_name="F5_contamination_probe",
        command=" ".join([sys.executable, *sys.argv]),
        gpu=gpu_name(),
        seed=args.seed,
        config={"models": args.models, "per_source": args.per_source, "k_fraction": args.k_fraction},
    )
    prov.finish().write(args.out.with_name("contamination_provenance.json"))
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
