#!/usr/bin/env python
"""Shared-prefix measurement: one state, many questions (loop task S3).

Real records carry several checks each, so per-*record* cost matters, not just per-question
cost. This measures, for zero-shot Qwen3.5-0.8B on v0.2 fresh test items:

* **(a) one prompt per question** — the ordinary path: state + question + options rendered and
  prefilled for every question;
* **(b) the state as a shared prefix** — the state is prefilled once and its KV cache is reused
  for every question of that record, so only the question's own tokens are computed.

Both paths are given the **identical token sequence** (the prefix/suffix split is done on the
token ids, and path (a) is the concatenation of the same ids), so the argmax comparison between
them is exact rather than approximate. The acceptance check is that (b) agrees with (a) on at
least 99 % of questions.

Usage:
    uv run python scripts/bench/measure_prefix.py --records 40 --out outputs/student_v0/S3/prefix.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.bench.schema import Item
from meddecide.eval.harness import Harness, ModelSpec
from meddecide.eval.readout import canonicalise_options, option_token_ids, read_option_probabilities
from meddecide.utils.io import read_jsonl, write_json
from meddecide.utils.provenance import Provenance, gpu_name, utcnow

FRESH_DIR = Path("data/bench/v0.2/fresh")


def _split_ids(tokenizer, state: str, rest: str) -> tuple[list[int], list[int]]:
    """Token ids of the state and of the rest of the prompt, as separate encodings."""
    prefix = tokenizer(state, add_special_tokens=False)["input_ids"]
    suffix = tokenizer(rest, add_special_tokens=False)["input_ids"]
    return prefix, suffix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=FRESH_DIR)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--records", type=int, default=40,
                        help="how many records with >= 2 questions to measure")
    parser.add_argument("--out", type=Path, default=Path("outputs/student_v0/S3/prefix.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch

    items: list[Item] = []
    for path in sorted(args.bench.glob("*.jsonl")):
        rows, report = read_jsonl(path, Item)
        report.check_closes()
        items.extend(row for row in rows if str(row.split) == "test")
    by_record: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for item in items:
        by_record[(item.source, item.source_record_id)].append(item)
    # deterministic sample of records that actually carry several questions
    multi = sorted(key for key, rows in by_record.items() if len(rows) >= 2)
    import random

    picked = random.Random(args.seed).sample(multi, min(args.records, len(multi)))

    harness = Harness(ModelSpec(model_id=args.model), batch_size=1, max_batch_tokens=1 << 20,
                      run_id="s3_prefix")
    model = harness.model
    tokenizer = harness.tokenizer
    variant = harness.variant

    per_record: list[dict[str, Any]] = []
    agreements = total_questions = 0
    margins: list[float] = []
    margins_agree: list[float] = []
    disagreements: list[dict[str, Any]] = []
    t_a_total = t_b_total = prefix_total = suffix_total = 0.0
    for key in picked:
        rows = by_record[key]
        # Build (prefix_ids, [(item, suffix_ids)]) with the *identical* token sequence for both
        # paths: the state is everything before the question text in the user message.
        prepared = []
        for item in rows:
            canonical, _ = canonicalise_options(item)
            from meddecide.eval.readout import render_prompt

            full = render_prompt(canonical, tokenizer, variant)
            marker = canonical.question
            split_at = full.find(marker)
            if split_at < 0:
                continue
            prefix_text, rest_text = full[:split_at], full[split_at:]
            prefix_ids, suffix_ids = _split_ids(tokenizer, prefix_text, rest_text)
            prepared.append((canonical, prefix_ids, suffix_ids))
        if not prepared:
            continue
        prefix_ids = prepared[0][1]  # every question of a record shares the same prefix
        if any(p[1] != prefix_ids for p in prepared):
            # a question that changes the prefix is not a shared-prefix case
            prepared = [p for p in prepared if p[1] == prefix_ids]
        if len(prepared) < 2:
            continue

        # ---- (a) one prompt per question -------------------------------------
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        picks_a = []
        probs_a: list[list[float]] = []
        for canonical, p_ids, s_ids in prepared:
            ids = torch.tensor([p_ids + s_ids], device="cuda")
            with torch.no_grad():
                out = model(input_ids=ids, use_cache=False)
            readout = read_option_probabilities(
                out.logits[0, -1].float().cpu().numpy(),
                option_token_ids(canonical, tokenizer, variant),
            )
            probs_a.append([float(x) for x in readout["option_probs"]])
            picks_a.append(int(np.argmax(readout["option_probs"])))
        torch.cuda.synchronize()
        t_a = time.perf_counter() - t0

        # ---- (b) shared prefix with the KV cache reused ----------------------
        # `cache_position` must be given explicitly: with a hybrid cache (attention layers plus
        # a recurrent state) the model cannot always derive the position of the first cached
        # token, and without it the rotary positions restart at 0 for the suffix — which changed
        # the argmax on 15 % of questions in the first version of this measurement.
        def _cached_pass(ids: list[int], cache, start: int):
            return model(
                input_ids=torch.tensor([ids], device="cuda"),
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(start, start + len(ids), device="cuda"),
            )

        # warm the cached path once per shape bucket, so Triton's first-call compile is not
        # charged to the measurement (the S0 lesson, measured again here: the first record's
        # cached path took 3.36 s against 0.06 s for the rest)
        with torch.no_grad():
            warm_cache = model(
                input_ids=torch.tensor([prefix_ids[:64]], device="cuda"), use_cache=True,
                cache_position=torch.arange(64, device="cuda"),
            ).past_key_values
            for warm_len in (32, 64, 128, 256):
                _cached_pass(list(range(warm_len)), warm_cache, 64)
        torch.cuda.empty_cache()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            prefix_out = model(
                input_ids=torch.tensor([prefix_ids], device="cuda"), use_cache=True,
                cache_position=torch.arange(len(prefix_ids), device="cuda"),
            )
        cache = prefix_out.past_key_values
        torch.cuda.synchronize()
        t_prefix = time.perf_counter() - t0
        picks_b = []
        probs_b: list[list[float]] = []
        t_suffix = 0.0
        position = len(prefix_ids)
        for canonical, _p_ids, s_ids in prepared:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            # Every question must branch from the **prefix** cache, not from the cache the
            # previous question left behind: appending question 2 to a cache that already holds
            # question 1 gives question 2 a different context from path (a), which flipped the
            # argmax on 15 % of questions in the first version of this measurement (one flip
            # with a 0.73 probability margin). The copy is charged to the shared-prefix time,
            # because it is a real cost of the approach.
            question_cache = copy.deepcopy(cache)
            with torch.no_grad():
                out = _cached_pass(s_ids, question_cache, position)
            torch.cuda.synchronize()
            t_suffix += time.perf_counter() - t1
            readout_b = read_option_probabilities(
                out.logits[0, -1].float().cpu().numpy(),
                option_token_ids(canonical, tokenizer, variant),
            )
            probs_b.append([float(x) for x in readout_b["option_probs"]])
            picks_b.append(int(np.argmax(readout_b["option_probs"])))
        t_b = t_prefix + t_suffix

        agree = sum(1 for a, b in zip(picks_a, picks_b, strict=True) if a == b)
        for item, pa, pb, a_pick, b_pick in zip(
            [p[0] for p in prepared], probs_a, probs_b, picks_a, picks_b, strict=True
        ):
            ordered = sorted(pa, reverse=True)
            margin = ordered[0] - ordered[1]
            margins.append(margin)
            if a_pick != b_pick:
                disagreements.append(
                    {
                        "item_id": item.item_id,
                        "template_id": item.template_id,
                        "margin_top1_top2": margin,
                        "probs_prefill": [round(x, 4) for x in pa],
                        "probs_cached": [round(x, 4) for x in pb],
                    }
                )
            else:
                margins_agree.append(margin)
        agreements += agree
        total_questions += len(prepared)
        t_a_total += t_a
        t_b_total += t_b
        prefix_total += t_prefix
        suffix_total += t_suffix
        per_record.append(
            {
                "record": f"{key[0]}:{key[1]}",
                "n_questions": len(prepared),
                "prefix_tokens": len(prefix_ids),
                "suffix_tokens_mean": float(np.mean([len(p[2]) for p in prepared])),
                "seconds_one_prompt_per_question": t_a,
                "seconds_shared_prefix": t_b,
                "seconds_prefix_prefill": t_prefix,
                "agreements": agree,
                "templates": sorted({item.template_id for item, _p, _s in
                                     [(p[0], p[1], p[2]) for p in prepared]}),
            }
        )

    report = {
        "generated_at_utc": utcnow(),
        "model": args.model,
        "n_records": len(per_record),
        "n_questions": total_questions,
        "questions_per_record_mean": total_questions / len(per_record) if per_record else None,
        "questions_per_record_histogram": dict(
            sorted(Counter(p["n_questions"] for p in per_record).items())
        ),
        "agreement": agreements / total_questions if total_questions else None,
        "n_agreements": agreements,
        "throughput": {
            "one_prompt_per_question": {
                "seconds_total": t_a_total,
                "questions_per_s": total_questions / t_a_total if t_a_total else None,
                "records_per_s": len(per_record) / t_a_total if t_a_total else None,
            },
            "shared_prefix": {
                "seconds_total": t_b_total,
                "seconds_prefix_prefill_total": prefix_total,
                "seconds_suffix_total": suffix_total,
                "questions_per_s": total_questions / t_b_total if t_b_total else None,
                "records_per_s": len(per_record) / t_b_total if t_b_total else None,
            },
            "speedup_questions": (t_a_total / t_b_total) if t_b_total else None,
        },
        "margin_top1_top2": {
            "n": len(margins),
            "median": float(np.median(margins)) if margins else None,
            "median_where_paths_agree": float(np.median(margins_agree)) if margins_agree else None,
            "max_among_disagreements": max(
                (d["margin_top1_top2"] for d in disagreements), default=None
            ),
        },
        "disagreements": disagreements,
        "checks": {
            "agreement_at_least_0.99": (agreements / total_questions) >= 0.99
            if total_questions
            else False,
            "agreement_at_least_0.99_above_0.05_margin": (
                all(d["margin_top1_top2"] <= 0.05 for d in disagreements)
                if total_questions
                else False
            ),
            "shared_prefix_faster_than_per_question": t_b_total < t_a_total,
        },
        "per_record": per_record,
        "protocol": (
            "both paths receive the identical token sequence: path (a) is the concatenation of "
            "the prefix and suffix ids, path (b) prefills the prefix once and feeds only the "
            "suffix with the returned cache. Batch size 1, use_cache=False for (a)."
        ),
    }
    prov = Provenance(
        run_name="S3_measure_prefix",
        command=" ".join([str(x) for x in [__file__, *[]]]) or "scripts/bench/measure_prefix.py",
        gpu=gpu_name(),
        seed=args.seed,
        config={"model": args.model, "records": args.records},
    )
    report["checks"]["verdict"] = "PASS" if all(report["checks"].values()) else "FAIL"
    write_json(args.out, report)
    prov.finish().write(args.out.with_name("prefix_provenance.json"))
    print(json.dumps({k: v for k, v in report.items() if k != "per_record"}, indent=2))
    print(f"wrote {args.out}")
    return 0 if report["checks"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
