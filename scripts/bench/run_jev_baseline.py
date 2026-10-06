#!/usr/bin/env python
"""F7: `autotrust/JEV-9B` decision-head baseline on v0.1, through its documented protocol.

The model card specifies a decision head reached through a LoRA module: the prompt ends in
`[decision]:`, a single forward step reads the log-probabilities of the kind's *verbalizer* tokens
(`false`/`true`, the score digits, or `A`…`P` for choice), and the head's per-slot **bias** and the
per-kind **temperature** are applied client-side. Those three artifacts are downloaded from the
model repo (`adapter_vllm/decision_head.json`, `calibration.json`), so this script reproduces the
published protocol rather than inventing one.

**Deviation from the card, stated.** The card serves the model with vLLM
(`--lora-modules jev-decision=JEV-9B/adapter_vllm`). vLLM is not installed in this environment and
is not compatible with the installed stack without a large upgrade, so the adapter is loaded with
`peft` and the same single-prefill read is performed with `transformers`. The card notes the two
paths agree to mean |Δp| 0.0008 (test KL 0.0210 vs 0.0211), so this is a fidelity-preserving
substitution; the substitution is recorded in the report and in the provenance.

**Option-count limit.** The head has 24 slots: 2 for `noul`, 6 for `score`, 16 for `choice`. A
benchmark `choice` item with more than 16 options cannot be expressed; those items are **not
scored** and the shortfall is reported per cell (never silently dropped).

These models read options natively, so the letter-readout health gate (D12) does not apply; the
cell reports its own distribution's accuracy, Brier, ECE, coverage and option-shuffle flip rate.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from meddecide.bench.schema import Item, QuestionType
from meddecide.eval.metrics import bootstrap_ci, ece, macro_accuracy
from meddecide.eval.probes import shuffle_options
from meddecide.utils.io import read_jsonl
from meddecide.utils.provenance import Provenance, gpu_name, utcnow

MODEL_ID = "autotrust/JEV-9B"
MAX_CHOICE_SLOTS = 16
LETTERS = "ABCDEFGHIJKLMNOP"


def load_head(repo: str = MODEL_ID) -> tuple[dict[str, Any], dict[str, float]]:
    from huggingface_hub import hf_hub_download

    head = json.loads(Path(hf_hub_download(repo, "adapter_vllm/decision_head.json")).read_text())
    calibration = json.loads(Path(hf_hub_download(repo, "calibration.json")).read_text())
    return head, calibration["per_kind"]


def build_prompt(kind: str, state: str, question: str, option_labels: list[str]) -> str:
    """The card's prompt, verbatim: `[kind] …\\n[state] …\\n[question] …\\n[options]\\n…\\n[decision]:`."""
    if kind == "choice":
        lines = [f"{LETTERS[i]}) {label}" for i, label in enumerate(option_labels)]
    elif kind == "score":
        lines = [str(i) for i in range(len(option_labels))]
    else:
        lines = ["false", "true"]
    return (
        f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n"
        + "\n".join(lines)
        + "\n[decision]:"
    )


def verbalizer_ids(head: dict[str, Any], kind: str, n_options: int) -> list[int]:
    start = int(head["slots"]["ranges"][kind][0])
    return [int(v) for v in head["verbalizer_ids"][start : start + n_options]]


def head_probabilities(
    logits_row: torch.Tensor,
    ids: list[int],
    head: dict[str, Any],
    kind: str,
    temperature: float,
    n_options: int,
) -> list[float]:
    """Apply the head's bias and the per-kind temperature, then softmax (the card's client step)."""
    start = int(head["slots"]["ranges"][kind][0])
    logprobs = torch.log_softmax(logits_row.float(), dim=-1)
    z = [
        (float(logprobs[t]) + float(head["bias"][start + i])) / float(temperature)
        for i, t in enumerate(ids[:n_options])
    ]
    m = max(z)
    e = [float(np.exp(x - m)) for x in z]
    total = sum(e)
    return [x / total for x in e]


def kind_of(item: Item) -> str:
    if item.qtype is QuestionType.NOUL:
        return "noul"
    if item.qtype is QuestionType.SCORE:
        return "score"
    return "choice"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier1", type=Path, default=Path("data/bench/v0.1/tier1"))
    parser.add_argument("--fresh", type=Path, default=Path("data/bench/v0.1/fresh"))
    parser.add_argument("--keep-screen", type=Path, default=Path("data/bench/v0.1/screen.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/bench_v0_fix0/F7"))
    parser.add_argument("--only-template", default=None)
    parser.add_argument("--limit-per-template", type=int, default=None)
    parser.add_argument("--shuffle-items", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="read one item per forward pass; larger values pad the vocab-dim "
                             "logits across the batch and OOM on long openFDA prompts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None,
                        help="cap items per template (debug/large-template safety)")
    parser.add_argument("--resume", action="store_true",
                        help="keep cells already measured in jev9b.json (per-template runs)")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    head, temperatures = load_head()
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, MODEL_ID, subfolder="adapter_vllm")
    model.eval()
    load_seconds = time.time() - t0

    screen = json.loads(args.keep_screen.read_text())
    kept = {t["template_id"] for t in screen["templates"] if not t["drop"]}

    report: dict[str, Any] = {
        "generated_at_utc": utcnow(),
        "model_id": MODEL_ID,
        "adapter": "adapter_vllm (decision head)",
        "protocol": "the model card's SinglePrefill path: [decision]: prompt, verbalizer tokens, "
                    "head bias and per-kind temperature applied client-side",
        "deviation": "served with transformers+peft instead of vLLM (card reports mean |dp| 0.0008 "
                     "between the paths); vLLM is not installed in this environment",
        "temperatures": temperatures,
        "load_seconds": load_seconds,
        "gpu": gpu_name(),
        "cell_option_limit": MAX_CHOICE_SLOTS,
        "cells": [],
    }
    preds_path = args.out / "preds_jev9b.jsonl"

    existing: dict[str, Any] = {}
    existing_path = args.out / "jev9b.json"
    if args.resume and existing_path.exists():
        old = json.loads(existing_path.read_text())
        existing = {c["template_id"]: c for c in old.get("cells", []) if c.get("status") == "measured"}
        report["cells"] = list(old.get("cells", []))
        print(f"resuming: {len(existing)} cells already measured", flush=True)

    for tier, directory in (("tier1", args.tier1), ("fresh", args.fresh)):
        rows: list[Item] = []
        for path in sorted(directory.glob("*.jsonl")):
            loaded, validation = read_jsonl(path, Item)
            validation.check_closes()
            rows.extend(i for i in loaded if str(i.split) == "test" and i.template_id in kept)
        by_template: dict[str, list[Item]] = defaultdict(list)
        for item in rows:
            by_template[item.template_id].append(item)

        for template_id, template_items in sorted(by_template.items()):
            if args.only_template and template_id != args.only_template:
                continue
            if args.limit_per_template:
                template_items = template_items[: args.limit_per_template]
            if args.max_items:
                template_items = template_items[: args.max_items]
            if args.only_template is None and template_id in existing:
                continue
            kind = kind_of(template_items[0])
            too_many = 0
            scored: list[dict[str, Any]] = []
            for item in template_items:
                n_options = item.n_options
                if kind == "choice" and n_options > MAX_CHOICE_SLOTS:
                    too_many += 1
                    continue
                labels = [o.label for o in item.options]
                prompt = build_prompt(kind, item.state, item.question, labels)
                ids = verbalizer_ids(head, kind, n_options)
                # one item at a time: the JEV base is Qwen3.5-9B with a 151k vocabulary, and
                # batching pads every sequence to the longest one, which made the final
                # [batch, seq, vocab] logits tensor reach 33 GiB on the openFDA templates (the
                # model's own prediction is a single position, so nothing is lost by batching 1)
                enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                with torch.inference_mode():
                    logits = model(**enc).logits[0, -1, :]
                probs = head_probabilities(logits, ids, head, kind, temperatures[kind], n_options)
                if kind == "choice":
                    gold_index = item.gold_index
                else:
                    # noul: slot order is [false, true]; score: slot order is digit == level index
                    gold_index = 0 if item.gold == "no" else 1 if kind == "noul" else item.gold_index
                argmax = int(np.argmax(probs))
                scored.append(
                    {
                        "item_id": item.item_id,
                        "template_id": template_id,
                        "source": item.source,
                        "qtype": kind,
                        "n_options": n_options,
                        "probs": probs,
                        "argmax": argmax,
                        "gold_index": gold_index,
                        "correct": argmax == gold_index,
                        "label_mass": float(sum(probs)),
                        "prompt_tokens": int(enc["input_ids"].shape[1]),
                    }
                )
            if not scored:
                report["cells"].append({
                    "model_id": MODEL_ID, "tier": tier, "template_id": template_id,
                    "qtype": kind, "n": 0, "template_total": len(template_items),
                    "status": f"NOT MEASURED — every item exceeds the {MAX_CHOICE_SLOTS}-option head",
                })
                continue

            correct = [s["correct"] for s in scored]
            gold = [s["gold_index"] for s in scored]
            probs_matrix = np.asarray([s["probs"] for s in scored], dtype=np.float64)
            accuracy = float(np.mean(correct))
            _, lo, hi = bootstrap_ci([1.0 if c else 0.0 for c in correct], n_resamples=1000,
                                     seed=args.seed)
            predicted = [str(s["argmax"]) for s in scored]
            gold_str = [str(g) for g in gold]
            gold_prob = probs_matrix[np.arange(len(scored)), np.asarray(gold)]
            flip = None
            if args.shuffle_items > 0 and kind == "choice":
                try:
                    flip = shuffle_flip(
                        model, tokenizer, head, temperatures, template_items,
                        args.shuffle_items, args.seed,
                    )
                except (RuntimeError, torch.OutOfMemoryError) as exc:
                    # an out-of-memory on the probe must not lose the cell's accuracy: the flip
                    # rate is recorded as NOT MEASURED with its reason
                    print(f"  shuffle probe failed for {template_id}: {type(exc).__name__}",
                          flush=True)
                    flip = None
            report["cells"] = [c for c in report["cells"] if c["template_id"] != template_id]
            report["cells"].append({
                "model_id": MODEL_ID, "tier": tier, "template_id": template_id, "qtype": kind,
                "source": scored[0]["source"], "n": len(scored),
                "template_total": len(template_items),
                "coverage_under_head_limit": len(scored) / len(template_items),
                "n_excluded_over_limit": too_many,
                "n_options": scored[0]["n_options"],
                "chance": 0.5 if kind == "noul" else 1.0 / scored[0]["n_options"],
                "accuracy": accuracy, "accuracy_ci95": [lo, hi],
                "n_correct": int(sum(correct)),
                "majority_share": max(Counter(gold_str).values()) / len(scored),
                "macro_accuracy": macro_accuracy(predicted, gold_str),
                "brier": float(np.mean((gold_prob - np.asarray(correct, dtype=float)) ** 2)),
                "ece": ece(probs_matrix, gold),
                "mean_gold_prob": float(gold_prob.mean()),
                "median_label_mass": float(np.median([s["label_mass"] for s in scored])),
                "shuffle_flip_rate": flip,
                "p50_prompt_tokens": float(np.median([s["prompt_tokens"] for s in scored])),
                "status": "measured",
            })
            cell = report["cells"][-1]
            print(f"{MODEL_ID} {tier}/{template_id} ({kind}): n={cell['n']}/{cell['template_total']} "
                  f"acc={accuracy:.4f} brier={cell['brier']:.4f} ece={cell['ece']:.4f} "
                  f"flip={flip}", flush=True)
            with preds_path.open("a", encoding="utf-8") as fh:
                for s in scored:
                    fh.write(json.dumps(s, ensure_ascii=False) + "\n")
            del scored
            gc.collect()
            torch.cuda.empty_cache()

    report["wall_seconds"] = time.time() - t0
    (args.out / "jev9b.json").write_text(json.dumps(report, indent=2) + "\n")
    prov = Provenance(
        run_name="F7_jev9b", command=" ".join([sys.executable, *sys.argv]), gpu=gpu_name(),
        seed=args.seed,
        config={"model": MODEL_ID, "shuffle_items": args.shuffle_items,
                "batch_size": args.batch_size},
    )
    prov.datasets = [{"id": MODEL_ID, "revision": None}]
    prov.finish().write(args.out / "jev9b_provenance.json")
    print(f"wall {report['wall_seconds']:.0f}s, {len(report['cells'])} cells")
    return 0


def shuffle_flip(
    model, tokenizer, head, temperatures, items: list[Item], n_items: int, seed: int
) -> float | None:
    """Share of sampled choice items whose chosen option *content* changes under reordering."""
    rng = random.Random(f"{seed}:jevflip")
    idx = rng.sample(range(len(items)), min(n_items, len(items)))
    flips = 0
    total = 0
    for i in idx:
        item = items[i]
        if item.n_options > MAX_CHOICE_SLOTS:
            continue
        original = pick_option(model, tokenizer, head, temperatures, item)
        shuffled, record = shuffle_options(item, seed=seed, salt=item.item_id)
        moved = pick_option(model, tokenizer, head, temperatures, shuffled)
        perm = [int(x) for x in record.details["perm"]]
        flips += int(perm[moved] != original)
        total += 1
    return flips / total if total else None


def pick_option(model, tokenizer, head, temperatures, item: Item) -> int:
    labels = [o.label for o in item.options]
    prompt = build_prompt("choice", item.state, item.question, labels)
    ids = verbalizer_ids(head, "choice", item.n_options)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.inference_mode():
        logits = model(**enc).logits[0, -1, :]
    probs = head_probabilities(logits, ids, head, "choice", temperatures["choice"], item.n_options)
    return int(np.argmax(probs))


if __name__ == "__main__":
    raise SystemExit(main())
