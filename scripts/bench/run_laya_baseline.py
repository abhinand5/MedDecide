#!/usr/bin/env python
"""F7: `convaiinnovations/laya` (and its typed-decisions checkpoint) as a decision-model baseline.

Laya is a non-autoregressive System-1 decision model: one forward pass over
`[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]` returns a probability
per option. This script uses the repository's **own** implementation rather than a re-implementation:
`rl_agent_api.RLAgent.system_one` and the `rl_common` helpers are imported from the downloaded
snapshot, and the post-hoc temperature is applied exactly as the model's own API does it
(`logits / temperature_by_options[temp_bucket(qtype, k)]`, falling back to the per-qtype scalar).

Mapping onto v0.1 items:

* `noul`  — Laya's noul options are always `[false, true]`, so `p[1]` is p(yes):
  gold `yes` → index 1, gold `no` → index 0.
* `choice` — `criteria` is a list, and `render_options` renders it in that order, so index *i* is
  the item's option *i*; gold index passes through.
* `score` — criteria are the level descriptions in order, so index *i* is the item's level *i*.

The report states coverage explicitly: Laya truncates the state to its configured `max_len` and
refuses a question whose options do not fit its head budget, and those items are counted, not
silently dropped. Laya reads options natively, so the letter-readout health gate does not apply.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.bench.schema import Item, QuestionType
from meddecide.eval.metrics import bootstrap_ci, ece, macro_accuracy
from meddecide.eval.probes import shuffle_options
from meddecide.utils.io import read_jsonl
from meddecide.utils.provenance import Provenance, gpu_name, utcnow

MODELS = {
    "laya": "convaiinnovations/laya",
    "laya-typed-decisions": "convaiinnovations/laya-typed-decisions",
}


def kind_of(item: Item) -> str:
    if item.qtype is QuestionType.NOUL:
        return "noul"
    if item.qtype is QuestionType.SCORE:
        return "score"
    return "choice"


def questions_for(item: Item, kind: str) -> tuple[dict[str, Any], int]:
    """Laya question definition plus the index of the gold option in its order."""
    if kind == "choice":
        labels = [o.label for o in item.options]
        return (
            {"type": "choice", "instructions": item.question or "Which option applies?",
             "criteria": labels},
            item.gold_index,
        )
    if kind == "score":
        labels = [o.label for o in item.options]
        return (
            {"type": "score", "instructions": item.question or "Which level applies?",
             "criteria": labels},
            item.gold_index,
        )
    return (
        {"type": "noul", "instructions": item.question or "Does the statement hold?"},
        1 if item.gold == "yes" else 0,
    )


def load_agent(model_key: str, device: str = "cuda"):
    """Load a Laya checkpoint using the implementation published in `convaiinnovations/laya`.

    The two checkpoints differ in where the weights live and in which modules ship with them:

    * `convaiinnovations/laya` — weights at the root, and the implementation (`rl_agent_api.py`,
      `rl_common.py`) alongside them;
    * `convaiinnovations/laya-typed-decisions` — weights under `typed-decisions/`, and **no
      implementation at all** (the repo contains only the safetensors, configs and tokenizer).

    So the implementation is always taken from the main repo and the weights from the requested
    checkpoint; `RLAgent` receives the directory holding that checkpoint's `model.safetensors`.
    """
    import shutil
    import tempfile

    from huggingface_hub import snapshot_download

    code_dir = snapshot_download("convaiinnovations/laya", allow_patterns=["*.py", "*.json"])
    if model_key == "laya":
        agent_dir = code_dir
    else:
        weights_dir = snapshot_download(
            MODELS[model_key],
            allow_patterns=["*.json", "*.safetensors", "tokenizer/*", "encoder/*"],
        )
        sub = os.path.join(weights_dir, "typed-decisions")
        if not os.path.isdir(sub):
            # this checkpoint's weights are at its own root, not in a subdirectory
            sub = weights_dir
        # stage the implementation next to the weights so the modules' bare-name imports resolve
        workdir = Path(tempfile.mkdtemp(prefix="laya-typed-"))
        for name in ("rl_agent_api.py", "rl_common.py", "email_utils.py"):
            src = os.path.join(code_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, workdir / name)
        for entry in os.listdir(sub):
            src = os.path.join(sub, entry)
            dst = workdir / entry
            if os.path.isdir(src):
                if not dst.exists():
                    os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)
        agent_dir = str(workdir)
    sys.path.insert(0, agent_dir)
    from rl_agent_api import RLAgent  # type: ignore[import-not-found]

    agent = RLAgent(agent_dir, device=device)
    return agent, agent_dir


def probabilities(agent: Any, state: str, qdef: dict[str, Any], n_options: int, kind: str) -> list[float]:
    """Run the repository's own system_one and return option probabilities in display order."""
    out = agent.system_one(state, {"q": qdef})
    answer = out["answers"]["q"]
    # the API returns {"choice"|"score": <label>, "probabilities": {<label>: p}, "confidence": c},
    # so read the distribution from `probabilities` in the order `render_options` produced,
    # which is the same order the model's option head was built in
    probs = answer.get("probabilities")
    if kind == "noul":
        if isinstance(probs, dict) and len(probs) == 2:
            values = [float(v) for _, v in probs.items()]
            return values
        p_yes = float(answer.get("noul", 0.5))
        return [1.0 - p_yes, p_yes]
    if isinstance(probs, dict) and probs:
        values = [float(v) for _, v in probs.items()]
    elif isinstance(probs, list):
        values = [float(v) for v in probs]
    else:
        raise ValueError(f"no probabilities in answer: {sorted(answer)}")
    if len(values) != n_options:
        raise ValueError(f"got {len(values)} probabilities for {n_options} options")
    return values


def run(args: argparse.Namespace) -> int:
    agent, agent_dir = load_agent(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    screen = json.loads(args.keep_screen.read_text())
    kept = {t["template_id"] for t in screen["templates"] if not t["drop"]}

    report: dict[str, Any] = {
        "generated_at_utc": utcnow(),
        "model_id": MODELS[args.model],
        "implementation": f"the repository's own RLAgent.system_one from {agent_dir}",
        "protocol": "single forward pass; per-option [MASK] marker probabilities; post-hoc "
                    "temperature applied to logits exactly as rl_agent_api does",
        "max_len": agent.cfg.get("max_len"),
        "head_max_len": agent.cfg.get("head_max_len"),
        "temperature": agent.cfg.get("temperature"),
        "temperature_by_options": agent.cfg.get("temperature_by_options"),
        "gpu": gpu_name(),
        "cells": [],
    }
    preds_path = args.out / f"preds_{args.model}.jsonl"
    t0 = time.time()

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
            kind = kind_of(template_items[0])
            scored: list[dict[str, Any]] = []
            skipped_options = 0
            skipped_collision = 0
            skipped_single_option = 0
            skipped_error = 0
            for item in template_items:
                qdef, gold_index = questions_for(item, kind)
                if item.n_options < 2:
                    # the model's own head calls topk(2) and crashes on a single option, and a
                    # one-option question is not answerable anyway: counted, not scored
                    skipped_single_option += 1
                    continue
                try:
                    probs = probabilities(agent, item.state, qdef, item.n_options, kind)
                except ValueError as exc:
                    message = str(exc)
                    if "fit in head_max_len" in message:
                        skipped_options += 1
                    elif "probabilities for" in message:
                        # the API keys probabilities by the *rendered* option text and truncates
                        # each option to 48 tokens, so long options can collapse onto one key;
                        # the item cannot be expressed without silently relabelling options
                        skipped_collision += 1
                    else:
                        skipped_error += 1
                        if skipped_error <= args.max_error_traces:
                            print(f"  skip {item.item_id}: {type(exc).__name__}: {exc}",
                                  file=sys.stderr)
                    continue
                except RuntimeError as exc:
                    # a model-internal failure (e.g. its head's topk on a degenerate count) must
                    # not abort the cell; it is counted with its message
                    skipped_error += 1
                    if skipped_error <= args.max_error_traces:
                        print(f"  skip {item.item_id}: RuntimeError: {str(exc)[:120]}",
                              file=sys.stderr)
                    continue
                argmax = int(np.argmax(probs))
                scored.append({
                    "item_id": item.item_id, "template_id": template_id, "source": item.source,
                    "qtype": kind, "n_options": item.n_options, "probs": probs,
                    "argmax": argmax, "gold_index": gold_index,
                    "correct": argmax == gold_index,
                    "label_mass": float(sum(probs)),
                })
            if not scored:
                report["cells"].append({
                    "model_id": MODELS[args.model], "tier": tier, "template_id": template_id,
                    "qtype": kind, "n": 0, "template_total": len(template_items),
                    "status": "NOT MEASURED — no item could be expressed for this model "
                              f"({skipped_options} options did not fit, "
                              f"{skipped_collision} options collapsed to one key, "
                              f"{skipped_single_option} single-option, "
                              f"{skipped_error} other errors)",
                })
                continue
            correct = [s["correct"] for s in scored]
            gold = [int(s["gold_index"]) for s in scored]
            probs_matrix = np.asarray([s["probs"] for s in scored], dtype=np.float64)
            accuracy = float(np.mean(correct))
            _, lo, hi = bootstrap_ci([1.0 if c else 0.0 for c in correct], n_resamples=1000,
                                     seed=args.seed)
            gold_prob = probs_matrix[np.arange(len(scored)), np.asarray(gold)]
            flip = None
            if args.shuffle_items > 0 and kind == "choice":
                flip = shuffle_flip(agent, template_items, args.shuffle_items, args.seed)
            report["cells"].append({
                "model_id": MODELS[args.model], "tier": tier, "template_id": template_id,
                "qtype": kind, "source": scored[0]["source"], "n": len(scored),
                "template_total": len(template_items),
                "coverage": len(scored) / len(template_items),
                "n_skipped_options_too_long": skipped_options,
                "n_skipped_option_collision": skipped_collision,
                "n_skipped_single_option": skipped_single_option,
                "n_skipped_other_error": skipped_error,
                "n_options": scored[0]["n_options"],
                "chance": 0.5 if kind == "noul" else 1.0 / scored[0]["n_options"],
                "accuracy": accuracy, "accuracy_ci95": [lo, hi],
                "n_correct": int(sum(correct)),
                "majority_share": max(Counter(str(g) for g in gold).values()) / len(scored),
                "macro_accuracy": macro_accuracy([str(s["argmax"]) for s in scored],
                                                 [str(g) for g in gold]),
                "brier": float(np.mean((gold_prob - np.asarray(correct, dtype=float)) ** 2)),
                "ece": ece(probs_matrix, gold),
                "mean_gold_prob": float(gold_prob.mean()),
                "median_label_mass": float(np.median([s["label_mass"] for s in scored])),
                "shuffle_flip_rate": flip,
                "status": "measured",
            })
            cell = report["cells"][-1]
            print(f"{MODELS[args.model]} {tier}/{template_id} ({kind}): "
                  f"n={cell['n']}/{cell['template_total']} acc={accuracy:.4f} "
                  f"brier={cell['brier']:.4f} ece={cell['ece']:.4f} flip={flip}", flush=True)
            with preds_path.open("a", encoding="utf-8") as fh:
                for s in scored:
                    fh.write(json.dumps(s, ensure_ascii=False) + "\n")
            del scored
            gc.collect()

    report["wall_seconds"] = time.time() - t0
    (args.out / f"{args.model}.json").write_text(json.dumps(report, indent=2) + "\n")
    prov = Provenance(
        run_name=f"F7_{args.model}", command=" ".join([sys.executable, *sys.argv]),
        gpu=gpu_name(), seed=args.seed,
        config={"model": MODELS[args.model], "shuffle_items": args.shuffle_items},
    )
    prov.datasets = [{"id": MODELS[args.model], "revision": None}]
    prov.finish().write(args.out / f"{args.model}_provenance.json")
    print(f"wall {report['wall_seconds']:.0f}s, {len(report['cells'])} cells")
    return 0


def shuffle_flip(agent: Any, items: list[Item], n_items: int, seed: int) -> float | None:
    rng = random.Random(f"{seed}:layaflip")
    idx = rng.sample(range(len(items)), min(n_items, len(items)))
    flips = total = 0
    for i in idx:
        item = items[i]
        original = pick(agent, item)
        shuffled, record = shuffle_options(item, seed=seed, salt=item.item_id)
        moved = pick(agent, shuffled)
        perm = [int(x) for x in record.details["perm"]]
        if original is None or moved is None:
            continue
        flips += int(perm[moved] != original)
        total += 1
    return flips / total if total else None


def pick(agent: Any, item: Item) -> int | None:
    qdef, _gold = questions_for(item, "choice")
    try:
        probs = probabilities(agent, item.state, qdef, item.n_options, "choice")
    except ValueError:
        return None
    return int(np.argmax(probs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="laya")
    parser.add_argument("--tier1", type=Path, default=Path("data/bench/v0.1/tier1"))
    parser.add_argument("--fresh", type=Path, default=Path("data/bench/v0.1/fresh"))
    parser.add_argument("--keep-screen", type=Path, default=Path("data/bench/v0.1/screen.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/bench_v0_fix0/F7"))
    parser.add_argument("--only-template", default=None)
    parser.add_argument("--limit-per-template", type=int, default=None)
    parser.add_argument("--shuffle-items", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-error-traces", type=int, default=3,
                        help="print this many per-cell error messages so failures are diagnosable")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
