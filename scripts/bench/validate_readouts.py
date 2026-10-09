#!/usr/bin/env python
"""F2 validation: reproduce the `choice` path, and validate the `noul` readout against a
reference rule on prompt-identical items.

Three sections, all in one GPU session:

**(a) `choice` reproduction.** F2 must not change the path that loop bench_v0 validated against
lm-evaluation-harness (C041). The script checks that canonicalisation is a no-op for `choice`
items (byte-identical prompts) and re-runs MedQA, comparing with bench_v0's recorded accuracy for
the same model.

**(b) `noul` on relevance items.** The same letter readout applied to yes/no items, against
lm-eval's continuation rule on the *same prompt text*. Reported as-is, including any gap: these
prompts put a relevance question to the model, which is not a natural yes/no task.

**(c) lm-eval's own `pubmedqa` task.** The clean reference: lm-eval's prompt and its own choice
set (`yes`/`no`/`maybe`), scored by our letter readout and by lm-eval's continuation rule. This is
the comparison the ADVISORY ±2-point criterion is about.

Writes ``outputs/bench_v0_fix0/F2/reference_validation_<model>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from meddecide.bench.schema import Item, QuestionType
from meddecide.eval.harness import Harness, ModelSpec
from meddecide.eval.readout import canonicalise_options, render_prompt
from meddecide.utils.io import read_jsonl
from meddecide.utils.provenance import Provenance, gpu_name, utcnow

# bench_v0 MedQA test accuracies for the models it ran (C028 for 0.8B, C049 for 4B).
V0_MEDQA_ACCURACY = {
    "Qwen/Qwen3.5-0.8B": 0.40770,
    "Qwen/Qwen3.5-4B": 0.70149,
}
PUBMEDQA_TEMPLATE = "Abstract: {contexts}\nQuestion: {question}\nAnswer:"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def continuation_totals(
    model: Any, tokenizer: Any, prompt: str, choices: list[str]
) -> list[float]:
    """Summed log-probability of each choice as a space-separated continuation (lm-eval's rule)."""
    totals: list[float] = []
    for choice in choices:
        encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)
        encoded = tokenizer.encode(prompt + " " + choice, add_special_tokens=False)
        target_ids = encoded[len(encoded_prompt) :]
        inputs = torch.tensor([encoded], device=model.device)
        with torch.inference_mode():
            logits = model(inputs).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        total = 0.0
        for offset, token_id in enumerate(target_ids):
            position = len(encoded) - len(target_ids) + offset - 1
            total += float(logprobs[position, token_id])
        totals.append(total)
    return totals


def _medqa_choice_section(args: argparse.Namespace, harness: Harness) -> dict[str, Any]:
    v0_medqa, _ = read_jsonl(args.v0_tier1 / "medqa.jsonl", Item)
    v0_test = [i for i in v0_medqa if str(i.split) == "test"]
    v01_medqa, _ = read_jsonl(args.tier1 / "medqa.jsonl", Item)
    v01_test = [i for i in v01_medqa if str(i.split) == "test"]

    changed = 0
    for item in v01_test:
        canonical, keys = canonicalise_options(item)
        if canonical.option_keys != keys or canonical.item_id != item.item_id:
            changed += 1

    predictions = harness.score(v01_test)
    accuracy = sum(1 for p in predictions if p.correct) / len(predictions)
    reference = V0_MEDQA_ACCURACY.get(args.model)
    return {
        "n_v0_test": len(v0_test),
        "n_v0_1_test": len(v01_test),
        "same_item_ids": sorted(i.item_id for i in v0_test) == sorted(i.item_id for i in v01_test),
        "n_choice_items_changed_by_canonicalisation": changed,
        "canonicalisation_is_noop_for_choice": changed == 0,
        "n": len(predictions),
        "n_correct": sum(1 for p in predictions if p.correct),
        "accuracy_f2": accuracy,
        "v0_accuracy_reference": reference,
        "delta_vs_v0_pts": (accuracy - reference) * 100 if reference is not None else None,
        "prompt_sha256_first_item": _sha(
            render_prompt(canonicalise_options(v01_test[0])[0], harness.tokenizer, harness.variant)
        ),
    }


def _noul_section(args: argparse.Namespace, harness: Harness) -> dict[str, Any]:
    """Letter readout vs continuation rule on prompt-identical yes/no items."""
    items: list[Item] = []
    for path in sorted(args.tier1.glob("*.jsonl")):
        rows, _ = read_jsonl(path, Item)
        items.extend(i for i in rows if i.qtype is QuestionType.NOUL and str(i.split) == "test")
    rng = random.Random(f"{args.seed}:noul")
    rng.shuffle(items)
    items = items[: args.noul_items]
    if not items:
        return {"n_items": 0, "status": "NOT MEASURED — no noul items"}

    canonical = [canonicalise_options(i)[0] for i in items]
    prompts = [render_prompt(i, harness.tokenizer, harness.variant) for i in canonical]
    predictions = harness.score_prompts(
        prompts,
        [i.option_keys for i in canonical],
        [i.gold for i in canonical],
        qtypes=["noul"] * len(prompts),
        sources=[i.source for i in items],
        template_ids=["noul_reference"] * len(prompts),
        item_ids=[i.item_id for i in items],
        original_keys=[["yes", "no"]] * len(prompts),
    )
    ours_accuracy = sum(1 for p in predictions if p.correct) / len(predictions)
    gold_original = ["yes" if i.gold == "yes" else "no" for i in items]

    correct_cont = 0
    agree = 0
    margins: list[float] = []
    for prompt, gold, pred in zip(prompts, gold_original, predictions, strict=True):
        totals = continuation_totals(harness.model, harness.tokenizer, prompt, ["yes", "no"])
        pick = "yes" if totals[0] > totals[1] else "no"
        readout_pick = "yes" if pred.option_keys[pred.argmax_index] == "A" else "no"
        correct_cont += int(pick == gold)
        agree += int(pick == readout_pick)
        margins.append(abs(totals[0] - totals[1]))
    n = len(prompts)
    return {
        "n_items": n,
        "sources": sorted({i.source for i in items}),
        "prompt_kind": "harness chat template + single-letter instruction (identical on both sides)",
        "ours_accuracy": ours_accuracy,
        "ours_n_correct": sum(1 for p in predictions if p.correct),
        "lm_eval_rule_accuracy": correct_cont / n,
        "lm_eval_rule_n_correct": correct_cont,
        "per_item_agreement": agree / n,
        "agreement_points": (ours_accuracy - correct_cont / n) * 100,
        "median_label_mass": float(np.median([p.label_mass for p in predictions])),
        "min_label_mass": float(np.min([p.label_mass for p in predictions])),
        "mean_abs_logprob_margin": float(np.mean(margins)),
        "within_2_points": abs((ours_accuracy - correct_cont / n) * 100) <= 2.0,
    }


def _pubmedqa_section(args: argparse.Namespace, harness: Harness) -> dict[str, Any]:
    """lm-eval's own pubmedqa task: its prompt, its three choices, both scoring rules."""
    from datasets import load_dataset

    from meddecide.bench.schema import Tier, make_item

    # lm-eval's `pubmedqa` task points at bigbio/pubmed_qa, which is script-based and no longer
    # loadable by `datasets` 5.x ("Dataset scripts are no longer supported"). This uses the
    # parquet release of the same expert-labelled corpus (qiaojin/PubMedQA pqa_labeled == the
    # standard 1,000-item PubMedQA labelled set) and rebuilds lm-eval's prompt shape
    # ("Abstract: <contexts>\nQuestion: <question>\nAnswer:") from its fields, then scores both
    # rules on that identical prompt. The difference is recorded in the report.
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rows = list(ds)[: args.pubmedqa_items]
    prompts = [
        PUBMEDQA_TEMPLATE.format(
            contexts="\n".join(r["context"]["contexts"]), question=r["question"]
        )
        for r in rows
    ]
    gold = [str(r["final_decision"]).strip().lower() for r in rows]
    letter = {"yes": "A", "no": "B", "maybe": "C"}

    chat_prompts: list[str] = []
    for prompt, row in zip(prompts, rows, strict=True):
        probe = make_item(
            tier=Tier.ESTABLISHED,
            source="pubmedqa_reference",
            source_record_id=str(row["pubid"]),
            source_url="https://pubmed.ncbi.nlm.nih.gov/",
            source_license="mit",
            record_date="2019-01-01",
            split="test",
            template_id="pubmedqa_reference",
            skill="evidence",
            qtype=QuestionType.CHOICE,
            state=prompt,
            # the lm-eval prompt already ends with "Answer:"; this question stays empty of
            # instruction because the reference task's prompt shape is fixed above
            question="What does this study conclude?",
            options=[{"key": "A", "label": "yes"}, {"key": "B", "label": "no"},
                     {"key": "C", "label": "maybe"}],
            gold=letter[str(row["final_decision"]).strip().lower()],
            option_order_seed=0,
        )
        chat_prompts.append(render_prompt(probe, harness.tokenizer, harness.variant))

    predictions = harness.score_prompts(
        chat_prompts,
        [["A", "B", "C"]] * len(rows),
        [letter[g] for g in gold],
        qtypes=["noul"] * len(rows),
        sources=["pubmedqa"] * len(rows),
        template_ids=["pubmedqa_reference"] * len(rows),
        item_ids=[f"pmid-{r['pubid']}" for r in rows],
        original_keys=[["yes", "no", "maybe"]] * len(rows),
    )
    ours_correct = sum(1 for p in predictions if p.correct)

    cont_correct = 0
    agree = 0
    for prompt, gold_value, pred in zip(prompts, gold, predictions, strict=True):
        totals = continuation_totals(
            harness.model, harness.tokenizer, prompt, ["yes", "no", "maybe"]
        )
        pick = ("yes", "no", "maybe")[int(np.argmax(totals))]
        cont_correct += int(pick == gold_value)
        agree += int(pick == pred.original_option_keys[pred.argmax_index])
    n = len(rows)
    return {
        "status": "measured",
        "task": "lm-eval's `pubmedqa` prompt shape over qiaojin/PubMedQA pqa_labeled "
                "(bigbio/pubmed_qa is script-based and unloadable on datasets 5.x)",
        "n": n,
        "ours_accuracy": ours_correct / n,
        "lm_eval_rule_accuracy": cont_correct / n,
        "per_item_agreement": agree / n,
        "agreement_points": (ours_correct - cont_correct) / n * 100,
        "median_label_mass": float(np.median([p.label_mass for p in predictions])),
        "within_2_points": abs((ours_correct - cont_correct) / n * 100) <= 2.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--tier1", type=Path, default=Path("data/bench/v0.1/tier1"))
    parser.add_argument("--v0-tier1", type=Path, default=Path("data/bench/tier1"))
    parser.add_argument("--noul-items", type=int, default=200)
    parser.add_argument("--pubmedqa-items", type=int, default=477)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-continuation", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    harness = Harness(ModelSpec(model_id=args.model), batch_size=args.batch_size)
    report: dict[str, Any] = {
        "generated_at_utc": utcnow(),
        "model": args.model,
        "variant_detection": harness.variant_detection,
    }

    report["choice_reproduction"] = _medqa_choice_section(args, harness)
    choice = report["choice_reproduction"]
    print(f"choice: acc={choice['accuracy_f2']:.5f} delta={choice['delta_vs_v0_pts']} pts "
          f"noop={choice['canonicalisation_is_noop_for_choice']}", flush=True)

    if args.skip_continuation:
        report["noul_reference"] = {"status": "NOT MEASURED — --skip-continuation"}
        report["pubmedqa_reference"] = {"status": "NOT MEASURED — --skip-continuation"}
    else:
        report["noul_reference"] = _noul_section(args, harness)
        section = report["noul_reference"]
        print(f"noul: ours={section.get('ours_accuracy')} cont={section.get('lm_eval_rule_accuracy')} "
              f"agree={section.get('per_item_agreement')} mass={section.get('median_label_mass')}",
              flush=True)
        report["pubmedqa_reference"] = _pubmedqa_section(args, harness)
        section = report["pubmedqa_reference"]
        print(f"pubmedqa: ours={section.get('ours_accuracy')} cont={section.get('lm_eval_rule_accuracy')} "
              f"agree={section.get('per_item_agreement')} mass={section.get('median_label_mass')}",
              flush=True)

    checks = {
        "choice_unchanged": bool(
            report["choice_reproduction"]["canonicalisation_is_noop_for_choice"]
        ),
        "choice_item_ids_identical": bool(report["choice_reproduction"]["same_item_ids"]),
        "choice_reproduced_within_2pts": (
            report["choice_reproduction"]["delta_vs_v0_pts"] is not None
            and abs(report["choice_reproduction"]["delta_vs_v0_pts"]) <= 2.0
        ),
        "noul_reference_measured": report["noul_reference"].get("n_items", 0) > 0,
        "noul_reference_within_2pts": report["noul_reference"].get("within_2_points"),
        "pubmedqa_reference_within_2pts": report.get("pubmedqa_reference", {}).get(
            "within_2_points"
        ),
    }
    report["checks"] = checks
    required = [v for v in checks.values() if v is not None]
    report["verdict"] = "PASS" if required and all(required) else "FAIL"

    out = args.out or Path(
        f"outputs/bench_v0_fix0/F2/reference_validation_{args.model.split('/')[-1].lower()}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    prov = Provenance(
        run_name="F2_reference_validation",
        command=" ".join([sys.executable, *sys.argv]),
        gpu=gpu_name(),
        seed=args.seed,
        config={"model": args.model, "noul_items": args.noul_items,
                "pubmedqa_items": args.pubmedqa_items},
    )
    prov.finish().write(out.with_name(out.stem + "_provenance.json"))
    print(json.dumps({"checks": checks, "verdict": report["verdict"]}, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
