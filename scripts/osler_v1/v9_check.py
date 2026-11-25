"""V9 check (osler_v1): re-derive the Osler-4B final run's headline numbers from its raw artifacts (R1, R6).

Reads outputs/osler_v1/V9/osler_4b/ only and prints aggregates; writes nothing. The numbers printed here are the ones in
CLAIMS V156 to V161. The checkpoint's calibration must equal the arm's dev fits (ADVISORY §8, item 6).

Usage (CPU; no GPU):
    source scripts/pod_env.sh && uv run --frozen python -I scripts/osler_v1/v9_check.py
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from meddecide.eval.metrics import template_macro_accuracy
from meddecide.train.blocks import DEFECT_MIN_ITEMS

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "outputs" / "osler_v1" / "V9" / "osler_4b"
FINAL_STEP = 43136
FLAGGED = (
    "mr_icd_support_template_v2",
    "mr_page_category_tabular_v1",
    "synthea_indicator_evidence_v1",
    "synthea_page_category_v1",
)
PINNED_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PMC_TEMPLATE = "pmc_similar_patient_v1"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    rows = load_jsonl(RUN / "dev_predictions.jsonl")
    n = len(rows)
    correct = [1.0 if r["correct"] else 0.0 for r in rows]
    templates = [r["template_id"] for r in rows]
    n_correct = int(sum(correct))
    print(f"dev items {n}; accuracy {n_correct / n:.6f} ({n_correct} correct)")
    mismatch = sum(r["predicted_key"] != r["gold_key"] for r in rows)
    print(f"consistency: predicted_key != gold_key on {mismatch} items (expected {n - n_correct})")
    print(f"template macro {template_macro_accuracy(correct, templates):.6f} over {len(set(templates))} templates")
    majority = max(Counter(r["gold_key"] for r in rows).values()) / n
    print(f"majority baseline {majority:.6f}")

    brier = 0.0
    nll = 0.0
    for r in rows:
        probs = [float(p) for p in r["probs"]]
        gold = int(r["gold_index"])
        # Multi-class Brier per item: sum over options of (p_k - y_k)^2; NLL is -log p(gold).
        brier += sum((p - (1.0 if k == gold else 0.0)) ** 2 for k, p in enumerate(probs))
        nll += -math.log(max(probs[gold], 1e-12))
    print(f"brier {brier / n:.6f}; mean nll {nll / n:.6f}")

    by_template: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_template[r["template_id"]].append(bool(r["correct"]))
    for name in FLAGGED:
        flags = by_template[name]
        print(f"flagged {name}: items {len(flags)} correct {sum(flags)} accuracy {sum(flags) / len(flags):.4f}")
    pmc = by_template[PMC_TEMPLATE]
    print(f"reference {PMC_TEMPLATE} (no rows in B1): items {len(pmc)} correct {sum(pmc)} "
          f"accuracy {sum(pmc) / len(pmc):.4f}")
    small = [name for name, flags in by_template.items() if len(flags) < DEFECT_MIN_ITEMS]
    screened = len(by_template) - len(small)
    print(f"defect screen: templates {len(by_template)}; below {DEFECT_MIN_ITEMS} items {len(small)}; screened {screened}")

    arm = json.loads((RUN / "arm_result.json").read_text(encoding="utf-8"))
    model = json.loads((RUN / "checkpoints" / "best" / "model.json").read_text(encoding="utf-8"))
    fits = arm["temperature_fits"]
    for qtype, fit in sorted(fits.items()):
        print(f"temperature {qtype}: T {fit['temperature']:.6f}; items {fit['n_items']}; "
              f"nll {fit['nll_before']:.6f} -> {fit['nll_after']:.6f}")
    print(f"temperature items {sum(f['n_items'] for f in fits.values())} (full dev {n})")
    matches = all(abs(model["calibration"][q] - fits[q]["temperature"]) < 1e-12 for q in fits)
    print(f"checkpoint calibration equals the arm's dev fits: {matches}; revision pinned: "
          f"{model['revision'] == PINNED_REVISION}")

    block = json.loads((RUN / "block_result.json").read_text(encoding="utf-8"))
    full = block["full_dev_metrics"]
    print(f"stored full dev: accuracy {full['accuracy']:.6f} correct {full['n_correct']}; "
          f"template macro {full['template_macro_accuracy']:.6f}; brier {full['brier']:.6f}; "
          f"mean nll {full['mean_nll']:.6f}")

    train = load_jsonl(RUN / "logs" / "train.jsonl")
    steps = [r for r in train if r.get("event") == "step"]
    evals = [r for r in train if r.get("event") == "eval"]
    print(f"logged step rows {len(steps)} (last step {steps[-1]['step']}); "
          f"non-finite logged losses {sum(not math.isfinite(r['loss']) for r in steps)}")
    print(f"evaluation steps {[r['step'] for r in evals]}; evaluated items {sorted({r['metrics']['n'] for r in evals})}")
    print(f"evaluation template macro {[round(r['metrics']['template_macro_accuracy'], 4) for r in evals]}")
    print(f"evaluation accuracy {[round(r['metrics']['accuracy'], 4) for r in evals]}")
    final = next(r["metrics"] for r in evals if r["step"] == FINAL_STEP)
    print(f"in-run final evaluation: accuracy {final['accuracy']:.6f} correct {final['n_correct']}; "
          f"template macro {final['template_macro_accuracy']:.6f}; brier {final['brier']:.6f}; "
          f"mean nll {final['mean_nll']:.6f}")
    print(f"post-run minus in-run: correct {full['n_correct'] - final['n_correct']}; "
          f"template macro {full['template_macro_accuracy'] - final['template_macro_accuracy']:+.6f}; "
          f"brier {full['brier'] - final['brier']:+.6f}; mean nll {full['mean_nll'] - final['mean_nll']:+.6f}")

    print(f"run: steps {block['steps']}; realised items {block['realised_items']}; train items {block['train_items']}; "
          f"real tokens {block['real_tokens']}; padded tokens {block['padded_tokens']}")
    print(f"run: train wall clock {block['train_wall_clock_s']} s; total {block['total_wall_clock_s']} s; "
          f"best step {block['best_step']}; logged steps checked {block['validation']['logged_steps']['logged_steps_checked']}; "
          f"mismatches {block['validation']['logged_steps']['mismatches']}")


if __name__ == "__main__":
    main()
