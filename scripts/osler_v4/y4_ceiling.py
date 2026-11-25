"""Y4 ceiling screen (D35; ADVISORY Y4 §3): zero-shot macro accuracy on the MIMIC dev items, per template, with the D12
readout-health gate (meddecide.eval.health.evaluate_cell). Reads the predictions of scripts/osler_v4/y4_zeroshot.py.

A template whose cell fails the gate has macro_accuracy_dev null and its readout status: the ceiling is then not measured,
and the MIMIC audit suite excludes that template (it cannot be shown to be below the ceiling). Writes
outputs/osler_v4/Y4/ceiling_<name>.json (aggregates only).

Usage: uv run --frozen python scripts/osler_v4/y4_ceiling.py --name qwen35_4b_mimic_dev
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.eval.health import evaluate_cell
from meddecide.eval.metrics import macro_accuracy
from meddecide.utils.io import write_json

OUT = Path("outputs/osler_v4/Y4")
PRED_ROOT = OUT / "zeroshot"
MANIFEST = OUT / "mimic_dev_all_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (PRED_ROOT / args.name / "predictions.jsonl").read_text().splitlines() if line.strip()]
    # no silent drops: the screen covers every MIMIC dev item, template by template, and every row is scored
    manifest = json.loads(MANIFEST.read_text())
    not_scored = [row["item_id"] for row in rows if row["status"] != "scored"]
    if not_scored:
        raise SystemExit(f"{len(not_scored)} rows are not scored: the screen is not complete: stop")
    if dict(Counter(row["template_id"] for row in rows)) != manifest["per_template"]:
        raise SystemExit("the zero-shot rows do not match the MIMIC dev manifest per template: stop")
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_template[row["template_id"]].append(row)
    result: dict[str, Any] = {}
    for template, items in sorted(by_template.items()):
        predicted, gold, correct, masses, option_counts = [], [], [], [], []
        for row in items:
            probs = row["probs"]
            keys = row["option_keys"]
            choice = keys[max(range(len(probs)), key=lambda i: probs[i])]
            predicted.append(choice)
            gold.append(row["gold"])
            correct.append(choice == row["gold"])
            masses.append(float(row["extras"]["label_mass"]))
            option_counts.append(len(keys))
        n_options = max(2, round(sum(option_counts) / len(option_counts)))
        majority_share = Counter(gold).most_common(1)[0][1] / len(gold)
        report = evaluate_cell(model_id=args.name, template_id=template, label_masses=masses, correct=correct,
                               greedy_matches=None, n_options=n_options, predicted_labels=predicted,
                               majority_share=majority_share)
        macro = macro_accuracy(predicted, gold)
        result[template] = {
            "items": len(items),
            "macro_accuracy_dev": round(macro, 4) if report.passed else None,
            "macro_accuracy_raw": round(macro, 4),
            "readout_status": report.status,
            "readout_failures": report.failures,
            "accuracy_ci95": list(report.accuracy_ci95) if report.accuracy_ci95 else None,
            "median_label_mass": report.median_label_mass,
            "majority_share": round(majority_share, 4),
        }
    write_json(OUT / f"ceiling_{args.name}.json", result)
    print(json.dumps({t: {k: v[k] for k in ("items", "macro_accuracy_dev", "readout_status")} for t, v in result.items()},
                     indent=2))


if __name__ == "__main__":
    main()
