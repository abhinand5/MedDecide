"""osler_v5 Z10 re-derivation (CPU; R6 self-audit; no output written).

Recomputes the Z10 figures from the raw prediction files in a fresh process and checks them against outputs/osler_v5/Z10/report.json:
1. every prediction's item, gold label and option-key order against its Y9 item, and its scored probabilities normalised;
2. the item micro and the template macro of each set, with the report's own definitions (meddecide.eval.o2_metrics.score_row);
3. the knowledge pairing against osler_v4's pplx file, joined on mix_row_to_item's id (as in z10_report.py);
4. the chance level of every cell (1 / option count, as cell_health uses it), printed beside the cells that sit within 0.02 of it.
Exits non-zero if an item is misaligned or a figure differs from report.json by more than the report's rounding.

Usage (CPU):
    uv run --frozen python scripts/osler_v5/z10_rederive.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from meddecide.eval.o2_metrics import score_row
from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.io import iter_jsonl

REPO = Path(__file__).resolve().parents[2]
Y9_DIR = REPO / "data" / "bench" / "osler_v4" / "y9"
Z10 = REPO / "outputs" / "osler_v5" / "Z10"
PPLX_KNOWLEDGE = REPO / "outputs" / "osler_v4" / "Y9" / "baselines" / "pplx_knowledge" / "predictions.jsonl"
REPORT = Z10 / "report.json"
SETS = ("mr_verdict", "mr_reported", "mimic_suite", "mimic_heldout", "mednli_test", "panel", "knowledge")
MODELS = {"qwen35_4b": Z10 / "zeroshot" / "qwen35_4b", "qwen35_9b": Z10 / "zeroshot" / "qwen35_9b", "clef_flash": Z10 / "clef_flash"}
TOLERANCE = 1e-4  # report.json rounds to 4 decimals, so the largest honest difference is 5e-5


def misaligned(model_dir: Path, set_name: str) -> int:
    """Predictions whose item, gold label, option-key order or scored probability sum differs from the Y9 item (0 means aligned)."""
    y9 = {str(r["item_id"]): r for r in iter_jsonl(Y9_DIR / f"{set_name}.jsonl")}
    bad = 0
    for row in iter_jsonl(model_dir / f"{set_name}.jsonl"):
        ref = y9.get(str(row["item_id"]))
        keys_differ = ref is not None and [str(k) for k in row["option_keys"]] != [str(o["key"]) for o in ref["options"]]
        probs_off = row["status"] == "scored" and abs(sum(row["probs"]) - 1.0) > 0.01
        if ref is None or str(row["gold"]) != str(ref["gold"]) or keys_differ or probs_off:
            bad += 1
    return bad


def figures(model_dir: Path, set_name: str) -> tuple[float, float, int, list]:
    """Item micro, template macro, the scored count, and the scored items (the report's definitions)."""
    scored = [score_row(row) for row in iter_jsonl(model_dir / f"{set_name}.jsonl") if row["status"] == "scored"]
    per: dict[str, list[float]] = defaultdict(list)
    for item in scored:
        per[item.template_id].append(float(item.correct))
    micro = float(np.mean([float(item.correct) for item in scored]))
    macro = float(np.mean([np.mean(values) for values in per.values()]))
    return micro, macro, len(scored), scored


def knowledge_pairing(model_dir: Path) -> tuple[int, int, float]:
    """Paired items, unpaired items, and the unstratified paired difference (model minus pplx), joined on the osler_v4 id."""
    join = {str(row["item_id"]): mix_row_to_item(row).item_id for row in iter_jsonl(Y9_DIR / "knowledge.jsonl")}
    pplx = {str(row["item_id"]): bool(row["correct"]) for row in iter_jsonl(PPLX_KNOWLEDGE)}
    model = {join[str(row["item_id"])]: float(score_row(row).correct)
             for row in iter_jsonl(model_dir / "knowledge.jsonl") if row["status"] == "scored"}
    common = [key for key in model if key in pplx]
    diff = float(np.mean([model[key] - float(pplx[key]) for key in common]))
    return len(common), len(model) - len(common), diff


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    by_slug = {model["slug"]: model for model in report["models"]}
    status = 0
    worst = 0.0
    for slug, model_dir in MODELS.items():
        for set_name in SETS:
            bad = misaligned(model_dir, set_name)
            if bad:
                print(f"MISALIGNED {slug} {set_name}: {bad} predictions differ from their Y9 item")
                status = 1
            micro, macro, n, scored = figures(model_dir, set_name)
            block = by_slug[slug]["sets"][set_name]
            worst = max(worst, abs(micro - block["accuracy"]), abs(macro - block["macro_accuracy"]))
            print(f"{slug:12s} {set_name:14s} micro {micro:.4f} macro {macro:.4f} n {n:5d}  (report {block['accuracy']:.4f} / "
                  f"{block['macro_accuracy']:.4f})")
            cells: dict[str, list] = defaultdict(list)
            for item in scored:
                cells[item.template_id].append(item)
            for template, items in sorted(cells.items()):
                options = max(2, round(float(np.mean([len(item.option_keys) for item in items]))))
                chance = 1.0 / options
                accuracy = float(np.mean([float(item.correct) for item in items]))
                if accuracy < chance + 0.02:
                    print(f"  near or below chance: {template} n {len(items)} options {options} chance {chance:.4f} "
                          f"accuracy {accuracy:.4f} margin {accuracy - chance:+.4f}")
        paired, unpaired, diff = knowledge_pairing(model_dir)
        block = by_slug[slug]["sets"]["knowledge"]["paired_with_pplx"]
        worst = max(worst, abs(diff - block["difference_point"]))
        print(f"{slug:12s} knowledge paired n {paired} unpaired {unpaired} difference {diff:+.4f} (report {block['difference_point']:+.4f})")
    print(f"largest difference from report.json over the figures: {worst:.2e} (tolerance {TOLERANCE:.0e})")
    if worst > TOLERANCE:
        print("FAIL: a re-derived figure differs from report.json")
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
