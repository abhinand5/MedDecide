"""osler_v5 Z9 (ADVISORY §10 Z9; GPU, dev only; additional, labelled; choose nothing).

The osler_v4 final (Y8, 63,304 steps) is scored in two settings, both with its trained LoRA adapter:
- base_readout: the readout at step 0 (pplx's base readout, R_base). This is C035's part (a), measured in Z1 without temperatures.
- trained_readout: the final as it is (alpha = 1, C032's last row).

Calibration: one temperature per question type, fitted by NLL on fast dev v4 for each setting (meddecide.train.temperature.fit_per_qtype,
the fit osler_v4 used for Y8's saved temperatures), applied to every dev set of that setting. A temperature does not change any argmax,
so the accuracy figures do not depend on it; the script checks that per item. The Brier scores do depend on it. Z1 fitted no
temperatures, although ADVISORY §10 says "calibrated as in the Z1 probes"; loops/osler_v5/z9_z10.md records that reading.

Per setting, on fast dev v4, CG v2 (its DDXPlus items under the original order and two seeded permutations) and knowledge dev v2: S and
its parts, CG v2 micro, knowledge accuracy and the paired difference against zero-shot pplx (the Z1 alpha 0 rows, the same 4,000 items),
positional health in all three orders, DDXPlus two-letter and single-letter accuracy, the D21 cell checks on CG v2, and Brier.
Nothing is written to a test split, the panel, the MIMIC audit suite or a held-out file.

Usage (GPU; one job; detached; about 1 h):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v5/z9_readout_probe.py \
        > outputs/osler_v5/Z9/logs/probe.log 2>&1 < /dev/null &
A smoke run passes --limit N (the first N items of each set; outputs/osler_v5/Z9/smoke/; no pairing).
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from meddecide.eval.composite import composite, permutation_for, positional_health
from meddecide.eval.o2_metrics import cell_health
from meddecide.model.blend import Blend
from meddecide.model.meddecide_model import reorder_options
from meddecide.model.pplx import PplxDecisionModel
from meddecide.train.blocks import micro_difference, pair_by_item
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.train.temperature import fit_per_qtype
from meddecide.utils.io import iter_jsonl, write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO / "outputs" / "osler_v4" / "Y8" / "checkpoints" / "best"
CONFIG = REPO / "configs" / "osler_v5" / "arm_R0.yaml"
DEV = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"
CG = REPO / "data" / "bench" / "osler_v4" / "cg_v2.jsonl"
KNOW = REPO / "data" / "bench" / "osler_v2" / "knowledge_dev_v2.jsonl"
ZERO_SHOT_KNOWLEDGE = REPO / "outputs" / "osler_v5" / "Z1" / "alpha_0" / "knowledge_dev.jsonl"
OUT = REPO / "outputs" / "osler_v5" / "Z9"
SMOKE_OUT = OUT / "smoke"
DDX_PREFIX = "cg_v2_ddxplus"
POSITIONAL_SEEDS = (1, 2)
CELL_MODEL_ID = "osler_v4_final"


def score(model: PplxDecisionModel, config: Any, items: list) -> Any:
    """Raw ScoredItems with the trainer's evaluation settings (the Z1 path; no temperature)."""
    model.eval_mode()
    return model.score_items(
        items,
        batch_size=config.eval_batch_size or config.batch_size,
        max_batch_tokens=config.eval_max_batch_tokens or config.max_batch_tokens,
        max_prompt_tokens=config.max_prompt_tokens,
        round_to_chunk=config.eval_round_to_chunk,
        length_buckets=config.eval_length_buckets,
    )


def as_o2_rows(scored: Any) -> list[dict[str, Any]]:
    """Scored items in the O2 row shape that cell_health reads. A trained readout takes the D21 checks only (no label mass)."""
    return [
        {"status": "scored", "item_id": str(item.item_id), "template_id": str(item.template_id),
         "probs": [float(p) for p in probs], "option_keys": [str(k) for k in keys], "gold": str(keys[int(gold)])}
        for item, keys, probs, gold in zip(scored.items, scored.option_keys, scored.probs, scored.gold_indices, strict=True)
    ]


def cell_statuses(scored: Any) -> dict[str, dict[str, Any]]:
    """The D21 checks (constant answer; accuracy CI not below chance) for each CG v2 template."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in as_o2_rows(scored):
        groups[row["template_id"]].append(row)
    out: dict[str, dict[str, Any]] = {}
    for template, rows in sorted(groups.items()):
        cell = cell_health(CELL_MODEL_ID, template, rows, zero_shot=False)
        out[template] = {"status": cell["status"], "n": len(rows), "accuracy": cell.get("accuracy")}
    return out


def brier(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Brier per item (sum over options of (p - y)^2): item micro, and macro over templates (the O2 definition)."""
    per_item = [float(sum((p - (1.0 if k == int(r["gold_index"]) else 0.0)) ** 2 for k, p in enumerate(r["probs"]))) for r in rows]
    groups: dict[str, list[float]] = defaultdict(list)
    for r, value in zip(rows, per_item, strict=True):
        groups[str(r["template_id"])].append(value)
    return {"micro": float(np.mean(per_item)), "macro": float(np.mean([np.mean(v) for v in groups.values()]))}


def run_setting(model: PplxDecisionModel, config: Any, name: str, sets: dict[str, list], ddx_index: list[int],
                reference: list[dict[str, Any]] | None, out_dir: Path) -> dict[str, Any]:
    """One setting: score the dev sets and the permuted DDXPlus items, fit the temperatures on fast dev v4, save the rows, summarise."""
    started = time.time()
    raw = {key: score(model, config, items) for key, items in sets.items()}
    ddx_items = [sets["cg_v2"][i] for i in ddx_index]
    raw_perm: dict[str, Any] = {}
    for seed in POSITIONAL_SEEDS:
        permuted = [reorder_options(item, permutation_for(item.n_options, seed=seed, index=i))
                    for item, i in zip(ddx_items, ddx_index, strict=True)]
        raw_perm[str(seed)] = score(model, config, permuted)
    fits = fit_per_qtype(raw["fast_dev_v4"])
    temperatures = {qtype: fit.temperature for qtype, fit in fits.items()}
    calibrated = {key: model.apply_temperature(s, temperature=temperatures) for key, s in raw.items()}
    calibrated_perm = {seed: model.apply_temperature(s, temperature=temperatures) for seed, s in raw_perm.items()}
    pairs = [(key, raw[key], calibrated[key]) for key in raw]
    pairs += [(f"positional seed {seed}", raw_perm[seed], calibrated_perm[seed]) for seed in raw_perm]
    for label, before, after in pairs:
        if not all(int(np.argmax(a)) == int(np.argmax(b)) for a, b in zip(before.probs, after.probs, strict=True)):
            raise AssertionError(f"{name}: calibration changed an argmax in {label}")

    rows = {key: dev_prediction_rows(s) for key, s in calibrated.items()}
    perm_rows = {seed: dev_prediction_rows(s) for seed, s in calibrated_perm.items()}
    uncal_rows = {key: dev_prediction_rows(raw[key]) for key in ("fast_dev_v4", "cg_v2")}

    parts = composite(rows["fast_dev_v4"], rows["cg_v2"])
    ddx_rows = [r for r in rows["cg_v2"] if str(r["template_id"]).startswith(DDX_PREFIX)]
    positional: dict[str, dict[str, Any]] = {"original": positional_health(ddx_rows).to_dict()}
    positional.update({f"seed{seed}": positional_health(pred).to_dict() for seed, pred in perm_rows.items()})
    cg_by_template: dict[str, list[bool]] = defaultdict(list)
    for r in rows["cg_v2"]:
        cg_by_template[str(r["template_id"])].append(bool(r["correct"]))
    know_correct = [bool(r["correct"]) for r in rows["knowledge_dev_v2"]]

    summary: dict[str, Any] = {
        "temperatures": {q: fit.to_dict() for q, fit in fits.items()},
        "composite": parts.to_dict(),
        "knowledge_dev_accuracy": float(np.mean(know_correct)),
        "knowledge_dev_items": len(know_correct),
        "positional": positional,
        "positional_passes_all_orders": all(bool(h["passes"]) for h in positional.values()),
        "ddx_two_letter_accuracy": positional["original"]["two_letter_accuracy"],
        "ddx_single_letter_accuracy": positional["original"]["single_letter_accuracy"],
        "cg_micro_per_template": {t: {"accuracy": float(sum(v) / len(v)), "items": len(v)} for t, v in sorted(cg_by_template.items())},
        "brier_calibrated": {"fast_dev_v4": brier(rows["fast_dev_v4"]), "cg_v2": brier(rows["cg_v2"])},
        "brier_uncalibrated": {"fast_dev_v4": brier(uncal_rows["fast_dev_v4"]), "cg_v2": brier(uncal_rows["cg_v2"])},
        "d21_cells_cg_v2": cell_statuses(calibrated["cg_v2"]),
    }
    if reference is not None:
        diff = micro_difference(pair_by_item(rows["knowledge_dev_v2"], reference))
        summary["knowledge_vs_zero_shot"] = {"point": float(diff.point), "lo": float(diff.lo), "hi": float(diff.hi),
                                             "items": len(know_correct), "reference": "Z1 alpha 0 (zero-shot pplx)"}

    folder = out_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    write_jsonl(folder / "fast_dev.jsonl", rows["fast_dev_v4"])
    write_jsonl(folder / "cg_v2.jsonl", rows["cg_v2"])
    write_jsonl(folder / "knowledge_dev.jsonl", rows["knowledge_dev_v2"])
    for seed, pred in perm_rows.items():
        write_jsonl(folder / f"positional_seed{seed}.jsonl", pred)
    summary["seconds"] = round(time.time() - started, 1)
    print(f"{name}: S {parts.s:.4f} (MR {parts.mr:.4f}, MIMIC {parts.mimic:.4f}, CG micro {parts.cg_micro:.4f}); "
          f"knowledge {summary['knowledge_dev_accuracy']:.4f}; DDX two-letter {summary['ddx_two_letter_accuracy']:.4f}, "
          f"single {summary['ddx_single_letter_accuracy']:.4f}; positional passes {[h['passes'] for h in positional.values()]}; "
          f"temperatures {temperatures}; {summary['seconds']} s", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=0, help="score only the first N items of each set (smoke run)")
    args = parser.parse_args()
    out_dir = SMOKE_OUT if args.limit else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    config = dataclasses.replace(load_config(CONFIG), device="cuda:0", dtype="bfloat16", backbone="pplx")
    model = PplxDecisionModel.load(CHECKPOINT, device="cuda:0", dtype="bfloat16")
    model.eval_mode()
    sets = {
        "fast_dev_v4": [mix_row_to_item(r) for r in iter_jsonl(DEV)],
        "cg_v2": [mix_row_to_item(r) for r in iter_jsonl(CG)],
        "knowledge_dev_v2": [mix_row_to_item(r) for r in iter_jsonl(KNOW)],
    }
    if args.limit:
        sets = {key: items[: args.limit] for key, items in sets.items()}
    ddx_index = [i for i, item in enumerate(sets["cg_v2"]) if str(item.template_id).startswith(DDX_PREFIX)]
    reference = None if args.limit else list(iter_jsonl(ZERO_SHOT_KNOWLEDGE))
    blend = Blend(model)
    results: dict[str, Any] = {
        "started_utc": started, "git_commit": git_commit(REPO), "checkpoint": str(CHECKPOINT.relative_to(REPO)),
        "dtype": "bfloat16", "device": "cuda:0", "limit": args.limit, "ddx_items": len(ddx_index),
        "items": {key: len(items) for key, items in sets.items()}, "settings": {},
    }
    # base readout: the trained adapter with the readout at step 0 (C035 part (a))
    blend.apply(1.0)
    with torch.no_grad():
        model.optcode.weight.copy_(blend.base_readout.to(model.optcode.weight.device))
    results["settings"]["base_readout"] = run_setting(model, config, "base_readout", sets, ddx_index, reference, out_dir)
    write_json(out_dir / "summary.json", results)
    # trained readout: the final as it is (alpha = 1)
    blend.apply(1.0)
    results["settings"]["trained_readout"] = run_setting(model, config, "trained_readout", sets, ddx_index, reference, out_dir)
    results["finished_utc"] = utcnow()
    write_json(out_dir / "summary.json", results)
    print("done", flush=True)


if __name__ == "__main__":
    main()
