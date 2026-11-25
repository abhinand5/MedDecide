"""osler_v5 Z4 (ADVISORY Z4; GPU, dev only): the blend toward the base on the selected checkpoint, then the export.

For alpha in {0 (the reference), 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}: the blend is set (meddecide.model.blend.Blend); the dev sets are
scored once (fast dev v4 for the MR and MIMIC template macros, the calibration and the Brier score; CG v2 for the micro accuracy; the
DDXPlus items under the original order and two seeded permutations for positional health); the temperatures are refitted on fast dev
v4 for that alpha (the same fit as Trainer.fit_calibration, from the same scoring pass); the D12 readout gate is read per template on
fast dev v4. The knowledge set is not scored here: the Z4 rule does not use it.

The choice is meddecide.eval.blend_choice.choose_alpha (pre-registered). The chosen alpha is exported as a standard checkpoint: the LoRA
B matrices are multiplied by alpha (equal to the adapter scaling by alpha), the readout is (1 - alpha) R_base + alpha R_trained, and
the temperatures of that alpha are saved. The export is reloaded and re-scored on fast dev v4; its calibrated probabilities must equal
the in-memory ones (outputs/osler_v5/Z4/blend.json).

Usage (GPU; one job; detached; about 3 h):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v5/z4_blend.py \
        --checkpoint outputs/osler_v5/Z3/checkpoints/step_XXXXXX --config configs/osler_v5/final.yaml \
        > outputs/osler_v5/Z4/logs/blend.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.bench.schema import Item
from meddecide.eval.blend_choice import AlphaStats, choose_alpha
from meddecide.eval.composite import (
    accuracy,
    composite,
    mimic_partition,
    permutation_for,
    positional_health,
    template_macro_accuracy,
)
from meddecide.eval.readout_gate import gate_by_template
from meddecide.model.blend import Blend, export_blend
from meddecide.model.meddecide_model import reorder_options
from meddecide.model.pplx import PplxDecisionModel
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.train.temperature import fit_per_qtype, fit_temperature
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"
CG = REPO / "data" / "bench" / "osler_v4" / "cg_v2.jsonl"
OUT = REPO / "outputs" / "osler_v5" / "Z4"
ALPHAS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
POSITIONAL_SEEDS = (1, 2)
DDX_PREFIX = "cg_v2_ddxplus"


def read_items(path: Path) -> list[Item]:
    with path.open(encoding="utf-8") as fh:
        return [mix_row_to_item(json.loads(line)) for line in fh if line.strip()]


def brier(rows: list[dict[str, Any]]) -> float:
    """Item mean of the multi-class Brier score over the options each item offers (D31)."""
    values = []
    for row in rows:
        probs = np.asarray(row["probs"], dtype=np.float64)
        target = np.zeros_like(probs)
        target[int(row["gold_index"])] = 1.0
        values.append(float(np.sum((probs - target) ** 2)))
    return float(np.mean(values))


def score_all(model: PplxDecisionModel, config: Any, fast: list[Item], cg: list[Item], ddx_index: list[int]) -> dict[str, Any]:
    """Dev rows at the current blend: the calibrated fast dev rows (for Brier and D12), CG rows and the permuted DDXPlus rows."""
    model.eval_mode()

    def score(items: list[Item]) -> Any:
        return model.score_items(
            items, batch_size=config.eval_batch_size or config.batch_size,
            max_batch_tokens=config.eval_max_batch_tokens or config.max_batch_tokens, max_prompt_tokens=config.max_prompt_tokens,
            round_to_chunk=config.eval_round_to_chunk, length_buckets=config.eval_length_buckets)

    raw_fast = score(fast)
    if config.temperature_per_qtype:
        model.calibration = {k: v.temperature for k, v in fit_per_qtype(raw_fast).items()}
    else:
        model.calibration = {"*": fit_temperature(raw_fast.logits, raw_fast.gold_indices, qtype="*").temperature}
    calibrated_fast = dev_prediction_rows(model.apply_temperature(raw_fast))
    cg_rows = dev_prediction_rows(score(cg))
    ddx_items = [cg[i] for i in ddx_index]
    permuted: dict[str, list[dict[str, Any]]] = {}
    for seed in POSITIONAL_SEEDS:
        items = [reorder_options(item, permutation_for(item.n_options, seed=seed, index=i)) for item, i in zip(ddx_items, ddx_index, strict=True)]
        permuted[str(seed)] = dev_prediction_rows(score(items))
    return {"fast": calibrated_fast, "cg": cg_rows, "perm": permuted, "temperatures": dict(model.calibration)}


def alpha_stats(alpha: float, data: dict[str, Any], ddx_rows: list[dict[str, Any]]) -> tuple[AlphaStats, dict[str, Any]]:
    fast = data["fast"]
    mr = [r for r in fast if str(r["source"]) in {"mr_v1", "mr_synthea", "mr_teacher", "mr_mtsamples", "mr_codiesp", "mr_pmc"}]
    mimic = mimic_partition(fast)
    parts = composite(fast, data["cg"])
    health = {"original": positional_health(ddx_rows).to_dict()}
    for seed, rows in data["perm"].items():
        health[f"seed{seed}"] = positional_health(rows).to_dict()
    passes = all(h["passes"] for h in health.values())
    stats = AlphaStats(alpha=alpha, mr_macro=template_macro_accuracy(mr), mimic_macro=template_macro_accuracy(mimic),
                       cg_micro=accuracy(data["cg"]), positional_passes=passes)
    cells = gate_by_template(fast)
    failing = {t: c.failures for t, c in cells.items() if c.failures}
    record = {"alpha": alpha, "composite": parts.to_dict(), "mr_template_macro": stats.mr_macro,
              "mimic_template_macro": stats.mimic_macro, "cg_micro": stats.cg_micro, "positional": health,
              "positional_passes_all_orders": passes, "brier_fast_dev_calibrated": brier(fast),
              "readout_gate_failing_cells": failing, "temperatures": data["temperatures"]}
    return stats, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    config = dataclasses.replace(load_config(REPO / args.config), device="cuda:0", dtype="bfloat16", backbone="pplx")
    checkpoint = REPO / args.checkpoint
    model = PplxDecisionModel.load(checkpoint, device="cuda:0", dtype="bfloat16")
    model.eval_mode()
    blend = Blend(model)
    fast = read_items(DEV)
    cg = read_items(CG)
    ddx_index = [i for i, item in enumerate(cg) if str(item.template_id).startswith(DDX_PREFIX)]
    records: dict[str, Any] = {}
    stats: list[AlphaStats] = []
    cal_probs: dict[float, list[np.ndarray]] = {}
    for alpha in ALPHAS:
        t0 = time.time()
        blend.apply(alpha)
        data = score_all(model, config, fast, cg, ddx_index)
        original_ddx = [r for r in data["cg"] if str(r["template_id"]).startswith(DDX_PREFIX)]
        s, record = alpha_stats(alpha, data, original_ddx)
        record["seconds"] = round(time.time() - t0, 1)
        records[f"{alpha:g}"] = record
        stats.append(s)
        cal_probs[alpha] = [np.asarray(r["probs"]) for r in data["fast"]]
        print(f"alpha {alpha:g}: MR {s.mr_macro:.4f}, MIMIC {s.mimic_macro:.4f}, CG micro {s.cg_micro:.4f}, positional "
              f"{s.positional_passes}, Brier {record['brier_fast_dev_calibrated']:.4f}; {record['seconds']} s", flush=True)
        write_json(OUT / "blend_partial.json", {"records": records})
    decision = choose_alpha(stats)
    chosen = float(decision["chosen_alpha"])
    print(f"choice: alpha {chosen:g} (blend used: {decision['blend_used']})", flush=True)

    # export: the chosen alpha's temperatures, then the model carries the blend (LoRA B scaled, readout blended)
    model.calibration = dict(records[f"{chosen:g}"]["temperatures"])
    export_blend(blend, chosen)
    export = OUT / "final_blended"
    model.save(export)
    reloaded = PplxDecisionModel.load(export, device="cuda:0", dtype="bfloat16")
    reloaded.eval_mode()
    cal_reloaded = reloaded.calibration
    raw = reloaded.score_items(fast, batch_size=config.eval_batch_size or config.batch_size,
                               max_batch_tokens=config.eval_max_batch_tokens or config.max_batch_tokens,
                               max_prompt_tokens=config.max_prompt_tokens, round_to_chunk=config.eval_round_to_chunk,
                               length_buckets=config.eval_length_buckets)
    reloaded_probs = [np.asarray(p) for p in reloaded.apply_temperature(raw).probs]
    max_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(reloaded_probs, cal_probs[chosen], strict=True))
    result = {"started_utc": started, "finished_utc": utcnow(), "git_commit": git_commit(REPO), "checkpoint": args.checkpoint,
              "alphas": records, "decision": decision, "chosen_alpha": chosen, "export": str(export.relative_to(REPO)),
              "export_temperatures": cal_reloaded, "export_reload_max_abs_diff_calibrated": max_diff}
    write_json(OUT / "blend.json", result)
    print(f"export reload: max abs diff of calibrated probabilities {max_diff:.3e}", flush=True)


if __name__ == "__main__":
    main()
