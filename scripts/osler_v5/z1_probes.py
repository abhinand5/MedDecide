"""osler_v5 Z1 (ADVISORY Z1; GPU, dev only, choose nothing): probes on the osler_v4 final checkpoint.

Settings, each scored on fast dev v4, CG v2, knowledge dev v2 and the DDXPlus items of CG v2 under two seeded permutations:
- the blend curve: alpha = 0 (the reference path, which is also the zero-shot-like reference of D54), 0.3, 0.5, 0.7, 0.85, 1.0;
- the readout split: (a) the trained adapter with the base readout; (b) the base adapter (adapter off) with the trained readout.
The alpha = 0 scores are checked against MedDecideModel.reference_option_probs on 16 CG items (bf16, the GPU).

Per setting: S and its parts, knowledge dev accuracy, positional health (original order and the two permutations), CG v2 micro per
template, and the DDXPlus two-letter accuracy. The per-item rows are kept under outputs/osler_v5/Z1/<setting>/. Temperatures are not fitted
here (accuracy only). Nothing is written to a test split, the panel, the MIMIC audit suite or a held-out file.

Usage (GPU; one job; detached; about 1 h):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v5/z1_probes.py \
        > outputs/osler_v5/Z1/logs/probes.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from meddecide.eval.composite import composite, permutation_for, positional_health
from meddecide.model.blend import Blend
from meddecide.model.meddecide_model import reorder_options
from meddecide.model.pplx import PplxDecisionModel
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO / "outputs" / "osler_v4" / "Y8" / "checkpoints" / "best"
CONFIG = REPO / "configs" / "osler_v5" / "arm_R0.yaml"
DEV = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"
CG = REPO / "data" / "bench" / "osler_v4" / "cg_v2.jsonl"
KNOW = REPO / "data" / "bench" / "osler_v2" / "knowledge_dev_v2.jsonl"
OUT = REPO / "outputs" / "osler_v5" / "Z1"
DDX_PREFIX = "cg_v2_ddxplus"
POSITIONAL_SEEDS = (1, 2)
ALPHAS = (0.0, 0.3, 0.5, 0.7, 0.85, 1.0)
REFERENCE_CHECK_ITEMS = 16


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def score(model: PplxDecisionModel, config: Any, items: list) -> list[dict[str, Any]]:
    """Per-item rows with the trainer's evaluation settings (Trainer.score, without an optimiser)."""
    model.eval_mode()
    scored = model.score_items(
        items,
        batch_size=config.eval_batch_size or config.batch_size,
        max_batch_tokens=config.eval_max_batch_tokens or config.max_batch_tokens,
        max_prompt_tokens=config.max_prompt_tokens,
        round_to_chunk=config.eval_round_to_chunk,
        length_buckets=config.eval_length_buckets,
    )
    return dev_prediction_rows(scored)


def probe(model: PplxDecisionModel, config: Any, blend_setting: str, sets: dict[str, list], ddx_index: list[int]) -> dict[str, Any]:
    """One setting: score the dev sets and the permuted DDXPlus items; save the rows; return the summary."""
    started = time.time()
    rows = {name: score(model, config, items) for name, items in sets.items()}
    ddx_items = [sets["cg_v2"][i] for i in ddx_index]
    perm_rows: dict[str, list[dict[str, Any]]] = {}
    for seed in POSITIONAL_SEEDS:
        permuted = [reorder_options(item, permutation_for(item.n_options, seed=seed, index=i))
                    for item, i in zip(ddx_items, ddx_index, strict=True)]
        perm_rows[str(seed)] = score(model, config, permuted)
    folder = OUT / blend_setting
    folder.mkdir(parents=True, exist_ok=True)
    write_jsonl(folder / "fast_dev.jsonl", rows["fast_dev_v4"])
    write_jsonl(folder / "cg_v2.jsonl", rows["cg_v2"])
    write_jsonl(folder / "knowledge_dev.jsonl", rows["knowledge_dev_v2"])
    for seed, pred in perm_rows.items():
        write_jsonl(folder / f"positional_seed{seed}.jsonl", pred)
    parts = composite(rows["fast_dev_v4"], rows["cg_v2"])
    cg_by_template: dict[str, list[bool]] = defaultdict(list)
    for r in rows["cg_v2"]:
        cg_by_template[str(r["template_id"])].append(bool(r["correct"]))
    ddx_rows = [r for r in rows["cg_v2"] if str(r["template_id"]).startswith(DDX_PREFIX)]
    summary = {
        "composite": parts.to_dict(),
        "knowledge_dev_accuracy": sum(bool(r["correct"]) for r in rows["knowledge_dev_v2"]) / len(rows["knowledge_dev_v2"]),
        "positional_original": positional_health(ddx_rows).to_dict(),
        "positional_permuted": {seed: positional_health(pred).to_dict() for seed, pred in perm_rows.items()},
        "cg_micro_per_template": {t: {"accuracy": sum(v) / len(v), "items": len(v)} for t, v in sorted(cg_by_template.items())},
        "ddx_two_letter_accuracy_original": positional_health(ddx_rows).two_letter_accuracy,
        "seconds": round(time.time() - started, 1),
    }
    print(f"{blend_setting}: S {parts.s:.4f} (MR {parts.mr:.4f}, MIMIC {parts.mimic:.4f}, CG micro {parts.cg_micro:.4f}); "
          f"knowledge {summary['knowledge_dev_accuracy']:.4f}; DDX two-letter {summary['ddx_two_letter_accuracy_original']:.4f}; "
          f"positional original {summary['positional_original']['passes']}, seeds "
          f"{[h['passes'] for h in summary['positional_permuted'].values()]}; {summary['seconds']} s", flush=True)
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    config = load_config(CONFIG)
    config = dataclasses.replace(config, device="cuda:0", dtype="bfloat16", backbone="pplx")
    model = PplxDecisionModel.load(CHECKPOINT, device="cuda:0", dtype="bfloat16")
    model.eval_mode()
    sets = {
        "fast_dev_v4": [mix_row_to_item(r) for r in read_jsonl(DEV)],
        "cg_v2": [mix_row_to_item(r) for r in read_jsonl(CG)],
        "knowledge_dev_v2": [mix_row_to_item(r) for r in read_jsonl(KNOW)],
    }
    ddx_index = [i for i, item in enumerate(sets["cg_v2"]) if str(item.template_id).startswith(DDX_PREFIX)]
    blend = Blend(model)

    # the blend must reproduce the reference path at alpha 0 on the GPU too (16 CG items, bf16)
    check_items = sets["cg_v2"][:REFERENCE_CHECK_ITEMS]
    with torch.no_grad():
        batch = model.collate([model.encode_item(item) for item in check_items])
        reference = model.reference_option_probs(batch).detach().float().cpu()
    blend.apply(0.0)
    with torch.no_grad():
        batch = model.collate([model.encode_item(item) for item in check_items])
        _, flat = model.batch_logits(batch)
    reference_diff = float((flat.detach().float().cpu() - reference).abs().max())
    print(f"alpha 0 against the reference path (bf16, GPU, {REFERENCE_CHECK_ITEMS} items): max abs diff {reference_diff:.3e}", flush=True)

    results: dict[str, Any] = {"started_utc": started, "git_commit": git_commit(REPO), "checkpoint": str(CHECKPOINT.relative_to(REPO)),
                               "dtype": "bfloat16", "device": "cuda:0", "items": {k: len(v) for k, v in sets.items()},
                               "ddx_items": len(ddx_index), "reference_check": {"items": REFERENCE_CHECK_ITEMS,
                                                                                 "max_abs_diff_alpha0": reference_diff},
                               "blend_curve": {}, "readout_split": {}}
    for alpha in ALPHAS:
        blend.apply(alpha)
        results["blend_curve"][f"{alpha:g}"] = probe(model, config, f"alpha_{alpha:g}", sets, ddx_index)
        write_json(OUT / "probes.json", results)
    # (a) the trained adapter with the base readout
    blend.apply(1.0)
    with torch.no_grad():
        model.optcode.weight.copy_(blend.base_readout.to(model.optcode.weight.device))
    results["readout_split"]["trained_adapter_base_readout"] = probe(model, config, "split_trained_adapter_base_readout", sets, ddx_index)
    write_json(OUT / "probes.json", results)
    # (b) the base adapter (off) with the trained readout
    blend.apply(0.0)
    with torch.no_grad():
        model.optcode.weight.copy_(blend.trained_readout.to(model.optcode.weight.device))
    results["readout_split"]["base_adapter_trained_readout"] = probe(model, config, "split_base_adapter_trained_readout", sets, ddx_index)
    blend.apply(1.0)  # restore the trained model
    results["finished_utc"] = utcnow()
    write_json(OUT / "probes.json", results)
    print("done", flush=True)


if __name__ == "__main__":
    main()
