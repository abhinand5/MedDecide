"""D3: the Clef-Flash text-tower port check (distill_v0; GOAL §3.1, timebox 3 h).

Clef-Flash is a Qwen3.5 vision-language checkpoint (`Qwen3_5ForConditionalGeneration`) with its own
joint schema head. Osler-9B needs the *text* tower with our option-code readout. This script loads
the pinned snapshot **in memory** through ``meddecide.model.clef`` (the vision tower is dropped and
counted; no converted copy is written — the workspace is quota-bound and a copy would duplicate 18 GB
of weights already in the HF cache), builds the model through ``MedDecideModel`` with
``base_override``, checks that the readout rows equal the LM head's rows for the code tokens, runs a
20-item smoke on the dev set (fastdev v4) and a backward pass, and reports probability sums, label
mass, argmax spread and gradient norms. This is a code-path check, not a result.

If it fails, the pre-registered fallback is ``Qwen/Qwen3.5-9B`` at its pinned revision, recorded in
STATE.md as a deviation.

Usage (GPU; one job; detached):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/distill_v0/d3_clef_port.py \
        --out outputs/distill_v0/D3 > outputs/distill_v0/D3/logs/run.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from meddecide.model.clef import REPO as CLEF
from meddecide.model.clef import REVISION as CLEF_REVISION
from meddecide.model.clef import checkpoint_files, load_clef_text_backbone
from meddecide.model.meddecide_model import LoraSettings, MedDecideModel
from meddecide.train.losses import ce_plus_brier
from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
FALLBACK = "Qwen/Qwen3.5-9B"
FALLBACK_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DEV = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"


def read_items(limit: int) -> list[Any]:
    """The first dev rows, converted as the training path converts them (the dev rows carry their own tier field)."""
    items: list[Any] = []
    with DEV.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            items.append(mix_row_to_item(json.loads(line)))
            if len(items) >= limit:
                break
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="outputs/distill_v0/D3")
    parser.add_argument("--smoke-items", type=int, default=20)
    args = parser.parse_args()
    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    clock = time.time()
    result: dict[str, Any] = {"task": "D3 Clef-Flash port check", "git_commit": git_commit(REPO),
                              "started_utc": started, "source_repo": CLEF, "source_revision": CLEF_REVISION,
                              "approach": "in-memory: Qwen3_5ForConditionalGeneration.model.language_model + lm_head, "
                                          "no disk copy",
                              "fallback": {"repo": FALLBACK, "revision": FALLBACK_REVISION}}
    files = checkpoint_files()
    result["snapshot"] = {"files": [{"name": path.name, "bytes": path.stat().st_size} for path in files],
                          "total_bytes": sum(path.stat().st_size for path in files)}
    print(f"loading the Clef text tower ({len(files)} shards, {result['snapshot']['total_bytes'] / 1e9:.1f} GB) ...",
          flush=True)
    backbone, clef_info = load_clef_text_backbone(dtype="bfloat16", device="cuda:0")
    result["clef"] = clef_info
    print(f"backbone: {clef_info}", flush=True)
    lora = LoraSettings(r=32, alpha=32, dropout=0.0, target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    model = MedDecideModel(CLEF, revision=CLEF_REVISION, lora=lora, readout="option_code", dtype="bfloat16",
                           device="cuda:0", variant="bare", max_prompt_tokens=16384, base_override=backbone)
    result["model"] = {"readout_codes": len(model.optcode_names), "code_token_ids": len(model.optcode_token_ids),
                       "first_codes": model.optcode_names[:8],
                       "trainable_parameters": model.trainable_parameter_count,
                       "hidden_size": int(model.get_decoder().config.hidden_size)}
    head = model.base.lm_head.weight.detach()
    with torch.no_grad():
        expected = head[torch.tensor(model.optcode_token_ids, device=head.device)].float().cpu()
        equal = torch.equal(expected, model.optcode.weight.detach().float().cpu())
    result["model"]["rows_equal_lm_head"] = bool(equal)
    print(f"readout rows equal lm_head: {equal}", flush=True)
    if not equal:
        result["verdict"] = "PORT_FAIL - readout rows do not equal the LM head rows"
        (out / "port_check.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit(result["verdict"])

    print(f"smoke: scoring {args.smoke_items} dev items ...", flush=True)
    items = read_items(args.smoke_items)
    smoke_started = time.time()
    decisions = model.predict(items, batch_size=4)
    sums = [sum(decision.probs) for decision in decisions]
    predicted = [decision.argmax_key for decision in decisions]
    result["smoke"] = {"items": len(items), "seconds": round(time.time() - smoke_started, 1),
                       "prob_sums_min": min(sums), "prob_sums_max": max(sums),
                       "label_mass_min": min(sums), "label_mass_mean": sum(sums) / len(sums),
                       "distinct_predicted_keys": len(set(predicted)),
                       "correct": sum(int(decision.correct) for decision in decisions)}
    print(f"smoke: {json.dumps(result['smoke'])}", flush=True)

    print("backward pass on 4 items ...", flush=True)
    model.train_mode()
    encoded = [model.encode_item(item) for item in items[:4]]
    batch = model.collate(encoded)
    _logits, probs = model.batch_logits(batch)
    loss = ce_plus_brier(probs, batch.gold_flat, batch.marker_item, batch.batch_size, lam=1.0).total
    loss.backward()
    lora_grad = sum(float(parameter.grad.norm()) for parameter in model.peft_model.parameters()
                    if parameter.grad is not None)
    head_grad = float(model.optcode.weight.grad.norm()) if model.optcode.weight.grad is not None else 0.0
    result["backward"] = {"loss": float(loss.detach()), "lora_grad_norm_sum": lora_grad,
                          "optcode_grad_norm": head_grad}
    ok = (result["model"]["rows_equal_lm_head"] and lora_grad > 0 and head_grad > 0
          and result["smoke"]["prob_sums_min"] > 0.99)
    result["verdict"] = "PORT_OK" if ok else "PORT_FAIL"
    result["wall_clock_seconds"] = round(time.time() - clock, 1)
    result["finished_utc"] = utcnow()
    result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    result["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1) if torch.cuda.is_available() else None
    (out / "port_check.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("verdict", "clef", "model", "smoke", "backward",
                                                    "wall_clock_seconds", "peak_gpu_gb")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
