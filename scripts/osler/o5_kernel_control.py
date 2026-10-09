"""O5 kernel-path control (diagnostic, GPU): how reproducible and how sensitive is the fp32 forward of the pinned
Qwen3.5-4B, by precision setting (``meddecide.precision``)?

One process runs one setting on one synthetic item (no benchmark, training or teacher text), with the O5 adapter shape and
the random LoRA and head of the O5 checks. It measures, on the last hidden state:
    rep       two identical separate-adapter forwards (run-to-run noise inside one process)
    merge     separate-adapter forward vs merge_and_unload forward (the O5 merged export, hidden-state version)
    sens_*    merged forward vs the merged forward after every decoder-layer weight is multiplied by (1 + s N(0,1)),
              for s = 1e-7 (about one fp32 ulp) and s = 1e-5
Two processes with the same setting can be compared through the saved last-layer hidden states (cross-process
reproducibility). This is a diagnostic: the O5 verdict is ``scripts/osler/o5_checks.py``.

Usage (one GPU job at a time):
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o5_kernel_control.py --precision as_run --tag run1
Writes outputs/osler_v0/O5/kernel_control_<precision>_<tag>.json and ..._hidden.pt (both gitignored).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import time
from pathlib import Path

from meddecide.precision import PRECISIONS, prepare_precision

REPO = Path(__file__).resolve().parents[2]
BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
OUT_DIR = REPO / "outputs/osler_v0/O5"
SENSITIVITY_SCALES = (1e-7, 1e-5)
KERNEL_PACKAGES = ("triton", "flash-linear-attention", "causal-conv1d")


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in KERNEL_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=PRECISIONS, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=0, help="seeds the global RNG before the adapter is built")
    args = parser.parse_args()
    # Before transformers or fla is imported: both read the environment and the module table at import time.
    prepare_precision(args.precision)

    import peft
    import torch
    import transformers

    from meddecide.model.meddecide_model import LoraSettings, MedDecideModel
    from meddecide.model.readout_checks import perturb_head, randomise_lora, synthetic_choice_item
    from meddecide.precision import (
        apply_after_import,
        kernel_implementations,
        max_abs_gap,
        perturb_layer_weights,
        restore_layer_weights,
        snapshot_layer_weights,
    )
    from meddecide.utils.io import write_json
    from meddecide.utils.provenance import git_commit, utcnow

    started = time.time()
    apply_after_import(args.precision)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    implementations = kernel_implementations()
    # PEFT draws lora_A from the global RNG; without this seed every process builds a different adapter.
    torch.manual_seed(args.seed)
    model = MedDecideModel(BASE_MODEL, revision=BASE_REVISION, lora=LoraSettings(r=32, alpha=32), device="cuda:0",
                           dtype="float32", readout="option_code", variant="bare")
    randomise_lora(model)
    perturb_head(model)
    item = synthetic_choice_item(2, 0)
    batch = model.collate([model.encode_item(item)]).to("cuda:0")
    mask = batch.attention_mask
    positions = (mask.long().cumsum(-1) - 1).clamp(min=0)

    def last_hidden(module: torch.nn.Module) -> torch.Tensor:
        with torch.no_grad():
            out = module(input_ids=batch.input_ids, attention_mask=mask, position_ids=positions, use_cache=False,
                         output_hidden_states=True)
        return out.hidden_states[-1].float().cpu()

    separate = last_hidden(model.peft_model)
    separate_again = last_hidden(model.peft_model)
    merged_model = model.peft_model.merge_and_unload().eval()
    merged = last_hidden(merged_model)
    snapshot = snapshot_layer_weights(merged_model)
    sensitivity: dict[str, float] = {}
    for scale in SENSITIVITY_SCALES:
        perturb_layer_weights(merged_model, snapshot, scale, seed=0)
        sensitivity[f"{scale:.0e}"] = max_abs_gap(merged, last_hidden(merged_model))
        restore_layer_weights(merged_model, snapshot)
    torch.save({"separate_last": separate, "merged_last": merged},
               OUT_DIR / f"kernel_control_{args.precision}_{args.tag}_hidden.pt")

    record = {
        "kind": "osler_v0_O5_kernel_control",
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "command": f"scripts/osler/o5_kernel_control.py --precision {args.precision} --tag {args.tag} --seed {args.seed}",
        "precision": args.precision,
        "tag": args.tag,
        "seed": args.seed,
        "environment": {"USE_HUB_KERNELS": os.environ.get("USE_HUB_KERNELS", "(unset)"),
                        "TRITON_F32_DEFAULT": os.environ.get("TRITON_F32_DEFAULT", "(unset)")},
        "implementations": implementations,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "dtype": "float32",
        "item": "synthetic_choice_item(2, 0)",
        "tokens": int(mask.sum().item()),
        "last_hidden_max_abs": float(separate.abs().max().item()),
        "rep_max_abs_diff": max_abs_gap(separate, separate_again),
        "merge_max_abs_diff": max_abs_gap(separate, merged),
        "sensitivity_max_abs_diff": sensitivity,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                     "gpu": torch.cuda.get_device_name(0), **package_versions()},
        "wall_clock_s": round(time.time() - started, 1),
    }
    write_json(OUT_DIR / f"kernel_control_{args.precision}_{args.tag}.json", record)
    print(f"{args.precision}/{args.tag}: implementations {implementations}", flush=True)
    print(f"rep {record['rep_max_abs_diff']:.6g}  merge {record['merge_max_abs_diff']:.6g}  "
          f"sens {sensitivity}  last-hidden max|h| {record['last_hidden_max_abs']:.4g}", flush=True)


if __name__ == "__main__":
    main()
