#!/usr/bin/env python
"""Shape-noise control for the padding probe.

Each item is scored alone (batch of one) and alone with ``--extra-pad`` additional left pad tokens.
The extra pads are whole linear-attention chunks, so the item keeps its chunk offset and the pads
are masked. Any difference between the two is therefore matmul-shape numerical noise, not a
padding effect. Compare its size with the alone-vs-batch difference from v2_layer_probe.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from v2_layer_probe import layer_states  # noqa: E402
from v2_padding_probe import dev_items, set_fp32_precision  # noqa: E402

from meddecide.model.meddecide_model import LINEAR_ATTENTION_CHUNK, MedDecideModel  # noqa: E402


def with_extra_pad(model: MedDecideModel, enc, extra: int):
    batch = model.collate([enc])
    pad_id = int(model.tokenizer.pad_token_id)
    length = batch.input_ids.shape[1] + extra
    input_ids = torch.full((1, length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((1, length), dtype=torch.long)
    input_ids[0, extra:] = batch.input_ids[0]
    attention_mask[0, extra:] = batch.attention_mask[0]
    batch.input_ids = input_ids
    batch.attention_mask = attention_mask
    batch.answer_positions = torch.full((1,), length - 1, dtype=torch.long)
    batch.marker_positions = batch.marker_positions + extra
    batch.option_end_positions = batch.option_end_positions + extra
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bench", type=Path, default=ROOT / "data" / "bench" / "v0.2")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-items", type=int, default=8)
    parser.add_argument("--extra-pad", type=int, default=LINEAR_ATTENTION_CHUNK)
    parser.add_argument("--tf32-conv", action="store_true",
                        help="keep PyTorch's default TF32 for cuDNN convolutions (as-run mode)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    precision = set_fp32_precision(tf32_conv=args.tf32_conv)
    items = dev_items(args.bench)
    step = max(1, (len(items) - 1) // (args.n_items - 1))
    items.sort(key=lambda i: (len(i.state) + len(i.question), i.item_id))
    sample = [items[k * step] for k in range(args.n_items)]
    model = MedDecideModel.load(args.checkpoint, device=args.device, dtype=args.dtype)
    per_item = []
    with torch.no_grad():
        for item in sample:
            enc = model.encode_item(item)
            alone = layer_states(model, model.collate([enc]).to(model.device))
            shifted = layer_states(model, with_extra_pad(model, enc, args.extra_pad).to(model.device))
            n = enc.n_tokens
            a_shift = alone[0].shape[1] - n
            s_shift = shifted[0].shape[1] - n
            deltas = [
                float((alone[k][0, a_shift:].float() - shifted[k][0, s_shift:].float()).abs().max())
                for k in range(len(alone))
            ]
            per_item.append({"prompt_tokens": n, "max_abs_delta_last_layer": deltas[-1],
                             "max_abs_delta_per_layer": deltas})
    worst = [max(p["max_abs_delta_per_layer"][k] for p in per_item) for k in range(len(per_item[0]["max_abs_delta_per_layer"]))]
    result = {
        "checkpoint": str(args.checkpoint),
        "dtype": args.dtype,
        "precision": precision,
        "extra_pad": args.extra_pad,
        "n_items": len(sample),
        "max_abs_delta_per_layer": worst,
        "items": per_item,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([round(v, 6) for v in worst]))


if __name__ == "__main__":
    main()
