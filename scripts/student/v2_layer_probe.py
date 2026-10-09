#!/usr/bin/env python
"""Per-layer padding probe: where does a left-padded item's hidden state first differ from alone?

For each of the probe items, the real-token hidden states of every layer are compared between a
batch of one (alone) and the item's row in a left-padded batch of all the items. A jump at one
layer points to a leak in that block; a gradual rise is numerical drift that compounds with depth.
Dev items only; nothing is fitted.
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

from v2_padding_probe import dev_items  # noqa: E402

from meddecide.model.meddecide_model import MedDecideModel  # noqa: E402


def layer_states(model: MedDecideModel, batch) -> tuple[torch.Tensor, ...]:
    position_ids = (batch.attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    net = model.peft_model if model.peft_model is not None else model.base
    out = net(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        position_ids=position_ids,
        use_cache=False,
        output_hidden_states=True,
        logits_to_keep=1,
    )
    return out.hidden_states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bench", type=Path, default=ROOT / "data" / "bench" / "v0.2")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-items", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    items = dev_items(args.bench)
    step = max(1, (len(items) - 1) // (args.n_items - 1))
    items.sort(key=lambda i: (len(i.state) + len(i.question), i.item_id))
    sample = [items[k * step] for k in range(args.n_items)]
    model = MedDecideModel.load(args.checkpoint, device=args.device, dtype=args.dtype)
    encoded = [model.encode_item(i) for i in sample]

    n_layers = None
    worst: list[float] = []
    with torch.no_grad():
        padded = layer_states(model, model.collate(encoded).to(model.device))
        n_layers = len(padded)
        per_layer = [[0.0] * n_layers for _ in sample]
        for j, enc in enumerate(encoded):
            alone = layer_states(model, model.collate([enc]).to(model.device))
            alone_shift = alone[0].shape[1] - enc.n_tokens
            shift = padded[0].shape[1] - enc.n_tokens
            for layer in range(n_layers):
                a = alone[layer][0, alone_shift:].float()
                p = padded[layer][j, shift:].float()
                per_layer[j][layer] = float((a - p).abs().max().item())
    for layer in range(n_layers):
        worst.append(max(per_layer[j][layer] for j in range(len(sample))))

    result = {
        "checkpoint": str(args.checkpoint),
        "dtype": args.dtype,
        "n_items": len(sample),
        "n_layers_incl_embedding": n_layers,
        "prompt_tokens": [e.n_tokens for e in encoded],
        "max_abs_delta_per_layer": worst,
        "max_abs_delta_per_item_last_layer": [per_layer[j][-1] for j in range(len(sample))],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([round(v, 6) for v in worst]))


if __name__ == "__main__":
    main()
