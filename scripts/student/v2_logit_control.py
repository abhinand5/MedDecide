#!/usr/bin/env python
"""V2: is the padded-batch logit residual larger than pure sequence-length shape noise?

For each of the same 8 dev items the probe uses (strict fp32), compare pointer logits under three
layouts of the *same* item:

  alone        batch of one, collated as the fixed code does (rounded to the chunk length)
  unpadded     batch of one with no padding at all (the item's own length, no pad tokens)
  shifted      alone plus 64 extra inert left pads: identical chunk offset, identical real positions,
               masked pads, so exactly the same computation in exact arithmetic
  batched      the padded batch of all 8 items (the probe's test condition)

``shift_delta`` = |alone - shifted| is the shape-only noise floor for this item. ``batch_delta`` =
|alone - batched| is the probe's quantity. ``rounding_delta`` = |alone - unpadded| is the cost of the
chunk rounding itself, which the bs=1 evaluation path carries. If ``batch_delta`` is no larger than ``shift_delta``,
the padded-batch residual is explained by sequence-length shape noise, not by pad leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from v2_padding_probe import dev_items, set_fp32_precision  # noqa: E402
from v2_shape_control import with_extra_pad  # noqa: E402

from meddecide.model.meddecide_model import MedDecideModel  # noqa: E402


def unpadded_batch(model: MedDecideModel, enc):
    """A batch of one with no pad tokens: the item exactly as encoded."""
    batch = model.collate([enc])
    n = enc.n_tokens
    batch.input_ids = torch.tensor([enc.input_ids], dtype=torch.long)
    batch.attention_mask = torch.ones((1, n), dtype=torch.long)
    batch.answer_positions = torch.full((1,), n - 1, dtype=torch.long)
    batch.marker_positions = torch.tensor(enc.marker_positions, dtype=torch.long)
    batch.option_end_positions = torch.tensor(enc.option_end_positions, dtype=torch.long)
    return batch


def per_item_logits(model: MedDecideModel, batch, n_options: list[int]) -> list[np.ndarray]:
    flat, _ = model.batch_logits(batch)
    flat_np = flat.float().cpu().numpy().astype(np.float64)
    out, cursor = [], 0
    for n in n_options:
        out.append(flat_np[cursor : cursor + n])
        cursor += n
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bench", type=Path, default=ROOT / "data" / "bench" / "v0.2")
    parser.add_argument("--n-items", type=int, default=8)
    parser.add_argument("--extra-pad", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    precision = set_fp32_precision(tf32_conv=False)
    items = dev_items(args.bench)
    # the same selection rule as padding_check, so the same items are compared with the probe
    order = sorted(range(len(items)), key=lambda i: (len(items[i].state) + len(items[i].question)))
    picks = sorted({order[round(k * (len(order) - 1) / max(1, args.n_items - 1))]
                    for k in range(args.n_items)})
    sample = [items[i] for i in picks]
    model = MedDecideModel.load(args.checkpoint, device="cuda:0", dtype="float32")
    encoded = [model.encode_item(item) for item in sample]
    n_options = [e.n_options for e in encoded]
    rows = []
    with torch.no_grad():
        alone = [per_item_logits(model, model.collate([e]).to(model.device), [e.n_options])[0]
                 for e in encoded]
        unpadded = [per_item_logits(model, unpadded_batch(model, e).to(model.device),
                                    [e.n_options])[0] for e in encoded]
        shifted = [per_item_logits(model, with_extra_pad(model, e, args.extra_pad).to(model.device),
                                   [e.n_options])[0] for e in encoded]
        batched = per_item_logits(model, model.collate(encoded).to(model.device), n_options)
    for k, enc in enumerate(encoded):
        rows.append({
            "item_id": sample[k].item_id,
            "prompt_tokens": enc.n_tokens,
            "shift_delta": float(np.max(np.abs(alone[k] - shifted[k]))),
            "batch_delta": float(np.max(np.abs(alone[k] - batched[k]))),
            "rounding_delta": float(np.max(np.abs(alone[k] - unpadded[k]))),
        })
    result = {
        "kind": "logit_shape_control",
        "task": "V2",
        "checkpoint": str(args.checkpoint),
        "dtype": "float32",
        "precision": precision,
        "extra_pad": args.extra_pad,
        "n_items": len(sample),
        "max_shift_delta": max(r["shift_delta"] for r in rows),
        "max_batch_delta": max(r["batch_delta"] for r in rows),
        "max_rounding_delta": max(r["rounding_delta"] for r in rows),
        "items": rows,
        "meddecide_module": sys.modules["meddecide"].__file__,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("max_shift_delta", "max_batch_delta", "max_rounding_delta")}))


if __name__ == "__main__":
    main()
