"""osler_v5 Z0 step 10 (D53): the blend on the real osler_v4 final checkpoint, in fp32 on the CPU, against the reference path. Four fast-dev
items (dev only; their option probabilities are numbers, not text, and nothing is printed but the maxima and counts).

Checks: alpha = 0 reproduces MedDecideModel.reference_option_probs (adapter off, the base readout) to < 1e-5; alpha = 1 reproduces the
trained model's probabilities (the unblended scores) to < 1e-6. Writes outputs/osler_v5/Z0/blend_check.json.

Usage (CPU; loads 4B weights in fp32, about 16 GB; detached):
    uv run --frozen python scripts/osler_v5/z0_blend_check.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# the CUDA-only linear-attention kernels are blocked, as in the CPU tests: the torch fallbacks compute the same model
os.environ.setdefault("USE_HUB_KERNELS", "NO")
sys.modules["causal_conv1d"] = None
sys.modules["fla"] = None

import torch  # noqa: E402

from meddecide.model.blend import Blend  # noqa: E402
from meddecide.model.pplx import PplxDecisionModel  # noqa: E402
from meddecide.train.mix_items import mix_row_to_item  # noqa: E402
from meddecide.utils.io import write_json  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO / "outputs" / "osler_v4" / "Y8" / "checkpoints" / "best"
DEV = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"
OUT = REPO / "outputs" / "osler_v5" / "Z0" / "blend_check.json"
TOLERANCE_ALPHA_0 = 1e-5
TOLERANCE_ALPHA_1 = 1e-6


def read_items(n: int) -> list:
    items = []
    with DEV.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                items.append(mix_row_to_item(json.loads(line)))
            if len(items) == n:
                break
    return items


def probs(model: PplxDecisionModel, items: list) -> torch.Tensor:
    batch = model.collate([model.encode_item(item) for item in items])
    _, flat = model.batch_logits(batch)
    return flat.detach().clone()


def reference(model: PplxDecisionModel, items: list) -> torch.Tensor:
    batch = model.collate([model.encode_item(item) for item in items])
    return model.reference_option_probs(batch).detach().clone()


def main() -> None:
    started = utcnow()
    t0 = time.time()
    model = PplxDecisionModel.load(CHECKPOINT, device="cpu", dtype="float32")
    model.eval_mode()
    print(f"loaded in {round(time.time() - t0)} s", flush=True)
    items = read_items(4)
    option_counts = [item.n_options for item in items]
    trained = probs(model, items)
    ref = reference(model, items)
    blend = Blend(model)
    blend.apply(0.0)
    alpha0 = probs(model, items)
    diff_alpha0_reference = float((alpha0 - ref).abs().max())
    diff_alpha0_trained = float((alpha0 - trained).abs().max())
    blend.apply(1.0)
    alpha1 = probs(model, items)
    diff_alpha1_trained = float((alpha1 - trained).abs().max())
    result = {
        "started_utc": started, "finished_utc": utcnow(), "git_commit": git_commit(REPO),
        "checkpoint": str(CHECKPOINT.relative_to(REPO)), "dtype": "float32", "device": "cpu",
        "items": len(items), "option_counts": option_counts, "flat_options": int(trained.numel()),
        "max_abs_diff_alpha0_vs_reference": diff_alpha0_reference, "tolerance_alpha0": TOLERANCE_ALPHA_0,
        "alpha0_matches_reference": diff_alpha0_reference < TOLERANCE_ALPHA_0,
        "max_abs_diff_alpha0_vs_trained": diff_alpha0_trained,
        "max_abs_diff_alpha1_vs_trained": diff_alpha1_trained, "tolerance_alpha1": TOLERANCE_ALPHA_1,
        "alpha1_matches_trained": diff_alpha1_trained < TOLERANCE_ALPHA_1,
        "seconds": round(time.time() - t0, 1),
    }
    write_json(OUT, result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
