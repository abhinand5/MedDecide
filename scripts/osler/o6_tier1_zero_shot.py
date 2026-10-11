"""Zero-shot Qwen3.5-4B on the O6-O8 tier-1 dev diagnostic items (CPU; the reference the ADVISORY's failure mode 3 asks for).

The arms report accuracy by source on the tier-1 dev items (MedQA and MedMCQA dev rows of the training mix) at every
evaluation. This script scores the same items with the base model alone (no LoRA): the letter readout of the base's LM
head over the offered options, bare letter variant, the same prompt cap and tokenisation as the arms. It runs on the CPU
because the GPU is busy with the O6-O8 chain (one GPU job at a time).

Writes outputs/osler_v0/O6/tier1_zero_shot/tier1_zero_shot.json (gitignored); the committed record is the CLAIMS row.

Usage (after the GPU is free; the reference takes minutes on the GPU, hours on the CPU):
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o6_tier1_zero_shot.py --device cuda:0 --dtype bfloat16

The CPU variant (--device cpu --dtype float32) takes the torch fallback kernels; it was stopped after 43 minutes because it
slowed the GPU training run (the training loop shares the host's CPUs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# CPU runs only: the fast causal-conv1d and fla kernels are CUDA-only and reject CPU tensors, so the torch fallback path is
# used (the same switch the CPU tests use). On the GPU the arms' own kernels are used.
_CPU_RUN = "--device" in sys.argv and sys.argv[sys.argv.index("--device") + 1].startswith("cpu")
if _CPU_RUN:
    os.environ.setdefault("USE_HUB_KERNELS", "NO")
    sys.modules["causal_conv1d"] = None
    sys.modules["fla"] = None

from meddecide.model.meddecide_model import MedDecideModel  # noqa: E402
from meddecide.train.config import load_config  # noqa: E402
from meddecide.train.mix_items import mix_row_to_item  # noqa: E402
from meddecide.train.osler_arm import accuracy_by_source, tier1_items  # noqa: E402
from meddecide.utils.io import write_json  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REFERENCE_CONFIG = REPO / "configs/osler_v0/arm_L.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    started = time.time()
    config = load_config(REFERENCE_CONFIG)
    dev_path = REPO / config.dev_path
    dev_items = []
    with dev_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                dev_items.append(mix_row_to_item(json.loads(line)))
    tier1 = tier1_items(dev_items)
    print(f"tier-1 dev items: {len(tier1)} of {len(dev_items)} dev items; sources {dict(Counter(i.source for i in tier1))}",
          flush=True)
    model = MedDecideModel(
        config.base_model,
        revision=config.revision,
        lora=None,
        dtype=args.dtype,
        device=args.device,
        variant=config.variant,
        readout="letter",
        max_prompt_tokens=config.max_prompt_tokens,
    )
    model.eval_mode()
    decisions = model.predict(tier1, batch_size=args.batch_size)
    per_source = accuracy_by_source(decisions, tier1)
    counts = dict(Counter(i.source for i in tier1))
    record = {
        "kind": "osler_v0_O6_tier1_zero_shot",
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "command": "scripts/osler/o6_tier1_zero_shot.py",
        "model": config.base_model,
        "revision": config.revision,
        "readout": "letter (zero-shot base, no LoRA)",
        "variant": config.variant,
        "dtype": args.dtype,
        "device": args.device,
        "dev_path": str(dev_path.relative_to(REPO)),
        "tier1_items": len(tier1),
        "items_by_source": counts,
        "accuracy_by_source": per_source,
        "wall_clock_s": round(time.time() - started, 1),
    }
    out = REPO / "outputs/osler_v0/O6/tier1_zero_shot"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "tier1_zero_shot.json", record)
    print(json.dumps({"accuracy_by_source": {k: round(v, 4) for k, v in per_source.items()}, "items_by_source": counts,
                      "wall_clock_s": record["wall_clock_s"]}), flush=True)


if __name__ == "__main__":
    main()
