#!/usr/bin/env python
"""Padding probe on a trained checkpoint: one dev item alone vs inside a left-padded batch.

Thin wrapper around ``scripts/bench/train_student.py::padding_check`` (the S9-diag probe) so the
same check runs on the trained step-1,000 checkpoint in fp32. Dev items only; nothing is fitted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "bench"))
sys.path.insert(0, str(ROOT / "src"))

from train_student import padding_check  # noqa: E402

from meddecide.model.meddecide_model import MedDecideModel  # noqa: E402
from meddecide.train.data import read_items  # noqa: E402


def set_fp32_precision(tf32_conv: bool) -> dict[str, bool]:
    """Pin the fp32 precision this probe is documented as using, and return the flags for provenance.

    PyTorch enables TF32 for cuDNN convolutions by default. Measured in V2, the flag does not change
    the probe's numbers (identical values in both modes; see SELF_AUDIT), so the pin is for
    documentation and provenance. ``tf32_conv=True`` reproduces the earlier as-run commands.
    """
    torch.backends.cudnn.allow_tf32 = tf32_conv
    torch.backends.cuda.matmul.allow_tf32 = False
    return {
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }


def dev_items(bench: Path) -> list:
    items = []
    for directory in ("tier1", "fresh"):
        for path in sorted((bench / directory).glob("*.jsonl")):
            items.extend(i for i in read_items(path) if str(i.split) == "dev")
    return items


def git_head() -> str:
    """HEAD of the checkout this script runs from, or ``none`` for an exported copy."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bench", type=Path, default=ROOT / "data" / "bench" / "v0.2")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-items", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--tf32-conv", action="store_true",
                        help="keep PyTorch's default TF32 for cuDNN convolutions (as-run mode)")
    parser.add_argument("--code-label", default="",
                        help="commit of an exported copy of the code (the probe cannot read it)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    precision = set_fp32_precision(tf32_conv=args.tf32_conv)
    items = dev_items(args.bench)
    model = MedDecideModel.load(args.checkpoint, device=args.device, dtype=args.dtype)
    probe = padding_check(model, items, n_items=args.n_items, tolerance=args.tolerance)
    probe.update(
        {
            "checkpoint": str(args.checkpoint),
            "dtype": args.dtype,
            "precision": precision,
            "git_commit": git_head(),
            "code_label": args.code_label,
            "n_dev_items_available": len(items),
            "meddecide_module": sys.modules["meddecide"].__file__,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: probe[k] for k in ("verdict", "n_items", "tolerance")}))
    print(json.dumps(probe["alone_vs_left_padded_batch"]["max_abs_delta"]))
    print(json.dumps(probe["alone_vs_uniform_batch_control"]["max_abs_delta"]))


if __name__ == "__main__":
    main()
