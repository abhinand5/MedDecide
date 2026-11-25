"""osler_v5 D56d (ADVISORY §8.6; CPU only, additional, chooses nothing): the readout scale of the osler_v4 final, a corrected-mix checkpoint
(F2) and pplx's base. For the option-code rows A-Z (rows 0-25) and rows 26-254 (the two-letter codes): the mean row norm, the mean distance
from the base row, and the paired difference of the corrected checkpoint against the osler_v4 final (row by row, then averaged). The base
readout is pplx's code-token rows of the tied embedding (decision_config's token_ids; the LM head is tied). Only weights are read; no MIMIC
text is loaded. Writes outputs/osler_v5/Z2/readout_scale_<label>.json.

Usage (CPU):
    uv run --frozen python scripts/osler_v5/readout_scale.py --checkpoint outputs/osler_v5/Z2/F2/checkpoints/step_006996 --label F2_6996
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from meddecide.model.pplx import REPO as PPLX_REPO
from meddecide.model.pplx import checkpoint_dir
from meddecide.utils.io import write_json

REPO = Path(__file__).resolve().parents[2]
OSLER_V4_FINAL = REPO / "outputs" / "osler_v4" / "Y8" / "checkpoints" / "best" / "optcode.pt"
EMBED = "language_model.embed_tokens.weight"


def base_readout() -> np.ndarray:
    ckpt = checkpoint_dir(PPLX_REPO)
    dc = json.loads((ckpt / "decision_config.json").read_text(encoding="utf-8"))
    ids = [int(i) for i in dc["token_ids"]]
    index = json.loads((ckpt / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    shard = ckpt / index[EMBED]
    with safe_open(str(shard), framework="pt") as fh:
        table = fh.get_tensor(EMBED).float()
    return table[torch.tensor(ids, dtype=torch.long)].numpy()


def readout(path: Path) -> np.ndarray:
    target = path if path.suffix == ".pt" else path / "optcode.pt"
    return torch.load(target, map_location="cpu", weights_only=True)["weight"].float().numpy()


def groups(w: np.ndarray, base: np.ndarray) -> dict:
    norms = np.linalg.norm(w, axis=1)
    dist = np.linalg.norm(w - base, axis=1)
    out = {}
    for name, sl in (("A_Z_rows_0_25", slice(0, 26)), ("rows_26_254", slice(26, 255))):
        out[name] = {"mean_row_norm": float(norms[sl].mean()), "mean_distance_from_base": float(dist[sl].mean())}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    base = base_readout()
    final = readout(OSLER_V4_FINAL)
    corrected = readout(REPO / args.checkpoint)
    result = {"label": args.label, "checkpoint": args.checkpoint, "base": "pplx code-token rows of the tied embedding",
              "osler_v4_final": groups(final, base), "corrected": groups(corrected, base)}
    for name, sl in (("A_Z_rows_0_25", slice(0, 26)), ("rows_26_254", slice(26, 255))):
        d_norm = np.linalg.norm(corrected[sl], axis=1) - np.linalg.norm(final[sl], axis=1)
        d_dist = np.linalg.norm(corrected[sl] - base[sl], axis=1) - np.linalg.norm(final[sl] - base[sl], axis=1)
        result.setdefault("paired_corrected_minus_osler_v4", {})[name] = {
            "mean_row_norm_difference": float(d_norm.mean()), "mean_distance_difference": float(d_dist.mean()),
            "rows_with_larger_distance": int((d_dist > 0).sum()), "rows": int(d_dist.size)}
    result["base_row_norm_mean"] = {"A_Z_rows_0_25": float(np.linalg.norm(base[:26], axis=1).mean()),
                                    "rows_26_254": float(np.linalg.norm(base[26:], axis=1).mean())}
    write_json(REPO / "outputs" / "osler_v5" / "Z2" / f"readout_scale_{args.label}.json", result)
    print(json.dumps(result, indent=1, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
