"""osler_v4 Y9 (parity of the repaired zero-shot pplx-flash run; CPU): the same scorer and flags as the Y4 chain, rerun on the first
items of the MIMIC audit suite, compared with Y4's predictions for the same items (outputs/osler_v4/Y4/baselines/suite_flash).

The Y9 chain's pplx steps failed at processor load (torchvision; STATE §7 of 2026-10-25T01:44Z) and were rerun with the Y4 flags
(`--with pillow==12.3.0 --with "torchvision==0.29.*"`). The gate's MIMIC check takes Y4's suite_flash as its pplx baseline, so the
repaired invocation must reproduce Y4's predictions on the same items. Counts and figures only; no item text, identifier or prediction
is printed or committed. Writes outputs/osler_v4/Y9/parity.json.

Usage (CPU; after the 200-item run in outputs/osler_v4/Y9/parity/parity_pplx_mimic200):
    CUDA_VISIBLE_DEVICES= uv run --frozen python scripts/osler_v4/y9_parity.py
"""

from __future__ import annotations

import json
from pathlib import Path

from meddecide.utils.io import write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
Y4 = REPO / "outputs" / "osler_v4" / "Y4" / "baselines" / "suite_flash" / "predictions.jsonl"
PARITY = REPO / "outputs" / "osler_v4" / "Y9" / "parity" / "parity_pplx_mimic200" / "predictions.jsonl"
OUT = REPO / "outputs" / "osler_v4" / "Y9" / "parity.json"


def rows(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as fh:
        return {str(r["item_id"]): r for r in (json.loads(line) for line in fh if line.strip())}


def main() -> None:
    y4 = rows(Y4)
    new = rows(PARITY)
    shared = [i for i in new if i in y4]
    same_argmax = sum(str(new[i]["predicted_key"]) == str(y4[i]["predicted_key"]) for i in shared)
    same_correct = sum(bool(new[i]["correct"]) == bool(y4[i]["correct"]) for i in shared)
    diffs = [abs(float(p) - float(q)) for i in shared for p, q in zip(new[i]["probs"], y4[i]["probs"], strict=True)]
    same_length = sum(len(new[i]["probs"]) == len(y4[i]["probs"]) for i in shared)
    figures = {
        "generated_utc": utcnow(), "y4_file": str(Y4.relative_to(REPO)), "parity_file": str(PARITY.relative_to(REPO)),
        "parity_items": len(new), "y4_items": len(y4), "items_shared": len(shared),
        "same_option_count": same_length, "same_argmax": same_argmax, "same_correct": same_correct,
        "max_abs_prob_difference": max(diffs) if diffs else None,
        "mean_abs_prob_difference": round(sum(diffs) / len(diffs), 8) if diffs else None,
        "items_with_prob_difference_above_1e-3": sum(
            1 for i in shared if max(abs(float(p) - float(q)) for p, q in zip(new[i]["probs"], y4[i]["probs"], strict=True)) > 1e-3),
    }
    write_json(OUT, figures)
    print(json.dumps({k: v for k, v in figures.items() if k not in ("generated_utc", "y4_file", "parity_file")}, sort_keys=True))


if __name__ == "__main__":
    main()
