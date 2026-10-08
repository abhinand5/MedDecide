#!/usr/bin/env python
"""Fit per-qtype temperatures on the V4 dev set for one saved letter-readout checkpoint (arm B, V7/V9).

Used when the checkpoint to be evaluated is not the trainer's own selected one (the trainer fits temperatures
only for ``best/``). Dev only: no test item is read. Uses the repository's ``fit_per_qtype`` on the no-padding
scoring of the dev set, the same fit the trainer applies. Writes ``<out>`` as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.bench.schema import Item  # noqa: E402
from meddecide.model.meddecide_model import MedDecideModel  # noqa: E402
from meddecide.train.temperature import fit_per_qtype  # noqa: E402


def shown(path: Path) -> str:
    """The path relative to the repository when it is inside it, else as given."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dev", type=Path, default=ROOT / "data" / "train" / "student_v1" / "dev.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    items = [Item.model_validate_json(line) for line in args.dev.open(encoding="utf-8")]
    model = MedDecideModel.load(args.checkpoint, device="cuda:0", dtype="bfloat16")
    scored = model.score_items(items, batch_size=1, round_to_chunk=False)
    fits = fit_per_qtype(scored)
    payload = {
        "kind": "letter_temperature_fit",
        "checkpoint": shown(args.checkpoint),
        "dev": shown(args.dev),
        "dev_items": len(items),
        "fits": {q: {"temperature": f.temperature, "n_items": f.n_items, "nll_before": f.nll_before,
                     "nll_after": f.nll_after} for q, f in fits.items()},
        "temperature": {q: f.temperature for q, f in fits.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["temperature"]))


if __name__ == "__main__":
    main()
