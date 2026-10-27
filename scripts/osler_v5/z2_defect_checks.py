"""osler_v5 Z2 defect checks (CPU): the measured facts behind the R0 prime collapse at step 6,996 and the readout rows the mix trains.
(1) DDXPlus predicted answers per arm and step (the modal answers and the two-letter gold items); (2) the option-count histogram of
the composed mix and the rows with more than 12 options; (3) the readout rows that changed between two checkpoints of an arm.
Writes outputs/osler_v5/Z2/defect_checks.json. Usage (CPU):
    uv run --frozen python scripts/osler_v5/z2_defect_checks.py"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from meddecide.train.final_mix import compose_final_mix
from meddecide.utils.io import write_json

REPO = Path(__file__).resolve().parents[2]
Z2 = REPO / "outputs" / "osler_v5" / "Z2"


def rows_of(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def ddx_answers(arm: str, step: str) -> dict:
    cg = rows_of(Z2 / arm / "evals" / f"step_{int(step):06d}" / "cg_v2.jsonl")
    ddx = [r for r in cg if str(r["template_id"]).startswith("cg_v2_ddxplus")]
    two = [r for r in ddx if int(r["gold_index"]) >= 26]
    top = Counter(r["predicted_key"] for r in ddx).most_common(5)
    top_two = Counter(r["predicted_key"] for r in two).most_common(5)
    return {"arm": arm, "step": int(step), "ddx_items": len(ddx), "predicted_top5": top,
            "modal_share": top[0][1] / len(ddx), "two_letter_gold_items": len(two),
            "two_letter_gold_predicted_top5": top_two, "two_letter_correct": sum(bool(r["correct"]) for r in two)}


def option_histogram() -> dict:
    spec = importlib.util.spec_from_file_location("y8", REPO / "scripts/osler_v4/y8_final_mix.py")
    assert spec is not None and spec.loader is not None
    y8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(y8)
    dec = {n: y8.entering(n) for n in ("B1", "B2", "B3", "B4")}
    family = {n: y8.add_files(n) for n in y8.FAMILY if dec[n][0]}
    other = y8.add_files("B4") if dec["B4"][0] else []
    rows, _ = compose_final_mix(y8.BASE, family, other, y8.BROAD)
    counts = Counter(len(r["options"]) for r in rows)
    wide = [r for r in rows if len(r["options"]) > 12]
    return {"rows": len(rows), "histogram": {str(k): v for k, v in sorted(counts.items())},
            "rows_over_12_options": len(wide),
            "templates_over_12_options": dict(Counter(str(r["template_id"]) for r in wide)),
            "max_options": max(counts)}


def changed_rows(arm: str, step_a: int, step_b: int) -> dict:
    a = torch.load(Z2 / arm / "full" / f"step_{step_a:06d}" / "trainable.pt", map_location="cpu", weights_only=False)
    a_w = next(v.float().numpy() for v in a.values() if hasattr(v, "shape") and tuple(v.shape) == (255, 2560))
    b_w = torch.load(Z2 / arm / "checkpoints" / f"step_{step_b:06d}" / "optcode.pt", map_location="cpu", weights_only=True)["weight"].float().numpy()
    d = np.linalg.norm(b_w - a_w, axis=1)
    changed = [i for i in range(len(d)) if d[i] > 1e-6]
    return {"arm": arm, "from_step": step_a, "to_step": step_b, "rows_changed": changed,
            "max_changed_row": max(changed) if changed else None, "distance_rows_12_to_17": [round(float(x), 4) for x in d[12:18]]}


def training_window(arm: str, lo: int = 5000, hi: int = 7000) -> dict:
    """The logged training statistics of an arm between two steps (CE mean, grad-norm median and maximum, train accuracy mean)."""
    recs = [json.loads(line) for line in (Z2 / arm / "logs" / "train.jsonl").open(encoding="utf-8") if line.strip()]
    window = [r for r in recs if r.get("event") == "step" and lo <= int(r["step"]) <= hi]
    g = np.array([r["grad_norm"] for r in window])
    return {"arm": arm, "window": [lo, hi], "logged_steps": len(window), "ce_mean": float(np.mean([r["ce"] for r in window])),
            "grad_norm_median": float(np.median(g)), "grad_norm_max": float(g.max()),
            "train_accuracy_mean": float(np.mean([r["accuracy"] for r in window]))}


def main() -> None:
    out = {"ddx": [ddx_answers("R0", "6996"), ddx_answers("R0p", "6996"), ddx_answers("R0", "13992")],
           "option_histogram": option_histogram(),
           "training_window": [training_window("R0"), training_window("R0p")]}
    if (Z2 / "R0p" / "full" / "step_002500" / "trainable.pt").exists() and (Z2 / "R0p" / "checkpoints" / "step_006996").exists():
        out["readout_changes"] = changed_rows("R0p", 2500, 6996)
    write_json(Z2 / "defect_checks.json", out)
    print(json.dumps(out, indent=1, sort_keys=True, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
