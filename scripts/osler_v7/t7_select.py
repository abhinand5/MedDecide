#!/usr/bin/env python
"""osler_v7 T7 — Lever-2 interpolation and the pre-registered selection for the 4B arm (ADVISORY §4, §7).

For s in {0.25, 0.5, 0.75, 1.0} the model is blended toward its base: the LoRA scaling is multiplied by s and the readout
is ``(1 - s) * R_base + s * R_trained`` (the osler_v5 `Blend` semantics, both deltas together as §4 specifies). The
zero-shot base is the s = 0 blend. Every candidate is scored once on the frozen format-matched dev set (all 22,934
items); the composite S (MR and MIMIC template macros + CG v2 micro, osler_v5's S) is computed from the same
partitions, and the paired template-macro difference against the base is bootstrapped (1,000 resamples, seed 0).

Selection rule (§7): a candidate is eligible when the lower bound of that paired difference is > -0.01
(non-inferiority); among eligible candidates the highest dev S is chosen. If none is eligible the arm is reported and
stops; this script selects no gate and reads no test, panel, MIMIC-audit or held-out item.

Run (GPU; one job; detached after T6)::

    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v7/t7_select.py \
        --arm outputs/osler_v7/T6/arm --out outputs/osler_v7/T7 \
        > outputs/osler_v7/T7/logs/select.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.composite import composite  # noqa: E402
from meddecide.model.blend import Blend, export_blend  # noqa: E402
from meddecide.model.pplx import PplxDecisionModel  # noqa: E402
from meddecide.train.blocks import macro_difference, pair_by_item  # noqa: E402
from meddecide.train.config import load_config  # noqa: E402
from meddecide.train.mix_items import mix_row_to_item  # noqa: E402
from meddecide.train.osler_arm import dev_prediction_rows  # noqa: E402
from meddecide.train.trainer import Trainer  # noqa: E402
from meddecide.utils.io import write_json  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

DEV = ROOT / "data" / "bench" / "osler_v7" / "dev_format_matched.jsonl"
SCALES = (0.25, 0.5, 0.75, 1.0)
NON_INFERIORITY = -0.01
OUT = ROOT / "outputs" / "osler_v7" / "T7"


def apply_blend(model: PplxDecisionModel, blend: Blend, scale: float) -> None:
    """Set the blend weight for both deltas (LoRA scaling and readout), as ADVISORY §4 specifies."""
    if not 0.0 <= scale <= 1.0:
        raise ValueError("the blend scale must be in [0, 1]")
    with torch.no_grad():
        readout = (1.0 - scale) * blend.base_readout + scale * blend.trained_readout
        model.optcode.weight.copy_(readout.to(device=model.optcode.weight.device, dtype=model.optcode.weight.dtype))
        for module, scales in blend.base_scaling:
            for name, value in scales.items():
                module.scaling[name] = value * scale


def score_dev(model: PplxDecisionModel, items: list[Any], *, batch_size: int, max_batch_tokens: int) -> list[dict[str, Any]]:
    scored = model.score_items(items, batch_size=batch_size, max_batch_tokens=max_batch_tokens)
    return dev_prediction_rows(scored)


def metrics(rows: list[dict[str, Any]], base_rows: list[dict[str, Any]], origins: dict[str, str]) -> dict[str, Any]:
    """The composite S and the paired non-inferiority difference of one candidate against the base."""
    fast = [row for row in rows if origins[str(row["item_id"])] == "fastdev_v4"]
    cg = [row for row in rows if origins[str(row["item_id"])] == "cg_v2"]
    parts = composite(fast, cg)
    diff = macro_difference(pair_by_item(rows, base_rows))
    item_diff = sum(bool(row["correct"]) for row in rows) / len(rows) - sum(
        bool(row["correct"]) for row in base_rows) / len(base_rows)
    return {
        "S": parts.to_dict()["S"],
        "composite": parts.to_dict(),
        "accuracy_item": round(sum(bool(row["correct"]) for row in rows) / len(rows), 8),
        "paired_template_macro_difference": diff.as_dict(),
        "paired_item_accuracy_difference": round(item_diff, 8),
        "eligible": bool(diff.lo > NON_INFERIORITY),
    }


def choose_scale(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """The pre-registered §7 rule over the per-scale records: highest S among eligible, else no choice."""
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        return {"chosen_scale": None, "reason": "no candidate is eligible (lower bound not above -0.01)"}
    best = max(eligible, key=lambda candidate: candidate["S"])
    return {"chosen_scale": best["scale"], "reason": "highest dev S among eligible candidates",
            "S": best["S"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True, help="the arm directory (with checkpoints/step_013992)")
    parser.add_argument("--step", type=int, default=13_992)
    parser.add_argument("--dev", type=Path, default=DEV)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--resume", action="store_true", help="reuse the per-scale prediction files already written")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "osler_v7" / "arm_4b.yaml")
    parser.add_argument("--no-export", action="store_true", help="skip calibration and the export of the chosen blend")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)
    checkpoint = args.arm / "checkpoints" / f"step_{args.step:06d}"
    if not checkpoint.is_dir():
        raise SystemExit(f"{checkpoint} is missing: T7 depends on the T6 arm's step-{args.step} checkpoint")
    if (args.out / "selection.json").exists() and not args.resume:
        raise SystemExit(f"{args.out}/selection.json exists: the selection is run once")
    with args.dev.open(encoding="utf-8") as handle:
        dev_rows = [json.loads(line) for line in handle if line.strip()]
    items = [mix_row_to_item(row) for row in dev_rows]
    origins = {item.item_id: str(row.get("dev_origin")) for item, row in zip(items, dev_rows, strict=True)}
    items.sort(key=lambda item: len(item.state) + len(item.question))
    print(f"[t7] {len(items)} dev items; checkpoint {checkpoint}", file=sys.stderr, flush=True)

    model = PplxDecisionModel.load(checkpoint, device=args.device, dtype=args.dtype)
    model.eval_mode()
    blend = Blend(model)
    started = time.time()

    base_path = args.out / "predictions" / "scale_0.jsonl"
    base_path.parent.mkdir(exist_ok=True)
    if args.resume and base_path.exists():
        with base_path.open(encoding="utf-8") as handle:
            base_rows = [json.loads(line) for line in handle if line.strip()]
    else:
        apply_blend(model, blend, 0.0)
        base_rows = score_dev(model, items, batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens)
        with base_path.open("w", encoding="utf-8") as handle:
            for row in base_rows:
                handle.write(json.dumps(row) + "\n")
    print(f"[t7] base scored ({len(base_rows)} rows, {time.time() - started:.0f}s)", file=sys.stderr, flush=True)

    candidates: list[dict[str, Any]] = []
    for scale in SCALES:
        path = args.out / "predictions" / f"scale_{scale:g}.jsonl"
        if args.resume and path.exists():
            with path.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        else:
            apply_blend(model, blend, scale)
            t0 = time.time()
            rows = score_dev(model, items, batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens)
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            print(f"[t7] scale {scale:g} scored in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        record = {"scale": scale, **metrics(rows, base_rows, origins)}
        candidates.append(record)
        print(f"[t7] scale {scale:g}: S {record['S']:.6f} diff lower "
              f"{record['paired_template_macro_difference']['ci_lo']:.6f} eligible {record['eligible']}",
              file=sys.stderr, flush=True)
    result = {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t7_select.py",
        "arm": str(args.arm.relative_to(ROOT)),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "dev": {"path": str(args.dev.relative_to(ROOT)), "items": len(items)},
        "rule": {"scales": list(SCALES), "non_inferiority_lower_bound": NON_INFERIORITY,
                 "statistic": "paired template-macro difference against the zero-shot base, 1,000 resamples, seed 0",
                 "choice": "highest dev S among eligible"},
        "candidates": candidates,
        "choice": choose_scale(candidates),
        "env": {"device": args.device, "dtype": args.dtype, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "wall_seconds": round(time.time() - started, 1),
    }
    chosen = result["choice"].get("chosen_scale")
    if chosen is not None and not args.no_export:
        # Calibrate the chosen blend on the fast-dev partition only (never on test/confirmation), then bake the scale
        # into the weights and export a standard checkpoint for the gate (osler_v5's Blend/export_blend semantics).
        apply_blend(model, blend, float(chosen))
        fast_rows = [row for row in dev_rows if str(row.get("dev_origin")) == "fastdev_v4"]
        config = load_config(args.config)
        fits = Trainer(model, config).fit_calibration([mix_row_to_item(row) for row in fast_rows])
        result["calibration"] = {"items": len(fast_rows),
                                 "fits": {key: fit.to_dict() for key, fit in fits.items()}}
        export_blend(blend, float(chosen))
        model.save(args.out / "chosen")
        result["chosen_checkpoint"] = str((args.out / "chosen").relative_to(ROOT))
        result["chosen_model_meta"] = {"temperature_per_qtype": config.temperature_per_qtype}
    write_json(args.out / "selection.json", result)
    print(json.dumps({"choice": result["choice"], "calibration": result.get("calibration")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
