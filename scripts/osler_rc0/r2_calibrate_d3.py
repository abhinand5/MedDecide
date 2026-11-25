#!/usr/bin/env python
"""osler_rc0 R2 — dev-only per-qtype temperature calibration for Osler-9B-rc0 (CPU).

Two subcommands:

``fit``   join a prediction file to the dev item file by the recomputed item id, fit one temperature per qtype on
          ``fastdev_v4`` only, and write ``9b_calibration.json`` with dev pre/post NLL, accuracy, Brier and ECE.
``apply`` re-scale a prediction file's ``probs`` with frozen dev temperatures and write a calibrated copy; the argmax
          is checked row by row (it can never change for a positive temperature, and the check proves it did not).

Run::

    uv run --frozen python scripts/osler_rc0/r2_calibrate_d3.py fit \
        --items outputs/osler_rc0/R2/fastdev_v4.jsonl \
        --predictions outputs/osler_rc0/R2/preds/9b_rc0_fastdev_v4/predictions.jsonl \
        --out outputs/osler_rc0/R2/9b_calibration.json
    uv run --frozen python scripts/osler_rc0/r2_calibrate_d3.py apply \
        --items data/bench/osler_v4/y9/panel.jsonl \
        --predictions outputs/eval0/panel/9b_final_panel/predictions.jsonl \
        --calibration outputs/osler_rc0/R2/9b_calibration.json \
        --out outputs/osler_rc0/R2/preds_cal/9b_rc0_panel/predictions.jsonl \
        --summary outputs/osler_rc0/R2/preds_cal/9b_rc0_panel/calibration_apply.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.calibration import (  # noqa: E402
    apply_temperature,
    fit_temperature,
    summary,
)
from meddecide.train.mix_items import mix_row_to_item  # noqa: E402
from meddecide.utils.io import write_json  # noqa: E402

QTYPES = ("choice", "noul", "score")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def item_qtypes(items_path: Path) -> dict[str, str]:
    """recomputed item id -> qtype, from the frozen item file."""
    rows = read_jsonl(items_path)
    out = {mix_row_to_item(row).item_id: str(row["qtype"]) for row in rows}
    if len(out) != len(rows):
        raise SystemExit(f"{items_path}: duplicate recomputed item ids")
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def join(predictions: list[dict[str, Any]], qtypes: dict[str, str], source: Path) -> list[tuple[dict[str, Any], str]]:
    joined = []
    missing = 0
    for row in predictions:
        qtype = qtypes.get(str(row["item_id"]))
        if qtype is None:
            missing += 1
            continue
        joined.append((row, qtype))
    if missing:
        raise SystemExit(f"{source}: {missing} prediction ids do not map into the item file: stop")
    if not joined:
        raise SystemExit(f"{source}: no usable rows")
    return joined


def cmd_fit(args: argparse.Namespace) -> None:
    qtypes = item_qtypes(args.items)
    rows = join(read_jsonl(args.predictions), qtypes, args.predictions)
    by_qtype: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for row, qtype in rows:
        by_qtype.setdefault(qtype, []).append((row, qtype))
    temperatures: dict[str, float] = {}
    per_qtype: dict[str, Any] = {}
    for qtype, group in sorted(by_qtype.items()):
        gold = [int(row["gold_index"]) for row, _ in group]
        probs = [row["probs"] for row, _ in group]
        temperature = fit_temperature(probs, gold)
        calibrated = apply_temperature(probs, temperature)
        argmax_before = np.asarray([np.asarray(row, dtype=np.float64).argmax() for row in probs], dtype=np.int64)
        argmax_after = np.asarray([np.asarray(row, dtype=np.float64).argmax() for row in calibrated], dtype=np.int64)
        if not np.array_equal(argmax_before, argmax_after):
            raise SystemExit(f"{qtype}: calibration changed an argmax on dev: stop")
        temperatures[qtype] = temperature
        per_qtype[qtype] = {
            "n_items": len(group),
            "temperature": round(temperature, 6),
            "pre": {k: round(v, 6) if isinstance(v, float) else v for k, v in summary(probs, gold).items()},
            "post": {k: round(v, 6) if isinstance(v, float) else v for k, v in summary(calibrated, gold).items()},
        }
    missing = [q for q in QTYPES if q not in temperatures]
    all_gold = [int(row["gold_index"]) for row, _ in rows]
    all_probs = [row["probs"] for row, _ in rows]
    all_cal = [apply_temperature([row["probs"]], temperatures[qtype])[0] for row, qtype in rows]
    out = {
        "fit_set": str(args.items),
        "fit_predictions": str(args.predictions),
        "fit_predictions_sha256": sha256_file(args.predictions),
        "fit_items_sha256": sha256_file(args.items),
        "temperatures": {k: round(v, 6) for k, v in temperatures.items()},
        "qtypes_missing": missing,
        "per_qtype": per_qtype,
        "all_items": {
            "n_items": len(rows),
            "pre": {k: round(v, 6) if isinstance(v, float) else v for k, v in summary(all_probs, all_gold).items()},
            "post": {k: round(v, 6) if isinstance(v, float) else v for k, v in summary(all_cal, all_gold).items()},
        },
        "argmax_changes_on_dev": 0,
        "definition": ("p' = p ** (1/T) renormalised (exact softmax(logit/T) when p = softmax(logit)); fitted by minimising "
                       "the mean NLL of the gold option on fastdev_v4 only; strictly monotone, so argmax is unchanged."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, out)
    print(json.dumps({"temperatures": out["temperatures"], "pre_nll": out["all_items"]["pre"]["nll"],
                      "post_nll": out["all_items"]["post"]["nll"], "pre_ece": out["all_items"]["pre"]["ece"],
                      "post_ece": out["all_items"]["post"]["ece"]}, indent=2))


def cmd_apply(args: argparse.Namespace) -> None:
    calibration = json.loads(args.calibration.read_text())
    temperatures = calibration["temperatures"]
    qtypes = item_qtypes(args.items)
    rows = read_jsonl(args.predictions)
    joined = join(rows, qtypes, args.predictions)
    if len(joined) != len(rows):
        raise SystemExit("apply: prediction rows missing an item in the item file: stop")
    out_rows = []
    argmax_changes = 0
    max_delta = 0.0
    gold: list[int] = []
    probs_before: list[list[float]] = []
    probs_after: list[list[float]] = []
    for row, qtype in joined:
        before = np.asarray(row["probs"], dtype=np.float64)
        after = apply_temperature(before, temperatures[qtype])
        if int(before.argmax()) != int(after.argmax()):
            argmax_changes += 1
        max_delta = max(max_delta, float(np.abs(after - before / before.sum()).max()))
        gold.append(int(row["gold_index"]))
        probs_before.append((before / before.sum()).tolist())
        probs_after.append(after.tolist())
        out_rows.append({**row, "probs": [float(p) for p in after], "temperature": temperatures[qtype]})
    if argmax_changes:
        raise SystemExit(f"apply: {argmax_changes} argmax changes: stop (temperature must be monotone)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row) + "\n")
    pre = summary(probs_before, gold)
    post = summary(probs_after, gold)
    result = {
        "items": str(args.items),
        "predictions": str(args.predictions),
        "predictions_sha256": sha256_file(args.predictions),
        "calibration": str(args.calibration),
        "rows": len(out_rows),
        "argmax_changes": argmax_changes,
        "max_abs_probability_shift_before_renormalisation": round(max_delta, 8),
        "temperatures_used": temperatures,
        "pre": {k: round(v, 6) if isinstance(v, float) else v for k, v in pre.items()},
        "post": {k: round(v, 6) if isinstance(v, float) else v for k, v in post.items()},
    }
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.summary, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--items", type=Path, required=True)
    fit.add_argument("--predictions", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    apply = sub.add_parser("apply")
    apply.add_argument("--items", type=Path, required=True)
    apply.add_argument("--predictions", type=Path, required=True)
    apply.add_argument("--calibration", type=Path, required=True)
    apply.add_argument("--out", type=Path, required=True)
    apply.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()
    for name in ("items", "predictions", "out"):
        setattr(args, name, getattr(args, name) if getattr(args, name).is_absolute() else ROOT / getattr(args, name))
    if args.command == "fit":
        cmd_fit(args)
    else:
        args.calibration = args.calibration if args.calibration.is_absolute() else ROOT / args.calibration
        args.summary = None if args.summary is None else (
            args.summary if args.summary.is_absolute() else ROOT / args.summary)
        cmd_apply(args)


if __name__ == "__main__":
    main()
