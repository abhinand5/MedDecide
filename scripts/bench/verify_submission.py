#!/usr/bin/env python
"""Verify a community submission against `abhinand/MedDecide-Bench` (`confirm_general`, 5,988 items).

Recomputes micro accuracy (+95% bootstrap CI), 12-template macro accuracy, multi-class Brier,
15-bin ECE, coverage, and D12/D21 readout health directly from per-item prediction probabilities,
then optionally emits a ready-to-PR submission JSON for `site/submissions/<org>__<model>.json`.

Prediction JSONL format (one line per scored item):
    {"item_id": "0005c8143090f0e6", "probs": {"A": 0.91, "B": 0.04, "C": 0.03, "D": 0.02}}
    # OR list aligned to item["options"]:
    {"item_id": "0005c8143090f0e6", "probs": [0.91, 0.04, 0.03, 0.02]}
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def load_confirm_general(local_jsonl: Path | None = None) -> list[dict[str, Any]]:
    if local_jsonl is not None and local_jsonl.is_file():
        return [json.loads(line) for line in local_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    from datasets import load_dataset

    ds = load_dataset("abhinand/MedDecide-Bench", "confirm_general", split="test")
    return [dict(row) for row in ds]


def compute_ece(confs: np.ndarray, corrects: np.ndarray, n_bins: int = 15) -> float:
    if len(confs) == 0:
        return float("nan")
    bin_idx = np.minimum((confs * n_bins).astype(int), n_bins - 1)
    total = 0.0
    n = len(confs)
    for b in range(n_bins):
        mask = bin_idx == b
        cnt = int(mask.sum())
        if cnt:
            total += (cnt / n) * abs(float(corrects[mask].mean()) - float(confs[mask].mean()))
    return total


def bootstrap_ci(values: np.ndarray, n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def verify(gold_items: list[dict[str, Any]], pred_path: Path) -> dict[str, Any]:
    gold_by_id = {str(item["item_id"]): item for item in gold_items}
    preds_by_id: dict[str, dict[str, Any]] = {}
    with pred_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            iid = str(row["item_id"])
            if iid not in gold_by_id:
                raise ValueError(f"Line {lineno}: unknown item_id {iid!r} not in confirm_general")
            if iid in preds_by_id:
                raise ValueError(f"Line {lineno}: duplicate item_id {iid!r}")
            preds_by_id[iid] = row

    if not preds_by_id:
        raise ValueError("Prediction file is empty")

    corrects: list[float] = []
    briers: list[float] = []
    confs: list[float] = []
    by_template: dict[str, list[tuple[int, int, int]]] = defaultdict(list)

    for iid, pred in preds_by_id.items():
        item = gold_by_id[iid]
        keys = [str(o["key"]) for o in item["options"]]
        gold_idx = keys.index(str(item["gold"]))
        raw_probs = pred["probs"]
        if isinstance(raw_probs, dict):
            probs = [float(raw_probs[k]) for k in keys]
        else:
            probs = [float(p) for p in raw_probs]
        if len(probs) != len(keys):
            raise ValueError(f"{iid}: expected {len(keys)} probs, got {len(probs)}")
        if any(not math.isfinite(p) or p < -1e-6 for p in probs):
            raise ValueError(f"{iid}: invalid probability values {probs}")
        s = sum(probs)
        if abs(s - 1.0) > 1e-2:
            raise ValueError(f"{iid}: probabilities sum to {s:.4f}, expected ~1.0")
        probs = [p / s for p in probs]

        pred_idx = max(range(len(probs)), key=lambda i: probs[i])
        is_correct = 1.0 if pred_idx == gold_idx else 0.0
        brier = sum((p - (1.0 if i == gold_idx else 0.0)) ** 2 for i, p in enumerate(probs))
        corrects.append(is_correct)
        briers.append(brier)
        confs.append(max(probs))
        by_template[str(item["template_id"])].append((pred_idx, gold_idx, len(keys)))

    c_arr = np.asarray(corrects, dtype=np.float64)
    b_arr = np.asarray(briers, dtype=np.float64)
    f_arr = np.asarray(confs, dtype=np.float64)
    ci_lo, ci_hi = bootstrap_ci(c_arr)

    tpl_accs: list[float] = []
    flags: list[str] = []
    for tpl, records in sorted(by_template.items()):
        t_corr = np.asarray([1.0 if p == g else 0.0 for p, g, _ in records], dtype=np.float64)
        t_acc = float(t_corr.mean())
        tpl_accs.append(t_acc)
        _, t_hi = bootstrap_ci(t_corr, n_resamples=1000)
        mean_chance = float(np.mean([1.0 / n for _, _, n in records]))
        if t_hi < mean_chance:
            flags.append(f"{tpl}: below_chance (acc={t_acc:.3f}, ci_hi={t_hi:.3f} < chance={mean_chance:.3f})")

    n_universe = len(gold_items)
    scored = len(preds_by_id)
    return {
        "n_universe": n_universe,
        "scored": scored,
        "coverage": round(scored / n_universe, 4),
        "accuracy": round(float(c_arr.mean()), 4),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "macro": round(float(np.mean(tpl_accs)), 4),
        "brier": round(float(b_arr.mean()), 4),
        "ece": round(compute_ece(f_arr, c_arr), 4),
        "readout_health": "PASS" if not flags else f"READOUT_FAIL — {len(flags)} templates",
        "readout_flags": flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions JSONL")
    parser.add_argument("--gold-jsonl", type=Path, default=None, help="Optional local confirm/general.jsonl")
    parser.add_argument("--model-name", type=str, default=None, help="Display name for submission JSON")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face repo ID (e.g. org/model)")
    parser.add_argument("--params-b", type=float, default=None, help="Parameter count in billions")
    parser.add_argument("--license", type=str, default="Apache-2.0")
    parser.add_argument("--out-submission", type=Path, default=None, help="Write ready-to-PR submission JSON")
    args = parser.parse_args()

    gold = load_confirm_general(args.gold_jsonl)
    metrics = verify(gold, args.predictions)
    print(json.dumps(metrics, indent=2))

    if args.out_submission is not None:
        if not args.model_name or not args.hf_repo or args.params_b is None:
            raise SystemExit("--out-submission requires --model-name, --hf-repo, and --params-b")
        tier = "<=4B" if args.params_b <= 4.6 else ("<=10B" if args.params_b <= 10.0 else ">10B")
        sub = {
            "id": args.model_name,
            "raw_id": args.model_name,
            "is_osler": False,
            "verification_tier": "public_verified",
            "params_b": round(float(args.params_b), 2),
            "size_tier": tier,
            "role": "Community Submission (Public Tier)",
            "base_repo": args.hf_repo,
            "hf_repo": args.hf_repo,
            "hf_url": f"https://huggingface.co/{args.hf_repo}",
            "readout_type": "community_submitted",
            "non_generative": False,
            "license": args.license,
            "submitted_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "p50_ms_b1": None,
            "peak_vram_gb": None,
            "general_medical_avg": None,
            "ehr_docs_avg": None,
            "five_set_macro": None,
            "on_pareto_front": False,
            "undominated_le10b": False,
            "undominated_le4b": False,
            "latency": {},
            "sets": {
                "confirm_general": {
                    "status": "MEASURED",
                    "n": metrics["n_universe"],
                    "scored": metrics["scored"],
                    "coverage": metrics["coverage"],
                    "accuracy": metrics["accuracy"],
                    "ci_lo": metrics["ci_lo"],
                    "ci_hi": metrics["ci_hi"],
                    "macro": metrics["macro"],
                    "brier": metrics["brier"],
                    "ece": metrics["ece"],
                    "readout_health": metrics["readout_health"],
                }
            },
        }
        args.out_submission.parent.mkdir(parents=True, exist_ok=True)
        args.out_submission.write_text(json.dumps(sub, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote submission JSON -> {args.out_submission}")


if __name__ == "__main__":
    main()
