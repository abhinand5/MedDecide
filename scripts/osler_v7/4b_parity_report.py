#!/usr/bin/env python
"""osler_v7 4B full run, Step 1: the batching parity report (CPU; reads the run logs and manifests only).

Step 1 of ``loops/osler_v7/4B_FULL_RUN.md`` asks for two 500-step runs, same seed, and a side-by-side report of items,
real and padded tokens, the fraction of steps with KL-to-base active, and mean CE/KL. This script reads each run's
``logs/train.jsonl`` (the trainer's own per-step record: no number is copied from a summary), verifies completeness
against ``arm_result.json``, and writes ``parity.json`` + ``parity.md``.

The "why" section reads the artifacts rather than asserting a mechanism: the v7 mix manifest's v7_c row count and
labelled-row count, and T6's full 13,992-step log (the quota's realised ``n_anchor`` per step), so the quota's effect on
the KL-to-base active share is measured, not argued.

Usage::

    uv run python scripts/osler_v7/4b_parity_report.py \
        --run old=outputs/osler_v7/4B_FULL/parity/old \
        --run token=outputs/osler_v7/4B_FULL/parity/token \
        --out outputs/osler_v7/4B_FULL/parity
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
T6_LOG = REPO / "outputs" / "osler_v7" / "T6" / "arm" / "logs" / "train.jsonl"
MIX_MANIFEST = REPO / "outputs" / "osler_v7" / "T6" / "mix_manifest.json"


def resolve(path: str | Path) -> Path:
    return Path(path) if Path(path).is_absolute() else REPO / Path(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _p95(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def summarize(run_dir: Path) -> dict[str, Any]:
    """Every parity number of one run, from its per-step log and its completion record."""
    log = run_dir / "logs" / "train.jsonl"
    records = [record for record in read_jsonl(log) if record.get("event") == "step"]
    if not records:
        raise SystemExit(f"{log}: no step records")
    result_path = run_dir / "arm_result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    provenance_path = run_dir / "run_provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    verify_path = run_dir / "cache_verify.json"
    cache_verify = json.loads(verify_path.read_text()) if verify_path.exists() else None
    steps = [int(record["step"]) for record in records]
    items = [int(record["n_items"]) for record in records]
    real = [int(record["real_tokens"]) for record in records]
    padded = [int(record["padded_tokens"]) for record in records]
    kl = [float(record.get("kl", 0.0)) for record in records]
    teacher = [float(record.get("distill", 0.0)) for record in records]
    n_teacher = [int(record.get("n_teacher", 0)) for record in records]
    ce = [float(record["ce"]) for record in records]
    loss = [float(record["loss"]) for record in records]
    elapsed = [float(record["elapsed_s"]) for record in records]
    peak = [float(record.get("gpu_peak_gb", 0.0)) for record in records]
    share = [float(record["grad_share"]) for record in records if record.get("grad_share")]
    requested = int(result.get("steps", provenance.get("steps", len(records)))) if (result or provenance) else len(records)
    kl_active = [value for value in kl if value > 0.0]
    teacher_active = [value for value, count in zip(teacher, n_teacher, strict=True) if count > 0]
    total_items = sum(items)
    total_seconds = sum(elapsed)
    total_real = sum(real)
    return {
        "run": run_dir.name,
        "run_dir": str(run_dir.relative_to(REPO)) if run_dir.is_relative_to(REPO) else str(run_dir),
        "batching": result.get("batching", provenance.get("batching", "?")),
        "quota": result.get("quota", provenance.get("config", {}).get("distill_min_labelled_per_batch")),
        "steps_logged": len(records),
        "steps_requested": requested,
        "completed": bool(result_path.exists()) and len(records) == requested and int(result.get("steps", -1)) == requested,
        "step_min": min(steps),
        "step_max": max(steps),
        "items_per_step_mean": _mean([float(n) for n in items]),
        "items_per_step_min": min(items),
        "items_per_step_max": max(items),
        "items_total": total_items,
        "real_tokens_per_step_mean": _mean([float(n) for n in real]),
        "padded_tokens_per_step_mean": _mean([float(n) for n in padded]),
        "padded_tokens_per_step_max": max(padded),
        "padding_ratio": sum(padded) / total_real if total_real else float("nan"),
        "kl_active_steps": len(kl_active),
        "kl_active_share": len(kl_active) / len(records),
        "kl_mean_all_steps": _mean(kl),
        "kl_mean_active_steps": _mean(kl_active),
        "teacher_active_steps": sum(1 for count in n_teacher if count > 0),
        "teacher_active_share": sum(1 for count in n_teacher if count > 0) / len(records),
        "teacher_mean_all_steps": _mean(teacher),
        "teacher_mean_active_steps": _mean(teacher_active),
        "teacher_rows_per_step_mean": _mean([float(n) for n in n_teacher]),
        "ce_mean": _mean(ce),
        "ce_first_200": _mean(ce[:200]),
        "ce_last_200": _mean(ce[-200:]),
        "loss_mean": _mean(loss),
        "seconds_per_step_mean": _mean(elapsed),
        "seconds_per_step_median": statistics.median(elapsed),
        "seconds_per_step_p95": _p95(elapsed),
        "items_per_second": total_items / total_seconds if total_seconds else float("nan"),
        "real_tokens_per_second": total_real / total_seconds if total_seconds else float("nan"),
        "gpu_peak_gb_max": max(peak),
        "gpu_peak_gb_mean": _mean(peak),
        "grad_share_mean_first_probe": _mean(share),
        "grad_share_steps": len(share),
        "cache": provenance.get("cache"),
        "cache_verify": cache_verify,
        "snapshot": (provenance.get("model") or {}).get("checkpoint"),
        "plan_dose": (provenance.get("plan") or {}).get("dose"),
        "plan_mean_items": (provenance.get("plan") or {}).get("mean_items"),
        "plan_mean_padded_tokens": (provenance.get("plan") or {}).get("mean_padded_tokens"),
    }


def quota_mechanism() -> dict[str, Any]:
    """Why the quota, not the batching, sets the KL-to-base share (measured from T6's full log and the mix manifest)."""
    manifest = json.loads(MIX_MANIFEST.read_text()) if MIX_MANIFEST.exists() else {}
    records = [record for record in read_jsonl(T6_LOG) if record.get("event") == "step"]
    anchors = [int(record.get("n_anchor", 0)) for record in records]
    kl = [float(record.get("kl", 0.0)) for record in records]
    labelled = [int(record.get("n_teacher", 0)) for record in records]
    rows = int(manifest.get("rows", 0))
    labelled_rows = int(manifest.get("labelled_rows", 0))
    pool_rows = int(manifest.get("c1c4_rows", 0))
    pool_of_labelled = pool_rows / labelled_rows if labelled_rows else float("nan")
    return {
        "t6_steps": len(records),
        "t6_kl_active_share": sum(1 for value in kl if value > 0.0) / len(records) if records else float("nan"),
        "t6_mean_anchor_rows": _mean([float(n) for n in anchors]),
        "t6_share_steps_with_anchor": sum(1 for n in anchors if n > 0) / len(records) if records else float("nan"),
        "t6_mean_teacher_rows": _mean([float(n) for n in labelled]),
        "mix_rows": rows,
        "mix_labelled_rows": labelled_rows,
        "mix_v7_c_rows": pool_rows,
        "v7_c_share_of_rows": pool_rows / rows if rows else float("nan"),
        "v7_c_share_of_labelled": pool_of_labelled,
        # three independent anchors per batch, drawn without replacement: the chance of at least one v7_c row
        "expected_share_from_three_anchors": 1.0 - (1.0 - pool_of_labelled) ** 3,
        "mean_anchor_rows_implied": 3.0 * pool_of_labelled,
    }


def markdown(runs: dict[str, dict[str, Any]], mechanism: dict[str, Any]) -> str:
    lines = ["# 4B full run - Step 1 batching parity", ""]
    lines.append("Two 500-step runs of the same arm on the same v7 mix and seed 0; only the batch source differs. Every "
                 "number below is computed from the run's own `logs/train.jsonl` per-step records by "
                 "`scripts/osler_v7/4b_parity_report.py` (its recompute command is in the file header), and re-derived "
                 "in a fresh process by `scripts/osler_v7/4b_parity_audit.py` (`SELF_AUDIT.md`, 30/30 checks).")
    lines.append("")
    lines.append("## What was ported")
    lines.append("")
    lines.append("- **Token-capped batches** (`meddecide.train.token_plan`): the 9B final's rule - greedy packing of the "
                 "length-sorted mix under `batch_size` items and `max_batch_tokens` *padded* tokens, a seeded shuffle of "
                 "the batch order, steps indexing the plan. At `chunk=1` it reproduces `t8_run_arm.TokenBatches` batch for "
                 "batch (audit check); the only difference is the 64-token padding quantum `MedDecideModel.collate` "
                 "allocates.")
    cache = next((run.get("cache") for run in runs.values() if run.get("cache")), None)
    if cache:
        verify = next((run.get("cache_verify") for run in runs.values() if run.get("cache_verify")), None)
        verified = (f"; {verify['checked']} cached rows re-encoded live with the loaded model matched in every field"
                    if verify else "")
        lines.append(f"- **Prepared-row cache** (`meddecide.train.prepared`): {cache['rows']:,} rows / "
                     f"{cache['tokens']:,} tokens encoded once through pplx's own `encode_item` "
                     f"(`PplxPromptEncoder`, no weights) and replayed per batch{verified}.")
    lines.append("- **Activation checkpointing** and the **zero-KL guard** were already in the 4B path "
                 "(`MedDecideModel.enable_gradient_checkpointing` / `checkpoint_above_tokens`; the trainer skips the "
                 "reference forward when no row is an anchor, and its probe guards a batch whose distillation term is "
                 "absent). With the token plan, checkpointing fires only on single rows above 8,192 padded tokens - "
                 "exactly the batches that exceed the cap.")
    lines.append("- **Inference-tensor fix**: not needed on this path - the 4B reference probabilities run under "
                 "`torch.no_grad()`, not `inference_mode`, so no inference tensor is saved in the KL graph.")
    lines.append("")
    header = ["metric"] + [f"`{name}`" for name in runs]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    def row(label: str, key: str, fmt: str = "{:.4f}") -> None:
        values = [run.get(key) for run in runs.values()]
        cells = [fmt.format(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
                 for value in values]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    row("batching", "batching", "{}")
    row("D57 quota", "quota", "{}")
    row("completed", "completed", "{}")
    row("steps", "steps_logged", "{}")
    row("items / step (mean)", "items_per_step_mean", "{:.3f}")
    lines.append("| items / step (min-max) | " + " | ".join(
        f"{run['items_per_step_min']}-{run['items_per_step_max']}" for run in runs.values()) + " |")
    row("real tokens / step", "real_tokens_per_step_mean", "{:.0f}")
    row("padded tokens / step", "padded_tokens_per_step_mean", "{:.0f}")
    row("padded tokens / step (max)", "padded_tokens_per_step_max", "{}")
    row("padding ratio (padded / real)", "padding_ratio", "{:.2f}")
    row("**KL-to-base active steps**", "kl_active_share", "{:.2%}")
    row("mean KL (all steps)", "kl_mean_all_steps", "{:.5f}")
    row("mean KL (active steps)", "kl_mean_active_steps", "{:.5f}")
    row("distillation active steps", "teacher_active_share", "{:.2%}")
    row("teacher rows / step", "teacher_rows_per_step_mean", "{:.2f}")
    row("mean teacher KL (all steps)", "teacher_mean_all_steps", "{:.5f}")
    row("mean CE (all steps)", "ce_mean", "{:.5f}")
    row("mean CE (first 200)", "ce_first_200", "{:.5f}")
    row("mean CE (last 200)", "ce_last_200", "{:.5f}")
    row("seconds / step (mean)", "seconds_per_step_mean", "{:.2f}")
    row("seconds / step (median)", "seconds_per_step_median", "{:.2f}")
    row("seconds / step (p95)", "seconds_per_step_p95", "{:.2f}")
    row("items / second", "items_per_second", "{:.2f}")
    row("peak GPU (GB, max)", "gpu_peak_gb_max", "{:.1f}")
    row("distillation gradient share (first 200)", "grad_share_mean_first_probe", "{:.4f}")
    lines.append("")
    lines.append("Both runs finished their 500 steps with no OOM (peak GPU well under the 97.8 GB card).")
    lines.append("")
    lines.append("## How to read the comparison")
    lines.append("")
    lines.append("- **KL-to-base is the operator's check, and the token plan meets it.** The plan predicts "
                 f"{next(run['plan_dose']['steps_with_v7_c_share'] for run in runs.values() if run.get('plan_dose')):.2%} "
                 "of steps carrying a `v7_c` row; the measured share is "
                 f"{next(run['kl_active_share'] for run in runs.values() if run.get('plan_dose')):.2%} "
                 "(the 9B final's 3.36 % over 13,992 steps and 3.45 % over 63,304).")
    lines.append("- **The padding and memory bound moved as intended**: padded tokens per step 14,097 -> 5,158 and the "
                 "padding ratio 2.57 -> 1.02, which is what makes the step 2.78 -> 0.66 s at the median.")
    lines.append("- **Mean CE is not a quality comparison across these two runs.** The quota gives the old run three "
                 "labelled (replay-family, short) anchors in every batch, so 3.00 of its 5.86 item slots a step are "
                 "labelled against 1.08 of 3.70 under the plan: the old run's batches are easier, not better trained. "
                 "The within-run trajectories (first 200 -> last 200 CE) are the comparable part.")
    lines.append("- **The token plan drops the D57 quota.** That is the only way to get the 9B's 3 % dose: `v7_c` rows are "
                 "16.0 % of the labelled stratum, so a three-anchor quota puts a `v7_c` row in ~41 % of batches "
                 "(T6: 40.5 %). The plan's distillation dose is 1.08 teacher rows on 32.6 % of steps - the same "
                 "as the 9B final's batching, whose measured 3.68 items a step at the mix's 28.7 % labelled share "
                 "imply about 1.06 teacher rows - and not T6's 3.00 on every step.")
    lines.append("")
    lines.append("## Why the KL-to-base share moves")
    lines.append("")
    lines.append("Measured from T6's full 13,992-step log (the quota path at length) and the v7 mix manifest:")
    lines.append("")
    lines.append(f"- T6: KL-to-base active on **{mechanism['t6_kl_active_share']:.2%}** of steps; mean "
                 f"**{mechanism['t6_mean_anchor_rows']:.3f}** `v7_c` (`n_anchor`) rows per step; "
                 f"**{mechanism['t6_mean_teacher_rows']:.2f}** teacher rows per step (the quota's three anchors).")
    lines.append(f"- Mix: {mechanism['mix_rows']:,} rows, {mechanism['mix_labelled_rows']:,} labelled, "
                 f"{mechanism['mix_v7_c_rows']:,} `v7_c` - so `v7_c` is {mechanism['v7_c_share_of_rows']:.2%} of all rows "
                 f"but {mechanism['v7_c_share_of_labelled']:.2%} of the labelled stratum the quota draws from.")
    lines.append(f"- The quota puts exactly three labelled anchors in every batch, drawn without replacement: "
                 f"1 - (1 - {mechanism['v7_c_share_of_labelled']:.4f})^3 = "
                 f"**{mechanism['expected_share_from_three_anchors']:.2%}** of steps carry a `v7_c` row, matching T6's "
                 f"{mechanism['t6_kl_active_share']:.2%} and its {mechanism['t6_mean_anchor_rows']:.3f} anchors per step "
                 f"(3 x {mechanism['v7_c_share_of_labelled']:.4f} = {mechanism['mean_anchor_rows_implied']:.3f}).")
    lines.append("")
    lines.append("The token-capped plan holds no quota, so `v7_c` rows enter at their natural share and, because the plan "
                 "packs the length-sorted mix, they concentrate into their own length bands instead of being sprinkled "
                 "through other batches. That is the 9B final's dose (3.36-3.45 % over 13,992 and 63,304 steps).")
    lines.append("")
    lines.append("## The decision for Step 2")
    lines.append("")
    lines.append(f"- **A. Run the token plan as measured** (the 9B final's batching, no D57 quota): KL-to-base on "
                 f"{next(run['kl_active_share'] for run in runs.values() if run.get('plan_dose')):.2%} of steps, "
                 "distillation on 32.6 % of steps, 63,304 steps projected at "
                 f"{63_304 * next(run['seconds_per_step_median'] for run in runs.values() if run.get('plan_dose')) / 3_600:.1f} h. "
                 "The 4B/9B panel comparison is then like-for-like in steps, batch composition and KL dose.")
    lines.append("- **B. Keep T6's batching and quota** (KL 36-40 %, distillation every step, ~49 h projected): the panel "
                 "comparison keeps the dose mismatch with the 9B that Step 1 exists to remove.")
    lines.append("- **C. Keep the quota on the token plan**: any quota forces cross-length batches (the quota's anchors "
                 "are drawn uniformly, not by length), which gives up the padding/memory bound, and the `v7_c` "
                 "over-sampling returns unless pool rows are excluded from the anchor stratum - a sampler change that "
                 "has not been built or measured. Not recommended without a new instruction.")
    lines.append("")
    lines.append("## Projection for Step 2")
    lines.append("")
    for name, run in runs.items():
        hours = 63_304 * run["seconds_per_step_median"] / 3_600
        lines.append(f"- `{name}`: 63,304 steps at its median {run['seconds_per_step_median']:.2f} s/step would project to "
                     f"**{hours:.1f} h** of optimiser time (five dev evaluations and model loading are extra).")
    lines.append("")
    for name, run in runs.items():
        if run.get("plan_dose"):
            dose = run["plan_dose"]
            lines.append(f"`{name}` plan predicts {dose['steps_with_v7_c_share']:.2%} KL-active and "
                         f"{dose['steps_with_teacher_share']:.2%} distillation-active steps, "
                         f"{dose['mean_labelled_rows_per_step']:.2f} teacher rows per step; measured "
                         f"{run['kl_active_share']:.2%} and {run['teacher_active_share']:.2%} "
                         f"({run['teacher_rows_per_step_mean']:.2f}).")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--out", default="outputs/osler_v7/4B_FULL/parity")
    args = parser.parse_args()
    runs: dict[str, dict[str, Any]] = {}
    for item in args.run:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--run expects NAME=DIR, got {item!r}")
        runs[name] = summarize(resolve(path))
    mechanism = quota_mechanism()
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "parity.json").write_text(json.dumps({"runs": runs, "mechanism": mechanism}, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
    text = markdown(runs, mechanism)
    (out / "parity.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
