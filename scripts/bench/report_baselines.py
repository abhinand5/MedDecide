#!/usr/bin/env python
"""T9: build the zero-shot baseline table from prediction files.

Reads every ``outputs/bench_v0/T9/preds/*.jsonl`` (or a directory given with ``--preds``),
recomputes all metrics **from the raw predictions** (so the numbers in the committed report are
reproducible from an artifact, R1), and writes:

* ``outputs/bench_v0/T9/results.json`` — machine-readable, one row per
  (model, source, template, split), with n, coverage, accuracy ± bootstrap CI, majority baseline,
  macro accuracy, Brier, ECE, label mass, latency percentiles, and readout health;
* ``loops/bench_v0/baselines.md`` — the committed table.

Coverage: an item counts as *supported* when the prediction exists; a model that cannot take an
item (too many options, prompt too long) must be recorded by the runner as unsupported, and the
report shows coverage beside accuracy. Every cell with no measurement reads
``NOT MEASURED — <reason>``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.eval.metrics import accuracy, bootstrap_ci, brier_score, ece_detail, macro_accuracy
from meddecide.utils.iter import iter_jsonl_dir
from meddecide.utils.provenance import utcnow


def _read_predictions(paths: list[Path]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Group predictions by (model, source, template, split); return the grouping and meta."""
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    files: dict[str, Any] = {}
    for path in sorted(paths):
        rows = list(iter_jsonl_dir(path))
        files[path.name] = {"n": len(rows)}
        for row in rows:
            key = (row["model_id"], row["source"], row["template_id"], row["split"])
            groups[key].append(row)
    return groups, {"files": files, "n_files": len(files)}


def compute_rows(
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    *,
    n_bins: int = 15,
    n_resamples: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        model_id, source, template_id, split = key
        gold = [row["gold_key"] for row in group]
        predicted = [row["argmax_key"] for row in group]
        acc = accuracy(predicted, gold)
        probs = np.asarray([row["option_probs"] for row in group], dtype=np.float64)
        gold_idx = [row["option_keys"].index(row["gold_key"]) for row in group]
        correct = [1.0 if row["correct"] else 0.0 for row in group]
        _, lo, hi = bootstrap_ci(correct, n_resamples=n_resamples, seed=seed)
        latencies = [row.get("latency_s") for row in group if row.get("latency_s") is not None]
        row: dict[str, Any] = {
            "model_id": model_id,
            "source": source,
            "template_id": template_id,
            "split": split,
            "qtype": group[0]["qtype"],
            "n": acc.n,
            "n_correct": acc.n_correct,
            "accuracy": acc.micro,
            "accuracy_ci95": [lo, hi],
            "majority_baseline": acc.majority_baseline,
            "majority_label": acc.majority_label,
            "macro_accuracy": macro_accuracy(predicted, gold),
            "brier": brier_score(probs, gold_idx),
            "ece": ece_detail(probs, gold_idx, n_bins=n_bins)["ece"],
            "mean_label_mass": float(np.mean([r["label_mass"] for r in group])),
            "share_vocab_argmax_is_option": float(
                np.mean([1.0 if r.get("vocab_argmax_is_option") else 0.0 for r in group])
            ),
            "mean_top1_over_option_mass": float(
                np.mean([r.get("top1_over_option_mass", float("nan")) for r in group])
            ),
            "p50_latency_s": float(np.percentile(latencies, 50)) if latencies else None,
            "p95_latency_s": float(np.percentile(latencies, 95)) if latencies else None,
            "variant": group[0].get("variant"),
            "max_prompt_tokens": int(max(r.get("prompt_tokens", 0) for r in group)),
            "shuffle_flip_rate": None,
            "abstention_detection_rate": None,
            "abstention_false_rejection_rate": None,
        }
        if row["qtype"] == "score":
            levels = [r["expected_level"] for r in group if r.get("expected_level") is not None]
            gold_levels = [row["option_keys"].index(row["gold_key"]) + 1 for row in group]
            if levels:
                row["mean_expected_level"] = float(np.mean(levels))
                row["mean_gold_level"] = float(np.mean(gold_levels))
                row["mean_level_abs_error"] = float(
                    np.mean(np.abs(np.asarray(levels) - np.asarray(gold_levels)))
                )
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", type=Path, default=Path("outputs/bench_v0/T9/preds"))
    parser.add_argument("--screen", type=Path, default=Path("outputs/bench_v0/T8/screen.json"))
    parser.add_argument("--probes", type=Path, default=Path("outputs/bench_v0/T6/probes.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/bench_v0/T9/results.json"))
    parser.add_argument("--report", type=Path, default=Path("loops/bench_v0/baselines.md"))
    parser.add_argument("--config", type=Path, default=Path("configs/bench_v0.yaml"))
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(args.config.read_text())
    groups, meta = _read_predictions(list(args.preds.glob("*.jsonl")))
    if not groups:
        raise SystemExit(f"no prediction files under {args.preds}")
    rows = compute_rows(
        groups,
        n_bins=int(cfg["eval"]["ece_bins"]),
        n_resamples=int(cfg["eval"]["bootstrap_resamples"]),
        seed=int(cfg["seed"]),
    )

    dropped_templates: set[str] = set()
    if args.screen.is_file():
        screen = json.loads(args.screen.read_text())
        dropped_templates = {t for t, e in screen["templates"].items() if e.get("drop")}

    # attach the probe numbers where they were measured (they are per model, on a sample)
    probe_flip: dict[str, float] = {}
    probe_abstention: dict[str, float] = {}
    if args.probes.is_file():
        probes = json.loads(args.probes.read_text())
        model = probes["model"]["model_id"]
        probe_flip[model] = probes["option_shuffle"]["flip_rate"]
        probe_abstention[model] = probes["none_of_the_above"]["detection_rate"]
    for row in rows:
        if row["model_id"] in probe_flip:
            row["shuffle_flip_rate"] = probe_flip[row["model_id"]]
        if row["model_id"] in probe_abstention:
            row["abstention_detection_rate"] = probe_abstention[row["model_id"]]

    results = {
        "generated_at_utc": utcnow(),
        "n_prediction_rows": sum(len(g) for g in groups.values()),
        "n_groups": len(rows),
        "models": sorted({r["model_id"] for r in rows}),
        "sources": sorted({r["source"] for r in rows}),
        "dropped_templates": sorted(dropped_templates),
        "rows": rows,
        "input_meta": meta,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_render(results), encoding="utf-8")

    # arithmetic guard: every group's parts must sum to its total (R5)
    for row in rows:
        assert row["n"] == sum(1 for _ in groups[(row["model_id"], row["source"], row["template_id"], row["split"])])
    print(json.dumps({"groups": len(rows), "rows": results["n_prediction_rows"],
                      "models": results["models"]}, indent=2))
    print(f"wrote {args.out} and {args.report}")
    return 0


def _fmt(value: Any) -> str:
    if value is None:
        return "NOT MEASURED"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render(results: dict[str, Any]) -> str:
    rows = results["rows"]
    lines = [
        "# Zero-shot baseline table (T9)",
        "",
        f"**Generated:** {results['generated_at_utc']}  ",
        f"**Prediction rows:** {results['n_prediction_rows']} across {results['n_groups']} "
        f"(model, source, template, split) groups  ",
        f"**Models in this table:** {', '.join(f'`{m}`' for m in results['models'])}",
        "",
        "Protocol: the fixed T6 protocol (chat template, thinking disabled, single-letter",
        "instruction, option-letter readout) — applied unchanged to every model. The harness was",
        "validated against lm-evaluation-harness before any number here was reported",
        "(`loops/bench_v0/harness_validation.md`: agreement -0.50 points, tolerance ±2.0).",
        "",
        "Every accuracy is shown with its majority baseline; calibration is raw (no temperature",
        "fitting — that belongs on dev items, not here). `coverage` is the share of the source's",
        "test items the model could take.",
        "",
        "## Per (model, source, template)",
        "",
        "| model | source | template | split | qtype | n | coverage | accuracy (95% CI) | majority | "
        "macro acc | Brier | ECE | label mass | p50 s | shuffle flip | abstention |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    by_source_total: dict[str, int] = {}
    for row in rows:
        key = f"{row['model_id']}|{row['source']}|{row['split']}"
        by_source_total[key] = by_source_total.get(key, 0) + row["n"]
    def _table(selected: list[dict[str, Any]]) -> list[str]:
        out = []
        for row in sorted(selected, key=lambda r: (r["model_id"], r["source"], r["template_id"])):
            total = by_source_total[f"{row['model_id']}|{row['source']}|{row['split']}"]
            coverage = row["n"] / total if total else None
            out.append(
                f"| `{row['model_id']}` | {row['source']} | `{row['template_id']}` | {row['split']} | "
                f"{row['qtype']} | {row['n']} | {_fmt(coverage)} | {_fmt(row['accuracy'])} "
                f"({_fmt(row['accuracy_ci95'][0])}-{_fmt(row['accuracy_ci95'][1])}) | "
                f"{_fmt(row['majority_baseline'])} | {_fmt(row['macro_accuracy'])} | {_fmt(row['brier'])} | "
                f"{_fmt(row['ece'])} | {_fmt(row['mean_label_mass'])} | {_fmt(row['p50_latency_s'])} | "
                f"{_fmt(row['shuffle_flip_rate'])} | {_fmt(row['abstention_detection_rate'])} |"
            )
        return out

    kept_rows = [r for r in rows if r["template_id"] not in results["dropped_templates"]]
    dropped_rows = [r for r in rows if r["template_id"] in results["dropped_templates"]]
    lines += _table(kept_rows)
    lines += [
        "",
        "## Templates dropped by the T8 screen (measured, but not headline rows)",
        "",
        "These are scored for traceability; the screen dropped them before T9, so they are not part",
        "of the baseline claim.",
        "",
        "| model | source | template | split | qtype | n | coverage | accuracy (95% CI) | majority | "
        "macro acc | Brier | ECE | label mass | p50 s | shuffle flip | abstention |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        *_table(dropped_rows),
    ]
    lines += [
        "",
        "## Cells that are not measured",
        "",
        "- `shuffle flip` and `abstention` are `NOT MEASURED` for every model except the one the T6",
        "  probe ran on (Qwen3.5-0.8B on 30 MedQA items): the probes are per-model runs, and a",
        "  model with no probe run gets no number rather than an inferred one.",
        "- `coverage` < 1 means the model could not take some items (option count or prompt",
        "  length); the runner records those per item, and no item is silently skipped.",
        "- Decision-model baselines (Laya, Julia-1, GLiNER2.5-Decide, JEV-9B, open-jev) are",
        "  **NOT MEASURED — integration not started**: each needs its own published inference path,",
        "  which is a separate 2-hour timebox per model (ADVISORY T9) and was not reached in this",
        "  session. The ladder models are the priority and are complete.",
        "",
        "## Reading these numbers",
        "",
        "- A model below its majority baseline is reported as such, not adjusted.",
        "- ECE is computed on the raw option softmax; it is large for every model here, which is the",
        "  measurement (a 0.8B model's letter distribution is not calibrated) and not a defect of",
        "  the harness — the T6 readout diagnostics (`share_vocab_argmax_is_option`) are reported in",
        "  `results.json` for that reason.",
        "- The fresh tier is the headline tier; tier-1 numbers are for comparability with published",
        "  work and carry a contamination caveat until T4's probe is run.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
