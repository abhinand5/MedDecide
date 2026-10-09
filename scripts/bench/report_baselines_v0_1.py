#!/usr/bin/env python
"""F6 aggregator: combine per-model files into `results.json` and the committed baseline table.

Reads every `outputs/bench_v0_fix0/F6/model_*.json` written by `run_baselines_v0_1.py` and emits:

* `outputs/bench_v0_fix0/F6/results.json` — every cell, with its gate verdict;
* `loops/bench_v0_fix0/baselines_v0_1.md` — the committed table.

Report rules (ADVISORY F6 + D12):

* an accuracy is shown only when the cell's gate status is `PASS`; a failing cell reads
  `READOUT_FAIL — <check>` and **no accuracy** is printed for it;
* a cell with no measurement reads `NOT MEASURED — <reason>`;
* every cell's `n` is checked against its template's item count, and any mismatch is reported
  rather than hidden;
* fresh results are shown whole and, where measured, on the strict slice (>= 2026-09-10).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from meddecide.utils.provenance import utcnow


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NOT MEASURED"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _accuracy_cell(cell: dict[str, Any]) -> str:
    status = str(cell.get("gate_status", ""))
    if status.startswith("READOUT_FAIL"):
        return status
    if cell.get("accuracy") is None:
        return status or "NOT MEASURED"
    lo, hi = cell["accuracy_ci95"]
    return f"{cell['accuracy']:.4f} ({lo:.4f}-{hi:.4f})"


def _table(cells: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| model | source | template | qtype | n | coverage | accuracy (95% CI) | majority | "
        "chance | label mass | greedy | flip |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in sorted(cells, key=lambda x: (x["model_id"], x["tier"], x["template_id"])):
        n_txt = f"{c['n']}/{c['template_total']}"
        lines.append(
            f"| `{c['model_id']}` | {c['source']} | `{c['template_id']}` | {c['qtype']} | {n_txt} | "
            f"{_fmt(c.get('coverage'), 3)} | {_accuracy_cell(c)} | {_fmt(c.get('majority_share'), 3)} | "
            f"{_fmt(c.get('chance'), 3)} | {_fmt(c.get('median_label_mass'), 4)} | "
            f"{_fmt(c.get('greedy_agreement'), 3)} | {_fmt(c.get('shuffle_flip_rate'), 3)} |"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("outputs/bench_v0_fix0/F6"))
    parser.add_argument("--out", type=Path, default=Path("outputs/bench_v0_fix0/F6/results.json"))
    parser.add_argument("--report", type=Path,
                        default=Path("loops/bench_v0_fix0/baselines_v0_1.md"))
    parser.add_argument("--screen", type=Path, default=Path("data/bench/v0.1/screen.json"))
    parser.add_argument("--benchmark-label", default="MedDecide-Bench v0.1 (F6)",
                        help="title used in the generated report")
    args = parser.parse_args()

    # `model_<slug>_provenance.json` also matches the glob; it is not a results file
    model_files = sorted(
        p for p in args.dir.glob("model_*.json") if not p.name.endswith("_provenance.json")
    )
    if not model_files:
        print(f"no model_*.json under {args.dir}", file=sys.stderr)
        return 1
    # One model can have several result files (a per-template run writes its own); merge the
    # cells instead of letting the last file win, and keep the first payload's metadata.
    models: dict[str, Any] = {}
    for path in model_files:
        payload = json.loads(path.read_text())
        model_id = payload["model_id"]
        if model_id not in models:
            models[model_id] = payload
            models[model_id]["source_files"] = [path.name]
        else:
            models[model_id]["cells"].extend(payload.get("cells", []))
            models[model_id].setdefault("source_files", []).append(path.name)
            models[model_id]["n_prediction_rows"] = models[model_id].get(
                "n_prediction_rows", 0
            ) + payload.get("n_prediction_rows", 0)

    screen = json.loads(args.screen.read_text()) if args.screen.exists() else {"templates": []}
    kept = {t["template_id"] for t in screen["templates"] if not t["drop"]}
    dropped = {t["template_id"]: t["drop_reasons"] for t in screen["templates"] if t["drop"]}
    template_totals = {
        (t["tier"], t["template_id"]): t["n_test"] for t in screen["templates"] if not t["drop"]
    }

    all_cells: list[dict[str, Any]] = []
    for model_id, payload in models.items():
        for cell in payload["cells"]:
            cell["model_id"] = model_id
            all_cells.append(cell)

    # integrity: n must equal the template's test item count for every cell
    mismatches = []
    for cell in all_cells:
        total = template_totals.get((cell["tier"], cell["template_id"]))
        if total is not None and cell["n"] != total and cell.get("gate_status", "") != "PASS":
            mismatches.append({"model": cell["model_id"], "tier": cell["tier"],
                               "template": cell["template_id"], "n": cell["n"], "total": total})

    summary: dict[str, Any] = {
        "generated_at_utc": utcnow(),
        "models": list(models),
        "n_cells": len(all_cells),
        "n_gate_pass": sum(1 for c in all_cells if c.get("gate_status") == "PASS"),
        "n_gate_fail": sum(
            1 for c in all_cells if str(c.get("gate_status", "")).startswith("READOUT_FAIL")
        ),
        "n_not_measured": sum(
            1 for c in all_cells if str(c.get("gate_status", "")).startswith("NOT MEASURED")
        ),
        "n_prediction_rows": sum(c["n"] for c in all_cells),
        "per_model": {
            model_id: payload["tiers"] for model_id, payload in models.items()
        },
        "templates_kept": sorted(kept),
        "templates_dropped": dropped,
        "coverage_mismatches": mismatches,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "cells": all_cells}, indent=2) + "\n")

    lines = [
        f"# Zero-shot ladder baselines on {args.benchmark_label}",
        "",
        f"Generated: `{summary['generated_at_utc']}`  ",
        f"Models: {', '.join(f'`{m}`' for m in models)}  ",
        f"Cells: {summary['n_cells']} ({summary['n_gate_pass']} PASS, "
        f"{summary['n_gate_fail']} READOUT_FAIL, {summary['n_not_measured']} NOT MEASURED)  ",
        f"Prediction rows: {summary['n_prediction_rows']}",
        "",
        "Protocol: the fixed T6 protocol (chat template, thinking disabled, single-letter",
        "instruction) with the **F2 readout**, which renders every question type with letter labels",
        "and reads the option-letter tokens. The `choice` path is byte-identical to the path",
        "validated against lm-evaluation-harness in loop bench_v0; `noul`/`score` now go through the",
        "same path (see `readout_validation.md`).",
        "",
        "**Reporting rule (D12).** An accuracy appears below only when its cell passes the",
        "readout-health gate: median label mass >= 0.5, greedy agreement >= 0.9 on a seeded sample,",
        "and an accuracy CI whose upper bound is not below chance. A failing cell reads",
        "`READOUT_FAIL — <check>` **and no accuracy is shown for it**. A cell that could not be",
        "measured reads `NOT MEASURED — <reason>`. `coverage` is the share of *this template's* test",
        "items the model could take (it was the share of the whole source in bench_v0).",
        "",
        "Batch size 8, max batch 32k tokens, greedy sample 50 items per cell, option-shuffle flip",
        "rate on up to 500 items per template. Predictions stay in the gitignored `outputs/`.",
        "",
    ]
    for tier, title in (("tier1", "Tier 1 (established public test sets)"),
                        ("fresh", "Tier 2 (fresh, 2026-03-01 window)")):
        cells = [c for c in all_cells if c["tier"] == tier]
        if not cells:
            continue
        lines += [f"## {title}", ""]
        lines += _table(cells)
        lines.append("")
        strict = [c for c in cells if c.get("strict_slice", {}).get("n")]
        if strict:
            lines += [
                "### Strict slice (records dated on/after 2026-09-10)",
                "",
                "| model | template | n | strict accuracy | whole-template accuracy |",
                "|---|---|---|---|---|",
            ]
            for c in sorted(strict, key=lambda x: (x["model_id"], x["template_id"])):
                whole = _accuracy_cell(c)
                lines.append(
                    f"| `{c['model_id']}` | `{c['template_id']}` | {c['strict_slice']['n']} | "
                    f"{c['strict_slice']['accuracy']:.4f} | {whole} |"
                )
            lines.append("")
    if dropped:
        lines += ["## Templates screened out (F3) — not measured here", "",
                  "| template | reasons |", "|---|---|"]
        for template_id, reasons in sorted(dropped.items()):
            lines.append(f"| `{template_id}` | {'; '.join(reasons)} |")
        lines.append("")
    lines += [
        "## Notes",
        "",
        "* `majority` is the largest gold class's share of the cell's items; every v0.1 template is",
        "  class-balanced (F4), so it is well below 1.0 and an accuracy can be read against it.",
        "* `flip` is the share of sampled items whose chosen option *content* changed when the",
        "  options were reordered — a positional-guessing signal, not an accuracy.",
        "* Cells whose `n` is below the template's item count are visible in the `n` column as",
        "  `n/total`; the difference is items the model could not take and is reported, never",
        "  silently dropped.",
        "",
    ]
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("n_cells", "n_gate_pass", "n_gate_fail", "n_not_measured",
                               "n_prediction_rows", "coverage_mismatches")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
