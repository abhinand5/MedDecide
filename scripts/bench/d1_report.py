"""d1-2B aggregates and the committed results file (bench/d1-2b), CPU.

Reads outputs/d1_bench/<set>/predictions.jsonl and manifest.json for the five frozen sets, checks each set against its frozen
items (counts, ids, gold labels, option keys, completeness), computes the template macro accuracy, the item micro accuracy and the
D12 readout status of every template cell (meddecide.eval.d1_bench), and writes:

- outputs/d1_bench/aggregates.json (every figure; gitignored);
- loops/d1_bench/results.md (aggregates only, committed): d1-2B's own figures, the readout cells, and the comparison with the
  zero-shot comparators, whose figures are transcribed from the claim rows named next to each value (no value is computed here).

Usage (CPU):
    uv run --frozen --no-sync python scripts/bench/d1_report.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.eval.d1_bench import (
    cell_statuses,
    item_micro_accuracy,
    scored_rows,
    template_macro_accuracy,
)
from meddecide.model.d1 import REPO, REVISION
from meddecide.utils.io import iter_jsonl, read_json, write_json
from meddecide.utils.provenance import utcnow

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/workspace/MedDecide/data/bench/osler_v4/y9")  # the frozen Y9 sets (main checkout; gitignored)
PRED_ROOT = REPO_ROOT / "outputs" / "d1_bench"
AGG_JSON = PRED_ROOT / "aggregates.json"
RESULTS_MD = REPO_ROOT / "loops" / "d1_bench" / "results.md"
NOT = "NOT MEASURED"

SETS = ("knowledge", "mednli_test", "panel", "mr_verdict", "mimic_suite")
TITLES = {
    "knowledge": "Knowledge test (MedQA + MedMCQA)",
    "mednli_test": "MedNLI test",
    "panel": "External clinical panel (7 datasets)",
    "mr_verdict": "MR verdict suite v1",
    "mimic_suite": "MIMIC audit suite v1 (local only)",
}
# Comparators: (template macro, item micro) with the claim row each value comes from. None = NOT MEASURED in the permitted sources.
COMPARATORS: dict[str, dict[str, tuple[float | None, float | None, str]]] = {
    "knowledge": {
        "Qwen3.5-4B zero-shot": (0.6450, 0.6162, "C095"),
        "Clef-Flash": (0.7105, 0.6864, "C095"),
        "pplx-decider-flash-v1.2 zero-shot": (None, 0.6835, "C096 (micro on the 5,435 items it scored; macro not stated)"),
    },
    "mednli_test": {
        "Qwen3.5-4B zero-shot": (0.7862, 0.7862, "C092 (one template; macro = micro)"),
        "Clef-Flash": (0.8636, 0.8636, "C092 (one template; macro = micro)"),
        "pplx-decider-flash-v1.2 zero-shot": (None, None, "NOT MEASURED in the permitted sources"),
    },
    "panel": {
        "Qwen3.5-4B zero-shot": (0.5483, 0.4831, "C093"),
        "Clef-Flash": (0.6502, 0.5711, "C093"),
        "pplx-decider-flash-v1.2 zero-shot": (None, None, "NOT MEASURED in the permitted sources"),
    },
    "mr_verdict": {
        "Qwen3.5-4B zero-shot": (0.5574, 0.6176, "C090"),
        "Clef-Flash": (0.6830, 0.6830, "C090"),
        "pplx-decider-flash-v1.2 zero-shot": (0.7044, None, "macro: EVIDENCE.md 5.2 (osler_v4 scoreboard, C228); micro NOT MEASURED"),
    },
    "mimic_suite": {
        "Qwen3.5-4B zero-shot": (0.5697, 0.5603, "C091"),
        "Clef-Flash": (0.6391, 0.6306, "C091"),
        "pplx-decider-flash-v1.2 zero-shot": (0.6341, None, "macro: EVIDENCE.md 5.2 (osler_v4 scoreboard, C228); micro NOT MEASURED"),
    },
}


def load_set(name: str) -> dict[str, Any]:
    """The set's d1 predictions, checked against the frozen items; any mismatch stops the report."""
    items = {str(r["item_id"]): r for r in iter_jsonl(DATA / f"{name}.jsonl")}
    pred_dir = PRED_ROOT / name
    manifest = read_json(pred_dir / "manifest.json")
    preds = list(iter_jsonl(pred_dir / "predictions.jsonl"))
    if not manifest.get("complete"):
        raise SystemExit(f"{name}: the run is not complete ({manifest.get('scored_total')} of {len(items)})")
    if len(preds) != len(items) or {str(p["item_id"]) for p in preds} != set(items):
        raise SystemExit(f"{name}: predictions do not cover the frozen items one to one")
    for p in preds:
        item = items[str(p["item_id"])]
        keys = [str(o["key"]) for o in item["options"]]
        if p["option_keys"] != keys or p["gold"] != str(item["gold"]) or p["template_id"] != str(item["template_id"]):
            raise SystemExit(f"{name}: prediction {p['item_id']} does not match its item")
    rows = scored_rows(preds, items)
    cells = cell_statuses(rows)
    failing = sorted(t for t, c in cells.items() if c["status"] != "PASS")
    extra: dict[str, Any] = {"answer_types": dict(Counter(str(p["answer_type"]) for p in preds))}
    scores = [p for p in preds if p["qtype"] == "score"]
    if scores:  # additional diagnostic: the card's median level against the argmax of the distribution
        disagree = 0
        for p in scores:
            probs = p["probs"]
            argmax = max(range(len(probs)), key=lambda i: probs[i])
            disagree += int(argmax != int(p["score_median"]))
        extra["score_items"] = len(scores)
        extra["score_argmax_vs_median_disagree"] = disagree
    return {
        "items": len(items), "scored": len(rows), "templates": len(cells),
        "qtypes": dict(Counter(str(p["qtype"]) for p in preds)),
        "template_macro_accuracy": round(template_macro_accuracy(rows), 4),
        "item_micro_accuracy": round(item_micro_accuracy(rows), 4),
        "readout": "PASS" if not failing else f"READOUT_FAIL on {len(failing)} template(s): {', '.join(failing)}",
        "cells": cells, "extra": extra,
        "manifest": {k: manifest[k] for k in ("items_sha256", "transformers", "torch", "gpu", "git_commit", "wall_s_this_run")},
    }


def fmt(x: float | None) -> str:
    return NOT if x is None else f"{x:.4f}"


def render(agg: dict[str, Any]) -> str:
    any_manifest = next(iter(agg["sets"].values()))["manifest"]
    lines = [
        "# d1-2B local benchmark (bench/d1-2b): aggregates only",
        "",
        f"Generated {utcnow()} by `scripts/bench/d1_report.py` from `outputs/d1_bench/<set>/predictions.jsonl` (gitignored; "
        "private). Aggregates only: no item, note, prediction or label text is in this file.",
        "",
        f"- Model: `{REPO}` at revision `{REVISION}` (LFM Open License v1.0; the license terms are in `loops/d1_bench/STATE.md`).",
        f"- Software: transformers {any_manifest['transformers']} (the card asks for >=5.19; see STATE), torch {any_manifest['torch']}, "
        f"GPU {any_manifest['gpu']}. Calibration: the checkpoint's `calibration.json` is applied by the card's engine.",
        "- Definitions (as `loops/osler_v4/results.md`): template macro accuracy = mean over templates of each template's mean "
        "correctness; item micro accuracy = mean correctness over all scored items. The readout status is the D12 gate "
        "(`meddecide.eval.readout_gate`, D21 variant for a trained readout that normalises over the offered options): a cell "
        "fails on a constant answer or when its 95 % upper bound is below chance.",
        "- Prompt protocol: one state and one question per call, via the card's `system_one` (noul: P(yes); choice: a "
        "distribution over the item's option keys; score: a distribution over its levels, lowest first). No text is generated.",
        "",
        "## 1. d1-2B on the frozen sets",
        "",
        "| set | items | scored | templates | template macro accuracy | item micro accuracy | readout gate (D12/D21) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in SETS:
        s = agg["sets"][name]
        lines.append(f"| {TITLES[name]} | {s['items']} | {s['scored']} | {s['templates']} | {s['template_macro_accuracy']:.4f} | "
                     f"{s['item_micro_accuracy']:.4f} | {s['readout']} |")
    lines += ["", "## 2. Readout-gate cells (every template of every set)", "",
              "| set | template | items | accuracy | chance (mean 1/n) | modal answer share | 95 % upper bound | status |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for name in SETS:
        for template, c in agg["sets"][name]["cells"].items():
            lines.append(f"| {name} | {template} | {c['items']} | {c['accuracy']:.4f} | {c['chance']:.4f} | {c['modal_share']:.4f} | "
                         f"{c['ci_upper']:.4f} | {c['status']} |")
    lines += ["", "## 3. Comparison with zero-shot comparators (additional; no verdict)", "",
              "Comparator values are transcribed from the claim rows named in the last column (`loops/osler_v5/CLAIMS.md` on "
              "`loop/osler_v5`, and `docs/paper/EVIDENCE.md` §5.2 on the same branch). The readout status of Qwen3.5-4B and Clef-Flash "
              "is PASS for every cell (C089). `NOT MEASURED` means the permitted sources give no such figure.", "",
              "| set | model | template macro accuracy | item micro accuracy | source |", "|---|---|---:|---:|---|"]
    for name in SETS:
        d1 = agg["sets"][name]
        lines.append(f"| {TITLES[name]} | d1-2B (this run) | {d1['template_macro_accuracy']:.4f} | {d1['item_micro_accuracy']:.4f} | "
                     "this run (section 1) |")
        for model, (macro, micro, source) in COMPARATORS[name].items():
            lines.append(f"| {TITLES[name]} | {model} | {fmt(macro)} | {fmt(micro)} | {source} |")
    lines += ["", "## 4. Additional diagnostics (labelled; no verdict)", ""]
    for name in SETS:
        extra = agg["sets"][name]["extra"]
        parts = [f"answer types {extra['answer_types']}"]
        if "score_items" in extra:
            parts.append(f"score items {extra['score_items']}; argmax of the distribution differs from the card's median level on "
                         f"{extra['score_argmax_vs_median_disagree']}")
        lines.append(f"- {TITLES[name]}: " + "; ".join(parts) + ".")
    lines += ["", "## 5. Claims (R1)", "", "| id | claim | value | artifact | recompute |", "|---|---|---|---|---|"]
    for i, name in enumerate(SETS, start=1):
        s = agg["sets"][name]
        lines.append(f"| D1-{i} | {TITLES[name]}: template macro, item micro, items | macro {s['template_macro_accuracy']:.4f}; micro "
                     f"{s['item_micro_accuracy']:.4f}; {s['scored']} items | `outputs/d1_bench/{name}/predictions.jsonl` | "
                     "`uv run --frozen --no-sync python scripts/bench/d1_report.py` |")
    lines += ["", "Re-derivation and self-audit: see `loops/d1_bench/STATE.md`.", ""]
    return "\n".join(lines)


def main() -> None:
    sets = {name: load_set(name) for name in SETS}
    agg = {"generated_utc": utcnow(), "model": REPO, "revision": REVISION, "sets": sets}
    write_json(AGG_JSON, agg)
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text(render(agg), encoding="utf-8")
    print(json.dumps({k: {"scored": v["scored"], "macro": v["template_macro_accuracy"], "micro": v["item_micro_accuracy"],
                          "readout": v["readout"]} for k, v in sets.items()}, indent=1))


if __name__ == "__main__":
    main()
