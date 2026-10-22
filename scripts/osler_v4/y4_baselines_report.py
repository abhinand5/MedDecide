"""Y4 baselines report (osler_v4 ADVISORY Y4 §4): the accuracy of each baseline on the frozen MIMIC audit suite, the held-out items
(additional rows), the MIMIC partition of fast dev v4 and CG v2, with the readout checks of meddecide.eval.health.

Readout checks, per (set, model, template): the constant-answer check (the modal answer on at least DEFAULT_MAX_MODAL_SHARE of the
items, on a template whose majority share is at most DEFAULT_DEGENERATE_MAJORITY_CEILING) and the below-chance check (the 95 % CI
upper bound of the accuracy under the mean chance 1/n). The label-mass check is not applied to the authors' rows: their decision
models are trained readouts that normalise over the offered options, so label mass is a diagnostic there (D21), and the label mass
is not read here. Cells that fail a check read READOUT_FAIL and their accuracy is not reported as a result (D12).

Writes outputs/osler_v4/Y4/baselines_report.json (aggregates) and loops/osler_v4/baselines.md (tables; no item text).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.eval.health import DEFAULT_DEGENERATE_MAJORITY_CEILING, DEFAULT_MAX_MODAL_SHARE
from meddecide.eval.metrics import bootstrap_ci, macro_accuracy
from meddecide.utils.io import iter_jsonl, write_json

BASELINES = Path("outputs/osler_v4/Y4/baselines")
OUT = Path("outputs/osler_v4/Y4/baselines_report.json")
MD = Path("loops/osler_v4/baselines.md")
SETS = ("suite", "heldout", "fastdev_mimic", "cg_v2")
MODELS = {"flash": "pplx-decider-flash-v1.2 (zero-shot, authors' code)",
          "meddecider4b": "MedDecider-4B (authors' code)",
          "meddecider9b": "MedDecider-9B (authors' code; extra row)",
          "osler_v2": "osler_v2 Osler-4B (trainer-based)"}


def cell_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [str(r["gold_key"]) for r in rows]
    predicted = [str(r["predicted_key"]) for r in rows]
    correct = [bool(r["correct"]) for r in rows]
    n = len(rows)
    chance = sum(1.0 / len(r["probs"]) for r in rows) / n
    ci = bootstrap_ci([1.0 if c else 0.0 for c in correct])
    counts = Counter(gold)
    majority_share = counts.most_common(1)[0][1] / n
    modal_key, modal_count = Counter(predicted).most_common(1)[0]
    modal_share = modal_count / n
    failures = []
    if modal_share >= DEFAULT_MAX_MODAL_SHARE and majority_share <= DEFAULT_DEGENERATE_MAJORITY_CEILING:
        failures.append(f"constant answer ({modal_key}) on {modal_share:.3f} of items")
    if ci[2] < chance:
        failures.append(f"below chance (CI upper {ci[2]:.4f} < chance {chance:.4f})")
    return {"items": n, "micro_accuracy": round(sum(correct) / n, 4), "macro_accuracy": round(macro_accuracy(predicted, gold), 4),
            "chance": round(chance, 4), "majority_share": round(majority_share, 4), "modal_share": round(modal_share, 4),
            "modal_key": modal_key, "status": "PASS" if not failures else "READOUT_FAIL — " + "; ".join(failures)}


def main() -> None:
    report: dict[str, Any] = {"checks": {"max_modal_share": DEFAULT_MAX_MODAL_SHARE,
                                         "degenerate_majority_ceiling": DEFAULT_DEGENERATE_MAJORITY_CEILING,
                                         "label_mass": "not applied (D21: trained readouts normalising over offered options)"},
                              "files": {}, "sets": {}}
    for set_name in SETS:
        for model in MODELS:
            path = BASELINES / f"{set_name}_{model}" / "predictions.jsonl"
            if not path.exists():
                report["files"][f"{set_name}_{model}"] = "not run"
                continue
            rows = [r for r in iter_jsonl(path) if r.get("status", "scored") == "scored"]
            report["files"][f"{set_name}_{model}"] = {"scored": len(rows)}
            by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in rows:
                by_template[str(r["template_id"])].append(r)
            cells = {t: cell_report(v) for t, v in sorted(by_template.items())}
            overall = cell_report(rows) if rows else None
            report["sets"][f"{set_name}_{model}"] = {"overall": overall, "templates": cells}
    write_json(OUT, report)
    lines = ["# osler_v4 Y4 — baselines (aggregates only)", "",
             "Readout checks per (set, model, template) follow D12 and D21 as written in the report script; label mass is not applied",
             "to the authors' rows. Cells that fail a check read READOUT_FAIL and are not reported as results.", ""]
    for key, value in report["sets"].items():
        if value["overall"] is None:
            continue
        o = value["overall"]
        lines += [f"## {key}", "", f"overall: {o['items']} items; micro {o['micro_accuracy']}; macro {o['macro_accuracy']}; "
                  f"chance {o['chance']}; status {o['status']}", "",
                  "| template | items | micro | macro | majority | modal share | status |", "|---|---|---|---|---|---|---|"]
        for t, c in value["templates"].items():
            lines.append(f"| {t} | {c['items']} | {c['micro_accuracy']} | {c['macro_accuracy']} | {c['majority_share']} | "
                         f"{c['modal_share']} | {c['status']} |")
        lines.append("")
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({k: v["overall"]["micro_accuracy"] for k, v in report["sets"].items() if v["overall"]})


if __name__ == "__main__":
    main()
