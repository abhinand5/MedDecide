"""D6 self-audit (R6): re-derive the Gate O7 numbers from the stored predictions.

Reads ``gate_o7.json`` and the per-model prediction files under the D6 output directory (and the Y9/Z10
baseline directories), recomputes from scratch with its own arithmetic: every set's micro and template
macro accuracy for both students and every frontier model, the paired accuracy differences of the four
checks (the same paired-bootstrap estimator from ``meddecide.eval.gate_o4``, but driven here over the
stored rows), the Gate O7 verdict, and the frontier relation. It then compares each number with what
``gate_o7.json`` claims and writes ``audit.json``; a mismatch fails the audit.

Usage (CPU):
    uv run --frozen python scripts/distill_v0/d6_rederive.py --out outputs/distill_v0/D6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from meddecide.eval.gate_o4 import comparison, pair_rows
from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.io import iter_jsonl
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
Y9 = REPO / "data" / "bench" / "osler_v4" / "y9"
BASELINES = REPO / "outputs" / "osler_v4" / "Y9" / "baselines"
Z10 = REPO / "outputs" / "osler_v5" / "Z10"
SETS = ("knowledge", "panel", "mr_verdict", "mednli_test", "mr_reported", "mr_heldout_test",
        "mimic_suite", "pairs_test", "mimic_heldout")
CHECKS = (("i", "panel", "pplx_flash", 0.0), ("ii", "knowledge", "pplx_flash", -0.02),
          ("iii", "mr_verdict", "final", -0.02), ("iv", "mimic_suite", "final", -0.02))
Y4 = REPO / "outputs" / "osler_v4" / "Y4" / "baselines"


def items_for(set_name: str) -> list[dict[str, Any]]:
    items = []
    for row in iter_jsonl(Y9 / f"{set_name}.jsonl"):
        items.append({**row, "item_id": mix_row_to_item(row).item_id})
    return items


def rows_from(path: Path, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The stored rows normalised from the item's gold, keys and the stored probabilities (computed-id keyed)."""
    by_computed = {str(i["item_id"]): i for i in items}
    rows = []
    for raw in iter_jsonl(path):
        item = by_computed.get(str(raw["item_id"]))
        if item is None:
            continue
        probs = [float(p) for p in raw["probs"]]
        gold = str(item["gold"])
        option_keys = [str(o["key"]) for o in item["options"]]
        predicted = option_keys[max(range(len(probs)), key=probs.__getitem__)]
        rows.append({"item_id": str(item["item_id"]), "template_id": str(item["template_id"]),
                     "gold_key": gold, "predicted_key": predicted, "correct": predicted == gold, "probs": probs})
    return rows


def baseline_path(model: str, set_name: str) -> Path | None:
    """The baseline rows for a check: zero-shot pplx (Y4 runs for the MIMIC sets) and the osler_v4 final."""
    if model == "pplx_flash":
        path = BASELINES / f"pplx_{set_name}" / "predictions.jsonl"
        if path.exists():
            return path
        if set_name in {"mimic_suite", "mimic_heldout"}:
            run = "suite_flash" if set_name == "mimic_suite" else "heldout_flash"
            path = Y4 / run / "predictions.jsonl"
            return path if path.exists() else None
        return None
    if model == "final":
        path = BASELINES / f"final_{set_name}" / "predictions.jsonl"
        return path if path.exists() else None
    return None


def scoreboard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"items": 0, "accuracy_micro": None, "accuracy_template_macro": None}
    correct = sum(int(r["correct"]) for r in rows)
    templates: dict[str, list[float]] = {}
    for r in rows:
        templates.setdefault(str(r["template_id"]), []).append(1.0 if r["correct"] else 0.0)
    macro = sum(sum(v) / len(v) for v in templates.values()) / len(templates)
    return {"items": len(rows), "correct": correct, "accuracy_micro": correct / len(rows),
            "accuracy_template_macro": macro}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="outputs/distill_v0/D6")
    args = parser.parse_args()
    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    gate = json.loads((out / "gate_o7.json").read_text(encoding="utf-8"))
    items_by_set = {name: items_for(name) for name in SETS}
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    students = gate["students"]
    student_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for student in students:
        student_rows[student] = {}
        for name in SETS:
            path = out / f"{student}_{name}" / "predictions.jsonl"
            student_rows[student][name] = rows_from(path, items_by_set[name])
        # scoreboard recomputation
        mismatched = []
        for name in SETS:
            recomputed = scoreboard(student_rows[student][name])
            recorded = gate["scoreboard"][student][name]
            if recomputed["items"] != recorded["items"] or (
                    recomputed["accuracy_micro"] is not None
                    and abs(recomputed["accuracy_micro"] - recorded["accuracy_micro"]) > 1e-12):
                mismatched.append(name)
        check(f"{student}_scoreboard", not mismatched, {"mismatched_sets": mismatched})

        # the four checks, recomputed
        for check_id, set_name, baseline, limit in CHECKS:
            base_rows = rows_from(baseline_path(baseline, set_name), items_by_set[set_name]) if baseline_path(baseline, set_name) else []
            if not base_rows or not student_rows[student][set_name]:
                check(f"{student}_{check_id}", False, "missing rows")
                continue
            pairing = pair_rows(student_rows[student][set_name], base_rows, items_by_set[set_name])
            paired = comparison(pairing)
            verdict = "PASS" if paired.accuracy.lo > limit else "FAIL"
            recorded = gate["gate_o7"][student]["checks"][check_id]
            same = (recorded["verdict"] == verdict
                    and recorded["accuracy"] is not None
                    and abs(recorded["accuracy"]["lower"] - paired.accuracy.lo) <= 1e-9
                    and abs(recorded["accuracy"]["point"] - paired.accuracy.point) <= 1e-9)
            check(f"{student}_{check_id}", same,
                  {"recomputed": {"point": paired.accuracy.point, "lower": paired.accuracy.lo, "verdict": verdict,
                                  "items": paired.items},
                   "recorded": recorded.get("accuracy"), "recorded_verdict": recorded["verdict"]})
        expected_gate = "PASS" if all(gate["gate_o7"][student]["checks"][c[0]]["verdict"] == "PASS" for c in CHECKS) else (
            "FAIL" if any(gate["gate_o7"][student]["checks"][c[0]]["verdict"] == "FAIL" for c in CHECKS) else "NOT MEASURED")
        check(f"{student}_gate_verdict", gate["gate_o7"][student]["verdict"] == expected_gate,
              {"recomputed": expected_gate, "recorded": gate["gate_o7"][student]["verdict"]})

    audit = {"task": "D6 self-audit", "audited_utc": utcnow(), "out": str(out), "checks": checks,
             "all_passed": all(entry["ok"] for entry in checks)}
    (out / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for entry in checks:
        print(f"[{'PASS' if entry['ok'] else 'FAIL'}] {entry['check']}")
    if not audit["all_passed"]:
        raise SystemExit("D6 self-audit failed")


if __name__ == "__main__":
    main()
