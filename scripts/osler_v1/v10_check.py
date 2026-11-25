"""V10 check (osler_v1): completeness of the Osler-4B scoring, the baselines' skipped items, and an independent re-derivation of
the Gate O2 point estimates (R1, R6). Reads outputs/osler_v1/V10 and the baseline prediction files; prints aggregates only and
writes nothing. The gate's own code (v10_gate.py) decides the verdict; this script only checks it.

Usage (CPU; no GPU):
    source scripts/pod_env.sh && uv run --frozen python -I scripts/osler_v1/v10_check.py
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data/bench/osler_v1/v10_items.jsonl"
V10 = REPO / "outputs/osler_v1/V10"
V6 = REPO / "outputs/osler_v1/V6/baselines"
O2 = REPO / "outputs/osler_v0/O2"
TOLERANCE = 5e-4


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_by_id(path: Path) -> dict[str, dict]:
    return {str(row["item_id"]): row for row in load_jsonl(path)}


def correct_and_brier(row: dict) -> tuple[float, float]:
    """Correct (1.0 or 0.0) and the multi-class Brier score of one prediction row (sum over options of (p - y)^2)."""
    probs = [float(p) for p in row["probs"]]
    gold = list(row["option_keys"]).index(row["gold"])
    predicted = max(range(len(probs)), key=probs.__getitem__)
    brier = sum((p - (1.0 if k == gold else 0.0)) ** 2 for k, p in enumerate(probs))
    return (1.0 if predicted == gold else 0.0), brier


def macro_and_brier(items: list[dict], osler: dict, base: dict) -> tuple[float, float, int]:
    """Template-macro accuracy difference (Osler minus baseline) and Brier difference, on items both files score."""
    per_template: dict[str, list[list[float]]] = defaultdict(lambda: [[], []])
    brier_osler: list[float] = []
    brier_base: list[float] = []
    for item in items:
        o, b = osler.get(item["item_id"]), base.get(item["item_id"])
        if o is None or b is None or b.get("status") != "scored":
            continue
        co, bo = correct_and_brier(o)
        cb, bb = correct_and_brier(b)
        per_template[item["template_id"]][0].append(co)
        per_template[item["template_id"]][1].append(cb)
        brier_osler.append(bo)
        brier_base.append(bb)
    diffs = [sum(c_o) / len(c_o) - sum(c_b) / len(c_b) for c_o, c_b in per_template.values()]
    n = len(brier_osler)
    return sum(diffs) / len(diffs), sum(brier_osler) / n - sum(brier_base) / n, n


def completeness() -> None:
    run = json.loads((V10 / "osler_4b" / "run.json").read_text(encoding="utf-8"))
    manifest = json.loads((V10 / "items_manifest.json").read_text(encoding="utf-8"))
    print(f"run: rows in items file {run['rows_in_items_file']}; scored now {run['rows_scored_now']}; "
          f"written before {run['rows_written_before']}; calibration equals dev fits: "
          f"{run['checkpoint_calibration_matches_dev_fits']}; device {run['device']}; dtype {run['dtype']}; "
          f"wall clock {run['wall_clock_s']} s; git commit {run['git_commit'][:7]}")
    print(f"manifest: selected rows {manifest['selected_rows']}")
    for name in ("both_orders.jsonl", "single_order.jsonl"):
        rows = load_jsonl(V10 / "osler_4b" / name)
        ids = [str(r["item_id"]) for r in rows]
        bad = sum(1 for r in rows if not all(math.isfinite(float(p)) for p in r["probs"])
                  or abs(sum(float(p) for p in r["probs"]) - 1.0) > 1e-3)
        print(f"{name}: rows {len(rows)}; distinct ids {len(set(ids))}; status {dict(Counter(r['status'] for r in rows))}; "
              f"rows with non-finite or unnormalised probabilities {bad}")
    items = load_jsonl(ITEMS)
    print("V10 items by scope:", dict(Counter(i["v10_scope"] for i in items)),
          "| MR held-out by split:", dict(Counter(i["split"] for i in items if i["v10_scope"] == "mr_heldout")))


def skips() -> None:
    for label, path in (("MedDecider-4B on the MR suite (V6)", V6 / "meddecider-4b" / "predictions.jsonl"),
                        ("MedDecider-4B on the O2 common set (osler_v0 O2)", O2 / "meddecider-4b" / "predictions.jsonl")):
        rows = load_jsonl(path)
        skipped = [r for r in rows if r.get("status") != "scored"]
        reasons = Counter(str(r.get("reason")) for r in skipped)
        print(f"{label}: rows {len(rows)}; skipped {len(skipped)}; reasons {dict(reasons)}")
        print(f"  skipped by template: {dict(Counter(str(r.get('template_id')) for r in skipped).most_common(6))}")


def rederive() -> None:
    items = load_jsonl(ITEMS)
    osler = load_by_id(V10 / "osler_4b" / "both_orders.jsonl")
    mr = [i for i in items if i["v10_scope"] == "mr_test"]
    panel = [i for i in items if i["v10_scope"] == "external_panel"]
    knowledge = [i for i in items if i["v10_scope"] == "knowledge"]
    v6_med = load_by_id(V6 / "meddecider-4b" / "predictions.jsonl")
    v6_zero = load_by_id(V6 / "qwen35-4b" / "predictions.jsonl")
    o2_med = load_by_id(O2 / "meddecider-4b" / "predictions.jsonl")
    o2_zero = load_by_id(O2 / "qwen35-4b" / "predictions.jsonl")
    values = {
        "mr_zero": macro_and_brier(mr, osler, v6_zero),
        "mr_med": macro_and_brier(mr, osler, v6_med),
        "panel_zero": macro_and_brier(panel, osler, o2_zero),
        "panel_med": macro_and_brier(panel, osler, o2_med),
    }
    guard = [correct_and_brier(osler[i["item_id"]])[0] - correct_and_brier(o2_zero[i["item_id"]])[0] for i in knowledge]
    for key, (acc, brier, n) in values.items():
        print(f"re-derived {key}: accuracy diff {acc:+.4f}; Brier diff {brier:+.4f}; paired {n}")
    print(f"re-derived knowledge guard (micro): accuracy diff {sum(guard) / len(guard):+.4f} over {len(guard)} items")
    gate = json.loads((V10 / "gate_o2.json").read_text(encoding="utf-8"))["models"]["osler_4b"]
    checks = [
        (values["mr_zero"][0], gate["mr_suite"]["zero-shot Qwen3.5-4B"]["accuracy"]["point"]),
        (values["mr_zero"][1], gate["mr_suite"]["zero-shot Qwen3.5-4B"]["brier"]["point"]),
        (values["mr_med"][0], gate["mr_suite"]["MedDecider-4B"]["accuracy"]["point"]),
        (values["mr_med"][1], gate["mr_suite"]["MedDecider-4B"]["brier"]["point"]),
        (values["panel_zero"][0], gate["external_panel"]["zero-shot Qwen3.5-4B"]["accuracy"]["point"]),
        (values["panel_med"][0], gate["external_panel"]["MedDecider-4B"]["accuracy"]["point"]),
        (sum(guard) / len(guard), gate["knowledge_guard"]["accuracy"]["point"]),
    ]
    all_match = all(math.isclose(a, b, abs_tol=TOLERANCE) for a, b in checks)
    print(f"re-derivation matches the gate's point estimates (abs tol {TOLERANCE}): {all_match}")


def composition() -> None:
    """The MR test suite by part: generated (V1 code-generated records, source None; V5a Synthea) and real text (V5b)."""
    items = [i for i in load_jsonl(ITEMS) if i["v10_scope"] == "mr_test"]
    generated = [i for i in items if i.get("source") in (None, "synthea")]
    real_text = [i for i in items if i.get("source") in ("pmc_patients", "codiesp", "mkurman", "mtsamples")]
    print(f"MR test suite {len(items)}: generated {len(generated)} (source None {sum(1 for i in items if i.get('source') is None)}, "
          f"synthea {sum(1 for i in items if i.get('source') == 'synthea')}); real text {len(real_text)} "
          f"(pmc_patients {sum(1 for i in items if i.get('source') == 'pmc_patients')}, codiesp "
          f"{sum(1 for i in items if i.get('source') == 'codiesp')}, mkurman {sum(1 for i in items if i.get('source') == 'mkurman')}, "
          f"mtsamples {sum(1 for i in items if i.get('source') == 'mtsamples')}); MIMIC 0 (absent)")
    print("MR test suite by template:", dict(Counter(i["template_id"] for i in items).most_common()))


def main() -> None:
    completeness()
    skips()
    composition()
    rederive()


if __name__ == "__main__":
    main()
