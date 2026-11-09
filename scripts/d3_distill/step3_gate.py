"""d3_distill step 3 (CPU): Gate O7 from the scored predictions on the frozen Y9 sets.

Gate O7 (PLAN.md, "unchanged from distill_v0"):
  (iii) external panel: lower bound of (student - zero-shot pplx-decider-flash-v1.2) > 0;
  (iv)  knowledge:      lower bound > -0.02 against the same baseline;
  (i,ii) MR verdict and MIMIC: within -0.02 of the osler_v4 final.

What is measured here: the panel and knowledge checks against pplx-flash, whose predictions are produced on this machine
with the authors' code (scripts/osler_v4/y4_pplx_flash.py, snapshot 6d771ad7); the mr_verdict comparison against
pplx-flash is reported as an **additional** row (no PLAN limit), and the PLAN's MR check against the osler_v4 final is
NOT MEASURED (its predictions and weights are not on this machine, operator instruction 2026-11-09). MIMIC is NOT
MEASURED: no MIMIC data is allowed on this machine.

Pairing, unpaired rule (D44(a), > 1 % ⇒ NOT MEASURED), readout gate (D12, a failing cell for either model ⇒ NOT
MEASURED) and the paired percentile bootstrap are the Gate O4 machinery (meddecide.eval.gate_o4).

Usage:
    python scripts/d3_distill/step3_gate.py --out loops/d3_distill/GATE_O7.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
Y9 = REPO / "data" / "bench" / "osler_v4" / "y9"
STEP2 = REPO / "outputs" / "d3_distill" / "step2"
BASELINES = REPO / "outputs" / "d3_distill" / "step3" / "baselines"
STEP3 = REPO / "outputs" / "d3_distill" / "step3"
STUDENTS = ("mini", "flash")
SETS = ("panel", "knowledge", "mr_verdict")
PANEL_LIMIT = 0.0  # Gate O7 (iii): the lower bound must be above 0
KNOWLEDGE_LIMIT = -0.02  # Gate O7 (iv)
UNPAIRED_LIMIT = 0.01
# The osler_v4 Y9 scoreboard's zero-shot pplx-decider-flash-v1.2 row (loops/osler_v4/FINDINGS.md section 6), template macro
# accuracy on the same frozen sets. Our freshly scored baseline should reproduce it; a mismatch means the pipeline is off.
PPPLX_PUBLISHED = {"knowledge": 0.7097, "panel": 0.6390, "mr_verdict": 0.7044}
PPPLX_TOLERANCE = 5e-3  # cross-check only: our environment's bf16 numerics move a few items


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def y9_items(set_name: str) -> list[dict]:
    from meddecide.train.mix_items import mix_row_to_item

    items = []
    for row in read_jsonl(Y9 / f"{set_name}.jsonl"):
        items.append({**row, "raw_id": str(row["item_id"]), "item_id": mix_row_to_item(row).item_id})
    if len({i["item_id"] for i in items}) != len(items):
        raise SystemExit(f"{set_name}: the recomputed item ids are not unique")
    return items


def normalised(path: Path, items: list[dict]) -> list[dict] | None:
    from meddecide.eval.gate_o4 import normalise_row

    if not path.exists():
        return None
    by_id = {str(i["item_id"]): i for i in items}
    rows = []
    for raw in read_jsonl(path):
        if raw.get("status", "scored") != "scored":
            continue
        item = by_id.get(str(raw["item_id"]))
        if item is None:
            raise SystemExit(f"{path}: a prediction for an item outside the set")
        rows.append(normalise_row(raw, item))
    return rows


def compare(student_rows: list[dict], base_rows: list[dict], items: list[dict], *, limit: float, name: str,
            verdict: bool = True) -> dict:
    from meddecide.eval.gate_o4 import comparison, pair_rows, readout_failures

    pairing = pair_rows(student_rows, base_rows, items)
    student_fail = readout_failures(student_rows)
    base_fail = readout_failures(base_rows)
    result = {
        "items_in_set": len(items),
        "paired": len(pairing.pairs),
        "unpaired": pairing.unpaired,
        "unpaired_share": round(pairing.unpaired_share, 6),
        "student_readout_failures": student_fail,
        "baseline_readout_failures": base_fail,
    }
    if pairing.unpaired_share > UNPAIRED_LIMIT:
        result.update(status="NOT MEASURED", reason=f"{pairing.unpaired_total} of {pairing.items} items unpaired "
                      f"({pairing.unpaired_share:.2%}, above 1 %; D44(a))")
        return result
    if student_fail or base_fail:
        result.update(status="NOT MEASURED", reason=f"readout cells fail (D12): student {student_fail}, baseline {base_fail}")
        return result
    cmp = comparison(pairing)
    result.update(
        status=("PASS" if cmp.accuracy.lo > limit else "FAIL") if verdict else "ADDITIONAL (no verdict)",
        rule=f"accuracy lower bound > {limit:+.2f}" if verdict else "additional comparison against pplx-flash (no PLAN limit)",
        accuracy={"point": cmp.accuracy.point, "lower": cmp.accuracy.lo, "upper": cmp.accuracy.hi},
        brier={"point": cmp.brier.point, "lower": cmp.brier.lo, "upper": cmp.brier.hi},
        templates=list(cmp.templates),
        paired_items=cmp.items,
    )
    return result


def own_figures(rows: list[dict], items: list[dict]) -> dict:
    from meddecide.eval.paired import item_brier

    if not rows:
        return {"status": "NOT RUN"}
    by_id = {str(i["item_id"]): i for i in items}
    by_template: dict[str, list[float]] = {}
    briers = []
    for row in rows:
        by_template.setdefault(row["template_id"], []).append(1.0 if row["correct"] else 0.0)
        item = by_id[row["item_id"]]
        gold = [str(o["key"]) for o in item["options"]].index(str(item["gold"]))
        briers.append(item_brier(row["probs"], gold))
    return {
        "scored_items": len(rows),
        "template_macro_accuracy": sum(sum(v) / len(v) for v in by_template.values()) / len(by_template),
        "brier_item_mean": sum(briers) / len(briers),
        "templates": len(by_template),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "loops" / "d3_distill" / "GATE_O7.md")
    ap.add_argument("--step2", type=Path, default=STEP2, help="root of the student score directories")
    ap.add_argument("--baselines", type=Path, default=BASELINES, help="root of the pplx baseline predictions")
    args = ap.parse_args()
    STEP3.mkdir(parents=True, exist_ok=True)

    items_by_set = {name: y9_items(name) for name in SETS}
    report: dict = {
        "gate": "O7",
        "finished_utc": utcnow(),
        "sets": {name: str(Y9 / f"{name}.jsonl") for name in SETS},
        "checks": {},
        "students": {},
    }
    for student in STUDENTS:
        base_rows = {
            name: normalised(args.baselines / f"pplx_{name}" / "predictions.jsonl", items_by_set[name]) for name in SETS
        }
        final = {
            name: normalised(args.step2 / student / "final_score" / f"{name}.jsonl", items_by_set[name]) for name in SETS
        }
        zero = {
            name: normalised(args.step2 / student / "zero_shot_score" / f"{name}.jsonl", items_by_set[name]) for name in SETS
        }
        per_student: dict = {"sets": {}, "checks": {}, "baseline_reproduces_published": {}}
        for name in SETS:
            if base_rows[name]:
                figures = own_figures(base_rows[name], items_by_set[name])
                if figures.get("template_macro_accuracy") is not None:
                    published = PPPLX_PUBLISHED[name]
                    per_student["baseline_reproduces_published"][name] = {
                        "ours": figures["template_macro_accuracy"], "published": published,
                        "delta": figures["template_macro_accuracy"] - published,
                        "matches": abs(figures["template_macro_accuracy"] - published) <= PPPLX_TOLERANCE,
                    }
        for name in SETS:
            per_student["sets"][name] = {
                "student": own_figures(final[name], items_by_set[name]) if final[name] else {"status": "NOT RUN"},
                "zero_shot": own_figures(zero[name], items_by_set[name]) if zero[name] else {"status": "NOT RUN"},
                "pplx_flash": own_figures(base_rows[name], items_by_set[name]) if base_rows[name] else {"status": "NOT RUN"},
            }
        if final["panel"] and base_rows["panel"]:
            per_student["checks"]["panel"] = compare(final["panel"], base_rows["panel"], items_by_set["panel"],
                                                     limit=PANEL_LIMIT, name="panel")
            per_student["checks"]["knowledge"] = compare(final["knowledge"], base_rows["knowledge"],
                                                         items_by_set["knowledge"], limit=KNOWLEDGE_LIMIT, name="knowledge")
            per_student["checks"]["mr_verdict_additional"] = compare(
                final["mr_verdict"], base_rows["mr_verdict"], items_by_set["mr_verdict"], limit=float("-inf"),
                name="mr_verdict_additional", verdict=False)
        else:
            per_student["checks"]["panel"] = {"status": "NOT RUN"}
            per_student["checks"]["knowledge"] = {"status": "NOT RUN"}
            per_student["checks"]["mr_verdict_additional"] = {"status": "NOT RUN"}
        per_student["checks"]["mr_verdict_vs_osler_v4_final"] = {
            "status": "NOT MEASURED",
            "reason": "the osler_v4 final's predictions and weights are not on this machine (operator instruction 2026-11-09)",
        }
        per_student["checks"]["mimic"] = {
            "status": "NOT MEASURED", "reason": "no MIMIC data on this machine (operator instruction: do not touch MIMIC)",
        }
        report["students"][student] = per_student

    # The overall verdict: FAIL if a measured check fails; NOT MEASURED while any O7 check is missing; PASS only if all pass.
    gate_names = ("panel", "knowledge", "mr_verdict_vs_osler_v4_final", "mimic")
    for data in report["students"].values():
        verdicts = [data["checks"][name].get("status") for name in gate_names]
        if "FAIL" in verdicts:
            data["verdict"] = "FAIL"
        elif all(v == "PASS" for v in verdicts):
            data["verdict"] = "PASS"
        else:
            data["verdict"] = "NOT MEASURED (missing checks: mr_verdict vs osler_v4 final, MIMIC)"
    (STEP3 / "gate_o7.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    lines = [
        "# d3_distill — Gate O7", "",
        f"Written `{utcnow()}` by `scripts/d3_distill/step3_gate.py`. Aggregates only; no item text.", "",
        "## Rule (PLAN.md step 3, unchanged from distill_v0)", "",
        "Panel lower bound > 0 against zero-shot pplx-decider-flash-v1.2; knowledge lower bound > -0.02; MR verdict and",
        "MIMIC within -0.02 of the osler_v4 final. The teacher is an additional row; the students are judged on the gate.",
        "Pairing by item id, unpaired above 1 % ⇒ NOT MEASURED (D44(a)); a failing readout cell for either model ⇒ NOT",
        "MEASURED (D12); paired percentile bootstrap, 1,000 resamples, seed 0 (the Gate O4 machinery).", "",
        "## Results", "",
    ]
    for student_name, data in report["students"].items():
        lines += [f"### {student_name}", "", f"Verdict: **{data['verdict']}**", "",
                  "| check | status | rule | accuracy delta [95 %] | Brier delta | paired / unpaired |",
                  "|---|---|---|---|---|---|"]
        for name in ("panel", "knowledge", "mr_verdict_additional", "mr_verdict_vs_osler_v4_final", "mimic"):
            check = data["checks"].get(name, {"status": "NOT RUN"})
            acc = check.get("accuracy")
            brier = check.get("brier")
            accs = "—" if not acc else f"{acc['point']:+.4f} [{acc['lower']:+.4f}, {acc['upper']:+.4f}]"
            bs = "—" if not brier else f"{brier['point']:+.4f} [{brier['lower']:+.4f}, {brier['upper']:+.4f}]"
            pair = "—" if "paired" not in check else f"{check['paired']} / {sum(check['unpaired'].values())}"
            lines.append(f"| {name} | {check.get('status')} | {check.get('rule', check.get('reason', ''))} | {accs} | {bs} | {pair} |")
        lines += ["", "Own figures (template macro accuracy / item-mean Brier):", "",
                  "| set | final student | zero-shot student | pplx-flash |", "|---|---|---|---|"]
        for name in SETS:
            s = data["sets"][name]
            def fmt(entry: dict) -> str:
                if entry.get("status") == "NOT RUN" or "template_macro_accuracy" not in entry:
                    return "NOT RUN"
                return f"{entry['template_macro_accuracy']:.4f} / {entry['brier_item_mean']:.4f} ({entry['scored_items']})"
            lines.append(f"| {name} | {fmt(s['student'])} | {fmt(s['zero_shot'])} | {fmt(s['pplx_flash'])} |")
        lines.append("")
    lines += ["## Notes", "",
              "- The MR check the PLAN defines is against the osler_v4 final; that model's predictions and weights are not",
              "  on this machine, so it is NOT MEASURED and no other number is substituted. The mr_verdict row against",
              "  pplx-flash is additional (the operator's 2026-11-09 instruction), not the PLAN's MR check.",
              "- MIMIC is NOT MEASURED: the operator forbids MIMIC data on this machine.",
              "- The d3_bench 500/1,000-item subsamples are not used as the gate; results.md says they cannot stand in for it.",
              "- Teacher row (additional, d3_bench seeded subsamples only, not the full sets): d3-ultra knowledge 0.805 (1,000),",
              "  panel 0.638 (500), mr_verdict 0.842 (500); d3_bench also has d1-ultra, Clef-Flash and pplx-flash rows there.",
              "- Majority (most frequent gold) baselines of the full sets (osler_v4 C231): mr_verdict 0.4177, panel 0.1941,",
              "  knowledge 0.3118.",
              "- Baseline reproduction check: our locally scored zero-shot pplx run gives template macro knowledge 0.71051",
              "  (+0.0008 vs the published 0.7097), panel 0.63617 (-0.0028 vs 0.6390), mr_verdict 0.70500 (+0.0006 vs",
              "  0.7044); every template passes the readout gate. The paired gate uses these local rows (same items, same",
              "  code) and records the cross-check deltas under `baseline_reproduces_published`; the published row is a",
              "  provenance check, not the comparator.",
              ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({s: d["verdict"] for s, d in report["students"].items()}, indent=1))


if __name__ == "__main__":
    main()
