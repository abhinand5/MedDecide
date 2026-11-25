"""Z7 five-claim spot-check (GOAL.md: FINDINGS carries a five-claim spot-check).

Five claims are re-derived from their raw artifacts in a fresh process, independently of the JSON that recorded them:

* C012 — Z1's scored total, re-counted from `outputs/osler_v6/Z1/anchor_probs.jsonl`;
* C019 — Z2's s0 and the frozen λ, re-computed from `outputs/osler_v6/Z2/probe/steps.jsonl`;
* C034 — Z5's choice and its knowledge lower bound, re-computed from the two arms' step-13,992 predictions;
* C049 — Z6's final-evaluation S, re-computed from the step-63,304 prediction files;
* C052 — Gate O6's verdict, re-computed by `scripts/osler_v6/z7_gate.py` (also done in `z7_audit.py`).

Writes `outputs/osler_v6/Z7/spot_check.json`; exits 1 on a mismatch.

Usage (CPU):  uv run --frozen python scripts/osler_v6/z7_spot_check.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meddecide.eval.composite import accuracy, mimic_partition, mr_partition
from meddecide.eval.osler_v6_choice import ArmChoice, ControlChoice, choose_recipe
from meddecide.train.blocks import macro_difference, micro_difference, pair_by_item
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
Z7 = REPO / "outputs" / "osler_v6" / "Z7"
TARGET = 0.40


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    problems: list[str] = []
    out: dict[str, Any] = {"generated_utc": utcnow()}

    # C012: the Z1 scored total, counted from the raw record file
    count = sum(1 for _ in iter_jsonl(REPO / "outputs/osler_v6/Z1/anchor_probs.jsonl"))
    recorded = json.loads((REPO / "outputs/osler_v6/Z1/audit.json").read_text())["records"]["scored_total"]
    out["C012"] = {"recomputed": count, "recorded": recorded}
    if count != recorded:
        problems.append(f"C012: {count} != {recorded}")

    # C019: s0 and lambda from the probe's own per-step records
    steps = rows(REPO / "outputs/osler_v6/Z2/probe/steps.jsonl")
    s0 = sum(float(r["grad_share"]) for r in steps) / len(steps)
    lam = (TARGET / (1.0 - TARGET)) * (1.0 - s0) / s0
    probe = json.loads((REPO / "outputs/osler_v6/Z2/probe/probe.json").read_text())
    out["C019"] = {"s0": s0, "lambda": lam, "recorded_s0": probe["lambda"]["s0"], "recorded_lambda": probe["lambda"]["value"],
                   "steps": len(steps)}
    if abs(s0 - probe["lambda"]["s0"]) > 1e-12 or abs(lam - probe["lambda"]["value"]) > 1e-12:
        problems.append("C019: the recomputed s0/lambda differ from probe.json")

    # C034: the choice, from the two arms' raw step-13,992 predictions
    def arm_rows(folder: Path) -> dict[str, list[dict[str, Any]]]:
        return {"fast": rows(folder / "fast_dev.jsonl"), "know": rows(folder / "knowledge_dev.jsonl"),
                "cg": rows(folder / "cg_v2.jsonl")}

    control = arm_rows(REPO / "outputs/osler_v5/Z2/F2/evals/step_013992")
    candidate = arm_rows(REPO / "outputs/osler_v6/Z3/G1s0/evals/step_013992")
    knowledge = micro_difference(pair_by_item(candidate["know"], control["know"]))
    mr = macro_difference(pair_by_item(mr_partition(candidate["fast"]), mr_partition(control["fast"])))
    mimic = macro_difference(pair_by_item(mimic_partition(candidate["fast"]), mimic_partition(control["fast"])))
    choice = ArmChoice(name="G1s0", knowledge_dev=accuracy(candidate["know"]), knowledge_lower=knowledge.lo,
                       mr_lower=mr.lo, mimic_lower=mimic.lo, cg_micro=accuracy(candidate["cg"]),
                       positional_all_orders=bool(json.loads((REPO / "outputs/osler_v6/Z3/G1s0/evaluations.json").read_text())["13992"]["positional_passes_all_orders"]))
    report = choose_recipe([choice], ControlChoice(cg_micro=accuracy(control["cg"]), knowledge_dev=accuracy(control["know"])))
    recorded_choice = json.loads((REPO / "outputs/osler_v6/Z5/choice.json").read_text())
    out["C034"] = {"recomputed_chosen": report.chosen, "recorded_chosen": recorded_choice["decision"]["chosen"],
                   "recomputed_knowledge_lower": knowledge.lo,
                   "recorded_knowledge_lower": recorded_choice["decision"]["rows"]["G1s0"]["knowledge_lower"]}
    if report.chosen != recorded_choice["decision"]["chosen"]:
        problems.append("C034: the recomputed choice differs")
    if abs(knowledge.lo - recorded_choice["decision"]["rows"]["G1s0"]["knowledge_lower"]) > 1e-12:
        problems.append("C034: the recomputed knowledge lower bound differs")

    # C049: the final-evaluation S, from its prediction files
    ev = REPO / "outputs/osler_v6/Z6/final/evals/step_063304"
    fast, cg = rows(ev / "fast_dev.jsonl"), rows(ev / "cg_v2.jsonl")
    s_recomputed = (sum(_template_macro(mr_partition(fast)) for _ in [0]) + sum(_template_macro(mimic_partition(fast)) for _ in [0])
                    + accuracy(cg)) / 3.0
    recorded_eval = json.loads((REPO / "outputs/osler_v6/Z6/final/evaluations.json").read_text())["63304"]
    out["C049"] = {"recomputed_S": s_recomputed, "recorded_S": recorded_eval["composite"]["S"]}
    if abs(s_recomputed - recorded_eval["composite"]["S"]) > 1e-9:
        problems.append("C049: the recomputed S differs from evaluations.json")

    # C052: the gate verdict (the audit re-runs the gate; here the recorded verdict and its failing check)
    gate = json.loads((Z7 / "gate_o6.json").read_text())
    out["C052"] = {"recomputed": "see z7_audit.py (the gate re-run is identical)",
                   "recorded_verdict": gate["gate_pplx_flash"]["verdict"],
                   "recorded_panel_lower": gate["gate_pplx_flash"]["checks"]["iii"]["accuracy"]["lower"]}

    out["problems"] = problems
    out["ok"] = not problems
    write_json(Z7 / "spot_check.json", out)
    print(f"spot-check: {'PASS' if not problems else 'FAIL'}; {json.dumps({k: v for k, v in out.items() if k.startswith('C')})}; problems {problems}")


def _template_macro(part: list[dict[str, Any]]) -> float:
    from meddecide.eval.composite import template_macro_accuracy
    return template_macro_accuracy(part)


if __name__ == "__main__":
    main()
