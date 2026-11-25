"""osler_v5 Z2 choice under D56c (CPU): the rule of ADVISORY §8.4 applied to the three corrected arms at step 13,992.

Inputs: each arm's summary (outputs/osler_v5/Z2/<arm>/summary.json, written by z2_arm_summary.py) and the paired bounds against F0
(outputs/osler_v5/Z2/pairs_vs_F0.json, written by z2_pairs.py --control F0 F2 F2s1). F0 is the control, so its own bounds are zero.
Writes outputs/osler_v5/Z2/choice_d56c.json and prints one line per arm. The rule and its limits are in meddecide.eval.d56c_choice.
The R-arm chooser (scripts/osler_v5/z2_choose.py, with meddecide.eval.arm_choice) is the superseded design and is not used here.

Usage (CPU):  uv run --frozen python scripts/osler_v5/z2_choose_d56c.py
"""

from __future__ import annotations

import json
from pathlib import Path

from meddecide.eval.d56c_choice import ArmResult, choose_arm
from meddecide.utils.io import write_json

REPO = Path(__file__).resolve().parents[2]
Z2 = REPO / "outputs" / "osler_v5" / "Z2"
FINAL_STEP = "13992"
CONTROL = "F0"
FIX_LEVEL = {"F0": 0, "F2": 1, "F2s1": 1}  # F2s1 is F2 at seed 1, so it is not a fuller fix than F2


def arm_result(name: str, pairs: dict, summary: dict) -> ArmResult:
    if name == CONTROL:
        bounds = {"knowledge": 0.0, "mr": 0.0, "mimic": 0.0}
    else:
        bounds = {"knowledge": pairs[name]["knowledge_micro_vs_F0"]["ci_lo"], "mr": pairs[name]["mr_macro_vs_F0"]["ci_lo"],
                  "mimic": pairs[name]["mimic_macro_vs_F0"]["ci_lo"]}
    return ArmResult(name=name, fix_level=FIX_LEVEL[name], positional_all_orders=bool(summary["positional_passes_all_orders"]),
                     knowledge_lower=bounds["knowledge"], mr_lower=bounds["mr"], mimic_lower=bounds["mimic"],
                     cg_micro=float(summary["cg_micro"]))


def main() -> None:
    pairs = json.loads((Z2 / f"pairs_vs_{CONTROL}.json").read_text(encoding="utf-8"))
    arms = []
    for name in FIX_LEVEL:
        summary = json.loads((Z2 / name / "summary.json").read_text(encoding="utf-8"))[FINAL_STEP]
        arms.append(arm_result(name, pairs, summary))
    report = choose_arm(arms)
    report.update({"step": int(FINAL_STEP), "control": CONTROL, "rule": "ADVISORY §8.4 (D56c); meddecide.eval.d56c_choice"})
    write_json(Z2 / "choice_d56c.json", report)
    for name, row in report["rows"].items():
        print(f"{name}: qualifies {row['qualifies']}; failed {row['failed'] or 'none'}; CG micro {row['cg_micro']:.4f}", flush=True)
    print(f"choice: {report['chosen'] or 'none'}; {report['outcome']}", flush=True)


if __name__ == "__main__":
    main()
