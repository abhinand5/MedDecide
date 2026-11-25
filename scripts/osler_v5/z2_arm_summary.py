"""osler_v5 Z2 per-arm summary (CPU): each arm's evaluations as recorded by run_arm.py (outputs/osler_v5/Z2/<arm>/evaluations.json):
the composite parts (MR and MIMIC template macro, CG micro), knowledge dev, and the positional health in the three orders.
Writes outputs/osler_v5/Z2/<arm>/summary.json. Usage (CPU):  uv run --frozen python scripts/osler_v5/z2_arm_summary.py R0 [R0p R2 R3]"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
Z2 = REPO / "outputs" / "osler_v5" / "Z2"


def main() -> None:
    arms = sys.argv[1:] or ["R0", "R0p", "R2", "R3"]
    for arm in arms:
        path = Z2 / arm / "evaluations.json"
        if not path.exists():
            print(f"{arm}: NOT MEASURED — no evaluations.json yet", flush=True)
            continue
        ev = json.loads(path.read_text(encoding="utf-8"))
        out = {}
        for step, r in sorted(ev.items(), key=lambda kv: int(kv[0])):
            po, pp, c = r["positional_original"], r["positional_permuted"], r["composite"]
            out[step] = {
                "S": c["S"], "mr_macro": c["mr"], "mimic_macro": c["mimic"], "cg_micro": c["cg_micro"],
                "knowledge_dev_micro": r["knowledge_dev_accuracy"],
                "positional": {"original": [po["two_letter_accuracy"], po["single_letter_accuracy"], po["passes"]],
                               "seed1": [pp["1"]["two_letter_accuracy"], pp["1"]["single_letter_accuracy"], pp["1"]["passes"]],
                               "seed2": [pp["2"]["two_letter_accuracy"], pp["2"]["single_letter_accuracy"], pp["2"]["passes"]]},
                "positional_passes_all_orders": r["positional_passes_all_orders"],
            }
            print(f"{arm} step {step}: S {c['S']:.4f} MR {c['mr']:.4f} MIMIC {c['mimic']:.4f} CG micro {c['cg_micro']:.4f} "
                  f"knowledge {r['knowledge_dev_accuracy']:.4f} | positional all orders {r['positional_passes_all_orders']} "
                  f"(original two/single {po['two_letter_accuracy']:.4f}/{po['single_letter_accuracy']:.4f})", flush=True)
        (Z2 / arm / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
