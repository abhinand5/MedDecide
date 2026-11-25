"""osler_v5 Z2 (ADVISORY Z2): apply the pre-registered choice rule to the four recipe arms at step 13,992 (CPU; dev rows only).

For each arm, from its evaluation rows at step 13,992: the composite parts (MR and MIMIC template macro; CG v2 micro), the positional
health in the three orders, and the paired bootstrap lower bounds against R0 (within templates, 1,000 resamples, seed 0): knowledge dev
on micro accuracy (the keep rule's knowledge statistic), MR and MIMIC on template macro (the keep rule's MR statistic and the MIMIC
suite's Gate O4 statistic). The rule itself is in meddecide.eval.arm_choice and is not changed here.

Writes outputs/osler_v5/Z2/choice.json. Usage (CPU):
    uv run --frozen python scripts/osler_v5/z2_choose.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meddecide.eval.arm_choice import ArmStats, choose
from meddecide.eval.composite import (
    accuracy,
    composite,
    mimic_partition,
    mr_partition,
    positional_health,
    template_macro_accuracy,
)
from meddecide.train.blocks import macro_difference, micro_difference, pair_by_item
from meddecide.utils.io import write_json

REPO = Path(__file__).resolve().parents[2]
Z2 = REPO / "outputs" / "osler_v5" / "Z2"
STEP = 13_992
ARMS = ("R0", "R0p", "R2", "R3")  # the tie order: a higher index is the fuller fix
CONTROL, REPEAT = "R0", "R0p"
DDX_PREFIX = "cg_v2_ddxplus"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load(arm: str) -> dict[str, list[dict[str, Any]]]:
    folder = Z2 / arm / "evals" / f"step_{STEP:06d}"
    return {
        "fast": read_jsonl(folder / "fast_dev.jsonl"),
        "cg": read_jsonl(folder / "cg_v2.jsonl"),
        "know": read_jsonl(folder / "knowledge_dev.jsonl"),
        "seed1": read_jsonl(folder / "positional_seed1.jsonl"),
        "seed2": read_jsonl(folder / "positional_seed2.jsonl"),
    }


def main() -> None:
    rows = {arm: load(arm) for arm in ARMS}
    control = rows[CONTROL]
    report_arms: dict[str, Any] = {}
    stats: list[ArmStats] = []
    for arm in ARMS:
        data = rows[arm]
        parts = composite(data["fast"], data["cg"])
        ddx = [r for r in data["cg"] if str(r["template_id"]).startswith(DDX_PREFIX)]
        health = {"original": positional_health(ddx).to_dict(), "seed1": positional_health(data["seed1"]).to_dict(),
                  "seed2": positional_health(data["seed2"]).to_dict()}
        passes = all(h["passes"] for h in health.values())
        if arm == CONTROL:
            know_lo = mr_lo = mimic_lo = None
            diffs: dict[str, Any] = {}
        else:
            k = micro_difference(pair_by_item(data["know"], control["know"]))
            m = macro_difference(pair_by_item(mr_partition(data["fast"]), mr_partition(control["fast"])))
            mi = macro_difference(pair_by_item(mimic_partition(data["fast"]), mimic_partition(control["fast"])))
            know_lo, mr_lo, mimic_lo = k.lo, m.lo, mi.lo
            diffs = {"knowledge_micro_vs_R0": k.as_dict(), "mr_macro_vs_R0": m.as_dict(), "mimic_macro_vs_R0": mi.as_dict()}
        stats.append(ArmStats(name=arm, cg_micro=parts.cg_micro, positional_passes=passes, knowledge_lower=know_lo,
                              mr_lower=mr_lo, mimic_lower=mimic_lo))
        report_arms[arm] = {"composite": parts.to_dict(), "knowledge_micro": accuracy(data["know"]),
                            "knowledge_rows": len(data["know"]), "positional": health, "differences": diffs,
                            "mr_template_macro": template_macro_accuracy(mr_partition(data["fast"])),
                            "mimic_template_macro": template_macro_accuracy(mimic_partition(data["fast"]))}
    decision = choose(stats, control=CONTROL, repeat=REPEAT, order=ARMS)
    out = {"step": STEP, "arms": report_arms, "decision": decision}
    write_json(Z2 / "choice.json", out)
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
