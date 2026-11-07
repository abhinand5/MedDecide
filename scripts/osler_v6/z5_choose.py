"""Z5 (ADVISORY §2.6): the pre-registered choice rule at 13,992 steps, on dev only.

Inputs are the step-13,992 evaluation files of the G1 arms (`outputs/osler_v6/Z3/<arm>/evals/step_013992/`) and of the G0
reference, osler_v5's F2 seed-0 arm (`outputs/osler_v5/Z2/F2/evals/step_013992/`, C062 — checked, not re-scored). G0 is not
a candidate; it supplies the comparison and its CG v2 micro accuracy. Seed 0 is the candidate; a seed-1 arm is a stability
check that is reported beside it and never replaces it.

The statistics are osler_v5's paired dev statistics, unchanged: knowledge dev v2 by item micro, MR and MIMIC by template
macro, paired within templates, 1,000 resamples, seed 0 (`meddecide.train.blocks`). The rule and its strict limits are in
`meddecide.eval.osler_v6_choice`.

Usage (CPU):  uv run --frozen python scripts/osler_v6/z5_choose.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meddecide.eval.composite import accuracy, mimic_partition, mr_partition
from meddecide.eval.osler_v6_choice import ArmChoice, ChoiceReport, ControlChoice, choose_recipe
from meddecide.train.blocks import macro_difference, micro_difference, pair_by_item
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
V5 = REPO / "outputs" / "osler_v5" / "Z2"
V6 = REPO / "outputs" / "osler_v6"
Z5 = V6 / "Z5"
PROBE = V6 / "Z2" / "probe" / "probe.json"
STEP = 13_992
CONTROL_ARM = "F2"  # osler_v5 F2 seed 0: the G0 reference (distill_weight 0)
CANDIDATE = "G1s0"  # seed 0 is the candidate (ADVISORY §2.6 seed rule)
STABILITY = "G1s1"


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_arm(folder: Path) -> dict[str, Any] | None:
    """The arm's step-13,992 predictions and evaluation record, or None when the arm has no such evaluation."""
    evaluation = folder / "evals" / f"step_{STEP:06d}"
    result_path = folder / "arm_result.json"
    if not (evaluation / "fast_dev.jsonl").exists() or not result_path.exists():
        return None
    result = json.loads(result_path.read_text())
    record = result["evaluations"][str(STEP)]
    return {
        "fast": read_rows(evaluation / "fast_dev.jsonl"),
        "know": read_rows(evaluation / "knowledge_dev.jsonl"),
        "cg": read_rows(evaluation / "cg_v2.jsonl"),
        "evaluation": record,
        "config": result.get("config", {}),
        "git_commit": result.get("git_commit"),
        "finished_utc": result.get("finished_utc"),
    }


def arm_choice(name: str, arm: dict[str, Any], control: dict[str, Any]) -> tuple[ArmChoice, dict[str, Any]]:
    """The rule's inputs for one arm and the underlying paired statistics."""
    knowledge = micro_difference(pair_by_item(arm["know"], control["know"]))
    mr = macro_difference(pair_by_item(mr_partition(arm["fast"]), mr_partition(control["fast"])))
    mimic = macro_difference(pair_by_item(mimic_partition(arm["fast"]), mimic_partition(control["fast"])))
    choice = ArmChoice(
        name=name,
        knowledge_dev=accuracy(arm["know"]),
        knowledge_lower=knowledge.lo,
        mr_lower=mr.lo,
        mimic_lower=mimic.lo,
        cg_micro=accuracy(arm["cg"]),
        positional_all_orders=bool(arm["evaluation"]["positional_passes_all_orders"]),
    )
    stats = {
        "knowledge_micro_vs_G0": knowledge.as_dict(),
        "mr_macro_vs_G0": mr.as_dict(),
        "mimic_macro_vs_G0": mimic.as_dict(),
        "items": {"knowledge": len(arm["know"]), "mr": len(mr_partition(arm["fast"])),
                  "mimic": len(mimic_partition(arm["fast"])), "cg": len(arm["cg"])},
        "positional": {
            "original": arm["evaluation"]["positional_original"],
            "permuted": arm["evaluation"]["positional_permuted"],
            "passes_all_orders": bool(arm["evaluation"]["positional_passes_all_orders"]),
        },
        "training_git_commit": arm["git_commit"],
        "finished_utc": arm["finished_utc"],
    }
    return choice, stats


def main() -> None:
    control_arm = load_arm(V5 / CONTROL_ARM)
    if control_arm is None:
        raise SystemExit(f"{V5 / CONTROL_ARM} has no step-{STEP} evaluation: the G0 reference is missing")
    if float(control_arm["config"].get("distill_weight", 0.0)) != 0.0:
        raise SystemExit("the G0 reference was not run with distill_weight 0: stop")
    if int(control_arm["config"].get("seed", -1)) != 0:
        raise SystemExit("the G0 reference is not seed 0: stop")
    candidate_arm = load_arm(V6 / "Z3" / CANDIDATE)
    if candidate_arm is None:
        raise SystemExit(f"{V6 / 'Z3' / CANDIDATE} has no step-{STEP} evaluation: Z5 depends on Z3")
    probe = json.loads(PROBE.read_text()) if PROBE.exists() else None
    if (probe is not None and probe.get("status") == "DONE"
            and float(candidate_arm["config"].get("distill_weight", -1.0)) != float(probe["lambda"]["value"])):
        raise SystemExit("the arm's distill_weight is not the frozen λ from Z2: stop")
    if int(candidate_arm["config"].get("distill_min_labelled_per_batch", 0)) != 3:
        raise SystemExit("the arm was not run with the D57 quota of 3 labelled anchors per batch: stop")

    control = ControlChoice(name=f"G0 ({CONTROL_ARM} seed 0)", cg_micro=accuracy(control_arm["cg"]),
                            knowledge_dev=accuracy(control_arm["know"]))
    candidates: list[ArmChoice] = []
    stats: dict[str, Any] = {}
    choice, stats[CANDIDATE] = arm_choice(CANDIDATE, candidate_arm, control_arm)
    candidates.append(choice)
    stability = load_arm(V6 / "Z3" / STABILITY)
    if stability is not None:
        _, stats[STABILITY] = arm_choice(STABILITY, stability, control_arm)
        if int(stability["config"].get("seed", -1)) != 1:
            raise SystemExit("the stability arm is not seed 1: stop")
    report: ChoiceReport = choose_recipe(candidates, control)
    payload = {
        "task": "Z5",
        "step": STEP,
        "rule": "ADVISORY §2.6, implemented in meddecide.eval.osler_v6_choice",
        "candidate": CANDIDATE,
        "stability_check": STABILITY if stability is not None else "NOT RUN",
        "control": {"name": control.name, "cg_micro": control.cg_micro, "knowledge_dev": control.knowledge_dev,
                    "folder": str((V5 / CONTROL_ARM).relative_to(REPO)), "config": control_arm["config"],
                    "git_commit": control_arm["git_commit"]},
        "arm_statistics": stats,
        "decision": report.to_dict(),
        "probe_lambda": probe.get("lambda") if probe else None,
        "generated_utc": utcnow(),
        "git_commit": git_commit(REPO),
    }
    Z5.mkdir(parents=True, exist_ok=True)
    write_json(Z5 / "choice.json", payload)
    for name, row in report.rows.items():
        print(f"{name}: qualifies {row['qualifies']}; failed {row['failed'] or 'none'}; "
              f"knowledge {row['knowledge_dev']:.4f} (lower {row['knowledge_lower']:+.4f}); "
              f"MR {row['mr_lower']:+.4f}; MIMIC {row['mimic_lower']:+.4f}; CG {row['cg_micro']:.4f}")
    print(f"choice: {report.chosen or 'none'}; {report.outcome}", flush=True)


if __name__ == "__main__":
    main()
