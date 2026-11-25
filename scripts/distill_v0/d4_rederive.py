"""D4/D5 self-audit (R6): re-derive the training run's numbers from its raw artifacts.

Reads ``arm_result.json``, ``run_provenance.json``, ``logs/train.jsonl``, ``logs/dose.jsonl``,
``logs/lambda_calibration.json``, ``evaluations.json``, ``dose_tripwire_1000.json`` and the checkpoint
directories, and recomputes: the dose summary, the calibration consistency, the evaluation grid and
positional health, the checkpoint presence, the temperature fits, and the provenance links to the D2
pool and label file. Writes ``audit.json`` and exits non-zero on any failure. This is the recompute
command behind the D4/D5 CLAIMS rows.

Usage (CPU):
    uv run --frozen python scripts/distill_v0/d4_rederive.py --out outputs/distill_v0/D4 --steps 63304
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from meddecide.train.trainer import evaluation_grid
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
D2 = REPO / "outputs" / "distill_v0" / "D2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="outputs/distill_v0/D4")
    parser.add_argument("--steps", type=int, default=63_304)
    args = parser.parse_args()
    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    arm = json.loads((out / "arm_result.json").read_text(encoding="utf-8"))
    provenance = json.loads((out / "run_provenance.json").read_text(encoding="utf-8"))
    dose = read_jsonl(out / "logs" / "dose.jsonl")
    train = read_jsonl(out / "logs" / "train.jsonl")
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # --- run completion and schedule ---------------------------------------------------------------------------------
    check("steps", arm["steps"] == args.steps or bool(arm.get("stopped_early")),
          {"steps": arm["steps"], "target": args.steps, "stopped_early": arm.get("stopped_early")})
    check("evaluation_grid", [int(s) for s in arm["evaluation_steps"]] == list(evaluation_grid(args.steps, 10)),
          {"recorded": arm["evaluation_steps"], "expected": list(evaluation_grid(args.steps, 10))})
    evaluations = arm.get("evaluations") or {}
    check("evaluations_complete", len(evaluations) == len(arm["evaluation_steps"]),
          {"evaluations": len(evaluations), "grid": len(arm["evaluation_steps"])})
    positional_ok = all(bool(entry.get("positional_passes_all_orders")) for entry in evaluations.values())
    check("positional_health_all_evaluations", positional_ok,
          {step: entry.get("positional_passes_all_orders") for step, entry in evaluations.items()})

    # --- calibration ---------------------------------------------------------------------------------------------------
    lambda_path = out / "logs" / "lambda_calibration.json"
    if lambda_path.exists():
        calibration = json.loads(lambda_path.read_text(encoding="utf-8"))
        measured = calibration["records"]
        check("lambda_records_within_window", len(measured) <= 200 and all(record["step"] <= 200 for record in measured),
              {"records": len(measured), "max_step": max(record["step"] for record in measured) if measured else None})
        check("lambda_share_target", abs(float(calibration["predicted_share"]) - float(calibration["target"])) <= 0.1,
              {"predicted_share": calibration["predicted_share"], "target": calibration["target"],
               "lambda": calibration["lambda"]})
        events = [row for row in train if row.get("event") == "distill_lambda"]
        check("lambda_frozen_once", len(events) == 1 and float(events[0]["lambda"]) == float(calibration["lambda"]),
              {"events": events})
        check("lambda_positive", float(calibration["lambda"]) > 0, calibration["lambda"])
    else:
        check("lambda_calibration_present", False, "missing lambda_calibration.json")

    # --- dose, recomputed from the per-step log --------------------------------------------------------------------------
    steps = [row["step"] for row in dose]
    check("dose_step_range", bool(steps) and steps[0] == 1 and steps[-1] == arm["steps"],
          {"first": steps[0] if steps else None, "last": steps[-1] if steps else None, "arm_steps": arm["steps"]})
    items = sum(int(row["n_items"]) for row in dose)
    teacher = sum(int(row["n_teacher"]) for row in dose)
    mimic = sum(1 for row in dose for template in row["templates"] if str(template).startswith("mimic_"))
    per_batch = [int(row["n_teacher"]) for row in dose]
    recomputed = {
        "steps": len(dose), "items": items, "teacher_items": teacher,
        "teacher_share": teacher / items if items else None, "mimic_share": mimic / items if items else None,
        "mean_teacher_per_batch": sum(per_batch) / len(per_batch) if per_batch else None,
        "mean_items_per_batch": items / len(dose) if dose else None,
        "mean_teacher_per_8_slots": 8 * teacher / items if items else None,
        "share_batches_below_3": sum(1 for value in per_batch if value < 3) / len(per_batch) if per_batch else None,
    }
    recorded = arm.get("dose") or {}
    check("dose_steps", recorded.get("steps") == recomputed["steps"], {"recorded": recorded.get("steps"),
                                                                      "recomputed": recomputed["steps"]})
    check("dose_teacher_share", abs(float(recorded.get("teacher_share") or -1) - recomputed["teacher_share"]) <= 1e-9,
          {"recorded": recorded.get("teacher_share"), "recomputed": recomputed["teacher_share"]})
    check("dose_mimic_share", abs(float(recorded.get("mimic_share") or -1) - recomputed["mimic_share"]) <= 1e-9,
          {"recorded": recorded.get("mimic_share"), "recomputed": recomputed["mimic_share"]})
    check("dose_items", recorded.get("items") == recomputed["items"], {"recorded": recorded.get("items"),
                                                                      "recomputed": recomputed["items"]})
    tripwire_path = out / "dose_tripwire_1000.json"
    if tripwire_path.exists():
        tripwire = json.loads(tripwire_path.read_text(encoding="utf-8"))
        check("dose_tripwire_bands",
              tripwire["teacher_band"][0] <= tripwire["teacher_share"] <= tripwire["teacher_band"][1]
              and tripwire["mimic_band"][0] <= tripwire["mimic_share"] <= tripwire["mimic_band"][1],
              tripwire)
    else:
        check("dose_tripwire_present", False, "missing dose_tripwire_1000.json")

    # --- checkpoints and calibration ----------------------------------------------------------------------------------
    final_dir = out / "checkpoints" / f"final_step_{arm['final_step']:06d}"
    check("final_checkpoint", final_dir.is_dir() and (final_dir / "model.json").exists(), str(final_dir))
    full = sorted((out / "full").glob("step_*")) if (out / "full").exists() else []
    check("full_checkpoints_present", bool(full), {"count": len(full), "newest": full[-1].name if full else None})
    check("temperature_fits", bool(arm.get("temperature_fits")),
          {key: value.get("temperature") if isinstance(value, dict) else value
           for key, value in (arm.get("temperature_fits") or {}).items()})
    config = arm.get("config") or {}
    check("recipe", config.get("seed") == 0 and float(config.get("lambda_brier", 0)) == 1.0
          and config.get("loss_weighting") == "per_item" and config.get("lora", {}).get("r") == 32
          and config.get("lora", {}).get("alpha") == 32 and config.get("distill_calibration_steps") == 200,
          {key: config.get(key) for key in ("seed", "lambda_brier", "loss_weighting", "distill_calibration_steps")})

    # --- provenance: the training pool and labels are D2's --------------------------------------------------------------
    d2_plan = json.loads((D2 / "composition.json").read_text(encoding="utf-8"))
    d2_manifest = json.loads((D2 / "label_manifest.json").read_text(encoding="utf-8"))
    check("pool_matches_d2", arm.get("pool_text_sha256") == d2_plan["pool"]["pool_text_sha256"],
          {"train": arm.get("pool_text_sha256"), "d2": d2_plan["pool"]["pool_text_sha256"]})
    check("labels_match_d2", arm.get("labels_sha256") == d2_manifest["labels_sha256"],
          {"train": arm.get("labels_sha256"), "d2": d2_manifest["labels_sha256"]})
    check("provenance_teacher_items", int(provenance.get("teacher_items") or -1) == int((arm.get("attach") or {}).get("attached") or -2),
          {"provenance": provenance.get("teacher_items"), "arm": (arm.get("attach") or {}).get("attached")})
    check("attach_counts", (arm.get("attach") or {}).get("attached") == (arm.get("attach") or {}).get("basket_rows")
          and (arm.get("attach") or {}).get("missing_label") == 0,
          arm.get("attach"))

    audit = {"task": "training self-audit", "audited_utc": utcnow(), "out": str(out),
             "checks": checks, "all_passed": all(entry["ok"] for entry in checks),
             "recomputed_dose": recomputed, "train_log_lines": len(train)}
    (out / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for entry in checks:
        print(f"[{'PASS' if entry['ok'] else 'FAIL'}] {entry['check']}")
    print(json.dumps(recomputed, indent=1))
    if not audit["all_passed"]:
        raise SystemExit("training self-audit failed")


if __name__ == "__main__":
    main()
