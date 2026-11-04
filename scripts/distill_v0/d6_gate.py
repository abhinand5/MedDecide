"""D6: Gate O7 for both distillation students, the ≤10B frontier and the scoreboard (distill_v0).

Scores each student on the nine frozen Y9 sets with the temperatures fitted on dev, then runs the four
pre-registered checks with Gate O4's estimator (item-paired bootstrap within templates, 1,000
resamples, seed 0; ``meddecide.eval.gate_o4``):

| id | check | limit |
|---|---|---|
| i | panel accuracy, student - zero-shot pplx-decider-flash-v1.2 | lower bound > 0 |
| ii | knowledge accuracy, student - zero-shot pplx-decider-flash-v1.2 | lower bound > -0.02 |
| iii | MR verdict accuracy, student - osler_v4 final | lower bound > -0.02 |
| iv | MIMIC suite accuracy, student - osler_v4 final | lower bound > -0.02 |

Gate O7 PASS iff i-iv hold for at least one student; the other student is reported as a row, not a
verdict. The ≤10B frontier is a claim, not a gate: a student is on it when no measured ≤10B model is
≥ it on every set it has and > it on at least one.

Usage (GPU for scoring; CPU for --reuse):
    source scripts/pod_env.sh && uv run --frozen python scripts/distill_v0/d6_gate.py --out outputs/distill_v0/D6 \
        --student4b outputs/distill_v0/D4/checkpoints/final_step_063304 \
        --student9b outputs/distill_v0/D5/checkpoints/final_step_063304
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from meddecide.eval.gate_o4 import comparison, normalise_row, pair_rows, readout_failures
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.utils.io import iter_jsonl, write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
Y9 = REPO / "data" / "bench" / "osler_v4" / "y9"
BASELINES = REPO / "outputs" / "osler_v4" / "Y9" / "baselines"
Z10 = REPO / "outputs" / "osler_v5" / "Z10"
RECIPE = REPO / "configs" / "distill_v0" / "student_4b.yaml"
SETS = ("knowledge", "panel", "mr_verdict", "mednli_test", "mr_reported", "mr_heldout_test",
        "mimic_suite", "pairs_test", "mimic_heldout")
CHECKS = (
    ("i", "panel", "pplx_flash", 0.0),
    ("ii", "knowledge", "pplx_flash", -0.02),
    ("iii", "mr_verdict", "final", -0.02),
    ("iv", "mimic_suite", "final", -0.02),
)
# model key -> (label, baseline directory prefix, ≤10B?)
FRONTIER_MODELS = {
    "student4b": ("Osler-4B (distill_v0)", None, True),
    "student9b": ("Osler-9B (distill_v0)", None, True),
    "pplx_flash": ("zero-shot pplx-decider-flash-v1.2", "pplx", True),
    "final": ("osler_v4 final (Osler-4B)", "final", True),
    "md4b": ("MedDecider-4B", "md4b", True),
    "md9b": ("MedDecider-9B", "md9b", True),
    "jev9b": ("JEV-9B", "jev9b", True),
    "osler_v2": ("osler_v2 Osler-4B", "osler_v2", True),
    "pplx27b": ("pplx-decider-v1.2-27b (ceiling, outside the class)", "pplx27b", False),
    "clef_flash": ("zero-shot Clef-Flash", "clef_flash", True),
    "qwen4b": ("zero-shot Qwen3.5-4B", "qwen35_4b", True),
    "qwen9b": ("zero-shot Qwen3.5-9B", "qwen35_9b", True),
}
MEDDECIDER_CAP = 10


def load_items(set_name: str) -> list[dict[str, Any]]:
    items = []
    for row in iter_jsonl(Y9 / f"{set_name}.jsonl"):
        items.append({**row, "raw_id": str(row["item_id"]), "item_id": mix_row_to_item(row).item_id})
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError(f"{set_name}: the recomputed item ids are not unique")
    return items


def load_rows(path: Path, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Rows keyed either by the recomputed item id (trainer-based runners) or by the set file's own id (zero-shot runners)."""
    by_computed = {str(item["item_id"]): item for item in items}
    by_raw = {str(item["raw_id"]): item for item in items}
    rows = []
    skipped = 0
    for raw in iter_jsonl(path):
        item = by_computed.get(str(raw["item_id"])) or by_raw.get(str(raw["item_id"]))
        if item is None:
            raise ValueError(f"{path}: a prediction outside the set")
        if raw.get("status", "scored") != "scored":
            skipped += 1
            continue
        rows.append(normalise_row(raw, item))
    return {"rows": rows, "path": str(path), "skipped_by_runner": skipped}


Y4_BASELINES = REPO / "outputs" / "osler_v4" / "Y4" / "baselines"
Z10_PATHS = {"clef_flash": Z10 / "clef_flash", "qwen4b": Z10 / "zeroshot" / "qwen35_4b",
             "qwen9b": Z10 / "zeroshot" / "qwen35_9b"}
# the MIMIC sets were scored before the Y9 grid existed; the Y9 gate reads the Y4 runs for them
Y4_RUNS = {"pplx_flash": {"mimic_suite": "suite_flash", "mimic_heldout": "heldout_flash"},
           "md4b": {"mimic_suite": "suite_meddecider4b", "mimic_heldout": "heldout_meddecider4b"},
           "md9b": {"mimic_suite": "suite_meddecider9b"},
           "osler_v2": {"mimic_suite": "suite_osler_v2", "mimic_heldout": "heldout_osler_v2"}}


def baseline_path(model: str, set_name: str) -> Path | None:
    if model in Z10_PATHS:
        path = Z10_PATHS[model] / f"{set_name}.jsonl"
        return path if path.exists() else None
    if model == "jev9b":
        path = BASELINES / f"jev9b_{set_name}" / "jev-9b" / "predictions.jsonl"
        return path if path.exists() else None
    if set_name in Y4_RUNS.get(model, {}):
        path = Y4_BASELINES / Y4_RUNS[model][set_name] / "predictions.jsonl"
        return path if path.exists() else None
    prefix = FRONTIER_MODELS[model][1]
    if prefix is None:
        return None
    path = BASELINES / f"{prefix}_{set_name}" / "predictions.jsonl"
    return path if path.exists() else None


def load_model(checkpoint: Path) -> tuple[Any, str]:
    from meddecide.model.meddecide_model import MedDecideModel
    from meddecide.model.pplx import PplxDecisionModel

    if (checkpoint / "decision_config.json").exists():
        return PplxDecisionModel.load(checkpoint, device="cuda:0", dtype="bfloat16"), "pplx"
    return MedDecideModel.load(checkpoint, device="cuda:0", dtype="bfloat16"), "meddecide"


def score_student(model: Any, prefix: str, out: Path, sets: tuple[str, ...]) -> dict[str, Any]:
    config = load_config(RECIPE)
    model.eval_mode()
    if not model.calibration:
        raise SystemExit(f"{prefix}: the checkpoint carries no temperatures; not scored (ADVISORY §5)")
    counts: dict[str, Any] = {}
    for set_name in sets:
        items = [mix_row_to_item(row) for row in iter_jsonl(Y9 / f"{set_name}.jsonl")]
        scored = model.score_items(items, batch_size=config.eval_batch_size or config.batch_size,
                                   max_batch_tokens=config.eval_max_batch_tokens or config.max_batch_tokens,
                                   max_prompt_tokens=config.max_prompt_tokens,
                                   round_to_chunk=config.eval_round_to_chunk,
                                   length_buckets=config.eval_length_buckets)
        calibrated = model.apply_temperature(scored)
        directory = out / f"{prefix}_{set_name}"
        directory.mkdir(parents=True, exist_ok=True)
        write_jsonl(directory / "predictions.jsonl", dev_prediction_rows(calibrated))
        counts[set_name] = {"items": len(items), "scored": len(items)}
        print(f"{prefix} {set_name}: {len(items)} items scored", flush=True)
    return counts


def run_checks(student: dict[str, Any], baselines: dict[str, dict[str, Any]],
               items_by_set: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for check_id, set_name, base, limit in CHECKS:
        rows = student.get(set_name)
        base_rows = baselines[base].get(set_name)
        if rows is None or base_rows is None:
            results[check_id] = {"check": check_id, "set": set_name, "baseline": base, "verdict": "NOT MEASURED",
                                 "reason": "no rows for the student or the baseline"}
            continue
        cap = MEDDECIDER_CAP if base in {"md4b", "md9b"} else None
        pairing = pair_rows(rows["rows"], base_rows["rows"], items_by_set[set_name], option_cap=cap)
        student_failures = readout_failures(rows["rows"])
        base_failures = readout_failures(base_rows["rows"])
        if pairing.unpaired_share > 0.01:
            verdict, reason, stats = "NOT MEASURED", f"{pairing.unpaired_total} of {pairing.items} unpaired", None
        elif student_failures or base_failures:
            verdict, reason, stats = "NOT MEASURED", f"readout gate fails: {student_failures + base_failures} (D12)", None
        else:
            paired = comparison(pairing)
            stats = {"point": paired.accuracy.point, "lower": paired.accuracy.lo, "upper": paired.accuracy.hi,
                     "brier_point": paired.brier.point, "brier_lower": paired.brier.lo, "brier_upper": paired.brier.hi,
                     "items": paired.items, "templates": len(paired.templates)}
            verdict = "PASS" if paired.accuracy.lo > limit else "FAIL"
            reason = f"accuracy lower bound {paired.accuracy.lo:+.4f} {'>' if verdict == 'PASS' else '<='} {limit:+.2f}"
        results[check_id] = {"check": check_id, "set": set_name, "baseline": base, "limit": limit,
                             "verdict": verdict, "reason": reason, "accuracy": stats,
                             "paired_items": len(pairing.pairs), "unpaired": pairing.unpaired,
                             "unpaired_share": round(pairing.unpaired_share, 6),
                             "student_readout_failures": student_failures, "baseline_readout_failures": base_failures}
    verdicts = [entry["verdict"] for entry in results.values()]
    gate_verdict = "PASS" if all(v == "PASS" for v in verdicts) else (
        "FAIL" if any(v == "FAIL" for v in verdicts) else "NOT MEASURED")
    return {"verdict": gate_verdict, "checks": results}


def scoreboard_row(rows: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows["rows"]:
        return {"items": 0, "accuracy_micro": None, "accuracy_template_macro": None}
    templates: dict[str, list[float]] = {}
    correct = 0
    for row in rows["rows"]:
        templates.setdefault(str(row["template_id"]), []).append(1.0 if row["correct"] else 0.0)
        correct += int(row["correct"])
    macro = sum(sum(v) / len(v) for v in templates.values()) / len(templates)
    return {"items": len(rows["rows"]), "correct": correct, "accuracy_micro": correct / len(rows["rows"]),
            "accuracy_template_macro": macro, "templates": len(templates),
            "skipped_by_runner": rows["skipped_by_runner"]}


def frontier(scoreboard: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """For each student, the ≤10B models that dominate it (no worse on every shared set, better on one)."""
    students = ("student4b", "student9b")
    out: dict[str, Any] = {}
    for student in students:
        own = scoreboard.get(student, {})
        dominators = []
        for model, (label, _prefix, inside) in FRONTIER_MODELS.items():
            if model == student or not inside:
                continue
            theirs = scoreboard.get(model, {})
            shared = [s for s in SETS if own.get(s, {}).get("accuracy_micro") is not None
                      and theirs.get(s, {}).get("accuracy_micro") is not None]
            if not shared:
                continue
            no_worse = all(theirs[s]["accuracy_micro"] >= own[s]["accuracy_micro"] for s in shared)
            better = any(theirs[s]["accuracy_micro"] > own[s]["accuracy_micro"] for s in shared)
            if no_worse and better:
                dominators.append({"model": model, "label": label, "sets": len(shared),
                                   "min_margin": min(theirs[s]["accuracy_micro"] - own[s]["accuracy_micro"]
                                                     for s in shared)})
        out[student] = {"on_frontier": not dominators, "dominated_by": dominators}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="outputs/distill_v0/D6")
    parser.add_argument("--student4b", default="outputs/distill_v0/D4/checkpoints/final_step_063304")
    parser.add_argument("--student9b", default="outputs/distill_v0/D5/checkpoints/final_step_063304")
    parser.add_argument("--students", nargs="*", default=["student4b", "student9b"])
    parser.add_argument("--reuse", action="store_true", help="score nothing; read existing predictions under --out")
    parser.add_argument("--validate", action="store_true",
                        help="wiring check: treat the osler_v4 final's Y9 rows as the student and compare against zero-shot "
                             "pplx on all four gate sets (must reproduce osler_v4's published Gate O4 numbers)")
    args = parser.parse_args()
    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    started = utcnow()
    items_by_set = {name: load_items(name) for name in SETS}
    checkpoints = {"student4b": Path(args.student4b), "student9b": Path(args.student9b)}
    checkpoints = {key: (path if path.is_absolute() else REPO / path) for key, path in checkpoints.items()}

    baselines: dict[str, dict[str, Any]] = {}
    for key in ("pplx_flash", "final", "md4b", "md9b", "jev9b", "osler_v2", "pplx27b", "clef_flash", "qwen4b", "qwen9b"):
        baselines[key] = {}
        for set_name in SETS:
            path = baseline_path(key, set_name)
            if path is not None:
                baselines[key][set_name] = load_rows(path, items_by_set[set_name])


    if args.validate:
        student = {"final": baselines["final"]}
        validation: dict[str, Any] = {}
        for set_name in ("panel", "knowledge", "mr_verdict", "mimic_suite"):
            pairing = pair_rows(student["final"][set_name]["rows"], baselines["pplx_flash"][set_name]["rows"],
                                items_by_set[set_name])
            paired = comparison(pairing)
            validation[set_name] = {"point": paired.accuracy.point, "lower": paired.accuracy.lo,
                                    "upper": paired.accuracy.hi, "items": paired.items,
                                    "templates": len(paired.templates)}
        write_json(out / "validate_gate.json", {"task": "D6 wiring validation (osler_v4 final vs pplx)",
                                                "validation": validation})
        print(json.dumps(validation, indent=1), flush=True)
        print("published Gate O4 (osler_v4): panel -0.0421 [-0.058,-0.027]; knowledge -0.0067; "
              "MR +0.1387; MIMIC +0.1654", flush=True)
        return


    scored: dict[str, Any] = {}
    for student in args.students:
        prefix = student
        if args.reuse:
            rows = {}
            for set_name in SETS:
                path = out / f"{prefix}_{set_name}" / "predictions.jsonl"
                if path.exists():
                    rows[set_name] = load_rows(path, items_by_set[set_name])
                else:
                    rows[set_name] = {"rows": [], "path": "", "skipped_by_runner": 0}
        else:
            model, _kind = load_model(checkpoints[student])
            score_student(model, prefix, out, SETS)
            del model
            import torch
            torch.cuda.empty_cache()
            rows = {name: load_rows(out / f"{prefix}_{name}" / "predictions.jsonl", items_by_set[name]) for name in SETS}
        scored[student] = rows

    gate: dict[str, Any] = {}
    for student in args.students:
        gate[student] = run_checks(scored[student], baselines, items_by_set)

    scoreboard: dict[str, dict[str, Any]] = {}
    for student in args.students:
        scoreboard[student] = {name: scoreboard_row(scored[student][name], items_by_set[name]) for name in SETS}
    for key, rows in baselines.items():
        scoreboard[key] = {name: scoreboard_row(rows[name], items_by_set[name]) for name in SETS if name in rows}

    result = {
        "task": "D6 Gate O7", "git_commit": git_commit(REPO), "started_utc": started, "finished_utc": utcnow(),
        "checkpoints": {key: str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
                        for key, path in checkpoints.items()},
        "students": args.students,
        "gate_o7": gate,
        "verdict": ("PASS" if any(entry["verdict"] == "PASS" for entry in gate.values())
                    else ("FAIL" if any(entry["verdict"] == "FAIL" for entry in gate.values()) else "NOT MEASURED")),
        "scoreboard": scoreboard,
        "frontier": frontier(scoreboard),
        "sets": list(SETS),
    }
    write_json(out / "gate_o7.json", result)
    print(json.dumps({student: {"verdict": gate[student]["verdict"],
                                "checks": {k: v["verdict"] for k, v in gate[student]["checks"].items()}}
                      for student in args.students}, indent=1), flush=True)
    print(f"Gate O7: {result['verdict']}", flush=True)


if __name__ == "__main__":
    main()
