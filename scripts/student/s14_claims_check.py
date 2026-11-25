"""S14 — recompute the numbers FINDINGS.md quotes, from the artifacts it names (R1).

One function per CLAIMS row added at closure. Every value is read from a raw artifact under
``outputs/`` or ``data/``; nothing is hand-copied. Run one key or all of them:

    uv run python scripts/student/s14_claims_check.py --key s9_run3_run_facts
    uv run python scripts/student/s14_claims_check.py --all --out outputs/student_v0/S14/claims_values.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
S9_RUN3 = REPO / "outputs/student_v0/S9_run3"
S10 = REPO / "outputs/student_v0/S10"
S9_RUN2 = REPO / "outputs/student_v0/S9"
S9_RUN1 = REPO / "outputs/student_v0/S9_run1_sorted"
DIAG = REPO / "outputs/student_v0/S9_diag"
S11 = REPO / "outputs/student_v0/S11"
S12 = REPO / "outputs/student_v0/S12"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return round(value, digits)


# --------------------------------------------------------------------------- S9 run 3
def s9_run3_run_facts() -> dict[str, Any]:
    r = _load(S9_RUN3 / "run.json")
    t = r["training"]
    b = _load(S9_RUN3 / "best.json")
    return {
        "exit": r["exit"],
        "stopped_early": t["stopped_early"],
        "pass_completed": t["pass_completed"],
        "steps": t["steps"],
        "epochs_completed": t["epochs_completed"],
        "items_seen": r["data"]["items_seen"],
        "tokens_seen": r["data"]["tokens_seen"],
        "train_seconds": _round(t["train_seconds"], 1),
        "train_budget_seconds": _round(t["train_budget_seconds"], 1),
        "command": r["command"],
        "items_per_s": _round(t["throughput"]["items_per_s"], 2),
        "tokens_per_s": _round(t["throughput"]["tokens_per_s"], 1),
        "gpu_peak_memory_gb": _round(t["throughput"]["gpu_peak_memory_gb"], 2),
        "n_dev_evals": r["dev_evals"]["n_evals"],
        "n_dev_eval_errors": r["dev_evals"]["n_errors"],
        "dev_sample_sha256": r["dev_evals"]["sample_sha256"],
        "dev_sample_n": r["dev_evals"]["n_items_per_eval"],
        "n_checkpoints_saved": len(r["dev_evals"]["checkpoints_per_eval"]),
        "checkpoint_every_eval": len(r["dev_evals"]["checkpoints_per_eval"]) == r["dev_evals"]["n_evals"],
        "selection_rule": r["dev_evals"]["selection_rule"],
        "selected_step": b["step"],
        "selected_dev_metrics": {
            k: _round(v, 6) for k, v in b["dev_metrics_at_selection"].items() if k != "per_qtype"
        },
        "schedule": t["schedule"],
        "base_model": r["provenance"]["config"]["base_model"],
        "git_commit": r["provenance"]["git_commit"],
        "wall_clock_s": _round(r["wall_clock_s"], 1),
    }


def s9_run3_selection_rows() -> dict[str, Any]:
    r = _load(S9_RUN3 / "run.json")
    traj = {t["step"]: t for t in r["dev_evals"]["trajectory"]}
    out = {"n_evals": len(traj)}
    for step in (1000, 39500, 44000):
        row = traj.get(step)
        if row:
            out[f"step_{step}"] = {
                k: _round(row[k], 6) for k in ("accuracy", "macro_accuracy", "brier", "mean_nll")
            }
    out["alternatives"] = r["dev_evals"].get("additional_brier_first_rule")
    out["best_by_macro"] = r["dev_evals"]["best_step_by_selection_rule"]
    out["max_macro_over_all_evals"] = max(t["macro_accuracy"] for t in traj.values())
    return out


def s9_run3_temperature() -> dict[str, Any]:
    t = _load(S9_RUN3 / "temperature.json")
    return {
        "checkpoint_step": t["checkpoint_step"],
        "n_dev_items_scored": t["n_dev_items_scored"],
        "per_qtype": {
            q: {
                "temperature": _round(v["temperature"], 6),
                "n_items": v["n_items"],
                "status": v["status"],
                "fitted": v["fitted"],
                "nll_before": _round(v["nll_before"], 6),
                "nll_after": _round(v["nll_after"], 6),
            }
            for q, v in t["per_qtype"].items()
        },
    }


def s9_run3_dev_final() -> dict[str, Any]:
    d = _load(S9_RUN3 / "dev_final.json")
    overall = d["calibrated"]["overall"]
    return {
        "n": overall["n"],
        "accuracy": _round(overall["accuracy"], 6),
        "macro_accuracy": _round(overall["macro_accuracy"], 6),
        "brier": _round(overall["brier"], 6),
        "uncalibrated_overall": {
            k: _round(d["uncalibrated"]["overall"][k], 6)
            for k in ("accuracy", "macro_accuracy", "brier")
        },
    }


def s9_run3_cells() -> dict[str, Any]:
    out = {}
    for tag, name in (
        ("official_step_1000", "model_meddecide-0p8b-lora-pointer__test.json"),
        ("converged_step_44000", "model_meddecide-0p8b-lora-pointer__converged.json"),
    ):
        d = _load(S9_RUN3 / name)
        tiers = {}
        for tier in ("tier1", "fresh"):
            cells = [c for c in d["cells"] if c["tier"] == tier]
            scored = d["tiers"][tier]["n_scored"]
            tiers[tier] = {
                "n_scored": scored,
                "coverage": d["tiers"][tier]["coverage"],
                "micro_accuracy": _round(sum(c["n_correct"] for c in cells) / scored, 4),
                "n_gate_pass": d["tiers"][tier]["n_gate_pass"],
                "n_gate_fail": d["tiers"][tier]["n_gate_fail"],
                "per_template": {
                    c["template_id"]: {
                        "n": c["n"],
                        "accuracy": _round(c["accuracy"], 4) if c.get("accuracy") is not None else None,
                        "status": c.get("readout_status") or c.get("status"),
                    }
                    for c in cells
                },
            }
        out[tag] = {"run_id": d["run_id"], "checkpoint_step": d.get("checkpoint_step"), "tiers": tiers}
    return out


# --------------------------------------------------------------------------- S10
def s10_run_facts() -> dict[str, Any]:
    r = _load(S10 / "run.json")
    t = r["training"]
    b = _load(S10 / "best.json")
    traj = r["dev_evals"]["trajectory"]
    return {
        "exit": r["exit"],
        "stopped_early": t["stopped_early"],
        "steps": t["steps"],
        "items_seen": r["data"]["items_seen"],
        "tokens_seen": r["data"]["tokens_seen"],
        "train_seconds": _round(t["train_seconds"], 1),
        "train_budget_seconds": _round(t["train_budget_seconds"], 1),
        "items_per_s": _round(t["throughput"]["items_per_s"], 2),
        "tokens_per_s": _round(t["throughput"]["tokens_per_s"], 1),
        "gpu_peak_memory_gb": _round(t["throughput"]["gpu_peak_memory_gb"], 2),
        "n_dev_evals": r["dev_evals"]["n_evals"],
        "selected_step": b["step"],
        "selected_dev_macro": _round(b["dev_metrics_at_selection"]["macro_accuracy"], 6),
        "runner_up_step": r["dev_evals"]["additional_brier_first_rule"].get("brier_best_step"),
        "last_20_evals_macro_mean": _round(
            sum(x["macro_accuracy"] for x in traj[-20:]) / 20, 4
        ),
        "last_20_evals_macro_stdev": _round(
            float(__import__("statistics").stdev(x["macro_accuracy"] for x in traj[-20:])), 4
        ),
        "command": r["command"],
        "git_commit": r["provenance"]["git_commit"],
    }


def s10_temperature() -> dict[str, Any]:
    t = _load(S10 / "temperature.json")
    return {
        "checkpoint_step": t["checkpoint_step"],
        "per_qtype": {
            q: {"temperature": _round(v["temperature"], 6), "n_items": v["n_items"], "status": v["status"]}
            for q, v in t["per_qtype"].items()
        },
    }


def s10_dev_final() -> dict[str, Any]:
    overall = _load(S10 / "dev_final.json")["calibrated"]["overall"]
    return {
        "n": overall["n"],
        "accuracy": _round(overall["accuracy"], 6),
        "macro_accuracy": _round(overall["macro_accuracy"], 6),
        "brier": _round(overall["brier"], 6),
    }


def s10_cells() -> dict[str, Any]:
    out = {}
    for tag, name in (
        ("official_step_17000", "model_meddecide-0p8b-lora-pointer__test.json"),
        ("converged_step_44000", "model_meddecide-0p8b-lora-pointer__converged.json"),
    ):
        d = _load(S10 / name)
        tiers = {}
        for tier in ("tier1", "fresh"):
            cells = [c for c in d["cells"] if c["tier"] == tier]
            scored = d["tiers"][tier]["n_scored"]
            tiers[tier] = {
                "n_scored": scored,
                "coverage": d["tiers"][tier]["coverage"],
                "micro_accuracy": _round(sum(c["n_correct"] for c in cells) / scored, 4),
                "n_gate_pass": d["tiers"][tier]["n_gate_pass"],
                "n_gate_fail": d["tiers"][tier]["n_gate_fail"],
            }
        out[tag] = {
            "run_id": d["run_id"],
            "model_id": d["model_id"],
            "checkpoint_meta_base_model_id": d.get("checkpoint_meta", {}).get("base_model_id"),
            "tiers": tiers,
        }
    return out


def readout_health_pair() -> dict[str, Any]:
    """Instruct (S9 run 3) vs Base ablation (S10): D12 gates passed, same item sets."""
    out: dict[str, Any] = {}
    for tag, path in (
        ("instruct_s9_run3_official", S9_RUN3 / "model_meddecide-0p8b-lora-pointer__test.json"),
        ("base_s10_official", S10 / "model_meddecide-0p8b-lora-pointer__test.json"),
        ("instruct_s9_run3_converged", S9_RUN3 / "model_meddecide-0p8b-lora-pointer__converged.json"),
        ("base_s10_converged", S10 / "model_meddecide-0p8b-lora-pointer__converged.json"),
    ):
        d = _load(path)
        tier1 = (d["tiers"]["tier1"]["n_gate_pass"], d["tiers"]["tier1"]["n_gate_fail"])
        fresh = (d["tiers"]["fresh"]["n_gate_pass"], d["tiers"]["fresh"]["n_gate_fail"])
        out[tag] = {
            "tier1_pass": tier1[0],
            "tier1_total": tier1[0] + tier1[1],
            "fresh_pass": fresh[0],
            "fresh_total": fresh[0] + fresh[1],
            "total_pass": tier1[0] + fresh[0],
            "total_cells": tier1[0] + tier1[1] + fresh[0] + fresh[1],
        }
    return out


# --------------------------------------------------------------------------- failed runs
def s9_run1_failure() -> dict[str, Any]:
    b = _load(S9_RUN1 / "best.json")
    started = _load(S9_RUN1 / "run_started.json")
    steps = [r for r in _jsonl(S9_RUN1 / "logs/train_steps.jsonl") if r.get("event") == "step"]
    devs = [r for r in _jsonl(S9_RUN1 / "logs/dev_evals.jsonl") if r.get("event") == "dev_eval"]
    best_acc = max(devs, key=lambda d: d["metrics"]["accuracy"])
    tail = devs[-3:]
    return {
        "run_dir": "outputs/student_v0/S9_run1_sorted",
        "started_at_utc": started["started_at_utc"],
        "command": started["command"],
        "git_commit": started["provenance"]["git_commit"],
        "dev_sample_n": started["eval_sample"]["n_items"],
        "selection_rule": started["selection_rule"],
        "best_checkpoint_step": b["step"],
        "best_checkpoint_dev_metrics": {
            k: _round(b["dev_metrics_at_selection"][k], 4)
            for k in ("accuracy", "macro_accuracy", "brier")
        },
        "steps_logged": len(steps),
        "dev_evals_logged": len(devs),
        "best_accuracy_eval": {
            "step": best_acc["step"],
            **{k: _round(best_acc["metrics"][k], 4) for k in ("accuracy", "macro_accuracy", "brier")},
        },
        "last_three_evals": [
            {"step": d["step"], **{k: _round(d["metrics"][k], 4) for k in ("accuracy", "macro_accuracy", "brier")}}
            for d in tail
        ],
        "operator_diagnosis": (
            "iter_batches sorted the whole epoch by char_proxy after shuffling -> an accidental "
            "length curriculum; stopped by operator instruction, kept as failed-by-pipeline"
        ),
        "no_planned_epoch_artifact": not (S9_RUN1 / "planned_epoch.json").exists(),
        "guard_planned_epoch_rho_tokens_run3": _round(
            _load(S9_RUN3 / "planned_epoch.json")["length_trend"]["rho_tokens"], 8
        ),
        "guard_planned_epoch_rho_tokens_s10": _round(
            _load(S10 / "planned_epoch.json")["length_trend"]["rho_tokens"], 8
        ),
        "guard_test": "tests/test_train_student.py::test_iter_batches_has_no_length_curriculum",
    }


def s9_run2_divergence() -> dict[str, Any]:
    r = _load(S9_RUN2 / "run.json")
    steps = [x for x in _jsonl(S9_RUN2 / "logs/train_steps.jsonl") if x.get("event") == "step"]
    devs = [x for x in _jsonl(S9_RUN2 / "logs/dev_evals.jsonl") if x.get("event") == "dev_eval"]
    gnorms = [x["grad_norm"] for x in steps if "grad_norm" in x]
    half = len(gnorms) // 2
    tail = [d for d in devs if d["step"] >= 18000]

    def window(lo: int, hi: int) -> dict[str, Any]:
        sel = sorted(g for g, s in zip(gnorms, (x["step"] for x in steps), strict=True) if lo <= s <= hi)
        if not sel:
            return {}
        return {
            "n": len(sel),
            "p50": _round(sel[len(sel) // 2], 2),
            "p95": _round(sel[int(0.95 * (len(sel) - 1))], 2),
            "max": _round(sel[-1], 1),
            "n_over_100": sum(g > 100 for g in sel),
            "n_over_1000": sum(g > 1000 for g in sel),
        }

    losses = [x["loss"] for x in steps if "loss" in x]
    late = [x["loss"] for x in steps if x["step"] > 18000]
    return {
        "exit": r["exit"],
        "stopped_early": r["training"]["stopped_early"],
        "steps_logged": r["training"]["steps"],
        "items_seen": r["data"]["items_seen"],
        "tokens_seen": r["data"]["tokens_seen"],
        "planned_epoch_rho_tokens": _round(
            _load(S9_RUN2 / "planned_epoch.json")["length_trend"]["rho_tokens"], 8
        ),
        "grad_p50_first_half": _round(sorted(gnorms[:half])[half // 2], 2),
        "grad_p50_second_half": _round(sorted(gnorms[half:])[len(gnorms[half:]) // 2], 2),
        "grad_max": _round(max(gnorms), 1),
        "n_grad_over_1000": sum(g > 1000 for g in gnorms),
        "grad_by_step_window": {
            f"{lo}-{hi}": window(lo, hi)
            for lo, hi in ((1, 4000), (4001, 8000), (8001, 12000), (12001, 16000), (16001, 18000), (18001, 20000), (20001, 21574))
        },
        "loss_over_2_share_after_18000": _round(sum(x > 2 for x in late) / len(late), 4),
        "loss_over_2_share_overall": _round(sum(x > 2 for x in losses) / len(losses), 4),
        "n_dev_evals": len(devs),
        "last_dev_eval": {
            "step": devs[-1]["step"],
            **{k: _round(devs[-1]["metrics"][k], 4) for k in ("accuracy", "macro_accuracy", "brier")},
        },
        "min_dev_macro_after_step_18000": _round(
            min(d["metrics"]["macro_accuracy"] for d in tail), 4
        ),
        "max_dev_macro_after_step_18000": _round(
            max(d["metrics"]["macro_accuracy"] for d in tail), 4
        ),
    }


def train_by_template() -> dict[str, Any]:
    m = _load(REPO / "data/train/student_v0/manifest.json")
    by_template: dict[str, int] = {}
    for _source, templates in m["by_source_template_qtype"]["train"].items():
        for template, count in templates.items():
            by_template[template] = by_template.get(template, 0) + (
                count if isinstance(count, int) else sum(count.values())
            )
    return {
        "by_template": dict(sorted(by_template.items(), key=lambda kv: -kv[1])),
        "sum": sum(by_template.values()),
        "n_train_total": m["totals"]["train"],
        "held_out_templates": m["held_out_templates"],
        "held_out_in_train": {
            t: by_template.get(t, 0) for t in m["held_out_templates"]
        },
        "train_sha256": m["files_sha256"]["train"],
        "leakage_recheck": m["checks"]["leakage_recheck_on_written_train"],
        "built_at_utc": m["built_at_utc"],
        "git_commit": m["git_commit"],
    }


def diag_arms() -> dict[str, Any]:
    d = _load(DIAG / "diag.json")
    return {
        "subset": d["subset"],
        "arms": [
            {
                "arm": a["arm"],
                "recipe": a["recipe"],
                "plan": a["plan"],
                "loss_first_window": _round(a["train_loss"]["first_window_mean"], 4),
                "loss_last_window": _round(a["train_loss"]["last_window_mean"], 4),
                "slope_per_1k": _round(a["train_loss"]["slope_per_1k_steps"], 4),
                "slope_ci95": _round(a["train_loss"]["slope_ci95"], 4),
                "grad_p50": _round(a["grad"]["p50"], 2),
                "grad_p95": _round(a["grad"]["p95"], 2),
                "grad_max": _round(a["grad"]["max"], 1),
                "n_over_100": a["grad"]["n_over_100"],
                "n_over_1000": a["grad"]["n_over_1000"],
                "dev_macro_by_step": {k: _round(v, 4) for k, v in a["dev_by_step"].items()},
            }
            for a in d["arms"]
        ],
        "picked": "d_rank8",
    }


def diag_spikes() -> dict[str, Any]:
    d = _load(DIAG / "diag.json")
    by_arm = {a["arm"]: a["batch_composition"] for a in d["arms"]}
    baseline = by_arm["baseline"]
    cap = by_arm["b_cap2048"]
    return {
        "baseline": {
            "mean_batch_max_item_tokens": _round(baseline["mean_max_item_tokens"], 1),
            "spike_mean_batch_max_item_tokens": _round(baseline["spike_mean_max_item_tokens"], 1),
            "spike_template_share": {k: _round(v, 4) for k, v in baseline["spike_template_share"].items()},
            "overall_template_share": {
                k: _round(v, 4) for k, v in baseline["overall_template_share"].items()
            },
        },
        "b_cap2048": {
            "mean_batch_max_item_tokens": _round(cap["mean_max_item_tokens"], 1),
            "spike_mean_batch_max_item_tokens": _round(cap["spike_mean_max_item_tokens"], 1),
        },
        "grad_max_baseline": _round(
            next(a["grad"]["max"] for a in d["arms"] if a["arm"] == "baseline"), 1
        ),
        "grad_max_b_cap2048": _round(
            next(a["grad"]["max"] for a in d["arms"] if a["arm"] == "b_cap2048"), 1
        ),
        "grad_p95_baseline": _round(next(a["grad"]["p95"] for a in d["arms"] if a["arm"] == "baseline"), 2),
        "grad_p95_b_cap2048": _round(
            next(a["grad"]["p95"] for a in d["arms"] if a["arm"] == "b_cap2048"), 2
        ),
    }


def padding_checks() -> dict[str, Any]:
    out: dict[str, Any] = {"from_file": {}, "from_diag_json": {}}
    for name in (
        "padding_check_fresh.json",
        "padding_check_fresh_fp32.json",
        "padding_check_trained.json",
        "padding_check_trained_fp32.json",
    ):
        p = _load(DIAG / name)
        out["from_file"][name] = {
            "verdict": p["verdict"],
            "tolerance": p["tolerance"],
            "max_abs_delta_padded": _round(p["alone_vs_left_padded_batch"]["max_abs_delta"], 6),
            "mean_abs_delta_padded": _round(p["alone_vs_left_padded_batch"]["mean_abs_delta"], 6),
            "max_abs_delta_uniform_control": _round(
                p["alone_vs_uniform_batch_control"]["max_abs_delta"], 6
            ),
            "dtype": p["model"]["dtype"],
            "checkpoint": p["checkpoint"],
        }
    trained = out["from_file"]["padding_check_trained_fp32.json"]
    fresh = out["from_file"]["padding_check_fresh_fp32.json"]
    out["trained_fp32_over_control"] = _round(
        trained["max_abs_delta_padded"] / trained["max_abs_delta_uniform_control"], 2
    )
    out["trained_fp32_over_fresh_fp32"] = _round(
        trained["max_abs_delta_padded"] / fresh["max_abs_delta_padded"], 2
    )
    return out


# --------------------------------------------------------------------------- G1 / S11
def g1_verdict() -> dict[str, Any]:
    g = _load(S11 / "g1.json")
    return {
        "verdicts": g["verdicts"],
        "provenance": g["provenance"],
        "set_sizes": g["set_sizes"],
        "not_measured": g["not_measured"],
        "dropped": g["dropped"],
        "excluded_cells": [
            {
                "model": c["model"],
                "tier": c["tier"],
                "template_id": c["template_id"],
                "status": c["status"],
                "failures": c["failures"],
            }
            for c in g["excluded_cells"]
        ],
    }


def g1_paired() -> dict[str, Any]:
    g = _load(S11 / "g1.json")
    out: dict[str, Any] = {}
    for set_name, per_baseline in g["comparisons"].items():
        for baseline, c in per_baseline.items():
            metrics = {}
            for name, m in (c.get("metrics") or {}).items():
                metrics[name] = {
                    "point": _round(m.get("point", m.get("difference")), 6),
                    "ci95": [_round(x, 6) for x in (m.get("ci95") or [])],
                    "reference_point": _round(m.get("reference_marginal_point", m.get("reference_point")), 6),
                    "baseline_point": _round(m.get("baseline_marginal_point", m.get("baseline_point")), 6),
                    "reference_marginal_ci95": [
                        _round(x, 6) for x in (m.get("reference_marginal_ci95") or [])
                    ],
                    "baseline_marginal_ci95": [
                        _round(x, 6) for x in (m.get("baseline_marginal_ci95") or [])
                    ],
                    "unpaired_ci95": [_round(x, 6) for x in (m.get("unpaired_ci95") or [])],
                }
            out[f"{set_name}|{baseline}"] = {
                "per_template": {
                    r["template_id"]: {
                        "n": r["n"],
                        "reference_accuracy": _round(r["reference_accuracy"], 4),
                        "baseline_accuracy": _round(r["baseline_accuracy"], 4),
                        "difference": _round(r["difference"], 4),
                        "reference_brier": _round(r["reference_brier"], 4),
                        "baseline_brier": _round(r["baseline_brier"], 4),
                    }
                    for r in (c.get("per_template") or [])
                },
                "n_items": c.get("n_items"),
                "n_reference_scored": c.get("n_reference_scored"),
                "n_baseline_scored": c.get("n_baseline_scored"),
                "n_templates": c.get("n_templates"),
                "templates": c.get("templates"),
                "metrics": metrics,
            }
    return out


def g1_heldout_templates() -> dict[str, Any]:
    g = _load(S11 / "g1.json")
    out: dict[str, Any] = {}
    for baseline, c in g["comparisons"]["fresh_heldout"].items():
        out[baseline] = {
            "n_items": c.get("n_items"),
            "per_template": {
                r["template_id"]: {
                    "n": r["n"],
                    "reference_accuracy": _round(r["reference_accuracy"], 4),
                    "baseline_accuracy": _round(r["baseline_accuracy"], 4),
                    "difference": _round(r["difference"], 4),
                    "reference_brier": _round(r["reference_brier"], 4),
                    "baseline_brier": _round(r["baseline_brier"], 4),
                }
                for r in c.get("per_template") or []
            },
        }
    return out


def item_set_identity() -> dict[str, Any]:
    def ids(path: Path) -> set[str]:
        return {r["item_id"] for r in _jsonl(path)}

    trained = ids(S9_RUN3 / "preds_meddecide-0p8b-lora-pointer__test.jsonl")
    base = ids(S10 / "preds_meddecide-0p8b-lora-pointer__test.jsonl")
    zero = ids(S11 / "preds_qwen3p5-0p8b.jsonl")
    return {
        "n_trained_run3": len(trained),
        "n_base_s10": len(base),
        "n_zeroshot": len(zero),
        "shared_all_three": len(trained & base & zero),
        "only_trained": len(trained - base - zero),
        "only_base": len(base - trained - zero),
        "only_zeroshot": len(zero - trained - base),
    }


def zeroshot_cells() -> dict[str, Any]:
    d = _load(S11 / "model_qwen3p5-0p8b.json")
    out = {}
    for tier in ("tier1", "fresh"):
        cells = [c for c in d["cells"] if c["tier"] == tier]
        scored = d["tiers"][tier]["n_scored"]
        out[tier] = {
            "n_scored": scored,
            "coverage": d["tiers"][tier]["coverage"],
            "micro_accuracy": _round(sum(c["n_correct"] for c in cells) / scored, 4),
            "n_gate_pass": d["tiers"][tier]["n_gate_pass"],
            "n_gate_fail": d["tiers"][tier]["n_gate_fail"],
        }
    return out


def jev9b_not_measured() -> dict[str, Any]:
    present = sorted(p.name for p in S11.iterdir() if "jev" in p.name.lower())
    return {
        "s11_files_with_jev_in_name": present,
        "v0_2_jev_predictions_exist": bool(present),
        "note": "F7 measured JEV-9B on v0.1, not v0.2; G1 reports it as a missing baseline",
    }


# --------------------------------------------------------------------------- S12
def s12_byte_identity() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tag, name in (
        ("s9_run3_best", "byte_identity_S9_run3_best.json"),
        ("s10_best", "byte_identity_S10_best.json"),
    ):
        p = S12 / name
        if not p.exists():
            out[tag] = "NOT MEASURED — file not written"
            continue
        d = _load(p)
        out[tag] = {
            "checkpoint": d["checkpoint"],
            "base_model_id": d["base_model_id"],
            "dtype": d["dtype"],
            "device": d["device"],
            "kernel_path": d["kernel_path"],
            "checkpoint_files": d["checkpoint_files"],
            "sets": {
                name_: {
                    k: v for k, v in s.items() if k not in ("rows", "templates")
                }
                for name_, s in d["prompt_sets"].items()
            },
            "wall_seconds": d["wall_seconds"],
            "git_commit": d["git_commit"],
        }
    return out


def s12_audit_readback() -> dict[str, Any]:
    audit = REPO / "outputs/bench_v0_fix0/F10/audit_v0.jsonl"
    staged = REPO / "outputs/bench_v0_fix0/F10/audit_sample.jsonl"
    return {
        "audit_v0_jsonl_exists": audit.exists(),
        "audit_sample_jsonl_exists": staged.exists(),
        "verdict": (
            "NOT MEASURED — audit pending (the operator's completed audit file "
            "outputs/bench_v0_fix0/F10/audit_v0.jsonl does not exist)"
            if not audit.exists()
            else "measured"
        ),
    }


def checkpoint_meta() -> dict[str, Any]:
    """The shipped checkpoints' reconstructing metadata (rank, head, parameter count)."""
    out: dict[str, Any] = {}
    for tag, path in (("s9_run3_best", S9_RUN3 / "best"), ("s10_best", S10 / "best")):
        m = _load(path / "model.json")
        out[tag] = {
            "base_model_id": m["base_model_id"],
            "dtype": m["dtype"],
            "variant": m["variant"],
            "max_prompt_tokens": m["max_prompt_tokens"],
            "lora": {
                "r": m["lora"]["r"],
                "alpha": m["lora"]["alpha"],
                "n_target_modules": len(m["lora"]["target_modules"]),
            },
            "head": m["head"],
            "calibration": m["calibration"],
            "trainable_parameter_count": m["trainable_parameter_count"],
        }
    return out


KEYS = {
    "s9_run3_run_facts": s9_run3_run_facts,
    "s9_run3_selection_rows": s9_run3_selection_rows,
    "s9_run3_temperature": s9_run3_temperature,
    "s9_run3_dev_final": s9_run3_dev_final,
    "s9_run3_cells": s9_run3_cells,
    "s10_run_facts": s10_run_facts,
    "s10_temperature": s10_temperature,
    "checkpoint_meta": checkpoint_meta,
    "s10_dev_final": s10_dev_final,
    "s10_cells": s10_cells,
    "readout_health_pair": readout_health_pair,
    "s9_run1_failure": s9_run1_failure,
    "s9_run2_divergence": s9_run2_divergence,
    "train_by_template": train_by_template,
    "diag_arms": diag_arms,
    "diag_spikes": diag_spikes,
    "padding_checks": padding_checks,
    "g1_verdict": g1_verdict,
    "g1_paired": g1_paired,
    "g1_heldout_templates": g1_heldout_templates,
    "item_set_identity": item_set_identity,
    "zeroshot_cells": zeroshot_cells,
    "jev9b_not_measured": jev9b_not_measured,
    "s12_byte_identity": s12_byte_identity,
    "s12_audit_readback": s12_audit_readback,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", choices=sorted(KEYS), help="one key to recompute")
    parser.add_argument("--all", action="store_true", help="recompute every key")
    parser.add_argument("--out", type=Path, help="write the JSON payload here")
    args = parser.parse_args()
    if not args.key and not args.all:
        parser.error("give --key <name> or --all")
    keys = sorted(KEYS) if args.all else [args.key]
    payload = {key: KEYS[key]() for key in keys}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
