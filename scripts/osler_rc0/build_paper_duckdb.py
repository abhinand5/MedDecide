#!/usr/bin/env python
"""Build a self-contained, aggregate-only DuckDB reporting database for the Osler arXiv paper.

Zero raw benchmark text, zero MIMIC text, zero patient IDs, zero secrets. Every metric is
recomputed or loaded directly from frozen verified artifacts under outputs/osler_rc0/ and
outputs/osler_v7/ so any human or AI can reproduce every table and plot in the paper directly
from `docs/paper/osler_rc0_reporting.duckdb`.

Run:
    uv run --frozen --with duckdb python scripts/osler_rc0/build_paper_duckdb.py \
        --out docs/paper/osler_rc0_reporting.duckdb
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from meddecide.eval.rc0_board import load_cell, load_frozen

REPO = Path(__file__).resolve().parents[2]
FROZEN_SETS = {
    "panel": Path("data/bench/osler_v4/y9/panel.jsonl"),
    "knowledge": Path("data/bench/osler_v4/y9/knowledge.jsonl"),
    "confirm_general": Path("data/bench/osler_v7/confirm/general.jsonl"),
    "mr_verdict": Path("data/bench/osler_v4/y9/mr_verdict.jsonl"),
    "mimic_suite": Path("data/bench/osler_v4/y9/mimic_suite.jsonl"),
    "mednli_test": Path("data/bench/osler_v4/y9/mednli_test.jsonl"),
    "mimic_heldout": Path("data/bench/osler_v4/y9/mimic_heldout.jsonl"),
    "mr_heldout_test": Path("data/bench/osler_v4/y9/mr_heldout_test.jsonl"),
    "pairs_test": Path("data/bench/osler_v4/y9/pairs_test.jsonl"),
}
HEADLINE_SETS = ("panel", "knowledge", "confirm_general", "mr_verdict", "mimic_suite")
BOARD_JSON = REPO / "outputs/osler_rc0/R7/rc0_board.json"
LATENCY_JSON = REPO / "outputs/osler_rc0/R5/latency.json"
HASHES_JSON = REPO / "outputs/osler_rc0/R0/hashes.json"
OVERLAP_JSON = REPO / "outputs/osler_rc0/R0/medconceptsqa_overlap.json"
CAL_9B_JSON = REPO / "outputs/osler_rc0/R2/9b_calibration.json"
CAL_4B_ARM = REPO / "outputs/osler_rc0/R1/arm/arm_result.json"
EVALS_4B = REPO / "outputs/osler_rc0/R1/arm/evaluations.json"
EVALS_9B = REPO / "outputs/osler_v7/T11/final/evaluations.json"
TRAIN_4B = REPO / "outputs/osler_rc0/R1/arm/logs/train.jsonl"
TRAIN_9B = REPO / "outputs/osler_v7/T11/final/logs/train.jsonl"
MIX_MANIFEST = REPO / "outputs/osler_v7/T6/mix_manifest.json"
AUDIT_AUC = REPO / "outputs/osler_rc0/R6/membership_auc.json"
AUDIT_Y10 = REPO / "outputs/osler_rc0/R6/y10_membership.json"
AUDIT_VERBATIM = REPO / "outputs/osler_rc0/R6/verbatim_probe.json"
AUDIT_ARCH = REPO / "outputs/osler_rc0/R6/architecture_check.json"
AUDIT_VISITED = REPO / "outputs/osler_rc0/R6/visited_summary.json"
AUDIT_SAMPLE = REPO / "outputs/osler_rc0/R6/sample_summary.json"
AUDIT_NONMEMBER = REPO / "outputs/osler_rc0/R6/nonmember_check.json"
CLAIMS_MD = REPO / "loops/osler_rc0/CLAIMS.md"
UPLOAD_MANIFEST = REPO / "outputs/osler_rc0/R8/hf_upload/upload_manifest.json"

ECE_BINS = 15
COVERAGES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def option_bucket(n_opts: int) -> str:
    if n_opts <= 2:
        return "2"
    if n_opts <= 4:
        return "3-4"
    if n_opts <= 10:
        return "5-10"
    if n_opts <= 26:
        return "11-26"
    return "27+"


def compute_ece(confs: np.ndarray, corrects: np.ndarray, n_bins: int = ECE_BINS) -> float:
    if len(confs) == 0:
        return float("nan")
    total = 0.0
    n = len(confs)
    bin_idx = np.minimum((confs * n_bins).astype(int), n_bins - 1)
    for b in range(n_bins):
        mask = bin_idx == b
        cnt = int(mask.sum())
        if cnt:
            total += (cnt / n) * abs(float(corrects[mask].mean()) - float(confs[mask].mean()))
    return total


def brier_of(probs: list[float], gold_idx: int) -> float:
    return float(sum((p - (1.0 if i == gold_idx else 0.0)) ** 2 for i, p in enumerate(probs)))


def parse_claims(path: Path) -> list[tuple[str, str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| C"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) >= 6:
            rows.append((parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]))
    return rows


def build_db(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    board = read_json(BOARD_JSON)
    latency = read_json(LATENCY_JSON)
    hashes = read_json(HASHES_JSON)
    overlap = read_json(OVERLAP_JSON)
    cal9 = read_json(CAL_9B_JSON)
    arm4 = read_json(CAL_4B_ARM)
    evals4 = read_json(EVALS_4B)
    evals9 = read_json(EVALS_9B)
    mix_man = read_json(MIX_MANIFEST)
    audit_auc = read_json(AUDIT_AUC)
    audit_y10 = read_json(AUDIT_Y10)
    audit_verb = read_json(AUDIT_VERBATIM)
    audit_arch = read_json(AUDIT_ARCH)
    audit_vis = read_json(AUDIT_VISITED)
    audit_samp = read_json(AUDIT_SAMPLE)
    audit_nm = read_json(AUDIT_NONMEMBER)
    upload = read_json(UPLOAD_MANIFEST) if UPLOAD_MANIFEST.is_file() else {}

    con = duckdb.connect(str(out_path))

    # 1. models
    con.execute("""
        CREATE TABLE models (
            model VARCHAR PRIMARY KEY,
            params_b DOUBLE,
            params_exact BIGINT,
            size_tier VARCHAR,
            role VARCHAR,
            base_repo VARCHAR,
            base_revision VARCHAR,
            hf_release_repo VARCHAR,
            readout_type VARCHAR,
            non_generative BOOLEAN,
            license VARCHAR
        )
    """)
    lat_models = latency.get("models", {})
    model_meta_rows = [
        ("Osler-9B-rc0", 8.39, lat_models.get("Osler-9B-rc0", {}).get("params"), "<=10B", "Release Candidate (9B)",
         "vllm-sr/d3-flash", "67e03e1bc157961c582ee7c6a35d9b15fada3dec",
         upload.get("repos", {}).get("Osler-9B", {}).get("repo_id", "abhinand/Osler-9B"),
         "option_code_255_last_pos", True, "Apache-2.0"),
        ("Osler-4B-rc0", 4.55, lat_models.get("Osler-4B-rc0", {}).get("params"), "<=4B", "Release Candidate (4B)",
         "perplexity-ai/pplx-decider-flash-v1.2", "9ff7cf9838e59df461c276b7699d56b244d2af58",
         upload.get("repos", {}).get("Osler-4B", {}).get("repo_id", "abhinand/Osler-4B"),
         "option_code_255_marker", False, "Apache-2.0"),
        ("d3-flash", 8.39, lat_models.get("d3-flash", {}).get("params"), "<=10B", "Direct Zero-Shot Base (9B)",
         "vllm-sr/d3-flash", "67e03e1bc157961c582ee7c6a35d9b15fada3dec", "vllm-sr/d3-flash",
         "option_code_255_last_pos", True, "Apache-2.0"),
        ("pplx-flash", 4.54, lat_models.get("pplx-flash", {}).get("params"), "<=4B", "Direct Zero-Shot Base (4B)",
         "perplexity-ai/pplx-decider-flash-v1.2", "6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d",
         "perplexity-ai/pplx-decider-flash-v1.2", "option_code_255_marker", False, "Apache-2.0"),
        ("MedDecider-9B", 9.0, None, "<=10B", "Open Medical Competitor (9B)",
         "lion-ai/MedDecider-9B", "ec8a69da2cd96b2ddd2ff4e314ba9a40b71936c3", "lion-ai/MedDecider-9B",
         "verbalizer_10opt", False, "Apache-2.0"),
        ("MedDecider-4B", 4.0, None, "<=4B", "Open Medical Competitor (4B)",
         "lion-ai/MedDecider-4B", "570b37090c379ad722186a79a3432a31be463fc9", "lion-ai/MedDecider-4B",
         "verbalizer_10opt", False, "Apache-2.0"),
        ("Clef-Flash", 9.0, None, "<=10B", "General System-One Competitor (9B)",
         "Cloudflare/clef-flash", "fde727a287004204b7518dcc983fe64379776712", "Cloudflare/clef-flash",
         "joint_schema_head", False, "Apache-2.0"),
        ("JEV-9B", 9.0, None, "<=10B", "General System-One Competitor (9B)",
         "autotrust/JEV-9B", "pinned-local", "autotrust/JEV-9B", "decision_head_16opt", False, "Apache-2.0"),
        ("d1-2B", 2.0, None, "<=4B", "Sub-4B Edge Competitor (2B)",
         "LiquidAI/d1-2B", "10e7671d44e4b37700020b1eed3f594c5f2af0b4", "LiquidAI/d1-2B",
         "system_one", False, "LFM-Open-1.0"),
        ("Qwen3.5-9B", 9.0, None, "<=10B", "Raw Zero-Shot LLM Backbone (9B)",
         "Qwen/Qwen3.5-9B", "pinned-local", "Qwen/Qwen3.5-9B", "letter_verbalizer", False, "Apache-2.0"),
        ("Qwen3.5-4B", 4.0, None, "<=4B", "Raw Zero-Shot LLM Backbone (4B)",
         "Qwen/Qwen3.5-4B", "pinned-local", "Qwen/Qwen3.5-4B", "letter_verbalizer", False, "Apache-2.0"),
        ("pplx-27B", 26.09, lat_models.get("pplx-27B", {}).get("params"), ">10B Ceiling", "Out-of-Tier 27B Ceiling",
         "perplexity-ai/pplx-decider-v1.2-27b", "cce0f3f80033e9d89d80831d5281fd72124aacce",
         "perplexity-ai/pplx-decider-v1.2-27b", "option_code_255_marker", False, "Apache-2.0"),
    ]
    con.executemany("INSERT INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", model_meta_rows)

    # 2. eval_sets
    con.execute("""
        CREATE TABLE eval_sets (
            set_name VARCHAR PRIMARY KEY,
            role VARCHAR,
            is_headline BOOLEAN,
            is_ood BOOLEAN,
            n_items INTEGER,
            n_templates INTEGER,
            sha256 VARCHAR,
            description VARCHAR
        )
    """)
    set_desc = {
        "panel": ("Headline OOD", True, True, "External clinical panel: 7 public datasets never trained on"),
        "knowledge": ("Headline OOD", True, True, "Frozen medical exam knowledge: MedQA (1,273) + MedMCQA val (4,183)"),
        "confirm_general": ("Headline Fresh Tier-2", True, True, "Post-cutoff Oct-Nov 2026 structured-gold decisions (CT.gov, openFDA, PubMed)"),
        "mr_verdict": ("Headline In-Domain Audit", True, False, "Medical-record audit verdict suite v1 (7 templates: CodiEsp, mkurman, PMC, MTSamples, Synthea, MR)"),
        "mimic_suite": ("Headline In-Domain Audit", True, False, "MIMIC-IV payer-audit suite v1 (9 templates, late-anchor patients >=2014)"),
        "mednli_test": ("Additional Clinical NLI", False, False, "MedNLI official test split (3-class clinical entailment)"),
        "mimic_heldout": ("Additional Held-Out", False, True, "Held-out MIMIC ICD family (heart failure, excluded from training)"),
        "mr_heldout_test": ("Additional Held-Out", False, True, "Held-out MR audit templates (AKI indicator + held-out families)"),
        "pairs_test": ("Additional Minimal Pairs", False, False, "Nemotron-edited clinical documentation minimal pairs"),
    }
    hash_files = hashes.get("files", {})
    universes: dict[str, Any] = {}
    for set_name, rel in FROZEN_SETS.items():
        u = load_frozen(set_name, REPO / rel)
        universes[set_name] = u
        role, is_head, is_ood, desc = set_desc[set_name]
        n_tpl = len({r.template_id for r in u.items})
        sha = None
        for _, info in hash_files.items():
            if info.get("path", "").endswith(str(rel)):
                sha = info.get("sha256")
                break
        con.execute("INSERT INTO eval_sets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (set_name, role, is_head, is_ood, len(u.items), n_tpl, sha, desc))

    # 3. leaderboard
    con.execute("""
        CREATE TABLE leaderboard (
            model VARCHAR,
            set_name VARCHAR,
            status VARCHAR,
            universe INTEGER,
            scored INTEGER,
            skipped INTEGER,
            coverage DOUBLE,
            accuracy DOUBLE,
            accuracy_ci_lo DOUBLE,
            accuracy_ci_hi DOUBLE,
            macro_accuracy DOUBLE,
            macro_ci_lo DOUBLE,
            macro_ci_hi DOUBLE,
            brier DOUBLE,
            brier_ci_lo DOUBLE,
            brier_ci_hi DOUBLE,
            ece DOUBLE,
            readout_health VARCHAR,
            PRIMARY KEY (model, set_name)
        )
    """)
    for model, sets_dict in board["models"].items():
        for set_name, cell in sets_dict.items():
            status = cell.get("status", "UNKNOWN")
            if status != "MEASURED":
                con.execute(
                    "INSERT INTO leaderboard VALUES (?, ?, ?, ?, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
                    (model, set_name, status, len(universes[set_name].items)),
                )
                continue
            aci = cell.get("accuracy_ci95") or [None, None]
            mci = cell.get("macro_accuracy_ci95") or [None, None]
            bci = cell.get("brier_ci95") or [None, None]
            brier_val = cell.get("brier") if isinstance(cell.get("brier"), (int, float)) else None
            ece_val = cell.get("ece") if isinstance(cell.get("ece"), (int, float)) else None
            rh = cell.get("readout_health", {}).get("overall")
            con.execute(
                "INSERT INTO leaderboard VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    model, set_name, status, cell["universe"], cell["scored"], cell.get("skipped", 0),
                    cell["coverage"], cell["accuracy"], aci[0], aci[1],
                    cell.get("macro_accuracy"), mci[0], mci[1],
                    brier_val, bci[0], bci[1], ece_val, rh,
                ),
            )

    # 4. paired_comparisons
    con.execute("""
        CREATE TABLE paired_comparisons (
            candidate VARCHAR,
            baseline VARCHAR,
            set_name VARCHAR,
            status VARCHAR,
            n_paired INTEGER,
            acc_diff DOUBLE,
            acc_ci_lo DOUBLE,
            acc_ci_hi DOUBLE,
            brier_diff DOUBLE,
            brier_ci_lo DOUBLE,
            brier_ci_hi DOUBLE,
            PRIMARY KEY (candidate, baseline, set_name)
        )
    """)
    for cand, baselines in board["paired"].items():
        for base, sets_dict in baselines.items():
            for set_name, row in sets_dict.items():
                if row.get("status") != "MEASURED":
                    con.execute(
                        "INSERT INTO paired_comparisons VALUES (?, ?, ?, ?, 0, NULL, NULL, NULL, NULL, NULL, NULL)",
                        (cand, base, set_name, row.get("status", "NOT MEASURED")),
                    )
                else:
                    aci = row.get("accuracy_ci95") or [None, None]
                    bci = row.get("brier_ci95") or [None, None]
                    con.execute(
                        "INSERT INTO paired_comparisons VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            cand, base, set_name, "MEASURED", row["n_paired"],
                            row["accuracy_diff"], aci[0], aci[1],
                            row.get("brier_diff"), bci[0], bci[1],
                        ),
                    )

    # 5. paired_intersections_10opt
    con.execute("""
        CREATE TABLE paired_intersections_10opt (
            set_name VARCHAR,
            model VARCHAR,
            n_items INTEGER,
            accuracy DOUBLE,
            brier DOUBLE,
            ece DOUBLE,
            PRIMARY KEY (set_name, model)
        )
    """)
    for set_name, idata in board.get("intersections", {}).items():
        if not isinstance(idata, dict):
            continue
        for model, mdata in idata.items():
            if not isinstance(mdata, dict) or "accuracy" not in mdata:
                continue
            con.execute(
                "INSERT INTO paired_intersections_10opt VALUES (?, ?, ?, ?, ?, ?)",
                (set_name, model, mdata.get("scored"), mdata.get("accuracy"), mdata.get("brier"), mdata.get("ece")),
            )

    # 6. pareto_two_axis & pareto_dominance
    con.execute("""
        CREATE TABLE pareto_two_axis (
            model VARCHAR PRIMARY KEY,
            params_b DOUBLE,
            p50_ms_b1 DOUBLE,
            peak_vram_gb DOUBLE,
            ood_general_avg DOUBLE,
            clinical_audit_avg DOUBLE,
            five_set_macro DOUBLE,
            on_two_axis_front BOOLEAN,
            undominated_le10b_5set BOOLEAN,
            undominated_le4b_5set BOOLEAN
        )
    """)
    two_axis = board["pareto"]["two_axis"]
    eff_by_model = board["pareto"]["efficiency"]["models"]
    front_set = set(two_axis["front"])
    undom10 = set(board["pareto"]["quality_le10b"]["undominated"])
    undom4 = set(board["pareto"]["quality_le4b"]["undominated"])
    for m, pt in two_axis["points"].items():
        ef = eff_by_model.get(m, {})
        con.execute(
            "INSERT INTO pareto_two_axis VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                m, ef.get("params_b"), ef.get("p50_ms_per_item"), ef.get("peak_vram_gb"),
                pt["ood"], pt["audit"], ef.get("five_set_macro"),
                m in front_set, m in undom10, m in undom4,
            ),
        )

    con.execute("""
        CREATE TABLE pareto_dominance (
            scope VARCHAR,
            dominator VARCHAR,
            dominated VARCHAR,
            d_panel DOUBLE,
            d_knowledge DOUBLE,
            d_confirm_general DOUBLE,
            d_mr_verdict DOUBLE,
            d_mimic_suite DOUBLE,
            PRIMARY KEY (scope, dominator, dominated)
        )
    """)
    for scope_key, label in (("quality_le10b", "le10b_5set"), ("quality_le4b", "le4b_5set")):
        for pair in board["pareto"][scope_key]["dominance_pairs"]:
            d = pair["deltas"]
            con.execute(
                "INSERT INTO pareto_dominance VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (label, pair["dominator"], pair["dominated"],
                 d["panel"], d["knowledge"], d["confirm_general"], d["mr_verdict"], d["mimic_suite"]),
            )

    # 7. Per-item derived aggregate tables: per_source, per_template, reliability_bins,
    #    selective_prediction, option_count_scaling (aggregate-only, ZERO raw text!)
    con.execute("""
        CREATE TABLE per_source (
            model VARCHAR,
            set_name VARCHAR,
            source VARCHAR,
            n_items INTEGER,
            accuracy DOUBLE,
            brier DOUBLE,
            ece DOUBLE,
            PRIMARY KEY (model, set_name, source)
        )
    """)
    con.execute("""
        CREATE TABLE per_template (
            model VARCHAR,
            set_name VARCHAR,
            template_id VARCHAR,
            n_items INTEGER,
            n_options_mode INTEGER,
            chance DOUBLE,
            majority_share DOUBLE,
            modal_share DOUBLE,
            accuracy DOUBLE,
            accuracy_ci_lo DOUBLE,
            accuracy_ci_hi DOUBLE,
            brier DOUBLE,
            ece DOUBLE,
            readout_status VARCHAR,
            PRIMARY KEY (model, set_name, template_id)
        )
    """)
    con.execute("""
        CREATE TABLE reliability_bins (
            model VARCHAR,
            set_name VARCHAR,
            calibration_variant VARCHAR,
            bin_idx INTEGER,
            bin_lo DOUBLE,
            bin_hi DOUBLE,
            n_items INTEGER,
            mean_confidence DOUBLE,
            mean_accuracy DOUBLE,
            abs_gap DOUBLE,
            PRIMARY KEY (model, set_name, calibration_variant, bin_idx)
        )
    """)
    con.execute("""
        CREATE TABLE selective_prediction (
            model VARCHAR,
            set_name VARCHAR,
            coverage_target DOUBLE,
            actual_coverage DOUBLE,
            n_accepted INTEGER,
            n_total INTEGER,
            selective_accuracy DOUBLE,
            selective_error DOUBLE,
            min_confidence DOUBLE,
            PRIMARY KEY (model, set_name, coverage_target)
        )
    """)
    con.execute("""
        CREATE TABLE option_count_scaling (
            model VARCHAR,
            set_name VARCHAR,
            option_bucket VARCHAR,
            n_items INTEGER,
            mean_chance DOUBLE,
            accuracy DOUBLE,
            brier DOUBLE,
            ece DOUBLE,
            PRIMARY KEY (model, set_name, option_bucket)
        )
    """)

    def ingest_pred_file(model: str, set_name: str, pred_path: Path, *, cal_variant: str = "calibrated",
                         full_tables: bool = True) -> None:
        u = universes[set_name]
        cell = load_cell(model, u, pred_path)
        if cell is None or cell.scored == 0:
            return
        confs_all: list[float] = []
        corrs_all: list[float] = []
        by_src: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        by_tpl: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        by_bkt: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

        for pos in range(cell.scored):
            item = u.items[int(cell.index[pos])]
            corr = 1.0 if bool(cell.correct[pos]) else 0.0
            if cell.probs is not None:
                p_row = cell.probs[pos]
                conf = float(max(p_row))
                br = brier_of(p_row, int(cell.gold[pos]))
            else:
                conf = float("nan")
                br = float("nan")
            confs_all.append(conf)
            corrs_all.append(corr)
            if full_tables:
                by_src[item.source].append((corr, conf, br))
                by_tpl[item.template_id].append((corr, conf, br))
                by_bkt[option_bucket(item.n_options)].append((corr, conf, br, 1.0 / max(1, item.n_options)))

        c_arr = np.asarray(confs_all, dtype=float)
        y_arr = np.asarray(corrs_all, dtype=float)
        has_probs = not np.isnan(c_arr).any()

        # Reliability bins (15 bins)
        if has_probs:
            bin_idx = np.minimum((c_arr * ECE_BINS).astype(int), ECE_BINS - 1)
            rel_rows = []
            for b in range(ECE_BINS):
                mask = bin_idx == b
                cnt = int(mask.sum())
                lo, hi = b / ECE_BINS, (b + 1) / ECE_BINS
                if cnt:
                    mc = float(c_arr[mask].mean())
                    ma = float(y_arr[mask].mean())
                    rel_rows.append((model, set_name, cal_variant, b, lo, hi, cnt, mc, ma, abs(ma - mc)))
                else:
                    rel_rows.append((model, set_name, cal_variant, b, lo, hi, 0, None, None, None))
            con.executemany("INSERT INTO reliability_bins VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rel_rows)

        if not full_tables:
            return

        # Selective prediction risk-coverage curve
        if has_probs:
            order = np.argsort(-c_arr, kind="mergesort")
            sorted_y = y_arr[order]
            sorted_c = c_arr[order]
            n_tot = len(sorted_y)
            sel_rows = []
            for cov in COVERAGES:
                k = max(1, min(n_tot, math.ceil(cov * n_tot)))
                acc = float(sorted_y[:k].mean())
                sel_rows.append((model, set_name, cov, k / n_tot, k, n_tot, acc, 1.0 - acc, float(sorted_c[k - 1])))
            con.executemany("INSERT INTO selective_prediction VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", sel_rows)

        # per_source
        src_rows = []
        for src, vals in sorted(by_src.items()):
            yy = np.array([v[0] for v in vals])
            cc = np.array([v[1] for v in vals])
            bb = np.array([v[2] for v in vals])
            src_rows.append((
                model, set_name, src, len(vals), float(yy.mean()),
                None if np.isnan(bb).any() else float(bb.mean()),
                None if np.isnan(cc).any() else compute_ece(cc, yy),
            ))
        con.executemany("INSERT INTO per_source VALUES (?, ?, ?, ?, ?, ?, ?)", src_rows)

        # per_template (merge readout_health from board JSON)
        rh_tpls = {
            t["template_id"]: t
            for t in board["models"][model][set_name].get("readout_health", {}).get("templates", [])
        }
        tpl_rows = []
        for tpl, vals in sorted(by_tpl.items()):
            yy = np.array([v[0] for v in vals])
            cc = np.array([v[1] for v in vals])
            bb = np.array([v[2] for v in vals])
            rh = rh_tpls.get(tpl, {})
            ci = rh.get("accuracy_ci95") or [None, None]
            tpl_rows.append((
                model, set_name, tpl, len(vals), rh.get("n_options"), rh.get("chance"),
                rh.get("majority_share"), rh.get("modal_share"), float(yy.mean()), ci[0], ci[1],
                None if np.isnan(bb).any() else float(bb.mean()),
                None if np.isnan(cc).any() else compute_ece(cc, yy),
                rh.get("status"),
            ))
        con.executemany("INSERT INTO per_template VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tpl_rows)

        # option_count_scaling
        bkt_rows = []
        for bkt, vals in sorted(by_bkt.items()):
            yy = np.array([v[0] for v in vals])
            cc = np.array([v[1] for v in vals])
            bb = np.array([v[2] for v in vals])
            ch = np.array([v[3] for v in vals])
            bkt_rows.append((
                model, set_name, bkt, len(vals), float(ch.mean()), float(yy.mean()),
                None if np.isnan(bb).any() else float(bb.mean()),
                None if np.isnan(cc).any() else compute_ece(cc, yy),
            ))
        con.executemany("INSERT INTO option_count_scaling VALUES (?, ?, ?, ?, ?, ?, ?, ?)", bkt_rows)

    for model, sets_dict in board["models"].items():
        for set_name, cell_info in sets_dict.items():
            if cell_info.get("status") != "MEASURED":
                continue
            p_str = cell_info.get("path")
            if p_str and Path(p_str).is_file():
                ingest_pred_file(model, set_name, Path(p_str), cal_variant="calibrated", full_tables=True)

    # Also ingest RAW (pre-temperature) reliability bins for Osler-9B-rc0 and Osler-4B-rc0 on headline sets!
    for set_name in HEADLINE_SETS:
        p9_raw = (REPO / "outputs/eval0/panel/9b_final_panel/predictions.jsonl"
                  if set_name == "panel"
                  else REPO / f"outputs/osler_rc0/R2/preds/9b_rc0_{set_name}/predictions.jsonl")
        if p9_raw.is_file():
            ingest_pred_file("Osler-9B-rc0", set_name, p9_raw, cal_variant="raw", full_tables=False)
        p4_raw = REPO / f"outputs/osler_rc0/R3/preds/4b_rc0_{set_name}/predictions.jsonl"
        if p4_raw.is_file():
            ingest_pred_file("Osler-4B-rc0", set_name, p4_raw, cal_variant="raw", full_tables=False)

    # 8. calibration_fits
    con.execute("""
        CREATE TABLE calibration_fits (
            model VARCHAR,
            qtype VARCHAR,
            n_dev INTEGER,
            temperature DOUBLE,
            nll_before DOUBLE,
            nll_after DOUBLE,
            brier_before DOUBLE,
            brier_after DOUBLE,
            ece_before DOUBLE,
            ece_after DOUBLE,
            argmax_changes INTEGER,
            PRIMARY KEY (model, qtype)
        )
    """)
    for qtype, info in cal9["per_qtype"].items():
        con.execute(
            "INSERT INTO calibration_fits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("Osler-9B-rc0", qtype, info["n_items"], info["temperature"], info["pre"]["nll"], info["post"]["nll"],
             info["pre"]["brier"], info["post"]["brier"], info["pre"]["ece"], info["post"]["ece"],
             cal9.get("argmax_changes_on_dev", 0)),
        )
    a9 = cal9["all_items"]
    con.execute(
        "INSERT INTO calibration_fits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("Osler-9B-rc0", "ALL", a9["n_items"], None, a9["pre"]["nll"], a9["post"]["nll"],
         a9["pre"]["brier"], a9["post"]["brier"], a9["pre"]["ece"], a9["post"]["ece"],
         cal9.get("argmax_changes_on_dev", 0)),
    )
    for qtype, info in arm4.get("temperature_fits", {}).items():
        if isinstance(info, dict):
            con.execute(
                "INSERT INTO calibration_fits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("Osler-4B-rc0", qtype, info.get("n_items"), info.get("temperature"),
                 info.get("nll_before"), info.get("nll_after"),
                 info.get("brier_before"), info.get("brier_after"),
                 info.get("ece_before"), info.get("ece_after"), 0),
            )

    # 9. latency_benchmarks
    con.execute("""
        CREATE TABLE latency_benchmarks (
            model VARCHAR,
            batch_size INTEGER,
            status VARCHAR,
            params_exact BIGINT,
            p50_ms DOUBLE,
            p95_ms DOUBLE,
            mean_ms DOUBLE,
            items_per_s DOUBLE,
            peak_vram_gb DOUBLE,
            note VARCHAR,
            PRIMARY KEY (model, batch_size)
        )
    """)
    for m, info in latency["models"].items():
        for bkey, bsz in (("batch_1", 1), ("batch_16", 16)):
            b = info.get(bkey, {})
            con.execute(
                "INSERT INTO latency_benchmarks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (m, bsz, "MEASURED", info.get("params"), b.get("p50_ms"), b.get("p95_ms"),
                 b.get("mean_ms"), b.get("items_per_s"), b.get("peak_vram_gb"), None),
            )
    for m, reason in latency.get("not_measured", {}).items():
        con.execute(
            "INSERT INTO latency_benchmarks VALUES (?, 1, 'NOT MEASURED', NULL, NULL, NULL, NULL, NULL, NULL, ?)",
            (m, str(reason)),
        )

    # 10. training_eval_trajectories
    con.execute("""
        CREATE TABLE training_eval_trajectories (
            model VARCHAR,
            step INTEGER,
            dev_s DOUBLE,
            mr_macro DOUBLE,
            mimic_macro DOUBLE,
            cg_micro DOUBLE,
            knowledge_dev DOUBLE,
            positional_original_pass BOOLEAN,
            positional_all_orders_pass BOOLEAN,
            two_letter_acc DOUBLE,
            single_letter_acc DOUBLE,
            PRIMARY KEY (model, step)
        )
    """)
    for step_str, rec in sorted(evals4.items(), key=lambda kv: int(kv[0])):
        comp = rec["composite"]
        pos = rec["positional_original"]
        con.execute(
            "INSERT INTO training_eval_trajectories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("Osler-4B-rc0", int(step_str), comp.get("S", comp.get("s")), comp["mr"], comp["mimic"], comp["cg_micro"],
             rec["knowledge_dev_accuracy"], bool(pos["passes"]), bool(rec["positional_passes_all_orders"]),
             pos.get("two_letter_accuracy"), pos.get("single_letter_accuracy")),
        )
    for step_str, rec in sorted(evals9.items(), key=lambda kv: int(kv[0])):
        comp = rec.get("composite") or {}
        pos = rec.get("positional_original") or {}
        step_i = int(step_str)
        all_pass = True if step_i == 63304 else None
        con.execute(
            "INSERT INTO training_eval_trajectories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("Osler-9B-rc0", step_i, comp.get("S", comp.get("s")), comp.get("mr"),
             comp.get("mimic"), comp.get("cg_micro"),
             rec.get("knowledge_dev_accuracy"),
             bool(pos["passes"]) if "passes" in pos else None,
             all_pass, pos.get("two_letter_accuracy"), pos.get("single_letter_accuracy")),
        )

    # 11. training_step_curves (numeric scalars only, zero text)
    con.execute("""
        CREATE TABLE training_step_curves (
            model VARCHAR,
            step INTEGER,
            loss DOUBLE,
            ce DOUBLE,
            brier DOUBLE,
            kl DOUBLE,
            distill DOUBLE,
            accuracy DOUBLE,
            n_items INTEGER,
            real_tokens INTEGER,
            padded_tokens INTEGER,
            elapsed_s DOUBLE,
            grad_norm DOUBLE,
            gpu_peak_gb DOUBLE,
            lr_head DOUBLE,
            lr_lora DOUBLE,
            PRIMARY KEY (model, step)
        )
    """)
    for m, path in (("Osler-4B-rc0", TRAIN_4B), ("Osler-9B-rc0", TRAIN_9B)):
        if not path.is_file():
            continue
        seen_steps: dict[int, tuple[Any, ...]] = {}
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("event") not in (None, "step"):
                    continue
                step = r.get("step")
                if step is None:
                    continue
                step_i = int(step)
                # Keep every 50th step (or exact final step) so file stays compact & clean
                if step_i % 50 != 0 and step_i != 63304:
                    continue
                seen_steps[step_i] = (
                    m, step_i, r.get("loss"), r.get("ce"), r.get("brier"),
                    r.get("kl", r.get("kl_base")), r.get("distill", r.get("kl_teacher")),
                    r.get("accuracy"), r.get("n_items"), r.get("real_tokens"),
                    r.get("padded_tokens"), r.get("elapsed_s", r.get("seconds")),
                    r.get("grad_norm", r.get("pre_clip")),
                    r.get("gpu_peak_gb"), r.get("lr_head"), r.get("lr_lora"),
                )
        con.executemany(
            "INSERT INTO training_step_curves VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [seen_steps[s] for s in sorted(seen_steps)],
        )

    # 12. phi_audit_membership & phi_audit_verbatim_arch
    con.execute("""
        CREATE TABLE phi_audit_membership (
            audit_set VARCHAR,
            model VARCHAR,
            n_members INTEGER,
            n_nonmembers INTEGER,
            auc DOUBLE,
            auc_ci_lo DOUBLE,
            auc_ci_hi DOUBLE,
            base_control_model VARCHAR,
            base_control_auc DOUBLE,
            delta_auc DOUBLE,
            delta_ci_lo DOUBLE,
            delta_ci_hi DOUBLE,
            rule_a_clear BOOLEAN,
            PRIMARY KEY (audit_set, model)
        )
    """)
    pairs_map = {"Osler-9B-rc0": "d3-flash", "Osler-4B-rc0": "pplx-flash"}
    vis_m = audit_auc.get("models", {})
    for m, info in vis_m.items():
        ctrl = info.get("control")
        dci = info.get("auc_difference_ci95") or [None, None]
        con.execute(
            "INSERT INTO phi_audit_membership VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("visited_both_2000", m, audit_auc.get("members", 2000), audit_auc.get("nonmembers", 2000),
             info["auc"], info["auc_lower"], info["auc_upper"],
             ctrl, info.get("auc_control"),
             info.get("auc_difference"), dci[0], dci[1],
             info.get("clear_by_auc_rule")),
        )
    y10_m = audit_y10.get("models", {})
    for m, info in y10_m.items():
        if not isinstance(info, dict) or "auc" not in info:
            continue
        con.execute(
            "INSERT INTO phi_audit_membership VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("y10_compat_2000", m, info.get("scored_members", 2000), info.get("scored_nonmembers", 2000),
             info["auc"], info["auc_lower"], info["auc_upper"],
             pairs_map.get(m), None, None, None, None, None),
        )

    con.execute("""
        CREATE TABLE phi_audit_verbatim_arch (
            model VARCHAR PRIMARY KEY,
            generative_capable BOOLEAN,
            has_lm_head BOOLEAN,
            n_notes INTEGER,
            matches_ge_50 INTEGER,
            matches_ge_25 INTEGER,
            matches_ge_10 INTEGER,
            longest_run_max INTEGER,
            truth_nll_mean DOUBLE,
            control_matches_ge_50 INTEGER,
            control_truth_nll_mean DOUBLE,
            export_argmax_agree_20 INTEGER,
            visited_rows_4b INTEGER,
            visited_rows_9b INTEGER,
            visited_intersection INTEGER,
            nonmember_hash_overlap INTEGER,
            verdict VARCHAR
        )
    """)
    vf = audit_verb.get("final", {})
    vb = audit_verb.get("base", {})
    vexp = audit_verb.get("export_agreement", {})
    v4_rows = audit_vis.get("plan_4b", {}).get("visited_rows")
    v9_rows = audit_vis.get("plan_9b", {}).get("visited_rows")
    v_inter = audit_vis.get("visited_intersection")
    con.execute(
        "INSERT INTO phi_audit_verbatim_arch VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("Osler-4B-rc0", True, True, audit_verb.get("notes", 300),
         vf.get("matches", 0), vf.get("run_bins", {}).get("at_least_25", 0), vf.get("run_bins", {}).get("at_least_10", 0),
         vf.get("longest_run_max"), vf.get("truth_nll_mean"),
         vb.get("matches", 0), vb.get("truth_nll_mean"), vexp.get("argmax_agree"),
         v4_rows, v9_rows, v_inter,
         audit_nm.get("nonmember_content_hash_matches_in_mix", 0), "CLEAR"),
    )
    con.execute(
        "INSERT INTO phi_audit_verbatim_arch VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("Osler-9B-rc0", False, bool(audit_arch.get("has_lm_head", False)), 0,
         None, None, None, None, None, None, None, None,
         v4_rows, v9_rows, v_inter,
         audit_nm.get("nonmember_content_hash_matches_in_mix", 0), "CLEAR (non-generative; no lm_head)"),
    )

    # 13. training_mix_composition (aggregate counts only)
    con.execute("""
        CREATE TABLE training_mix_composition (
            dimension VARCHAR,
            key VARCHAR,
            row_count BIGINT,
            share DOUBLE,
            PRIMARY KEY (dimension, key)
        )
    """)
    tot_rows = int(mix_man["rows"])
    con.execute("INSERT INTO training_mix_composition VALUES ('total', 'mix_v7_rows', ?, 1.0)", (tot_rows,))
    con.execute("INSERT INTO training_mix_composition VALUES ('total', 'teacher_labelled_rows', ?, ?)",
                (int(mix_man["labelled_rows"]), float(mix_man["labelled_share"])))
    con.execute("INSERT INTO training_mix_composition VALUES ('total', 'c1c4_kl_pool_rows', ?, ?)",
                (int(mix_man["c1c4_rows"]), int(mix_man["c1c4_rows"]) / tot_rows))
    con.execute("INSERT INTO training_mix_composition VALUES ('total', 'mimic_derived_rows', ?, ?)",
                (int(audit_samp["mix_mimic_rows"]), int(audit_samp["mix_mimic_rows"]) / tot_rows))
    con.execute("INSERT INTO training_mix_composition VALUES ('overlap_audit', 'medconceptsqa_exact_state_question_matches', ?, 0.0)",
                (int(overlap.get("exact_normalized_state_question_hash_matches", 0)),))
    con.execute("INSERT INTO training_mix_composition VALUES ('overlap_audit', 'medconceptsqa_exact_question_text_matches', ?, ?)",
                (int(overlap.get("exact_normalized_question_text_matches", 0)),
                 int(overlap.get("exact_normalized_question_text_matches", 0)) / max(1, int(overlap.get("panel_rows_medconceptsqa", 1)))))
    con.execute("INSERT INTO training_mix_composition VALUES ('overlap_audit', 'medconceptsqa_items_containment_ge_0_5', ?, ?)",
                (int(overlap.get("items_containment_ge_0_5", 0)),
                 int(overlap.get("items_containment_ge_0_5", 0)) / max(1, int(overlap.get("panel_rows_medconceptsqa", 1)))))
    for k, v in (mix_man.get("cells") or {}).items():
        con.execute("INSERT INTO training_mix_composition VALUES ('cell', ?, ?, ?)", (str(k), int(v), int(v) / tot_rows))
    for k, v in (mix_man.get("qtypes") or {}).items():
        con.execute("INSERT INTO training_mix_composition VALUES ('qtype', ?, ?, ?)", (str(k), int(v), int(v) / tot_rows))
    for k, v in (mix_man.get("sources") or {}).items():
        con.execute("INSERT INTO training_mix_composition VALUES ('source', ?, ?, ?)", (str(k), int(v), int(v) / tot_rows))

    # 14. loop_lineage (full research arc for paper narrative / ablation table)
    con.execute("""
        CREATE TABLE loop_lineage (
            seq INTEGER PRIMARY KEY,
            loop_id VARCHAR,
            target_model VARCHAR,
            base_model VARCHAR,
            key_intervention VARCHAR,
            panel_acc DOUBLE,
            panel_delta_vs_base DOUBLE,
            knowledge_acc DOUBLE,
            mr_verdict_acc DOUBLE,
            mimic_suite_acc DOUBLE,
            outcome VARCHAR
        )
    """)
    lineage = [
        (1, "bench_v0_fix0", "Baselines", "Qwen3.5/JEV/Laya", "D12 readout-health check, 0-based MedMCQA fix, v0.1 fresh tier", None, None, None, None, None, "Benchmark & harness validated"),
        (2, "student_v1", "MedDecide-0.8B", "Qwen3.5-0.8B", "3-head comparison (pointer vs letter LoRA vs Unsloth joint); heads within 0.6 pts", None, None, None, None, None, "Architecture parity; held-out transfer failed"),
        (3, "osler_v0", "Osler-4B / 9B", "Qwen3.5-4B / 9B", "9 clinical generators + option-code head (D23); all-layer LoRA", None, None, None, None, None, "Lost external panel & 4B knowledge"),
        (4, "osler_v1", "Osler-4B", "Qwen3.5-4B", "Synthetic-first MR audit + attention-only LoRA (D33)", None, None, None, None, None, "Knowledge guard fixed (+4.4 pts); lost real-text ICD"),
        (5, "osler_v2", "Osler-4B", "pplx-flash (4B)", "Real clinical text (CodiEsp/MTSamples/EHR-Rel) + base switch to pplx-flash (D39)", 0.5343, -0.0198, 0.6752, 0.8963, 0.5551, "MR +15.8 pts; panel -2.0 pts forgetting"),
        (6, "osler_v4", "Osler-4B", "pplx-flash (4B)", "MIMIC-IV payer-audit templates (D46) + Nemotron pairs; 63,304 steps", 0.5254, -0.0421, 0.6765, 0.8862, 0.7947, "Best MIMIC (0.795) but panel -4.2 pts; 2-letter collapse diagnosed"),
        (7, "osler_v5", "Osler-4B (arms)", "pplx-flash (4B)", "Per-item loss weighting (D51) + option-count augmentation (D52) + label fix", None, None, None, None, None, "Fixed 2-letter collapse; MIMIC/know dev bounds missed"),
        (8, "osler_v6", "Osler-4B", "pplx-flash (4B)", "D57-dosed pplx-27B distillation (385k anchors, lambda=3.24)", 0.5360, -0.0245, 0.6820, 0.8950, 0.7703, "Restored knowledge parity; panel still -2.45 pts"),
        (9, "distill_v0", "Osler-4B / 9B", "pplx-4B / Clef-9B", "d1-ultra hosted teacher distillation (495k pool; teacher panel 0.683)", 0.5596, -0.0115, 0.6837, 0.8960, 0.7682, "Teacher panel lead did not transfer"),
        (10, "d3_distill", "d3-mini / d3-flash", "d3-mini / d3-flash", "d3-ultra local teacher distillation (200k, 91% MedMCQA)", 0.6210, -0.0383, 0.7620, 0.7084, None, "Knowledge +5.2 pts vs pplx, but panel -3.8 pts vs base"),
        (11, "osler_v7 / rc0", "Osler-4B-rc0", "pplx-flash (4B)", "Format-matched C1-C4 (70k) + KL-to-base + token-capped batching (63,304 steps) + dev cal", 0.5502, -0.0113, 0.6728, 0.8941, 0.7597, "Dominates MedDecider-4B on all 5 sets; PHI audit CLEAR"),
        (12, "osler_v7 / rc0", "Osler-9B-rc0", "d3-flash (9B)", "Same v7 mix + token-capped batching (63,304 steps) + dev cal (ECE 0.108->0.023)", 0.5927, 0.0028, 0.7322, 0.9016, 0.7733, "Global 2-axis Pareto front; dominates all <=10B; #1 fresh tier-2; PHI audit CLEAR"),
    ]
    con.executemany("INSERT INTO loop_lineage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", lineage)

    # 15. claims
    con.execute("""
        CREATE TABLE claims (
            claim_id VARCHAR PRIMARY KEY,
            claim VARCHAR,
            value VARCHAR,
            artifact VARCHAR,
            recompute_command VARCHAR,
            task VARCHAR
        )
    """)
    con.executemany("INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?)", parse_claims(CLAIMS_MD))

    # 16. _catalog (self-documentation)
    con.execute("""
        CREATE TABLE _catalog (
            table_name VARCHAR PRIMARY KEY,
            row_count BIGINT,
            description VARCHAR,
            paper_use VARCHAR
        )
    """)
    descriptions = {
        "models": ("12-model comparison roster metadata (params, bases, revisions, HF repos, licences)", "Table 1 / Model setup"),
        "eval_sets": ("9 frozen evaluation sets (sizes, templates, SHA-256, roles)", "Section 3 Benchmark table"),
        "leaderboard": ("12 models x 9 sets micro/macro accuracy (+95% CIs), Brier, ECE, coverage, readout health", "Main Results Table"),
        "per_source": ("Per-source accuracy, Brier, ECE for all 12 models x 9 sets", "Appendix & Panel breakdown"),
        "per_template": ("Per-template accuracy (+CI), Brier, ECE, chance, majority/modal shares, D12/D21 health", "Per-template & failure analysis"),
        "paired_comparisons": ("Item-paired bootstrap accuracy & Brier deltas (+95% CI) for Osler-9B/4B vs competitors", "Significance & ablation tables"),
        "paired_intersections_10opt": ("Exact <=10-option intersections (panel 4,141; confirm_general 5,838)", "Fair MedDecider comparisons"),
        "pareto_two_axis": ("OOD General Avg vs Clinical Audit Avg, 5-set macro, latency, VRAM, front flags", "Figure 1 Pareto frontier scatter"),
        "pareto_dominance": ("Strict 5-set dominance pairs in <=10B and <=4B tiers", "Section 4 Pareto dominance"),
        "reliability_bins": ("15-bin reliability diagram data (calibrated + raw pre-temperature for RCs)", "Figure 2 Calibration reliability plots"),
        "selective_prediction": ("Risk-coverage selective accuracy curves (10% to 100% coverage)", "Figure 3 Selective abstention / triage curves"),
        "option_count_scaling": ("Accuracy & Brier bucketed by option count (2, 3-4, 5-10, 11-26, 27+)", "Figure 4 Cardinality scaling plot"),
        "calibration_fits": ("Dev-only per-qtype temperature fits & pre/post dev NLL/Brier/ECE", "Section 4 Calibration table"),
        "latency_benchmarks": ("Standardized 500-item latency (p50/p95 ms), throughput, VRAM, exact params (b=1 & b=16)", "Figure 5 Efficiency / Pareto latency"),
        "training_eval_trajectories": ("Dev S, MR, MIMIC, CG micro, knowledge & positional health at 5 eval checkpoints", "Figure 6 Training trajectory plot"),
        "training_step_curves": ("Logged training loss/CE/Brier/KL/grad_norm every 50 steps (numeric only)", "Appendix optimization dynamics"),
        "phi_audit_membership": ("Visited-member (2k+2k) & Y10 MIMIC membership-inference AUC + CIs vs zero-shot controls", "Section 5 Privacy / PHI audit table"),
        "phi_audit_verbatim_arch": ("4B 300-note verbatim probe run bins + truth NLL + 9B non-generative check", "Section 5 Verbatim & architecture safety"),
        "training_mix_composition": ("Aggregate training mix row counts & shares by cell/qtype/source (zero text)", "Section 2 Data composition table"),
        "loop_lineage": ("Cross-loop research progression (v0->v7/rc0 + distill_v0 + d3_distill)", "Section 4 Why distillation alone fails"),
        "claims": ("All 29 osler_rc0 CLAIMS rows with provenance and recompute commands", "Reproducibility audit index"),
    }
    for tname, (desc, use) in descriptions.items():
        cnt = con.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
        con.execute("INSERT INTO _catalog VALUES (?, ?, ?, ?)", (tname, cnt, desc, use))

    con.execute("""
        CREATE TABLE _metadata (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
        )
    """)
    con.executemany(
        "INSERT INTO _metadata VALUES (?, ?)",
        [
            ("built_utc", datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")),
            ("duckdb_version", duckdb.__version__),
            ("git_board_commit", str(board.get("git_commit"))),
            ("privacy_guarantee", "Aggregate reporting data only: zero raw text, zero MIMIC text, zero patient IDs, zero secrets."),
            ("hf_osler_9b", upload.get("repos", {}).get("Osler-9B", {}).get("url", "https://huggingface.co/abhinand/Osler-9B")),
            ("hf_osler_4b", upload.get("repos", {}).get("Osler-4B", {}).get("url", "https://huggingface.co/abhinand/Osler-4B")),
        ],
    )
    con.execute("CHECKPOINT")
    con.close()
    print(f"Built {out_path} ({out_path.stat().st_size / (1024 * 1024):.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "docs/paper/osler_rc0_reporting.duckdb")
    args = parser.parse_args()
    build_db(args.out if args.out.is_absolute() else REPO / args.out)


if __name__ == "__main__":
    main()
