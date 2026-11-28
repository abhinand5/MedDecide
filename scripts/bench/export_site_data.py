#!/usr/bin/env python
"""Export strictly aggregate reporting JSON for the MedDecide landing page & HF Leaderboard Space.

Reads `docs/paper/osler_rc0_reporting.duckdb` (plus any verified community submission files in
`site/submissions/*.json`) and writes `site/data/leaderboard.json`.

Privacy guarantee: exports aggregate numbers ONLY — zero raw benchmark text, zero MIMIC text,
zero patient IDs, zero internal loop metadata.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO / "docs/paper/osler_rc0_reporting.duckdb"
DEFAULT_SUBMISSIONS = REPO / "site/submissions"
DEFAULT_OUT = REPO / "site/data/leaderboard.json"

HEADLINE_SETS = ("panel", "knowledge", "confirm_general", "mr_verdict", "mimic_suite")
ADDITIONAL_SETS = ("mednli_test", "pairs_test", "mimic_heldout", "mr_heldout_test")

# Real-world synthetic clinical vignettes (measured live on standalone merged Osler-9B & Osler-4B)
CLINICAL_EXAMPLES: list[dict[str, Any]] = [
    {
        "id": "negation_pe",
        "tag": "Negation & Ruled-Out Dx",
        "title": "Chest Pain — Ruled-Out PE vs. GERD",
        "state": (
            "54yo F presented to ED with substernal burning chest discomfort after dinner. "
            "Given recent 6-hour flight, CTA chest was obtained and showed NO pulmonary embolism "
            "and no aortic dissection. High-sensitivity troponin T was <6 ng/L at 0h and 3h; ECG normal sinus rhythm. "
            "Symptoms resolved completely after viscous lidocaine and IV pantoprazole. Discharged on daily omeprazole."
        ),
        "questions": [
            {
                "id": "pe_confirmed",
                "type": "noul",
                "prompt": "Does this note confirm a diagnosis of pulmonary embolism?",
                "expected": "No",
                "osler_9b": {"yes": 0.0426, "no": 0.9574},
                "osler_4b": {"yes": 0.0272, "no": 0.9728},
            },
            {
                "id": "primary_dx",
                "type": "choice",
                "prompt": "Most likely final diagnosis explaining this visit?",
                "expected": "GERD / Esophagitis",
                "options": [
                    {"key": "gerd", "label": "GERD / Esophagitis", "p9b": 0.993, "p4b": 0.991},
                    {"key": "pneumothorax", "label": "Spontaneous pneumothorax", "p9b": 0.003, "p4b": 0.003},
                    {"key": "nstemi", "label": "NSTEMI", "p9b": 0.002, "p4b": 0.004},
                    {"key": "pe", "label": "Acute pulmonary embolism", "p9b": 0.002, "p4b": 0.002},
                ],
            },
        ],
    },
    {
        "id": "ehr_hfref",
        "tag": "Messy EHR Distractor",
        "title": "Cardiac Asthma vs. Decompensated HFrEF",
        "state": (
            "71yo M with childhood asthma (no inhaler use in 15 yrs), HTN, and ischemic cardiomyopathy (EF 25%) "
            "admitted for 4 days of worsening orthopnea, 3-pillow sleep, and 4 kg weight gain. "
            "Exam: JVP 14 cm, bibasilar crackles, 2+ pitting edema to knees; mild expiratory wheeze ('cardiac asthma'). "
            "Two albuterol nebulizers in ED gave no relief. BNP 2,650 pg/mL. CXR: vascular congestion and Kerley B lines."
        ),
        "questions": [
            {
                "id": "cause",
                "type": "choice",
                "prompt": "Primary driver of this patient's acute dyspnea?",
                "expected": "Acute decompensated HFrEF",
                "options": [
                    {"key": "hfref_exacerbation", "label": "Acute decompensated HFrEF", "p9b": 0.997, "p4b": 0.992},
                    {"key": "asthma", "label": "Acute asthma exacerbation", "p9b": 0.001, "p4b": 0.003},
                    {"key": "copd", "label": "COPD exacerbation", "p9b": 0.001, "p4b": 0.002},
                    {"key": "pneumonia", "label": "Bacterial pneumonia", "p9b": 0.000, "p4b": 0.002},
                ],
            },
            {
                "id": "diuresis_needed",
                "type": "noul",
                "prompt": "Is IV loop diuretic therapy clinically indicated?",
                "expected": "Yes",
                "osler_9b": {"yes": 0.9558, "no": 0.0442},
                "osler_4b": {"yes": 0.9527, "no": 0.0473},
            },
        ],
    },
    {
        "id": "metabolic_mala",
        "tag": "Metabolic Differential",
        "title": "Metformin-Associated Lactic Acidosis vs. DKA",
        "state": (
            "66yo F with T2DM on metformin 1000 mg BID and CKD3b had 3 days of gastroenteritis with poor PO intake. "
            "Labs: Na 138, K 5.4, Cl 98, HCO3 10, BUN 58, Cr 4.1 (baseline 1.5), glucose 114 mg/dL, "
            "beta-hydroxybutyrate 0.3 mmol/L (normal), venous pH 7.16, serum lactate 10.2 mmol/L. Urine ketones negative."
        ),
        "questions": [
            {
                "id": "syndrome",
                "type": "choice",
                "prompt": "Which metabolic emergency best explains these labs?",
                "expected": "Metformin-associated lactic acidosis (MALA)",
                "options": [
                    {"key": "mala", "label": "Metformin-associated lactic acidosis (MALA)", "p9b": 0.985, "p4b": 0.991},
                    {"key": "salicylate", "label": "Salicylate toxicity", "p9b": 0.008, "p4b": 0.002},
                    {"key": "hhs", "label": "Hyperosmolar hyperglycemic state (HHS)", "p9b": 0.005, "p4b": 0.001},
                    {"key": "dka", "label": "Diabetic ketoacidosis (DKA)", "p9b": 0.003, "p4b": 0.006},
                ],
            },
            {
                "id": "dka",
                "type": "noul",
                "prompt": "Do the laboratory values support diabetic ketoacidosis?",
                "expected": "No",
                "osler_9b": {"yes": 0.0855, "no": 0.9145},
                "osler_4b": {"yes": 0.0888, "no": 0.9112},
            },
        ],
    },
    {
        "id": "abstention_nodule",
        "tag": "Honest Abstention",
        "title": "Unbiopsied Pulmonary Nodule — Refusing to Guess Stage",
        "state": (
            "59yo M with 35 pack-year smoking history had a screening low-dose chest CT showing a 1.8 cm spiculated "
            "right upper lobe nodule. Pulmonology referral placed for PET-CT and biopsy. No pathology or mediastinal "
            "lymph node sampling has been performed yet."
        ),
        "questions": [
            {
                "id": "staging",
                "type": "choice",
                "prompt": "What is the confirmed TNM stage of lung cancer based on this note?",
                "expected": "Insufficient information",
                "options": [
                    {"key": "insufficient_info", "label": "Insufficient information (no tissue/staging yet)", "p9b": 0.968, "p4b": 0.990},
                    {"key": "stage_i", "label": "Confirmed Stage I NSCLC", "p9b": 0.026, "p4b": 0.006},
                    {"key": "stage_iii", "label": "Confirmed Stage III NSCLC", "p9b": 0.004, "p4b": 0.002},
                    {"key": "stage_iv", "label": "Confirmed Stage IV metastatic NSCLC", "p9b": 0.002, "p4b": 0.002},
                ],
            },
        ],
    },
    {
        "id": "minimal_pair_sepsis",
        "tag": "Minimal-Pair Grounding",
        "title": "Sepsis-3 Septic Shock — Single-Sentence Flip",
        "state": (
            "Case A (Positive): 64yo F with E. coli bacteremia. Despite 30 mL/kg IV crystalloid, MAP remains 56 mmHg "
            "requiring norepinephrine at 12 mcg/min. Repeat lactate is 4.6 mmol/L.\n"
            "Case B (Negative): Same patient — following 30 mL/kg crystalloid, MAP improved to 78 mmHg with no "
            "vasopressors required. Repeat lactate normalized to 1.2 mmol/L."
        ),
        "questions": [
            {
                "id": "sepsis_pos",
                "type": "noul",
                "prompt": "Case A — Meets Sepsis-3 septic shock criteria (vasopressor + lactate > 2)?",
                "expected": "Yes",
                "osler_9b": {"yes": 0.9732, "no": 0.0268},
                "osler_4b": {"yes": 0.9731, "no": 0.0269},
            },
            {
                "id": "sepsis_neg",
                "type": "noul",
                "prompt": "Case B — Meets Sepsis-3 septic shock criteria (vasopressor + lactate > 2)?",
                "expected": "No",
                "osler_9b": {"yes": 0.0415, "no": 0.9585},
                "osler_4b": {"yes": 0.0369, "no": 0.9631},
            },
        ],
    },
    {
        "id": "trial_exclusion",
        "tag": "Trial Eligibility",
        "title": "Multi-Clause Clinical Trial Exclusion Trap",
        "state": (
            "Trial Criteria: Adults >=18y with symptomatic HFrEF (LVEF <=40%) on stable GDMT. "
            "Exclusions: eGFR <30 mL/min/1.73m2, active malignancy receiving chemotherapy, or pregnancy.\n"
            "Patient: 69yo M with NYHA III HFrEF (LVEF 30%) on carvedilol and sacubitril-valsartan, "
            "currently undergoing FOLFOX chemotherapy for stage III colon cancer, eGFR 28 mL/min/1.73m2."
        ),
        "questions": [
            {
                "id": "eligible",
                "type": "noul",
                "prompt": "Is this patient eligible to enroll in the clinical trial?",
                "expected": "No",
                "osler_9b": {"yes": 0.0188, "no": 0.9812},
                "osler_4b": {"yes": 0.0242, "no": 0.9758},
            },
        ],
    },
]

DISPLAY_NAMES = {
    "Osler-9B-rc0": "Osler-9B",
    "Osler-4B-rc0": "Osler-4B",
}


def r4(v: float | None) -> float | None:
    return None if v is None else round(float(v), 4)


def export_payload(db_path: Path, submissions_dir: Path) -> dict[str, Any]:
    con = duckdb.connect(str(db_path), read_only=True)

    # 1. Models + Pareto metadata
    model_rows = con.execute(
        """
        SELECT
            m.model,
            m.params_b,
            m.size_tier,
            m.role,
            m.base_repo,
            m.hf_release_repo,
            m.readout_type,
            m.non_generative,
            m.license,
            p.p50_ms_b1,
            p.peak_vram_gb,
            p.ood_general_avg,
            p.clinical_audit_avg,
            p.five_set_macro,
            p.on_two_axis_front,
            p.undominated_le10b_5set,
            p.undominated_le4b_5set
        FROM models m
        JOIN pareto_two_axis p USING (model)
        ORDER BY p.five_set_macro DESC NULLS LAST, m.params_b DESC
        """
    ).fetchall()

    # 2. Per-set leaderboard cells
    lb_rows = con.execute(
        """
        SELECT
            model, set_name, status, universe, scored, coverage,
            accuracy, accuracy_ci_lo, accuracy_ci_hi,
            macro_accuracy, brier, ece, readout_health
        FROM leaderboard
        """
    ).fetchall()
    cells_by_model: dict[str, dict[str, Any]] = {}
    for row in lb_rows:
        m = row[0]
        cells_by_model.setdefault(m, {})[row[1]] = {
            "status": row[2],
            "n": row[3],
            "scored": row[4],
            "coverage": r4(row[5]),
            "accuracy": r4(row[6]),
            "ci_lo": r4(row[7]),
            "ci_hi": r4(row[8]),
            "macro": r4(row[9]),
            "brier": r4(row[10]),
            "ece": r4(row[11]),
            "readout_health": row[12],
        }

    # 3. Latency (batch 1 & 16)
    lat_rows = con.execute(
        """
        SELECT model, batch_size, status, p50_ms, p95_ms, items_per_s, peak_vram_gb
        FROM latency_benchmarks
        WHERE status = 'MEASURED'
        ORDER BY model, batch_size
        """
    ).fetchall()
    latency_by_model: dict[str, dict[str, Any]] = {}
    for m, bsz, _st, p50, p95, ips, vram in lat_rows:
        latency_by_model.setdefault(m, {})[f"b{bsz}"] = {
            "p50_ms": None if p50 is None else round(float(p50), 2),
            "p95_ms": None if p95 is None else round(float(p95), 2),
            "items_per_s": None if ips is None else round(float(ips), 2),
            "peak_vram_gb": None if vram is None else round(float(vram), 2),
        }

    models_out: list[dict[str, Any]] = []
    for r in model_rows:
        raw_name = r[0]
        disp = DISPLAY_NAMES.get(raw_name, raw_name)
        m_cells = cells_by_model.get(raw_name, {})
        has_all_5 = all(m_cells.get(s, {}).get("status") == "MEASURED" for s in HEADLINE_SETS)
        models_out.append(
            {
                "id": disp,
                "raw_id": raw_name,
                "is_osler": raw_name.startswith("Osler-"),
                "verification_tier": "verified_full" if has_all_5 else "verified_partial",
                "params_b": round(float(r[1]), 2),
                "size_tier": r[2],
                "role": r[3],
                "base_repo": r[4],
                "hf_repo": r[5],
                "hf_url": f"https://huggingface.co/{r[5]}" if r[5] else None,
                "readout_type": r[6],
                "non_generative": bool(r[7]),
                "license": r[8],
                "p50_ms_b1": round(float(r[9]), 2) if r[9] is not None else None,
                "peak_vram_gb": round(float(r[10]), 2) if r[10] is not None else None,
                "general_medical_avg": r4(r[11]),
                "ehr_docs_avg": r4(r[12]),
                "five_set_macro": r4(r[13]) if has_all_5 else None,
                "on_pareto_front": bool(r[14]),
                "undominated_le10b": bool(r[15]),
                "undominated_le4b": bool(r[16]),
                "latency": latency_by_model.get(raw_name, {}),
                "sets": m_cells,
            }
        )

    # 4. Merge any approved community submissions from site/submissions/*.json
    if submissions_dir.is_dir():
        for sf in sorted(submissions_dir.glob("*.json")):
            if sf.name.startswith("_") or sf.name == "schema.json":
                continue
            sub = json.loads(sf.read_text(encoding="utf-8"))
            models_out.append(sub)

    # 5. Per-source breakdown for panel & confirm_general (headline models)
    ps_rows = con.execute(
        """
        SELECT set_name, source, n_items, model, accuracy, brier, ece
        FROM per_source
        WHERE set_name IN ('panel', 'confirm_general', 'knowledge')
        ORDER BY set_name, source, accuracy DESC
        """
    ).fetchall()
    per_source: dict[str, dict[str, Any]] = {}
    for sname, src, n_items, m, acc, br, ece in ps_rows:
        per_source.setdefault(sname, {}).setdefault(src, {"n": n_items, "models": {}})[
            "models"
        ][DISPLAY_NAMES.get(m, m)] = {"accuracy": r4(acc), "brier": r4(br), "ece": r4(ece)}

    # 6. Calibration reliability curves (15-bin, panel / mr_verdict / confirm_general)
    rel_rows = con.execute(
        """
        SELECT set_name, model, calibration_variant, bin_idx, bin_lo, bin_hi, n_items, mean_confidence, mean_accuracy
        FROM reliability_bins
        WHERE set_name IN ('panel', 'mr_verdict', 'confirm_general')
          AND model IN ('Osler-9B-rc0', 'Osler-4B-rc0', 'd3-flash', 'pplx-flash', 'pplx-27B')
          AND n_items > 0
        ORDER BY set_name, model, calibration_variant, bin_idx
        """
    ).fetchall()
    reliability: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for sname, m, var, bidx, blo, bhi, cnt, mc, ma in rel_rows:
        key = f"{DISPLAY_NAMES.get(m, m)} ({var})" if var == "raw" else DISPLAY_NAMES.get(m, m)
        reliability.setdefault(sname, {}).setdefault(key, []).append(
            {
                "bin": bidx,
                "mid": round((blo + bhi) / 2.0, 3),
                "n": cnt,
                "conf": r4(mc),
                "acc": r4(ma),
            }
        )

    # 7. Selective prediction curves (mr_verdict, mimic_suite, panel)
    sel_rows = con.execute(
        """
        SELECT set_name, model, actual_coverage, selective_accuracy, selective_error, min_confidence
        FROM selective_prediction
        WHERE set_name IN ('mr_verdict', 'mimic_suite', 'panel')
          AND model IN ('Osler-9B-rc0', 'Osler-4B-rc0', 'd3-flash', 'pplx-flash', 'MedDecider-9B', 'pplx-27B')
        ORDER BY set_name, model, actual_coverage
        """
    ).fetchall()
    selective: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for sname, m, cov, sacc, serr, mconf in sel_rows:
        selective.setdefault(sname, {}).setdefault(DISPLAY_NAMES.get(m, m), []).append(
            {
                "coverage": r4(cov),
                "accuracy": r4(sacc),
                "error": r4(serr),
                "min_conf": r4(mconf),
            }
        )

    # 8. Option-count cardinality scaling
    opt_rows = con.execute(
        """
        SELECT set_name, option_bucket, model, n_items, accuracy, brier
        FROM option_count_scaling
        WHERE set_name IN ('panel', 'mr_verdict', 'mimic_suite')
          AND model IN ('Osler-9B-rc0', 'Osler-4B-rc0', 'd3-flash', 'pplx-flash', 'MedDecider-9B', 'pplx-27B')
        ORDER BY set_name, option_bucket, model
        """
    ).fetchall()
    option_scaling: dict[str, dict[str, dict[str, Any]]] = {}
    for sname, bucket, m, n_items, acc, br in opt_rows:
        option_scaling.setdefault(sname, {}).setdefault(bucket, {"n": n_items, "models": {}})[
            "models"
        ][DISPLAY_NAMES.get(m, m)] = {"accuracy": r4(acc), "brier": r4(br)}

    # 9. Eval sets summary
    es_rows = con.execute(
        """
        SELECT set_name, role, is_headline, is_ood, n_items, n_templates, substr(sha256, 1, 16), description
        FROM eval_sets
        ORDER BY is_headline DESC, set_name
        """
    ).fetchall()
    eval_sets = [
        {
            "id": r[0],
            "role": r[1],
            "is_headline": bool(r[2]),
            "is_public_tier": bool(r[3]) and r[0] in ("panel", "knowledge", "confirm_general"),
            "n_items": r[4],
            "n_templates": r[5],
            "sha256_prefix": r[6],
            "description": r[7],
        }
        for r in es_rows
    ]

    # 10. Privacy / PHI audit summary (strictly aggregate)
    phi_rows = con.execute(
        """
        SELECT model, auc, auc_ci_lo, auc_ci_hi, base_control_model, base_control_auc, delta_auc, rule_a_clear
        FROM phi_audit_membership
        WHERE audit_set = 'visited_both_2000' AND base_control_model IS NOT NULL
        """
    ).fetchall()
    phi_audit = [
        {
            "model": DISPLAY_NAMES.get(r[0], r[0]),
            "auc": r4(r[1]),
            "ci_lo": r4(r[2]),
            "ci_hi": r4(r[3]),
            "control": r[4],
            "control_auc": r4(r[5]),
            "delta_auc": r4(r[6]),
            "clear": bool(r[7]),
        }
        for r in phi_rows
    ]

    con.close()
    return {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "links": {
            "github": "https://github.com/abhinand5/MedDecide",
            "landing_page": "https://abhinand5.github.io/MedDecide",
            "hf_osler_9b": "https://huggingface.co/abhinand/Osler-9B",
            "hf_osler_4b": "https://huggingface.co/abhinand/Osler-4B",
            "hf_bench": "https://huggingface.co/datasets/abhinand/MedDecide-Bench",
            "hf_space": "https://huggingface.co/spaces/abhinand/MedDecide-Leaderboard",
            "hf_collection": "https://huggingface.co/collections/abhinand/osler-calibrated-medical-decision-models-6b0af4d7e2e7a5eed75e3619",
            "private_eval_issue": "https://github.com/abhinand5/MedDecide/issues/new?template=private_eval_request.yml",
        },
        "eval_sets": eval_sets,
        "models": models_out,
        "per_source": per_source,
        "reliability": reliability,
        "selective_prediction": selective,
        "option_scaling": option_scaling,
        "phi_audit": phi_audit,
        "clinical_examples": CLINICAL_EXAMPLES,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--submissions", type=Path, default=DEFAULT_SUBMISSIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = export_payload(args.db, args.submissions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KB, {len(payload['models'])} models)")


if __name__ == "__main__":
    main()
