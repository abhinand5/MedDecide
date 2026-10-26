"""osler_v5 Z0 self-audit (R6): re-derive the Z0 headline numbers from the raw artifacts in a fresh process, with plain numpy and json (no
meddecide code except the item-id conversion, which the prediction files depend on). Prints each figure and whether it matches the
recorded artifact. Counts and figures only.

Usage (CPU; fresh process):
    uv run --frozen python scripts/osler_v5/z0_rederive.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from meddecide.train.mix_items import mix_row_to_item

REPO = Path(__file__).resolve().parents[2]
Z0 = REPO / "outputs" / "osler_v5" / "Z0"
Y8 = REPO / "outputs" / "osler_v4" / "Y8"
MR = {"mr_v1", "mr_synthea", "mr_teacher", "mr_mtsamples", "mr_codiesp", "mr_pmc"}


def check(name: str, derived: float, recorded: float, tol: float = 1e-9) -> bool:
    ok = abs(derived - recorded) <= tol
    print(f"{name}: derived {derived:.10f}, recorded {recorded:.10f}, {'match' if ok else 'MISMATCH'}", flush=True)
    return ok


def main() -> None:
    results: list[bool] = []
    # 1. shares from the per-step arrays and the Y8 step log (the logged pre-clip norms), in plain numpy
    arrays = np.load(Z0 / "replay" / "steps.npz")
    n = arrays["n_items"].astype(np.float64)
    m = arrays["mimic_items"].astype(np.float64)
    log_steps = arrays["logged_step_index"]
    norms = arrays["logged_grad_norm"]
    applied = np.minimum(norms, 1.0)
    summary = json.loads((Z0 / "replay" / "summary.json").read_text(encoding="utf-8"))
    results.append(check("per_step share (clip-aware, logged steps)", float(np.sum(m[log_steps] / n[log_steps] * applied) / np.sum(applied)),
                         summary["shares"]["per_step_clip_aware_logged"]))
    results.append(check("per_item share (clip-aware, logged steps)",
                         float(np.sum(m[log_steps] / 8 * applied) / np.sum(n[log_steps] / 8 * applied)),
                         summary["shares"]["per_item_clip_aware_logged"]))
    results.append(check("per_item share (full run, unclipped)", float(m.sum() / n.sum()), summary["shares"]["per_item_full_run_unclipped"]))
    results.append(check("per_step share (full run, unclipped)", float(np.mean(m / n)), summary["shares"]["per_step_full_run_unclipped"]))
    results.append(check("realised items", float(n.sum()), float(summary["realised_items"])))
    # the step log in plain json: the logged norms and template counts must equal the arrays
    logged_norms, count_bad = [], 0
    with (Y8 / "logs" / "train.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("event") == "step":
                logged_norms.append(float(record["grad_norm"]))
                t = int(record["step"]) - 1
                ids = record["template_ids"]
                count_bad += int(len(ids) != int(n[t])) + int(sum(i.startswith("mimic_") for i in ids) != int(m[t]))
    results.append(check("logged grad norms equal the replay's (max abs diff)", float(np.max(np.abs(np.array(logged_norms) - norms))), 0.0))
    results.append(check("logged-step count mismatches", float(count_bad), 0.0))
    results.append(check("clipped share of logged steps", float(np.mean(np.array(logged_norms) > 1.0)), summary["shares"]["logged_clipped_share"]))

    # 2. the Y8 replay record against the replay arrays
    y8 = json.loads((Y8 / "replays" / "final_mix_replay.json").read_text(encoding="utf-8"))
    results.append(check("items differing from the Y8 record", float(np.count_nonzero(n != np.array(y8["sizes"]))), 0.0))
    results.append(check("real tokens differing from the Y8 record", float(np.count_nonzero(arrays["real_tokens"] != np.array(y8["real_tokens"]))), 0.0))

    # 3. the S composite from the prediction files, in plain python (ids through the item conversion)
    def ids_and_rows(item_path: Path, pred_path: Path) -> tuple[dict[str, dict], list[dict]]:
        gold_positions = {}
        with item_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    item = mix_row_to_item(json.loads(line))
                    gold_positions[item.item_id] = (item.template_id, item.source)
        rows = [json.loads(line) for line in pred_path.open(encoding="utf-8") if line.strip()]
        return gold_positions, rows

    fast_path = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"
    cg_path = REPO / "data" / "bench" / "osler_v4" / "cg_v2.jsonl"
    composite_recorded = json.loads((Z0 / "composite.json").read_text(encoding="utf-8"))
    for name, fast_pred, cg_pred in (
        ("osler_v4_final", Y8 / "dev_predictions.jsonl", Y8 / "cg_v2_predictions.jsonl"),
        ("A0", REPO / "outputs/osler_v4/Y6/A0/fastdev_v4_predictions.jsonl", REPO / "outputs/osler_v4/Y6/A0/cg_v2_predictions.jsonl"),
    ):
        _, fast_rows = ids_and_rows(fast_path, fast_pred)
        _, cg_rows = ids_and_rows(cg_path, cg_pred)
        mr = [r["correct"] for r in fast_rows if r["source"] in MR]
        mimic = [r["correct"] for r in fast_rows if str(r["template_id"]).startswith("mimic_")]
        cg = [r["correct"] for r in cg_rows]
        s = (np.mean(mr) + np.mean(mimic) + np.mean(cg)) / 3
        recorded = composite_recorded["models"][name]["composite"]
        results.append(check(f"{name} S", float(s), recorded["S"]))
        results.append(check(f"{name} CG v2 micro", float(np.mean(cg)), recorded["cg_micro"]))

    # 4. the two-letter expectation at arm length, in plain numpy
    eligible = float(np.sum(arrays["closed_items"][:13_992]))
    g = []
    for r in range(26, 64):
        g.append(sum(1.0 / c for c in range(max(8, r + 1), 65)) / 57)
    mean_per_row = eligible * float(np.mean(g))
    recorded_exp = json.loads((Z0 / "augment_expectation.json").read_text(encoding="utf-8"))
    results.append(check("expected two-letter golds per row (mean 26..63), arm length, p = 1",
                         mean_per_row, recorded_exp["runs"]["13992"]["expected_per_row_mean_26_63"], tol=1e-6))

    verdict = "ALL MATCH" if all(results) else "SOME MISMATCH"
    print(f"re-derivation: {sum(results)} of {len(results)} figures match: {verdict}", flush=True)


if __name__ == "__main__":
    main()
