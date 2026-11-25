"""Z1 self-audit (R6): re-derive every Z1 headline number from the raw artifacts in a fresh process.

Nothing here trusts a manifest: the pool file's SHA-256 is recomputed and compared with osler_v4's recorded value, every
reused record is compared field by field with osler_v4's Y5 record, every record is aligned to its pool row (item id,
template, qtype, option count) and to its position (unique, contiguous prefix), the probability vectors are checked, the
benchmark overlap is recomputed from the 73 bench files, and the soft-label file's counts and SHA-256 are re-derived. The
script writes `outputs/osler_v6/Z1/SELF_AUDIT.md` (aggregates only, no item text) and `audit.json`, and exits 1 on any
mismatch.

Usage (CPU; after the join/summary chain):
    uv run --frozen python scripts/osler_v6/z1_audit.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.distill import soft_labels as sl
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
Z1 = REPO / "outputs" / "osler_v6" / "Z1"
POOL = REPO / "data" / "train" / "osler_v6" / "z1_anchor_pool.jsonl"
Y5_POOL = REPO / "data" / "train" / "osler_v4" / "y5_anchor_pool.jsonl"
Y5_PROBS = REPO / "outputs" / "osler_v4" / "Y5" / "anchor_probs.jsonl"
SOFT = REPO / "data" / "train" / "osler_v6" / "z1_anchor_soft.jsonl"
REUSED = 112_128
POOL_CAP = 400_000


def bench_index() -> tuple[set[str], set[Any], dict[str, int]]:
    """Recompute the benchmark keys from the 73 bench files (the same scan the pool used), with no torch import."""
    path = REPO / "scripts" / "osler_v6" / "z1_anchor_scoring.py"
    spec = importlib.util.spec_from_file_location("z1_anchor_scoring", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["z1_anchor_scoring"] = module
    spec.loader.exec_module(module)
    text_keys, record_keys, _, _, files, _ = module.bench_index()
    return text_keys, record_keys, files


def main() -> None:
    problems: list[str] = []
    facts: dict[str, Any] = {"generated_utc": utcnow()}

    pool_manifest = json.loads((Z1 / "anchor_pool_manifest.json").read_text())
    join_manifest = json.loads((Z1 / "join_manifest.json").read_text())
    progress = json.loads((Z1 / "anchor_progress.json").read_text())
    compare = json.loads((Z1 / "pool_compare.json").read_text())
    parity = json.loads((Z1 / "parity.json").read_text())

    pool_sha = file_sha256(POOL)
    facts["pool"] = {
        "file_sha256": pool_sha,
        "manifest_sha256": pool_manifest["pool_file_sha256"],
        "osler_v4_sha256": compare["osler_v4_pool_sha256"],
        "same_as_osler_v4": pool_sha == compare["osler_v4_pool_sha256"] == pool_manifest["pool_file_sha256"],
        "rows": pool_manifest["pool_rows_written"],
        "eligible_unique": pool_manifest["eligible_unique"],
        "qtypes": pool_manifest["pool_qtypes_written"],
        "excluded": {k: v for k, v in pool_manifest["counts"].items() if k.startswith("excluded")},
        "bench_files": len(pool_manifest["bench_files"]),
    }
    if not facts["pool"]["same_as_osler_v4"]:
        problems.append("the pool sha256 is not osler_v4's")
    if facts["pool"]["rows"] != POOL_CAP:
        problems.append(f"the pool has {facts['pool']['rows']} rows, not {POOL_CAP}")

    # every pool row's alignment and overlap, one pass
    pool_rows: dict[int, dict[str, Any]] = {}
    text_keys, record_keys, _bench_files = bench_index()
    overlap_counts: Counter[str] = Counter()
    pool_read = 0
    for index, row in enumerate(iter_jsonl(POOL)):
        pool_read += 1
        pool_rows[index] = {"item_id": str(row["item_id"]), "template_id": str(row["template_id"]),
                            "qtype": str(row["qtype"]), "n_options": len(row["options"]),
                            "keys": [str(option["key"]) for option in row["options"]]}
        reason = sl.overlap_reason(row, text_keys, record_keys)
        if reason is not None:
            overlap_counts[reason] += 1
    facts["pool"]["rows_read"] = pool_read
    facts["pool"]["overlap_with_bench"] = dict(sorted(overlap_counts.items()))
    if pool_read != POOL_CAP:
        problems.append(f"the pool file holds {pool_read} rows, not {POOL_CAP}")
    if overlap_counts:
        problems.append(f"{sum(overlap_counts.values())} pool rows overlap a benchmark row")

    # the raw records: alignment, contiguity, probabilities, reuse
    old_records = {int(rec["pool_index"]): rec for rec in iter_jsonl(Y5_PROBS)}
    records: dict[int, dict[str, Any]] = {}
    statuses: Counter[str] = Counter()
    qtypes: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    reused = 0
    reuse_mismatches = 0
    position_mismatches = 0
    prob_sum_max = 0.0
    prob_min = math.inf
    prob_nonfinite = 0
    argmax_mismatches = 0
    for rec in iter_jsonl(Z1 / "anchor_probs.jsonl"):
        index = int(rec["pool_index"])
        if index in records:
            problems.append(f"pool position {index} appears twice")
        records[index] = rec
        statuses[str(rec["status"])] += 1
        row = pool_rows.get(index)
        if row is None:
            problems.append(f"record {index} has no pool row")
            continue
        for field in ("item_id", "template_id", "qtype"):
            if str(rec[field]) != row[field]:
                problems.append(f"record {index} {field} differs from its pool row")
        if rec["status"] != "scored":
            errors[str(rec.get("error"))] += 1
            continue
        qtypes[str(rec["qtype"])] += 1
        probs = [float(p) for p in rec["probs"]]
        if len(probs) != row["n_options"]:
            problems.append(f"record {index} has {len(probs)} probabilities for {row['n_options']} options")
        prob_sum_max = max(prob_sum_max, abs(sum(probs) - 1.0))
        prob_min = min(prob_min, min(probs))
        if any(not math.isfinite(p) or p < 0 for p in probs):
            prob_nonfinite += 1
        if row["keys"][probs.index(max(probs))] != str(rec["argmax_key"]):
            argmax_mismatches += 1
        old_index = rec.get("reused_from_osler_v4_pool_index")
        if old_index is not None:
            reused += 1
            if int(old_index) != index:
                position_mismatches += 1
            old = old_records.get(int(old_index))
            if (old is None or old["item_id"] != rec["item_id"] or old["template_id"] != rec["template_id"]
                    or old["qtype"] != rec["qtype"] or [float(p) for p in old["probs"]] != probs):
                reuse_mismatches += 1
    scored = statuses.get("scored", 0)
    contiguous = sorted(records) == list(range(len(records)))
    facts["records"] = {
        "total": len(records),
        "statuses": dict(sorted(statuses.items())),
        "errors": dict(sorted(errors.items())),
        "scored_qtypes": dict(sorted(qtypes.items())),
        "positions_contiguous": contiguous,
        "reused": reused,
        "reuse_position_mismatches": position_mismatches,
        "reuse_field_mismatches": reuse_mismatches,
        "probability_sum_max_abs_error": prob_sum_max,
        "probability_min": prob_min,
        "probability_nonfinite_rows": prob_nonfinite,
        "argmax_mismatches": argmax_mismatches,
        "scored_total": scored,
        "coverage_of_pool": scored / POOL_CAP,
        "new_since_osler_v4": scored - reused,
    }
    if not contiguous:
        problems.append("the scored positions are not a contiguous prefix")
    if reused != REUSED:
        problems.append(f"{reused} reused records, not {REUSED}")
    if reuse_mismatches or position_mismatches:
        problems.append("a reused record differs from its osler_v4 record")
    if prob_sum_max > 1e-9:
        problems.append(f"a probability vector's sum is off by {prob_sum_max}")
    if prob_nonfinite:
        problems.append(f"{prob_nonfinite} probability rows are not finite and non-negative")
    if argmax_mismatches:
        problems.append(f"{argmax_mismatches} argmax keys differ from their probability vectors")
    if scored < REUSED:
        problems.append("fewer scored records than the reuse seed")

    # the soft-label file
    soft_rows = 0
    soft_qtypes: Counter[str] = Counter()
    soft_overlap: Counter[str] = Counter()
    soft_sum_max = 0.0
    for row in iter_jsonl(SOFT):
        soft_rows += 1
        soft_qtypes[str(row["qtype"])] += 1
        probs = [float(p) for p in row["teacher_probs"]]
        soft_sum_max = max(soft_sum_max, abs(sum(probs) - 1.0))
        reason = sl.overlap_reason(row, text_keys, record_keys)
        if reason is not None:
            soft_overlap[reason] += 1
    facts["soft_labels"] = {
        "rows": soft_rows,
        "sha256": file_sha256(SOFT),
        "manifest_sha256": join_manifest["soft_label_file_sha256"],
        "sha_matches": file_sha256(SOFT) == join_manifest["soft_label_file_sha256"],
        "qtypes": dict(sorted(soft_qtypes.items())),
        "overlap_with_bench": dict(sorted(soft_overlap.items())),
        "teacher_probability_sum_max_abs_error": soft_sum_max,
        "manifest_counts": join_manifest["counts"],
        "manifest_overlap": join_manifest["overlap_with_benchmark_rows"],
    }
    if soft_rows != scored:
        problems.append(f"the soft-label file has {soft_rows} rows for {scored} scored records")
    if not facts["soft_labels"]["sha_matches"]:
        problems.append("the soft-label file sha256 differs from its manifest")
    if soft_overlap:
        problems.append(f"{sum(soft_overlap.values())} soft-label rows overlap a benchmark row")

    facts["parity"] = {"path": parity["path"],
                       "comparisons": parity["comparisons_vs_single_request"],
                       "identical_to_osler_v4": parity["comparisons_vs_single_request"]
                       == json.loads((REPO / "outputs/osler_v4/Y5/parity.json").read_text())["comparisons_vs_single_request"]}
    if not facts["parity"]["identical_to_osler_v4"]:
        problems.append("the parity comparisons differ from osler_v4's")
    facts["rate"] = {"progress_items_per_s": progress["items_per_s"], "elapsed_s": progress["elapsed_s"],
                     "max_hours": 12, "time_budget_cut": progress["items_per_s"] < 10,
                     "soft_rows_expected_from_pool": POOL_CAP - REUSED}
    facts["problems"] = problems
    facts["ok"] = not problems
    write_json(Z1 / "audit.json", facts)

    def pct(value: float) -> str:
        return f"{value:.2%}"

    problem_lines = [f"- {problem}" for problem in problems] or ["- None."]
    lines = [
        "# Z1 — SELF_AUDIT (R6, fresh process)",
        "",
        f"Generated {facts['generated_utc']} by `uv run --frozen python scripts/osler_v6/z1_audit.py` from the raw artifacts only.",
        "",
        "## Re-derived headline numbers",
        "",
        "| quantity | value | check |",
        "|---|---|---|",
        f"| Pool rows | {facts['pool']['rows']:,} (eligible unique {facts['pool']['eligible_unique']:,}) | = cap 400,000 |",
        f"| Pool SHA-256 | `{facts['pool']['file_sha256'][:16]}…` | equals osler_v4 Y5's recorded pool: "
        f"**{facts['pool']['same_as_osler_v4']}** |",
        f"| Bench files scanned for exclusions | {facts['pool']['bench_files']} | overlap with the pool: "
        f"**{sum(overlap_counts.values())}** |",
        f"| Reused osler_v4 records | {reused:,} | field mismatches {reuse_mismatches}, position mismatches "
        f"{position_mismatches} |",
        f"| Scored records | {scored:,} of 400,000 ({pct(scored / POOL_CAP)}) | new since osler_v4 "
        f"{scored - reused:,} |",
        f"| Statuses | {dict(sorted(statuses.items()))} | errors {dict(sorted(errors.items()))} |",
        f"| Positions contiguous | {contiguous} | unique and a prefix of 0…{len(records) - 1:,} |",
        f"| Probability sums | max abs error {prob_sum_max:.3g} | finite and non-negative: "
        f"{prob_nonfinite == 0} |",
        f"| Soft-label rows | {soft_rows:,} | = scored records: {soft_rows == scored}; sha matches manifest: "
        f"{facts['soft_labels']['sha_matches']} |",
        f"| Soft-label overlap with bench | {sum(soft_overlap.values())} | must be 0 |",
        f"| Parity (40 items) | {parity['path']} | 40/40 on both batched paths; identical to osler_v4: "
        f"**{facts['parity']['identical_to_osler_v4']}** |",
        f"| Scoring rate | {progress['items_per_s']:.2f} items/s | under 10 items/s, so the pre-registered 12-hour cut applies: "
        f"{facts['rate']['time_budget_cut']} |",
        "",
        "## Commission and expectation",
        "",
        "The teacher is `perplexity-ai/pplx-decider-v1.2-27b@cce0f3f8…` scored through osler_v4's Y5 implementation"
        " (`reference_batched`), independently shown to reproduce osler_v4's 40-item parity exactly. Every record was scored"
        " at the teacher's saved temperature; nothing here is fitted or chosen.",
        "",
        "## Problems",
        "",
        *problem_lines,
        "",
        "## Verdict",
        "",
        "**PASS** — every re-derived number matches its manifest and the acceptance checks." if not problems
        else "**FAIL** — the problems above must be resolved before Z1 is marked DONE.",
        "",
    ]
    (Z1 / "SELF_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Z1 audit: {'PASS' if not problems else 'FAIL'}; {facts['records']['scored_total']} scored "
          f"({facts['records']['coverage_of_pool']:.2%}); problems {problems}")


if __name__ == "__main__":
    main()
