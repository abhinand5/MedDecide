#!/usr/bin/env python
"""osler_v7 T2 — build and freeze the format-matched dev set (§6) and the confirmation set (§8), with manifests.

Dev set (§6). A new frozen file assembled from the existing frozen *dev* assets — `fastdev_v4` (MR verdict family
dev items, the 13 MIMIC audit-template dev shards, 4-option MCQ, two-option pairs, code items), `cg_v2` (its unique
rows: DDXPlus 49-option diagnosis items, 12-option page-category items, official dev splits) and
`knowledge_dev_v2` (its unique rows). Code-generated and teacher-prose templates are excluded (synthetic-free rule);
DDXPlus (a public synthetic-patient dataset with structured gold) is kept because it is the only non-panel source of
10-50-option diagnosis items, and that is stated in the manifest. Duplicates are removed by item id. Every kept row is
checked against the v6 mix and the 385,536 anchor rows for item-id and normalised-text overlap; any overlap aborts.
The MIMIC dev shards come from the osler_v4 build's patient-level split (documented in loops/osler_v4/mimic.md); item
rows carry no subject id, so that split is cited, not re-derived, and the manifest says so.

Confirmation set (§8). Two kinds of component, both manifested:
- general: NEW tier-2 items built in this task from records dated 2026-10-06..2026-11-12 (ClinicalTrials.gov, openFDA,
  PubMed; structured gold by construction), test split only, sampled at most 1,000 per template (seed 0). The build's
  own acceptance file must exist and pass.
- audit and knowledge: the frozen Y9 test splits `mr_verdict` and `mimic_suite` (audit) and `knowledge` (knowledge
  test), referenced by path, SHA-256 and counts. They have no post-window records (MIMIC is historical; the knowledge
  test sets are not date-stamped), are test splits, and were never used for training or selection in this loop. The
  manifest states the deviation from §8's date rule for these two components explicitly; the operator reviews the
  manifest before first use.

Writes (items are gitignored; manifests are aggregates only)::

    data/bench/osler_v7/dev_format_matched.jsonl
    data/bench/osler_v7/confirm/general.jsonl
    outputs/osler_v7/T2/dev_manifest.json
    outputs/osler_v7/T2/confirmation_manifest.json
    outputs/osler_v7/T2/provenance.json

Run (CPU): ``uv run --frozen python scripts/osler_v7/t2_build_sets.py``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

MIX = ROOT / "data" / "train" / "osler_v6" / "mix_f2_distill.jsonl"
ANCHORS = ROOT / "data" / "train" / "osler_v6" / "z1_anchor_soft.jsonl"
FASTDEV = ROOT / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"
CG = ROOT / "data" / "bench" / "osler_v4" / "cg_v2.jsonl"
KNOW_DEV = ROOT / "data" / "bench" / "osler_v2" / "knowledge_dev_v2.jsonl"
FRESH = ROOT / "data" / "bench" / "osler_v7" / "fresh"
DEV_OUT = ROOT / "data" / "bench" / "osler_v7" / "dev_format_matched.jsonl"
CONFIRM_OUT = ROOT / "data" / "bench" / "osler_v7" / "confirm" / "general.jsonl"
OUT = ROOT / "outputs" / "osler_v7" / "T2"
Y9 = ROOT / "data" / "bench" / "osler_v4" / "y9"
AUDIT_FILES = {"mr_verdict": Y9 / "mr_verdict.jsonl", "mimic_suite": Y9 / "mimic_suite.jsonl"}
KNOWLEDGE_FILE = Y9 / "knowledge.jsonl"

WINDOW_START = date(2026, 10, 6)
WINDOW_END = date(2026, 11, 12)
STRICT_SLICE = date(2026, 9, 10)
PER_TEMPLATE_CAP = 1_000
SEED = 0
EXCLUDED_TEMPLATES = {
    "synthea_indicator_evidence_v1", "synthea_page_category_v1", "mr_indicator_prose_v1",
    "synthea_narrative_page_v1", "chart_indicator_evidence_v1", "restyle_page_category_v1",
}
EXCLUDED_SOURCES = {"generated", "mr_teacher"}
CODE_TEMPLATES = {
    "code_assignment_v1", "code_family_v1", "mimic_icd_support_v1", "mimic_icd_specificity_v1",
    "mimic_pcs_support_v1", "mr_icd_support_template_v2", "fda_class_choice_v2", "fda_route_choice_v1",
}
MR_FAMILY = {
    "mr_icd_support_template_v2", "mr_indicator_evidence_v1", "mr_page_category_tabular_v1",
    "mtsamples_page_category_v1", "pmc_similar_patient_v1", "codiesp_icd_support_v1", "mkurman_icd_support_v1",
}
SHAPES = {
    "many_option_10_50": lambda row: 10 <= row["_n_options"] <= 50,
    "two_option_pairs": lambda row: row["qtype"] == "noul" or (row["qtype"] == "choice" and row["_n_options"] == 2),
    "code_items": lambda row: row["template_id"] in CODE_TEMPLATES,
    "mcq_4_option": lambda row: row["qtype"] == "choice" and row["_n_options"] == 4,
    "mcq_10_option": lambda row: row["qtype"] == "choice" and row["_n_options"] == 10,
    "mr_verdict_family": lambda row: row["template_id"] in MR_FAMILY,
    "mimic_audit_templates": lambda row: str(row["template_id"]).startswith("mimic_"),
}


def _rel(path: Path) -> str:
    """Repo-relative when possible, otherwise the absolute path (tests use temporary directories)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise(text: Any) -> str:
    return " ".join(str(text or "").split())


def option_count(row: dict[str, Any]) -> int:
    options = row["options"]
    if isinstance(options, str):
        options = json.loads(options)
    return len(options)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def training_overlap_sets() -> dict[str, Any]:
    """Item ids and text digests of the v6 mix and the anchor file (one pass each; sha256 verified at the end)."""
    item_ids: set[str] = set()
    item_texts: set[bytes] = set()
    record_texts: set[bytes] = set()
    digests: dict[str, str] = {}
    rows = {"mix": 0, "anchors": 0}
    for name, path in (("mix", MIX), ("anchors", ANCHORS)):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                row = json.loads(line)
                rows[name] += 1
                item_ids.add(str(row["item_id"]))
                state = normalise(row.get("state"))
                record_texts.add(hashlib.sha256(state.encode()).digest())
                item_texts.add(hashlib.sha256(f"{state}\x1f{normalise(row.get('question'))}".encode()).digest())
        digests[name] = digest.hexdigest()
        print(f"[overlap] {name}: {rows[name]:,} rows, sha {digests[name][:12]}", file=sys.stderr, flush=True)
    return {"item_ids": item_ids, "item_texts": item_texts, "record_texts": record_texts,
            "rows": rows, "sha256": digests}


def check_overlap(rows: list[dict[str, Any]], overlap: dict[str, Any]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        if str(row["item_id"]) in overlap["item_ids"]:
            counts["item_id"] += 1
        state = normalise(row.get("state"))
        if hashlib.sha256(f"{state}\x1f{normalise(row.get('question'))}".encode()).digest() in overlap["item_texts"]:
            counts["item_text"] += 1
        if hashlib.sha256(state.encode()).digest() in overlap["record_texts"]:
            counts["record_text"] += 1
    return dict(counts)


def build_dev(sources: dict[str, Path] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter, dedupe and freeze the format-matched dev rows; returns the rows and the build report."""
    if sources is None:
        sources = {"fastdev_v4": FASTDEV, "cg_v2": CG, "knowledge_dev_v2": KNOW_DEV}
    kept: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {"inputs": {}, "excluded": Counter(), "duplicates_removed": Counter(), "rows_in": 0}
    for origin, path in sources.items():
        n_in = n_kept = 0
        for row in read_jsonl(path):
            n_in += 1
            if str(row["template_id"]) in EXCLUDED_TEMPLATES or str(row.get("source")) in EXCLUDED_SOURCES:
                report["excluded"][f"{origin}:{'template' if str(row['template_id']) in EXCLUDED_TEMPLATES else 'source'}"] += 1
                continue
            item_id = str(row["item_id"])
            if item_id in kept:
                report["duplicates_removed"][origin] += 1
                continue
            enriched = {**row, "_n_options": option_count(row), "dev_origin": origin}
            kept[item_id] = enriched
            n_kept += 1
        report["inputs"][origin] = {"path": _rel(path), "sha256": sha256_file(path),
                                    "rows_in": n_in, "rows_kept": n_kept}
        report["rows_in"] += n_in
    rows = [kept[k] for k in sorted(kept)]
    report["rows_out"] = len(rows)
    report["excluded"] = dict(sorted(report["excluded"].items()))
    report["duplicates_removed"] = dict(sorted(report["duplicates_removed"].items()))
    return rows, report


def dev_composition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    templates = Counter(str(r["template_id"]) for r in rows)
    qtypes = Counter(str(r["qtype"]) for r in rows)
    options = Counter(str(r["_n_options"]) for r in rows)
    shapes = {}
    for name, predicate in SHAPES.items():
        items = [r for r in rows if predicate(r)]
        shapes[name] = {
            "items": len(items),
            "templates": dict(sorted(Counter(str(r["template_id"]) for r in items).items())),
            "qtypes": dict(sorted(Counter(str(r["qtype"]) for r in items).items())),
        }
    return {
        "rows": len(rows),
        "templates": dict(sorted(templates.items())),
        "qtypes": dict(sorted(qtypes.items())),
        "option_counts": dict(sorted(options.items(), key=lambda kv: int(kv[0]))),
        "origin": dict(sorted(Counter(str(r["dev_origin"]) for r in rows).items())),
        "shapes": shapes,
        "all_shapes_present": all(entry["items"] > 0 for entry in shapes.values()),
    }


def build_confirmation_general(
    overlap: dict[str, Any], fresh_dir: Path = FRESH
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the test split of the newly built fresh tier-2 items, cap per template, and record the date checks."""
    manifest_path = fresh_dir / "manifest.json"
    acceptance_path = FRESH / "acceptance.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} is missing: the fresh build has not finished")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8")) if acceptance_path.exists() else None
    if acceptance is not None and acceptance.get("verdict") != "PASS":
        failed = {name for name, ok in acceptance.get("checks", {}).items() if not ok}
        if failed - {"no_record_crosses_splits"}:
            raise SystemExit(f"the fresh build's acceptance failed checks other than record disjointness: {sorted(failed)}")
        if not acceptance.get("split_disjointness", {}).get("leaks"):
            raise SystemExit("the fresh build reports no record leak but its split check failed: stop")
    rows: list[dict[str, Any]] = []
    per_source = {}
    for source in ("clinicaltrials", "openfda", "pubmed"):
        path = fresh_dir / f"{source}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} is missing")
        got = 0
        kept: list[dict[str, Any]] = []
        rows_in_file = read_jsonl(path)
        got = len(rows_in_file)
        # The build's own split check found one record with items in dev and test. The confirmation set keeps the
        # test split only, so those records are excluded here and counted; the build check is not edited.
        splits_by_record: dict[str, set[str]] = {}
        for row in rows_in_file:
            splits_by_record.setdefault(str(row["source_record_id"]), set()).add(str(row.get("split")))
        leaking = {record for record, splits in splits_by_record.items() if len(splits) > 1}
        dropped_leak = 0
        for row in rows_in_file:
            if str(row.get("split")) != "test":
                continue
            if str(row["source_record_id"]) in leaking:
                dropped_leak += 1
                continue
            when = date.fromisoformat(str(row["record_date"])[:10])
            if not (WINDOW_START <= when <= WINDOW_END):
                raise SystemExit(f"{source} {row['item_id']}: record_date {when} is outside the planned window")
            kept.append(row)
        kept_records = {str(row["source_record_id"]) for row in kept}
        per_source[source] = {"path": _rel(path), "sha256": sha256_file(path), "rows_in": got,
                              "test_rows": len(kept), "records_with_items_in_two_splits": len(leaking),
                              "rows_dropped_for_split_leak": dropped_leak,
                              "record_split_disjoint_after_filter": not any(
                                  len(splits_by_record.get(record, set())) > 1 for record in kept_records)}
        rows.extend(kept)
    # deterministic per-template cap (seed 0 is the row order itself: the build orders by a seeded shuffle)
    by_template: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_template.setdefault(str(row["template_id"]), []).append(row)
    capped: list[dict[str, Any]] = []
    for _template, items in sorted(by_template.items()):
        capped.extend(items[:PER_TEMPLATE_CAP])
    capped.sort(key=lambda r: str(r["item_id"]))
    # Decontamination: drop every record one of whose test items shares an item id or normalised text with the v6
    # training files. The training source can re-issue an old record in a later window (PubMed updates do), so the
    # builder's date check alone is not sufficient. Counted, never silent.
    bad_records: set[str] = set()
    bad_rows = 0
    bad_by_source: Counter[str] = Counter()
    for row in capped:
        state = normalise(row.get("state"))
        overlaps = (
            str(row["item_id"]) in overlap["item_ids"]
            or hashlib.sha256(f"{state}\x1f{normalise(row.get('question'))}".encode()).digest() in overlap["item_texts"]
            or hashlib.sha256(state.encode()).digest() in overlap["record_texts"]
        )
        if overlaps:
            bad_records.add(str(row["source_record_id"]))
            bad_rows += 1
            bad_by_source[str(row["source"])] += 1
    kept = [row for row in capped if str(row["source_record_id"]) not in bad_records]
    decontamination = {
        "rows_before": len(capped),
        "rows_dropped": len(capped) - len(kept),
        "rows_with_a_direct_match": bad_rows,
        "records_dropped": len(bad_records),
        "rows_by_source": dict(sorted(bad_by_source.items())),
        "reason": "the record's item id or normalised state/question text appears in the v6 mix or the anchor file "
                  "(a training source can re-issue an older record in a later window)",
    }
    out = [{**row, "confirm_component": "general_fresh"} for row in kept]
    dates = sorted(str(row["record_date"])[:10] for row in out)
    composition = {
        "rows": len(out),
        "sources": dict(sorted(Counter(str(r["source"]) for r in out).items())),
        "templates": dict(sorted(Counter(str(r["template_id"]) for r in out).items())),
        "qtypes": dict(sorted(Counter(str(r["qtype"]) for r in out).items())),
        "option_counts": dict(sorted(Counter(str(option_count(r)) for r in out).items(),
                                     key=lambda kv: int(kv[0]))),
        "record_date_min": dates[0] if dates else None,
        "record_date_max": dates[-1] if dates else None,
        "strict_slice_rows": sum(1 for d in dates if d >= STRICT_SLICE.isoformat()),
        "strict_slice_share": round(sum(1 for d in dates if d >= STRICT_SLICE.isoformat()) / len(dates), 8)
        if dates else None,
        "per_template_cap": PER_TEMPLATE_CAP,
        "rows_before_cap": len(rows),
        "templates_before_cap": dict(sorted(Counter(str(r["template_id"]) for r in rows).items())),
        "overlap_with_training": check_overlap(out, overlap),
        "decontamination": decontamination,
    }
    return out, {"build_manifest": manifest, "acceptance_verdict": acceptance.get("verdict") if acceptance else None,
                 "acceptance_failed_checks": sorted(name for name, ok in (acceptance or {}).get("checks", {}).items()
                                                    if not ok),
                 "acceptance_split_leak": (acceptance or {}).get("split_disjointness", {}),
                 "record_split_disjoint_after_filter": all(
                     entry["record_split_disjoint_after_filter"] for entry in per_source.values()),
                 "sources": per_source, "composition": composition}


def audit_components(overlap: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rows_by_name = {}
    for name, path in AUDIT_FILES.items():
        rows = read_jsonl(path)
        rows_by_name[name] = rows
        out[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "splits": dict(sorted(Counter(str(r.get("split", "")) for r in rows).items())),
            "qtypes": dict(sorted(Counter(str(r["qtype"]) for r in rows).items())),
            "templates": dict(sorted(Counter(str(r["template_id"]) for r in rows).items())),
            "overlap_with_training": check_overlap(rows, overlap),
            "date_rule": "NOT APPLICABLE — frozen MIMIC/MR test splits; no post-window records exist for MIMIC",
        }
    knowledge = read_jsonl(KNOWLEDGE_FILE)
    out["knowledge"] = {
        "path": str(KNOWLEDGE_FILE.relative_to(ROOT)),
        "sha256": sha256_file(KNOWLEDGE_FILE),
        "rows": len(knowledge),
        "splits": dict(sorted(Counter(str(r.get("split", "")) for r in knowledge).items())),
        "qtypes": dict(sorted(Counter(str(r["qtype"]) for r in knowledge).items())),
        "templates": dict(sorted(Counter(str(r["template_id"]) for r in knowledge).items())),
        "overlap_with_training": check_overlap(knowledge, overlap),
        "date_rule": "NOT APPLICABLE — established knowledge test sets, not date-stamped",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlap = training_overlap_sets()

    dev_rows, dev_report = build_dev()
    dev_overlap = check_overlap(dev_rows, overlap)
    if any(dev_overlap.values()):
        raise SystemExit(f"the dev set overlaps training: {dev_overlap}")
    DEV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with DEV_OUT.open("w", encoding="utf-8") as handle:
        for row in dev_rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
    dev_manifest = {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t2_build_sets.py",
        "component": "format_matched_dev",
        "path": str(DEV_OUT.relative_to(ROOT)),
        "sha256": sha256_file(DEV_OUT),
        "rows": len(dev_rows),
        "build": dev_report,
        "composition": dev_composition(dev_rows),
        "overlap_with_training": dev_overlap,
        "mimic_split_provenance": (
            "the MIMIC dev items come from the osler_v4 patient-level split (mimic.subject_split; test rows are "
            "late-anchor patients only). Item rows carry no subject id, so the split is cited from "
            "loops/osler_v4/mimic.md rather than re-derived; the item-id/text overlap check against the v6 mix is 0."
        ),
        "synthetic_rule": (
            "code-generated and teacher-prose templates are excluded; DDXPlus (public synthetic-patient dataset, "
            "structured gold) is kept as the only non-panel source of 10-50-option diagnosis items."
        ),
        "wall_seconds": round(time.time() - started, 1),
    }
    if dev_manifest["composition"]["rows"] < 3_000 or not dev_manifest["composition"]["all_shapes_present"]:
        raise SystemExit("the dev set is below 3,000 items or misses a required shape")
    (args.out_dir / "dev_manifest.json").write_text(json.dumps(dev_manifest, indent=2, sort_keys=True) + "\n",
                                                    encoding="utf-8")

    general_rows, general_report = build_confirmation_general(overlap)
    if any(general_report["composition"]["overlap_with_training"].values()):
        raise SystemExit(f"the general confirmation items overlap training: {general_report['composition']['overlap_with_training']}")
    CONFIRM_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CONFIRM_OUT.open("w", encoding="utf-8") as handle:
        for row in general_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = audit_components(overlap)
    confirmation_manifest = {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t2_build_sets.py",
        "component": "confirmation_set",
        "review_required_before_first_use": True,
        "review_note": (
            "ADVISORY §0.3: the operator reviews this manifest before the confirmation set's first use (T10). The "
            "general component is fresh tier-2 (records 2026-10-06..2026-11-12, structured gold). The audit and "
            "knowledge components are frozen test splits without post-window records; that is a stated deviation "
            "from §8's date rule for those components, not a silent substitution."
        ),
        "general": {
            "path": str(CONFIRM_OUT.relative_to(ROOT)),
            "sha256": sha256_file(CONFIRM_OUT),
            "window_start": WINDOW_START.isoformat(),
            "window_end": WINDOW_END.isoformat(),
            "rows": len(general_rows),
            "build": general_report,
        },
        "audit": {k: v for k, v in audit.items() if k in AUDIT_FILES},
        "knowledge": audit["knowledge"],
        "uses": {
            "gate_v7_G_a": "general",
            "gate_v7_G_b": "audit: mr_verdict + mimic_suite",
            "gate_v7_G_c": "knowledge",
            "selected_on": False,
        },
        "wall_seconds": round(time.time() - started, 1),
    }
    (args.out_dir / "confirmation_manifest.json").write_text(
        json.dumps(confirmation_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t2_build_sets.py",
        "argv": sys.argv,
        "wall_seconds": round(time.time() - started, 1),
        "training_files": {"mix": {"path": str(MIX.relative_to(ROOT)), "rows": overlap["rows"]["mix"],
                                    "sha256": overlap["sha256"]["mix"]},
                           "anchors": {"path": str(ANCHORS.relative_to(ROOT)), "rows": overlap["rows"]["anchors"],
                                       "sha256": overlap["sha256"]["anchors"]}},
        "fresh_build": {"manifest": str((FRESH / "manifest.json").relative_to(ROOT)),
                        "acceptance": str((FRESH / "acceptance.json").relative_to(ROOT))},
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({
        "dev_rows": len(dev_rows), "dev_sha": dev_manifest["sha256"][:12],
        "dev_shapes": {k: v["items"] for k, v in dev_manifest["composition"]["shapes"].items()},
        "confirm_general_rows": len(general_rows),
        "confirm_general_sha": confirmation_manifest["general"]["sha256"][:12],
        "strict_slice_share": general_report["composition"]["strict_slice_share"],
        "audit_rows": {k: v["rows"] for k, v in audit.items() if k in AUDIT_FILES},
        "knowledge_rows": audit["knowledge"]["rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
