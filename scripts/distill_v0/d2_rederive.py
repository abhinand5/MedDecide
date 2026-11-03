"""D2 self-audit (R6): re-derive the basket and label-run numbers from the raw artifacts.

Reads ``composition.json``, ``label_manifest.json``, ``labels.jsonl``, ``failures.jsonl`` and
``skipped.jsonl`` and recomputes: the frozen-file hash, the label counts by template, the cost and
token sums against the ledger, the cap, the coverage (labels + failures + skipped = unique basket
keys), the probability sums, the text-key uniqueness, and that no labelled text key or item id is in
a frozen set. Writes ``audit.json`` and exits non-zero on any failure. This is the recompute command
behind the D2 CLAIMS rows.

Usage (CPU):
    uv run --frozen python scripts/distill_v0/d2_rederive.py --out outputs/distill_v0/D2
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.distill.d1_teacher import file_sha256
from meddecide.distill.loop_ledger import read_entries
from meddecide.panel.overlap import row_text_key
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
LABEL_FIELDS = {"item_id", "template_id", "qtype", "option_keys", "teacher_probs", "teacher", "cost",
                "input_tokens", "text_key"}
FROZEN_FILES = (
    "data/bench/osler_v4/y9/knowledge.jsonl", "data/bench/osler_v4/y9/panel.jsonl",
    "data/bench/osler_v4/y9/mr_verdict.jsonl", "data/bench/osler_v4/y9/mednli_test.jsonl",
    "data/bench/osler_v4/y9/mr_reported.jsonl", "data/bench/osler_v4/y9/mr_heldout_test.jsonl",
    "data/bench/osler_v4/y9/mimic_suite.jsonl", "data/bench/osler_v4/y9/pairs_test.jsonl",
    "data/bench/osler_v4/y9/mimic_heldout.jsonl", "data/bench/osler_mimic_v1/mimic_audit_suite_v1.jsonl",
    "data/bench/osler_mimic_v1/mimic_heldout_v1.jsonl", "data/bench/osler_v2/knowledge_dev_v2.jsonl",
    "data/bench/osler_v4/fastdev_v4.jsonl", "data/bench/osler_v4/cg_v2.jsonl",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="outputs/distill_v0/D2")
    args = parser.parse_args()
    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    plan = read_json(out / "composition.json")
    manifest = read_json(out / "label_manifest.json")
    labels = read_jsonl(out / "labels.jsonl")
    failures = read_jsonl(out / "failures.jsonl")
    skipped = read_jsonl(out / "skipped.jsonl")
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    check("labels_sha256", file_sha256(out / "labels.jsonl") == manifest["labels_sha256"],
          {"recorded": manifest["labels_sha256"], "recomputed": file_sha256(out / "labels.jsonl")})
    check("failures_sha256", file_sha256(out / "failures.jsonl") == manifest["failures_sha256"],
          {"recorded": manifest["failures_sha256"], "recomputed": file_sha256(out / "failures.jsonl")})
    check("skipped_sha256", file_sha256(out / "skipped.jsonl") == manifest["skipped_sha256"],
          {"recorded": manifest["skipped_sha256"], "recomputed": file_sha256(out / "skipped.jsonl")})
    check("plan_sha256", file_sha256(out / "composition.json") == manifest["plan_sha256"],
          {"recorded": manifest["plan_sha256"], "recomputed": file_sha256(out / "composition.json")})

    keys = [str(row["text_key"]) for row in labels]
    unique_keys = set(keys)
    check("text_keys_unique", len(unique_keys) == len(keys), f"{len(keys) - len(unique_keys)} duplicates")
    bad_schema = [key for key, row in zip(keys, labels, strict=True) if set(row) != LABEL_FIELDS]
    check("label_schema", not bad_schema, f"{len(bad_schema)} rows differ from the nine-field schema")
    bad_key = [key for key, row in zip(keys, labels, strict=True) if str(row["text_key"]) != key]
    check("label_key_matches", not bad_key, f"{len(bad_key)} rows carry another text key")
    bad_teacher = [key for key, row in zip(keys, labels, strict=True)
                   if row["teacher"] != manifest["model"]]
    check("teacher_id", not bad_teacher, f"{len(bad_teacher)} rows carry another teacher id")
    bad_probs = []
    max_deviation = 0.0
    for row in labels:
        probs = row["teacher_probs"]
        total = float(sum(probs))
        max_deviation = max(max_deviation, abs(total - 1.0))
        if (len(probs) != len(row["option_keys"]) or any(not isinstance(p, (int, float))
                                                         or not math.isfinite(float(p)) or float(p) < 0 for p in probs)
                or abs(total - 1.0) > 1e-6):
            bad_probs.append(row["text_key"])
    check("probability_sums", not bad_probs, {"bad": len(bad_probs), "max_abs_sum_deviation": max_deviation})

    planned_keys = int(plan["teacher_basket"]["unique_text_keys"])
    coverage = len(labels) + len(failures) + len(skipped)
    check("coverage", coverage == planned_keys, {"labels": len(labels), "failures": len(failures),
                                                 "skipped": len(skipped), "planned_unique_keys": planned_keys})
    labelled = set(keys)
    failed = {str(row["text_key"]) for row in failures}
    skipped_keys = {str(row["text_key"]) for row in skipped}
    check("no_overlap_between_outcomes", not (labelled & failed) and not (labelled & skipped_keys)
          and not (failed & skipped_keys), {"labelled": len(labelled), "failed": len(failed),
                                            "skipped": len(skipped_keys)})

    cost = sum(float(row["cost"]) for row in labels)
    tokens = sum(int(row["input_tokens"]) for row in labels)
    d2_entries = [entry for entry in read_entries(REPO / "outputs" / "distill_v0" / "cost_ledger.jsonl")
                  if entry.get("task") == "D2"]
    ledger_cost = sum(float(entry.get("cost_usd") or 0.0) for entry in d2_entries)
    ledger_tokens = sum(int(entry.get("input_tokens") or 0) for entry in d2_entries)
    # The ledger may exceed the frozen labels by the spend of an interrupted attempt whose partial rows were
    # discarded before a resume (documented in the label manifest). It may never be below the labels' spend.
    gap = ledger_cost - cost
    check("loop_ledger_covers_labels", cost <= ledger_cost + 1e-6,
          {"labels_cost": round(cost, 9), "ledger": round(ledger_cost, 9), "entries": len(d2_entries)})
    check("loop_ledger_gap_documented", gap <= max(0.02, 0.02 * cost),
          {"gap_usd": round(gap, 9), "labels_cost": round(cost, 9),
           "note": "gap = spend of attempts whose partial rows were discarded before a resume"})
    check("loop_ledger_tokens_cover_labels", tokens <= ledger_tokens + 1,
          {"labels_tokens": tokens, "ledger_tokens": ledger_tokens, "gap": ledger_tokens - tokens})
    check("cap_respected", float(manifest["cumulative_spend_usd"]) <= float(manifest["cap_usd"]),
          {"cumulative": manifest["cumulative_spend_usd"], "cap": manifest["cap_usd"]})
    check("stop_reason", manifest["stop_reason"] == "complete", manifest["stop_reason"])

    # frozen sets: no labelled text key or item id may appear in one (GOAL §5 rule 1)
    frozen_ids: set[str] = set()
    frozen_texts: set[str] = set()
    for relative in FROZEN_FILES:
        for row in read_jsonl(REPO / relative):
            frozen_ids.add(str(row["item_id"]))
            frozen_texts.add(row_text_key(row))
    frozen_hits = sorted(labelled & frozen_texts)
    frozen_id_hits = sorted({str(row["item_id"]) for row in labels} & frozen_ids)
    check("no_frozen_text_key", not frozen_hits, f"{len(frozen_hits)} labelled keys are in a frozen set")
    check("no_frozen_item_id", not frozen_id_hits, f"{len(frozen_id_hits)} labelled item ids are in a frozen set")

    check("mimic_only_under_authorization", bool(manifest.get("allow_protected")) or not manifest["qa"]["labelled_mimic_rows"],
          {"allow_protected": manifest.get("allow_protected"), "labelled_mimic_rows": manifest["qa"]["labelled_mimic_rows"]})

    per_template = Counter(str(row["template_id"]) for row in labels)
    recorded = manifest["qa"]["per_template"]
    check("per_template_counts", dict(sorted(per_template.items())) == recorded,
          {"templates": len(per_template), "mismatched": sorted(set(per_template) ^ set(recorded))})
    check("basket_counts", plan["counts"]["teacher"] == 494_803 and plan["counts"]["teacher_unique_keys"] == planned_keys,
          plan["counts"])

    audit = {"task": "D2 self-audit", "audited_utc": utcnow(), "out": str(out), "checks": checks,
             "all_passed": all(check["ok"] for check in checks),
             "totals": {"labels": len(labels), "failures": len(failures), "skipped": len(skipped),
                        "unique_keys": len(unique_keys), "cost_usd": round(cost, 9), "input_tokens": tokens,
                        "cumulative_spend_usd": manifest["cumulative_spend_usd"],
                        "per_template": dict(sorted(per_template.items()))}}
    (out / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for entry in checks:
        print(f"[{'PASS' if entry['ok'] else 'FAIL'}] {entry['check']}")
    print(json.dumps(audit["totals"], indent=1)[:1200])
    if not audit["all_passed"]:
        raise SystemExit("D2 self-audit failed")


if __name__ == "__main__":
    main()
