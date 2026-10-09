"""O4 verifier: re-derive the training mix v2 headline numbers from the written files, in a fresh process (R6).

Reads data/train/osler_v0/{train,dev}.jsonl and the protected sets (v0.2 test and dev, external panel, robustness pack,
held-out generators, held-out templates) and recomputes, without the build script's code path: row counts, the source
and template counts, the augmentation counts by transform, the held-out absence, the date window, and the leakage counts
by text key and by dataset-qualified record id. Bare-id collisions that disappear under qualification are counted by
source. Train rows are checked against the v0.2 dev text too; dev rows are not (the dev split is drawn from it).
Writes outputs/osler_v0/O4/verify_mix.json (gitignored).

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o4_verify_mix.py
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from meddecide.mix.leakage import row_text_key
from meddecide.utils.io import write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "data/train/osler_v0/train.jsonl"
DEV = REPO / "data/train/osler_v0/dev.jsonl"
OUT = REPO / "outputs/osler_v0/O4/verify_mix.json"
WINDOW = "2026-03-01"
HELD_OUT_TEMPLATES = {"ct_arm_role_noul_v1", "ct_phase_choice_v1", "fda_boxed_warning_noul_v1", "pubmed_humans_noul_v1"}
HELD_OUT_GENERATORS = {"note_lab_range_v1", "policy_triage_v1"}
TEXT_NAMES = ("v02_test", "v02_dev", "panel", "robustness", "heldout_generators", "v02_heldout_templates")
QUAL_NAMES = ("v02_test", "panel", "robustness")


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def protected() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Text keys and dataset-qualified record keys of the protected sets."""
    text: dict[str, set[str]] = {k: set() for k in TEXT_NAMES}
    qual: dict[str, set[str]] = {k: set() for k in QUAL_NAMES}
    for path in sorted((REPO / "data/bench/v0.2/fresh").glob("*.jsonl")) + sorted(
            (REPO / "data/bench/v0.2/tier1").glob("*.jsonl")):
        for rec in rows(path):
            if rec["template_id"] in HELD_OUT_TEMPLATES:
                text["v02_heldout_templates"].add(row_text_key(rec))
            if rec["split"] == "test":
                text["v02_test"].add(row_text_key(rec))
                qual["v02_test"].add(f"{rec['source']}/{rec['source_record_id']}")
            if rec["split"] == "dev":
                text["v02_dev"].add(row_text_key(rec))
    for name, path in (("panel", REPO / "data/bench/v0.3_ext/panel.jsonl"),
                       ("robustness", REPO / "data/bench/v0.3_ext/robustness.jsonl")):
        for rec in rows(path):
            text[name].add(row_text_key(rec))
            qual[name].add(f"{rec['set_name']}/{rec['source_record_id']}")
    for path in (REPO / "data/gen/osler_v0/dev.jsonl", REPO / "data/gen/osler_v0/test.jsonl"):
        for rec in rows(path):
            if rec["held_out"]:
                state_row = {"state": rec["state"], "question": rec["question"],
                             "options": [{"label": o["label"]} for o in rec["options"]]}
                text["heldout_generators"].add(row_text_key(state_row))
    return text, qual


def main() -> None:
    text_sets, qual_sets = protected()
    bare_sets = {name: {value.partition("/")[2] for value in values} for name, values in qual_sets.items()}
    dev_text_sets = {k: v for k, v in text_sets.items() if k != "v02_dev"}
    report: dict[str, Any] = {"computed_at_utc": utcnow(), "protected_set_sizes": {
        **{f"text:{k}": len(v) for k, v in text_sets.items()},
        **{f"record_qualified:{k}": len(v) for k, v in qual_sets.items()}}}
    for label, path in (("train", TRAIN), ("dev", DEV)):
        active_text = text_sets if label == "train" else dev_text_sets
        n = 0
        by_source: Counter = Counter()
        by_template: Counter = Counter()
        by_transform: Counter = Counter()
        dates: Counter = Counter()
        held_rows = 0
        text_hits: Counter = Counter()
        qual_hits: Counter = Counter()
        bare_hits: Counter = Counter()
        bare_by_source: Counter = Counter()
        for row in rows(path):
            n += 1
            by_source[row["source"]] += 1
            by_template[row["template_id"]] += 1
            transform = row["meta"].get("augmentation")
            if transform:
                by_transform[transform] += 1
            if not row["record_date"]:
                dates["undated"] += 1
            elif row["record_date"] >= WINDOW:
                dates["in_window"] += 1
            else:
                dates["pre_window"] += 1
            if row["template_id"] in HELD_OUT_TEMPLATES or (
                    row["source"] == "generated" and row["template_id"] in HELD_OUT_GENERATORS):
                held_rows += 1
            key = row_text_key(row)
            for name, keys in active_text.items():
                if key in keys:
                    text_hits[name] += 1
            qualified = f"{row['source']}/{row['source_record_id']}"
            bare = str(row["source_record_id"])
            for name, keys in qual_sets.items():
                if qualified in keys:
                    qual_hits[name] += 1
                elif bare in bare_sets[name]:
                    bare_hits[name] += 1
                    bare_by_source[f"{name}:{row['source']}"] += 1
        report[label] = {
            "rows": n,
            "by_source": dict(by_source),
            "pubmed_pubtype_choice_v1_rows": by_template["pubmed_pubtype_choice_v1"],
            "max_template_share": round(max(by_template.values()) / n, 4),
            "augmented_by_transform": dict(by_transform),
            "augmented_total": sum(by_transform.values()),
            "held_out_rows": held_rows,
            "dates": dict(dates),
            "text_hits": {f"text:{k}": text_hits.get(k, 0) for k in active_text},
            "qualified_record_hits": {f"record_qualified:{k}": qual_hits.get(k, 0) for k in qual_sets},
            "bare_record_id_collisions_not_same_dataset": {f"record_bare:{k}": bare_hits.get(k, 0)
                                                           for k in qual_sets},
            "bare_collisions_by_protected_set_and_source": dict(bare_by_source),
        }
        print(f"{label}: rows {n}; text hits {sum(text_hits.values())}; qualified hits {sum(qual_hits.values())}; "
              f"bare collisions {sum(bare_hits.values())}; held-out rows {held_rows}", flush=True)
    write_json(OUT, report)


if __name__ == "__main__":
    main()
