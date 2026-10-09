#!/usr/bin/env python
"""T2 acceptance check: schema + JSONL helpers + manifest builder on synthetic items.

Demonstrates the published constructor rejects gold outside the options and score items
with fewer than two levels, that ids are deterministic, that a JSONL round-trip preserves
items, and that the manifest counts close — and writes the evidence to
``outputs/bench_v0/T2/acceptance.json``.

This uses **synthetic** items only: T2 defines the format, it does not load real data
(that is T3/T5). No item text from a real source is written anywhere.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from meddecide.bench.schema import (
    Item,
    Option,
    QuestionType,
    Split,
    Tier,
    compute_item_id,
    make_item,
)
from meddecide.utils.io import (
    build_manifest,
    check_no_record_crosses_splits,
    read_jsonl,
    write_jsonl,
)

SYNTHETIC_STATE = "Synthetic state text used only to exercise the schema (not a real record)."
SYNTHETIC_QUESTION = "Which option is correct?"


def _synthetic(record_id: str, *, split: Split, template: str, seed: int, qtype=QuestionType.CHOICE):
    options = (
        [Option(key="yes", label="Yes"), Option(key="no", label="No")]
        if qtype is QuestionType.NOUL
        else [Option(key="A", label="Option one"), Option(key="B", label="Option two")]
    )
    return make_item(
        tier=Tier.FRESH,
        source="synthetic",
        source_record_id=record_id,
        source_url=f"https://example.org/record/{record_id}",
        source_license="synthetic",
        record_date=date(2026, 1, 1),
        split=split,
        template_id=template,
        skill="synthetic_check",
        qtype=qtype,
        state=SYNTHETIC_STATE,
        question=SYNTHETIC_QUESTION,
        options=options,
        gold=options[0].key,
        option_order_seed=seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/bench_v0/T2"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {}

    # 1. validation failures are rejected
    failures: dict[str, str] = {}
    good_payload = _synthetic("r1", split=Split.TEST, template="t1", seed=1).model_dump()
    try:
        # gold outside the options: mutate the payload, not the model
        Item.model_validate({**good_payload, "gold": "Z"})
        failures["gold_not_in_options"] = "NOT REJECTED"
    except ValidationError as exc:
        failures["gold_not_in_options"] = str(exc).splitlines()[1].strip()

    try:
        make_item(
            tier=Tier.FRESH, source="synthetic", source_record_id="r2",
            source_url="https://example.org/record/r2", source_license="synthetic",
            record_date=date(2026, 1, 1), split=Split.TEST, template_id="t_score",
            skill="synthetic_check", qtype=QuestionType.SCORE,
            state=SYNTHETIC_STATE, question=SYNTHETIC_QUESTION,
            options=[Option(key="1", label="only level")], gold="1", option_order_seed=1,
        )
        failures["score_single_level"] = "NOT REJECTED"
    except ValidationError as exc:
        failures["score_single_level"] = str(exc).splitlines()[1].strip()

    results["validation_failures"] = failures

    # 2. deterministic ids
    a = _synthetic("r1", split=Split.TEST, template="t1", seed=1)
    b = _synthetic("r1", split=Split.TEST, template="t1", seed=1)
    c = _synthetic("r1", split=Split.TEST, template="t1", seed=2)
    results["deterministic_ids"] = {
        "same_inputs_same_id": a.item_id == b.item_id,
        "different_seed_different_id": a.item_id != c.item_id,
        "matches_compute_item_id": a.item_id == compute_item_id("synthetic", "r1", "t1", 1, "test"),
        "id_length": len(a.item_id),
    }

    # 3. JSONL round-trip
    rows = [
        _synthetic("r1", split=Split.TEST, template="t1", seed=1),
        _synthetic("r2", split=Split.TEST, template="t1", seed=1),
        _synthetic("r3", split=Split.DEV, template="t2", seed=1, qtype=QuestionType.NOUL),
    ]
    items_path = write_jsonl(args.out / "synthetic_items.jsonl", rows)
    loaded, report = read_jsonl(items_path, type(rows[0]))
    results["round_trip"] = {
        "n_written": len(rows),
        "n_loaded": len(loaded),
        "identical": [x.item_id for x in loaded] == [x.item_id for x in rows],
        "load_report": report.to_dict(),
    }

    # 4. manifest closes, and split disjointness holds
    manifest = build_manifest({"synthetic_items": items_path}, {"synthetic_items": loaded})
    entry = manifest["files"]["synthetic_items"]
    results["manifest"] = {
        "n_rows": entry["n_rows"],
        "sha256": entry["sha256"],
        "count_by_split": entry["count_by_split"],
        "count_by_qtype": entry["count_by_qtype"],
        "counts_close": sum(entry["count_by_split"].values()) == entry["n_rows"]
        and sum(entry["count_by_qtype"].values()) == entry["n_rows"],
    }
    results["split_disjointness"] = check_no_record_crosses_splits(loaded)

    # 5. arithmetic: parts sum to the whole
    results["checks"] = {
        "round_trip_complete": len(loaded) == len(rows),
        "manifest_closes": bool(results["manifest"]["counts_close"]),
        "no_split_leaks": bool(results["split_disjointness"]["ok"]),
        "all_validation_failures_rejected": all(
            v != "NOT REJECTED" for v in failures.values()
        ),
        "ids_deterministic": bool(results["deterministic_ids"]["same_inputs_same_id"]),
    }
    results["verdict"] = "PASS" if all(results["checks"].values()) else "FAIL"

    out = args.out / "acceptance.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results["checks"], indent=2))
    print(f"verdict: {results['verdict']} -> {out}")
    return 0 if results["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
