"""Y9 (osler_v4 ADVISORY Y9; D49; STATE §7 Y9 pre-registration): the Gate O4 evaluation sets, built from the frozen sources (CPU).

The V10 verdict file (data/bench/osler_v1/v10_items.jsonl) holds the MR test set (19,294 items: seven verdict templates and two
reported-only templates), the MR held-out rows, the MedQA and MedMCQA test items (the knowledge set, 5,456 items) and the panel. This
script writes the MR, MR held-out, knowledge and panel sets under data/bench/osler_v4/y9/ (gitignored; the item text stays private).
The panel and knowledge rows carry no `source` field (they have `benchmark` and `set_name`), which the scorers' reader requires; the
copies add `source` = `set_name` and change nothing else. The MIMIC audit suite, the MIMIC held-out file, MedNLI and the Y3 pair test are
copied with `benchmark` (`v0.2`) and `set_name` added, which the JEV-9B runner needs. Each copy's source file and hash are in the manifest. Every row is checked against the item schema the
scorers use (meddecide.train.mix_items.mix_row_to_item); a row that fails is counted, and any count mismatch with the frozen figures
stops the script.

Writes outputs/osler_v4/Y9/sets_manifest.json (counts, template counts, option counts, hashes; no item text).

Usage (CPU):
    uv run --frozen python scripts/osler_v4/y9_sets.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import iter_jsonl, read_json, write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
V10 = REPO / "data" / "bench" / "osler_v1" / "v10_items.jsonl"
MR_MANIFEST = REPO / "data" / "bench" / "osler_mr_v1" / "manifest.json"
MIMIC_SUITE = REPO / "data" / "bench" / "osler_mimic_v1" / "mimic_audit_suite_v1.jsonl"
MIMIC_HELDOUT = REPO / "data" / "bench" / "osler_mimic_v1" / "mimic_heldout_v1.jsonl"
PANEL = REPO / "data" / "bench" / "v0.3_ext" / "panel.jsonl"
MEDNLI_TEST = REPO / "data" / "mimic_build" / "items" / "mednli_v1__test.jsonl"
PAIRS_TEST = REPO / "data" / "mimic_build" / "pairs" / "build_off" / "rows_test.jsonl"
OUT_DATA = REPO / "data" / "bench" / "osler_v4" / "y9"
OUT_MANIFEST = REPO / "outputs" / "osler_v4" / "Y9" / "sets_manifest.json"
OPTION_CAP = 10  # the MedDecider prompt letters A to J (STATE §7, reading 3)

# The frozen counts (STATE §2, the MR manifest, D44, the Y4 record). Any mismatch stops the script.
EXPECTED = {
    "mr_verdict": 13_103, "mr_reported": 6_191, "mr_heldout_test": 584, "knowledge": 5_456,
    "panel": 4_353, "mimic_suite": 4_253, "mimic_heldout": 110, "mednli_test": 1_422, "pairs_test": 500,
}


def summarise(name: str, path: Path, rows: list[dict[str, Any]], failures: int) -> dict[str, Any]:
    templates = collections.Counter(str(r["template_id"]) for r in rows)
    options = collections.Counter(len(r["options"]) for r in rows)
    return {
        "file": str(path.relative_to(REPO)), "sha256": file_sha256(path), "rows": len(rows),
        "expected_rows": EXPECTED[name], "schema_failures": failures,
        "templates": dict(sorted(templates.items())), "option_counts": {str(k): v for k, v in sorted(options.items())},
        "more_than_ten_options": sum(v for k, v in options.items() if k > OPTION_CAP),
    }


def check_schema(rows: list[dict[str, Any]]) -> int:
    failures = 0
    for row in rows:
        try:
            mix_row_to_item(row)
        except (ValueError, KeyError, TypeError):
            failures += 1
    return failures


def with_benchmark(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """The rows with ``benchmark`` (``v0.2``, the all-sets scope of the O2 runners) and ``set_name`` where the row has none.

    The JEV-9B runner (meddecide.eval.o2_io) selects rows by ``benchmark``; the MIMIC, MedNLI and pair rows carry neither key.
    """
    return [{**r, "benchmark": r.get("benchmark", "v0.2"), "set_name": r.get("set_name", name)} for r in rows]


def with_source(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows with ``source`` set to ``set_name`` where the row has no source (the V10 panel and knowledge rows)."""
    return [r if "source" in r else {**r, "source": r["set_name"]} for r in rows]


def write_subset(name: str, rows: list[dict[str, Any]], source_file: Path | None = None) -> dict[str, Any]:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    path = OUT_DATA / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarise(name, path, rows, check_schema(rows))
    if source_file is not None:
        summary["copied_from"] = str(source_file.relative_to(REPO))
        summary["copied_from_sha256"] = file_sha256(source_file)
    return summary


def main() -> None:
    mr = read_json(MR_MANIFEST)
    verdict_templates = set(mr["verdict_templates"])
    reported_templates = set(mr["reported_only_templates"])
    if len(verdict_templates) != 7 or len(reported_templates) != 2:
        raise SystemExit("the MR manifest does not list seven verdict and two reported-only templates")
    v10 = list(iter_jsonl(V10))
    mr_verdict = [r for r in v10 if r.get("set_name") == "mr_test" and r["template_id"] in verdict_templates]
    mr_reported = [r for r in v10 if r.get("set_name") == "mr_test" and r["template_id"] in reported_templates]
    mr_heldout_test = [r for r in v10 if r.get("set_name") == "mr_heldout" and r.get("split") == "test"]
    knowledge = [r for r in v10 if r.get("set_name") in ("medqa", "medmcqa")]
    if {r["template_id"] for r in knowledge} != {"medqa_usmle4_v1", "medmcqa_4opt_v1"}:
        raise SystemExit("the knowledge rows are not the MedQA and MedMCQA templates")
    if len(mr_verdict) + len(mr_reported) != mr["test_items"]["total"]:
        raise SystemExit("the MR test rows do not add to the frozen total")
    summaries = {
        "mr_verdict": write_subset("mr_verdict", mr_verdict),
        "mr_reported": write_subset("mr_reported", mr_reported),
        "mr_heldout_test": write_subset("mr_heldout_test", mr_heldout_test),
        "knowledge": write_subset("knowledge", with_source(knowledge)),
        "panel": write_subset("panel", with_source(list(iter_jsonl(PANEL))), source_file=PANEL),
    }
    copies = {
        "mimic_suite": (MIMIC_SUITE, list(iter_jsonl(MIMIC_SUITE))),
        "mimic_heldout": (MIMIC_HELDOUT, list(iter_jsonl(MIMIC_HELDOUT))),
        "mednli_test": (MEDNLI_TEST, list(iter_jsonl(MEDNLI_TEST))),
        "pairs_test": (PAIRS_TEST, list(iter_jsonl(PAIRS_TEST))),
    }
    for name, (path, rows) in copies.items():
        summaries[name] = write_subset(name, with_benchmark(rows, name), source_file=path)
    bad = {name: s["rows"] for name, s in summaries.items() if s["rows"] != s["expected_rows"]}
    failed = {name: s["schema_failures"] for name, s in summaries.items() if s["schema_failures"]}
    if bad or failed:
        raise SystemExit(f"count or schema mismatch: counts {bad}, schema failures {failed}")
    panel_ids = {str(r["item_id"]) for r in iter_jsonl(OUT_DATA / "panel.jsonl")}
    v10_panel = {str(r["item_id"]) for r in v10 if r.get("benchmark") == "ext_panel"}
    if panel_ids != v10_panel:
        raise SystemExit("the panel copy and the V10 panel rows differ")
    write_json(OUT_MANIFEST, {"generated_utc": utcnow(), "option_cap_for_medDecider": OPTION_CAP,
                              "mr_verdict_templates": sorted(verdict_templates),
                              "mr_reported_templates": sorted(reported_templates), "sets": summaries})
    for name, s in summaries.items():
        print(f"{name}: {s['rows']} rows, {len(s['templates'])} templates, more than ten options {s['more_than_ten_options']}, "
              f"schema failures {s['schema_failures']}", flush=True)


if __name__ == "__main__":
    main()
