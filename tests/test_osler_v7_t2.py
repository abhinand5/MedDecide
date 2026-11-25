"""osler_v7 T2 set-builder helpers on synthetic rows (CPU; no real data).

The script is loaded by path (the repo's pattern for loop scripts); every row below is invented, so the tests never
read the v6 mix, the fresh build or any MIMIC-derived file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "osler_v7" / "t2_build_sets.py"
spec = importlib.util.spec_from_file_location("t2_build_sets", SCRIPT)
assert spec is not None and spec.loader is not None
t2 = importlib.util.module_from_spec(spec)
sys.modules["t2_build_sets"] = t2
spec.loader.exec_module(t2)


def _row(
    item_id: str,
    template_id: str,
    qtype: str,
    keys: list[str],
    *,
    source: str = "medqa",
    split: str = "dev",
    state: str = "state",
    question: str = "question",
) -> dict[str, object]:
    return {
        "item_id": item_id,
        "template_id": template_id,
        "source": source,
        "qtype": qtype,
        "options": json.dumps([{"key": key, "label": key.upper()} for key in keys]),
        "gold": keys[0],
        "state": state,
        "question": question,
        "split": split,
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_normalise_and_option_count() -> None:
    assert t2.normalise("  a  b\n c ") == "a b c"
    assert t2.normalise(None) == ""
    assert t2.option_count({"options": '[{"key": "A"}, {"key": "B"}]'}) == 2
    assert t2.option_count({"options": [{"key": "A"}, {"key": "B"}, {"key": "C"}]}) == 3


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("abc", encoding="utf-8")
    assert t2.sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_check_overlap_detects_item_id_and_texts() -> None:
    row = _row("i1", "medqa_usmle4_v1", "choice", ["A", "B", "C", "D"], state="The state", question="The question?")
    overlap = {
        "item_ids": {"i1"},
        "item_texts": {hashlib.sha256(b"The state\x1fThe question?").digest()},
        "record_texts": {hashlib.sha256(b"The state").digest()},
    }
    counts = t2.check_overlap([row], overlap)
    assert counts == {"item_id": 1, "item_text": 1, "record_text": 1}
    assert t2.check_overlap([_row("i2", "x", "noul", ["yes", "no"])], overlap) == {}


def test_build_dev_excludes_synthetic_templates_and_dedupes(tmp_path: Path) -> None:
    fast = tmp_path / "fast.jsonl"
    cg = tmp_path / "cg.jsonl"
    know = tmp_path / "know.jsonl"
    _write(fast, [
        _row("keep1", "medqa_usmle4_v1", "choice", ["A", "B", "C", "D"]),
        _row("drop_synth", "synthea_page_category_v1", "choice", ["A", "B"], source="mr_synthea"),
        _row("drop_gen", "ct_randomised_noul_v1", "noul", ["yes", "no"], source="generated"),
    ])
    _write(cg, [
        _row("keep1", "medqa_usmle4_v1", "choice", ["A", "B", "C", "D"]),  # duplicate of fast
        _row("keep2", "cg_v2_ddxplus_validate_v1", "choice", [str(i) for i in range(49)]),
    ])
    _write(know, [_row("keep3", "medmcqa_4opt_v1", "choice", ["A", "B", "C", "D"])])
    rows, report = t2.build_dev({"fastdev_v4": fast, "cg_v2": cg, "knowledge_dev_v2": know})
    ids = [row["item_id"] for row in rows]
    assert ids == ["keep1", "keep2", "keep3"]
    assert report["duplicates_removed"].get("cg_v2") == 1
    assert rows[1]["dev_origin"] == "cg_v2"
    assert rows[1]["_n_options"] == 49
    comp = t2.dev_composition(rows)
    assert comp["rows"] == 3
    assert comp["shapes"]["many_option_10_50"]["items"] == 1
    assert comp["shapes"]["two_option_pairs"]["items"] == 0  # no noul kept here
    assert comp["all_shapes_present"] is False


def test_build_confirmation_general_keeps_test_split_and_caps(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "manifest.json").write_text(json.dumps({"window_start": "2026-10-06"}), encoding="utf-8")
    (fresh / "acceptance.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
    rows = [
        {**_row(f"c{i}", "ct_phase_choice_v1", "choice", ["A", "B"], source="clinicaltrials", split="test",
               state=f"s{i}"), "record_date": "2026-10-20", "source_record_id": f"rec{i}"}
        for i in range(t2.PER_TEMPLATE_CAP + 5)
    ]
    rows.append({**_row("dev1", "ct_phase_choice_v1", "choice", ["A", "B"], source="clinicaltrials", split="dev"),
                 "record_date": "2026-10-21", "source_record_id": "rec_dev"})
    _write(fresh / "clinicaltrials.jsonl", rows)
    _write(fresh / "openfda.jsonl", [])
    _write(fresh / "pubmed.jsonl", [])
    overlap = {"item_ids": set(), "item_texts": set(), "record_texts": set()}
    out, report = t2.build_confirmation_general(overlap, fresh_dir=fresh)
    assert len(out) == t2.PER_TEMPLATE_CAP
    assert report["sources"]["clinicaltrials"]["test_rows"] == t2.PER_TEMPLATE_CAP + 5
    assert report["composition"]["rows_before_cap"] == t2.PER_TEMPLATE_CAP + 5
    assert all(row["confirm_component"] == "general_fresh" for row in out)


def test_build_confirmation_general_rejects_a_row_outside_the_window(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "manifest.json").write_text(json.dumps({"window_start": "2026-10-06"}), encoding="utf-8")
    (fresh / "acceptance.json").write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
    rows = [{**_row("old", "ct_phase_choice_v1", "choice", ["A", "B"], source="clinicaltrials", split="test"),
             "record_date": "2026-09-01", "source_record_id": "rec_old"}]
    _write(fresh / "clinicaltrials.jsonl", rows)
    _write(fresh / "openfda.jsonl", [])
    _write(fresh / "pubmed.jsonl", [])
    overlap = {"item_ids": set(), "item_texts": set(), "record_texts": set()}
    with pytest.raises(SystemExit):
        t2.build_confirmation_general(overlap, fresh_dir=fresh)
