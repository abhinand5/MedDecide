"""Tests for the manifest builder and the no-silent-drops accounting."""

from __future__ import annotations

from datetime import date

import pytest

from meddecide.bench.schema import Option, QuestionType, Split, Tier, make_item
from meddecide.utils.io import build_manifest, read_jsonl, write_jsonl
from meddecide.utils.provenance import Provenance


def _item(record_id: str, *, split=Split.TEST, template="t1", qtype=QuestionType.CHOICE, source="src_a"):
    options = (
        [Option(key="yes", label="Yes"), Option(key="no", label="No")]
        if qtype is QuestionType.NOUL
        else [Option(key="A", label="one"), Option(key="B", label="two")]
    )
    return make_item(
        tier=Tier.FRESH,
        source=source,
        source_record_id=record_id,
        source_url=f"https://example.org/{record_id}",
        source_license="public-domain",
        record_date=date(2026, 1, 1),
        split=split,
        template_id=template,
        skill="trial_design",
        qtype=qtype,
        state="state text",
        question="question text?",
        options=options,
        gold=options[0].key,
        option_order_seed=1,
    )


def test_manifest_counts_close_and_hash_is_recorded(tmp_path) -> None:
    rows = [
        _item("r1", split=Split.TEST, template="t1"),
        _item("r2", split=Split.TEST, template="t1"),
        _item("r3", split=Split.DEV, template="t2", source="src_b"),
        _item("r4", split=Split.DEV, template="t2", qtype=QuestionType.NOUL, source="src_b"),
    ]
    path = write_jsonl(tmp_path / "fresh.jsonl", rows)
    loaded, report = read_jsonl(path, type(rows[0]))
    assert report.n_kept == 4

    manifest = build_manifest({"fresh": path}, {"fresh": loaded})
    entry = manifest["files"]["fresh"]
    assert entry["exists"] is True
    assert len(entry["sha256"]) == 64
    assert entry["n_rows"] == 4

    # every breakdown sums to the file total (R5)
    assert sum(entry["count_by_source"].values()) == 4
    assert sum(entry["count_by_split"].values()) == 4
    assert sum(entry["count_by_qtype"].values()) == 4
    assert entry["count_by_split"] == {"dev": 2, "test": 2}
    assert entry["count_by_source"] == {"src_a": 2, "src_b": 2}
    assert entry["count_by_qtype"] == {"choice": 3, "noul": 1}

    # nested source -> split -> qtype -> template counts
    counts = manifest["counts"]["fresh"]
    assert set(counts) == {"src_a", "src_b"}
    assert counts["src_a"]["test"]["choice"]["t1"] == 2
    assert counts["src_b"]["dev"]["choice"]["t2"] == 1
    assert counts["src_b"]["dev"]["noul"]["t2"] == 1
    assert manifest["totals"]["fresh"] == 4


def test_manifest_reports_missing_file_honestly(tmp_path) -> None:
    """R2: a manifest entry for a file that does not exist says so, with sha256 None."""
    missing = tmp_path / "nope.jsonl"
    manifest = build_manifest({"missing": missing}, {"missing": []})
    assert manifest["files"]["missing"]["exists"] is False
    assert manifest["files"]["missing"]["sha256"] is None
    assert manifest["files"]["missing"]["n_rows"] == 0


def test_manifest_hash_matches_file_bytes(tmp_path) -> None:
    from meddecide.utils.hashing import file_sha256

    rows = [_item("r1")]
    path = write_jsonl(tmp_path / "x.jsonl", rows)
    manifest = build_manifest({"x": path}, {"x": rows})
    assert manifest["files"]["x"]["sha256"] == file_sha256(path)


def test_write_jsonl_round_trip_is_byte_stable(tmp_path) -> None:
    rows = [_item("r1"), _item("r2")]
    p1 = write_jsonl(tmp_path / "a.jsonl", rows)
    p2 = write_jsonl(tmp_path / "b.jsonl", rows)
    assert p1.read_bytes() == p2.read_bytes()


def test_provenance_write_is_json_and_records_command(tmp_path) -> None:
    prov = Provenance(run_name="unit", command="uv run pytest").finish()
    out = prov.write(tmp_path / "prov.json")
    import json

    payload = json.loads(out.read_text())
    assert payload["run_name"] == "unit"
    assert payload["command"] == "uv run pytest"
    assert payload["finished_at"] and payload["wall_clock_s"] is not None
    assert "HF_TOKEN" not in out.read_text()


def test_stable_hash_normalises_unicode_and_key_order() -> None:
    from meddecide.utils.hashing import stable_hash

    # NFC vs NFD spelling of "é" must hash the same
    assert stable_hash({"a": "caf\u00e9"}) == stable_hash({"a": "cafe\u0301"})
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


@pytest.mark.parametrize("bad", ["", " ", "\t"])
def test_item_rejects_empty_state(bad: str) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        make_item(
            tier=Tier.FRESH,
            source="s",
            source_record_id="r",
            source_url="https://example.org/r",
            source_license="x",
            record_date=date(2026, 1, 1),
            split=Split.TEST,
            template_id="t",
            skill="sk",
            qtype=QuestionType.CHOICE,
            state=bad,
            question="q",
            options=[Option(key="A", label="a"), Option(key="B", label="b")],
            gold="A",
            option_order_seed=0,
        )


def test_no_record_crosses_splits_detects_a_leak() -> None:
    from meddecide.utils.io import check_no_record_crosses_splits

    clean = [_item("r1", split=Split.TEST), _item("r1", split=Split.TEST), _item("r2", split=Split.DEV)]
    report = check_no_record_crosses_splits(clean)
    assert report["ok"] is True
    assert report["n_rows"] == 3 and report["n_unique_records"] == 2
    assert report["n_leaking_records"] == 0

    leaky = [*clean, _item("r2", split=Split.TEST)]
    report = check_no_record_crosses_splits(leaky)
    assert report["ok"] is False
    assert report["n_leaking_records"] == 1
    assert report["leaks"][0]["source_record_id"] == "r2"
    assert report["leaks"][0]["splits"] == ["dev", "test"]
    assert report["leaks"][0]["n_items"] == 2
