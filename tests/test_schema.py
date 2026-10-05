"""Tests for the item schema: round-trip, deterministic ids, validation failures."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from meddecide.bench.schema import (
    Item,
    Option,
    QuestionType,
    Split,
    Tier,
    compute_item_id,
    make_item,
    split_by_record_hash,
)
from meddecide.utils.io import read_jsonl, write_jsonl


def _choice_item(**overrides):
    kwargs = {
        "tier": Tier.FRESH,
        "source": "clinicaltrials",
        "source_record_id": "NCT00000001",
        "source_url": "https://clinicaltrials.gov/study/NCT00000001",
        "source_license": "public-domain",
        "record_date": date(2026, 1, 2),
        "split": Split.TEST,
        "template_id": "ct_phase_v1",
        "skill": "trial_design",
        "qtype": QuestionType.CHOICE,
        "state": "A study of drug X in adults with condition Y.",
        "question": "What is the phase of this study?",
        "options": [
            Option(key="A", label="Phase 1"),
            Option(key="B", label="Phase 2"),
            Option(key="C", label="Phase 3"),
        ],
        "gold": "B",
        "option_order_seed": 7,
    }
    kwargs.update(overrides)
    return make_item(**kwargs)


def test_round_trip_jsonl(tmp_path) -> None:
    item = _choice_item()
    path = write_jsonl(tmp_path / "items.jsonl", [item])
    rows, report = read_jsonl(path, Item)
    assert report.n_lines == 1 and report.n_kept == 1 and report.n_dropped == 0
    assert rows[0] == item


def test_item_id_is_deterministic_and_content_sensitive() -> None:
    a = _choice_item()
    b = _choice_item()
    assert a.item_id == b.item_id
    assert a.item_id == compute_item_id("clinicaltrials", "NCT00000001", "ct_phase_v1", 7)
    assert len(a.item_id) == 16

    # different option order seed -> different id for the same source record + template
    assert _choice_item(option_order_seed=8).item_id != a.item_id
    # different source record -> different id
    assert _choice_item(source_record_id="NCT00000002").item_id != a.item_id


def test_item_id_ignores_dict_order_and_whitespace() -> None:
    assert compute_item_id("a b", "X", "t", 1) == compute_item_id("a  b", "X", "t", 1)


def test_gold_must_be_an_option() -> None:
    with pytest.raises(ValidationError, match="gold"):
        _choice_item(gold="Z")


def test_choice_needs_at_least_two_options() -> None:
    with pytest.raises(ValidationError):
        _choice_item(options=[Option(key="A", label="only")], gold="A")


def test_score_needs_two_levels_and_ordered_keys() -> None:
    common = {
        "qtype": QuestionType.SCORE,
        "options": [Option(key="1", label="low"), Option(key="2", label="high")],
        "gold": "2",
    }
    item = _choice_item(**common)
    assert item.n_options == 2 and item.gold_index == 1

    # 1 level: rejected by the field-level 2..255 bound, before the score-specific check
    with pytest.raises(ValidationError, match="at least 2 items"):
        _choice_item(**{**common, "options": [Option(key="1", label="only")], "gold": "1"})
    # levels must ascend 1..N
    with pytest.raises(ValidationError, match="score"):
        _choice_item(
            **{
                **common,
                "options": [Option(key="2", label="high"), Option(key="1", label="low")],
            }
        )


def test_score_rejects_more_than_ten_levels() -> None:
    options = [Option(key=str(i), label=f"level {i}") for i in range(1, 12)]
    with pytest.raises(ValidationError, match="score"):
        _choice_item(qtype=QuestionType.SCORE, options=options, gold="1")


def test_noul_option_keys_are_fixed() -> None:
    item = _choice_item(
        qtype=QuestionType.NOUL,
        options=[Option(key="yes", label="Yes"), Option(key="no", label="No")],
        gold="yes",
    )
    assert item.gold_index == 0
    with pytest.raises(ValidationError, match="noul"):
        _choice_item(
            qtype=QuestionType.NOUL,
            options=[Option(key="A", label="Yes"), Option(key="B", label="No")],
            gold="A",
        )
    with pytest.raises(ValidationError, match="noul"):
        _choice_item(
            qtype=QuestionType.NOUL,
            options=[Option(key="yes", label="Yes"), Option(key="no", label="No")],
            gold="maybe",
        )


def test_duplicate_option_keys_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        _choice_item(
            options=[Option(key="A", label="x"), Option(key="A", label="y")],
            gold="A",
        )


def test_source_url_must_be_http() -> None:
    with pytest.raises(ValidationError, match="source_url"):
        _choice_item(source_url="ftp://example.org/x")


def test_extra_fields_rejected() -> None:
    payload = _choice_item().model_dump(mode="json")
    payload["secret_field"] = 1
    with pytest.raises(ValidationError):
        Item.model_validate(payload)


def test_split_by_record_hash_is_deterministic_and_by_record() -> None:
    first = split_by_record_hash("NCT00000001")
    assert first == split_by_record_hash("NCT00000001")
    assert first in (Split.DEV, Split.TEST)
    # a record must never be split across dev and test
    assert len({split_by_record_hash(f"NCT{i:08d}") for i in range(200)}) == 2


def test_read_jsonl_accounts_for_every_line(tmp_path) -> None:
    """Every non-blank line is accounted for: 2 kept, 1 schema failure, 1 bad JSON.

    Corrected during T1: the first draft asserted 3 non-blank lines and 1 drop, which was
    simply wrong (the file has 4 non-blank lines). The reader was right; the expectation
    was fixed to match the file, not the other way round.
    """
    path = tmp_path / "mixed.jsonl"
    good = _choice_item().model_dump_json()
    path.write_text(
        good + "\n" + '{"item_id": "short"}\n' + "{not json}\n" + good + "\n",
        encoding="utf-8",
    )
    rows, report = read_jsonl(path, Item)
    assert report.n_lines == 4, "blank lines are not counted as records"
    assert report.n_kept == 2 and rows[0] == rows[1]
    assert report.n_dropped == 2
    assert report.dropped["schema_validation"] == 1
    assert report.dropped["invalid_json"] == 1
    report.check_closes()
