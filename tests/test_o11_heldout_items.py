"""The O11 held-out generator rows (ADVISORY O11; heldout.md rule 3): the O2-format conversion and the held-out checks (CPU)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from meddecide.bench.schema import QuestionType
from meddecide.eval.osler_scoring import o2_row_to_item

REPO = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("o11_heldout_items", REPO / "scripts/osler/o11_heldout_items.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rec(item_id: str = "a" * 16, generator: str = "note_lab_range_v1", qtype: str = "choice", held_out: bool = True,
         split: str = "test") -> dict:
    options = [{"key": "A", "label": "within"}, {"key": "B", "label": "below"}, {"key": "C", "label": "above"}]
    if qtype == "score":
        options = [{"key": "1", "label": "level 1"}, {"key": "2", "label": "level 2"}, {"key": "3", "label": "level 3"}]
    return {
        "item_id": item_id, "generator": generator, "family": "A_note_facts", "pair_id": "p-1", "role": "base",
        "split": split, "qtype": qtype, "state": "Lab: sodium 138 mmol/L (reference 135 to 145).",
        "question": "Is the value within the reference range?", "options": options, "gold": "2" if qtype == "score" else "B",
        "meta": {"condition": "sodium"}, "held_out": held_out, "patient_id": "p-1",
    }


def test_a_held_out_choice_item_converts_with_its_generator_as_the_template() -> None:
    module = _module()
    item = o2_row_to_item(module.to_o2_row(_rec()))
    assert item.qtype is QuestionType.CHOICE
    assert item.template_id == "note_lab_range_v1"
    assert item.option_keys == ["A", "B", "C"] and item.gold == "B"
    assert item.item_id == "a" * 16


def test_a_held_out_score_item_converts() -> None:
    module = _module()
    item = o2_row_to_item(module.to_o2_row(_rec(generator="policy_triage_v1", qtype="score")))
    assert item.qtype is QuestionType.SCORE
    assert item.template_id == "policy_triage_v1" and item.option_keys == ["1", "2", "3"] and item.gold == "2"


def test_held_out_records_keeps_the_registry_generators_on_the_test_split(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "test.jsonl"
    rows = [_rec("a" * 16), _rec("b" * 16, generator="policy_triage_v1", qtype="score"),
            _rec("c" * 16, generator="note_negation_v1", held_out=False)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    held = module.held_out_records(path)
    assert [r["item_id"] for r in held] == ["a" * 16, "b" * 16]


def test_a_held_out_row_outside_the_registry_is_refused(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(_rec(generator="note_negation_v1")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="registry lists"):
        module.held_out_records(path)


def test_a_held_out_row_on_another_split_is_refused(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "test.jsonl"
    rows = [_rec("a" * 16), _rec("b" * 16, generator="policy_triage_v1", qtype="score", split="dev")]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not on the test split"):
        module.held_out_records(path)


def test_duplicated_held_out_ids_are_refused(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "test.jsonl"
    rows = [_rec("a" * 16), _rec("a" * 16, generator="policy_triage_v1", qtype="score")]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not unique"):
        module.held_out_records(path)
