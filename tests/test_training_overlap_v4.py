"""The training-overlap check (osler_v4 Y4): text and record matches, and dev files are never training input. Synthetic only."""

from __future__ import annotations

import json
from pathlib import Path

from meddecide.panel import training_overlap as to


def _row(state: str, record: str, question: str = "Which is right?") -> dict:
    return {"state": state, "question": question, "options": [{"key": "A", "label": "yes"}, {"key": "B", "label": "no"}],
            "source_record_id": record}


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_dev_files_are_not_training_input(tmp_path: Path) -> None:
    root = tmp_path / "train"
    _write(root / "osler_v1" / "block_B1.jsonl", [_row("a", "r1")])
    _write(root / "osler_v0" / "dev.jsonl", [_row("b", "r2")])
    assert [p.name for p in to.training_files(root)] == ["block_B1.jsonl"]


def test_a_candidate_with_the_same_text_or_the_same_record_is_a_match(tmp_path: Path) -> None:
    train = _write(tmp_path / "block.jsonl", [_row("same text here", "other-record"),
                                                _row("different", "shared-record", question="Other question?")])
    candidates = [_row("same text here", "x1"), _row("candidate", "shared-record", question="Mine?"),
                  _row("untouched", "x3", question="Not in training?")]
    for i, c in enumerate(candidates):
        c["item_id"] = f"c{i}"
    found = to.match_training(candidates, [train])
    assert found["c0"].text and not found["c0"].record
    assert found["c1"].record and not found["c1"].text
    assert "c2" not in found


def test_a_text_match_needs_the_same_normalised_question(tmp_path: Path) -> None:
    train = _write(tmp_path / "block.jsonl", [_row("same", "r9", question="Question A?")])
    candidate = _row("same", "zz", question="Question B?")
    candidate["item_id"] = "c"
    assert "c" not in to.match_training([candidate], [train])
