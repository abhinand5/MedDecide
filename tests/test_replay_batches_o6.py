"""The CPU batch replay's log reader (scripts/osler/o6_replay_batches.py): a half-written last line is skipped, a broken line in
the middle is an error."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _replay_module():
    spec = importlib.util.spec_from_file_location("o6_replay_batches", REPO / "scripts/osler/o6_replay_batches.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _line(**row) -> str:
    return json.dumps(row)


def test_logged_step_sizes_read_every_step_record(tmp_path: Path) -> None:
    log = tmp_path / "train.jsonl"
    log.write_text("\n".join([_line(event="step", step=50, n_items=8), _line(event="eval", step=50),
                              _line(event="step", step=100, n_items=3)]) + "\n", encoding="utf-8")
    assert _replay_module().logged_step_sizes(log) == {50: 8, 100: 3}


def test_a_half_written_last_line_is_skipped(tmp_path: Path) -> None:
    log = tmp_path / "train.jsonl"
    log.write_text(_line(event="step", step=50, n_items=8) + "\n" + '{"event": "step", "step": 100, "n_it', encoding="utf-8")
    assert _replay_module().logged_step_sizes(log) == {50: 8}


def test_a_broken_line_in_the_middle_is_an_error(tmp_path: Path) -> None:
    log = tmp_path / "train.jsonl"
    log.write_text("not json\n" + _line(event="step", step=50, n_items=8) + "\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _replay_module().logged_step_sizes(log)
