"""The O11 leakage check's counts (ADVISORY failure mode 5): a training row matches a v0.2 test item or a panel item by id or text (CPU)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("o11_leakage_check", REPO / "scripts/osler/o11_leakage_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _item(item_id: str, state: str, question: str) -> dict:
    return {"item_id": item_id, "state": state, "question": question}


def test_a_training_row_matching_a_test_item_by_id_or_by_text_is_counted() -> None:
    module = _module()
    test = [_item("a" * 16, "A patient with a fever.", "Is the diagnosis influenza?")]
    panel = [_item("p" * 16, "A panel vignette.", "Which option?")]
    train = [
        _item("a" * 16, "unrelated state", "unrelated question"),  # same id as a test item
        _item("b" * 16, "A patient with a fever.", "Is the diagnosis influenza?"),  # same text as a test item
        _item("p" * 16, "unrelated state", "unrelated question"),  # same id as a panel item
        _item("c" * 16, "A panel vignette.", "Which option?"),  # same text as a panel item
        _item("d" * 16, "unrelated state", "unrelated question"),
    ]
    counts = module.overlaps(train, test, panel)
    assert counts == {"train_rows": 5, "v02_test_id": 1, "v02_test_text": 1, "panel_id": 1, "panel_text": 1}


def test_no_overlap_gives_zero_counts() -> None:
    module = _module()
    counts = module.overlaps([_item("d" * 16, "x", "y")], [_item("a" * 16, "p", "q")], [_item("p" * 16, "r", "s")])
    assert counts == {"train_rows": 1, "v02_test_id": 0, "v02_test_text": 0, "panel_id": 0, "panel_text": 0}
