"""The O11 scorer's pre-flight (ADVISORY O11): every in-scope row converts and every item id is unique (CPU; no model)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _scorer():
    spec = importlib.util.spec_from_file_location("o11_score", REPO / "scripts/osler/o11_score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _noul_row(item_id: str, keys: tuple[str, ...] = ("no", "yes")) -> dict:
    labels = {"yes": "Yes", "no": "No", "maybe": "Maybe"}
    return {
        "item_id": item_id, "benchmark": "robustness", "set_name": "pubmed", "template_id": "pubmed_example_noul_v1",
        "qtype": "noul", "state": "A patient presents with a headache after a fall.", "question": "Is the drug indicated?",
        "gold": "yes", "options": [{"key": k, "label": labels[k]} for k in keys],
        "split": "test", "licence": "CC0", "source_url": "https://example.org/record/1", "source_record_id": "rec-1",
        "meta": {"tier": "fresh", "skill": "triage"},
    }


def test_reversed_and_canonical_noul_rows_pass_the_preflight() -> None:
    _scorer().preflight([_noul_row("a" * 16), _noul_row("b" * 16, keys=("yes", "no"))])


def test_a_row_that_does_not_convert_stops_the_preflight_and_names_its_id() -> None:
    with pytest.raises(ValueError, match=f"row {'c' * 16} \\(pubmed, noul\\) does not convert"):
        _scorer().preflight([_noul_row("a" * 16), _noul_row("c" * 16, keys=("yes", "maybe"))])


def test_duplicated_item_ids_stop_the_preflight() -> None:
    with pytest.raises(ValueError, match="duplicated item ids"):
        _scorer().preflight([_noul_row("a" * 16), _noul_row("a" * 16)])
