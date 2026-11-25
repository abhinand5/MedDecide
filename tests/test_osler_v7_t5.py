"""osler_v7 T5 pool builders: pure helpers on synthetic inputs (CPU; no real data, no model)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "osler_v7"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


c1 = _load("c1_many_option", "c1_many_option.py")
c2 = _load("c2_pairs", "c2_pairs.py")
c4 = _load("c4_option_augment", "c4_option_augment.py")


def test_c1_option_keys_follow_the_255_code_scheme() -> None:
    assert [c1.option_key(i) for i in (0, 1, 25, 26, 27, 51, 52)] == ["A", "B", "Z", "AA", "AB", "AZ", "BA"]


def test_c1_code_normalisation_and_chapter() -> None:
    assert c1.normalise_code("j96.01") == "J9601"
    assert c1.normalise_code(None) == ""
    assert c1.chapter_of("J9601") == "respiratory"
    assert c1.chapter_of("I10") == "circulatory"
    assert c1.chapter_of("Z0000") == "factors"
    assert c1.chapter_of("") == "other"


def test_c2_negation_window() -> None:
    tokens = ["no", "evidence", "of", "pneumonia", "and", "sepsis", "documented"]
    assert c2.negated(tokens, 3) is True  # pneumonia, within the 5-token window of "no"
    assert c2.negated(tokens, 6) is False  # "documented", window starts after the negation words


def test_c4_key_at_matches_the_mix_code_scheme() -> None:
    assert [c4.key_at(i) for i in (0, 25, 26, 27, 51)] == ["A", "Z", "AA", "AB", "AZ"]
