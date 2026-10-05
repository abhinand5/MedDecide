"""Smoke tests: the package imports, exposes a version, and has the expected layout."""

from __future__ import annotations

import importlib

import meddecide


def test_package_imports() -> None:
    assert meddecide.__version__


def test_subpackages_import() -> None:
    for name in ("meddecide.bench", "meddecide.eval", "meddecide.teacher", "meddecide.utils"):
        assert importlib.import_module(name) is not None


def test_bench_schema_symbols() -> None:
    from meddecide.bench import Item, Option, QuestionType, Split, Tier, make_item

    assert all(x is not None for x in (Item, Option, QuestionType, Split, Tier, make_item))


def test_eval_metrics_symbols() -> None:
    from meddecide.eval import accuracy, brier_score, ece

    assert all(callable(f) for f in (accuracy, brier_score, ece))


def test_io_and_hashing_import() -> None:
    from meddecide.utils.io import build_manifest, read_jsonl, write_jsonl
    from meddecide.utils.provenance import Provenance, git_commit, utcnow

    assert all(callable(f) for f in (build_manifest, read_jsonl, write_jsonl, git_commit, utcnow))
    assert Provenance(run_name="x", command="y").git_commit
