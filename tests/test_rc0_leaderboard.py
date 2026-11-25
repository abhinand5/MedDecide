"""Unit tests for the RC0 leaderboard core (osler_rc0 R7): three schemas, id bijection, metrics, Pareto."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from meddecide.eval.rc0_board import (
    cell_metrics,
    exact_dominates,
    load_cell,
    load_frozen,
    paired_difference,
    pareto_front,
    restricted_to_n_options,
    stratified_macro_ci,
    template_macro,
)


def _row(item_id: str, source: str, template: str, qtype: str, options: list[str], gold: str) -> dict:
    return {
        "item_id": item_id,
        "tier": "fresh",
        "source": source,
        "source_record_id": item_id,
        "source_url": "https://example.org/x",
        "source_license": "public-domain",
        "record_date": "2026-01-01",
        "split": "test",
        "template_id": template,
        "skill": "test",
        "qtype": qtype,
        "state": f"state {item_id}",
        "question": f"question {item_id}",
        "options": [{"key": key, "label": f"label {key}"} for key in options],
        "gold": gold,
    }


@pytest.fixture()
def frozen(tmp_path: Path):
    rows = [
        _row("raw_1", "src_a", "tpl_a", "choice", ["A", "B", "C"], "B"),
        _row("raw_2", "src_a", "tpl_a", "noul", ["yes", "no"], "yes"),
        _row("raw_3", "src_b", "tpl_b", "score", ["1", "2", "3"], "3"),
        _row("raw_4", "src_b", "tpl_b", "choice", ["A", "B"], "A"),
    ]
    path = tmp_path / "frozen.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return load_frozen("test_set", path), rows


def _write(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_frozen_bijection_is_verified(frozen) -> None:
    fs, rows = frozen
    assert len(fs) == 4
    assert set(fs.raw_index) == {row["item_id"] for row in rows}
    assert set(fs.raw_index).isdisjoint(fs.recomp_index)
    assert len(fs.recomp_index) == 4


def test_schema_a_with_and_without_probs(tmp_path: Path, frozen) -> None:
    fs, _ = frozen
    recomp = [item.recomp_id for item in fs.items]
    rows = [
        {"item_id": recomp[0], "gold_index": 1, "predicted_index": 1, "probs": [0.1, 0.8, 0.1]},
        {"item_id": recomp[1], "gold_index": 0, "predicted_index": 1},
        {"item_id": recomp[2], "gold_index": 2, "predicted_index": 2, "probs": [0.05, 0.15, 0.8]},
        {"item_id": recomp[3], "gold_index": 0, "predicted_index": 1, "probs": [0.4, 0.6]},
    ]
    cell = load_cell("m", fs, _write(tmp_path, "a.jsonl", rows))
    assert cell is not None and cell.scheme == "recomp" and cell.schema == "A"
    assert cell.scored == 4 and cell.coverage == 1.0
    assert cell.correct.tolist() == [True, False, True, False]
    assert cell.probs is None  # one row without probs makes the cell's Brier NOT MEASURED


def test_schema_b(tmp_path: Path, frozen) -> None:
    fs, _ = frozen
    raw = [item.raw_id for item in fs.items]
    rows = [
        {"item_id": raw[0], "gold_key": "B", "predicted_key": "B", "probs": [0.1, 0.8, 0.1]},
        {"item_id": raw[1], "gold_key": "yes", "predicted_key": "no", "probs": [0.3, 0.7]},
        {"item_id": raw[2], "gold_key": "3", "predicted_key": "3", "probs": [0.1, 0.2, 0.7]},
        {"item_id": raw[3], "gold_key": "A", "predicted_key": "A", "probs": [0.9, 0.1]},
    ]
    cell = load_cell("m", fs, _write(tmp_path, "b.jsonl", rows))
    assert cell is not None and cell.scheme == "raw" and cell.schema == "B"
    assert cell.correct.tolist() == [True, False, True, True]
    assert cell.probs is not None and len(cell.probs) == 4


def test_schema_c_and_skipped_rows(tmp_path: Path, frozen) -> None:
    fs, _ = frozen
    raw = [item.raw_id for item in fs.items]
    rows = [
        {"item_id": raw[0], "status": "scored", "gold": "B", "option_keys": ["A", "B", "C"],
         "probs": [0.1, 0.8, 0.1], "extras": {"label_mass": 0.99}},
        {"item_id": raw[1], "status": "skipped", "gold": "yes", "option_keys": ["yes", "no"],
         "probs": [0.5, 0.5], "extras": {"label_mass": 0.5}},
        {"item_id": raw[2], "status": "scored", "gold": "3", "option_keys": ["1", "2", "3"],
         "probs": [0.1, 0.2, 0.7], "extras": {"label_mass": 0.9}},
        {"item_id": raw[3], "status": "scored", "gold": "A", "option_keys": ["A", "B"],
         "probs": [0.6, 0.4], "extras": {"label_mass": 0.8}},
    ]
    cell = load_cell("m", fs, _write(tmp_path, "c.jsonl", rows))
    assert cell is not None and cell.schema == "C" and cell.scheme == "raw"
    assert cell.scored == 3 and cell.skipped == 1
    assert cell.coverage == pytest.approx(3 / 4)
    assert cell.label_mass is not None and cell.label_mass.shape == (3,)


def test_unmapped_id_raises(tmp_path: Path, frozen) -> None:
    fs, _ = frozen
    rows = [{"item_id": "not_an_id", "gold_index": 0, "predicted_index": 0, "probs": [1.0, 0.0]}]
    with pytest.raises(ValueError, match="map raw="):
        load_cell("m", fs, _write(tmp_path, "bad.jsonl", rows))


def test_mixed_scheme_ids_raise(tmp_path: Path, frozen) -> None:
    fs, _ = frozen
    rows = [
        {"item_id": fs.items[0].raw_id, "gold_index": 1, "predicted_index": 1, "probs": [0.1, 0.8, 0.1]},
        {"item_id": fs.items[1].recomp_id, "gold_index": 0, "predicted_index": 0, "probs": [0.9, 0.1]},
    ]
    with pytest.raises(ValueError, match="map raw="):
        load_cell("m", fs, _write(tmp_path, "mixed.jsonl", rows))


def test_wrong_gold_raises(tmp_path: Path, frozen) -> None:
    fs, _ = frozen
    rows = [{"item_id": fs.items[0].recomp_id, "gold_index": 0, "predicted_index": 0, "probs": [1.0, 0.0, 0.0]}]
    with pytest.raises(ValueError, match="gold_index"):
        load_cell("m", fs, _write(tmp_path, "gold.jsonl", rows))


def test_bad_probabilities_raise(tmp_path: Path, frozen) -> None:
    fs, _ = frozen
    rows = [{"item_id": fs.items[0].recomp_id, "gold_index": 1, "predicted_index": 1, "probs": [0.5, 0.8, 0.1]}]
    with pytest.raises(ValueError, match="sum"):
        load_cell("m", fs, _write(tmp_path, "probs.jsonl", rows))


def test_metrics_brier_and_ece_by_hand(tmp_path: Path, frozen) -> None:
    fs, _ = frozen
    recomp = [item.recomp_id for item in fs.items]
    rows = [
        {"item_id": recomp[0], "gold_index": 1, "predicted_index": 1, "probs": [0.1, 0.8, 0.1]},
        {"item_id": recomp[2], "gold_index": 2, "predicted_index": 2, "probs": [0.05, 0.15, 0.8]},
        {"item_id": recomp[3], "gold_index": 0, "predicted_index": 1, "probs": [0.4, 0.6]},
    ]
    cell = load_cell("m", fs, _write(tmp_path, "metrics.jsonl", rows))
    metrics = cell_metrics(fs, cell)
    # Brier: item 1 = (0.1^2 + 0.2^2 + 0.1^2)=0.06; item 3 = (0.05^2+0.15^2+0.2^2)=0.065;
    # item 4 = (0.4-1)^2 + (0.6-0)^2 = 0.72
    assert metrics["brier"] == pytest.approx((0.06 + 0.065 + 0.72) / 3, abs=1e-6)
    # top-label ECE: confidences 0.8 and 0.8 correct (bin 12), 0.6 wrong (bin 9)
    # bin 12: |1 - 0.8| * 2/3 = 0.133333 ; bin 9: |0 - 0.6| * 1/3 = 0.2
    assert metrics["ece"] == pytest.approx(0.1333333 + 0.2, abs=1e-6)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["macro_accuracy"] == pytest.approx((1.0 + 0.5) / 2)  # tpl_a: 1/1, tpl_b: 1/2


def test_template_macro_and_stratified_ci() -> None:
    templates = np.array(["a", "a", "a", "b", "b"], dtype=object)
    correct = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    assert template_macro(templates, correct) == pytest.approx((2 / 3 + 0) / 2)
    lo, hi = stratified_macro_ci(templates, correct, n_resamples=200, seed=0)
    assert lo <= hi <= 1.0
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.5)


def test_paired_difference_and_restriction(tmp_path: Path, frozen) -> None:
    fs, _ = frozen
    recomp = [item.recomp_id for item in fs.items]
    a = load_cell("a", fs, _write(tmp_path, "pa.jsonl", [
        {"item_id": recomp[0], "gold_index": 1, "predicted_index": 1, "probs": [0.1, 0.8, 0.1]},
        {"item_id": recomp[2], "gold_index": 2, "predicted_index": 2, "probs": [0.05, 0.15, 0.8]},
    ]))
    b = load_cell("b", fs, _write(tmp_path, "pb.jsonl", [
        {"item_id": recomp[0], "gold_index": 1, "predicted_index": 0, "probs": [0.8, 0.1, 0.1]},
        {"item_id": recomp[2], "gold_index": 2, "predicted_index": 2, "probs": [0.0, 0.1, 0.9]},
    ]))
    diff = paired_difference(fs, a, b)
    assert diff["n"] == 2
    assert diff["accuracy_diff"] == pytest.approx(0.5)
    assert diff["brier_diff"] == pytest.approx(((0.06 + 0.065) - (1.46 + 0.02)) / 2)
    restricted = restricted_to_n_options(a, fs, 2)
    assert restricted.scored == 0 and restricted.n_universe == 2


def test_pareto_functions() -> None:
    a = {"x": 0.6, "y": 0.7}
    b = {"x": 0.5, "y": 0.7}
    c = {"x": 0.6, "y": 0.8}
    assert exact_dominates(c, a, ["x", "y"])
    assert exact_dominates(a, b, ["x", "y"])
    assert not exact_dominates(a, b, ["x", "y", "z"])  # incomparable when a key is missing
    assert not exact_dominates(a, a, ["x", "y"])  # strict on at least one
    front = pareto_front({"a": (0.6, 0.7), "b": (0.5, 0.7), "c": (0.6, 0.8), "d": (0.9, 0.1)})
    assert front == ["c", "d"]


def test_schema_a_accepts_the_d3_noul_rendering_swap(tmp_path: Path, frozen) -> None:
    """The d3 path renders noul as (no, yes): its gold index is the frozen one or its swap, and metrics follow it."""
    fs, _ = frozen
    recomp = [item.recomp_id for item in fs.items]
    rows = [
        {"item_id": recomp[1], "gold_index": 1, "predicted_index": 1, "probs": [0.3, 0.7]},  # gold yes, rendered (no, yes)
        {"item_id": recomp[2], "gold_index": 2, "predicted_index": 2, "probs": [0.05, 0.15, 0.8]},
    ]
    cell = load_cell("m", fs, _write(tmp_path, "noul_swap.jsonl", rows))
    assert cell is not None and cell.correct.tolist() == [True, True]
    metrics = cell_metrics(fs, cell)
    # item 1 Brier: (0.3-0)^2 + (0.7-1)^2 = 0.18 ; item 2: 0.065
    assert metrics["brier"] == pytest.approx((0.18 + 0.065) / 2, abs=1e-6)


def test_schema_a_rejects_a_gold_index_that_is_not_a_permutation(tmp_path: Path, frozen) -> None:
    fs, _ = frozen
    rows = [{"item_id": fs.items[0].recomp_id, "gold_index": 2, "predicted_index": 2, "probs": [0.1, 0.1, 0.8]}]
    with pytest.raises(ValueError, match="not a permutation"):
        load_cell("m", fs, _write(tmp_path, "bad_gold.jsonl", rows))
