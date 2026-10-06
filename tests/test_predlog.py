"""Unit tests for the append-only prediction-log reader (S1)."""

from __future__ import annotations

import json

from meddecide.eval.predlog import read_prediction_log


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_rows_without_run_id_still_read_as_one_run(tmp_path) -> None:
    path = _write(
        tmp_path / "preds.jsonl",
        [{"item_id": "a", "correct": True}, {"item_id": "b", "correct": False}],
    )
    log = read_prediction_log(path)
    assert log.n_raw == 2 and log.n_unique == 2 and log.n_distinct_items == 2
    assert log.n_without_run_id == 2
    assert log.run_ids == [""]


def test_reruns_are_kept_per_run_and_deduplicated_within_a_run(tmp_path) -> None:
    """Two runs of the same item are two rows (one per run), not one.

    Deduplication is by ``(run_id, item_id)``: within a run a repeated row collapses (the
    resumed-run case below), across runs the rows are kept so a consumer can choose a run.
    The whole-file row count is therefore a *log* size, and ``rows_for_run`` is the unit a
    metric is computed on.
    """
    path = _write(
        tmp_path / "preds.jsonl",
        [
            {"run_id": "r1", "item_id": "a", "correct": False},
            {"run_id": "r1", "item_id": "b", "correct": True},
            {"run_id": "r2", "item_id": "a", "correct": True},
            {"run_id": "r2", "item_id": "b", "correct": True},
        ],
    )
    log = read_prediction_log(path)
    assert log.n_raw == 4
    assert log.n_unique == 4  # four distinct (run, item) keys
    assert log.n_distinct_items == 2  # ... over two distinct items
    assert log.n_duplicate_rows == 0
    assert log.run_ids == ["r1", "r2"]
    assert log.per_run() == {
        "r1": {"n_rows": 2, "n_distinct_items": 2},
        "r2": {"n_rows": 2, "n_distinct_items": 2},
    }
    # one run is one row per item: this is what a metric must be computed on
    assert {row["item_id"]: row["correct"] for row in log.rows_for_run("r2")} == {
        "a": True,
        "b": True,
    }
    assert log.rows_for_run("r1")[0]["correct"] is False


def test_a_resumed_run_does_not_double_count_an_item(tmp_path) -> None:
    """The failure P7 describes: the same run appends the same item again after a resume."""
    path = _write(
        tmp_path / "preds.jsonl",
        [
            {"run_id": "r1", "item_id": "a", "correct": False},
            {"run_id": "r1", "item_id": "a", "correct": True},
        ],
    )
    log = read_prediction_log(path)
    assert log.n_raw == 2 and log.n_unique == 1 and log.n_duplicate_rows == 1
    # last row wins, so the resumed run's later answer is the one read
    assert log.rows[0]["correct"] is True


def test_unreadable_lines_are_counted_not_silently_skipped(tmp_path) -> None:
    path = tmp_path / "preds.jsonl"
    path.write_text('{"run_id": "r1", "item_id": "a"}\nnot json\n', encoding="utf-8")
    log = read_prediction_log(path)
    assert log.n_raw == 2 and log.unreadable_lines == 1 and log.n_unique == 1
    assert log.report()["unreadable_lines"] == 1


def test_missing_file_is_an_empty_log(tmp_path) -> None:
    log = read_prediction_log(tmp_path / "nope.jsonl")
    assert log.n_raw == 0 and log.n_unique == 0
    assert log.report()["dedupe_key"] == "(run_id, item_id), last row wins"
