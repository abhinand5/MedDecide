"""Unit tests for the distill_v0 pool/teacher attachment (synthetic rows only)."""

from __future__ import annotations

import json

import pytest

from meddecide.train.distill_pool import attach_teacher_probs, load_labels


def row(template: str = "medmcqa_4opt_v1", keys: tuple[str, ...] = ("A", "B", "C")) -> dict[str, object]:
    return {"item_id": "0123456789abcdef", "template_id": template, "qtype": "choice",
            "options": [{"key": key, "label": f"label {key}"} for key in keys], "gold": keys[0]}


def label(key: str = "k1", template: str = "medmcqa_4opt_v1", keys: tuple[str, ...] = ("A", "B", "C"),
          probs: tuple[float, ...] = (0.2, 0.5, 0.3), teacher: str = "liquid/d1-ultra-20261015") -> dict[str, object]:
    return {"item_id": "0123456789abcdef", "text_key": key, "template_id": template, "qtype": "choice",
            "option_keys": list(keys), "teacher_probs": list(probs), "teacher": teacher}


def test_attach_adds_the_distribution_only_to_teacher_templates() -> None:
    rows = [row(), row("mimic_msdrg_v1")]
    keys = ["k1", "k2"]
    labels = {"k1": label(), "k2": label("k2", template="mimic_msdrg_v1")}
    report = attach_teacher_probs(rows, keys, labels, model_id="liquid/d1-ultra-20261015")
    assert rows[0]["teacher_probs"] == [0.2, 0.5, 0.3]
    assert rows[0]["teacher"] == "liquid/d1-ultra-20261015"
    assert "teacher_probs" not in rows[1]  # MIMIC stays gold-only even when a label exists
    assert report["attached"] == 1


def test_attach_renormalises_and_records_the_deviation() -> None:
    rows = [row()]
    labels = {"k1": label(probs=(0.2, 0.5, 0.2999))}
    report = attach_teacher_probs(rows, ["k1"], labels, model_id="m")
    assert sum(rows[0]["teacher_probs"]) == pytest.approx(1.0)
    assert report["renormalised"] == 1
    assert report["max_abs_sum_deviation"] == pytest.approx(0.0001)


def test_attach_refuses_a_changed_option_order_or_template() -> None:
    rows = [row(), row()]
    labels = {"k1": label(keys=("B", "A", "C")), "k2": label("k2", template="another_template_v1")}
    report = attach_teacher_probs(rows, ["k1", "k2"], labels, model_id="m")
    assert report["attached"] == 0
    assert report["option_order_mismatch"] == 1
    assert report["template_mismatch"] == 1
    assert all("teacher_probs" not in item for item in rows)


def test_attach_counts_missing_labels() -> None:
    report = attach_teacher_probs([row()], ["absent"], {}, model_id="m")
    assert report["attached"] == 0 and report["missing_label"] == 1


def test_load_labels_rejects_a_duplicate_text_key(tmp_path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(label(key="k1")) for _ in range(2)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        load_labels(path)


def test_attach_with_a_basket_leaves_capped_rows_gold_only() -> None:
    rows = [row(), row()]
    labels = {"k1": label("k1"), "k2": label("k2")}
    report = attach_teacher_probs(rows, ["k1", "k2"], labels, model_id="m", basket_indices={0})
    assert report["attached"] == 1 and report["attached_keys"] == 1
    assert report["outside_basket_teacher_rows"] == 1
    assert "teacher_probs" not in rows[1]
