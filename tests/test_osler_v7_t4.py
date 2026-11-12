"""osler_v7 T4 grid metrics on synthetic prediction rows (CPU; no model, no real data)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "osler_v7" / "t4_grid.py"
spec = importlib.util.spec_from_file_location("t4_grid", SCRIPT)
assert spec is not None and spec.loader is not None
t4 = importlib.util.module_from_spec(spec)
sys.modules["t4_grid"] = t4
spec.loader.exec_module(t4)


def _pred(item_id: str, template_id: str, qtype: str, n_options: int, gold: int, predicted: int) -> dict[str, object]:
    return {
        "item_id": item_id,
        "source": "s",
        "template_id": template_id,
        "qtype": qtype,
        "n_options": n_options,
        "gold_index": gold,
        "predicted_index": predicted,
        "correct": predicted == gold,
    }


def test_option_group_buckets() -> None:
    assert [t4.option_group(n) for n in (2, 3, 4, 5, 6, 9, 10, 11, 49)] == [
        "2", "3", "4", "5", "6_9", "6_9", "10", "11_49", "11_49"]


def test_summarise_partitions_and_macro() -> None:
    rows = [
        _pred("m1", "mimic_icd_support_v1", "choice", 4, 0, 0),          # MIMIC correct
        _pred("m2", "mimic_ed_acuity_v1", "score", 4, 2, 2),             # MIMIC correct
        _pred("r1", "mr_icd_support_template_v2", "choice", 4, 1, 0),    # MR wrong
        _pred("c1", "cg_v2_ddxplus_validate_v1", "choice", 49, 30, 30),  # CG wide correct
        _pred("c2", "cg_v2_ddxplus_validate_v1", "choice", 49, 3, 3),    # CG wide correct
        _pred("k1", "medmcqa_4opt_v1", "choice", 4, 3, 2),               # knowledge wrong
    ]
    meta = {
        "m1": {"dev_origin": "fastdev_v4"}, "m2": {"dev_origin": "fastdev_v4"},
        "r1": {"dev_origin": "fastdev_v4"}, "c1": {"dev_origin": "cg_v2"},
        "c2": {"dev_origin": "cg_v2"}, "k1": {"dev_origin": "knowledge_dev_v2"},
    }
    out = t4.summarise(rows, meta)
    assert out["overall"]["items"] == 6
    assert out["overall"]["accuracy_item"] == round(4 / 6, 8)
    assert out["mr_family"]["items"] == 1 and out["mr_family"]["accuracy_item"] == 0.0
    assert out["mimic_templates"]["items"] == 2 and out["mimic_templates"]["accuracy_item"] == 1.0
    assert out["cg_origin_micro"] == {"items": 2, "accuracy_item": 1.0}
    assert out["knowledge_origin"] == {"items": 1, "accuracy_item": 0.0}
    assert out["two_letter"]["items"] == 2 and out["two_letter"]["accuracy_item"] == 1.0
    assert out["by_qtype"]["score"]["items"] == 1
    assert out["by_option_group"]["11_49"]["items"] == 2
    assert out["positional"]["passes"] is True
    assert out["positional"]["two_letter_accuracy"] == 1.0
    assert out["positional"]["single_letter_accuracy"] == 1.0


def test_summarise_handles_no_wide_rows() -> None:
    rows = [_pred("a", "mimic_icd_support_v1", "choice", 4, 0, 0)]
    out = t4.summarise(rows, {"a": {"dev_origin": "fastdev_v4"}})
    assert out["two_letter"]["items"] == 0
    assert out["positional"]["status"].startswith("NOT MEASURED")
