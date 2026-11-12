"""osler_v7 T1 composition/prior-shift helpers on synthetic rows (CPU; no real data).

The script is loaded by path (the repo's pattern for loop scripts, e.g. tests/test_o11_audit.py); every row below is
invented, so the test never touches the panel, the mix or any prediction file.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "osler_v7" / "t1_diagnose.py"
spec = importlib.util.spec_from_file_location("t1_diagnose", SCRIPT)
assert spec is not None and spec.loader is not None
t1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t1)


def _row(qtype: str, options: list[dict[str, str]], gold: str, **extra: object) -> dict[str, object]:
    return {
        "qtype": qtype,
        "options": options,
        "gold": gold,
        "template_id": extra.get("template_id", "tpl"),
        "source": extra.get("source", "src"),
        "split": extra.get("split", "train"),
        "tier": extra.get("tier", "fresh"),
        "state": extra.get("state", "state"),
        "question": extra.get("question", "question"),
        "record_date": extra.get("record_date", "2024-01-02"),
    }


def _opts(*keys: str) -> list[dict[str, str]]:
    return [{"key": key, "label": key.upper()} for key in keys]


def test_parse_options_accepts_the_mix_string_and_the_panel_list() -> None:
    as_list = [{"key": "A", "label": "alpha"}]
    as_string = '[{"key": "A", "label": "alpha"}]'
    assert t1.parse_options({"options": as_list}) == as_list
    assert t1.parse_options({"options": as_string}) == as_list


def test_gold_index_finds_the_key_position_or_none() -> None:
    row = _row("choice", _opts("A", "B", "C"), "C")
    assert t1.gold_index(row, t1.parse_options(row)) == 2
    missing = _row("choice", _opts("A", "B"), "Z")
    assert t1.gold_index(missing, t1.parse_options(missing)) is None


def test_relative_position_and_bins() -> None:
    assert t1.relative_position(0, 4) == 0.0
    assert t1.relative_position(3, 4) == 1.0
    assert t1.relative_position(1, 4) == 1 / 3
    assert t1.relative_position(5, 1) == 0.0
    assert [t1.relative_bin(v) for v in (0.0, 0.2, 0.3, 0.6, 0.9, 1.0)] == [
        "0_25",
        "0_25",
        "25_50",
        "50_75",
        "75_100",
        "75_100",
    ]


def test_char_bucket_boundaries() -> None:
    assert t1.char_bucket(0) == "0_99"
    assert t1.char_bucket(99) == "0_99"
    assert t1.char_bucket(100) == "100_249"
    assert t1.char_bucket(32_000) == "ge32000"


def test_total_variation_identical_and_disjoint() -> None:
    assert t1.total_variation({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0.0
    assert t1.total_variation({"a": 1.0}, {"b": 1.0}) == 1.0
    assert t1.total_variation({"a": 0.75, "b": 0.25}, {"a": 0.25, "b": 0.75}) == 0.5


def test_absorb_and_finalise_composition() -> None:
    comp = t1.new_composition()
    t1.absorb(comp, _row("choice", _opts("A", "B", "C", "D"), "D"), with_dates=True)
    t1.absorb(comp, _row("noul", _opts("yes", "no"), "yes", record_date="2026-04-01"), with_dates=True)
    t1.absorb(comp, _row("noul", _opts("yes", "no"), "no"), with_dates=True)
    t1.absorb(comp, _row("score", _opts("1", "2", "3"), "2"), with_dates=True)
    out = t1.finalise_composition(comp, with_dates=True)
    assert out["rows"] == 4
    assert out["qtype"] == {"choice": 1, "noul": 2, "score": 1}
    assert out["choice"]["gold_relative_mean"] == 1.0
    assert out["score"]["gold_index"] == {"1": 1}
    assert out["noul"]["gold"] == {"no": 1, "yes": 1}
    assert out["noul"]["yes_share"] == 0.5
    assert out["record_dates"]["dated_rows"] == 4
    assert out["record_dates"]["pre_window_share_of_dated"] == 0.75


def test_composition_compare_of_a_file_with_itself_is_zero() -> None:
    comp = t1.new_composition()
    for row in (
        _row("choice", _opts("A", "B"), "A"),
        _row("noul", _opts("yes", "no"), "yes"),
    ):
        t1.absorb(comp, row, with_dates=False)
    final = t1.finalise_composition(comp, with_dates=False)
    cmp = t1.composition_compare(final, final)
    assert cmp["qtype_tvd"] == 0.0
    assert cmp["choice_option_count_tvd"] == 0.0
    assert cmp["noul_yes_share_gap"] == 0.0
    assert cmp["templates_shared"] == 1
    assert cmp["fraction_of_panel_items_from_templates_seen_in_training"] == 1.0
    assert final["record_dates"]["status"].startswith("NOT MEASURED")


def _panel_item(item_id: str, source: str, qtype: str, keys: list[str], gold: str) -> dict[str, object]:
    return {
        "_item_id": item_id,
        "_options": [{"key": key, "label": key.upper()} for key in keys],
        "qtype": qtype,
        "source": source,
        "gold": gold,
    }


def _prediction(item_id: str, keys: list[str], probs: list[float]) -> dict[str, object]:
    index = max(range(len(probs)), key=probs.__getitem__)
    return {"item_id": item_id, "probs": probs, "predicted_key": keys[index]}


def test_analyse_panel_yes_rate_and_option_positions() -> None:
    keys4 = ["A", "B", "C", "D"]
    items = [
        _panel_item("i1", "noul_src", "noul", ["yes", "no"], "yes"),
        _panel_item("i2", "noul_src", "noul", ["yes", "no"], "no"),
        _panel_item("i3", "choice_src", "choice", keys4, "B"),
    ]
    final = {
        "i1": _prediction("i1", ["yes", "no"], [0.9, 0.1]),
        "i2": _prediction("i2", ["yes", "no"], [0.2, 0.8]),
        "i3": _prediction("i3", keys4, [0.1, 0.2, 0.3, 0.4]),
    }
    base = {
        "i1": _prediction("i1", ["yes", "no"], [0.6, 0.4]),
        "i2": _prediction("i2", ["yes", "no"], [0.7, 0.3]),
        "i3": _prediction("i3", keys4, [0.4, 0.3, 0.2, 0.1]),
    }
    out = t1.analyse_panel(items, final, base)
    assert out["noul"]["items"] == 2
    assert out["noul"]["gold_yes_share"] == 0.5
    assert out["noul"]["final"]["mean_p_yes"] == 0.55
    assert out["noul"]["final"]["argmax_yes_rate"] == 0.5
    assert out["noul"]["final"]["accuracy"] == 1.0
    assert out["noul"]["base"]["accuracy"] == 0.5
    assert out["noul"]["final"]["tp"] == 1
    assert out["noul"]["final"]["tn"] == 1
    assert out["per_dataset"]["choice_src"]["final_accuracy"] == 0.0
    assert out["per_dataset"]["choice_src"]["base_accuracy"] == 0.0
    assert out["per_dataset"]["choice_src"]["final_index_share"] == {"3": 1.0}
    assert out["per_dataset"]["choice_src"]["position_tvd_final_gold"] == 1.0
    assert out["summary"]["losing_sources"] == []
