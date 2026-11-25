"""osler_v7 T3 port-check helpers on synthetic rows (CPU; no real data, no model).

The script is loaded by path (the repo's pattern for loop scripts); every row below is invented, so the test never
reads the mix, the d3 snapshot or any prediction.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "osler_v7" / "t3_port_check.py"
spec = importlib.util.spec_from_file_location("t3_port_check", SCRIPT)
assert spec is not None and spec.loader is not None
t3 = importlib.util.module_from_spec(spec)
sys.modules["t3_port_check"] = t3
spec.loader.exec_module(t3)


def _row(
    item_id: str,
    qtype: str,
    options: list[dict[str, str]],
    gold: str,
    *,
    source: str = "mimic_iv",
    template_id: str = "tpl",
    state: str = "state",
    question: str = "question",
) -> dict[str, object]:
    return {
        "item_id": item_id,
        "qtype": qtype,
        "options": json.dumps(options),
        "gold": gold,
        "source": source,
        "template_id": template_id,
        "state": state,
        "question": question,
    }


def _opts(*keys: str) -> list[dict[str, str]]:
    return [{"key": key, "label": key.upper()} for key in keys]


def test_structural_report_score_levels() -> None:
    row = _row("s1", "score", _opts("1", "2", "3"), "2")
    report = t3.structural_report(row)
    assert report["all_checks_pass"] is True
    assert report["checks"]["score_renders_in_level_order"] is True
    assert report["checks"]["score_criteria_are_labels_in_order"] is True
    assert report["n_options"] == 3


def test_structural_report_noul_renders_no_then_yes() -> None:
    row = _row("n1", "noul", _opts("yes", "no"), "yes")
    report = t3.structural_report(row)
    assert report["all_checks_pass"] is True
    assert report["checks"]["noul_renders_no_then_yes"] is True
    assert report["checks"]["noul_has_no_criteria"] is True
    # gold "yes" is the second rendered option (No, Yes)
    assert report["checks"]["rendered_gold_matches_order"] is True


def test_structural_report_choice_row_order() -> None:
    row = _row("c1", "choice", _opts("A", "B", "C", "D"), "B")
    report = t3.structural_report(row)
    assert report["all_checks_pass"] is True
    assert report["checks"]["choice_renders_in_row_order"] is True
    assert report["checks"]["choice_criteria_are_the_options"] is True


def test_structural_report_flags_an_option_count_over_255(monkeypatch: object) -> None:
    row = _row("big", "choice", _opts(*[str(i) for i in range(256)]), "0")
    monkeypatch.setattr(t3, "MAX_CODES", 255)
    report = t3.structural_report(row)
    assert report["checks"]["option_count_within_255"] is False
    assert report["all_checks_pass"] is False


def test_select_rows_picks_each_category_and_the_longest_mimic(tmp_path: Path) -> None:
    rows = [
        _row("score1", "score", _opts("1", "2"), "1", source="clinicaltrials", state="x" * 10),
        _row("noul1", "noul", _opts("yes", "no"), "no", source="clinicaltrials", state="x" * 10),
        _row("choice4a", "choice", _opts("A", "B", "C", "D"), "A", source="medqa", state="x" * 10),
        _row("choice4b", "choice", _opts("A", "B", "C", "D"), "D", source="medqa", state="x" * 10),
        _row("mimic_short", "choice", _opts("A", "B", "C", "D"), "A", state="x" * 100),
        _row("mimic_long", "choice", _opts("A", "B", "C", "D"), "A", state="x" * 900),
        _row("mimic_mid", "choice", _opts("A", "B", "C", "D"), "A", state="x" * 400),
    ]
    path = tmp_path / "mix.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    selected, counts = t3.select_rows(use_score=1, use_noul=1, use_choice4=2, use_mimic_long=2,
                                      use_mimic_first=1, mix=path)
    tags = {tag for tags in selected.tags.values() for tag in tags}
    assert {"score", "noul", "choice4", "mimic_long", "mimic_first"} <= tags
    assert counts["mimic_longest_state_chars"] == 900
    long_tags = selected.tags["mimic_long"]
    assert "mimic_long" in long_tags
    # the longest MIMIC row is selected, the short one is not a "mimic_long" pick
    assert "mimic_long" not in selected.tags.get("mimic_short", [])
