"""osler_v7 T7 selection rule on synthetic scale records (CPU; no model, no real data)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "osler_v7" / "t7_select.py"
spec = importlib.util.spec_from_file_location("t7_select", SCRIPT)
assert spec is not None and spec.loader is not None
t7 = importlib.util.module_from_spec(spec)
sys.modules["t7_select"] = t7
spec.loader.exec_module(t7)


def _candidate(scale: float, s: float, lo: float) -> dict[str, object]:
    return {"scale": scale, "S": s, "paired_template_macro_difference": {"ci_lo": lo}, "eligible": lo > t7.NON_INFERIORITY}


def test_choose_picks_the_highest_s_among_eligible() -> None:
    choice = t7.choose_scale([_candidate(1.0, 0.80, -0.02), _candidate(0.75, 0.81, -0.005), _candidate(0.5, 0.79, 0.001)])
    assert choice["chosen_scale"] == 0.75


def test_choose_returns_none_when_no_candidate_is_eligible() -> None:
    choice = t7.choose_scale([_candidate(1.0, 0.80, -0.02), _candidate(0.5, 0.81, -0.011)])
    assert choice["chosen_scale"] is None
    assert "no candidate" in choice["reason"]


def test_apply_blend_rejects_out_of_range_scale() -> None:
    with pytest.raises(ValueError, match="must be in"):
        t7.apply_blend(None, None, 1.5)  # type: ignore[arg-type]
