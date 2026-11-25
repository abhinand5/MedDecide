"""osler_v4 keep rule v2 (D47): the margins and the verdict on synthetic lower bounds."""

import pytest

from meddecide.train.keep_v2 import KEEP_V2_MARGINS, keep_v2_verdict


def test_margins_are_the_d47_values() -> None:
    assert KEEP_V2_MARGINS == {"own": 0.02, "mr": -0.015, "general": -0.01, "knowledge": -0.02, "cg_v2": -0.01}


def test_a_block_is_kept_only_when_every_lower_bound_is_strictly_above_its_margin() -> None:
    passing = {"own": 0.03, "mr": -0.01, "general": -0.005, "knowledge": -0.01, "cg_v2": -0.005}
    assert keep_v2_verdict(passing)["kept"] is True
    on_the_margin = dict(passing, own=0.02)  # equal to the margin: not kept (strictly above)
    assert keep_v2_verdict(on_the_margin)["kept"] is False
    assert keep_v2_verdict(on_the_margin)["checks"]["own"] is False


def test_one_failing_check_rejects_the_block_and_the_others_are_still_reported() -> None:
    verdict = keep_v2_verdict({"own": 0.3, "mr": -0.02, "general": 0.0, "knowledge": 0.0, "cg_v2": 0.0})
    assert verdict["kept"] is False
    assert verdict["checks"] == {"own": True, "mr": False, "general": True, "knowledge": True, "cg_v2": True}


def test_a_missing_lower_bound_is_refused() -> None:
    with pytest.raises(ValueError):
        keep_v2_verdict({"own": 0.1, "mr": 0.0})
