"""osler_v5 Z4 blend rule: retention at 80 % of the full gain, positional health, CG micro against alpha 1, the smallest alpha, outcome 3."""

from __future__ import annotations

import pytest

from meddecide.eval.blend_choice import AlphaStats, choose_alpha


def _s(alpha: float, mr: float, mimic: float, cg: float, pos: bool = True) -> AlphaStats:
    return AlphaStats(alpha=alpha, mr_macro=mr, mimic_macro=mimic, cg_micro=cg, positional_passes=pos)


def test_the_smallest_qualifying_alpha_is_chosen() -> None:
    stats = [_s(0.0, 0.70, 0.60, 0.80), _s(0.5, 0.80, 0.70, 0.80), _s(0.8, 0.90, 0.80, 0.80), _s(1.0, 1.00, 0.90, 0.79)]
    # gains at alpha 1 are 0.30 (MR) and 0.30 (MIMIC); 0.5 keeps a third and 0.8 two thirds, below the 80 % retention, so only alpha 1 remains
    out = choose_alpha(stats)
    assert out["chosen_alpha"] == 1.0 and out["blend_used"] is False
    stats2 = [_s(0.0, 0.70, 0.60, 0.80), _s(0.5, 0.95, 0.85, 0.81), _s(0.8, 1.00, 0.90, 0.80), _s(1.0, 1.00, 0.90, 0.79)]
    out2 = choose_alpha(stats2)
    assert out2["chosen_alpha"] == 0.5 and out2["blend_used"] is True  # 0.5 keeps 0.25 of each 0.30 gain, above the 0.24 floor


def test_the_cg_comparison_is_against_alpha_one() -> None:
    stats = [_s(0.0, 0.70, 0.60, 0.80), _s(0.5, 0.95, 0.85, 0.78), _s(1.0, 1.00, 0.90, 0.79)]
    assert choose_alpha(stats)["alphas"]["0.5"]["cg_micro_at_least_full"] is False
    assert choose_alpha(stats)["chosen_alpha"] == 1.0


def test_positional_health_is_required() -> None:
    stats = [_s(0.0, 0.70, 0.60, 0.80), _s(0.5, 0.95, 0.85, 0.85, pos=False), _s(1.0, 1.00, 0.90, 0.79)]
    assert choose_alpha(stats)["chosen_alpha"] == 1.0


def test_no_qualifier_keeps_alpha_one_and_reports_outcome_three() -> None:
    stats = [_s(0.0, 0.70, 0.60, 0.80), _s(0.5, 0.71, 0.61, 0.80), _s(1.0, 1.00, 0.90, 0.79)]
    out = choose_alpha(stats)
    assert out["chosen_alpha"] == 1.0 and out["blend_used"] is False and "outcome 3" in out["outcome"]


def test_the_reference_and_the_full_model_must_both_be_present() -> None:
    with pytest.raises(ValueError, match="must be among"):
        choose_alpha([_s(0.5, 0.9, 0.9, 0.8), _s(1.0, 1.0, 0.9, 0.8)])
    with pytest.raises(ValueError, match="appears twice"):
        choose_alpha([_s(0.0, 0.7, 0.6, 0.8), _s(0.0, 0.7, 0.6, 0.8), _s(1.0, 1.0, 0.9, 0.8)])
