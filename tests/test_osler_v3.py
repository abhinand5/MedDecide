"""osler_v3 helpers (CPU): the fixed anchor prefixes (ADVISORY §3.2), the anchor rule, the timing summary, and the recipe
config checked against osler_v2's (only the keys the loop is allowed to change may differ)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from meddecide.train.config import load_config
from meddecide.train.osler_v3 import (
    DOMAIN_PREFIXES,
    TEMPLATE_CAP,
    anchor_share,
    capped_sizes,
    choose_recipe,
    is_anchor,
    timing_summary,
)

REPO = Path(__file__).resolve().parents[1]

ADVISORY_PREFIXES = (
    "mr_", "synthea_", "chart_", "codiesp_", "mtsamples_", "mimic_", "ehr_rel_", "pmc_", "ragtruth_", "drugprot_",
    "multicare_", "clinclaim_", "restyle_",
)


def test_domain_prefixes_are_the_advisory_list_in_order() -> None:
    assert DOMAIN_PREFIXES == ADVISORY_PREFIXES


@pytest.mark.parametrize("template_id", ["mr_icd_support_template_v2", "synthea_page_v1", "pmc_similar_patient_v1",
                                         "restyle_page_category_v1"])
def test_domain_templates_are_not_anchors(template_id: str) -> None:
    assert not is_anchor(template_id)


@pytest.mark.parametrize("template_id", ["replay_arc_v1", "boolq_v1", "snli_v1", "medqa_v1", "medmcqa_v1",
                                         "xmr_contains_but_does_not_start_with_a_prefix", "Mr_upper_case_is_case_sensitive"])
def test_general_templates_are_anchors(template_id: str) -> None:
    assert is_anchor(template_id)  # the rule is a case-sensitive prefix test, as the trainer's startswith is


def test_no_prefixes_makes_every_item_an_anchor() -> None:
    assert is_anchor("mr_icd_support_template_v2", prefixes=())


def test_anchor_share_counts_and_denominator() -> None:
    ids = ["mr_a", "medqa_b", "snli_c", "pmc_d", "boolq_e", "codiesp_f", "medmcqa_g", "medqa_h"]
    got = anchor_share(ids)
    assert got == {"items": 8, "anchors": 5, "domain": 3, "anchor_share": 5 / 8}


def test_anchor_share_refuses_an_empty_count() -> None:
    with pytest.raises(ValueError):
        anchor_share([])


def _record(elapsed: float, items: int, anchors: int, kl: float, peak: float, loss: float = 0.5) -> dict:
    return {"elapsed_s": elapsed, "n_items": items, "n_anchor": anchors, "kl": kl, "gpu_peak_gb": peak, "loss": loss}


def test_timing_summary_leaves_the_warm_up_step_out_of_the_mean() -> None:
    records = [
        _record(9.0, 8, 6, 0.10, 50.0),   # warm-up: not in the step-time mean
        _record(1.0, 8, 8, 0.20, 52.0),
        _record(1.2, 8, 0, 0.00, 58.5),   # no anchors: not in the KL mean
        _record(1.4, 8, 4, 0.40, 57.0),
    ]
    got = timing_summary(records)
    assert got["steps"] == 4
    assert got["s_per_step_mean_excl_first"] == pytest.approx((1.0 + 1.2 + 1.4) / 3)
    assert got["s_per_step_first"] == 9.0
    assert got["s_per_step_median"] == pytest.approx(1.3)
    assert got["items"] == 32 and got["anchor_items"] == 18
    assert got["anchor_share"] == pytest.approx(18 / 32)
    assert got["gpu_peak_gb_max"] == 58.5
    assert got["kl_mean_over_anchor_steps"] == pytest.approx((0.10 + 0.20 + 0.40) / 3)
    assert got["kl_steps_with_anchors"] == 3
    assert got["nonfinite_losses"] == 0


def test_timing_summary_counts_non_finite_losses_and_has_nan_kl_without_anchors() -> None:
    records = [_record(1.0, 8, 0, 0.0, 50.0, loss=float("nan")), _record(1.0, 8, 0, 0.0, 50.0, loss=0.3)]
    got = timing_summary(records)
    assert got["nonfinite_losses"] == 1
    assert math.isnan(got["kl_mean_over_anchor_steps"]) and got["kl_steps_with_anchors"] == 0


def test_timing_summary_needs_two_steps() -> None:
    with pytest.raises(ValueError):
        timing_summary([_record(1.0, 8, 0, 0.0, 50.0)])


def test_osler_v3_recipe_differs_from_osler_v2_only_in_allowed_keys() -> None:
    allowed = {"kl_to_base", "kl_exclude_template_prefixes", "output_dir", "train_path", "dev_path", "backbone"}
    v2 = yaml.safe_load((REPO / "configs" / "osler_v2" / "block_arm.yaml").read_text(encoding="utf-8"))
    v3 = yaml.safe_load((REPO / "configs" / "osler_v3" / "block_arm.yaml").read_text(encoding="utf-8"))
    assert set(v3) == set(v2) | {"kl_to_base", "kl_exclude_template_prefixes"}  # the only keys osler_v3 adds
    changed = {k for k in v2 if v3[k] != v2[k]}
    assert changed <= allowed, sorted(changed - allowed)
    assert v3["kl_to_base"] == 0.0
    assert tuple(v3["kl_exclude_template_prefixes"]) == DOMAIN_PREFIXES
    assert v3["lora"] == v2["lora"]  # D33: attention only, r 32


def test_osler_v3_recipe_loads_as_a_student_config() -> None:
    config = load_config(REPO / "configs" / "osler_v3" / "block_arm.yaml")
    assert config.kl_to_base == 0.0
    assert config.kl_exclude_template_prefixes == DOMAIN_PREFIXES
    assert config.lora.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")


def test_capped_sizes_hold_the_cap_at_the_fixed_point() -> None:
    # C1 (830,076 rows) plus the other broad rows (65,365) are fixed; AQuA-RAT (97,432) is the one template above 8 % of the block
    total, kept = capped_sizes(fixed_total=830076 + 65365, sizes={"broad_aqua_rat_v1": 97432})
    assert kept["broad_aqua_rat_v1"] <= TEMPLATE_CAP * total
    assert total == 830076 + 65365 + kept["broad_aqua_rat_v1"]
    assert abs(kept["broad_aqua_rat_v1"] - 77864) <= 1  # the X2 value: AQuA-RAT at 8 % of 973,305


def test_capped_sizes_keep_small_templates_whole() -> None:
    total, kept = capped_sizes(fixed_total=1000, sizes={"a": 10, "b": 20})
    assert kept == {"a": 10, "b": 20} and total == 1030


def test_stage_a_choice_takes_the_best_qualifying_run_and_the_control_wins_ties() -> None:
    assert choose_recipe({"A0": 0.10, "A1": 0.15, "A2": 0.12}, ["A0", "A1", "A2"]) == ("A1", False)
    # A1 scores highest but qualifies not: the control is the only candidate
    assert choose_recipe({"A0": 0.10, "A1": 0.30}, ["A0"]) == ("A0", False)
    # within 0.002 of the control: the control wins the tie
    assert choose_recipe({"A0": 0.100, "A1": 0.1015}, ["A0", "A1"]) == ("A0", True)
    # at or above 0.002 the variant replaces the control
    assert choose_recipe({"A0": 0.100, "A2": 0.1025}, ["A0", "A2"]) == ("A2", False)
    # a run without a CG score cannot be chosen
    assert choose_recipe({"A0": 0.10, "A3": None}, ["A0", "A3"]) == ("A0", False)
    with pytest.raises(ValueError):
        choose_recipe({"A0": None}, ["A0"])
