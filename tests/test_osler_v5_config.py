"""osler_v5 config keys: loss_weighting (D51) and the option-augmentation settings (D52a). Defaults reproduce the osler_v4 recipe."""

from __future__ import annotations

import pytest

from meddecide.train.config import StudentConfig, load_config


def test_the_defaults_are_the_osler_v4_behaviour() -> None:
    cfg = StudentConfig()
    assert cfg.loss_weighting == "per_step"
    assert cfg.option_augment_probability == 0.0
    assert cfg.option_augment_templates == ()


def test_per_item_and_a_closed_list_augmentation_are_accepted() -> None:
    cfg = StudentConfig.from_dict({"loss_weighting": "per_item", "option_augment_probability": 1.0,
                                   "option_augment_templates": ["mimic_msdrg_v1", "mimic_principal_dx_v1"]})
    assert cfg.loss_weighting == "per_item"
    assert cfg.option_augment_templates == ("mimic_msdrg_v1", "mimic_principal_dx_v1")


def test_an_unknown_loss_weighting_is_refused() -> None:
    with pytest.raises(ValueError, match="loss_weighting"):
        StudentConfig.from_dict({"loss_weighting": "per_token"})


def test_augmentation_needs_both_a_probability_and_templates() -> None:
    with pytest.raises(ValueError, match="both a probability"):
        StudentConfig.from_dict({"option_augment_probability": 0.5})
    with pytest.raises(ValueError, match="both a probability"):
        StudentConfig.from_dict({"option_augment_templates": ["mimic_msdrg_v1"]})
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        StudentConfig.from_dict({"option_augment_probability": 1.5, "option_augment_templates": ["mimic_msdrg_v1"]})


def test_a_template_off_the_closed_list_is_refused() -> None:
    with pytest.raises(ValueError, match="not on the closed list"):
        StudentConfig.from_dict({"option_augment_probability": 1.0, "option_augment_templates": ["csqa_choice_v1"]})


def test_the_osler_v5_recipe_configs_load() -> None:
    cfg = load_config("configs/osler_v5/arm_R2.yaml")
    assert cfg.loss_weighting == "per_item"
    assert cfg.option_augment_templates == ("mimic_principal_dx_v1", "mimic_msdrg_v1")
    assert cfg.kl_to_base == 0.0
    assert load_config("configs/osler_v5/arm_R0.yaml").loss_weighting == "per_step"


def test_the_corrected_mix_drop_is_validated_and_the_f_arms_load() -> None:
    cfg = StudentConfig.from_dict({"drop_mix_templates": ["fda_route_choice_v1"]})
    assert cfg.drop_mix_templates == ("fda_route_choice_v1",)
    assert StudentConfig().drop_mix_templates == ()
    with pytest.raises(ValueError, match="not allowed to be dropped"):
        StudentConfig.from_dict({"drop_mix_templates": ["csqa_choice_v1"]})
    assert load_config("configs/osler_v5/arm_F2.yaml").drop_mix_templates == ("fda_route_choice_v1",)
    assert load_config("configs/osler_v5/arm_F2s1.yaml").seed == 1
    assert load_config("configs/osler_v5/arm_F0.yaml").loss_weighting == "per_step"
