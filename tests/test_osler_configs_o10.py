"""The O10 config checks (src/meddecide/train/osler_configs.py): pinned bases, O9's head, and arm L's recipe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from meddecide.train.config import StudentConfig, load_config
from meddecide.train.osler_configs import (
    HEAD_SETTINGS,
    PINNED_BASES,
    check_head_matches,
    check_pinned_base,
    recipe_differences,
)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs/osler_v0"


def _arm(name: str) -> StudentConfig:
    return load_config(CONFIGS / f"arm_{name}.yaml")


def _with(config: StudentConfig, **changes: Any) -> StudentConfig:
    data = config.to_dict()
    data.update(changes)
    return StudentConfig.from_dict(data)


def test_the_matched_arms_differ_only_in_base_output_and_head() -> None:
    reference = _arm("L")
    for arm in ("P", "N"):
        config = _arm(arm)
        assert recipe_differences(config, reference) == [], arm
        assert (config.readout, config.bidirectional_full_attention) == HEAD_SETTINGS[arm], arm


def test_a_recipe_change_is_listed_and_base_output_and_head_changes_are_not() -> None:
    reference = _arm("L")
    assert recipe_differences(_with(reference, batch_size=16), reference) == ["batch_size"]
    assert recipe_differences(_with(reference, lr=0.002, lora_lr=0.0002), reference) == ["lora_lr", "lr"]
    moved = _with(reference, base_model="Qwen/Qwen3.5-9B", revision=PINNED_BASES["Qwen/Qwen3.5-9B"],
                  output_dir="outputs/osler_v0/O10/osler_9b", readout="pointer", bidirectional_full_attention=True)
    assert recipe_differences(moved, reference) == []


def test_the_base_must_be_an_o10_base_at_its_pinned_revision() -> None:
    reference = _arm("L")
    nine_b = _with(reference, base_model="Qwen/Qwen3.5-9B", revision=PINNED_BASES["Qwen/Qwen3.5-9B"])
    check_pinned_base(nine_b)  # passes
    with pytest.raises(ValueError, match="not the pinned"):
        check_pinned_base(_with(nine_b, revision="0" * 40))
    with pytest.raises(ValueError, match="not an O10 base"):
        check_pinned_base(_with(nine_b, base_model="Qwen/Qwen3.5-27B"))
    with pytest.raises(ValueError, match="not an O10 base"):
        check_pinned_base(reference)  # the 4B arm config is not an O10 config


def test_the_head_must_be_the_one_o9_chose() -> None:
    arm_l, arm_p, arm_n = _arm("L"), _arm("P"), _arm("N")
    check_head_matches(arm_l, "L")
    check_head_matches(arm_p, "P")
    check_head_matches(arm_n, "N")
    with pytest.raises(ValueError, match="is not O9's choice"):
        check_head_matches(arm_l, "P")
    with pytest.raises(ValueError, match="is not O9's choice"):
        check_head_matches(arm_p, "N")
    with pytest.raises(ValueError, match="expected one of"):
        check_head_matches(arm_l, "X")
