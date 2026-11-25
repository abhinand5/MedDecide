"""osler_v7 T6: the KL-to-base inclusion mask (CPU; no model)."""

from __future__ import annotations

import pytest
import torch

from meddecide.train.trainer import kl_anchor_mask


def test_mask_includes_v7_cells() -> None:
    mask = kl_anchor_mask(["v7_c1_dx", "mimic_icd_support_v1", "v7_c4_x"], include=("v7_c",))
    assert mask.dtype == torch.bool
    assert mask.tolist() == [True, False, True]


def test_mask_excludes_prefixes_as_before() -> None:
    mask = kl_anchor_mask(["v7_c1", "mimic_icd_support_v1", "medmcqa_4opt_v1"], exclude=("mimic_", "v7_"))
    assert mask.tolist() == [False, False, True]


def test_mask_defaults_to_every_row() -> None:
    assert kl_anchor_mask(["a", "b"]).tolist() == [True, True]


def test_mask_rejects_include_and_exclude_together() -> None:
    with pytest.raises(ValueError, match="cannot use both"):
        kl_anchor_mask(["a"], include=("v7_c",), exclude=("mimic_",))
