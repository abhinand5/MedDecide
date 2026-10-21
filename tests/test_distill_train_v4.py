"""Y6 distillation (osler_v4, D48): a teacher distribution travels with its item, follows option-order augmentation, and enters
the loss as KL(teacher || model) over the items that have one."""

from __future__ import annotations

import pytest
import torch

from meddecide.bench.schema import SoftItem
from meddecide.model.meddecide_model import EncodedItem, reorder_options
from meddecide.train.config import StudentConfig
from meddecide.train.losses import kl_to_reference, teacher_reference


def _item(options: int = 3, teacher: list[float] | None = None) -> SoftItem:
    keys = ["A", "B", "C", "D"][:options]
    return SoftItem.model_validate({
        "item_id": "0123456789abcdef", "tier": "fresh", "source": "src", "source_record_id": "r1",
        "source_url": "https://example.org/r1", "source_license": "public-domain", "record_date": "2026-01-01",
        "split": "train", "template_id": "t_choice_v1", "skill": "skill", "qtype": "choice",
        "state": "A patient has a finding.", "question": "Which option?",
        "options": [{"key": k, "label": f"label {k}"} for k in keys], "gold": "A", "teacher_probs": teacher,
    })


def test_a_soft_item_without_a_teacher_has_none_and_is_an_item():
    item = _item()
    assert item.teacher_probs is None and item.teacher is None


def test_a_teacher_that_is_not_a_distribution_over_the_options_is_rejected():
    with pytest.raises(ValueError, match="values for 3 options"):
        _item(3, [0.5, 0.5])
    with pytest.raises(ValueError, match="not a distribution"):
        _item(3, [0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="not a distribution"):
        _item(3, [1.2, -0.2, 0.0])


def test_reordering_moves_the_teacher_with_its_option_and_the_gold_follows_its_content():
    out = reorder_options(_item(3, [0.5, 0.3, 0.2]), [2, 0, 1])  # display order C, A, B
    assert [o.key for o in out.options] == ["C", "A", "B"]
    assert out.teacher_probs == pytest.approx([0.2, 0.5, 0.3])
    assert out.gold == "A"
    assert out.teacher_probs[[o.key for o in out.options].index("A")] == pytest.approx(0.5)


def test_reordering_an_item_without_a_teacher_keeps_none():
    assert reorder_options(_item(3), [1, 2, 0]).teacher_probs is None


def test_an_encoded_item_has_no_teacher_unless_one_is_given():
    enc = EncodedItem(item_id="x", qtype="choice", template_id="t", source="s", option_keys=["A", "B"],
                      original_option_keys=["A", "B"], gold_index=0, input_ids=[1, 2], marker_positions=[1, 1],
                      option_end_positions=[1, 1], answer_position=1, prompt_tokens=2, truncated=False)
    assert enc.teacher_probs is None


def test_the_teacher_reference_is_flat_in_display_order_with_a_mask_for_the_items_with_one():
    flat, mask = teacher_reference([[0.6, 0.4], None, [0.1, 0.9, 0.0]], [2, 2, 3])
    assert flat.tolist() == pytest.approx([0.6, 0.4, 0.5, 0.5, 0.1, 0.9, 0.0])
    assert mask.tolist() == [True, False, True]


def test_the_teacher_reference_rejects_a_row_with_the_wrong_number_of_values():
    with pytest.raises(ValueError, match="values for 3 options"):
        teacher_reference([[0.5, 0.5]], [3])


def test_distillation_is_the_mean_kl_from_teacher_to_model_over_the_items_with_a_teacher():
    probs = torch.tensor([0.5, 0.5, 0.2, 0.3, 0.5, 0.7, 0.3], requires_grad=True)
    item_index = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    teacher_flat, mask = teacher_reference([[0.6, 0.4], None, [0.1, 0.9]], [2, 3, 2])
    value = kl_to_reference(probs, teacher_flat, item_index, 3, mask)
    kl_item0 = 0.6 * (torch.log(torch.tensor(0.6 / 0.5))) + 0.4 * (torch.log(torch.tensor(0.4 / 0.5)))
    kl_item2 = 0.1 * (torch.log(torch.tensor(0.1 / 0.7))) + 0.9 * (torch.log(torch.tensor(0.9 / 0.3)))
    assert float(value.detach()) == pytest.approx(float((kl_item0 + kl_item2) / 2), rel=1e-5)
    value.backward()
    assert probs.grad is not None and bool(torch.isfinite(probs.grad).all())


def test_distillation_with_no_teacher_items_is_zero():
    probs = torch.tensor([0.5, 0.5], requires_grad=True)
    flat, mask = teacher_reference([None], [2])
    value = kl_to_reference(probs, flat, torch.tensor([0, 0]), 1, mask)
    assert float(value.detach()) == 0.0


def test_distill_weight_defaults_off_and_must_not_be_negative():
    assert StudentConfig.__dataclass_fields__["distill_weight"].default == 0.0
    assert StudentConfig.from_dict({"distill_weight": 0.5}).distill_weight == 0.5
    with pytest.raises(ValueError, match="distill_weight must be >= 0"):
        StudentConfig.from_dict({"distill_weight": -0.1})


def test_a_mix_row_with_a_teacher_becomes_a_soft_item_and_one_without_stays_plain():
    from meddecide.train.mix_items import mix_row_to_item

    row = {"item_id": "0123456789abcdef", "tier": "fresh", "source": "src", "source_record_id": "r1",
           "source_url": "https://example.org/r1", "source_license": "public-domain", "record_date": "2026-01-01",
           "split": "train", "template_id": "broad_x_v1", "skill": "s", "qtype": "choice", "state": "A patient.",
           "question": "Which?", "options": [{"key": "A", "label": "a"}, {"key": "B", "label": "b"},
                                             {"key": "C", "label": "c"}], "gold": "B"}
    assert getattr(mix_row_to_item(row), "teacher_probs", None) is None
    soft = mix_row_to_item({**row, "teacher_probs": [0.2, 0.7, 0.1], "teacher": "t"})
    assert isinstance(soft, SoftItem)
    assert soft.teacher_probs == [0.2, 0.7, 0.1] and soft.teacher == "t"
    assert [o.key for o in soft.options] == ["A", "B", "C"] and soft.gold == "B"
