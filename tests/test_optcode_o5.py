"""O5 unit tests that need no model: option-code selection, the head's initialisation and export, and the bidirectional mask.

The equalities are the O5 acceptance statements in miniature: at initialisation the option-code readout equals the
letter readout renormalised over the offered letters, and the export replaces only the code rows.
"""

from __future__ import annotations

import pytest
import torch

from meddecide.model.bidirectional import bidirectional_mask, full_attention_bidirectional
from meddecide.model.head import segment_softmax
from meddecide.model.meddecide_model import letter_positions
from meddecide.model.optcode import (
    LETTERS,
    K,
    OptionCodeHead,
    code_names,
    code_token_ids,
    exported_lm_head,
)


def test_code_names_are_letters_then_single_token_two_letter_codes_in_order() -> None:
    names = code_names(30, lambda code: code != "AB")
    assert names[:26] == list(LETTERS)
    assert names[26:] == ["AA", "AC", "AD", "AE"]


def test_code_names_default_size_is_255_and_refuses_when_too_few_codes_exist() -> None:
    assert len(code_names(K, lambda code: True)) == K
    with pytest.raises(ValueError, match="only 26 single-token codes"):
        code_names(40, lambda code: len(code) == 1)


def test_code_names_refuses_a_tokenizer_that_splits_a_letter() -> None:
    with pytest.raises(ValueError, match="a single letter"):
        code_names(5, lambda code: code != "C")


def test_code_token_ids_require_one_token_per_code_and_distinct_ids() -> None:
    assert code_token_ids(["A", "B"], lambda code: [{"A": 10, "B": 11}[code]]) == [10, 11]
    with pytest.raises(ValueError, match="not one token"):
        code_token_ids(["A"], lambda code: [1, 2])
    with pytest.raises(ValueError, match="same token"):
        code_token_ids(["A", "B"], lambda code: [7])


def _random_lm_head(vocab: int = 40, d: int = 6, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(vocab, d, generator=gen)


def test_initialised_head_equals_the_letter_readout_renormalised_over_offered_letters() -> None:
    lm_head = _random_lm_head()
    token_ids = list(range(3, 3 + 26))  # one LM-head row per letter
    head = OptionCodeHead(d_model=6, k=26)
    head.init_from_lm_head(lm_head, token_ids)
    gen = torch.Generator().manual_seed(1)
    hidden = torch.randn(2, 6, generator=gen)
    n_options = [4, 3]
    marker_item = torch.tensor([0] * 4 + [1] * 3)
    code_index = torch.tensor(letter_positions(n_options))
    logits, probs = head(hidden, marker_item, code_index)
    # the zero-shot letter readout: the LM-head logit of each offered letter's token at the answer position
    letter_logits = hidden[marker_item] @ lm_head[token_ids].T
    letter_logits = letter_logits[torch.arange(7), code_index]
    expected = segment_softmax(letter_logits, marker_item, 2)
    assert torch.allclose(logits, letter_logits, atol=1e-5)
    assert torch.allclose(probs, expected, atol=1e-6)
    assert torch.allclose(probs.reshape(-1)[:4].sum(), torch.tensor(1.0), atol=1e-6)


def test_head_rows_follow_the_lm_head_rows_they_were_copied_from() -> None:
    lm_head = _random_lm_head(vocab=50, d=4)
    ids = [9, 2, 30]
    head = OptionCodeHead(d_model=4, k=3)
    head.init_from_lm_head(lm_head, ids)
    assert torch.equal(head.weight.detach(), lm_head[ids].float())
    assert head.token_ids.tolist() == ids


def test_forward_refuses_more_options_than_codes_and_bad_shapes() -> None:
    head = OptionCodeHead(d_model=3, k=2)
    hidden = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="more options"):
        head(hidden, torch.tensor([0, 0, 0]), torch.tensor([0, 1, 2]))
    with pytest.raises(ValueError, match="must be"):
        head(torch.zeros(1, 4), torch.tensor([0]), torch.tensor([0]))


def test_export_replaces_only_the_code_rows_and_native_and_exported_scores_agree() -> None:
    lm_head = _random_lm_head(vocab=40, d=6)
    ids = [5, 6, 7]
    head = OptionCodeHead(d_model=6, k=3)
    head.init_from_lm_head(lm_head, ids)
    with torch.no_grad():
        head.weight.add_(0.5 * torch.randn_like(head.weight))  # a trained head differs from its initialisation
    exported = exported_lm_head(lm_head, head)
    changed = torch.nonzero((exported != lm_head).any(dim=1)).flatten().tolist()
    assert changed == ids  # only the code-token rows moved
    assert torch.equal(exported[ids], head.weight.detach())
    gen = torch.Generator().manual_seed(2)
    hidden = torch.randn(1, 6, generator=gen)
    native_logits, _ = head(hidden, torch.tensor([0, 0, 0]), torch.tensor([0, 1, 2]))
    exported_logits = (hidden @ exported[ids].T)[0]
    assert torch.allclose(native_logits, exported_logits, atol=1e-6)


def test_export_keeps_the_matrix_dtype_and_refuses_nothing_else() -> None:
    lm_head = _random_lm_head(vocab=10, d=4).to(torch.bfloat16)
    head = OptionCodeHead(d_model=4, k=2)
    head.init_from_lm_head(lm_head, [1, 2])
    exported = exported_lm_head(lm_head, head)
    assert exported.dtype == torch.bfloat16
    assert torch.equal(exported[[0, 3]], lm_head[[0, 3]])


def test_bidirectional_mask_lets_every_query_see_every_real_key_and_no_padding() -> None:
    padding = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    mask = bidirectional_mask(padding, batch=2, seq=4, device=torch.device("cpu"))
    assert mask.shape == (2, 1, 4, 4) and mask.dtype == torch.bool
    assert mask[0, 0, :, 0].logical_not().all()  # padded key is never attended
    assert mask[0, 0, :, 1:].all()  # real keys are attended from every query, including earlier ones
    assert bidirectional_mask(None, 1, 3, torch.device("cpu")).all()


def test_full_attention_patch_is_scoped_and_absent_when_disabled() -> None:
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen3_5

    original = qwen3_5.create_causal_mask
    with full_attention_bidirectional(False):
        assert qwen3_5.create_causal_mask is original
    with full_attention_bidirectional(True):
        assert qwen3_5.create_causal_mask is not original
    assert qwen3_5.create_causal_mask is original
    with pytest.raises(RuntimeError), full_attention_bidirectional(True):
        raise RuntimeError("inside the block")
    assert qwen3_5.create_causal_mask is original  # restored even when the block raises
