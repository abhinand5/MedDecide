"""Batch layout: the default rounds rows to the chunk length; ``round_to_chunk=False`` does not."""

from __future__ import annotations

from types import SimpleNamespace

from meddecide.model.meddecide_model import LINEAR_ATTENTION_CHUNK, EncodedItem, MedDecideModel


def _encoded(n: int, item_id: str) -> EncodedItem:
    return EncodedItem(
        item_id=item_id, qtype="choice", template_id="t", source="s",
        option_keys=["A", "B"], original_option_keys=["A", "B"], gold_index=0,
        input_ids=list(range(1, n + 1)), marker_positions=[2, n - 3],
        option_end_positions=[3, n - 2], answer_position=n - 1, prompt_tokens=n, truncated=False,
    )


def _model() -> SimpleNamespace:
    return SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))


def test_default_rounds_the_row_length_up_to_the_chunk() -> None:
    batch = MedDecideModel.collate(_model(), [_encoded(100, "a"), _encoded(70, "b")])
    assert batch.input_ids.shape == (2, 2 * LINEAR_ATTENTION_CHUNK)
    assert int(batch.attention_mask[1].sum()) == 70


def test_unrounded_batch_is_as_long_as_its_longest_item() -> None:
    batch = MedDecideModel.collate(_model(), [_encoded(100, "a"), _encoded(70, "b")], round_to_chunk=False)
    assert batch.input_ids.shape == (2, 100)
    assert int(batch.attention_mask[1].sum()) == 70
    assert batch.answer_positions.tolist() == [99, 99]


def test_single_unrounded_item_has_no_pad_tokens_and_its_own_positions() -> None:
    batch = MedDecideModel.collate(_model(), [_encoded(70, "c")], round_to_chunk=False)
    assert int(batch.attention_mask.sum()) == 70
    assert batch.marker_positions.tolist() == [2, 67]
    assert batch.option_end_positions.tolist() == [3, 68]
    assert batch.answer_positions.tolist() == [69]


def test_single_rounded_item_shifts_markers_by_its_own_padding() -> None:
    batch = MedDecideModel.collate(_model(), [_encoded(70, "d")])
    shift = 2 * LINEAR_ATTENTION_CHUNK - 70
    assert batch.input_ids.shape == (1, 2 * LINEAR_ATTENTION_CHUNK)
    assert batch.marker_positions.tolist() == [2 + shift, 67 + shift]
    assert batch.answer_positions.tolist() == [2 * LINEAR_ATTENTION_CHUNK - 1]


def test_exact_length_batches_cover_every_position_once_and_never_mix_lengths() -> None:
    from meddecide.model.meddecide_model import exact_length_batches

    lengths = [70, 120, 70, 70, 9, 120, 70]
    batches = exact_length_batches(lengths, batch_size=2)
    assert sorted(p for b in batches for p in b) == list(range(len(lengths)))
    for batch in batches:
        assert len({lengths[p] for p in batch}) == 1
        assert len(batch) <= 2
    assert [lengths[b[0]] for b in batches] == [120, 70, 70, 9]
