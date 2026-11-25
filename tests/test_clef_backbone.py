"""Unit tests for the Clef text-tower wrapper (synthetic text tower; no GPU, no checkpoint)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from meddecide.model.clef import ClefTextBackbone


class DummyTower(nn.Module):
    """A stand-in for Qwen3_5TextModel: final hidden states, an optional hidden-state tuple, one extra attribute."""

    def __init__(self, hidden: int, vocab: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, num_hidden_layers=1, vocab_size=vocab)
        self.embed = nn.Embedding(vocab, hidden)
        self.extra_marker = "tower"

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False, **_: object) -> SimpleNamespace:
        hidden = self.embed(input_ids)
        states = (hidden, hidden) if output_hidden_states else None
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=states)


def build() -> tuple[ClefTextBackbone, nn.Linear, DummyTower]:
    tower = DummyTower(hidden=8, vocab=16)
    head = nn.Linear(8, 16, bias=False)
    return ClefTextBackbone(tower, head), head, tower


def test_forward_computes_logits_over_the_whole_sequence() -> None:
    backbone, head, _ = build()
    ids = torch.tensor([[1, 2, 3, 4]])
    out = backbone(input_ids=ids)
    assert out.logits.shape == (1, 4, 16)
    assert torch.allclose(out.logits, head(backbone.model.embed(ids)))
    assert out.hidden_states is None


def test_logits_to_keep_returns_only_the_last_token() -> None:
    backbone, _, _ = build()
    out = backbone(input_ids=torch.tensor([[1, 2, 3, 4]]), logits_to_keep=1)
    assert out.logits.shape == (1, 1, 16)


def test_hidden_states_are_passed_through_unchanged() -> None:
    backbone, _, tower = build()
    out = backbone(input_ids=torch.tensor([[1, 2]]), output_hidden_states=True)
    assert out.hidden_states is not None and len(out.hidden_states) == 2
    assert torch.equal(out.hidden_states[-1], tower.embed(torch.tensor([[1, 2]])))


def test_attributes_fall_through_to_the_tower_and_the_peft_surface_exists() -> None:
    backbone, head, _ = build()
    assert backbone.extra_marker == "tower"  # __getattr__ delegation
    assert backbone.base_model_prefix == "model"
    assert backbone.get_output_embeddings() is head
    assert callable(backbone.prepare_inputs_for_generation)
    assert backbone.prepare_inputs_for_generation(torch.tensor([[1]]))["input_ids"].shape == (1, 1)


def test_training_gradients_reach_the_head_and_the_tower() -> None:
    backbone, head, tower = build()
    out = backbone(input_ids=torch.tensor([[1, 2, 3]]))
    out.logits.sum().backward()
    assert head.weight.grad is not None and float(head.weight.grad.abs().sum()) > 0
    assert tower.embed.weight.grad is not None and float(tower.embed.weight.grad.abs().sum()) > 0
