"""MedDecide model: frozen base + decision-path LoRA + pointer head.

This is decision D3 of the program plan. The base model is frozen and never updated; a LoRA
adapter (rank and target modules from config; default r=16 on the attention and MLP
projections) is the only part of the *language* path that trains, and it can be disabled so
the untouched base can generate. The pointer head is a separate module that reads hidden
states out of the base and never writes to it.

Everything the head needs is derived from the existing harness rendering
(:func:`meddecide.eval.readout.render_prompt` over a canonicalised item), so a scored item
and a zero-shot item are the same prompt.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from meddecide.bench.schema import Item, QuestionType
from meddecide.eval.harness import plan_batches
from meddecide.eval.metrics import accuracy, brier_score, macro_accuracy
from meddecide.eval.readout import (
    LetterVariant,
    canonicalise_options,
    expected_level,
    render_prompt,
)
from meddecide.model.bidirectional import full_attention_bidirectional
from meddecide.model.head import HeadSettings, PointerHead
from meddecide.model.markers import locate_option_markers
from meddecide.model.optcode import K as OPTION_CODES
from meddecide.model.optcode import OptionCodeHead, code_names, code_token_ids, exported_lm_head

# "attention and MLP projections" for this hybrid stack: Qwen3.5 alternates full-attention
# and linear-attention layers, so the attention targets include both families (quantities
# that are *not* projections — the linear attention conv1d — are deliberately not adapted).
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class LoraSettings:
    """LoRA shape and placement. ``lora=None`` on the model means no adapter at all."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS

    def to_dict(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoraSettings:
        return cls(
            r=int(data.get("r", 16)),
            alpha=int(data.get("alpha", 32)),
            dropout=float(data.get("dropout", 0.0)),
            target_modules=tuple(data.get("target_modules", DEFAULT_LORA_TARGETS)),
        )


# module-level singleton so the constructor default is not a call (ruff B008); ``lora=None``
# passed to the constructor means "no adapter at all" (the base-ablation path)
DEFAULT_LORA = LoraSettings()

# The gated-delta reference kernel chunks each sequence from index 0 in blocks of this size.
# Left padding shifts where a real token falls in its chunk, so the batch length is rounded to
# a multiple of it: every row then has the same chunk offset as the item scored alone.
LINEAR_ATTENTION_CHUNK = 64


def segment_softmax(flat: Tensor, n_options: Sequence[int]) -> Tensor:
    """Softmax over each item's options, for flat per-option logits laid out item by item."""
    if int(flat.numel()) != int(sum(n_options)):
        raise ValueError("flat logits do not match the per-item option counts")
    pieces, start = [], 0
    for n in n_options:
        pieces.append(torch.softmax(flat[start : start + int(n)], dim=0))
        start += int(n)
    return torch.cat(pieces) if pieces else flat


def letter_positions(n_options: Sequence[int]) -> list[int]:
    """Within-item option index of every flat option (0 for A, 1 for B, ...)."""
    out: list[int] = []
    for n in n_options:
        out.extend(range(int(n)))
    return out


def exact_length_batches(lengths: Sequence[int], *, batch_size: int) -> list[list[int]]:
    """Group item positions so that every batch holds items of one exact length.

    A batch of equal-length rows has no padding at all, so each item is computed as it would be alone
    (up to batch-dimension noise). Items are grouped by length, longest group first, and each group is
    split into batches of at most ``batch_size``. Every position appears exactly once.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    groups: dict[int, list[int]] = {}
    for position, length in enumerate(lengths):
        groups.setdefault(int(length), []).append(position)
    batches: list[list[int]] = []
    for length in sorted(groups, reverse=True):
        members = groups[length]
        for start in range(0, len(members), batch_size):
            batches.append(members[start : start + batch_size])
    return batches


def reorder_options(item: Item, permutation: Sequence[int]) -> Item:
    """Copy of ``item`` with options permuted; the gold **follows its content**.

    ``permutation`` lists the source positions in display order. Gold is re-derived from the
    moved option's key, which is what makes option-order augmentation label-preserving.
    """
    order = list(permutation)
    if sorted(order) != list(range(item.n_options)):
        raise ValueError(f"permutation {order} is not a permutation of {item.n_options} options")
    options = [item.options[i] for i in order]
    new_gold_index = order.index(item.gold_index)
    return item.model_copy(update={"options": options, "gold": options[new_gold_index].key})


@dataclass(frozen=True)
class EncodedItem:
    """One item prepared for the head: padded-free token ids and marker positions."""

    item_id: str
    qtype: str
    template_id: str
    source: str
    option_keys: list[str]
    original_option_keys: list[str]
    gold_index: int
    input_ids: list[int]
    marker_positions: list[int]
    option_end_positions: list[int]
    answer_position: int
    prompt_tokens: int
    truncated: bool
    marker_notes: list[str] = field(default_factory=list)

    @property
    def n_options(self) -> int:
        return len(self.marker_positions)

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "qtype": self.qtype,
            "template_id": self.template_id,
            "n_options": self.n_options,
            "n_tokens": self.n_tokens,
            "option_line_tokens": float(
                sum(e - p + 1 for p, e in zip(
                    self.marker_positions, self.option_end_positions, strict=True))
            ) / max(1, self.n_options),
            "prompt_tokens": self.prompt_tokens,
            "truncated": self.truncated,
            "marker_notes": self.marker_notes,
        }


@dataclass
class ModelBatch:
    """A padded batch with flat marker indices, so mixed option counts are allowed."""

    input_ids: Tensor
    attention_mask: Tensor
    marker_positions: Tensor
    option_end_positions: Tensor
    marker_item: Tensor
    answer_positions: Tensor
    n_options: Tensor
    gold_flat: Tensor
    encoded: list[EncodedItem]

    @property
    def batch_size(self) -> int:
        return len(self.encoded)

    @property
    def real_tokens(self) -> int:
        return int(self.attention_mask.sum())

    @property
    def padded_tokens(self) -> int:
        return int(self.input_ids.numel())

    @property
    def n_markers(self) -> int:
        return int(self.marker_positions.numel())

    def to(self, device: str | torch.device) -> ModelBatch:
        """Move every tensor to ``device`` **in place** and return self.

        In place on purpose: callers keep index tensors (``gold_flat``, ``marker_item``) from
        the same object and mixing devices is a silent crash waiting to happen.
        """
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.marker_positions = self.marker_positions.to(device)
        self.option_end_positions = self.option_end_positions.to(device)
        self.marker_item = self.marker_item.to(device)
        self.answer_positions = self.answer_positions.to(device)
        self.n_options = self.n_options.to(device)
        self.gold_flat = self.gold_flat.to(device)
        return self


@dataclass
class ScoredItems:
    """Batched inference output before it is turned into public predictions."""

    items: list[Item]
    option_keys: list[list[str]]
    original_option_keys: list[list[str]]
    logits: list[np.ndarray]
    probs: list[np.ndarray]
    gold_indices: list[int]
    prompt_tokens: list[int]
    latency_s: list[float]
    wall_clock_s: float
    batch_sizes: list[int]

    def __len__(self) -> int:
        return len(self.items)

    @property
    def qtypes(self) -> list[str]:
        return [str(item.qtype) for item in self.items]

    @property
    def total_prompt_tokens(self) -> int:
        return int(sum(self.prompt_tokens))

    def to_probs_array(self, max_options: int | None = None) -> np.ndarray:
        """Probs as a padded ``(n_items, max_options)`` array; unused cells are NaN."""
        width = max_options or max(len(p) for p in self.probs)
        out = np.full((len(self.probs), width), np.nan, dtype=np.float64)
        for i, probs in enumerate(self.probs):
            out[i, : len(probs)] = probs
        return out

    def metrics(self) -> dict[str, Any]:
        """Accuracy, Brier and friends over this set, with explicit denominators (R5)."""
        predicted = [keys[int(np.argmax(p))] for keys, p in zip(self.option_keys, self.probs,
                                                                strict=True)]
        gold = [keys[g] for keys, g in zip(self.option_keys, self.gold_indices, strict=True)]
        acc = accuracy(predicted, gold)
        # brier_score wants rectangular rows, and items in one set may have different option
        # counts, so the tested harness metric is applied per option-count group and combined
        # with explicit weights (R5: the parts sum to the total).
        groups: dict[int, list[int]] = {}
        for i, probs in enumerate(self.probs):
            groups.setdefault(len(probs), []).append(i)
        weighted = 0.0
        counted = 0
        for width, indices in sorted(groups.items()):
            rows = np.stack([self.probs[i] for i in indices])
            if rows.shape[1] != width:  # pragma: no cover - shape guard
                raise AssertionError("option-count grouping is inconsistent")
            weighted += len(indices) * brier_score(
                rows, [self.gold_indices[i] for i in indices]
            )
            counted += len(indices)
        if counted != len(self.probs):  # pragma: no cover - shape guard
            raise AssertionError("Brier denominator disagrees with the item count")
        return {
            "n": acc.n,
            "n_correct": acc.n_correct,
            "accuracy": acc.micro,
            "majority_baseline": acc.majority_baseline,
            "majority_label": acc.majority_label,
            "macro_accuracy": macro_accuracy(predicted, gold),
            "brier": weighted / counted,
            "mean_nll": float(
                np.mean([-np.log(max(float(p[g]), 1e-12)) for p, g in
                         zip(self.probs, self.gold_indices, strict=True)])
            ),
            "wall_clock_s": self.wall_clock_s,
            "total_prompt_tokens": self.total_prompt_tokens,
            "p50_latency_s": float(np.percentile(self.latency_s, 50)),
            "p95_latency_s": float(np.percentile(self.latency_s, 95)),
            "batches": len(self.batch_sizes),
        }

    def per_qtype_metrics(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for qtype in sorted(set(self.qtypes)):
            idx = [i for i, q in enumerate(self.qtypes) if q == qtype]
            subset = ScoredItems(
                items=[self.items[i] for i in idx],
                option_keys=[self.option_keys[i] for i in idx],
                original_option_keys=[self.original_option_keys[i] for i in idx],
                logits=[self.logits[i] for i in idx],
                probs=[self.probs[i] for i in idx],
                gold_indices=[self.gold_indices[i] for i in idx],
                prompt_tokens=[self.prompt_tokens[i] for i in idx],
                latency_s=[self.latency_s[i] for i in idx],
                wall_clock_s=self.wall_clock_s,
                batch_sizes=self.batch_sizes,
            )
            out[qtype] = subset.metrics()
        return out


@dataclass(frozen=True)
class Decision:
    """One item's answer: the full distribution over exactly its offered options."""

    item_id: str
    qtype: str
    template_id: str
    source: str
    option_keys: list[str]
    original_option_keys: list[str]
    probs: list[float]
    argmax_index: int
    gold_key: str
    correct: bool
    expected_level: float | None
    n_options: int
    prompt_tokens: int
    latency_s: float
    temperature: float

    @property
    def argmax_key(self) -> str:
        return self.option_keys[self.argmax_index]

    def to_json(self) -> dict[str, Any]:
        """Serialisable row. Option labels are deliberately absent (the repo is public)."""
        return {
            "item_id": self.item_id,
            "qtype": self.qtype,
            "template_id": self.template_id,
            "source": self.source,
            "option_keys": self.option_keys,
            "original_option_keys": self.original_option_keys,
            "option_probs": [round(float(p), 6) for p in self.probs],
            "argmax_key": self.argmax_key,
            "gold_key": self.gold_key,
            "correct": self.correct,
            "expected_level": self.expected_level,
            "n_options": self.n_options,
            "prompt_tokens": self.prompt_tokens,
            "latency_s": round(float(self.latency_s), 5),
            "temperature": float(self.temperature),
        }


class MedDecideModel:
    """The trained architecture: frozen base (+ optional LoRA) and a pointer head."""

    def __init__(
        self,
        base_model_id: str,
        *,
        lora: LoraSettings | None = DEFAULT_LORA,
        head_settings: HeadSettings | None = None,
        dtype: str = "bfloat16",
        device: str = "cuda:0",
        variant: LetterVariant = "bare",
        revision: str | None = None,
        trust_remote_code: bool = False,
        max_prompt_tokens: int = 16384,
        head: PointerHead | None = None,
        calibration: dict[str, float] | None = None,
        readout: str = "pointer",
        bidirectional_full_attention: bool = False,
        code_count: int = OPTION_CODES,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if readout not in ("pointer", "letter", "option_code"):
            raise ValueError(f"readout must be 'pointer', 'letter' or 'option_code', got {readout!r}")
        self.readout = readout
        self.bidirectional_full_attention = bool(bidirectional_full_attention)
        self.optcode: OptionCodeHead | None = None
        self.optcode_names: list[str] = []
        self.optcode_token_ids: list[int] = []

        self.base_model_id = base_model_id
        self.revision = revision
        self.dtype_name = dtype
        self.device = device
        self.variant = variant
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.lora = lora
        self.head_settings = head_settings or HeadSettings()
        self.calibration: dict[str, float] = dict(calibration or {})
        self.torch_dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        # the answer position is the last token, so padding on the right would move it
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            revision=revision,
            dtype=self.torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        self.peft_model = None
        if lora is not None:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=lora.r,
                lora_alpha=lora.alpha,
                lora_dropout=lora.dropout,
                bias="none",
                target_modules=list(lora.target_modules),
                task_type="CAUSAL_LM",
            )
            self.peft_model = get_peft_model(self.base, lora_config)
            self.peft_model.eval()
        d_model = int(self.get_decoder().config.hidden_size)
        if readout == "pointer":
            self.head = head if head is not None else PointerHead(d_model, self.head_settings)
            self.head.to(device)
            self.head.eval()
        elif readout == "option_code":
            if head is not None:
                raise ValueError("the option-code readout takes no pointer head")
            self.head = None
            self.optcode = OptionCodeHead(d_model, code_count).to(device)
            self._init_option_code()
        else:
            # the letter readout has no head: the answer's LM-head logits for the option letters are
            # the scores, and the LoRA adapter is the only trainable part (ADVISORY student_v1 V7)
            if head is not None:
                raise ValueError("a letter readout takes no head")
            self.head = None
        self._key_token_cache: dict[str, int] = {}

    # ---- structure ----------------------------------------------------------
    def _init_option_code(self) -> None:
        """Copy the LM-head rows of the code tokens into the option-code head (so it equals the letter readout)."""
        if self.optcode is None:
            raise RuntimeError("the option-code head is not built")

        def encode(code: str) -> list[int]:
            text = f" {code}" if self.variant == "space" else code
            return self.tokenizer.encode(text, add_special_tokens=False)

        names = code_names(self.optcode.k, lambda code: len(encode(code)) == 1)
        ids = code_token_ids(names, encode)
        causal = self.peft_model.get_base_model() if self.peft_model is not None else self.base
        self.optcode.init_from_lm_head(causal.lm_head.weight, ids)
        self.optcode_names = names
        self.optcode_token_ids = ids

    def get_decoder(self) -> nn.Module:
        """The text model that produces hidden states (inside the PEFT wrapper if any)."""
        causal = self.peft_model.get_base_model() if self.peft_model is not None else self.base
        return causal.model if hasattr(causal, "model") else causal

    @property
    def d_model(self) -> int:
        if self.head is None:
            return int(self.get_decoder().config.hidden_size)
        return int(self.head.d_model)

    @property
    def trainable_parameter_count(self) -> int:
        head_params = 0 if self.head is None else int(
            sum(p.numel() for p in self.head.parameters() if p.requires_grad))
        if self.optcode is not None:
            head_params += int(sum(p.numel() for p in self.optcode.parameters() if p.requires_grad))
        return head_params + (
            int(sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad))
            if self.peft_model is not None
            else 0
        )

    def trainable_param_groups(
        self, *, lr: float, lora_lr: float, weight_decay: float = 0.0
    ) -> list[dict[str, Any]]:
        """AdamW parameter groups: the head at ``lr``, the adapter at ``lora_lr``.

        A single learning rate for both is possible (``lora_lr == lr``); they are separate
        because the head is randomly initialised and the adapter starts at zero, so they
        tolerate different step sizes.
        """
        groups: list[dict[str, Any]] = []
        if self.head is not None:
            groups.append({"params": [p for p in self.head.parameters() if p.requires_grad], "lr": lr,
                           "weight_decay": weight_decay, "name": "head"})
        if self.optcode is not None:
            groups.append({"params": [p for p in self.optcode.parameters() if p.requires_grad], "lr": lr,
                           "weight_decay": weight_decay, "name": "optcode"})
        if self.peft_model is not None:
            groups.append(
                {
                    "params": [p for p in self.peft_model.parameters() if p.requires_grad],
                    "lr": lora_lr,
                    "weight_decay": weight_decay,
                    "name": "lora",
                }
            )
        return groups

    @contextmanager
    def adapter_disabled(self) -> Iterator[None]:
        """Run the base exactly as it is, adapter bypassed (no-op with no adapter)."""
        if self.peft_model is None:
            with nullcontext():
                yield
        else:
            with self.peft_model.disable_adapter():
                yield

    @property
    def adapter_enabled(self) -> bool:
        return self.peft_model is not None

    def train_mode(self, adapter: bool = True) -> None:
        """LoRA/head in train mode, the frozen base always in eval mode.

        Only LoRA submodules are switched to train; the base weights are frozen and its
        dropout stays off, so a training step cannot change how the base itself behaves.
        """
        self.base.eval()
        if self.peft_model is not None:
            for name, module in self.peft_model.named_modules():
                if "lora_" in name:
                    module.train(adapter)
        if self.head is not None:
            self.head.train(adapter)

    def eval_mode(self) -> None:
        if self.peft_model is not None:
            self.peft_model.eval()
        self.base.eval()
        if self.head is not None:
            self.head.eval()

    # ---- encoding -----------------------------------------------------------
    def _key_token(self, key: str) -> int:
        if key not in self._key_token_cache:
            from meddecide.eval.readout import key_token_id

            self._key_token_cache[key] = key_token_id(key, self.tokenizer, self.variant)
        return self._key_token_cache[key]

    def encode_item(
        self,
        item: Item,
        *,
        permutation: Sequence[int] | None = None,
        max_prompt_tokens: int | None = None,
        allow_marker_mismatch: bool = False,
    ) -> EncodedItem:
        """Canonicalise, optionally permute options, render and tokenize one item.

        The marker positions are the option key tokens of the *rendered* prompt, verified
        against the harness's key tokens; the answer position is the final prompt token.

        With ``permutation`` the options are reordered **before** canonicalisation, so the
        prompt still lists its options as ``A.``, ``B.``, … and the gold letter is re-derived
        from the option's content: option-order augmentation never changes which answer is
        correct.
        """
        if permutation is not None:
            item = reorder_options(item, permutation)
        canonical, original_keys = canonicalise_options(item)
        markers = locate_option_markers(canonical, self.tokenizer, variant=self.variant)
        notes: list[str] = []
        if not markers.matches_harness_key_tokens and not allow_marker_mismatch:
            raise ValueError(
                f"{item.item_id}: located option marker tokens disagree with the harness key "
                f"tokens ({markers.mismatches}); the pointer head would score a token the "
                f"prompt does not show"
            )
        for mismatch in markers.mismatches:
            notes.append(f"marker_mismatch:{mismatch['option_key']}")
        for i, single in enumerate(markers.single_token_key):
            if not single:
                notes.append(f"multi_token_key:{markers.option_keys[i]}")
            elif markers.merged_with_prefix[i]:
                notes.append(f"merged_marker:{markers.option_keys[i]}")

        limit = self.max_prompt_tokens if max_prompt_tokens is None else int(max_prompt_tokens)
        input_ids = markers.input_ids
        positions = list(markers.positions)
        end_positions = list(markers.end_positions)
        answer_position = markers.answer_position
        truncated = False
        if len(input_ids) > limit:
            drop = len(input_ids) - limit
            input_ids = input_ids[drop:]
            positions = [p - drop for p in positions]
            end_positions = [p - drop for p in end_positions]
            answer_position -= drop
            truncated = True
        if any(p < 0 for p in positions) or any(p < 0 for p in end_positions):
            raise ValueError(
                f"{item.item_id}: max_prompt_tokens={limit} truncates the options block away"
            )
        canonical_gold_index = canonical.gold_index
        return EncodedItem(
            item_id=item.item_id,
            qtype=str(item.qtype),
            template_id=item.template_id,
            source=item.source,
            option_keys=list(markers.option_keys),
            original_option_keys=list(original_keys),
            gold_index=canonical_gold_index,
            input_ids=input_ids,
            marker_positions=positions,
            option_end_positions=end_positions,
            answer_position=answer_position,
            prompt_tokens=len(markers.input_ids),
            truncated=truncated,
            marker_notes=notes,
        )

    def collate(self, encoded: Sequence[EncodedItem], *, round_to_chunk: bool = True) -> ModelBatch:
        """Left-pad a list of encoded items and build the flat marker index tensors.

        ``round_to_chunk=True`` (the default, used for training and the V2 fixed batched path)
        rounds the row length up to a multiple of ``LINEAR_ATTENTION_CHUNK``. ``False`` gives a
        row exactly as long as its longest item: with one item that is the item with no padding
        at all, the canonical computation that V2 found the rounding to perturb.
        """
        if not encoded:
            raise ValueError("cannot collate an empty batch")
        pad_id = int(self.tokenizer.pad_token_id)
        longest = max(e.n_tokens for e in encoded)
        if round_to_chunk:
            longest = -(-longest // LINEAR_ATTENTION_CHUNK) * LINEAR_ATTENTION_CHUNK
        max_len = longest
        input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
        marker_positions: list[int] = []
        option_end_positions: list[int] = []
        marker_item: list[int] = []
        n_options: list[int] = []
        gold_flat: list[int] = []
        offset = 0
        for row, enc in enumerate(encoded):
            shift = max_len - enc.n_tokens
            row_ids = torch.tensor(enc.input_ids, dtype=torch.long)
            input_ids[row, shift:] = row_ids
            attention_mask[row, shift:] = 1
            for pos, end in zip(enc.marker_positions, enc.option_end_positions, strict=True):
                marker_positions.append(pos + shift)
                option_end_positions.append(end + shift)
                marker_item.append(row)
            n_options.append(enc.n_options)
            gold_flat.append(offset + enc.gold_index)
            offset += enc.n_options
        return ModelBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            marker_positions=torch.tensor(marker_positions, dtype=torch.long),
            option_end_positions=torch.tensor(option_end_positions, dtype=torch.long),
            marker_item=torch.tensor(marker_item, dtype=torch.long),
            answer_positions=torch.full((len(encoded),), max_len - 1, dtype=torch.long),
            n_options=torch.tensor(n_options, dtype=torch.long),
            gold_flat=torch.tensor(gold_flat, dtype=torch.long),
            encoded=list(encoded),
        )

    # ---- forward ------------------------------------------------------------
    def hidden_states(self, batch: ModelBatch) -> Tensor:
        """Final-layer hidden states of the frozen base (plus LoRA if enabled).

        The base is *frozen*, not detached: the LoRA adapter still needs gradients, so this
        must be called outside ``inference_mode`` during training.
        """
        # Without explicit positions the base numbers a left-padded row from the pad, so a real
        # token's RoPE position depends on how much padding its batch added. Counting only the
        # real tokens makes every item's positions 0..n-1 whatever its batch layout.
        position_ids = (batch.attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        kwargs: dict[str, Any] = {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "position_ids": position_ids,
            "use_cache": False,
            "output_hidden_states": True,
        }
        model = self.peft_model if self.peft_model is not None else self.base
        with full_attention_bidirectional(self.bidirectional_full_attention):
            try:
                output = model(**kwargs, logits_to_keep=1)
            except TypeError:
                # older transformers without logits_to_keep: the lm_head then runs over the whole
                # sequence, which is slower but numerically identical for the head
                output = model(**kwargs)
        return output.hidden_states[-1]

    def batch_logits(self, batch: ModelBatch) -> tuple[Tensor, Tensor]:
        """``(flat logits, flat probabilities)`` over exactly the batch's offered options."""
        dev = self.device
        batch = batch.to(dev)
        if self.readout == "option_code":
            return self._option_code_logits(batch)
        if self.head is None:
            return self._letter_logits(batch)
        hidden = self.hidden_states(batch)
        return self.head.option_logits(
            hidden,
            batch.marker_positions,
            batch.marker_item,
            batch.answer_positions,
            batch.batch_size,
            batch.option_end_positions,
        )

    def _letter_logits(self, batch: ModelBatch) -> tuple[Tensor, Tensor]:
        """Letter readout: per offered option, the LM-head logit of its letter at the answer position.

        The softmax is restricted to the offered letters, per item. The flat layout and the return
        shape match the pointer head's, so the same CE + Brier loss and the same metrics apply.
        """
        hidden = self.hidden_states(batch)
        rows = torch.arange(batch.batch_size, device=hidden.device)
        answer = batch.answer_positions.to(hidden.device)
        h = hidden[rows, answer]
        causal = self.peft_model.get_base_model() if self.peft_model is not None else self.base
        vocab = causal.lm_head(h).float()
        n_options = [int(n) for n in batch.n_options.tolist()]
        letters = torch.tensor(
            [self._key_token(chr(ord("A") + j)) for j in letter_positions(n_options)],
            device=vocab.device, dtype=torch.long,
        )
        flat = vocab[batch.marker_item.to(vocab.device), letters]
        return flat, segment_softmax(flat, n_options)

    def _option_code_logits(self, batch: ModelBatch) -> tuple[Tensor, Tensor]:
        """Option-code readout: the answer state dotted with each offered option's code row (D23)."""
        if self.optcode is None:
            raise RuntimeError("the option-code head is not built")
        hidden = self.hidden_states(batch)
        rows = torch.arange(batch.batch_size, device=hidden.device)
        answer = batch.answer_positions.to(hidden.device)
        h = hidden[rows, answer]
        n_options = [int(n) for n in batch.n_options.tolist()]
        code_index = torch.tensor(letter_positions(n_options), device=h.device, dtype=torch.long)
        return self.optcode(h, batch.marker_item.to(h.device), code_index)

    @torch.no_grad()
    def exported_option_code_lm_head(self) -> Tensor:
        """The LM-head matrix with the code-token rows replaced by the trained option-code rows (D23 export)."""
        if self.optcode is None:
            raise ValueError("this model has no option-code head")
        causal = self.peft_model.get_base_model() if self.peft_model is not None else self.base
        return exported_lm_head(causal.lm_head.weight, self.optcode)

    @torch.no_grad()
    def score_items(
        self,
        items: Sequence[Item],
        *,
        batch_size: int = 16,
        max_batch_tokens: int = 8192,
        max_prompt_tokens: int | None = None,
        allow_marker_mismatch: bool = False,
        round_to_chunk: bool = True,
        length_buckets: bool = False,
    ) -> ScoredItems:
        """Score items in batches; returns raw logits and probabilities per item.

        ``round_to_chunk`` is passed to :meth:`collate`; ``batch_size=1, round_to_chunk=False``
        scores every item with no padding (the V2 evaluation path). ``length_buckets=True`` batches
        only items of one exact length (:func:`exact_length_batches`): also padding-free, and faster
        for the same computation. It needs ``round_to_chunk=False``.

        ``latency_s`` is measured wall-clock per item (batch time / batch size), the same
        convention the zero-shot harness uses, so the two paths' latencies are comparable.
        """
        encoded = [
            self.encode_item(
                item,
                max_prompt_tokens=max_prompt_tokens,
                allow_marker_mismatch=allow_marker_mismatch,
            )
            for item in items
        ]
        canonical = [canonicalise_options(item) for item in items]
        if length_buckets:
            if round_to_chunk:
                raise ValueError("length_buckets needs round_to_chunk=False (equal-length rows unpadded)")
            batches = exact_length_batches([e.n_tokens for e in encoded], batch_size=batch_size)
        else:
            batches = plan_batches(
                [e.n_tokens for e in encoded],
                batch_size=batch_size,
                max_batch_tokens=max_batch_tokens,
            )
        logits: list[np.ndarray | None] = [None] * len(encoded)
        probs: list[np.ndarray | None] = [None] * len(encoded)
        latency: list[float] = [0.0] * len(encoded)
        batch_sizes: list[int] = []
        started = time.perf_counter()
        for indices in batches:
            group = [encoded[i] for i in indices]
            batch = self.collate(group, round_to_chunk=round_to_chunk)
            t0 = time.perf_counter()
            flat_logits, flat_probs = self.batch_logits(batch)
            elapsed = time.perf_counter() - t0
            flat_logits_np = flat_logits.float().cpu().numpy()
            flat_probs_np = flat_probs.float().cpu().numpy()
            per_item = elapsed / max(1, len(indices))
            batch_sizes.append(len(indices))
            cursor = 0
            for position, enc in zip(indices, group, strict=True):
                n = enc.n_options
                logits[position] = flat_logits_np[cursor : cursor + n].astype(np.float64)
                probs[position] = flat_probs_np[cursor : cursor + n].astype(np.float64)
                latency[position] = per_item
                cursor += n
        if any(p is None for p in probs):
            raise AssertionError("some items were not scored")
        return ScoredItems(
            items=list(items),
            option_keys=[c[0].option_keys for c in canonical],
            original_option_keys=[c[1] for c in canonical],
            logits=[np.asarray(v) for v in logits],
            probs=[np.asarray(v) for v in probs],
            gold_indices=[c[0].gold_index for c in canonical],
            prompt_tokens=[e.prompt_tokens for e in encoded],
            latency_s=latency,
            wall_clock_s=time.perf_counter() - started,
            batch_sizes=batch_sizes,
        )

    def apply_temperature(
        self, scored: ScoredItems, temperature: dict[str, float] | None = None
    ) -> ScoredItems:
        """Return a copy of ``scored`` with temperatures applied to the logits.

        ``temperature=None`` uses the model's fitted calibration; an explicit dict (including
        ``{}``) *replaces* it, so ``temperature={}`` means "deliberately uncalibrated" rather
        than "silently keep the fit".
        """
        temps = dict(self.calibration) if temperature is None else dict(temperature)
        probs: list[np.ndarray] = []
        for logits, qtype in zip(scored.logits, scored.qtypes, strict=True):
            t = float(temps.get(qtype, temps.get("*", 1.0)))
            if not np.isfinite(t) or t <= 0:
                raise ValueError(f"temperature for {qtype!r} must be positive, got {t}")
            shifted = logits / t
            shifted = shifted - shifted.max()
            exp = np.exp(shifted)
            probs.append(exp / exp.sum())
        return ScoredItems(
            items=scored.items,
            option_keys=scored.option_keys,
            original_option_keys=scored.original_option_keys,
            logits=scored.logits,
            probs=probs,
            gold_indices=scored.gold_indices,
            prompt_tokens=scored.prompt_tokens,
            latency_s=scored.latency_s,
            wall_clock_s=scored.wall_clock_s,
            batch_sizes=scored.batch_sizes,
        )

    def predict(
        self,
        items: Sequence[Item],
        *,
        batch_size: int = 16,
        max_batch_tokens: int = 8192,
        temperature: dict[str, float] | None = None,
        max_prompt_tokens: int | None = None,
    ) -> list[Decision]:
        """Batched inference: the full distribution over the offered options per item.

        ``score`` items also carry the expected level (1-based, lowest level first). The fitted
        per-qtype temperature is applied by default; ``temperature={}`` disables calibration.
        """
        scored = self.score_items(
            items,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
            max_prompt_tokens=max_prompt_tokens,
        )
        temps = dict(self.calibration) if temperature is None else dict(temperature)
        calibrated = self.apply_temperature(scored, temperature) if temps else scored
        decisions: list[Decision] = []
        for i, item in enumerate(scored.items):
            probs = calibrated.probs[i]
            argmax = int(np.argmax(probs))
            gold_index = scored.gold_indices[i]
            keys = scored.option_keys[i]
            decisions.append(
                Decision(
                    item_id=item.item_id,
                    qtype=str(item.qtype),
                    template_id=item.template_id,
                    source=item.source,
                    option_keys=list(keys),
                    original_option_keys=list(scored.original_option_keys[i]),
                    probs=[float(p) for p in probs],
                    argmax_index=argmax,
                    gold_key=keys[gold_index],
                    correct=argmax == gold_index,
                    expected_level=(
                        expected_level(probs) if item.qtype is QuestionType.SCORE else None
                    ),
                    n_options=len(keys),
                    prompt_tokens=scored.prompt_tokens[i],
                    latency_s=scored.latency_s[i],
                    temperature=float(temps.get(str(item.qtype), temps.get("*", 1.0))),
                )
            )
        return decisions

    # ---- generation (identity check) ---------------------------------------
    @torch.no_grad()
    def generate_greedy(
        self, prompts: Sequence[str], *, max_new_tokens: int = 8, seed: int = 0
    ) -> list[list[int]]:
        """Greedy generation, one prompt at a time, adapter state as it currently is.

        One prompt per forward pass is deliberate: batch composition can change kernel
        accumulation order, and the byte-identity check must isolate the adapter, not batching.
        """
        torch.manual_seed(seed)
        model = self.peft_model if self.peft_model is not None else self.base
        was_training = model.training
        model.eval()
        out: list[list[int]] = []
        for prompt in prompts:
            encoded = self.tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            generated = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False
            )
            new_tokens = generated[0, encoded["input_ids"].shape[1] :]
            out.append([int(t) for t in new_tokens])
        if was_training:
            model.train()
        return out

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=False)

    # ---- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Save adapter weights, head weights and the reconstructing config."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        if self.peft_model is not None:
            self.peft_model.save_pretrained(out / "adapter")
        if self.head is not None:
            torch.save(self.head.state_dict(), out / "head.pt")
        if self.optcode is not None:
            torch.save(self.optcode.state_dict(), out / "optcode.pt")
        meta = {
            "readout": self.readout,
            "bidirectional_full_attention": self.bidirectional_full_attention,
            "code_count": self.optcode.k if self.optcode is not None else None,
            "code_names": self.optcode_names,
            "base_model_id": self.base_model_id,
            "revision": self.revision,
            "dtype": self.dtype_name,
            "variant": self.variant,
            "max_prompt_tokens": self.max_prompt_tokens,
            "lora": self.lora.to_dict() if self.lora is not None else None,
            "head": self.head_settings.to_dict(),
            "calibration": self.calibration,
            "trainable_parameter_count": self.trainable_parameter_count,
        }
        (out / "model.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cuda:0",
        dtype: str | None = None,
        **overrides: Any,
    ) -> MedDecideModel:
        """Rebuild a saved model (adapter + head + calibration) from ``path``.

        The base is constructed **without** an adapter and the saved adapter is then wrapped
        around it. Building an adapter first and wrapping a second one on top makes peft warn
        about "multiple adapters in the model", which is a real hazard for a checkpoint other
        tasks (S8, S11) load — verified by a save/load prediction-equality check.
        """
        src = Path(path)
        meta = json.loads((src / "model.json").read_text(encoding="utf-8"))
        lora = LoraSettings.from_dict(meta["lora"]) if meta.get("lora") else None
        adapter_dir = src / "adapter"
        model = cls(
            meta["base_model_id"],
            lora=None,
            head_settings=HeadSettings(**meta["head"]),
            readout=meta.get("readout", "pointer"),
            dtype=dtype or meta.get("dtype", "bfloat16"),
            device=device,
            variant=meta.get("variant", "bare"),
            revision=meta.get("revision"),
            max_prompt_tokens=int(meta.get("max_prompt_tokens", 16384)),
            calibration=meta.get("calibration") or {},
            bidirectional_full_attention=bool(meta.get("bidirectional_full_attention", False)),
            code_count=int(meta.get("code_count") or OPTION_CODES),
            **overrides,
        )
        model.lora = lora
        head_path = src / "head.pt"
        if head_path.exists() and model.head is not None:
            state = torch.load(head_path, map_location=device, weights_only=True)
            model.head.load_state_dict(state)
            model.head.to(device)
        optcode_path = src / "optcode.pt"
        if optcode_path.exists() and model.optcode is not None:
            state = torch.load(optcode_path, map_location=device, weights_only=True)
            model.optcode.load_state_dict(state)
            model.optcode.to(device)
        if lora is not None and adapter_dir.exists():
            from peft import PeftModel

            model.peft_model = PeftModel.from_pretrained(
                model.base, adapter_dir, is_trainable=False
            )
            model.peft_model.eval()
        model.eval_mode()
        return model

    def describe(self) -> dict[str, Any]:
        return {
            "base_model_id": self.base_model_id,
            "revision": self.revision,
            "dtype": self.dtype_name,
            "device": self.device,
            "variant": self.variant,
            "d_model": self.d_model,
            "lora": self.lora.to_dict() if self.lora is not None else None,
            "head": self.head_settings.to_dict(),
            "trainable_parameter_count": self.trainable_parameter_count,
            "calibration": self.calibration,
        }


def render_item_prompt(model: MedDecideModel, item: Item) -> str:
    """The prompt the head (and the harness) sees for one item, canonicalised."""
    canonical, _ = canonicalise_options(item)
    return render_prompt(canonical, model.tokenizer, model.variant)


__all__ = [
    "DEFAULT_LORA_TARGETS",
    "Decision",
    "EncodedItem",
    "HeadSettings",
    "LoraSettings",
    "MedDecideModel",
    "ModelBatch",
    "PointerHead",
    "ScoredItems",
    "render_item_prompt",
    "reorder_options",
]
