"""Trainable text-only harness for a d3 student (d3_distill step 2), built on the student's own runtime classes.

The student's ``d3_runtime.D3`` is loaded by the card's ``AutoModel`` path; this module then

- reuses ``D3.prepare`` for the prompt (render + tokenize exactly as inference) and ``D3.encoder`` +
  ``D3.backbone`` for the forward, so the training forward is the runtime's own reference path;
- wraps ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj`` of the full-attention layers in an attention-only LoRA
  (D33) and trains the 255-row decision readout; the rest of the backbone stays frozen;
- builds the CE(gold) and KL(teacher ‖ student) terms over the offered option codes, with the runtime's own
  temperature and option masking;
- schedules batches deterministically from the step number (resume-safe): epoch order by seed, windows of 64
  length-sorted into batches of 8.

No model step is re-implemented: the forward is ``backbone.get_input_embeddings`` → ``D3.encoder`` → readout matmul,
which is exactly what ``D3.logits`` does without ``torch.inference_mode``.
"""

from __future__ import annotations

import contextlib
import math
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from meddecide.distill.d3_batch import question_request

MAX_CODES = 255


# ---------------------------------------------------------------------------------------------
# LoRA (D33: attention only)
# ---------------------------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a rank-``r`` update: ``W x + (alpha/r) B A x``.

    ``A`` is Kaiming-uniform and ``B`` is zero, so the module starts exactly at the frozen model's output.
    The adapter is FP32 and the update is cast back to the base output's dtype.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        if r < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.A = nn.Parameter(torch.empty(self.r, base.in_features, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, self.r, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = ((x.float() @ self.A.T) @ self.B.T) * self.scaling
        return out + delta.to(out.dtype)


def inject_lora(
    backbone: nn.Module,
    *,
    r: int = 32,
    alpha: int = 32,
    targets: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
) -> dict[str, LoRALinear]:
    """Replace the attention projections of every full-attention layer; returns the adapters by dotted name."""
    language_model = backbone.language_model
    # The composite (multimodal-style) config nests the text config, whose layer_types the runtime's PromptEncoder reads.
    types = list(language_model.config.layer_types)
    adapters: dict[str, LoRALinear] = {}
    for index, layer in enumerate(language_model.layers):
        if types[index] != "full_attention":
            continue
        for name in targets:
            module = getattr(layer.self_attn, name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"layer {index}.self_attn.{name} is {type(module).__name__}, not nn.Linear")
            adapter = LoRALinear(module, r, alpha)
            setattr(layer.self_attn, name, adapter)
            adapters[f"layers.{index}.self_attn.{name}"] = adapter
    if not adapters:
        raise RuntimeError("no full-attention projection was wrapped: check the layer types and target names")
    return adapters


# ---------------------------------------------------------------------------------------------
# Student
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainRow:
    """One prepared training item: the token sequence, its option count/gold and the teacher distribution."""

    row_id: str
    sequence: tuple[int, ...]
    count: int
    gold: int
    teacher: tuple[float, ...] | None
    split: str
    source: str

    @property
    def length(self) -> int:
        return len(self.sequence)


def render_order(qtype: str, option_keys: Sequence[str]) -> list[int]:
    """The option index at each position of the *rendered* prompt, which is the readout's code order.

    ``choice`` and ``score`` render the options in the row's order. ``noul`` always renders its fixed
    ``false`` / ``true`` options (d3_format.NOUL_DESCRIPTIONS), i.e. ``no`` then ``yes``, whatever order the row
    stores them in, so the readout's first code is "No" even for a row whose options are ``[yes, no]``.
    """
    if qtype == "noul":
        if sorted(option_keys) != ["no", "yes"]:
            raise ValueError(f"a noul row needs the keys yes and no, got {list(option_keys)}")
        return [list(option_keys).index("no"), list(option_keys).index("yes")]
    return list(range(len(option_keys)))


def render_targets(
    qtype: str, option_keys: Sequence[str], gold_key: str, teacher: Sequence[float] | None
) -> tuple[int, tuple[float, ...] | None]:
    """Gold index and teacher distribution in the rendered (readout) code order."""
    order = render_order(qtype, option_keys)
    gold = next(i for i, key in enumerate(option_keys) if key == gold_key)
    return order.index(gold), None if teacher is None else tuple(float(teacher[i]) for i in order)


def prepare_rows(runtime: Any, rows: Sequence[dict[str, Any]]) -> list[TrainRow]:
    """Prepare pool rows through the student's own ``D3.prepare``; a row the runtime refuses raises, never drops."""
    out: list[TrainRow] = []
    for row in rows:
        options = [(o["key"], o["label"]) for o in row["options"]]
        request = question_request(row["qtype"], row["question"], options)
        ready = runtime.prepare(row["state"], {"q": request})
        if "q" in ready.errors:
            raise ValueError(f"{row['row_id']}: {ready.errors['q']['message']}")
        gold, teacher = render_targets(
            row["qtype"], [key for key, _ in options], row["gold"], row.get("teacher_probs")
        )
        out.append(
            TrainRow(
                row_id=row["row_id"],
                sequence=tuple(ready.sequences["q"]),
                count=len(options),
                gold=gold,
                teacher=teacher,
                split=row["split"],
                source=row["source"],
            )
        )
    return out


def batch_tensors(rows: Sequence[TrainRow], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Left-padded ids/mask of one batch, plus option counts and gold indices."""
    width = max(row.length for row in rows)
    ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, width - row.length :] = torch.as_tensor(row.sequence, dtype=torch.long)
        mask[i, width - row.length :] = 1
    return (
        ids.to(device, non_blocking=True),
        mask.to(device, non_blocking=True),
        [row.count for row in rows],
        [row.gold for row in rows],
    )


def masked_logits(logits: torch.Tensor, counts: Sequence[int]) -> torch.Tensor:
    """Logits with every code past its row's option count set to -inf."""
    limit = torch.as_tensor(list(counts), device=logits.device)[:, None]
    invalid = torch.arange(logits.shape[-1], device=logits.device)[None] >= limit
    return logits.masked_fill(invalid, float("-inf"))


def loss_from_logits(
    logits: torch.Tensor,
    counts: Sequence[int],
    golds: Sequence[int],
    teachers: Sequence[Sequence[float] | None],
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed CE(gold) and summed KL(teacher ‖ student) over the offered codes of one batch."""
    scaled = masked_logits(logits, counts) / float(temperature)
    log_probs = torch.log_softmax(scaled, dim=-1)
    rows = torch.arange(len(counts), device=logits.device)
    ce = -log_probs[rows, torch.as_tensor(list(golds), device=logits.device)].sum()
    kl = logits.new_zeros(())
    for i, teacher in enumerate(teachers):
        if teacher is None:
            continue
        target = torch.as_tensor(list(teacher), device=logits.device, dtype=torch.float32)
        target = target / target.sum()
        kl = kl + (target * (target.clamp_min(1e-12).log() - log_probs[i, : len(teacher)])).sum()
    return ce, kl


class D3Student(nn.Module):
    """A loaded d3 runtime with a trainable attention-only LoRA and decision readout."""

    def __init__(self, runtime: Any, *, lora_r: int = 32, lora_alpha: int = 32):
        super().__init__()
        self.runtime = runtime
        self.backbone = runtime.backbone
        # The runtime's inference path picks its SDPA backends explicitly (no cuDNN SDPA on CUDA); the training forward
        # must use the same selection or bf16 attention numerics drift between the harness and the model's own path.
        module = sys.modules.get(type(runtime).__module__)
        self._sdpa_backends = getattr(module, "sdpa_backends", None)
        self.pad_id = int(runtime.pad_id)
        self.temperature = float(runtime.temperature)
        self.device = runtime.device
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.adapters = inject_lora(self.backbone, r=lora_r, alpha=lora_alpha)
        self.readout = nn.Parameter(runtime.readout.detach().float().clone())
        self.readout.requires_grad_(True)
        self.backbone.eval()

    def lora_parameters(self) -> dict[str, nn.Parameter]:
        params: dict[str, nn.Parameter] = {}
        for name, adapter in self.adapters.items():
            params[f"{name}.A"] = adapter.A
            params[f"{name}.B"] = adapter.B
        return params

    def _sdpa(self):
        if self._sdpa_backends is None:
            return contextlib.nullcontext()
        return self._sdpa_backends(self.device)

    def forward_logits(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The runtime's reference forward, without ``inference_mode``: embeds -> encoder -> FP32 readout."""
        with self._sdpa():
            embeds = self.backbone.get_input_embeddings()(ids)
            hidden = self.runtime.encoder(embeds, mask)[:, -1]
        return hidden.float() @ self.readout.T

    def loss_terms(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        counts: Sequence[int],
        golds: Sequence[int],
        teachers: Sequence[Sequence[float] | None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Summed CE(gold) and summed KL(teacher ‖ student) over one batch, both over the offered codes."""
        return loss_from_logits(
            self.forward_logits(ids, mask), counts, golds, teachers, temperature=self.temperature
        )

    @torch.inference_mode()
    def probabilities(self, sequences: Sequence[Sequence[int]], counts: Sequence[int], *, batch_size: int = 32, length_sorted: bool = True) -> list[list[float]]:
        """The model's own probability path (``D3.probabilities``) with the current adapter and readout."""
        self.runtime.readout = self.readout.detach()
        order = list(range(len(sequences)))
        if length_sorted:
            order.sort(key=lambda i: len(sequences[i]))
        out: list[list[float]] = [[] for _ in sequences]
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            values = self.runtime.probabilities([sequences[i] for i in chunk], [counts[i] for i in chunk])
            for i, value in zip(chunk, values, strict=True):
                out[i] = value
        return out

    def state_dict_for_checkpoint(self) -> dict[str, Any]:
        return {
            "lora": {name: parameter.detach().cpu() for name, parameter in self.lora_parameters().items()},
            "readout": self.readout.detach().cpu(),
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        lora = self.lora_parameters()
        for name, value in state["lora"].items():
            if name not in lora:
                raise KeyError(f"checkpoint has LoRA parameter {name}, the model does not")
            lora[name].data.copy_(value.to(lora[name].device))
        self.readout.data.copy_(state["readout"].to(self.readout.device))


# ---------------------------------------------------------------------------------------------
# Deterministic step -> batch mapping (resume-safe)
# ---------------------------------------------------------------------------------------------


class BatchPlan:
    """Maps a global step to 8 items: epochs by seed, windows of ``window`` length-sorted, batches of ``batch``."""

    def __init__(self, lengths: Sequence[int], *, batch: int = 8, window: int = 64, seed: int = 0):
        if batch < 1 or window < batch:
            raise ValueError("need 1 <= batch <= window")
        self.lengths = list(lengths)
        self.batch_size = int(batch)
        self.window = int(window)
        self.seed = int(seed)
        self.windows = len(self.lengths) // self.window
        self.steps_per_epoch = self.windows * (self.window // self.batch_size)
        if self.steps_per_epoch < 1:
            raise ValueError("the data is smaller than one window")
        self._epoch: int | None = None
        self._order: list[int] = []

    def _epoch_order(self, epoch: int) -> list[int]:
        if self._epoch != epoch:
            order = list(range(len(self.lengths)))
            random.Random(f"{self.seed}:{epoch}").shuffle(order)
            self._order = order
            self._epoch = epoch
        return self._order

    def epoch_of(self, step: int) -> int:
        return step // self.steps_per_epoch

    def batch(self, step: int) -> list[int]:
        epoch = self.epoch_of(step)
        within = step % self.steps_per_epoch
        order = self._epoch_order(epoch)
        window_index = within // (self.window // self.batch_size)
        batch_index = within % (self.window // self.batch_size)
        chunk = order[window_index * self.window : (window_index + 1) * self.window]
        chunk = sorted(chunk, key=lambda i: self.lengths[i])
        return chunk[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]


def lr_at(step: int, total: int, base: float, *, warmup_fraction: float = 0.03, floor: float = 0.1) -> float:
    """Linear warm-up then cosine decay to ``floor * base`` (the D33 schedule)."""
    if total < 1:
        raise ValueError("the schedule needs at least one step")
    warmup = max(1, int(total * warmup_fraction))
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * max(floor, 0.5 * (1.0 + math.cos(math.pi * progress)))


def gradient_norm(parameters: Sequence[nn.Parameter], grads: Sequence[torch.Tensor | None]) -> float:
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().pow(2).sum().cpu())
    return math.sqrt(total)
