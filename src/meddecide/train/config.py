"""Student-model training config (``configs/student_v0.yaml``).

Every number that changes training lives here: model id, LoRA shape and targets, the Brier
weight, learning rates, batch limits, prompt cap, seed, and whether to fit temperatures. Code
reads the config; nothing important is hardcoded in the training loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from meddecide.model.head import HeadSettings
from meddecide.model.meddecide_model import DEFAULT_LORA_TARGETS, LoraSettings
from meddecide.train.corrected_mix import validate_drop
from meddecide.train.option_augment import validate_templates

DEFAULT_CONFIG_PATH = Path("configs/student_v0.yaml")

_TOP_LEVEL_KEYS = {
    "base_model",
    "revision",
    "train_path",
    "dev_path",
    "output_dir",
    "seed",
    "dtype",
    "device",
    "variant",
    "lora",
    "head",
    "readout",
    "bidirectional_full_attention",
    "lr_floor",
    "tripwire_grad_norm",
    "tripwire_macro_floor",
    "tripwire_consecutive_evals",
    "selection_metric",
    "lambda_brier",
    "lr",
    "lora_lr",
    "weight_decay",
    "grad_clip",
    "batch_size",
    "max_batch_tokens",
    "max_prompt_tokens",
    "epochs",
    "shuffle_options",
    "eval_every",
    "eval_steps",
    "eval_items",
    "eval_batch_size",
    "eval_max_batch_tokens",
    "eval_round_to_chunk",
    "eval_length_buckets",
    "log_every",
    "lr_schedule",
    "warmup_fraction",
    "batch_chunk_factor",
    "fit_temperature",
    "temperature_per_qtype",
    "temperature_items_per_qtype",
    "max_seconds",
    "gradient_checkpointing",
    "checkpoint_above_tokens",
    "backbone",
    "kl_to_base",
    "kl_exclude_template_prefixes",
    "distill_weight",
    "loss_weighting",
    "option_augment_probability",
    "option_augment_templates",
    "drop_mix_templates",
}


@dataclass(frozen=True)
class StudentConfig:
    """Training configuration for MedDecide-0.8B (S9) and its Base ablation (S10)."""

    base_model: str = "Qwen/Qwen3.5-0.8B"
    revision: str | None = None
    train_path: str = "data/train/student_v0/train.jsonl"
    dev_path: str = "data/train/student_v0/dev.jsonl"
    output_dir: str = "outputs/student_v0/S9"
    seed: int = 0
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    variant: str = "bare"
    lora: LoraSettings = field(default_factory=LoraSettings)
    head: HeadSettings = field(default_factory=HeadSettings)
    # "pointer" (student_v0's head) or "letter" (student_v1 arm B: no head, LM logits of option letters)
    readout: str = "pointer"
    # arm N (D23): the full-attention layers attend in both directions on the decision path (off for causal arms)
    bidirectional_full_attention: bool = False
    # the cosine decay ends at this fraction of the peak learning rate (O6-O8: 0.10; 0.0 is the student behaviour)
    lr_floor: float = 0.0
    # divergence tripwire (ADVISORY O6): None disables a check; the gradient check uses the pre-clip norm
    tripwire_grad_norm: float | None = None
    tripwire_macro_floor: float | None = None
    tripwire_consecutive_evals: int = 2
    # dev statistic for checkpoint selection and the dev tripwire: "macro_accuracy" (per-gold-key recall; the
    # student behaviour) or "template_macro_accuracy" (mean over templates; G1/D24's macro, osler_v0 Q20)
    selection_metric: str = "macro_accuracy"
    # cross-entropy + lambda * Brier; 1.0 is the plan's starting value
    lambda_brier: float = 1.0
    lr: float = 1.0e-3
    lora_lr: float = 2.0e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    batch_size: int = 8
    max_batch_tokens: int = 8192
    max_prompt_tokens: int = 8192
    epochs: int = 1
    shuffle_options: bool = True
    eval_every: int = 200
    # explicit evaluation steps (osler_v4 Y8: 10 evaluations ending at the step budget); None keeps the eval_every schedule
    eval_steps: tuple[int, ...] | None = None
    eval_items: int = 256
    # forward-only dev scoring may use bigger batches than training (no activations are kept);
    # None falls back to batch_size / max_batch_tokens
    eval_batch_size: int | None = None
    eval_max_batch_tokens: int | None = None
    # True (the student_v0 behaviour) rounds the dev batch length to the chunk; False scores each
    # item with no padding at batch size 1 (the V2 evaluation path, ADVISORY student_v1 V2)
    eval_round_to_chunk: bool = True
    # True batches dev items of one exact length (no padding; the faster route to the same computation)
    eval_length_buckets: bool = False
    log_every: int = 10
    # LR schedule, applied by Trainer.train only when it is given an explicit schedule_steps
    # total (S9 passes the planned batch count of its one epoch). "cosine" = linear warmup over
    # warmup_fraction of the steps, then cosine decay to zero, on **both** parameter groups.
    lr_schedule: str = "cosine"
    warmup_fraction: float = 0.03
    # length-bucketing chunk size in units of batch_size (see meddecide.train.data)
    batch_chunk_factor: int = 100
    fit_temperature: bool = True
    temperature_per_qtype: bool = True
    temperature_items_per_qtype: int | None = None
    max_seconds: float | None = None
    # recompute decoder layers in the backward pass (same numbers; needed for the 16,384-token cap on one GPU)
    gradient_checkpointing: bool = False
    # with gradient_checkpointing on: recompute only for batches whose padded token count exceeds this (None = always).
    # Same numbers either way; osler_v2 profiling: about 1.4x faster on the batches that fit without recomputation
    checkpoint_above_tokens: int | None = None
    # "qwen": MedDecideModel on base_model (Qwen3.5 causal LM, our prompt); "pplx": PplxDecisionModel on
    # perplexity-ai/pplx-decider-flash-v1.2 (its backbone, prompt and readout; base_model is ignored) — osler_v2
    backbone: str = "qwen"
    # osler_v3 (D41): + kl_to_base * mean KL(base || model) over the batch's anchor items, the base being the model with
    # its adapter off and its pre-training readout; anchors are items whose template_id starts with none of
    # kl_exclude_template_prefixes (the domain templates the model is meant to move on). 0.0 = off
    kl_to_base: float = 0.0
    kl_exclude_template_prefixes: tuple[str, ...] = ()
    # osler_v4 Y6 (D48): + distill_weight * KL(teacher || model) over the batch's items that carry a teacher distribution
    # (SoftItem.teacher_probs, the pplx 27B soft labels); needs no base model. 0.0 = off
    distill_weight: float = 0.0
    # osler_v5 (D51): "per_step" (the osler_v4 behaviour: the loss is a mean over the batch's items) or "per_item" (after clipping,
    # the gradients are multiplied by n_items / batch_size; KL and distillation are sums over the masked items divided by batch_size)
    loss_weighting: str = "per_step"
    # osler_v5 (D52a): the probability that an eligible choice row of a closed-list template is rebuilt with more options
    # (meddecide.train.option_augment); the templates must be on that list. 0.0 and no templates = off
    option_augment_probability: float = 0.0
    option_augment_templates: tuple[str, ...] = ()
    # osler_v5 (D56a): the training-mix templates dropped whole (meddecide.train.corrected_mix.DROP_ALLOWED); () = osler_v4's mix
    drop_mix_templates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lora"] = self.lora.to_dict()
        data["head"] = self.head.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StudentConfig:
        unknown = sorted(set(data) - _TOP_LEVEL_KEYS)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        lora_data = data.get("lora") or {}
        head_data = data.get("head") or {}
        lora = LoraSettings(
            r=int(lora_data.get("r", 16)),
            alpha=int(lora_data.get("alpha", 32)),
            dropout=float(lora_data.get("dropout", 0.0)),
            target_modules=tuple(lora_data.get("target_modules", DEFAULT_LORA_TARGETS)),
        )
        head = HeadSettings(
            hidden=int(head_data.get("hidden", 256)),
            dropout=float(head_data.get("dropout", 0.0)),
        )
        allowed_lora = set(LoraSettings().to_dict())
        unknown_lora = sorted(set(lora_data) - allowed_lora)
        if unknown_lora:
            raise ValueError(f"unknown lora keys: {unknown_lora}")
        unknown_head = sorted(set(head_data) - set(HeadSettings().to_dict()))
        if unknown_head:
            raise ValueError(f"unknown head keys: {unknown_head}")
        kwargs = {k: v for k, v in data.items() if k not in {"lora", "head"}}
        if "kl_exclude_template_prefixes" in kwargs:
            kwargs["kl_exclude_template_prefixes"] = tuple(kwargs["kl_exclude_template_prefixes"] or ())
        if "option_augment_templates" in kwargs:
            kwargs["option_augment_templates"] = validate_templates(kwargs["option_augment_templates"] or ())
        if "drop_mix_templates" in kwargs:
            kwargs["drop_mix_templates"] = validate_drop(kwargs["drop_mix_templates"] or ())
        cfg = cls(lora=lora, head=head, **kwargs)
        # arithmetic and range checks that would otherwise surface as a confusing crash later
        if cfg.batch_size < 1 or cfg.max_batch_tokens < 1 or cfg.max_prompt_tokens < 1:
            raise ValueError("batch_size, max_batch_tokens and max_prompt_tokens must be >= 1")
        if cfg.lambda_brier < 0:
            raise ValueError("lambda_brier must be >= 0")
        if cfg.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if cfg.lr_schedule not in {"cosine", "constant"}:
            raise ValueError(f"lr_schedule must be 'cosine' or 'constant', got {cfg.lr_schedule!r}")
        if not 0.0 <= cfg.warmup_fraction < 1.0:
            raise ValueError(f"warmup_fraction must be in [0, 1), got {cfg.warmup_fraction}")
        if cfg.batch_chunk_factor < 1:
            raise ValueError("batch_chunk_factor must be >= 1")
        if not 0.0 <= cfg.lr_floor < 1.0:
            raise ValueError(f"lr_floor must be in [0, 1), got {cfg.lr_floor}")
        if cfg.tripwire_consecutive_evals < 1:
            raise ValueError("tripwire_consecutive_evals must be >= 1")
        if cfg.checkpoint_above_tokens is not None and (cfg.checkpoint_above_tokens < 1 or not cfg.gradient_checkpointing):
            raise ValueError("checkpoint_above_tokens needs gradient_checkpointing and a positive token count")
        if cfg.kl_to_base < 0:
            raise ValueError("kl_to_base must be >= 0")
        if cfg.kl_to_base > 0 and cfg.readout != "option_code":
            raise ValueError("kl_to_base needs the option-code readout")
        if cfg.distill_weight < 0:
            raise ValueError("distill_weight must be >= 0")
        if cfg.loss_weighting not in ("per_step", "per_item"):
            raise ValueError(f"loss_weighting must be 'per_step' or 'per_item', got {cfg.loss_weighting!r}")
        if not 0.0 <= cfg.option_augment_probability <= 1.0:
            raise ValueError("option_augment_probability must be in [0, 1]")
        if (cfg.option_augment_probability > 0) != bool(cfg.option_augment_templates):
            raise ValueError("option augmentation needs both a probability above 0 and at least one closed-list template")
        if cfg.backbone not in ("qwen", "pplx"):
            raise ValueError(f"backbone must be 'qwen' or 'pplx', got {cfg.backbone!r}")
        if cfg.backbone == "pplx" and (cfg.readout != "option_code" or cfg.bidirectional_full_attention):
            raise ValueError("the pplx backbone uses its own readout (readout: option_code) and its own attention mask")
        if cfg.selection_metric not in ("macro_accuracy", "template_macro_accuracy"):
            raise ValueError(f"selection_metric must be 'macro_accuracy' or 'template_macro_accuracy', "
                             f"got {cfg.selection_metric!r}")
        return cfg


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> StudentConfig:
    """Read a YAML config file and validate it."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: config must be a mapping")
    return StudentConfig.from_dict(raw)
