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
        return cfg


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> StudentConfig:
    """Read a YAML config file and validate it."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: config must be a mapping")
    return StudentConfig.from_dict(raw)
