"""Training loop for the pointer head and the decision-path LoRA.

What the loop does, in order, for every step:

1. encode a batch of items through the **existing harness rendering** (option order permuted
   per epoch when ``shuffle_options`` is on, gold following its content);
2. run the frozen base (+ LoRA) once, without ``inference_mode`` so the adapter gets gradients;
3. read the pointer head's logits over exactly the offered options;
4. backprop ``cross-entropy + lambda * Brier`` into the head and the adapter only;
5. clip, step, and record throughput and memory.

Dev evaluation and temperature fitting are separate methods so a caller can decide when to pay
for them. Nothing here touches a test split.

``train`` can also run a **linear-warmup + cosine-decay** learning-rate schedule on **both**
parameter groups (head and LoRA). It is opt-in through ``schedule_steps``: a total is needed to
know what "3 % of the steps" and "the end of training" mean, and callers that do not pass one keep
the constant-LR behaviour they had before (the overfit smoke test relies on it).
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from meddecide.bench.schema import Item
from meddecide.train.config import StudentConfig
from meddecide.train.data import iter_batches, shuffled_order
from meddecide.train.losses import ce_plus_brier
from meddecide.train.temperature import TemperatureFit, fit_per_qtype, fit_temperature


def git_commit() -> str:
    """Current commit, or ``unknown`` outside a checkout (provenance, R7)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):  # pragma: no cover
        return "unknown"


def provenance(config: StudentConfig, model: Any) -> dict[str, Any]:
    """Everything R7 asks for that is knowable at runtime."""
    info: dict[str, Any] = {
        "git_commit": git_commit(),
        "config": config.to_dict(),
        "seed": config.seed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "model": model.describe() if hasattr(model, "describe") else str(model),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        import peft

        info["peft"] = peft.__version__
    except ImportError:  # pragma: no cover
        pass
    return info


@dataclass
class StepRecord:
    """One optimiser step's numbers."""

    step: int
    epoch: int
    loss: float
    ce: float
    brier: float
    accuracy: float
    n_items: int
    n_options: int
    real_tokens: int
    padded_tokens: int
    elapsed_s: float
    grad_norm: float
    gpu_peak_gb: float
    lr_head: float
    lr_lora: float

    @property
    def items_per_s(self) -> float:
        return self.n_items / self.elapsed_s if self.elapsed_s > 0 else float("nan")

    @property
    def tokens_per_s(self) -> float:
        return self.real_tokens / self.elapsed_s if self.elapsed_s > 0 else float("nan")

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["items_per_s"] = self.items_per_s
        data["tokens_per_s"] = self.tokens_per_s
        return data


@dataclass
class TrainResult:
    """What a training run produced."""

    steps: int
    epochs_completed: int
    wall_clock_s: float
    history: list[StepRecord] = field(default_factory=list)
    log_path: str | None = None
    best_step: int | None = None
    best_dev: dict[str, Any] | None = None
    stopped_early: str | None = None
    schedule: dict[str, Any] | None = None

    def to_dict(self, *, include_history: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "steps": self.steps,
            "epochs_completed": self.epochs_completed,
            "wall_clock_s": self.wall_clock_s,
            "best_step": self.best_step,
            "best_dev": self.best_dev,
            "stopped_early": self.stopped_early,
            "schedule": self.schedule,
        }
        if include_history:
            out["history"] = [r.to_dict() for r in self.history]
        return out

    def loss_at(self, step: int) -> float | None:
        """Training loss at a 1-based step number (the smoke run reports these)."""
        for record in self.history:
            if record.step == step:
                return record.loss
        return None


class Trainer:
    """Trains the pointer head and LoRA adapter of a :class:`MedDecideModel`."""

    def __init__(
        self,
        model: Any,
        config: StudentConfig,
        *,
        log_path: str | Path | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.log_path = Path(log_path) if log_path else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.optimizer = optimizer or torch.optim.AdamW(
            model.trainable_param_groups(
                lr=config.lr, lora_lr=config.lora_lr, weight_decay=config.weight_decay
            )
        )
        # the peak LRs the schedule scales; set here so a caller-supplied optimizer works too
        for group in self.optimizer.param_groups:
            group.setdefault("base_lr", float(group["lr"]))
        self.history: list[StepRecord] = []

    # ---- helpers ------------------------------------------------------------
    def _log(self, record: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def evaluate(self, items: Sequence[Item], *, batch_size: int | None = None) -> dict[str, Any]:
        """Dev metrics via batched inference; also returned per qtype."""
        self.model.eval_mode()
        scored = self.model.score_items(
            items,
            batch_size=batch_size or self.config.eval_batch_size or self.config.batch_size,
            max_batch_tokens=(
                self.config.eval_max_batch_tokens or self.config.max_batch_tokens
            ),
            max_prompt_tokens=self.config.max_prompt_tokens,
        )
        overall = scored.metrics()
        overall["per_qtype"] = scored.per_qtype_metrics()
        return overall

    def evaluate_loss(self, items: Sequence[Item]) -> dict[str, Any]:
        """Forward-only loss on a fixed set of items (no optimiser step, no gradients).

        This is what makes a training-loss curve comparable: raw per-step losses move with the
        item mix (the batch plan is length-sorted, so later steps hold longer, harder items),
        while this number is measured on the same items every time.
        """
        self.model.eval_mode()
        totals = {"loss": 0.0, "ce": 0.0, "brier": 0.0}
        n_items = 0
        correct = 0.0
        n_batches = 0
        with torch.no_grad():
            for _, batch in iter_batches(
                self.model,
                items,
                batch_size=self.config.batch_size,
                max_batch_tokens=self.config.max_batch_tokens,
                max_prompt_tokens=self.config.max_prompt_tokens,
                augment_options=False,
            ):
                model_batch = self.model.collate(batch)
                _logits, probs = self.model.batch_logits(model_batch)
                breakdown = ce_plus_brier(
                    probs,
                    model_batch.gold_flat,
                    model_batch.marker_item,
                    model_batch.batch_size,
                    lam=self.config.lambda_brier,
                )
                totals["loss"] += float(breakdown.total) * breakdown.n_items
                totals["ce"] += float(breakdown.ce) * breakdown.n_items
                totals["brier"] += float(breakdown.brier) * breakdown.n_items
                correct += breakdown.accuracy * breakdown.n_items
                n_items += breakdown.n_items
                n_batches += 1
        if n_items == 0:
            raise ValueError("evaluate_loss got no items")
        out = {k: v / n_items for k, v in totals.items()}
        out.update({"accuracy": correct / n_items, "n_items": n_items, "n_batches": n_batches})
        self.model.train_mode()
        return out

    def fit_calibration(
        self,
        items: Sequence[Item],
        *,
        per_qtype: bool | None = None,
        limit_per_qtype: int | None = None,
        seed: int | None = None,
    ) -> dict[str, TemperatureFit]:
        """Fit temperatures on dev and store them on the model.

        ``per_qtype`` defaults to the config switch. ``limit_per_qtype`` caps how many items of
        each qtype are scored (a speed knob for smoke runs); ``None`` uses every dev item.
        """
        per_qtype = self.config.temperature_per_qtype if per_qtype is None else per_qtype
        seed = self.config.seed if seed is None else seed
        selected = list(items)
        if limit_per_qtype is not None:
            rng = np.random.default_rng([seed, 4242])
            keep: list[int] = []
            counts: dict[str, int] = {}
            for i in rng.permutation(len(selected)):
                qtype = str(selected[int(i)].qtype)
                if counts.get(qtype, 0) >= limit_per_qtype:
                    continue
                counts[qtype] = counts.get(qtype, 0) + 1
                keep.append(int(i))
            selected = [selected[i] for i in sorted(keep)]
        self.model.eval_mode()
        scored = self.model.score_items(
            selected,
            batch_size=self.config.batch_size,
            max_batch_tokens=self.config.max_batch_tokens,
            max_prompt_tokens=self.config.max_prompt_tokens,
        )
        if per_qtype:
            fits = fit_per_qtype(scored)
        else:  # one temperature for every qtype
            fits = {
                "*": fit_temperature(scored.logits, scored.gold_indices, qtype="*"),
            }
        self.model.calibration = {k: v.temperature for k, v in fits.items()}
        return fits

    def save_checkpoint(self, path: str | Path) -> Path:
        return self.model.save(path)

    # ---- learning-rate schedule --------------------------------------------
    @staticmethod
    def lr_factor(step: int, *, total_steps: int, warmup_steps: int) -> float:
        """Linear warmup then cosine decay, as a multiplier in ``[0, 1]``.

        ``step`` is 1-based. The warmup reaches 1.0 at ``warmup_steps``; the cosine reaches 0.0
        at ``total_steps``. ``warmup_steps <= 0`` means "no warmup, start at the peak".
        """
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if warmup_steps > 0 and step <= warmup_steps:
            return max(0.0, min(1.0, step / warmup_steps))
        span = total_steps - max(0, warmup_steps)
        if span <= 0:
            return 1.0
        progress = min(1.0, max(0.0, (step - max(0, warmup_steps)) / span))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _apply_lr(self, step: int, schedule: dict[str, Any] | None) -> None:
        """Set every parameter group's LR for ``step`` from the schedule (no-op when None)."""
        if schedule is None:
            return
        factor = self.lr_factor(
            step, total_steps=int(schedule["total_steps"]), warmup_steps=int(schedule["warmup_steps"])
        )
        for group in self.optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * factor

    # ---- training -----------------------------------------------------------
    def train(
        self,
        items: Sequence[Item],
        *,
        steps: int | None = None,
        eval_items: Sequence[Item] | None = None,
        checkpoint_dir: str | Path | None = None,
        max_seconds: float | None = None,
        on_step: Callable[[int], None] | None = None,
        schedule_steps: int | None = None,
    ) -> TrainResult:
        """Run the optimiser. ``steps`` caps the total across epochs (None = all epochs).

        ``schedule_steps`` is the total step count the LR schedule is laid out over (the caller's
        planned epoch). ``None`` (the default, and every caller before S9) keeps the constant LRs
        the optimizer was built with; the schedule is also skipped when ``lr_schedule`` is
        ``"constant"``.
        """
        config = self.config
        if config.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.model.train_mode()
        started = time.perf_counter()
        total_steps = 0
        epochs_completed = 0
        best: dict[str, Any] | None = None
        best_step: int | None = None
        stopped_early: str | None = None
        deadline = (
            (max_seconds if max_seconds is not None else config.max_seconds) or None
        )
        epoch = 0
        schedule: dict[str, Any] | None = None
        if schedule_steps is not None and config.lr_schedule == "cosine":
            warmup_steps = round(float(config.warmup_fraction) * int(schedule_steps))
            schedule = {
                "kind": "linear_warmup_cosine_decay",
                "total_steps": int(schedule_steps),
                "warmup_steps": max(0, warmup_steps),
                "warmup_fraction": float(config.warmup_fraction),
                "peak_lr_by_group": [float(g["base_lr"]) for g in self.optimizer.param_groups],
            }
        # a step budget keeps going past ``config.epochs`` epochs: "train for N steps" must mean
        # N steps, not "N steps or one pass, whichever comes first" (the overfit test relies on
        # this to see its 64 items many times)
        while epoch < config.epochs or (
            steps is not None and total_steps < steps and stopped_early is None
        ):
            order = shuffled_order(len(items), seed=config.seed, epoch=epoch)
            for _, batch in iter_batches(
                self.model,
                items,
                batch_size=config.batch_size,
                max_batch_tokens=config.max_batch_tokens,
                max_prompt_tokens=config.max_prompt_tokens,
                order=order,
                augment_options=config.shuffle_options,
                epoch=epoch,
                seed=config.seed,
            ):
                if steps is not None and total_steps >= steps:
                    stopped_early = f"step budget {steps} reached"
                    break
                if deadline is not None and time.perf_counter() - started >= deadline:
                    stopped_early = f"time budget {deadline:.0f}s reached"
                    break
                if config.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                self._apply_lr(total_steps + 1, schedule)
                t0 = time.perf_counter()
                model_batch = self.model.collate(batch)
                _logits, probs = self.model.batch_logits(model_batch)
                breakdown = ce_plus_brier(
                    probs,
                    model_batch.gold_flat,
                    model_batch.marker_item,
                    model_batch.batch_size,
                    lam=config.lambda_brier,
                )
                self.optimizer.zero_grad(set_to_none=True)
                breakdown.total.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in self.optimizer.param_groups for p in group["params"]],
                        config.grad_clip,
                    )
                )
                self.optimizer.step()
                elapsed = time.perf_counter() - t0
                peak = (
                    torch.cuda.max_memory_allocated() / 1e9
                    if config.device.startswith("cuda") and torch.cuda.is_available()
                    else 0.0
                )
                total_steps += 1
                record = StepRecord(
                    step=total_steps,
                    epoch=epoch,
                    loss=float(breakdown.total.detach()),
                    ce=float(breakdown.ce.detach()),
                    brier=float(breakdown.brier.detach()),
                    accuracy=breakdown.accuracy,
                    n_items=model_batch.batch_size,
                    n_options=breakdown.n_options,
                    real_tokens=model_batch.real_tokens,
                    padded_tokens=model_batch.padded_tokens,
                    elapsed_s=elapsed,
                    grad_norm=grad_norm,
                    gpu_peak_gb=peak,
                    lr_head=float(self.optimizer.param_groups[0]["lr"]),
                    lr_lora=float(self.optimizer.param_groups[-1]["lr"]),
                )
                self.history.append(record)
                if on_step is not None:
                    on_step(total_steps)
                if config.log_every and total_steps % config.log_every == 0:
                    self._log({"event": "step", **record.to_dict()})
                if (
                    eval_items is not None
                    and config.eval_every
                    and total_steps % config.eval_every == 0
                ):
                    metrics = self.evaluate(eval_items, batch_size=config.batch_size)
                    self._log({"event": "eval", "step": total_steps, "metrics": metrics})
                    if best is None or (
                        metrics["macro_accuracy"],
                        -metrics["brier"],
                    ) > (best["macro_accuracy"], -best["brier"]):
                        best, best_step = metrics, total_steps
                        if checkpoint_dir is not None:
                            self.save_checkpoint(checkpoint_dir)
                    self.model.train_mode()
            epochs_completed = epoch + 1
            epoch += 1
            if stopped_early:
                break
        wall = time.perf_counter() - started
        result = TrainResult(
            steps=total_steps,
            epochs_completed=epochs_completed,
            wall_clock_s=wall,
            history=self.history,
            log_path=str(self.log_path) if self.log_path else None,
            best_step=best_step,
            best_dev=best,
            stopped_early=stopped_early,
            schedule=schedule,
        )
        self._log({"event": "end", **result.to_dict(include_history=False)})
        self.model.eval_mode()
        return result
