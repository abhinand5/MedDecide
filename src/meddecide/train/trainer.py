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
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from meddecide.bench.schema import Item
from meddecide.train.config import StudentConfig
from meddecide.train.data import iter_batches, shuffled_order
from meddecide.train.losses import ce_plus_brier, kl_to_reference, teacher_reference
from meddecide.train.resume import (
    config_fingerprint,
    read_full_checkpoint,
    restore_optimizer_params,
    set_rng_state,
    write_full_checkpoint,
)
from meddecide.train.temperature import TemperatureFit, fit_per_qtype, fit_temperature

if TYPE_CHECKING:
    from meddecide.model.meddecide_model import ScoredItems


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
    # per-batch composition: what the optimiser actually saw, so a grad spike can be attributed
    # to a batch's length (and to which templates it held) instead of guessed at
    max_item_tokens: int = 0
    mean_item_tokens: float = 0.0
    template_ids: list[str] = field(default_factory=list)
    # osler_v3 KL-to-base (D41): the penalty's value on this step and the number of anchor items it covered
    kl: float = 0.0
    n_anchor: int = 0
    # osler_v4 Y6 (D48): the distillation penalty's value on this step and the number of items with a teacher distribution
    distill: float = 0.0
    n_teacher: int = 0

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


def is_eval_step(step: int, eval_every: int | None, eval_steps: Sequence[int] | None = None) -> bool:
    """Whether a step is an evaluation step: the explicit list when one is given, else every eval_every steps (0 or None: never)."""
    if eval_steps is not None:
        return step in eval_steps
    return bool(eval_every) and step % eval_every == 0


def evaluation_grid(steps: int, count: int) -> tuple[int, ...]:
    """count evaluation steps spread evenly over steps, the last one at the step budget (osler_v4 Y8: 10 evaluations at 63,304)."""
    if count < 1 or steps < count:
        raise ValueError(f"cannot place {count} evaluations in {steps} steps")
    return tuple(round(steps * k / count) for k in range(1, count + 1))


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
        # optional observer of every evaluation (step, metrics); the O6 driver uses it for the tier-1 diagnostic
        self.eval_hook: Callable[[int, dict[str, Any]], None] | None = None
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

    def score(self, items: Sequence[Item], *, batch_size: int | None = None) -> ScoredItems:
        """Per-item probabilities over the offered options (batched inference; no metrics)."""
        self.model.eval_mode()
        return self.model.score_items(
            items,
            batch_size=batch_size or self.config.eval_batch_size or self.config.batch_size,
            max_batch_tokens=(
                self.config.eval_max_batch_tokens or self.config.max_batch_tokens
            ),
            max_prompt_tokens=self.config.max_prompt_tokens,
            round_to_chunk=self.config.eval_round_to_chunk,
            length_buckets=self.config.eval_length_buckets,
        )

    def evaluate(self, items: Sequence[Item], *, batch_size: int | None = None) -> dict[str, Any]:
        """Dev metrics via batched inference; also returned per qtype."""
        scored = self.score(items, batch_size=batch_size)
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
    def lr_factor(step: int, *, total_steps: int, warmup_steps: int, floor: float = 0.0) -> float:
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
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _apply_lr(self, step: int, schedule: dict[str, Any] | None) -> None:
        """Set every parameter group's LR for ``step`` from the schedule (no-op when None)."""
        if schedule is None:
            return
        factor = self.lr_factor(
            step,
            total_steps=int(schedule["total_steps"]),
            warmup_steps=int(schedule["warmup_steps"]),
            floor=float(schedule.get("floor", 0.0)),
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
        full_checkpoint_dir: str | Path | None = None,
        full_checkpoint_every: int = 0,
        resume_from: str | Path | None = None,
    ) -> TrainResult:
        """Run the optimiser. ``steps`` caps the total across epochs (None = all epochs).

        ``schedule_steps`` is the total step count the LR schedule is laid out over (the caller's
        planned epoch). ``None`` (the default, and every caller before S9) keeps the constant LRs
        the optimizer was built with; the schedule is also skipped when ``lr_schedule`` is
        ``"constant"``.

        ``full_checkpoint_every`` > 0 with ``full_checkpoint_dir`` writes a full checkpoint (the trainable tensors, the optimizer,
        the run state and the RNG states) every that many steps; ``resume_from`` continues from one with the same data order and
        LR schedule (meddecide.train.resume). Both default off, and the run is then the same as before.
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
        consecutive_below = 0
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
                "floor": float(config.lr_floor),
                "peak_lr_by_group": [float(g["base_lr"]) for g in self.optimizer.param_groups],
            }
        # resume (osler_v4 Y8): the state of a full checkpoint, restored before the first batch. The batches the checkpoint already took
        # in its epoch are skipped, so the data order continues exactly (the order is a pure function of seed, epoch and batch index)
        skip_in_epoch = 0
        epoch_start_step = 0
        if resume_from is not None:
            saved = read_full_checkpoint(Path(resume_from))
            run = saved.run_state
            if run["config_fingerprint"] != config_fingerprint(config.to_dict()):
                raise ValueError("resume: the training configuration differs from the checkpoint's")
            if run["schedule"] != schedule:
                raise ValueError("resume: the LR schedule differs from the checkpoint's")
            restore_optimizer_params(self.optimizer, saved.trainable)
            self.optimizer.load_state_dict(saved.optimizer)
            set_rng_state(saved.rng)
            total_steps = int(run["step"])
            epoch = int(run["epoch"])
            epoch_start_step = int(run["epoch_start_step"])
            skip_in_epoch = int(run["batches_in_epoch"])
            epochs_completed = int(run["epochs_completed"])
            best = run["best"]
            best_step = run["best_step"]
            consecutive_below = int(run["consecutive_below"])
            self._log({"event": "resume", "from_step": total_steps, "epoch": epoch, "batches_in_epoch": skip_in_epoch})
        # a step budget keeps going past ``config.epochs`` epochs: "train for N steps" must mean
        # N steps, not "N steps or one pass, whichever comes first" (the overfit test relies on
        # this to see its 64 items many times)
        while epoch < config.epochs or (
            steps is not None and total_steps < steps and stopped_early is None
        ):
            if skip_in_epoch == 0:
                epoch_start_step = total_steps
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
                if skip_in_epoch > 0:
                    skip_in_epoch -= 1  # a batch the checkpoint already took: its step is not repeated
                    continue
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
                if config.checkpoint_above_tokens is not None:
                    self.model.checkpoint_this_batch(model_batch.padded_tokens > config.checkpoint_above_tokens)
                anchor = None
                ref_probs = None
                if config.kl_to_base > 0:
                    prefixes = tuple(config.kl_exclude_template_prefixes)
                    anchor = torch.tensor([not str(e.template_id).startswith(prefixes) if prefixes else True
                                           for e in model_batch.encoded], dtype=torch.bool)
                    if bool(anchor.any()):
                        ref_probs = self.model.reference_option_probs(model_batch)
                _logits, probs = self.model.batch_logits(model_batch)
                breakdown = ce_plus_brier(
                    probs,
                    model_batch.gold_flat,
                    model_batch.marker_item,
                    model_batch.batch_size,
                    lam=config.lambda_brier,
                )
                total_loss = breakdown.total
                kl_value = 0.0
                if ref_probs is not None and anchor is not None:
                    kl = kl_to_reference(probs, ref_probs.to(probs.device), model_batch.marker_item.to(probs.device),
                                         model_batch.batch_size, anchor)
                    total_loss = total_loss + config.kl_to_base * kl
                    kl_value = float(kl.detach())
                distill_value = 0.0
                n_teacher = 0
                if config.distill_weight > 0:
                    teacher_flat, teacher_mask = teacher_reference(
                        [enc.teacher_probs for enc in model_batch.encoded],
                        [int(n) for n in model_batch.n_options.tolist()],
                    )
                    n_teacher = int(teacher_mask.sum())
                    if n_teacher:
                        distill = kl_to_reference(probs, teacher_flat.to(probs.device, probs.dtype),
                                                  model_batch.marker_item.to(probs.device), model_batch.batch_size,
                                                  teacher_mask)
                        total_loss = total_loss + config.distill_weight * distill
                        distill_value = float(distill.detach())
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in self.optimizer.param_groups for p in group["params"]],
                        config.grad_clip,
                    )
                )
                if config.tripwire_grad_norm is not None and grad_norm > config.tripwire_grad_norm:
                    stopped_early = (
                        f"gradient-norm tripwire at step {total_steps + 1}: {grad_norm:.1f} > "
                        f"{config.tripwire_grad_norm:g}"
                    )
                    self._log({"event": "tripwire", "step": total_steps + 1, "grad_norm": grad_norm})
                    break
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
                    loss=float(total_loss.detach()),
                    ce=float(breakdown.ce.detach()),
                    brier=float(breakdown.brier.detach()),
                    accuracy=breakdown.accuracy,
                    n_items=model_batch.batch_size,
                    n_options=breakdown.n_options,
                    real_tokens=model_batch.real_tokens,
                    padded_tokens=model_batch.padded_tokens,
                    max_item_tokens=max(e.n_tokens for e in model_batch.encoded),
                    mean_item_tokens=float(
                        sum(e.n_tokens for e in model_batch.encoded) / len(model_batch.encoded)
                    ),
                    template_ids=[str(e.template_id) for e in model_batch.encoded],
                    elapsed_s=elapsed,
                    grad_norm=grad_norm,
                    gpu_peak_gb=peak,
                    lr_head=float(self.optimizer.param_groups[0]["lr"]),
                    lr_lora=float(self.optimizer.param_groups[-1]["lr"]),
                    kl=kl_value,
                    n_anchor=int(anchor.sum()) if anchor is not None else 0,
                    distill=distill_value,
                    n_teacher=n_teacher,
                )
                self.history.append(record)
                if on_step is not None:
                    on_step(total_steps)
                if config.log_every and total_steps % config.log_every == 0:
                    self._log({"event": "step", **record.to_dict()})
                if eval_items is not None and is_eval_step(total_steps, config.eval_every, config.eval_steps):
                    metrics = self.evaluate(eval_items, batch_size=config.batch_size)
                    self._log({"event": "eval", "step": total_steps, "metrics": metrics})
                    if self.eval_hook is not None:
                        self.eval_hook(total_steps, metrics)
                    key = config.selection_metric
                    if best is None or (
                        metrics[key],
                        -metrics["brier"],
                    ) > (best[key], -best["brier"]):
                        best, best_step = metrics, total_steps
                        if checkpoint_dir is not None:
                            self.save_checkpoint(checkpoint_dir)
                    self.model.train_mode()
                    if config.tripwire_macro_floor is not None:
                        if float(metrics[key]) < float(config.tripwire_macro_floor):
                            consecutive_below += 1
                        else:
                            consecutive_below = 0
                        if consecutive_below >= config.tripwire_consecutive_evals:
                            stopped_early = (
                                f"dev-macro tripwire at step {total_steps}: {consecutive_below} consecutive evals "
                                f"of {key} below {config.tripwire_macro_floor:g}"
                            )
                            self._log({"event": "tripwire", "step": total_steps,
                                       "metric": key, key: float(metrics[key])})
                            break
                if full_checkpoint_dir is not None and full_checkpoint_every and total_steps % full_checkpoint_every == 0:
                    write_full_checkpoint(
                        Path(full_checkpoint_dir) / f"step_{total_steps:06d}",
                        optimizer=self.optimizer,
                        run_state={
                            "step": total_steps, "epoch": epoch, "epoch_start_step": epoch_start_step,
                            "batches_in_epoch": total_steps - epoch_start_step, "epochs_completed": epochs_completed,
                            "best": best, "best_step": best_step, "consecutive_below": consecutive_below,
                            "schedule": schedule, "seed": config.seed,
                            "config_fingerprint": config_fingerprint(config.to_dict()),
                        },
                    )
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
