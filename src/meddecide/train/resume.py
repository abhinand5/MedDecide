"""osler_v4 Y8 (operator steer, 2026-10-23): full training checkpoints for a resumable final run (pure I/O helpers).

A full checkpoint is a directory step_<NNNNNN>/ holding:
- trainable.pt: the trainable tensors of the model by parameter name (the LoRA adapter and the readout head);
- optimizer.pt: the optimizer state (the AdamW moments and step counts) with its parameter groups;
- rng.pt: the Python, NumPy and torch random states at the checkpoint;
- run_state.json: the step, the epoch, the step at which the epoch began, the batches already taken in the epoch, the best metric and
  step, the tripwire counter, the LR schedule, the seed and a fingerprint of the training configuration;
- COMPLETE: written last, so a checkpoint interrupted while being written is never used.

The data order needs no saved state: the epoch order and every batch's option permutation are pure functions of (seed, epoch, batch
index) (meddecide.train.data), so a resumed run skips the batches its checkpoint already took in the epoch and continues identically.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

STEP_DIR = re.compile(r"^step_(\d{6})$")
COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class FullCheckpoint:
    trainable: dict[str, torch.Tensor]
    optimizer: dict[str, Any]
    rng: dict[str, Any]
    run_state: dict[str, Any]


def _json_default(value: Any) -> Any:
    """NumPy scalars (metric values) become Python numbers; anything else is an error, not a silent string."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def truncate_log_after(path: Path, step: int) -> int:
    """Drop the JSONL records of a training log written after ``step`` (a resume rewrites them); returns the number dropped.

    Records without a ``step`` field (run events) are kept. The file is replaced atomically.
    """
    if not path.exists():
        return 0
    kept: list[str] = []
    dropped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "step" in record and int(record["step"]) > step:
            dropped += 1
        else:
            kept.append(line)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    tmp.replace(path)
    return dropped


def config_fingerprint(config: dict[str, Any]) -> str:
    """A stable hash of the training configuration: a resume must use the configuration that was saved."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def optimizer_params(optimizer: torch.optim.Optimizer) -> dict[str, torch.Tensor]:
    """The trainable tensors, keyed by parameter-group and position. The optimizer holds exactly the trainable tensors (the model is
    not a torch module, so it has no named parameters); the keys are stable because the groups are built in a fixed order."""
    return {f"{g}:{i}": param.detach().cpu().clone()
            for g, group in enumerate(optimizer.param_groups) for i, param in enumerate(group["params"])}


def restore_optimizer_params(optimizer: torch.optim.Optimizer, values: dict[str, torch.Tensor]) -> None:
    """Copy saved values into the optimizer's tensors; every key must be present and every shape must match."""
    expected = optimizer_params(optimizer)
    if set(values) != set(expected):
        raise ValueError("the checkpoint's trainable tensors do not match the model's")
    for g, group in enumerate(optimizer.param_groups):
        for i, param in enumerate(group["params"]):
            saved = values[f"{g}:{i}"]
            if tuple(saved.shape) != tuple(param.shape):
                raise ValueError(f"trainable tensor {g}:{i} has shape {tuple(saved.shape)}, the model has {tuple(param.shape)}")
            param.data.copy_(saved.to(device=param.device, dtype=param.dtype))


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def write_full_checkpoint(directory: Path, *, optimizer: torch.optim.Optimizer, run_state: dict[str, Any],
                          keep_last: int = 0) -> Path:
    """Write a full checkpoint atomically: the files go to a partial directory, which is renamed into place last."""
    partial = directory.parent / f"{directory.name}.partial"
    if keep_last > 0:  # the disk quota is small: before the write, keep only keep_last - 1 older complete checkpoints
        older = [p for p in complete_checkpoints(directory.parent) if p != directory]
        for old in older[: max(0, len(older) - (keep_last - 1))]:
            shutil.rmtree(old)
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    torch.save(optimizer_params(optimizer), partial / "trainable.pt")
    torch.save(optimizer.state_dict(), partial / "optimizer.pt")
    torch.save(rng_state(), partial / "rng.pt")
    (partial / "run_state.json").write_text(json.dumps(run_state, indent=2, sort_keys=True, default=_json_default) + "\n",
                                            encoding="utf-8")
    (partial / COMPLETE).write_text("complete\n", encoding="utf-8")
    if directory.exists():
        shutil.rmtree(directory)
    partial.rename(directory)
    return directory


def read_full_checkpoint(directory: Path) -> FullCheckpoint:
    if not (directory / COMPLETE).exists():
        raise ValueError(f"{directory} is not a complete full checkpoint")
    return FullCheckpoint(
        trainable=torch.load(directory / "trainable.pt", map_location="cpu", weights_only=False),
        optimizer=torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=False),
        rng=torch.load(directory / "rng.pt", map_location="cpu", weights_only=False),
        run_state=json.loads((directory / "run_state.json").read_text(encoding="utf-8")),
    )


def complete_checkpoints(root: Path) -> list[Path]:
    """The complete step_<NNNNNN> directories under root, oldest first."""
    if not root.exists():
        return []
    found: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = STEP_DIR.match(child.name)
        if match and (child / COMPLETE).exists():
            found.append((int(match.group(1)), child))
    return [child for _, child in sorted(found)]


def latest_full_checkpoint(root: Path) -> Path | None:
    """The complete step_<NNNNNN> directory with the largest step under root, or None."""
    found = complete_checkpoints(root)
    return found[-1] if found else None
