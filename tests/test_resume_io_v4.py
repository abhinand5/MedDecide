"""osler_v4 Y8 resume (operator steer, 2026-10-23): the full-checkpoint I/O round-trips exactly and picks the latest complete checkpoint."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from meddecide.train.resume import (
    config_fingerprint,
    latest_full_checkpoint,
    optimizer_params,
    read_full_checkpoint,
    restore_optimizer_params,
    rng_state,
    set_rng_state,
    truncate_log_after,
    write_full_checkpoint,
)


def _model_and_optimizer() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    model[1].bias.requires_grad_(False)  # a frozen tensor: it is not in the optimizer and must not be saved
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
    return model, optimizer


def test_full_checkpoint_round_trips_the_trainable_tensors_and_the_optimizer(tmp_path: Path) -> None:
    _, optimizer = _model_and_optimizer()
    run_state = {"step": 3, "epoch": 0, "epoch_start_step": 0, "batches_in_epoch": 3}
    directory = write_full_checkpoint(tmp_path / "step_000003", optimizer=optimizer, run_state=run_state)
    saved = read_full_checkpoint(directory)
    assert saved.run_state == run_state
    expected = optimizer_params(optimizer)
    assert set(saved.trainable) == set(expected)
    assert len(expected) == 3  # the first layer's weight and bias and the second layer's weight; the frozen second bias is absent
    for key, tensor in expected.items():
        assert torch.equal(saved.trainable[key], tensor)
    original = optimizer.state_dict()["state"]
    for key, entry in saved.optimizer["state"].items():
        for field, value in entry.items():
            assert torch.equal(value, original[key][field])
    assert saved.optimizer["param_groups"] == optimizer.state_dict()["param_groups"]


def test_restore_copies_the_saved_values_into_a_fresh_model_exactly(tmp_path: Path) -> None:
    _, trained = _model_and_optimizer()
    directory = write_full_checkpoint(tmp_path / "step_000003", optimizer=trained, run_state={"step": 3})
    saved = read_full_checkpoint(directory)
    _, fresh = _model_and_optimizer()
    torch.manual_seed(7)  # a different initialisation, so the copy is what makes them equal
    with torch.no_grad():
        for group in fresh.param_groups:
            for param in group["params"]:
                param.add_(torch.randn_like(param))
    restore_optimizer_params(fresh, saved.trainable)
    for key, tensor in optimizer_params(trained).items():
        assert torch.equal(optimizer_params(fresh)[key], tensor)
    with pytest.raises(ValueError):
        restore_optimizer_params(fresh, {"0:0": saved.trainable["0:0"]})


def test_rng_state_restores_the_python_numpy_and_torch_streams() -> None:
    state = rng_state()
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_rng_state(state)
    again = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == again


def test_latest_full_checkpoint_takes_the_largest_complete_step(tmp_path: Path) -> None:
    _, optimizer = _model_and_optimizer()
    for step in (5000, 10000):
        write_full_checkpoint(tmp_path / f"step_{step:06d}", optimizer=optimizer, run_state={"step": step})
    partial = tmp_path / "step_015000"
    partial.mkdir()  # a checkpoint without COMPLETE is not used
    (partial / "run_state.json").write_text("{}", encoding="utf-8")
    assert latest_full_checkpoint(tmp_path) == tmp_path / "step_010000"
    assert latest_full_checkpoint(tmp_path / "missing") is None


def test_an_incomplete_checkpoint_is_refused(tmp_path: Path) -> None:
    (tmp_path / "step_000001").mkdir()
    with pytest.raises(ValueError):
        read_full_checkpoint(tmp_path / "step_000001")


def test_config_fingerprint_is_stable_and_sensitive() -> None:
    config = {"lr": 1e-4, "seed": 0, "kl_to_base": 0.0}
    assert config_fingerprint(config) == config_fingerprint(dict(reversed(list(config.items()))))
    assert config_fingerprint(config) != config_fingerprint({**config, "kl_to_base": 0.1})


def test_truncate_log_after_drops_only_later_steps(tmp_path: Path) -> None:
    log = tmp_path / "train.jsonl"
    log.write_text('{"event": "step", "step": 1}\n{"event": "step", "step": 2}\n{"event": "eval", "step": 3}\n'
                   '{"event": "run"}\n{"event": "step", "step": 4}\n', encoding="utf-8")
    assert truncate_log_after(log, 2) == 2
    kept = list(log.read_text(encoding="utf-8").splitlines())
    assert kept == ['{"event": "step", "step": 1}', '{"event": "step", "step": 2}', '{"event": "run"}']
    assert truncate_log_after(tmp_path / "missing.jsonl", 2) == 0
