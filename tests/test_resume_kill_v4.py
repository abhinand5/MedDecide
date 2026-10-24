"""osler_v4 Y8 resume (operator steer, 2026-10-23): a training process killed mid-run resumes from its latest full checkpoint and matches an
uninterrupted run. The victim runs in its own process, which is killed (SIGKILL) once its first full checkpoint is complete; the tiny CPU
model and the helpers are those of test_resume_trainer_v4.py."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from test_resume_trainer_v4 import _items, _trainer, build_tiny_dir

from meddecide.train.resume import latest_full_checkpoint, optimizer_params


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory) -> Path:
    return build_tiny_dir(tmp_path_factory)

VICTIM = """
import sys
from pathlib import Path
import torch
sys.path.insert(0, {tests!r})
from test_resume_trainer_v4 import _items, _trainer
trainer, _ = _trainer(Path({tiny!r}))
trainer.train(_items(), steps=100, schedule_steps=100, full_checkpoint_dir=Path({out!r}), full_checkpoint_every=3)
"""


def test_a_killed_run_resumes_to_the_uninterrupted_result(tiny_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "full"
    tests_dir = Path(__file__).resolve().parent
    code = VICTIM.format(tests=str(tests_dir), tiny=str(tiny_dir), out=str(out))
    victim = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 600
        while latest_full_checkpoint(out) is None:
            if victim.poll() is not None:
                raise AssertionError("the victim exited before its first full checkpoint")
            if time.monotonic() > deadline:
                raise AssertionError("no full checkpoint within 600 s")
            time.sleep(0.2)
    finally:
        victim.kill()
        victim.wait()
    checkpoint = latest_full_checkpoint(out)
    assert checkpoint is not None and victim.returncode != 0  # it was killed, not finished

    straight, _ = _trainer(tiny_dir)
    straight.train(_items(), steps=12, schedule_steps=100)

    resumed, _ = _trainer(tiny_dir)
    resumed.train(_items(), steps=12, schedule_steps=100, resume_from=checkpoint)
    expected, actual = optimizer_params(straight.optimizer), optimizer_params(resumed.optimizer)
    assert set(expected) == set(actual)
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name
