"""O9 reads each arm's dev predictions from the directory that arm's config writes them to (ADVISORY O6-O8)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from meddecide.train.config import load_config

REPO = Path(__file__).resolve().parents[1]


def _o9_module():
    spec = importlib.util.spec_from_file_location("o9_head_choice", REPO / "scripts/osler/o9_head_choice.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_o9_reads_each_arm_from_the_directory_its_config_writes() -> None:
    module = _o9_module()
    assert set(module.ARM_DIRS) == {"L", "P", "N"}
    for arm, arm_dir in module.ARM_DIRS.items():
        config = load_config(REPO / f"configs/osler_v0/arm_{arm}.yaml")
        assert Path(config.output_dir) == Path("outputs/osler_v0") / arm_dir, arm
