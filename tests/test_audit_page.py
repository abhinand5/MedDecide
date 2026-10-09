"""Runs the audit page's node-based logic test from pytest, when node is available."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TEST_SCRIPT = REPO / "tests" / "test_audit_page.mjs"
PAGE = REPO / "tools" / "audit" / "audit.html"


def test_audit_page_exists_and_is_self_contained() -> None:
    assert PAGE.is_file(), "tools/audit/audit.html must be committed"
    html = PAGE.read_text(encoding="utf-8")
    # no network, no external libraries: the operator opens the file directly
    for needle in ("http://", "https://", "<link", "src=\"http", "import "):
        assert needle not in html, f"audit.html must not reference {needle!r}"
    for key in ("localStorage", "audit_v0.jsonl", "ArrowRight", "ArrowLeft"):
        assert key in html, f"audit.html is missing {key!r}"


def test_audit_page_logic_via_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available; the browser logic test needs node")
    result = subprocess.run(
        [node, str(TEST_SCRIPT)], capture_output=True, text=True, cwd=str(REPO), timeout=120
    )
    assert result.returncode == 0, f"node test failed:\n{result.stdout}\n{result.stderr}"
    assert "all checks passed" in result.stdout
