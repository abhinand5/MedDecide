"""Convert v0.2 benchmark rows to the evaluation schema (osler_v0 O2).

The scoreboard scores every kept v0.2 test row (fresh and tier-1), so this conversion covers all kept
templates, not only the fresh ones used as robustness bases in O1.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from meddecide.bench.schema import Option, QuestionType
from meddecide.panel.schema import EvalItem


def eval_item_from_v02_row(row: Mapping[str, Any], *, revision: str) -> EvalItem:
    """One v0.2 row (the benchmark's Item JSON) as an EvalItem. Keys, labels and gold are copied unchanged."""
    return EvalItem(
        item_id=row["item_id"],
        benchmark="v0.2",
        set_name=row["source"],
        template_id=row["template_id"],
        qtype=QuestionType(row["qtype"]),
        state=row["state"],
        question=row["question"],
        options=[Option(key=o["key"], label=o["label"], description=o.get("description"))
                 for o in row["options"]],
        gold=row["gold"],
        source_record_id=row["source_record_id"],
        source_url=row["source_url"],
        licence=row["source_license"],
        revision=revision,
        split=row["split"],
        meta={"tier": row["tier"], "skill": row["skill"]},
    )


def kept_test_rows(bench_dir: Path) -> Iterator[dict[str, Any]]:
    """Test rows of the templates the v0.2 screen keeps, across every file in the manifest."""
    manifest = json.loads((bench_dir / "manifest.json").read_text())
    screen = json.loads((bench_dir / "screen.json").read_text())
    kept = {t["template_id"] for t in screen["templates"] if not t["drop"]}
    root = bench_dir.parents[2]
    for rec in manifest["files"].values():
        with (root / rec["path"]).open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row["split"] == "test" and row["template_id"] in kept:
                    yield row
