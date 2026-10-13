"""O11: the held-out generators' test items as O2 common-set rows (CPU; no model).

The two held-out generators (loops/osler_v0/heldout.md; ``meddecide.gen.families.HELD_OUT``) are evaluated on their test items
only, in their own row, never pooled with the seen generators (heldout.md rule 3). Each generator item becomes one O2-format row,
so the scorer reads it like every other O2 row: the generator name is the template id (the O4 mix's convention, see
``meddecide.mix.converters.generator_row``), the family is the skill, and the patient id is the record id. Writes
data/gen/osler_v0/o11_heldout_test.jsonl and a manifest with the counts and the file's sha256 (both gitignored).

Usage:
    uv run --frozen python scripts/osler/o11_heldout_items.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from meddecide.gen.families import HELD_OUT
from meddecide.utils.io import count_by, iter_jsonl, write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
TEST = REPO / "data/gen/osler_v0/test.jsonl"
OUT = REPO / "data/gen/osler_v0/o11_heldout_test.jsonl"
MANIFEST = REPO / "data/gen/osler_v0/o11_heldout_test.manifest.json"
BENCHMARK = "osler_v0_heldout"
LICENCE = "generated-by-code (this repository)"


def to_o2_row(rec: dict[str, Any]) -> dict[str, Any]:
    """One held-out generator item as an O2 common-set row (the fields ``o2_row_to_item`` reads, plus the provenance)."""
    return {
        "item_id": rec["item_id"],
        "benchmark": BENCHMARK,
        "set_name": "generated",
        "template_id": rec["generator"],
        "qtype": rec["qtype"],
        "state": rec["state"],
        "question": rec["question"],
        "options": [{"key": o["key"], "label": o["label"]} for o in rec["options"]],
        "gold": rec["gold"],
        "split": rec["split"],
        "licence": LICENCE,
        "source_url": "",
        "source_record_id": rec["patient_id"],
        "meta": {"skill": rec["family"], "pair_id": rec["pair_id"], "role": rec["role"]},
    }


def held_out_records(path: Path) -> list[dict[str, Any]]:
    """The held-out rows of the generator test file. No silent drops: the generators must be exactly the registry's, every
    row must be on the test split, and item ids must be unique. Any other case raises."""
    held = [rec for rec in iter_jsonl(path) if rec.get("held_out") is True]
    generators = {rec["generator"] for rec in held}
    if generators != set(HELD_OUT):
        raise ValueError(f"held-out rows come from {sorted(generators)}; the registry lists {sorted(HELD_OUT)}")
    if any(rec["split"] != "test" for rec in held):
        raise ValueError("a held-out row is not on the test split")
    if len({rec["item_id"] for rec in held}) != len(held):
        raise ValueError("held-out item ids are not unique")
    return held


def main() -> None:
    rows = [to_o2_row(rec) for rec in held_out_records(TEST)]
    write_jsonl(OUT, rows)
    manifest = {
        "kind": "osler_v0_O11_heldout_items",
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "command": "scripts/osler/o11_heldout_items.py",
        "source": str(TEST.relative_to(REPO)),
        "output": str(OUT.relative_to(REPO)),
        "rows": len(rows),
        "rows_by_generator": count_by(rows, lambda r: r["template_id"]),
        "rows_by_qtype": count_by(rows, lambda r: r["qtype"]),
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }
    write_json(MANIFEST, manifest)
    print(json.dumps({k: manifest[k] for k in ("rows", "rows_by_generator", "rows_by_qtype", "sha256")}))


if __name__ == "__main__":
    main()
