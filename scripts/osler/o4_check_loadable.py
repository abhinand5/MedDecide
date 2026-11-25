"""O4 addendum: every mix v2 row must load as a validated training item (the trainer's reader is strict).

Converts each row of data/train/osler_v0/{train,dev}.jsonl with ``meddecide.train.mix_items.mix_row_to_item`` and counts the
rows that load and the failures by reason. Writes outputs/osler_v0/O4/loadable.json (gitignored). A failure is reported, not
repaired (R4): a non-zero count means the trainer would stop on that row.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o4_check_loadable.py
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/osler_v0/O4/loadable.json"


def check(path: Path) -> dict[str, object]:
    loaded = 0
    failures: Counter = Counter()
    examples: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            try:
                mix_row_to_item(row)
                loaded += 1
            except (ValidationError, ValueError, KeyError) as exc:
                reason = type(exc).__name__ + ": " + str(exc).splitlines()[0][:120]
                failures[reason] += 1
                examples.setdefault(reason, row["template_id"])
    return {"rows": loaded + sum(failures.values()), "loaded": loaded,
            "failed": sum(failures.values()), "failures_by_reason": dict(failures),
            "example_template_per_reason": examples}


def main() -> None:
    started = time.time()
    report = {"built_at_utc": utcnow(), "git_commit": git_commit(REPO), "splits": {}}
    for name in ("train", "dev"):
        report["splits"][name] = check(REPO / f"data/train/osler_v0/{name}.jsonl")
        print(f"{name}: {report['splits'][name]['loaded']} loaded of {report['splits'][name]['rows']}", flush=True)
    report["wall_clock_s"] = round(time.time() - started, 1)
    write_json(OUT, report)


if __name__ == "__main__":
    main()
