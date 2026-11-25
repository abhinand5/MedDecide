"""osler_v4 Y8 (ADVISORY Y8; operator ruling item 3): compose the final mix in memory from its recorded sources and print its counts (CPU).

The same composition the Y8 driver makes (scripts/osler_v4/y8_train.py): the entering MIMIC-family blocks by the operator's item 1, the
replay top-up, and the three caps. Nothing is written to disk (the disk quota cannot hold the ~9.5 GB mix file). It prints counts, shares,
the cap checks and the composition hash only; no item text or identifier is printed.

Usage (CPU):
    uv run --frozen python scripts/osler_v4/y8_compose_check.py
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

from meddecide.train.final_mix import compose_final_mix

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    spec = importlib.util.spec_from_file_location("y8_final_mix", REPO / "scripts" / "osler_v4" / "y8_final_mix.py")
    assert spec is not None and spec.loader is not None
    y8_mix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(y8_mix)
    decisions = {name: y8_mix.entering(name) for name in ("B1", "B2", "B3", "B4")}
    family = {name: y8_mix.add_files(name) for name in y8_mix.FAMILY if decisions[name][0]}
    other = y8_mix.add_files("B4") if decisions["B4"][0] else []
    print("entering:", {name: decisions[name][0] for name in decisions}, flush=True)
    started = time.time()
    rows, report = compose_final_mix(y8_mix.BASE, family, other, y8_mix.BROAD)
    print(f"composed rows {len(rows)} in {round(time.time() - started)} s", flush=True)
    keys = ("rows", "base_rows", "family_available", "family_limit", "family_kept", "topup_rows", "topup_pool_rows", "replay_rows",
            "replay_share", "family_share", "largest_template", "largest_template_share", "templates", "checks", "composition_sha256")
    print(json.dumps({k: report[k] for k in keys}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
