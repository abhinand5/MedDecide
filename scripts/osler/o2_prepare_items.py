"""O2: the common item set that every competitor and baseline is scored on.

Sources: kept v0.2 test rows (37,289 items), the external panel (4,353) and the robustness pack (20,898).
Each item's prompt length is measured with the scoring prompt (meddecide.eval.readout.render_prompt) under
the Qwen3.5 tokenizer, the same definition as O1's long-record slice. Items over the common cap of 8,192
tokens are excluded for every model, so all models share one denominator; the exclusions are counted per
source and set in the manifest. Competitor code caps (JEV 8,192; pplx 8,192; Clef 16,384) are then
satisfied by construction for the common set, except where a model's own tokenizer is longer (recorded at
scoring time).

Outputs: data/bench/v0.3_ext/o2_items.jsonl (in-set items, gitignored) and the manifest section `o2_items`.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o2_prepare_items.py
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from meddecide.eval.readout import render_prompt
from meddecide.panel.schema import EvalItem
from meddecide.panel.v02 import eval_item_from_v02_row, kept_test_rows
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "data/bench/v0.3_ext"
OUT = EXT / "o2_items.jsonl"
MANIFEST = EXT / "manifest.json"
TOKENIZER = "Qwen/Qwen3.5-0.8B"
CAP = 8192
V02_REVISION = "benchmark v0.2 (manifest sha256 363c037e)"


def sources() -> list[tuple[str, EvalItem]]:
    items: list[tuple[str, EvalItem]] = []
    for row in kept_test_rows(REPO / "data/bench/v0.2"):
        items.append(("v0.2", eval_item_from_v02_row(row, revision=V02_REVISION)))
    for name in ["panel.jsonl", "robustness.jsonl"]:
        for line in (EXT / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append((name.split(".")[0], EvalItem.model_validate_json(line)))
    return items


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    counts: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    tokens_by_source: dict[str, list[int]] = {}
    kept_rows: list[dict[str, Any]] = []
    for source, item in sources():
        n = len(tok.encode(render_prompt(item, tok), add_special_tokens=False))
        tokens_by_source.setdefault(f"{source}:{item.set_name}", []).append(n)
        counts[f"{source}:{item.set_name}"] += 1
        if n > CAP:
            excluded[f"{source}:{item.set_name}"] += 1
            continue
        row = item.model_dump(mode="json")
        row["qwen_prompt_tokens"] = n
        kept_rows.append(row)
    EXT.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for row in kept_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    per_source = {
        key: {
            "items": counts[key],
            "over_common_cap_excluded": excluded.get(key, 0),
            "in_common_set": counts[key] - excluded.get(key, 0),
            "prompt_tokens_median": int(statistics.median(tokens_by_source[key])),
            "prompt_tokens_max": max(tokens_by_source[key]),
        }
        for key in sorted(counts)
    }
    section = {
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "cap_tokens": CAP,
        "tokenizer": TOKENIZER,
        "prompt": "meddecide.eval.readout.render_prompt (chat template, thinking off)",
        "items_total": sum(counts.values()),
        "items_in_common_set": len(kept_rows),
        "items_over_common_cap": sum(excluded.values()),
        "per_source_and_set": per_source,
        "file": str(OUT.relative_to(REPO)),
        "file_sha256": file_sha256(OUT),
        "rule": "every model is scored on the common set only; per-model code caps are recorded at scoring time",
    }
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest["o2_items"] = section
    write_json(MANIFEST, manifest)
    print(f"items total={section['items_total']} in common set={section['items_in_common_set']} "
          f"over cap={section['items_over_common_cap']}")
    for key, v in per_source.items():
        print(f"  {key:40s} items={v['items']:6d} in_set={v['in_common_set']:6d} over_cap={v['over_common_cap_excluded']:5d}")


if __name__ == "__main__":
    main()
