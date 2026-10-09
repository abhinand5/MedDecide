"""O1 fix 3 (student_v1 carry-over): long-record slice membership for v0.2 (ids only), and the flags
for templates whose items exceed the prompt cap.

The slice is defined as in student_v1: the scoring prompt (chat template, thinking off, as the harness
renders it) has more than 8,192 tokens under the Qwen3.5 tokenizer. Membership is stored as item ids
only (no item text). Per-template counts are compared with student_v1's recorded counts for the same
templates (the counts live in data/bench/v0.2/long_record.json).

Outputs: data/bench/v0.3_ext/long_record_ids.json (ids and counts) and a manifest section.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o1_long_slice.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from transformers import AutoTokenizer

from meddecide.eval.readout import render_prompt
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "data/bench/v0.2"
OUT_DIR = REPO / "data/bench/v0.3_ext"
TOKENIZER = "Qwen/Qwen3.5-0.8B"
CAP = 8192
OPENFDA_TEMPLATES_FLAGGED = ["fda_route_claim_noul_v1", "fda_boxed_warning_noul_v1", "fda_class_choice_v2"]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    manifest = json.loads((BENCH / "manifest.json").read_text())
    screen = json.loads((BENCH / "screen.json").read_text())
    kept = {t["template_id"] for t in screen["templates"] if not t["drop"]}
    tier_of = {t["template_id"]: t["tier"] for t in screen["templates"]}

    long_ids: dict[str, list[str]] = defaultdict(list)
    counts_total: Counter[str] = Counter()
    counts_long: Counter[str] = Counter()
    lengths: list[int] = []
    scanned = 0
    for rec in manifest["files"].values():
        for raw in (REPO / rec["path"]).open(encoding="utf-8"):
            row = json.loads(raw)
            if row["split"] != "test" or row["template_id"] not in kept:
                continue
            scanned += 1
            item = SimpleNamespace(
                state=row["state"], question=row["question"],
                options=[SimpleNamespace(key=o["key"], label=o["label"]) for o in row["options"]],
            )
            n = len(tok.encode(render_prompt(item, tok), add_special_tokens=False))
            lengths.append(n)
            counts_total[row["template_id"]] += 1
            if n > CAP:
                counts_long[row["template_id"]] += 1
                long_ids[row["template_id"]].append(row["item_id"])

    # student_v1's recorded counts, for the same templates where the file records them
    student = json.loads((BENCH / "long_record.json").read_text())
    recorded = {tid: v.get("n_long") for tid, v in student.get("per_template", {}).items()} \
        if isinstance(student.get("per_template"), dict) else {}
    comparison = {tid: {"ours_long": counts_long.get(tid, 0), "recorded_in_v0_2_long_record_json": recorded.get(tid)}
                  for tid in sorted(counts_total)}
    flagged = {tid: {"items": counts_total[tid], "over_cap": counts_long.get(tid, 0)}
               for tid in OPENFDA_TEMPLATES_FLAGGED if tid in counts_total}
    tier_counts = Counter(tier_of[tid] for tid in counts_total for _ in range(counts_total[tid]))
    payload: dict[str, Any] = {
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "definition": f"scoring prompt (render_prompt, chat template, thinking off) > {CAP} tokens under {TOKENIZER}",
        "scope": "v0.2 test rows of the 19 kept templates (screen.json); fresh and established",
        "rows_scanned": scanned,
        "over_cap_total": sum(counts_long.values()),
        "over_cap_by_tier": {t: sum(counts_long[tid] for tid in counts_long if tier_of[tid] == t)
                             for t in ("fresh", "tier1")},
        "tier_row_counts": dict(tier_counts),
        "prompt_tokens": {"median": sorted(lengths)[len(lengths) // 2], "p95": sorted(lengths)[int(0.95 * len(lengths))],
                          "max": max(lengths)},
        "flagged_openfda_templates": flagged,
        "compared_with_v0_2_long_record_json": comparison,
        "cap_policy": {
            "default_cap_tokens": CAP,
            "rule": "every model is scored with the same cap unless its authors' code cannot take it; items over the "
                    "cap are excluded for that model and the exclusion is counted per model at O2",
            "authors_code_caps": {"autotrust JEV (F7 harness)": 8192,
                                  "perplexity pplx-decider (DecisionModel.prepare max_length)": 8192,
                                  "lion-ai MedDecider (meddecider_infer.py)": "no cap in the code; scored in full"},
        },
    }
    write_json(OUT_DIR / "long_record_ids.json", {"payload": payload, "ids_over_cap": dict(sorted(long_ids.items()))})
    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["long_record"] = {k: v for k, v in payload.items() if k != "compared_with_v0_2_long_record_json"}
    manifest["long_record"]["ids_file"] = "data/bench/v0.3_ext/long_record_ids.json"
    manifest["long_record"]["compared_with_v0_2_long_record_json"] = comparison
    write_json(manifest_path, manifest)
    print(f"rows scanned={scanned} over_cap_total={payload['over_cap_total']} flagged={flagged}")
    print(f"over_cap_by_tier={payload['over_cap_by_tier']} prompt_tokens={payload['prompt_tokens']}")


if __name__ == "__main__":
    main()
