"""W2 step 1 (osler_v2 ADVISORY W2; D36): share-alike and permissive replay, one fixed set for every block.

Sources (train splits only; licences recorded in `meddecide.mix.replay_v2.CANDIDATES_V2`): ARC-Easy and ARC-Challenge (CC BY-SA
4.0), BoolQ (CC BY-SA 3.0), SNLI (CC BY-SA 4.0). OpenBookQA is not used (no licence on its Hub card).

The SNLI sample is the largest that keeps every template at or below 8 % of block C0 (ADVISORY W6 cap), chosen by a stable
hash of the item id, so the set does not depend on file order. The leakage check is the osler_v0 protected sets (every test
and dev set of the earlier loops), the same check as osler_v1's V3.

Shares are reported for the planned blocks: the replay set plus the B1 rows, then C1 (real text, upper bound), C2 (prose, upper
bound), and the chart rows at which C3 would fall below 20 %. Recompute at W6, where the real block sizes are known.

Outputs: data/train/osler_v2/replay_v2_train.jsonl (gitignored) and outputs/osler_v2/W2/replay_manifest.json.

Usage: source scripts/pod_env.sh && uv run --frozen python scripts/osler_v2/w2_replay.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from meddecide.mix.leakage import row_text_key
from meddecide.mix.replay_v2 import CANDIDATES_V2, arc_row, boolq_row, snli_row
from meddecide.utils.hashing import file_sha256, stable_hash
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
B1 = REPO / "data" / "train" / "osler_v1" / "block_B1.jsonl"
OUT_DATA = REPO / "data" / "train" / "osler_v2" / "replay_v2_train.jsonl"
OUT_MANIFEST = REPO / "outputs" / "osler_v2" / "W2" / "replay_manifest.json"
REPLAY_TEMPLATES_V1 = {"hellaswag_ending_choice_v1", "winogrande_blank_choice_v1", "csqa_choice_v1", "qasc_science_choice_v1"}
TEMPLATE_CAP = 0.08  # ADVISORY W6: no template above 8 % of a block
REPLAY_TARGET = 0.20  # ADVISORY W2 / D36
# Real-text rows that join at C1 (upper bounds from osler_v1's realtext files; EHR-Rel is added after its own step):
# pmc_similar_patient_v1 train 52,549; codiesp_icd_support_v1 train 3,374 (its dev carve-out is taken later, so this is an upper
# bound); mtsamples_page_category_v1 train 668.
REAL_TEXT_ROWS_C1 = 52_549 + 3_374 + 668
PROSE_ROWS_C2 = 14_903 + 16_450  # osler_v1 verified teacher pages: indicator pairs and narrative pages (upper bound)
SOURCES = {
    "ai2_arc": {"repo": "allenai/ai2_arc", "patterns": ["ARC-Easy/train-*.parquet", "ARC-Challenge/train-*.parquet"]},
    "boolq": {"repo": "google/boolq", "patterns": ["data/train-*.parquet"]},
    "snli": {"repo": "stanfordnlp/snli", "patterns": ["plain_text/train-*.parquet"]},
}


def load_protected() -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    """The osler_v0 protected sets, imported from the osler_v0 mix builder (the same check osler_v1's V3 used)."""
    path = REPO / "scripts" / "osler" / "o4_build_mix.py"
    spec = importlib.util.spec_from_file_location("o4_build_mix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protected = module.protected_sets()["train"]
    return protected["text"], protected["bare"], protected["qualified"]


def read_split(repo: str, patterns: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    local = Path(snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns))
    files = sorted(p for pattern in patterns for p in local.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no train parquet for {repo} under {patterns}")
    records: list[dict[str, Any]] = []
    for path in files:
        records.extend(pq.read_table(path).to_pylist())
    return records, [str(p.relative_to(local)) for p in files]


def protected_hits(row: dict[str, Any], protected: tuple[dict, dict, dict]) -> list[str]:
    """Every osler_v0 protected set the row overlaps (text, bare record id, or source-qualified record id)."""
    text_sets, bare_sets, qualified_sets = protected
    key = row_text_key(row)
    bare = str(row["source_record_id"])
    hits = [f"text:{n}" for n, keys in text_sets.items() if key in keys]
    hits += [f"record_bare:{n}" for n, ids in bare_sets.items() if bare in ids]
    hits += [f"record_qualified:{n}" for n, ids in qualified_sets.items() if f"{row['source']}/{bare}" in ids]
    return hits


def b1_counts() -> tuple[int, int]:
    """Rows of block B1 and how many of them are the osler_v1 replay templates (no text is kept)."""
    total = replay = 0
    with B1.open(encoding="utf-8") as fh:
        for line in fh:
            total += 1
            if json.loads(line)["template_id"] in REPLAY_TEMPLATES_V1:
                replay += 1
    return total, replay


def convert(name: str, records: list[dict[str, Any]], protected: tuple[dict, dict, dict]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    leaks: Counter[str] = Counter()
    for index, rec in enumerate(records):
        if name == "boolq":
            row, reason = boolq_row(rec, "train", index)
        else:
            row, reason = snli_row(rec, "train", index)
        reasons[reason] += 1
        if row is None:
            continue
        hits = protected_hits(row, protected)
        if hits:
            for h in hits:
                leaks[h] += 1
            reasons["dropped_protected_overlap"] += 1
            continue
        rows.append(row)
    return rows, {"records_read": len(records), "kept": len(rows), "reasons": dict(reasons),
                  "leakage_hits": dict(leaks)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    protected = load_protected()
    report: dict[str, Any] = {}
    arc_rows: list[dict[str, Any]] = []
    arc_report: dict[str, Any] = {}
    for config in ("ARC-Easy", "ARC-Challenge"):
        records, files = read_split("allenai/ai2_arc", [f"{config}/train-*.parquet"])
        kept: list[dict[str, Any]] = []
        reasons: Counter[str] = Counter()
        for rec in records:
            row, reason = arc_row(rec, "train", config)
            reasons[reason] += 1
            if row is not None:
                if protected_hits(row, protected):
                    reasons["dropped_protected_overlap"] += 1
                else:
                    kept.append(row)
        arc_rows.extend(kept)
        arc_report[config] = {"files": files, "records_read": len(records), "kept": len(kept), "reasons": dict(reasons)}
    report["ai2_arc"] = {"licence": CANDIDATES_V2["ai2_arc"][1], "configs": arc_report, "kept": len(arc_rows)}

    rows_by_source: dict[str, list[dict[str, Any]]] = {"ai2_arc": arc_rows}
    for name in ("boolq", "snli"):
        spec = SOURCES[name]
        records, files = read_split(spec["repo"], spec["patterns"])
        rows, info = convert(name, records, protected)
        rows_by_source[name] = rows
        report[name] = {"licence": CANDIDATES_V2[name][1], "files": files, **info}
        print(name, {k: v for k, v in info.items() if k != "leakage_hits"}, flush=True)

    # The SNLI cap: the largest sample that keeps the SNLI template at or below 8 % of C0 (B1 plus the other new rows).
    b1_total, b1_replay = b1_counts()
    other_new = len(rows_by_source["ai2_arc"]) + len(rows_by_source["boolq"])
    snli_cap = int(TEMPLATE_CAP * (b1_total + other_new) / (1 - TEMPLATE_CAP))
    snli_sorted = sorted(rows_by_source["snli"], key=lambda r: stable_hash({"item": r["item_id"]}, length=16))
    snli_kept = snli_sorted[:min(snli_cap, len(snli_sorted))]
    rows = rows_by_source["ai2_arc"] + rows_by_source["boolq"] + snli_kept
    new_replay = len(rows)

    def share(extra: int, replay_rows: int, base_total: int) -> float:
        return round((b1_replay + replay_rows) / (base_total + replay_rows + extra), 4)

    c0_total = b1_total + new_replay
    blocks = {
        "C0 (B1 + fixed replay)": share(0, new_replay, b1_total),
        "C1 (C0 + real text, upper bound)": share(REAL_TEXT_ROWS_C1, new_replay, b1_total),
        "C2 (C1 + teacher prose, upper bound)": share(REAL_TEXT_ROWS_C1 + PROSE_ROWS_C2, new_replay, b1_total),
    }
    # charts that C3 may add before its share falls below 20 %
    c2_extra = REAL_TEXT_ROWS_C1 + PROSE_ROWS_C2
    replay_total = b1_replay + new_replay
    max_total_for_target = int(replay_total / REPLAY_TARGET)
    charts_max_for_target = max_total_for_target - (b1_total + new_replay + c2_extra)
    write_jsonl(OUT_DATA, rows)
    manifest = {
        "task": "W2",
        "written_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "rule": "D36: share-alike allowed for training only; train splits only; leakage check against osler_v0 protected sets",
        "sources": report,
        "snli_cap_rows_at_8_percent_of_C0": snli_cap,
        "snli_available_after_filters": len(rows_by_source["snli"]),
        "new_replay_rows": {"ai2_arc": len(rows_by_source["ai2_arc"]), "boolq": len(rows_by_source["boolq"]),
                            "snli": len(snli_kept), "total": new_replay},
        "not_used": {"openbookqa": CANDIDATES_V2["openbookqa"]},
        "b1_rows": b1_total,
        "b1_replay_rows_v1_templates": b1_replay,
        "b1_replay_share": round(b1_replay / b1_total, 4),
        "c0_rows": c0_total,
        "replay_share_by_block": blocks,
        "replay_target": REPLAY_TARGET,
        "c3_chart_rows_for_target": charts_max_for_target,
        "rows_file": str(OUT_DATA.relative_to(REPO)),
        "rows_sha256": file_sha256(OUT_DATA),
    }
    write_json(OUT_MANIFEST, manifest)
    print(json.dumps({"new_replay_rows": manifest["new_replay_rows"], "replay_share_by_block": blocks,
                      "c3_chart_rows_for_target": charts_max_for_target}), flush=True)


if __name__ == "__main__":
    main()
