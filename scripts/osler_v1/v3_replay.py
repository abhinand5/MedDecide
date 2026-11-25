"""V3: general human-labelled replay for osler_v1 (ADVISORY V3; D20(b), permissive licences only).

Reads the train splits of the permissive candidates (HellaSwag, WinoGrande), converts them with
``meddecide.mix.replay_v1``, counts every kept and dropped row by reason, runs the leakage check of the osler_v0 mix builder
(v0.2 dev and test, the external panel, the robustness pack and the held-out sets) and writes the replay rows and a manifest.

The replay share is measured against mix v2 (the osler_v0 mix: 573,611 train rows, 19,829 of them replay). The ADVISORY's
20 % target is reported as it is measured; nothing is removed from the non-replay rows to reach it (ADVISORY V3: "add").

Outputs: data/train/osler_v1/replay_v1_train.jsonl (gitignored) and outputs/osler_v1/V3/replay_manifest.json.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v3_replay.py
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
from meddecide.mix.replay_v1 import CANDIDATES, hellaswag_row, winogrande_row
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
OUT_DATA = REPO / "data" / "train" / "osler_v1" / "replay_v1_train.jsonl"
OUT_MANIFEST = REPO / "outputs" / "osler_v1" / "V3" / "replay_manifest.json"
MIX_V2_TRAIN_ROWS = 573_611          # loops/osler_v0/mix_v2_manifest.json: train_rows
MIX_V2_REPLAY_ROWS = 19_829          # loops/osler_v0/mix_v2_manifest.json: replay_rows
ADVISORY_REPLAY_SHARE = 0.20

SOURCES: dict[str, dict[str, Any]] = {
    "hellaswag": {"repo": "Rowan/hellaswag", "patterns": ["data/train-*.parquet"], "convert": hellaswag_row},
    "winogrande": {"repo": "allenai/winogrande", "patterns": ["winogrande_xl/train-*.parquet"],
                   "convert": winogrande_row},
}


def load_protected() -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    """The osler_v0 protected sets, imported from the osler_v0 mix builder (same keys, same counting)."""
    path = REPO / "scripts" / "osler" / "o4_build_mix.py"
    spec = importlib.util.spec_from_file_location("o4_build_mix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protected = module.protected_sets()["train"]
    return protected["text"], protected["bare"], protected["qualified"]


def read_train(repo: str, patterns: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    local = Path(snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns))
    files = sorted(p for pattern in patterns for p in local.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no train parquet for {repo} under {patterns}")
    records: list[dict[str, Any]] = []
    for path in files:
        records.extend(pq.read_table(path).to_pylist())
    return records, [str(p.relative_to(local)) for p in files]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    text_sets, bare_sets, qualified_sets = load_protected()
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for name, spec in SOURCES.items():
        records, files = read_train(spec["repo"], spec["patterns"])
        kept = Counter()
        reasons: Counter[str] = Counter()
        leaks: Counter[str] = Counter()
        for rec in records:
            row, reason = spec["convert"](rec, "train")
            reasons[reason] += 1
            if row is None:
                continue
            key = row_text_key(row)
            bare = str(row["source_record_id"])
            hit = [f"text:{n}" for n, keys in text_sets.items() if key in keys]
            hit += [f"record_bare:{n}" for n, ids in bare_sets.items() if bare in ids]
            hit += [f"record_qualified:{n}" for n, ids in qualified_sets.items() if f"{row['source']}/{bare}" in ids]
            if hit:
                for h in hit:
                    leaks[h] += 1
                reasons["dropped_protected_overlap"] += 1
                continue
            kept["train"] += 1
            rows.append(row)
        licence = CANDIDATES[name]
        report[name] = {
            "repo": spec["repo"], "files": files, "licence": licence[1], "licence_source": licence[2],
            "records_read": len(records), "kept": kept["train"], "reasons": dict(reasons),
            "leakage_hits": dict(leaks), "split": "train only",
        }
        print(f"{name}: read {len(records)}, kept {kept['train']}, reasons {dict(reasons)}", flush=True)
    dropped_candidates = {k: {"decision": v[0], "licence": v[1], "why": v[2]} for k, v in CANDIDATES.items()
                          if v[0] != "kept"}
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DATA, rows)
    replay_total = MIX_V2_REPLAY_ROWS + len(rows)
    total = MIX_V2_TRAIN_ROWS + len(rows)
    share_now = replay_total / total
    needed = round(ADVISORY_REPLAY_SHARE * (MIX_V2_TRAIN_ROWS - MIX_V2_REPLAY_ROWS) / (1 - ADVISORY_REPLAY_SHARE))
    manifest = {
        "task": "V3",
        "written_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "rule": "permissive licences only (D20(b)); train splits only; leakage check against the osler_v0 protected sets",
        "sources": report,
        "not_converted": dropped_candidates,
        "replay_rows_added": len(rows),
        "mix_v2_train_rows": MIX_V2_TRAIN_ROWS,
        "mix_v2_replay_rows": MIX_V2_REPLAY_ROWS,
        "b0_train_rows": total,
        "b0_replay_rows": replay_total,
        "b0_replay_share": round(share_now, 4),
        "advisory_target_share": ADVISORY_REPLAY_SHARE,
        "replay_rows_needed_for_target": needed,
        "replay_rows_short_of_target": max(0, needed - replay_total),
        "target_met": share_now >= ADVISORY_REPLAY_SHARE,
        "rows_file": str(OUT_DATA.relative_to(REPO)),
        "rows_sha256": file_sha256(OUT_DATA),
    }
    write_json(OUT_MANIFEST, manifest)
    print(json.dumps({k: manifest[k] for k in ("replay_rows_added", "b0_train_rows", "b0_replay_rows",
                                                "b0_replay_share", "replay_rows_short_of_target", "target_met")}),
          flush=True)


if __name__ == "__main__":
    main()
