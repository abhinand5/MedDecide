"""Self-audit of V3 (osler_v1 ADVISORY V3; R6): the replay rows, re-read from the file in a fresh process, checked against the
MR test sets (V1 dev, test and heldout_test) as well as the osler_v0 protected sets, which the build already used.

Writes outputs/osler_v1/V3/SELF_AUDIT.md and prints the checks. Exits non-zero if a check fails.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v3_self_audit.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from meddecide.mix.leakage import row_text_key
from meddecide.utils.io import iter_jsonl, read_json

REPO = Path(__file__).resolve().parents[2]
REPLAY = REPO / "data" / "train" / "osler_v1" / "replay_v1_train.jsonl"
MR_DIR = REPO / "data" / "gen" / "osler_v1"
MANIFEST = REPO / "outputs" / "osler_v1" / "V3" / "replay_manifest.json"
OUT = REPO / "outputs" / "osler_v1" / "V3" / "SELF_AUDIT.md"


def main() -> None:
    manifest = read_json(MANIFEST)
    replay_rows = list(iter_jsonl(REPLAY))
    checks: list[tuple[str, bool, str]] = []

    counts = Counter(r["source"] for r in replay_rows)
    checks.append(("replay row count equals the manifest's added rows",
                   len(replay_rows) == manifest["replay_rows_added"],
                   f"{len(replay_rows)} rows; sources {dict(counts)}"))
    checks.append(("every replay row is a train-split row", all(r["split"] == "train" for r in replay_rows),
                   "split values read from the file"))
    checks.append(("every replay row has two or more options and a gold key among them",
                   all(len(r["options"]) >= 2 and r["gold"] in {o["key"] for o in r["options"]} for r in replay_rows),
                   "option counts and gold membership"))

    mr_keys: dict[str, set[str]] = {}
    for split in ("dev", "test", "heldout_test"):
        mr_keys[split] = {row_text_key(r) for r in iter_jsonl(MR_DIR / f"{split}.jsonl")}
    hits: Counter[str] = Counter()
    for r in replay_rows:
        key = row_text_key(r)
        for split, keys in mr_keys.items():
            if key in keys:
                hits[f"mr_{split}"] += 1
    checks.append(("no replay row's text key matches an MR dev, test or heldout_test item",
                   sum(hits.values()) == 0, f"hits {dict(hits) or 'none'}"))

    stored_leaks = {k: v for s in manifest["sources"].values() for k, v in s["leakage_hits"].items()}
    checks.append(("the build's osler_v0 protected-set check recorded no hits",
                   not stored_leaks, f"hits {stored_leaks or 'none'}"))

    share = (manifest["mix_v2_replay_rows"] + len(replay_rows)) / (manifest["mix_v2_train_rows"] + len(replay_rows))
    checks.append(("the replay share recomputed from the counts equals the manifest's share",
                   abs(share - manifest["b0_replay_share"]) < 1e-4, f"recomputed {share:.4f}"))
    checks.append(("the 20 % target is not met (recorded, not forced)", not manifest["target_met"],
                   f"B0 replay share {manifest['b0_replay_share']:.4f}; {manifest['replay_rows_short_of_target']} rows short"))

    lines = ["# V3 self-audit (osler_v1 ADVISORY V3; R6)", "",
             "Re-derived from the written replay file in a fresh process.", "",
             "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |" for name, ok, detail in checks]
    lines += ["", "Suspicious: nothing beyond the replay share, which is below the ADVISORY target (Question 2 in STATE §6)."]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, ok, detail in checks:
        print(("PASS " if ok else "FAIL ") + name + " | " + detail)
    if not all(ok for _, ok, _ in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
