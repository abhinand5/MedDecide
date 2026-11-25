"""Y3 self-audit (osler_v4 R6): the headline numbers of the mimic_doc_support_v1 build, re-derived from the raw attempts, the pair
rows and the admission table, in a fresh process. Counts only: no note text, identifiers or values are printed or committed.

Checks: attempts per stage and outcome sums; kept attempts whose admission lies in the stage's split (test: late-anchor); pair
rows per split, one original (gold yes) and one edit (gold no) per pair; every edit is a pure sentence deletion of its original
(meddecide.mr.mimic_pairs.check_pure_deletion); no pair selects more than 30 % of its sentences (ADVISORY Y3 §3). Writes
outputs/osler_v4/Y3/audit.json.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from meddecide.mr.mimic_pairs import check_pure_deletion
from meddecide.utils.io import iter_jsonl, write_json

BUILD = Path("outputs/osler_v4/Y3/build_off")
STAGES = ("dev", "test", "train")
ROW_FILES = tuple(Path(f"data/mimic_build/pairs/build_off/rows_{stage}.jsonl") for stage in STAGES)
ADMISSIONS = Path("data/mimic_build/admissions.parquet")
OUT = Path("outputs/osler_v4/Y3/audit.json")
MAX_SHARE = 0.30  # ADVISORY Y3 §3


def attempts_report() -> dict[str, Any]:
    total: Counter[str] = Counter()
    outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    for attempt in iter_jsonl(BUILD / "attempts.jsonl"):
        total[attempt["stage"]] += 1
        outcomes[attempt["stage"]][attempt["outcome"]] += 1
    report: dict[str, Any] = {}
    for stage in STAGES:
        n = total[stage]
        kept = outcomes[stage]["kept"]
        report[stage] = {"attempts": n, "kept": kept, "keep_rate": round(kept / n, 4) if n else None,
                         "outcome_sum_equals_attempts": sum(outcomes[stage].values()) == n}
    return report


def admission_splits() -> dict[int, tuple[str, bool]]:
    table = pq.read_table(ADMISSIONS, columns=["hadm_id", "split", "late_anchor"]).to_pylist()
    return {int(row["hadm_id"]): (str(row["split"]), bool(row["late_anchor"])) for row in table}


def kept_split_report(splits: dict[int, tuple[str, bool]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for stage in STAGES:
        matched = unmatched = late_violations = 0
        for attempt in iter_jsonl(BUILD / "attempts.jsonl"):
            if attempt["stage"] != stage or attempt["outcome"] != "kept":
                continue
            split, late = splits.get(int(attempt["hadm_id"]), (None, False))
            if split != stage:
                unmatched += 1
                continue
            if stage == "test" and not late:
                late_violations += 1
            matched += 1
        report[stage] = {"kept_in_stage_split": matched, "kept_outside_stage_split": unmatched,
                         "test_not_late_anchor": late_violations}
    return report


def pair_report() -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in ROW_FILES:
        for row in iter_jsonl(path):
            groups[str(row["meta"]["pair_id"])].append(row)
    by_split: Counter[str] = Counter()
    malformed = impure = over_share = 0
    for rows in groups.values():
        roles = sorted(str(r["meta"]["role"]) for r in rows)
        if roles != ["edit", "original"] or len({str(r["split"]) for r in rows}) != 1:
            malformed += 1
            continue
        original = next(r for r in rows if r["meta"]["role"] == "original")
        edit = next(r for r in rows if r["meta"]["role"] == "edit")
        if original["gold"] != "yes" or edit["gold"] != "no":
            malformed += 1
            continue
        by_split[str(original["split"])] += 1
        if not check_pure_deletion(str(original["state"]), str(edit["state"])):
            impure += 1
        if float(original["meta"]["share_selected"]) > MAX_SHARE:
            over_share += 1
    return {"pairs_by_split": dict(sorted(by_split.items())), "rows": sum(len(rows) for rows in groups.values()),
            "malformed_pairs": malformed, "impure_edits": impure, "pairs_over_30_percent": over_share}


def main() -> None:
    splits = admission_splits()
    report = {"attempts": attempts_report(), "kept_split": kept_split_report(splits), "pairs": pair_report()}
    write_json(OUT, report)
    print(report)


if __name__ == "__main__":
    main()
