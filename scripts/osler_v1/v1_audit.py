"""Self-audit of the V1 generator output, re-derived from the JSONL files in a fresh process (osler_v1 ADVISORY V1; R6).

Reads data/gen/osler_v1/{train,dev,test,heldout_test}.jsonl and checks, per generator:
- counts per split and the label balance (gold key counts, and the gold share of "none" for the ICD family);
- twins: every pair_id has exactly one base and one twin, and the two golds differ (the twin flips the label);
- patient-level splits: every row's split equals split_of(patient_id); no patient in two splits;
- held-out rows: none in train, dev or test; every row of heldout_test is held out (indicator or family), and the held-out
  rows never appear in the training splits.

Writes outputs/osler_v1/V1/self_audit.json. Aggregates only; nothing from an item is printed or committed.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v1_audit.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.gen.core import split_of
from meddecide.mr.generators import HELD_OUT_ICD_FAMILIES, HELD_OUT_INDICATORS, is_held_out
from meddecide.utils.io import read_json, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "gen" / "osler_v1"
OUT = REPO / "outputs" / "osler_v1" / "V1" / "self_audit.json"
SPLITS = ("train", "dev", "test", "heldout_test")


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def audit() -> dict[str, Any]:
    per_gen: dict[str, dict[str, Any]] = {}
    patients: dict[str, set[str]] = defaultdict(set)
    split_mismatch = 0
    held_in_training = 0
    heldout_not_held = 0
    for split in SPLITS:
        for row in iter_rows(DATA / f"{split}.jsonl"):
            gen = row["generator"]
            g = per_gen.setdefault(gen, {
                "rows": Counter(), "gold": defaultdict(Counter), "pairs": defaultdict(dict),
                "held_out_rows": Counter(), "trap_or_family": defaultdict(Counter),
                "dev_chance_sum": 0.0, "dev_n": 0,
            })
            g["rows"][split] += 1
            g["gold"][split][row["gold"]] += 1
            g["pairs"][row["pair_id"]].setdefault(row["role"], row["gold"])
            patients[split].add(row["patient_id"])
            # a row's split is its patient's hash split; the training-side files carry exactly their own split
            file_split_ok = split == "heldout_test" or row["split"] == split
            if split_of(row["patient_id"]) != row["split"] or not file_split_ok:
                split_mismatch += 1
            if split == "heldout_test":
                if not (row.get("held_out", False) or is_held_out(row["meta"])):
                    heldout_not_held += 1
                g["held_out_rows"]["heldout_test"] += 1
            elif row.get("held_out", False) or is_held_out(row["meta"]):
                held_in_training += 1
            meta = row["meta"]
            key = meta.get("indicator") or meta.get("family") or ""
            g["trap_or_family"][split][key] += 1
            if split == "dev":
                g["dev_chance_sum"] += 1.0 / len(row["options"])
                g["dev_n"] += 1
    all_patients = set().union(*patients.values())
    shared = sum(len(patients[a] & patients[b]) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:])

    report: dict[str, Any] = {}
    for gen, g in sorted(per_gen.items()):
        pairs = g["pairs"]
        complete = sum(1 for roles in pairs.values() if set(roles) == {"base", "twin"})
        flipped = sum(1 for roles in pairs.values() if len(set(roles.values())) == 2)
        report[gen] = {
            "rows": {s: g["rows"][s] for s in SPLITS},
            "gold_by_split": {s: dict(sorted(g["gold"][s].items())) for s in SPLITS if g["gold"][s]},
            "pairs": len(pairs),
            "pairs_with_base_and_twin": complete,
            "pairs_whose_gold_flips": flipped,
            "held_out_rows": dict(g["held_out_rows"]),
            "items_by_indicator_or_family": {s: dict(sorted(c.items())) for s, c in g["trap_or_family"].items()},
            "dev_uniform_chance": round(g["dev_chance_sum"] / g["dev_n"], 4) if g["dev_n"] else None,
            "dev_majority_gold_share": round(max(g["gold"]["dev"].values()) / sum(g["gold"]["dev"].values()), 4)
            if g["gold"]["dev"] else None,
        }
    return {
        "written_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "held_out_indicators": sorted(HELD_OUT_INDICATORS),
        "held_out_icd_families": sorted(HELD_OUT_ICD_FAMILIES),
        "generators": report,
        "patients_per_split": {s: len(patients[s]) for s in SPLITS},
        "patients_total": len(all_patients),
        "patients_in_two_splits": shared,
        "rows_whose_split_disagrees_with_patient_hash": split_mismatch,
        "held_out_rows_in_training_splits": held_in_training,
        "heldout_test_rows_not_held_out": heldout_not_held,
    }


SCREEN = REPO / "outputs" / "osler_v1" / "V1" / "screen.json"
SELF_AUDIT_MD = REPO / "outputs" / "osler_v1" / "V1" / "SELF_AUDIT.md"


def self_audit_markdown(result: dict[str, Any]) -> str:
    screen = {r["generator"]: r for r in read_json(SCREEN)["reports"]}
    checks: list[tuple[str, bool, str]] = [
        ("no patient in two splits", result["patients_in_two_splits"] == 0, str(result["patients_in_two_splits"])),
        ("every row's split equals its patient's hash split",
         result["rows_whose_split_disagrees_with_patient_hash"] == 0,
         str(result["rows_whose_split_disagrees_with_patient_hash"])),
        ("no held-out row in train, dev or test", result["held_out_rows_in_training_splits"] == 0,
         str(result["held_out_rows_in_training_splits"])),
        ("every heldout_test row is held out", result["heldout_test_rows_not_held_out"] == 0,
         str(result["heldout_test_rows_not_held_out"])),
    ]
    for gen, r in result["generators"].items():
        if r["pairs_with_base_and_twin"] or r["pairs_whose_gold_flips"]:
            checks.append((f"{gen}: every twin pair complete and its gold flips",
                           r["pairs_with_base_and_twin"] == r["pairs"] == r["pairs_whose_gold_flips"],
                           f"{r['pairs_with_base_and_twin']}/{r['pairs']} complete, {r['pairs_whose_gold_flips']} flip"))
    for gen, rep in screen.items():
        checks.append((f"{gen}: O3 screen (ADVISORY §3.6)", bool(rep["passes"]),
                       f"string presence {rep['string_presence_macro']}, Naive Bayes {rep['naive_bayes_macro']}; "
                       + ("; ".join(rep["failures"]) if rep["failures"] else "no failures")))
    lines = ["# V1 self-audit (osler_v1 ADVISORY V1; R6)", "",
             "Re-derived from the written files in a fresh process by `scripts/osler_v1/v1_audit.py`.", "",
             "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |" for name, ok, detail in checks]
    lines += ["", "## Expectations (chance and majority on dev, from the same files)", "",
              "| generator | uniform chance | majority gold share | Naive Bayes (screen) |", "|---|---|---|---|"]
    for gen, r in result["generators"].items():
        nb = screen.get(gen, {}).get("naive_bayes_macro")
        lines.append(f"| {gen} | {r['dev_uniform_chance']} | {r['dev_majority_gold_share']} | {nb} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    result = audit()
    write_json(OUT, result)
    SELF_AUDIT_MD.write_text(self_audit_markdown(result), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "generators"}, indent=1))
    for gen, r in result["generators"].items():
        print(gen, {s: r["rows"][s] for s in SPLITS}, "pairs", r["pairs"],
              "complete", r["pairs_with_base_and_twin"], "flipped", r["pairs_whose_gold_flips"])


if __name__ == "__main__":
    main()
