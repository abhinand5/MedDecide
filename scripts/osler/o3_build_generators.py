"""O3: build the clinical decision generators' splits, screen them, and write the manifest (ADVISORY O3, D20).

Patients are generated per generator with ids "<generator>-<k>" and assigned to a split by stable hash of the id
(80/10/10 by patient, so twins share a split). Quotas: seen generators 1,500 train patients (3,000 items), 150 dev and
150 test patients; held-out generators get 0 train, 150 dev and 150 test patients (their dev is used only for the
screen, and their test is for evaluation only).

Outputs (gitignored): data/gen/osler_v0/{train,dev,test}.jsonl, data/gen/osler_v0/manifest.json (counts, hashes);
outputs/osler_v0/O3/screen.json (screen report per generator).

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o3_build_generators.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.gen.core import split_of
from meddecide.gen.families import FAMILIES
from meddecide.gen.screen import screen_generator
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/gen/osler_v0"
SCREEN = REPO / "outputs/osler_v0/O3/screen.json"
QUOTA = {"train": 1500, "dev": 150, "test": 150}


def build_family(family) -> dict[str, list[dict[str, Any]]]:
    quota = dict(QUOTA)
    if family.held_out:
        quota["train"] = 0
    got: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    remaining = {s: quota[s] for s in quota}
    k = 0
    while any(v > 0 for v in remaining.values()):
        pid = f"{family.name}-{k:06d}"
        k += 1
        split = split_of(pid)
        if remaining[split] <= 0:
            continue
        items = family.make(pid)
        remaining[split] -= 1
        for it in items:
            row = it.as_dict()
            row["held_out"] = family.held_out
            row["patient_id"] = pid
            got[split].append(row)
    return got


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    screen_reports = []
    counts: dict[str, Any] = {}
    seen_train: dict[str, list[dict[str, Any]]] = defaultdict(list)
    built: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for fam in FAMILIES:
        built[fam.name] = build_family(fam)
        if not fam.held_out:
            seen_train[fam.name] = built[fam.name]["train"]
    for fam in FAMILIES:
        splits = built[fam.name]
        for s in ("train", "dev", "test"):
            per_split[s].extend(splits[s])
        counts[fam.name] = {
            "held_out": fam.held_out,
            "family": splits["dev"][0]["family"] if splits["dev"] else None,
            "items": {s: len(splits[s]) for s in ("train", "dev", "test")},
            "patients": {s: len({r["patient_id"] for r in splits[s]}) for s in ("train", "dev", "test")},
            "gold_counts_test": dict(sorted(Counter(gold_label_of(r) for r in splits["test"]).items())),
        }
        if fam.held_out:
            # held-out: screen with a baseline trained only on seen generators of the same family letter
            family_letter = splits["dev"][0]["family"][:1]
            nb_train = [r for rows in seen_train.values() if rows and rows[0]["family"][:1] == family_letter
                        for r in rows]
            source = f"seen generators of family {family_letter} (train split)"
        else:
            nb_train = splits["train"]
            source = "own train split"
        rep = screen_generator(fam.name, fam.held_out, splits["dev"], nb_train, source)
        screen_reports.append(rep.as_dict())
        print(f"{fam.name:28s} held_out={fam.held_out!s:5s} dev={len(splits['dev'])} "
              f"sp={rep.string_presence_macro} nb={rep.naive_bayes_macro} gold_in_state={rep.gold_in_state_hits} "
              f"pass={rep.passes} {rep.failures}", flush=True)

    hashes = {}
    for s, rows in per_split.items():
        path = OUT / f"{s}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        hashes[s] = {"file": str(path.relative_to(REPO)), "rows": len(rows), "sha256": file_sha256(path)}
    SCREEN.parent.mkdir(parents=True, exist_ok=True)
    write_json(SCREEN, {"built_at_utc": utcnow(), "git_commit": git_commit(REPO), "reports": screen_reports})
    manifest = {
        "built_at_utc": utcnow(), "git_commit": git_commit(REPO),
        "quota_patients": QUOTA, "held_out": [f.name for f in FAMILIES if f.held_out],
        "generators": counts, "files": hashes,
        "screen_passed": all(r["passes"] for r in screen_reports),
        "screen_file": str(SCREEN.relative_to(REPO)),
    }
    write_json(OUT / "manifest.json", manifest)
    print(f"wrote {OUT} (train {hashes['train']['rows']}, dev {hashes['dev']['rows']}, test {hashes['test']['rows']}); "
          f"screen passed={manifest['screen_passed']}")


def gold_label_of(row: dict[str, Any]) -> str:
    return next(o["label"] for o in row["options"] if o["key"] == row["gold"])


if __name__ == "__main__":
    main()
