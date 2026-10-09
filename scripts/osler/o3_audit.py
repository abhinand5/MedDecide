"""O3 audit: recompute the generator headline numbers from the built JSONL in a fresh process (R6).

Recomputes split sizes, held-out counts, gold distributions, gold-in-state hits, the shortcut baselines (string
presence and Naive Bayes, trained with the same rule as the screen), twin statistics and the majority baseline, then
compares them with outputs/osler_v0/O3/screen.json. Mismatches are listed and make the exit status non-zero; nothing
is repaired. Writes outputs/osler_v0/O3/recompute.json (gitignored).

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o3_audit.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.gen.families import FAMILIES
from meddecide.gen.screen import (
    NaiveBayes,
    gold_in_state,
    gold_label,
    macro_accuracy,
    string_presence_predict,
)
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/gen/osler_v0"
SCREEN = REPO / "outputs/osler_v0/O3/screen.json"
OUT = REPO / "outputs/osler_v0/O3/recompute.json"


def _load(split: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]


def _by_generator(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[r["generator"]].append(r)
    return out


def _twin_share(dev: list[dict[str, Any]], preds: list[str]) -> tuple[int, float | None]:
    by_pair: dict[str, list[bool]] = defaultdict(list)
    for r, p in zip(dev, preds, strict=True):
        by_pair[r["pair_id"]].append(gold_label(r) == p)
    full = [v for v in by_pair.values() if len(v) == 2]
    return len(full), (sum(all(v) for v in full) / len(full)) if full else None


def _majority(rows: list[dict[str, Any]]) -> str:
    return Counter(gold_label(r) for r in rows).most_common(1)[0][0] if rows else ""


def main() -> None:
    splits = {s: _load(s) for s in ("train", "dev", "test")}
    by_gen = {s: _by_generator(rows) for s, rows in splits.items()}
    seen_train = {f.name: by_gen["train"][f.name] for f in FAMILIES if not f.held_out}
    screen = {r["generator"]: r for r in json.loads(SCREEN.read_text(encoding="utf-8"))["reports"]}
    mismatches: list[str] = []
    gens: dict[str, Any] = {}
    for fam in FAMILIES:
        dev, test, own_train = by_gen["dev"][fam.name], by_gen["test"][fam.name], by_gen["train"][fam.name]
        if fam.held_out:
            letter = dev[0]["family"][:1]
            nb_train = [r for name, rs in seen_train.items() if rs and rs[0]["family"][:1] == letter for r in rs]
            nb_source = f"seen generators whose family starts with {letter!r} (train split)"
        else:
            nb_train, nb_source = own_train, "own train split"
        majority = _majority(nb_train)
        phrase = [r for r in dev if r["qtype"] != "noul"]
        sp_pred = [string_presence_predict(r, majority) for r in phrase]
        sp = macro_accuracy([(gold_label(r), p) for r, p in zip(phrase, sp_pred, strict=True)])
        nb = NaiveBayes().fit(nb_train)
        nb_pred = [nb.predict(r) for r in dev]
        nbm = macro_accuracy([(gold_label(r), p) for r, p in zip(dev, nb_pred, strict=True)])
        twin_pairs, both = _twin_share(dev, nb_pred)
        own_majority = _majority(own_train)
        majority_test = macro_accuracy([(gold_label(r), own_majority) for r in test]) if own_train else None
        s = screen[fam.name]
        recomputed = {
            "gold_in_state_hits": sum(gold_in_state(r) for r in dev),
            "twin_pairs": twin_pairs,
            "string_presence_macro": None if sp is None else round(sp, 4),
            "naive_bayes_macro": None if nbm is None else round(nbm, 4),
            "twin_both_correct_share": None if both is None else round(both, 4),
        }
        for key, mine in recomputed.items():
            if mine != s[key]:
                mismatches.append(f"{fam.name}.{key}: recomputed {mine} vs screen.json {s[key]}")
        gens[fam.name] = {
            "held_out": fam.held_out,
            "family": dev[0]["family"],
            "items": {"train": len(own_train), "dev": len(dev), "test": len(test)},
            "patients": {s_: len({r["patient_id"] for r in by_gen[s_][fam.name]}) for s_ in ("train", "dev", "test")},
            "nb_train_rows": len(nb_train),
            "nb_train_source": nb_source,
            "dev_gold_counts": dict(sorted(Counter(gold_label(r) for r in dev).items())),
            "test_gold_counts": dict(sorted(Counter(gold_label(r) for r in test).items())),
            "test_classes": len({gold_label(r) for r in test}),
            "gold_in_state_hits_dev": recomputed["gold_in_state_hits"],
            "gold_in_state_hits_test (additional)": sum(gold_in_state(r) for r in test),
            "string_presence_macro": recomputed["string_presence_macro"],
            "naive_bayes_macro": recomputed["naive_bayes_macro"],
            "distinct_naive_bayes_predictions_dev": len(set(nb_pred)),
            "distinct_string_presence_predictions_dev": len(set(sp_pred)),
            "twin_pairs": twin_pairs,
            "twin_both_correct_share": recomputed["twin_both_correct_share"],
            "own_majority_macro_test (additional)": None if majority_test is None else round(majority_test, 4),
        }
    report = {
        "computed_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "splits": {s: {"rows": len(rows), "patients": len({r["patient_id"] for r in rows}),
                       "held_out_rows": sum(1 for r in rows if r["held_out"])} for s, rows in splits.items()},
        "held_out_rows_in_train": sum(1 for r in splits["train"] if r["held_out"]),
        "generators": gens,
        "mismatches_vs_screen_json": mismatches,
    }
    write_json(OUT, report)
    for name, g in gens.items():
        print(f"{name:26s} held_out={g['held_out']!s:5s} dev={g['items']['dev']} test={g['items']['test']} "
              f"sp={g['string_presence_macro']} nb={g['naive_bayes_macro']} both={g['twin_both_correct_share']} "
              f"nb_rows={g['nb_train_rows']} nb_distinct={g['distinct_naive_bayes_predictions_dev']}")
    print(f"splits {report['splits']}; held_out_rows_in_train={report['held_out_rows_in_train']}")
    print(f"mismatches vs screen.json: {len(mismatches)}")
    for m in mismatches:
        print("  " + m)
    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
