"""Y4 self-audit (R6): re-derives the baseline headline numbers from the raw predictions in a fresh process.

For each run (default: every run under outputs/osler_v4/Y4/baselines) it checks the scored row count and the template
denominators; that the correct flag agrees with the key match; the micro accuracy and chance level against
baselines_report.json; that every probability vector is finite, non-negative and sums to one. For the trainer-based osler_v2
rows it also checks that the letter keys are positions in the item's own option list (the gold and predicted identities, and
the macro, majority and modal figures under the item's own keys). It prints counts and differences only, never item text.

Usage, from the repository root, after scripts/osler_v4/y4_baselines_report.py:
    uv run python scripts/osler_v4/y4_rederive.py [run ...]
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.eval.metrics import macro_accuracy
from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.io import iter_jsonl

BASELINES = Path("outputs/osler_v4/Y4/baselines")
REPORT = Path("outputs/osler_v4/Y4/baselines_report.json")
ITEMS = {
    "suite": Path("data/bench/osler_mimic_v1/mimic_audit_suite_v1.jsonl"),
    "heldout": Path("data/bench/osler_mimic_v1/mimic_heldout_v1.jsonl"),
    "fastdev_mimic": Path("data/bench/osler_v4/fastdev_v4_mimic.jsonl"),
    "cg_v2": Path("data/bench/osler_v4/cg_v2.jsonl"),
}
def position_key(index: int) -> str:
    """The trainer's positional key: the index in base 26 with the digits A (0) to Z (25), so 0 is A, 25 is Z, 26 is BA."""
    digits = ""
    while True:
        index, remainder = divmod(index, 26)
        digits = chr(ord("A") + remainder) + digits
        if index == 0:
            return digits


def scored_rows(run: str) -> list[dict[str, Any]]:
    return [r for r in iter_jsonl(BASELINES / run / "predictions.jsonl") if r.get("status", "scored") == "scored"]


def set_of(run: str) -> str:
    return max((s for s in ITEMS if run.startswith(s + "_")), key=len)


def item_options(set_name: str) -> dict[str, tuple[list[str], str]]:
    """Each item's option keys (in the item file's order) and gold key, keyed by the trainer's item id."""
    out: dict[str, tuple[list[str], str]] = {}
    for row in iter_jsonl(ITEMS[set_name]):
        out[mix_row_to_item(row).item_id] = ([str(o["key"]) for o in row["options"]], str(row["gold"]))
    return out


def check_run(run: str, report: dict[str, Any]) -> None:
    rows = scored_rows(run)
    n = len(rows)
    if n == 0:  # a run that is still writing its file has no complete rows yet
        print(f"{run}: no scored rows yet (not checked)")
        return
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        per[str(r["template_id"])][0] += bool(r["correct"])
        per[str(r["template_id"])][1] += 1
    flag_mismatch = sum((r["predicted_key"] == r["gold_key"]) != bool(r["correct"]) for r in rows)
    micro = sum(bool(r["correct"]) for r in rows) / n
    chance = sum(1 / len(r["probs"]) for r in rows) / n
    bad_sum = sum(abs(sum(r["probs"]) - 1.0) > 1e-3 for r in rows)
    not_finite = sum(not all(math.isfinite(x) and x >= -1e-9 for x in r["probs"]) for r in rows)
    entry = report["sets"][run]
    overall = entry["overall"]
    worst = max(abs(c[0] / c[1] - entry["templates"][t]["micro_accuracy"]) for t, c in per.items())
    print(f"{run}: rows {n}; templates {len(per)}; template items sum {sum(c[1] for c in per.values())} "
          f"(equals rows: {sum(c[1] for c in per.values()) == n}); correct-flag vs key mismatches {flag_mismatch}; "
          f"micro {micro:.4f} (report {overall['micro_accuracy']}, diff {abs(round(micro, 4) - overall['micro_accuracy']):.4f}); "
          f"chance {chance:.4f} (report {overall['chance']}); worst template diff {worst:.4f}; "
          f"probability vectors off sum-to-one {bad_sum}, not finite or negative {not_finite}")
    if "osler" in run:
        check_osler_keys(run, rows)


def check_osler_keys(run: str, rows: list[dict[str, Any]]) -> None:
    """The trainer's letter keys are positions: letter i is the item's option i (checked against the item file and the flash run)."""
    set_name = set_of(run)
    options = item_options(set_name)
    flash = {r["item_id"]: r["gold_key"] for r in scored_rows(f"{set_name}_flash")} if (BASELINES / f"{set_name}_flash").exists() else {}
    counts: Counter[str] = Counter()
    by_template: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        keys, gold = options[r["item_id"]]
        counts["gold key is the position of the gold index"] += r["gold_key"] == position_key(r["gold_index"])
        counts["predicted key is the position of the predicted index"] += r["predicted_key"] == position_key(r["predicted_index"])
        counts["n_options equals the item file's option count"] += r["n_options"] == len(keys)
        counts["item-file gold equals the option at the gold index"] += gold == keys[r["gold_index"]]
        if r["item_id"] in flash:  # the flash run skips the repeated-label items (C086); the trainer's runs keep them
            counts["items the flash run also scored"] += 1
            counts["flash gold key equals the option at the gold index"] += flash[r["item_id"]] == keys[r["gold_index"]]
        t = by_template[str(r["template_id"])]
        t["letter_gold"].append(r["gold_key"])
        t["letter_pred"].append(r["predicted_key"])
        t["item_gold"].append(keys[r["gold_index"]])
        t["item_pred"].append(keys[r["predicted_index"]])
    agree = 0
    for t in by_template.values():
        same_macro = macro_accuracy(t["letter_pred"], t["letter_gold"]) == macro_accuracy(t["item_pred"], t["item_gold"])
        same_major = Counter(t["letter_gold"]).most_common(1)[0][1] == Counter(t["item_gold"]).most_common(1)[0][1]
        same_modal = Counter(t["letter_pred"]).most_common(1)[0][1] == Counter(t["item_pred"]).most_common(1)[0][1]
        agree += same_macro and same_major and same_modal
    print(f"  osler key check, {run}: items {len(rows)}; " + "; ".join(f"{k} {v}" for k, v in sorted(counts.items()))
          + f"; templates where macro, majority and modal figures agree under both keyings {agree} of {len(by_template)}")


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    runs = sys.argv[1:] or sorted(p.parent.name for p in BASELINES.glob("*/manifest.json"))
    for run in runs:
        if not (BASELINES / run / "manifest.json").exists():  # every runner writes its manifest last, when the run is complete
            print(f"{run}: no manifest yet (the run is still writing; not checked)")
            continue
        check_run(run, report)


if __name__ == "__main__":
    main()
