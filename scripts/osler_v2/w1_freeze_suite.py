"""W1 step 4 (osler_v2 ADVISORY W1, D35): freeze the MR suite v1 from the screens and the ceiling statistic.

D35 excludes from the verdict (a) any template whose zero-shot Qwen3.5-4B accuracy on its dev items is >= 0.97 (ceiling) and
(b) any template whose Naive Bayes screen on dev is >= 0.90 (vocabulary shortcut). Both are reported, never in the verdict.
A template whose dev items do not exist cannot trigger either rule: it stays in the verdict, its screen is NOT MEASURED, and
its status is written out (no silent drop).

Reads outputs/osler_v2/W1/screens.json (from w1_screens.py). Writes:
- data/bench/osler_mr_v1/manifest.json (gitignored; template lists, counts, reasons);
- loops/osler_v2/mr_suite_v1.md (committed; aggregates only, no item text).

Usage: uv run --frozen python scripts/osler_v2/w1_freeze_suite.py
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCREENS = REPO / "outputs" / "osler_v2" / "W1" / "screens.json"
EVAL = REPO / "data" / "bench" / "osler_mr_v0" / "mr_eval.jsonl"
MANIFEST = REPO / "data" / "bench" / "osler_mr_v1" / "manifest.json"
MD = REPO / "loops" / "osler_v2" / "mr_suite_v1.md"
CEILING = 0.97
SCREEN = 0.90


def eval_counts() -> dict[tuple[str, str], int]:
    counts: Counter[tuple[str, str]] = Counter()
    with EVAL.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            counts[(row["set_name"], row["template_id"])] += 1
    return dict(counts)


def decide(name: str, entry: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    notes: list[str] = []
    ceiling = entry["ceiling"]
    screen = entry["screen"]
    if ceiling.get("accuracy") is not None and ceiling["accuracy"] >= CEILING:
        reasons.append(f"ceiling: zero-shot dev accuracy {ceiling['accuracy']:.4f} >= {CEILING}")
    if screen.get("status") == "MEASURED" and screen["naive_bayes_macro"] is not None \
            and screen["naive_bayes_macro"] >= SCREEN:
        reasons.append(f"vocabulary shortcut: Naive Bayes dev macro {screen['naive_bayes_macro']:.4f} >= {SCREEN}")
    if screen.get("status") != "MEASURED":
        notes.append("dev screens NOT MEASURED: " + screen["reason"])
        notes.append("D35 rules cannot trigger without dev items; the template stays in the verdict (D35's own exclusions "
                     "are the only drops). Operator question recorded in STATE section 6.")
    verdict = not reasons
    return {"template": name, "verdict": verdict, "reasons": reasons, "notes": notes}


def main() -> None:
    screens = json.loads(SCREENS.read_text(encoding="utf-8"))["templates"]
    counts = eval_counts()
    decisions = [decide(name, entry) for name, entry in screens.items()]
    for d in decisions:
        d["test_items"] = counts.get(("mr_test", d["template"]), 0)
        d["heldout_items"] = counts.get(("mr_heldout", d["template"]), 0)
    verdict_items = sum(d["test_items"] for d in decisions if d["verdict"])
    reported_items = sum(d["test_items"] for d in decisions if not d["verdict"])
    total = sum(d["test_items"] for d in decisions)
    assert verdict_items + reported_items == total, "arithmetic must close"
    manifest = {
        "frozen_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rules": {"ceiling": CEILING, "screen": SCREEN, "source": "D35 (PROGRAM.md)"},
        "verdict_templates": [d["template"] for d in decisions if d["verdict"]],
        "reported_only_templates": [d["template"] for d in decisions if not d["verdict"]],
        "decisions": decisions,
        "test_items": {"total": total, "verdict": verdict_items, "reported_only": reported_items},
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = ["# MR suite v1 (osler_v2, D35) — frozen at W1", "",
             "Aggregates only. Recompute: `uv run --frozen python scripts/osler_v2/w1_screens.py` then "
             "`uv run --frozen python scripts/osler_v2/w1_freeze_suite.py`. The rules are fixed (D35): zero-shot Qwen3.5-4B "
             "dev accuracy ≥ 0.97 (ceiling) and Naive Bayes dev macro ≥ 0.90 (vocabulary shortcut) exclude a template from the "
             "verdict; both are reported. Nothing here was chosen after an osler_v2 model result: none exists yet.", "",
             "| template | dev items | ceiling (zero-shot acc) | NB dev macro | string presence | verdict | test items | held-out items |",
             "|---|---|---|---|---|---|---|---|"]
    for d in decisions:
        entry = screens[d["template"]]
        ceil = entry["ceiling"].get("accuracy")
        ceil_s = "NOT MEASURED" if ceil is None else f"{ceil:.4f} (n={entry['ceiling']['n']})"
        nb = entry["screen"].get("naive_bayes_macro")
        nb_s = "NOT MEASURED" if nb is None else f"{nb:.4f}"
        sp = entry["screen"].get("string_presence_macro")
        sp_s = "—" if sp is None else f"{sp:.4f}"
        status = "IN VERDICT" if d["verdict"] else "REPORTED ONLY"
        lines.append(f"| `{d['template']}` | {entry['n_dev']} | {ceil_s} | {nb_s} | {sp_s} | {status} | {d['test_items']} "
                     f"| {d['heldout_items']} |")
    lines += ["", "## Reasons", ""]
    for d in decisions:
        for r in d["reasons"] + d["notes"]:
            lines.append(f"- `{d['template']}`: {r}")
    lines += ["", f"Test items: total {total}; in the verdict {verdict_items}; reported only {reported_items} "
              f"(sum checked in code).", ""]
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"verdict items {verdict_items}, reported only {reported_items}, total {total}")
    print("wrote", MANIFEST, MD)


if __name__ == "__main__":
    main()
