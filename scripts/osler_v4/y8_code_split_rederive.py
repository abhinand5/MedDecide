"""Y8 self-audit (R6) for the additional code-split rows: re-derives them from the CG v2 prediction files in a fresh process.

The check does not import the library's split or gate code. It reads the four prediction files, derives each gold index from the
run's own key (A-Z letters for the trainer; numbers from 1 and yes/no for the zero-shot runner), and recomputes the DDXPlus counts,
accuracies and two-letter masses and each template's count and accuracy. It also re-derives the constant-answer rule of the D12
gate (modal share against the health thresholds) for every cell, and compares everything with outputs/osler_v4/Y8/code_split.json.
Counts and figures only; exits 1 when any compared value differs.

Usage (CPU): CUDA_VISIBLE_DEVICES= uv run --frozen python scripts/osler_v4/y8_code_split_rederive.py [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.eval.health import DEFAULT_DEGENERATE_MAJORITY_CEILING, DEFAULT_MAX_MODAL_SHARE

REPO = Path(__file__).resolve().parents[2]
FILES = {
    "zero_shot_pplx_flash": REPO / "outputs/osler_v4/Y4/baselines/cg_v2_flash/predictions.jsonl",
    "osler_v2": REPO / "outputs/osler_v4/Y4/baselines/cg_v2_osler_v2/predictions.jsonl",
    "a0": REPO / "outputs/osler_v4/Y6/A0/cg_v2_predictions.jsonl",
    "y8_final": REPO / "outputs/osler_v4/Y8/cg_v2_predictions.jsonl",
}
DDX = "cg_v2_ddxplus_validate_v1"
TOLERANCE = 1e-9


def gold_index(key: str) -> int:
    """The run's own key as a 0-based index: yes is 0 and no is 1; a number n is n - 1; letters count from A = 0 in base 26."""
    if key == "yes":
        return 0
    if key == "no":
        return 1
    if key.isdigit():
        return int(key) - 1
    value = 0
    for ch in key:
        value = value * 26 + (ord(ch) - ord("A"))
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", type=Path, default=REPO / "outputs/osler_v4/Y8/code_split.json")
    args = parser.parse_args()
    published = json.loads(args.json.read_text(encoding="utf-8"))
    checks: list[tuple[str, Any, Any]] = []
    items: dict[str, dict[str, dict[str, Any]]] = {}
    for key, path in FILES.items():
        if not path.exists():
            continue
        items[key] = {}
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status", "scored") == "scored":
                items[key][row["item_id"]] = row

    for key in items:
        if key not in published["labels"]:
            continue
        rows = items[key].values()
        # DDXPlus: the split by gold index, two-letter from 26
        ddx = [r for r in rows if r["template_id"] == DDX]
        single = [r for r in ddx if gold_index(r["gold_key"]) < 26]
        two = [r for r in ddx if gold_index(r["gold_key"]) >= 26]
        pub = published["ddxplus"][key]["split"]
        checks += [
            (f"{key}: DDXPlus items", len(ddx), pub["items"]),
            (f"{key}: DDXPlus single-letter gold items", len(single), pub["single_gold"]["items"]),
            (f"{key}: DDXPlus single-letter gold correct", sum(bool(r["correct"]) for r in single), pub["single_gold"]["correct"]),
            (f"{key}: DDXPlus two-letter gold items", len(two), pub["two_letter_gold"]["items"]),
            (f"{key}: DDXPlus two-letter gold correct", sum(bool(r["correct"]) for r in two), pub["two_letter_gold"]["correct"]),
        ]
        for name, group in (("single", single), ("two", two)):
            pub_group = pub["single_gold" if name == "single" else "two_letter_gold"]
            mass = sum(sum(r["probs"][26:]) for r in group) / len(group) if group else None
            checks.append((f"{key}: DDXPlus {name}-gold mass mean", mass, pub_group["two_letter_mass_mean"]))
        overall = sum(sum(r["probs"][26:]) for r in ddx) / len(ddx)
        checks.append((f"{key}: DDXPlus two-letter mass mean, all items", overall, pub["two_letter_mass_mean"]))
        checks.append((f"{key}: DDXPlus probability vectors of 49 options", sum(len(r["probs"]) != 49 for r in ddx), 0))
        # per template: count and micro accuracy; the constant-answer rule of the gate for every cell
        by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_template[r["template_id"]].append(r)
        pub_rows = {t["template"]: t for t in published["templates"]}
        for template, cell in by_template.items():
            pub_cell = pub_rows[template]
            n = len(cell)
            checks.append((f"{key}: {template} items", n, pub_cell["items"][key]))
            checks.append((f"{key}: {template} accuracy", sum(bool(r["correct"]) for r in cell) / n, pub_cell["accuracy"][key]))
            predicted = Counter(r["predicted_key"] for r in cell)
            gold = Counter(r["gold_key"] for r in cell)
            modal_share = predicted.most_common(1)[0][1] / n
            majority_share = gold.most_common(1)[0][1] / n
            constant = modal_share >= DEFAULT_MAX_MODAL_SHARE and majority_share <= DEFAULT_DEGENERATE_MAJORITY_CEILING
            flagged = "constant answer" in pub_cell["status"][key]
            checks.append((f"{key}: {template} constant-answer flag agrees with the rule", constant, flagged))

    mismatches = []
    for name, recomputed, recorded in checks:
        if isinstance(recomputed, bool) or isinstance(recorded, bool) or recorded is None or recomputed is None:
            ok = recomputed == recorded
        else:
            ok = abs(float(recomputed) - float(recorded)) <= TOLERANCE
        if not ok:
            mismatches.append((name, recomputed, recorded))
    print(f"code split re-derivation: {len(checks)} values compared, {len(mismatches)} mismatches", flush=True)
    for name, recomputed, recorded in mismatches:
        print(f"  MISMATCH {name}: recomputed {recomputed} recorded {recorded}", flush=True)
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
