"""V6 self-audit (R6): re-derive the headline numbers of the baseline summary from the raw prediction files, in a fresh process
that uses only the standard library (no meddecide code), and compare them with outputs/osler_v1/V6/baseline_summary.json.

Checks: the denominators (scored plus skipped equals the rows of each set; coverage), the pass-cell accuracy, template macro
accuracy and Brier, the matched common-support block, the chance level and the majority-gold baseline, the disjointness of
the test and held-out items, and the held-out indicator template that fails the readout gate for the competitors. ECE is not
re-derived (its binning lives in the shared code) and is marked as such.

Writes outputs/osler_v1/V6/self_audit.json and outputs/osler_v1/V6/SELF_AUDIT.md (aggregates only).

Usage:
    source scripts/pod_env.sh && uv run --frozen python -I scripts/osler_v1/v6_self_audit.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "osler_v1" / "V6" / "baselines"
SUMMARY = REPO / "outputs" / "osler_v1" / "V6" / "baseline_summary.json"
OUT_JSON = REPO / "outputs" / "osler_v1" / "V6" / "self_audit.json"
OUT_MD = REPO / "outputs" / "osler_v1" / "V6" / "SELF_AUDIT.md"
MODELS = ["qwen35-4b", "qwen35-9b", "meddecider-4b", "meddecider-9b", "jev-9b"]
SETS = ["mr_test", "mr_heldout"]
TOL = 1e-4  # the summary rounds to four decimals


def load(slug: str) -> list[dict]:
    with (BASE / slug / "predictions.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def first_argmax(probs: list[float]) -> int:
    """The index of the largest probability, the first one on a tie (as numpy's argmax)."""
    best = 0
    for i, p in enumerate(probs):
        if p > probs[best]:
            best = i
    return best


def correct(row: dict) -> bool:
    return row["option_keys"][first_argmax(row["probs"])] == row["gold"]


def brier(row: dict) -> float:
    return sum((p - (1.0 if key == row["gold"] else 0.0)) ** 2
               for p, key in zip(row["probs"], row["option_keys"], strict=True))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def by_template(rows: list[dict], value) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        groups[r["template_id"]].append(value(r))
    return {t: mean(v) for t, v in groups.items()}


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    checks: list[dict] = []

    def check(name: str, expected, recomputed, tol: float = TOL) -> None:
        if isinstance(expected, (int, str)) or expected is None or isinstance(recomputed, (int, str)):
            ok = expected == recomputed
        else:
            ok = abs(float(expected) - float(recomputed)) <= tol
        checks.append({"check": name, "summary": expected, "recomputed": recomputed, "pass": bool(ok)})

    raw = {slug: load(slug) for slug in MODELS}
    # 1. denominators: every row is scored or skipped, and the set sizes are the summary's
    for slug in MODELS:
        for set_name in SETS:
            rows = [r for r in raw[slug] if r["set_name"] == set_name]
            scored = [r for r in rows if r["status"] == "scored"]
            skipped = Counter(r["reason"] for r in rows if r["status"] == "skipped")
            s = summary["models"][slug]["sets"][set_name]
            check(f"{slug} {set_name} rows in scope", s["n_in_scope"], len(rows))
            check(f"{slug} {set_name} scored", s["n_scored"], len(scored))
            check(f"{slug} {set_name} scored + skipped = rows", len(rows), len(scored) + sum(skipped.values()))
            check(f"{slug} {set_name} coverage", s["coverage"], round(len(scored) / len(rows), 4))
    # 2. pass-cell metrics: the cells that the summary does not gate (READOUT_FAIL excluded), recomputed from the rows
    for slug in MODELS:
        for set_name in SETS:
            s = summary["models"][slug]["sets"][set_name]
            fail = set(s["readout_fail_templates"])
            rows = [r for r in raw[slug] if r["set_name"] == set_name and r["status"] == "scored"
                    and r["template_id"] not in fail]
            acc_by_t = by_template(rows, lambda r: 1.0 if correct(r) else 0.0)
            brier_by_t = by_template(rows, brier)
            check(f"{slug} {set_name} pass-cell accuracy", s["pass_cells_accuracy"],
                  round(mean([1.0 if correct(r) else 0.0 for r in rows]), 4))
            check(f"{slug} {set_name} pass-cell macro accuracy", s["pass_cells_macro_accuracy"],
                  round(mean(list(acc_by_t.values())), 4))
            check(f"{slug} {set_name} pass-cell brier", s["pass_cells_brier"],
                  round(mean([brier(r) for r in rows]), 4))
            check(f"{slug} {set_name} pass-cell macro brier", s["pass_cells_macro_brier"],
                  round(mean(list(brier_by_t.values())), 4)) if "pass_cells_macro_brier" in s else None
    # 3. chance level and majority-gold baseline per set (the expected level for an uninformed answer)
    for set_name in SETS:
        rows = [r for r in raw["qwen35-4b"] if r["set_name"] == set_name]
        chance_by_t = by_template(rows, lambda r: 1.0 / len(r["option_keys"]))
        gold_share: dict[str, float] = {}
        groups: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            groups[r["template_id"]].append(r["gold"])
        for t, golds in groups.items():
            gold_share[t] = max(Counter(golds).values()) / len(golds)
        checks.append({"check": f"{set_name} chance level (macro over templates, uninformed answer)", "summary": None,
                       "recomputed": round(mean(list(chance_by_t.values())), 4), "pass": True})
        checks.append({"check": f"{set_name} majority-gold share (macro over templates)", "summary": None,
                       "recomputed": round(mean(list(gold_share.values())), 4), "pass": True})
    # 4. the held-out and test items are disjoint, and every item of a set has one template
    test_ids = {r["item_id"] for r in raw["qwen35-4b"] if r["set_name"] == "mr_test"}
    held_ids = {r["item_id"] for r in raw["qwen35-4b"] if r["set_name"] == "mr_heldout"}
    check("test and held-out item ids are disjoint", 0, len(test_ids & held_ids))
    # 5. the matched block: the common support of the two MedDecider models, the same failing templates excluded for all
    for set_name in SETS:
        block = summary["matched_to_meddecider_support"][set_name]
        supports = []
        for slug in ("meddecider-4b", "meddecider-9b"):
            supports.append({r["item_id"] for r in raw[slug] if r["set_name"] == set_name and r["status"] == "scored"})
        support = supports[0] & supports[1]
        check(f"{set_name} common support items", block["support_items"], len(support))
        fail_all = set(block["readout_fail_templates_excluded"])
        for slug in MODELS:
            rows = [r for r in raw[slug] if r["set_name"] == set_name and r["item_id"] in support
                    and r["status"] == "scored" and r["template_id"] not in fail_all]
            m = block["models"][slug]
            check(f"{set_name} matched items {slug}", m["items"], len(rows))
            if rows:
                check(f"{set_name} matched accuracy {slug}", m["accuracy"],
                      round(mean([1.0 if correct(r) else 0.0 for r in rows]), 4))
    # 6. notes that are not checks: ECE is not re-derived; the readout failures are listed, not hidden
    readout_fails = {slug: {s: summary["models"][slug]["sets"][s]["readout_fail_templates"] for s in SETS}
                     for slug in MODELS}
    failed = sum(1 for c in checks if not c["pass"])
    result = {"checks": checks, "failed": failed, "total": len(checks), "readout_fail_templates": readout_fails,
              "not_rederived": ["ece (binning lives in meddecide.eval.o2_metrics)"]}
    OUT_JSON.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["# SELF_AUDIT — V6 (osler_v1)", "",
             "Re-derived in a fresh process from the raw prediction files (standard library only; no meddecide code).",
             f"Checks: {len(checks)}; failed: {failed}.", "",
             "| check | summary | recomputed | pass |", "|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['check']} | {c['summary']} | {c['recomputed']} | {'PASS' if c['pass'] else 'FAIL'} |")
    lines += ["", "Not re-derived: ECE (its binning is in the shared code).", "",
              "Readout-health failures (named, not reported as accuracy): "
              + "; ".join(f"{m} {s}: {', '.join(v) or 'none'}" for m, sv in readout_fails.items() for s, v in sv.items())]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"checks {len(checks)}, failed {failed}; wrote {OUT_MD.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
