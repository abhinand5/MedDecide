"""osler_v3 X4 checks (ADVISORY X4; D44(b)): the label-tracking check, the stated-number check, the O3 Naive Bayes screen on dev,
and the leakage check, for the chart items built by osler_v2's generator (scripts/osler_v2/w3b_charts.py).

The D44(b) rule: a phrase that states the fact the label is computed from is exempt; any other label-tracking phrase blocks its
own template only. For chart_icd_support_v1 the label is computed from the discharge diagnosis, so its certainty words (ruled out,
confirmed, diagnosed) are the labelled fact and exempt. The indicator and evidence-page templates have no exempt phrase: any
flagged phrase blocks them.

The phrase test is osler_v2's (w3b_pilot_checks.py: narrative 2- and 3-grams, ratio at least 3, difference at least 0.10, at least
10 items), reused unchanged, so the threshold is the same as the one the osler_v2 pilots were judged by.

Writes outputs/osler_v3/X4/checks.json (aggregates; phrases are listed because they are not item text) and prints a summary.
Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x4_charts_check.py --dir data/gen/osler_v3/charts
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.gen.screen import NaiveBayes, gold_label, macro_accuracy
from meddecide.mix.protected import hit_total, leakage_counts, load_protected
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v3" / "X4" / "checks.json"
SPLITS = ("train", "dev", "test")
SCREEN_THRESHOLD = 0.90
# D44(b): the words that state the labelled fact of each template (exempt from the label-tracking check)
LABELLED_FACT = {  # fixed before the full build; each entry states whether the diagnosis is present or ruled out
    "chart_icd_support_v1": ("ruled out", "rule out", "out current", "confirmed", "diagnosed", "diagnosis of", "does have",
                             "does not have", "determined"),
}
MIN_GROUP = 10  # a group below this size has no rate to compare (osler_v2's test needed none: its groups were large)


def _osler_v2_checks():
    spec = importlib.util.spec_from_file_location("w3b_pilot_checks", REPO / "scripts" / "osler_v2" / "w3b_pilot_checks.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load osler_v2's chart checks")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_split_rows(directory: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        rows[split] = [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []
    return rows


def label_tracking(rows: list[dict[str, Any]], w3b: Any, template: str) -> dict[str, Any]:
    """Phrases that track a gold group, one group against the rest (osler_v2's test: ratio at least 3, difference at least 0.10,
    at least 10 items), with the labelled fact removed. With two groups this is osler_v2's comparison; with more it is run for
    each group, so no template goes unchecked."""
    groups: dict[str, list[set[str]]] = defaultdict(list)
    for r in rows:
        text = w3b.narrative_text(r).lower()
        words = [w for w in re.findall(r"[a-z]+", text) if w not in w3b.STOP]
        grams = {" ".join(words[i:i + 2]) for i in range(len(words) - 1)}
        grams |= {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}
        group = str(r["gold"])
        if template == "chart_icd_support_v1":
            gold_text = next(o["label"] for o in w3b.options_of(r) if o["key"] == r["gold"])
            group = "none" if gold_text.startswith("None") else "code"
        groups[group].append(grams)
    result: dict[str, Any] = {"groups": {g: len(v) for g, v in groups.items()}, "flagged": [], "exempt": []}
    if len(groups) < 2:
        result["note"] = "one gold group only: no comparison"
        return result
    small = sorted(g for g, v in groups.items() if len(v) < MIN_GROUP)
    result["groups_below_min_size_not_compared"] = small
    counts: dict[str, Counter[str]] = {g: Counter() for g in groups}
    for g, sets in groups.items():
        for s_ in sets:
            counts[g].update(s_)
    total = sum(len(v) for v in groups.values())
    exempt_words = LABELLED_FACT.get(template, ())
    seen: set[tuple[str, str]] = set()
    for g in groups:
        if len(groups[g]) < MIN_GROUP:
            continue
        n_in, n_out = len(groups[g]), total - len(groups[g])
        for phrase in set().union(*(c.keys() for c in counts.values())):
            c_in = counts[g][phrase]
            c_out = sum(counts[o][phrase] for o in groups if o != g)
            if max(c_in, c_out) < 10 or n_out == 0 or (g, phrase) in seen:
                continue
            seen.add((g, phrase))
            r_in, r_out = c_in / n_in, c_out / n_out
            ratio = (max(r_in, r_out) + 1e-9) / (min(r_in, r_out) + 1e-9)
            if not (ratio >= 3 and abs(r_in - r_out) >= 0.10):
                continue
            entry = {"phrase": phrase, "group": g, "share_in_group": round(r_in, 3), "share_rest": round(r_out, 3)}
            (result["exempt"] if any(word in phrase for word in exempt_words) else result["flagged"]).append(entry)
    result["flagged"].sort(key=lambda e: -abs(e["share_in_group"] - e["share_rest"]))
    result["flagged"] = result["flagged"][:30]
    result["exempt"] = result["exempt"][:30]
    return result


def stated_numbers(rows: list[dict[str, Any]], w3b: Any) -> dict[str, int]:
    """Narrative pages that state a unit value (the prompt forbids it): counts only."""
    pages_total = pages_with_unit = 0
    for r in rows:
        for header, body in w3b.pages(r["state"]):
            if not w3b.is_narrative(header):
                continue
            pages_total += 1
            pages_with_unit += bool(w3b.UNIT.search(body))
    return {"narrative_pages": pages_total, "narrative_pages_with_unit_value": pages_with_unit}


def screen(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> dict[str, Any]:
    if not train or not dev:
        return {"status": "NOT MEASURED — no train or dev rows"}
    nb = NaiveBayes().fit(train)
    macro = macro_accuracy([(gold_label(r), nb.predict(r)) for r in dev])
    return {"train_rows": len(train), "dev_rows": len(dev), "naive_bayes_macro": None if macro is None else round(macro, 4),
            "flag_at_or_above_0_90": bool(macro is not None and macro >= SCREEN_THRESHOLD)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/gen/osler_v3/charts")
    ap.add_argument("--first-charts", type=int, default=0,
                    help="check only the first N charts in file order (train, then dev, then test); 0 checks every chart")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--original-exemptions", action="store_true",
                    help="D-X4-2: chart_icd_support_v1 exempts only the original words (ruled out, rule out, out current)")
    args = ap.parse_args()
    if args.original_exemptions:
        LABELLED_FACT["chart_icd_support_v1"] = ("ruled out", "rule out", "out current")
    directory = REPO / args.dir
    w3b = _osler_v2_checks()
    rows = load_split_rows(directory)
    if args.first_charts:
        first: dict[str, None] = {}
        for s in SPLITS:
            for r in rows[s]:
                first.setdefault(r["pair_id"].split(":")[1], None)
        keep = set(list(first)[: args.first_charts])
        rows = {s: [r for r in rows[s] if r["pair_id"].split(":")[1] in keep] for s in SPLITS}
    all_rows = [r for s in SPLITS for r in rows[s]]
    templates = sorted({r["generator"] for r in all_rows})
    report: dict[str, Any] = {"checked_utc": utcnow(), "git_commit": git_commit(REPO), "dir": args.dir,
                              "first_charts": args.first_charts or None,
                              "labelled_fact": {k: list(v) for k, v in LABELLED_FACT.items()},
                              "items_by_split": {s: len(rows[s]) for s in SPLITS},
                              "items_by_template": dict(Counter(r["generator"] for r in all_rows)),
                              "templates": {}}
    text, bare, qualified = load_protected(REPO)
    adapt = [{"state": r["state"], "question": r["question"], "options": r["options"],
              "source_record_id": str(r.get("pair_id", r["item_id"])), "source": "synthea_chart", "meta": {}} for r in all_rows]
    adapted = {s_: [a for a, r in zip(adapt, all_rows, strict=True) if r["split"] == s_] for s_ in SPLITS}
    report["leakage"] = {s_: leakage_counts(adapted[s_], text, bare, qualified) for s_ in SPLITS if adapted[s_]}
    for template in templates:
        t_rows = {s: [r for r in rows[s] if r["generator"] == template] for s in SPLITS}
        tracking = label_tracking(t_rows["train"] + t_rows["dev"], w3b, template)
        report["templates"][template] = {
            "items": {s: len(t_rows[s]) for s in SPLITS},
            "label_tracking": tracking,
            "stated_numbers": stated_numbers(t_rows["train"] + t_rows["dev"] + t_rows["test"], w3b),
            "screen_naive_bayes": screen(t_rows["train"], t_rows["dev"]),
            "blocked": bool(tracking.get("flagged")),
        }
    write_json(args.out, report)
    for template, info in report["templates"].items():
        print(f"{template}: items {info['items']}, flagged phrases {len(info['label_tracking'].get('flagged', []))}, "
              f"exempt {len(info['label_tracking'].get('exempt', []))}, blocked {info['blocked']}, "
              f"NB {info['screen_naive_bayes'].get('naive_bayes_macro')}", flush=True)
    return 1 if hit_total(report["leakage"].get("train", {})) else 0


if __name__ == "__main__":
    sys.exit(main())
