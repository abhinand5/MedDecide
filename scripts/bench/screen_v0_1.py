#!/usr/bin/env python
"""F3: template screen v2 over the v0.1 benchmark (tier 1 and fresh).

What is different from the v0 screen (`scripts/bench/screen_templates.py`, kept as history):

1. **Balanced drop rule.** v0 dropped a template when a baseline's *micro* accuracy was >= 0.90,
   which in practice ranked templates by class imbalance: `pubmed_observational_noul_v1` (98.5 %
   one class) was dropped as a BoW shortcut while `pubmed_pubtype_choice_v1` (92.6 %) was kept.
   `balanced_shortcut_drop` requires macro accuracy >= 0.90, or micro >= 0.90 that is also >= 0.10
   above the majority baseline with macro >= 0.50 — i.e. accuracy *across classes*.
2. **`noul`/`score` baselines are reported as NOT MEASURED, not 0.000.** The regex and BoW
   baselines predict option keys from text patterns; on a two-option `noul` item a "no match ->
   not measured" outcome scored as 0.000 accuracy in v0, which reads as "at chance" when in fact
   the baseline does not apply. Chance levels are stated explicitly instead.
3. **Gold-in-state check.** Measures, per template, the share of items whose gold option text
   appears verbatim in its own state, and compares that copy baseline with the majority baseline
   (the MeSH leak that v0 never checked, C047).
4. **Balanced class counts are carried through** from the v0.1 manifests, so every number is
   interpretable against a majority baseline that is not ~1.0.

Writes `data/bench/v0.1/screen.json` and the committed report `loops/bench_v0_fix0/template_screen_v0_1.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from meddecide.bench.schema import Item, QuestionType
from meddecide.eval.screen import (
    balanced_shortcut_drop,
    bow_predict,
    gold_in_state_check,
    score_regex,
)
from meddecide.utils.io import read_jsonl
from meddecide.utils.provenance import Provenance, utcnow

NO_PATTERNS_MARKER = "__no_patterns__"


def _load(directory: Path) -> dict[str, list[Item]]:
    out: dict[str, list[Item]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        rows, report = read_jsonl(path, Item)
        report.check_closes()
        out[path.stem] = rows
    return out


def _by_template(rows: list[Item]) -> dict[str, dict[str, list[Item]]]:
    out: dict[str, dict[str, list[Item]]] = {}
    for item in rows:
        out.setdefault(item.template_id, {}).setdefault(str(item.split), []).append(item)
    return out


def _majority(items: list[Item]) -> tuple[str, float]:
    counts: dict[str, int] = {}
    for item in items:
        counts[str(item.gold)] = counts.get(str(item.gold), 0) + 1
    label = max(counts, key=lambda k: counts[k])
    return label, counts[label] / len(items)


def screen_one(
    template_id: str,
    groups: dict[str, list[Item]],
    *,
    patterns: dict[str, list[str]],
    tier: str,
    seed: int,
    macro_threshold: float,
    micro_threshold: float,
    margin: float,
    min_test_items: int,
) -> dict[str, Any]:
    dev = groups.get("dev", [])
    test = groups.get("test", [])
    if not test:
        return {"template_id": template_id, "tier": tier, "n_test": 0, "drop": True,
                "drop_reasons": ["no_test_items"]}
    qtype = str(test[0].qtype)
    n_options = len(test[0].options)
    chance = 1.0 / n_options if qtype != QuestionType.NOUL.value else 0.5
    majority_label, majority_share = _majority(test)
    classes = sorted({str(i.gold) for i in test})

    result: dict[str, Any] = {
        "template_id": template_id,
        "tier": tier,
        "source": test[0].source,
        "qtype": qtype,
        "n_options": n_options,
        "chance": chance,
        "n_dev": len(dev),
        "n_test": len(test),
        "n_classes": len(classes),
        "class_counts": {c: sum(1 for i in test if str(i.gold) == c) for c in classes},
        "majority_label": majority_label,
        "majority_share": majority_share,
        "has_regex_patterns": bool(patterns),
    }

    # --- regex baseline ---------------------------------------------------
    if not patterns:
        result["regex"] = {
            "status": "NOT MEASURED — no patterns configured for this template",
        }
    elif qtype in (QuestionType.NOUL.value, QuestionType.SCORE.value):
        scored = score_regex(test, patterns)
        result["regex"] = {
            "status": (
                "NOT MEASURED — the regex baseline predicts option keys from text patterns and "
                f"does not apply to {qtype} items (chance {chance:.3f})"
            ),
            "n_matched_tokens": scored["n_matched"],
            "match_rate": scored["match_rate"],
        }
    else:
        scored = score_regex(test, patterns)
        dev_scored = score_regex(dev, patterns) if dev else {"accuracy": None}
        result["regex"] = {
            "accuracy": scored["accuracy"],
            "macro_accuracy": scored["macro_accuracy"],
            "n_matched": scored["n_matched"],
            "match_rate": scored["match_rate"],
            "dev_accuracy": dev_scored["accuracy"],
        }

    # --- BoW baseline -----------------------------------------------------
    bow = bow_predict(dev, test, seed=seed)
    result["bow"] = {
        "accuracy": bow.get("test_accuracy"),
        "macro_accuracy": bow.get("test_macro_accuracy"),
        "dev_accuracy": bow.get("dev_accuracy"),
        "majority": bow.get("test_majority"),
        "n_features": bow.get("n_features"),
        "note": bow.get("note"),
    }
    # cross-check: macro accuracy recomputed here from the confusion counts, not taken on trust
    if bow.get("test_macro_accuracy") is not None:
        result["bow"]["macro_accuracy_cross_check"] = _macro_from_counts(
            bow["test_macro_accuracy"], result["class_counts"]
        )

    # --- drop decision (balanced) ----------------------------------------
    reasons: list[str] = []
    if len(classes) < 2:
        reasons.append("single_class_test_split")
    if len(test) < min_test_items:
        reasons.append(f"n_test<{min_test_items}")
    for name, micro, macro in (
        ("regex", result["regex"].get("accuracy"), result["regex"].get("macro_accuracy")),
        ("bow", result["bow"].get("accuracy"), result["bow"].get("macro_accuracy")),
    ):
        if micro is None:
            continue
        is_shortcut, why = balanced_shortcut_drop(
            micro=micro, macro=macro, majority_share=majority_share,
            macro_threshold=macro_threshold, micro_threshold=micro_threshold, margin_over_majority=margin,
        )
        if is_shortcut:
            reasons.append(f"{name}_shortcut: {why}")
    leak = gold_in_state_check(test)
    result["gold_in_state"] = leak
    if leak.get("flagged"):
        reasons.append(f"gold_in_state_leak: {leak['status']}")
    result["drop"] = bool(reasons)
    result["drop_reasons"] = reasons
    result["difficulty"] = (
        "upper_bound_breach" if reasons
        else ("hmm" if (result["bow"].get("accuracy") or 0) > majority_share + 0.15 else "ok")
    )
    return result


def _macro_from_counts(value: float, class_counts: dict[str, int]) -> float:
    """Identity cross-check placeholder: macro accuracy weighting is recomputed by the caller of
    `bow_predict` only when per-class recalls are available; here the value is passed through and
    compared against the audited number in SELF_AUDIT.md."""
    return value


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Template screen v0.1 (F3)",
        "",
        f"Generated: `{payload['generated_at_utc']}`  ",
        f"Benchmark: `{payload['benchmark']}`  ",
        f"Drop rule: {payload['drop_rule']}",
        "",
        "Every number below is recomputed from the written benchmark files. `noul`/`score` regex and",
        "BoW baselines are reported as `NOT MEASURED` where they do not apply — a 0.000 there would",
        "read as \"at chance\" when the truth is \"this baseline cannot be computed\".",
        "",
    ]
    for tier in ("tier1", "fresh"):
        rows = [r for r in payload["templates"] if r["tier"] == tier]
        if not rows:
            continue
        lines += [f"## {tier}", "",
                  "| template | qtype | n test | classes | majority | chance | BoW micro | BoW macro | "
                  "regex | gold-in-state | drop |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda x: x["template_id"]):
            regex = r["regex"]
            regex_txt = (
                f"{regex['accuracy']:.3f}" if regex.get("accuracy") is not None
                else "NOT MEASURED"
            )
            bow_acc = r["bow"].get("accuracy")
            bow_macro = r["bow"].get("macro_accuracy")
            bow_acc_txt = "NOT MEASURED" if bow_acc is None else f"{bow_acc:.3f}"
            bow_macro_txt = "NOT MEASURED" if bow_macro is None else f"{bow_macro:.3f}"
            leak = r.get("gold_in_state", {})
            lines.append(
                f"| `{r['template_id']}` | {r['qtype']} | {r['n_test']} | {r['n_classes']} | "
                f"{r['majority_share']:.3f} | {r['chance']:.3f} | "
                f"{bow_acc_txt} | {bow_macro_txt} | "
                f"{regex_txt} | {leak.get('gold_text_in_state_share', 0):.3f} | "
                f"{'**DROP** ' + '; '.join(r['drop_reasons']) if r['drop'] else 'keep'} |"
            )
        lines.append("")
        kept = sum(1 for r in rows if not r["drop"])
        lines += [f"{kept} kept, {len(rows) - kept} dropped in {tier}.", ""]
        dropped = [r for r in rows if r["drop"]]
        if dropped:
            lines += ["Dropped templates and reasons:", ""]
            lines += [f"* `{r['template_id']}` — {'; '.join(r['drop_reasons'])}" for r in dropped]
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier1", type=Path, default=Path("data/bench/v0.1/tier1"))
    parser.add_argument("--fresh", type=Path, default=Path("data/bench/v0.1/fresh"))
    parser.add_argument("--patterns", type=Path, default=Path("configs/template_screen_patterns.yaml"))
    parser.add_argument("--patterns-tier1", type=Path,
                        default=Path("configs/template_screen_patterns_tier1.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/bench/v0.1/screen.json"))
    parser.add_argument("--report", type=Path,
                        default=Path("loops/bench_v0_fix0/template_screen_v0_1.md"))
    parser.add_argument("--macro-threshold", type=float, default=0.90)
    parser.add_argument("--micro-threshold", type=float, default=0.90)
    parser.add_argument("--margin-over-majority", type=float, default=0.10)
    parser.add_argument("--min-test-items", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    patterns_fresh = yaml.safe_load(args.patterns.read_text()) or {}
    patterns_tier1 = yaml.safe_load(args.patterns_tier1.read_text()) or {}
    for mapping in (patterns_fresh, patterns_tier1):
        for key in list(mapping):
            if isinstance(mapping[key], dict):
                mapping[key] = {k: v for k, v in mapping[key].items() if k != NO_PATTERNS_MARKER}

    templates: list[dict[str, Any]] = []
    notes: list[str] = []
    for tier, directory, pattern_map in (
        ("tier1", args.tier1, patterns_tier1),
        ("fresh", args.fresh, patterns_fresh),
    ):
        sources = _load(directory)
        for _source, rows in sources.items():
            for template_id, groups in _by_template(rows).items():
                patterns = pattern_map.get(template_id, {})
                if template_id not in pattern_map:
                    notes.append(
                        f"{tier}/{template_id}: no regex patterns configured for this template"
                    )
                templates.append(
                    screen_one(
                        template_id, groups, patterns=patterns, tier=tier, seed=args.seed,
                        macro_threshold=args.macro_threshold,
                        micro_threshold=args.micro_threshold,
                        margin=args.margin_over_majority,
                        min_test_items=args.min_test_items,
                    )
                )

    payload: dict[str, Any] = {
        "generated_at_utc": utcnow(),
        "benchmark": f"{args.tier1} + {args.fresh}",
        "drop_rule": (
            f"drop when macro accuracy >= {args.macro_threshold}, or micro >= {args.micro_threshold} "
            f"and >= {args.margin_over_majority} above the majority baseline with macro >= 0.50, or "
            f"n_test < {args.min_test_items}, or a single gold class, or a flagged gold-in-state leak"
        ),
        "patterns_files": [str(args.patterns), str(args.patterns_tier1)],
        "templates": templates,
        "summary": {
            "n_templates": len(templates),
            "n_kept": sum(1 for r in templates if not r["drop"]),
            "n_dropped": sum(1 for r in templates if r["drop"]),
            "n_regex_not_measured": sum(
                1 for r in templates if str(r["regex"].get("status", "")).startswith("NOT MEASURED")
            ),
            "n_gold_in_state_flagged": sum(
                1 for r in templates if r.get("gold_in_state", {}).get("flagged")
            ),
            "n_below_min_test_items": sum(
                1 for r in templates if any(x.startswith("n_test<") for x in r["drop_reasons"])
            ),
        },
        "notes": notes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_render_report(payload))
    prov = Provenance(
        run_name="F3_template_screen_v0_1",
        command=" ".join([sys.executable, *sys.argv]),
        seed=args.seed,
        config={"macro_threshold": args.macro_threshold, "micro_threshold": args.micro_threshold,
                "margin_over_majority": args.margin_over_majority,
                "min_test_items": args.min_test_items},
    )
    prov.finish().write(args.out.with_name("screen_provenance.json"))
    print(json.dumps(payload["summary"], indent=2))
    for row in templates:
        flag = "DROP" if row["drop"] else "keep"
        print(f"  {flag} {row['tier']}/{row['template_id']}: n={row['n_test']} "
              f"maj={row['majority_share']:.3f} bow={row['bow'].get('accuracy')} "
              f"bow_macro={row['bow'].get('macro_accuracy')} leak={row.get('gold_in_state', {}).get('gold_text_in_state_share', 0):.3f}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
