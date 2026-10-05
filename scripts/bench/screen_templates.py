#!/usr/bin/env python
"""T8: screen fresh templates with regex and bag-of-words baselines.

Trains on each template's **dev** split, scores on **test**, drops templates where either
baseline reaches the configured threshold (0.95), and writes:

* ``outputs/bench_v0/T8/screen.json`` — every number, with denominators;
* ``loops/bench_v0/template_screen.md`` — the committed kept/dropped table;
* ``data/bench/fresh/manifest_screened.json`` — the build manifest plus a per-template
  ``screen`` block marking dropped templates (the raw manifest is left untouched, R9).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from meddecide.bench.schema import Item
from meddecide.eval.screen import TemplateScreenResult, screen_template
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import read_jsonl
from meddecide.utils.provenance import Provenance, utcnow


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", type=Path, default=Path("data/bench/fresh"))
    parser.add_argument("--patterns", type=Path, default=Path("configs/template_screen_patterns.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/bench_v0.yaml"))
    parser.add_argument("--report", type=Path, default=Path("loops/bench_v0/template_screen.md"))
    parser.add_argument("--out", type=Path, default=Path("outputs/bench_v0/T8/screen.json"))
    parser.add_argument("--task", default="T8")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    patterns_by_template = yaml.safe_load(args.patterns.read_text()) or {}
    regex_threshold = float(cfg.get("screen", {}).get("regex_drop_threshold", 0.95))
    bow_threshold = float(cfg.get("screen", {}).get("bow_drop_threshold", 0.95))
    seed = int(cfg.get("seed", 0))

    by_template: dict[str, dict[str, list[Item]]] = {}
    input_files: dict[str, str] = {}
    for path in sorted(args.fresh.glob("*.jsonl")):
        rows, report = read_jsonl(path, Item)
        report.check_closes()
        input_files[path.name] = file_sha256(path)
        for row in rows:
            by_template.setdefault(row.template_id, {"dev": [], "test": []})[str(row.split)].append(row)

    results: list[TemplateScreenResult] = []
    for template_id in sorted(by_template):
        splits = by_template[template_id]
        patterns = patterns_by_template.get(template_id, {})
        # a template with an explicit "no patterns" marker is screened with an empty mapping
        patterns = {k: v for k, v in patterns.items() if k != "__no_patterns__"}
        result = screen_template(
            template_id,
            splits.get("dev", []),
            splits.get("test", []),
            patterns=patterns,
            regex_threshold=regex_threshold,
            bow_threshold=bow_threshold,
            seed=seed,
        )
        if template_id not in patterns_by_template:
            result.notes.append("no regex patterns configured for this template; regex baseline abstains")
        results.append(result)

    dropped = [r for r in results if r.drop]
    kept = [r for r in results if not r.drop]
    screen = {
        "generated_at_utc": utcnow(),
        "thresholds": {"regex": regex_threshold, "bow": bow_threshold},
        "patterns_file": str(args.patterns),
        "patterns_file_sha256": file_sha256(args.patterns),
        "input_files": input_files,
        "n_templates": len(results),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "templates": {r.template_id: r.to_dict() for r in results},
        "checks": {
            "all_templates_screened": len(results) > 0,
            "kept_plus_dropped_equals_total": len(kept) + len(dropped) == len(results),
            "no_drop_without_reason": all(r.drop_reasons for r in dropped),
            # every drop must carry either a named threshold or the single-class condition
            "every_drop_has_a_rule": all(
                all((">=" in reason) or reason == "single_class_test_split" for reason in r.drop_reasons)
                for r in dropped
            ),
        },
    }
    screen["verdict"] = "PASS" if all(screen["checks"].values()) else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(screen, indent=2) + "\n")

    # manifest with the screen verdict attached (the raw manifest is not modified)
    manifest_path = args.fresh / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        manifest["screen"] = {
            "screen_file": str(args.out),
            "thresholds": screen["thresholds"],
            "templates_kept": sorted(r.template_id for r in kept),
            "templates_dropped": sorted(r.template_id for r in dropped),
            "by_template": {
                r.template_id: {
                    "drop": r.drop,
                    "drop_reasons": r.drop_reasons,
                    "n_test": r.n_test,
                    "majority_test_accuracy": r.majority_test_accuracy,
                    "regex_test_accuracy": r.regex_test_accuracy,
                    "bow_test_accuracy": r.bow_test_accuracy,
                }
                for r in results
            },
        }
        (args.fresh / "manifest_screened.json").write_text(json.dumps(manifest, indent=2) + "\n")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_render_report(screen, results), encoding="utf-8")

    prov = Provenance(
        run_name=f"{args.task}_template_screen",
        command=" ".join([sys.executable, *sys.argv]),
        seed=seed,
        config={"patterns": str(args.patterns), "thresholds": screen["thresholds"]},
    )
    prov.finish().write(args.out.with_name("screen_provenance.json"))

    print(json.dumps({k: v for k, v in screen.items() if k != "templates"}, indent=2))
    print(f"kept={len(kept)} dropped={len(dropped)} -> {args.out}")
    return 0 if screen["verdict"] == "PASS" else 1


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _render_report(screen: dict[str, Any], results: list[TemplateScreenResult]) -> str:
    lines = [
        "# Fresh-template screen (T8)",
        "",
        f"**Generated:** {screen['generated_at_utc']}  ",
        f"**Patterns:** `{screen['patterns_file']}` (sha256 `{screen['patterns_file_sha256'][:16]}…`)  ",
        f"**Thresholds:regex >= {screen['thresholds']['regex']}, bag-of-words >= "
        f"{screen['thresholds']['bow']} on the **test** split → template dropped.",
        "",
        "The bag-of-words model (TF-IDF 1-2 grams + logistic regression) is fitted on each",
        "template's **dev** split and scored on its **test** split; the regex baseline is",
        "hand-written per template and never fitted. Nothing here touches a model's predictions.",
        "",
        "## Kept templates",
        "",
        "| template | source | qtype | n_test | classes | majority | regex (test) | "
        "regex macro (test) | BoW (test) | BoW macro (test) | BoW (dev) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: r.template_id):
        if r.drop:
            continue
        lines.append(
            f"| `{r.template_id}` | {r.source} | {r.qtype} | {r.n_test} | {r.n_test_classes} | "
            f"{_fmt(r.majority_test_accuracy)} | {_fmt(r.regex_test_accuracy)} | "
            f"{_fmt(r.regex_test_macro_accuracy)} | {_fmt(r.bow_test_accuracy)} | "
            f"{_fmt(r.bow_test_macro_accuracy)} | {_fmt(r.bow_dev_accuracy)} |"
        )
    lines += ["", "## Dropped templates", ""]
    dropped = [r for r in results if r.drop]
    if not dropped:
        lines.append("None: no template reached either threshold.")
    else:
        lines += [
            "| template | source | n_test | classes | majority | regex (test) | BoW (test) | reason |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(dropped, key=lambda r: r.template_id):
            lines.append(
                f"| `{r.template_id}` | {r.source} | {r.n_test} | {r.n_test_classes} | "
                f"{_fmt(r.majority_test_accuracy)} | {_fmt(r.regex_test_accuracy)} | "
                f"{_fmt(r.bow_test_accuracy)} | {', '.join(r.drop_reasons)} |"
            )
    lines += [
        "",
        "## Notes carried with the numbers",
        "",
        f"- {screen['n_kept']} of {screen['n_templates']} templates kept; "
        f"{screen['n_dropped']} dropped.",
        "- A `noul` template has chance = 0.5, so a BoW score *below* 0.5 is not evidence the",
        "  template is hard — it can mean the classifier collapsed to the majority class.",
        "- Templates whose answer is not stated in the state (major MeSH topic, pharmacologic",
        "  class) have no honest regex cue; their regex rows are 0 by construction and the",
        "  baseline that matters for them is the bag-of-words one.",
        "- Dropping a template removes it from T9's headline table; it stays in the data so the",
        "  decision is reviewable.",
        "",
    ]
    for r in sorted(results, key=lambda r: r.template_id):
        if r.notes:
            lines.append(f"- `{r.template_id}`: {'; '.join(n for n in r.notes if n)}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
