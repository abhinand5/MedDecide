"""osler_v5 Z0 (ADVISORY Z0 steps 2-3): the osler_v4 final mix, composed in memory, and its per-template counts (CPU).

Prints and writes counts only: rows, option-count distribution, distinct gold labels, and the MIMIC flag per template. No item text,
label text, identifier or value is printed or written, because the mix holds MIMIC-derived rows that never leave this machine.
Used to write the closed list of the option-count augmentation (D52a) before any run.

Usage (CPU):
    uv run --frozen python scripts/osler_v5/z0_mix_stats.py
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from meddecide.train.final_mix import compose_final_mix
from meddecide.utils.io import write_json

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v5" / "Z0" / "mix_template_stats.json"
MIMIC_PREFIX = "mimic_"
TWO_LETTER_FROM = 26  # option positions 26 and above take the two-letter codes (D52b)


def _y8_module():
    spec = importlib.util.spec_from_file_location("y8_final_mix", REPO / "scripts" / "osler_v4" / "y8_final_mix.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    y8 = _y8_module()
    decisions = {name: y8.entering(name) for name in ("B1", "B2", "B3", "B4")}
    family = {name: y8.add_files(name) for name in y8.FAMILY if decisions[name][0]}
    other = y8.add_files("B4") if decisions["B4"][0] else []
    started = time.time()
    rows, report = compose_final_mix(y8.BASE, family, other, y8.BROAD)
    print(f"composed {len(rows)} rows in {round(time.time() - started)} s; composition {report['composition_sha256'][:8]}…", flush=True)

    counts: Counter[str] = Counter()
    qtypes: dict[str, set[str]] = defaultdict(set)
    n_options: dict[str, list[int]] = defaultdict(list)
    gold_labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        template = str(row["template_id"])
        counts[template] += 1
        qtypes[template].add(str(row["qtype"]))
        n_options[template].append(len(row["options"]))
        gold_key = str(row["gold"])
        gold_labels[template].add(next(str(o["label"]) for o in row["options"] if str(o["key"]) == gold_key))

    templates = {}
    for template in sorted(counts):
        options = n_options[template]
        templates[template] = {
            "rows": counts[template],
            "qtype": sorted(qtypes[template]),
            "mimic": template.startswith(MIMIC_PREFIX),
            "options_min": min(options),
            "options_median": statistics.median(options),
            "options_max": max(options),
            "rows_over_26_options": sum(n > TWO_LETTER_FROM for n in options),
            "distinct_gold_labels": len(gold_labels[template]),
        }
    total = len(rows)
    mimic_rows = sum(counts[t] for t in counts if t.startswith(MIMIC_PREFIX))
    summary = {
        "rows": total, "composition_sha256": report["composition_sha256"], "mimic_rows": mimic_rows,
        "mimic_share": mimic_rows / total,
        "rows_over_26_options": sum(v["rows_over_26_options"] for v in templates.values()),
        "share_over_26_options": sum(v["rows_over_26_options"] for v in templates.values()) / total,
        "templates": templates,
    }
    write_json(OUT, summary)
    print(f"rows {total}; MIMIC rows {mimic_rows} ({mimic_rows / total:.4f}); rows over 26 options "
          f"{summary['rows_over_26_options']} ({summary['share_over_26_options']:.4f}); templates {len(templates)}", flush=True)
    for template, stats in templates.items():
        if stats["qtype"] == ["choice"] or stats["options_max"] > 8:
            print(json.dumps({"template": template, **{k: v for k, v in stats.items() if k != "qtype"},
                              "qtype": stats["qtype"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
