#!/usr/bin/env python
"""V4 screen: each new training template must pass on its own pre-window dev split (ADVISORY V4).

For every template in the V4 component files, a bag-of-words classifier is fitted on the template's
pre-window **train** items and scored on its pre-window **dev** items. A template passes when all hold:

  * at least two gold classes appear in its dev items;
  * BoW **macro** accuracy on dev is below 0.90 (the ADVISORY threshold; the micro rule of the
    benchmark screen is reported beside it, not used for the decision);
  * ``gold_in_state_check`` does not flag the template (the gold text is not copied into the state).

Writes ``outputs/student_v1/V4/screen.json`` and ``data/train/student_v1/kept_templates.json``.
Held-out templates are refused by the builders and are not screened here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.bench.schema import Item  # noqa: E402
from meddecide.eval.screen import bow_predict, gold_in_state_check  # noqa: E402

MACRO_THRESHOLD = 0.90
COMPONENTS = ROOT / "data" / "train" / "student_v1" / "components"
OUT = ROOT / "outputs" / "student_v1" / "V4" / "screen.json"
KEPT = ROOT / "data" / "train" / "student_v1" / "kept_templates.json"


def load_by_template(paths: list[Path]) -> dict[str, dict[str, list[Item]]]:
    grouped: dict[str, dict[str, list[Item]]] = defaultdict(lambda: {"train": [], "dev": []})
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = Item.model_validate_json(line)
                grouped[item.template_id][str(item.split)].append(item)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, nargs="+",
                        default=sorted(COMPONENTS.glob("*_v1.jsonl")))
    args = parser.parse_args()

    grouped = load_by_template(list(args.components))
    results = {}
    kept, dropped = [], []
    for template_id in sorted(grouped):
        train, dev = grouped[template_id]["train"], grouped[template_id]["dev"]
        reasons = []
        classes = sorted({i.gold for i in dev})
        if len(classes) < 2:
            reasons.append("dev_single_class")
        if not dev or not train:
            reasons.append("empty_train_or_dev")
            bow = {"test_accuracy": None, "test_macro_accuracy": None}
        else:
            bow = bow_predict(train, dev, seed=0)
            if bow.get("test_macro_accuracy") is None or bow["test_macro_accuracy"] >= MACRO_THRESHOLD:
                reasons.append(f"bow_macro>={MACRO_THRESHOLD}")
        in_state = gold_in_state_check(dev) if dev else {"flagged": False}
        if in_state.get("flagged"):
            reasons.append("gold_text_in_state")
        results[template_id] = {
            "source": dev[0].source if dev else (train[0].source if train else None),
            "qtype": str(dev[0].qtype) if dev else None,
            "n_train": len(train),
            "n_dev": len(dev),
            "dev_gold_classes": classes,
            "bow_dev_macro_accuracy": bow.get("test_macro_accuracy"),
            "bow_dev_micro_accuracy": bow.get("test_accuracy"),
            "bow_dev_majority": bow.get("test_majority"),
            "gold_in_state": {k: v for k, v in in_state.items() if k != "per_class_share"},
            "decision": "PASS" if not reasons else "DROP",
            "reasons": reasons,
        }
        (kept if not reasons else dropped).append(template_id)

    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        head = "unknown"
    report = {
        "kind": "v1_screen",
        "rule": f"BoW macro accuracy on pre-window dev < {MACRO_THRESHOLD}; >=2 dev classes; "
                "gold_in_state_check not flagged",
        "inputs": [str(p.relative_to(ROOT)) for p in args.components],
        "git_commit": head,
        "kept": kept,
        "dropped": dropped,
        "templates": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    KEPT.parent.mkdir(parents=True, exist_ok=True)
    KEPT.write_text(json.dumps({"kept": kept, "dropped": dropped}, indent=2) + "\n", encoding="utf-8")
    for template_id in sorted(results):
        r = results[template_id]
        macro = r["bow_dev_macro_accuracy"]
        macro_s = f"{macro:.4f}" if macro is not None else "n/a"
        print(f"{r['decision']:4s} {template_id} train={r['n_train']} dev={r['n_dev']} "
              f"bow_macro={macro_s} {','.join(r['reasons'])}")


if __name__ == "__main__":
    main()
