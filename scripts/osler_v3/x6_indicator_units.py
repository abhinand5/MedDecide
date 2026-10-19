"""osler_v3 X6 check (ADVISORY section 7; D44; the 'too good' rule): do chart indicator items that state a unit value score higher?

The chart indicator template (chart_indicator_evidence_v1) is yes/no. Its narrative pages may state a unit value, which the prompt
forbids (X4, C068). This script splits the new-test-pool indicator items by whether any narrative page states a unit value (the
osler_v2 UNIT pattern, meddecide-free: scripts/osler_v2/w3b_pilot_checks.py), and reports the accuracy of two zero-shot baselines on
each part. It is an additional check (never a verdict). A high accuracy on the items WITHOUT a stated value would mean the record
carries the answer by other means; the split reports both parts.

Writes outputs/osler_v3/X6/indicator_units.json (aggregates only). Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x6_indicator_units.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from meddecide.eval.o2_metrics import score_row
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "data" / "bench" / "osler_v3" / "new_test_o2.jsonl"
ROOT = REPO / "outputs" / "osler_v3" / "X6" / "new_test"
OUT = REPO / "outputs" / "osler_v3" / "X6" / "indicator_units.json"
MODELS = {
    "zero-shot pplx-decider-flash-v1.2": "zeroshot_pplx/pplx-decider-flash-v1.2/predictions.jsonl",
    "zero-shot Qwen3.5-9B": "zeroshot_qwen9b/qwen35-9b/predictions.jsonl",
}
TEMPLATE = "chart_indicator_evidence_v1"


def _w3b() -> Any:
    spec = importlib.util.spec_from_file_location("w3b_pilot_checks", REPO / "scripts" / "osler_v2" / "w3b_pilot_checks.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    w3b = _w3b()

    def has_unit(state: str) -> bool:
        return any(w3b.UNIT.search(body) for header, body in w3b.pages(state) if w3b.is_narrative(header))

    items = {str(r["item_id"]): r for r in (json.loads(line) for line in POOL.open(encoding="utf-8") if line.strip())
             if r["template_id"] == TEMPLATE}
    report: dict[str, Any] = {"built_utc": utcnow(), "git_commit": git_commit(REPO), "template": TEMPLATE,
                              "note": "additional check; never a verdict", "models": {}}
    for name, rel in MODELS.items():
        preds = {str(r["item_id"]): r for r in (json.loads(line) for line in (ROOT / rel).open(encoding="utf-8") if line.strip())}
        parts: dict[str, list[bool]] = {"with_stated_unit_value": [], "without_stated_unit_value": []}
        for item_id, item in items.items():
            if item_id not in preds:
                continue
            key = "with_stated_unit_value" if has_unit(item["state"]) else "without_stated_unit_value"
            parts[key].append(bool(score_row(preds[item_id]).correct))
        report["models"][name] = {key: {"items": len(v), "accuracy": round(sum(v) / len(v), 4) if v else None}
                                  for key, v in parts.items()}
        print(name, report["models"][name], flush=True)
    write_json(OUT, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
