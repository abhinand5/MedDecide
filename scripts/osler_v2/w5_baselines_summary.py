"""osler_v2 W5: the baselines' aggregates (ADVISORY W5 step 4): zero-shot accuracy on knowledge dev v2 (with the D12 readout gate)
and the five baselines on the EHR-Rel test set (accuracy, and the mean absolute error of the expected level for the ordinal items).

Aggregates only. Items with a repeated option label (pplx's reference cannot score them) are excluded on every row, counted,
and named in the output. Writes outputs/osler_v2/W5/baselines_summary.json; loops/osler_v2/baselines_v2.md is written from it.

Usage: uv run --frozen python scripts/osler_v2/w5_baselines_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meddecide.eval.o2_metrics import cell_health
from meddecide.utils.io import iter_jsonl, write_json

REPO = Path(__file__).resolve().parents[2]
W5 = REPO / "outputs" / "osler_v2" / "W5"
KNOWLEDGE = REPO / "data" / "bench" / "osler_v2" / "knowledge_dev_v2.jsonl"
EHR = REPO / "data" / "bench" / "osler_v2" / "ehr_rel_test.jsonl"
ZS_KNOWLEDGE = {"qwen35-4b": W5 / "zeroshot_knowledge" / "qwen35-4b" / "predictions.jsonl",
                "pplx-decider-flash-v1.2": W5 / "zeroshot_knowledge" / "pplx-decider-flash-v1.2" / "predictions.jsonl"}
EHR_BASELINES = {name: W5 / "baselines" / name / "predictions.jsonl"
                 for name in ("qwen35-4b", "qwen35-9b", "meddecider-4b", "meddecider-9b", "jev-9b")}


def duplicate_label_ids() -> set[str]:
    out = set()
    for row in iter_jsonl(KNOWLEDGE):
        labels = [o["label"] for o in row["options"]]
        if len(set(labels)) != len(labels):
            out.add(row["item_id"])
    return out


def argmax_key(row: dict[str, Any]) -> str:
    keys = row["option_keys"] if "option_keys" in row else [o["key"] for o in row["options"]]
    probs = row["probs"]
    return keys[max(range(len(keys)), key=lambda i: probs[i])]


def expected_level(row: dict[str, Any]) -> float:
    """Expected level of an ordinal (score) item: the probability-weighted mean of the level numbers 1..N."""
    return sum((i + 1) * p for i, p in enumerate(row["probs"]))


def main() -> None:
    excluded = duplicate_label_ids()
    knowledge: dict[str, Any] = {"excluded_duplicate_label_items": len(excluded), "models": {}}
    for name, path in ZS_KNOWLEDGE.items():
        rows = [r for r in iter_jsonl(path) if r["item_id"] not in excluded]
        scored = [r for r in rows if r["status"] == "scored"]
        by_template: dict[str, list[dict[str, Any]]] = {}
        for r in scored:
            by_template.setdefault(r["template_id"], []).append(r)
        hits = sum(argmax_key(r) == r["gold"] for r in scored)
        if name.startswith("qwen"):  # the D12 gate reads the letter readout's label mass (Qwen rows only)
            health = {t: cell_health(f"zero-shot {name}", t, rs, zero_shot=True)["status"] for t, rs in by_template.items()}
        else:
            health = dict.fromkeys(by_template, "NOT APPLICABLE: reference readout (pplx's own 255-row readout, no letter mass)")
        knowledge["models"][name] = {
            "scored": len(scored), "skipped": len(rows) - len(scored),
            "accuracy": round(hits / len(scored), 4) if scored else None,
            "readout_health": health,
        }
    ehr: dict[str, Any] = {"items": 0, "models": {}}
    gold_by_id = {json.loads(line)["item_id"]: json.loads(line) for line in EHR.open(encoding="utf-8")}
    ehr["items"] = len(gold_by_id)
    for name, path in EHR_BASELINES.items():
        rows = list(iter_jsonl(path))
        hits = sum(argmax_key(r) == r["gold"] for r in rows)
        mae = sum(abs(expected_level(r) - int(r["gold"])) for r in rows) / len(rows)
        ehr["models"][name] = {"n": len(rows), "accuracy": round(hits / len(rows), 4),
                               "mean_abs_error_expected_level": round(mae, 4)}
    report = {"knowledge_dev_v2": knowledge, "ehr_rel_test": ehr,
              "note": "accuracy is the argmax against gold; ordinal error uses the expected level (levels 1-4)."}
    write_json(W5 / "baselines_summary.json", report)
    print(json.dumps(report, indent=1)[:2500])


if __name__ == "__main__":
    main()
