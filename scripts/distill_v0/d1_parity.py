"""D1 parity test: the same frozen items labelled twice, plus a comparison with the frozen D1 labels.

The teacher is not deterministic (ADVISORY §1.1: identical calls differ in the fourth decimal). This
script measures that on a fixed sample so the frozen label file's reproducibility is stated rather
than assumed: a deterministic sample of the panel set (first 40 by item id), every item labelled in
two independent rounds through the same code path, then compared by argmax and by probability.

Writes ``parity.json`` (comparisons, accuracies, cost) and ``parity_labels.jsonl`` (freeze rows for
both rounds; no item text). Run after ``d1_ceiling.py`` so the frozen D1 labels are present.

Usage (CPU/network):
    source /workspace/.secrets.env && uv run --frozen python scripts/distill_v0/d1_parity.py \
        --out outputs/distill_v0/D1 --sample 40
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from meddecide.distill.d1_client import credits, label_one
from meddecide.distill.d1_teacher import (
    CostLedger,
    file_sha256,
    hosted_guard,
    is_correct,
    load_frozen,
    request_for_item,
)
from meddecide.distill.soft_labels import compare_scorings
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
Y9 = REPO / "data" / "bench" / "osler_v4" / "y9"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_round(api_key: str, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import threading

    ledger = CostLedger(cap_usd=1.0)  # parity is a few cents; a separate small cap makes a runaway impossible
    fatal = threading.Event()
    rows: list[dict[str, Any]] = []
    for item in items:
        request, reason = request_for_item(item)
        if request is None:
            raise SystemExit(f"{item['item_id']}: unrequestable ({reason})")
        row, failure = label_one(api_key, request, ledger, fatal)
        if row is None:
            raise SystemExit(f"{item['item_id']}: parity call failed ({failure})")
        rows.append(row)
    return rows, ledger.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="outputs/distill_v0/D1")
    parser.add_argument("--sample", type=int, default=40)
    args = parser.parse_args()
    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set (source /workspace/.secrets.env)")
    started = utcnow()
    panel_path = Y9 / "panel.jsonl"
    items = sorted(read_jsonl(panel_path), key=lambda row: str(row["item_id"]))[: args.sample]
    for item in items:
        allowed, why = hosted_guard(item)
        if not allowed:
            raise SystemExit(f"{item['item_id']}: not allowed for the hosted teacher ({why})")
    frozen = load_frozen(out / "labels.jsonl")
    balance = credits(api_key)

    round_a, ledger_a = run_round(api_key, items)
    round_b, ledger_b = run_round(api_key, items)
    keys_a = {row["item_id"]: row["teacher_probs"] for row in round_a}
    keys_b = {row["item_id"]: row["teacher_probs"] for row in round_b}
    frozen_probs = {str(item["item_id"]): frozen[str(item["item_id"])]["teacher_probs"]
                    for item in items if str(item["item_id"]) in frozen}
    common_frozen = [str(item["item_id"]) for item in items if str(item["item_id"]) in frozen]
    order = [str(item["item_id"]) for item in items]
    acc = {}
    for label, probs in (("round_a", keys_a), ("round_b", keys_b)):
        correct = sum(int(is_correct(item, probs[str(item["item_id"])])) for item in items)
        acc[label] = {"items": len(items), "correct": correct, "accuracy": correct / len(items)}
    comparisons: dict[str, Any] = {"a_vs_b": compare_scorings([keys_a[i] for i in order], [keys_b[i] for i in order])}
    if common_frozen:
        comparisons["frozen_vs_a"] = compare_scorings([frozen_probs[i] for i in common_frozen],
                                                      [keys_a[i] for i in common_frozen])
        comparisons["frozen_vs_b"] = compare_scorings([frozen_probs[i] for i in common_frozen],
                                                      [keys_b[i] for i in common_frozen])
    label_rows = [*round_a, *round_b]
    labels_path = out / "parity_labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as handle:
        for row in label_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    result = {
        "task": "D1 parity",
        "model": round_a[0]["teacher"],
        "git_commit": git_commit(REPO),
        "started_utc": started,
        "finished_utc": utcnow(),
        "sample": {"set": "panel", "items": len(items), "selection": "first N by sorted item_id"},
        "frozen_labels_present": len(common_frozen),
        "comparisons": comparisons,
        "accuracy": acc,
        "rounds": {"round_a": ledger_a, "round_b": ledger_b},
        "balance_before": balance,
        "labels_sha256": file_sha256(labels_path),
        "note": "argmax_agree is the number of items whose modal option agrees; max_abs_diff and mean_abs_diff are "
                "over all option probabilities. A teacher that is deterministic under our call would show 0.0.",
    }
    write_json(out / "parity.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
