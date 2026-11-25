"""D1 self-audit (R6): re-derive every headline number from the raw artifacts in a fresh process.

Reads ``labels.jsonl``, ``failures.jsonl``, ``manifest.json`` and ``ceiling.json`` and the frozen sets,
recomputes the counts, the per-set accuracy (micro, template macro, majority baseline), the cost and
token totals, and the refusal counts from scratch, then compares each with what the manifest and the
ceiling claim. It also checks the label rows themselves (schema, key order, probability sums, no text
fields) and that every sendable frozen item is either labelled or failed — never silently missing.

Writes ``audit.json``; exits non-zero when any check fails. This is the recompute command behind the
D1 CLAIMS rows.

Usage (CPU):
    uv run --frozen python scripts/distill_v0/d1_rederive.py --out outputs/distill_v0/D1
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.distill.d1_teacher import file_sha256, hosted_guard, request_for_item
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
Y9 = REPO / "data" / "bench" / "osler_v4" / "y9"
FROZEN_SETS = ("knowledge", "panel", "mr_verdict", "mednli_test", "mr_reported", "mr_heldout_test",
               "mimic_suite", "pairs_test", "mimic_heldout")
ROW_FIELDS = {"item_id", "template_id", "qtype", "option_keys", "teacher_probs", "teacher", "cost", "input_tokens"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="outputs/distill_v0/D1")
    args = parser.parse_args()
    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    labels_path = out / "labels.jsonl"
    failures_path = out / "failures.jsonl"
    manifest = read_json(out / "manifest.json")
    allow_protected = bool(manifest.get("allow_protected"))
    ceiling = read_json(out / "ceiling.json")
    labels = {str(row["item_id"]): row for row in read_jsonl(labels_path)}
    failures = {str(row["item_id"]): row for row in read_jsonl(failures_path)}
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # --- artifacts and row schema -------------------------------------------------------------------------------------
    check("labels_sha256", file_sha256(labels_path) == manifest["labels_sha256"],
          {"recorded": manifest["labels_sha256"], "recomputed": file_sha256(labels_path)})
    check("failures_sha256", file_sha256(failures_path) == manifest["failures_sha256"],
          {"recorded": manifest["failures_sha256"], "recomputed": file_sha256(failures_path)})
    bad_fields = [item_id for item_id, row in labels.items() if set(row) != ROW_FIELDS]
    check("label_row_schema", not bad_fields, f"{len(bad_fields)} rows differ from the recorded schema")
    bad_teacher = [item_id for item_id, row in labels.items() if row["teacher"] != manifest["model"]]
    check("label_teacher_id", not bad_teacher, f"{len(bad_teacher)} rows carry another teacher id")
    bad_probs = []
    for item_id, row in labels.items():
        probs = row["teacher_probs"]
        if (len(probs) != len(row["option_keys"]) or any(not math.isfinite(p) or p < 0 for p in probs)
                or abs(sum(probs) - 1.0) > 1e-6):
            bad_probs.append(item_id)
    check("label_probability_sums", not bad_probs, f"{len(bad_probs)} rows are not distributions")
    duplicates = len(read_jsonl(labels_path)) - len(labels)
    check("no_duplicate_label_rows", duplicates == 0, f"{duplicates} duplicate item ids")

    # --- frozen sets, refusals and coverage ---------------------------------------------------------------------------
    sendable: dict[str, dict[str, Any]] = {}
    refused: dict[str, Counter[str]] = {}
    unrequestable: dict[str, Counter[str]] = {}
    for name in FROZEN_SETS:
        items = read_jsonl(Y9 / f"{name}.jsonl")
        refused[name] = Counter()
        unrequestable[name] = Counter()
        for item in items:
            allowed, why = hosted_guard(item, allow_protected=allow_protected)
            if not allowed:
                refused[name][str(why)] += 1
                continue
            request, reason = request_for_item(item)
            if request is None:
                unrequestable[name][str(reason)] += 1
                continue
            sendable[str(item["item_id"])] = (name, item)
    recomputed_refusals = {name: dict(sorted(counter.items())) for name, counter in refused.items()}
    recorded_refusals = {name: info["refused"] for name, info in manifest["sets"].items()}
    check("refusal_counts", recomputed_refusals == recorded_refusals,
          {"recomputed": recomputed_refusals, "recorded": recorded_refusals})
    protected_sets = {"mimic_suite", "pairs_test", "mimic_heldout", "mednli_test"}
    protected_labelled = sorted(item_id for item_id, (name, _) in sendable.items()
                                if name in protected_sets and item_id in labels)
    check("protected_rows_only_under_authorization", allow_protected or not protected_labelled,
          {"allow_protected": allow_protected, "protected_labelled": len(protected_labelled)})
    missing = sorted(item_id for item_id in sendable if item_id not in labels and item_id not in failures)
    check("every_sendable_item_labelled_or_failed", not missing, f"{len(missing)} sendable items have neither row")
    stray = sorted(item_id for item_id in labels if item_id not in sendable)
    check("no_label_for_unsendable_item", not stray, f"{len(stray)} label rows are outside the sendable set")
    mismatch = []
    for item_id, row in labels.items():
        if item_id not in sendable:
            continue
        _, item = sendable[item_id]
        if list(row["option_keys"]) != [str(option["key"]) for option in item["options"]]:
            mismatch.append(item_id)
    check("label_option_order", not mismatch, f"{len(mismatch)} rows disagree with the item's option order")

    # --- cost and tokens ----------------------------------------------------------------------------------------------
    cost = sum(float(row["cost"]) for row in labels.values())
    tokens = sum(int(row["input_tokens"]) for row in labels.values())
    check("cost_sum", abs(cost - float(manifest["ledger"]["spent_usd"])) <= 1e-6,
          {"recomputed": round(cost, 9), "ledger": manifest["ledger"]["spent_usd"]})
    check("token_sum", tokens == int(manifest["ledger"]["input_tokens"]),
          {"recomputed": tokens, "ledger": manifest["ledger"]["input_tokens"]})
    check("calls_count", len(labels) == int(manifest["ledger"]["calls"]),
          {"labelled": len(labels), "calls": manifest["ledger"]["calls"]})
    check("cap_respected", cost <= float(manifest["cap_usd"]), {"cost": round(cost, 6), "cap": manifest["cap_usd"]})

    # --- headline accuracies, recomputed independently ----------------------------------------------------------------
    per_set: dict[str, Any] = {}
    for item_id, (name, item) in sendable.items():
        bucket = per_set.setdefault(name, {"items": 0, "labelled": 0, "correct": 0, "gold": Counter(), "templates": {}})
        bucket["items"] += 1
        row = labels.get(item_id)
        if row is None:
            continue
        probs = row["teacher_probs"]
        predicted = row["option_keys"][max(range(len(probs)), key=probs.__getitem__)]
        correct = int(predicted == str(item["gold"]))
        bucket["labelled"] += 1
        bucket["correct"] += correct
        bucket["gold"][str(item["gold"])] += 1
        template = bucket["templates"].setdefault(str(item.get("template_id") or ""), [0, 0])
        template[0] += 1
        template[1] += correct
    recomputed: dict[str, Any] = {}
    accuracy_ok = True
    for name, bucket in sorted(per_set.items()):
        labelled = bucket["labelled"]
        templates = bucket["templates"]
        micro = bucket["correct"] / labelled if labelled else None
        macro = (sum(correct / items for items, correct in templates.values()) / len(templates)) if templates else None
        majority = max(bucket["gold"].values()) / labelled if labelled else None
        recorded = ceiling.get(name, {})
        same = (recorded.get("labelled") == labelled and recorded.get("correct") == bucket["correct"]
                and abs(float(recorded.get("accuracy_micro") or 0) - (micro or 0)) <= 1e-12
                and abs(float(recorded.get("accuracy_template_macro") or 0) - (macro or 0)) <= 1e-12)
        accuracy_ok &= same
        recomputed[name] = {"items": bucket["items"], "labelled": labelled, "correct": bucket["correct"],
                            "accuracy_micro": micro, "accuracy_template_macro": macro, "majority_baseline": majority,
                            "matches_recorded": same}
    check("per_set_accuracy", accuracy_ok, recomputed)
    check("counts", len(labels) + len(failures) == int(manifest["counts"]["planned_sendable"]),
          {"labels": len(labels), "failures": len(failures), "planned": manifest["counts"]["planned_sendable"]})
    check("manifest_counts", manifest["counts"]["frozen_rows"] == len(labels)
          and manifest["counts"]["failed_rows"] == len(failures),
          {"frozen_rows": manifest["counts"]["frozen_rows"], "labels": len(labels),
           "failed_rows": manifest["counts"]["failed_rows"], "failures": len(failures)})

    # --- parity recomputation (same command: the parity claims are checked here too) --------------------------------
    parity_path = out / "parity.json"
    parity_labels_path = out / "parity_labels.jsonl"
    if parity_path.exists() and parity_labels_path.exists():
        parity = read_json(parity_path)
        rows = read_jsonl(parity_labels_path)
        sample = int(parity["sample"]["items"])
        round_a = {str(row["item_id"]): row["teacher_probs"] for row in rows[:sample]}
        round_b = {str(row["item_id"]): row["teacher_probs"] for row in rows[sample:]}
        order = [str(row["item_id"]) for row in rows[:sample]]

        def agree_maxdiff(left: dict[str, list[float]], right: dict[str, list[float]],
                          ids: list[str]) -> tuple[int, float]:
            agree = 0
            max_diff = 0.0
            for item_id in ids:
                first, second = left[item_id], right[item_id]
                agree += int(first.index(max(first)) == second.index(max(second)))
                max_diff = max(max_diff, max(abs(x - y) for x, y in zip(first, second, strict=True)))
            return agree, max_diff

        agree_ab, diff_ab = agree_maxdiff(round_a, round_b, order)
        recorded = parity["comparisons"]["a_vs_b"]
        check("parity_a_vs_b", agree_ab == recorded["argmax_agree"]
              and abs(diff_ab - recorded["max_abs_diff"]) <= 1e-12,
              {"recomputed_agree": agree_ab, "recorded_agree": recorded["argmax_agree"],
               "recomputed_max_diff": diff_ab, "recorded_max_diff": recorded["max_abs_diff"]})
        correct = 0
        for item_id in order:
            item = sendable[item_id][1]
            probs = round_a[item_id]
            predicted = [str(option["key"]) for option in item["options"]][max(range(len(probs)), key=probs.__getitem__)]
            correct += int(predicted == str(item["gold"]))
        check("parity_round_a_accuracy", abs(correct / sample - parity["accuracy"]["round_a"]["accuracy"]) <= 1e-12,
              {"recomputed": correct / sample, "recorded": parity["accuracy"]["round_a"]["accuracy"]})

    audit = {"task": "D1 self-audit", "audited_utc": utcnow(), "out": str(out),
             "checks": checks, "all_passed": all(check["ok"] for check in checks), "recomputed": recomputed,
             "totals": {"labels": len(labels), "failures": len(failures), "cost_usd": round(cost, 9),
                        "input_tokens": tokens, "sendable": len(sendable),
                        "refused": sum(sum(c.values()) for c in refused.values())}}
    (out / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for check in checks:
        print(f"[{'PASS' if check['ok'] else 'FAIL'}] {check['check']}")
    print(json.dumps(audit["totals"], indent=2))
    if not audit["all_passed"]:
        raise SystemExit("D1 self-audit failed")


if __name__ == "__main__":
    main()
