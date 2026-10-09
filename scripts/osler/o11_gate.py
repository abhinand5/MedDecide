"""O11: Gate O1 (PROGRAM D24) for Osler-4B and Osler-9B. CPU only; reads prediction files, runs no model.

Scopes, all taken from the O2 common set (``data/bench/v0.3_ext/o2_items.jsonl``):
  headline_seen      v0.2 fresh-tier items whose template is neither a superseded v1 template nor one of the four D14
                     held-out templates (D16, D24 (i): the headline fresh set, seen templates)
  external_panel     the external clinical panel (D24 (ii))
  heldout_templates  v0.2 fresh-tier items of the four D14 held-out templates (reported separately)
  robustness         the robustness pack (reported separately, same rule)
  knowledge          v0.2 established items from MedQA and MedMCQA (the knowledge guard's scope)

Comparisons (D24): Osler-4B against MedDecider-4B and zero-shot Qwen3.5-4B; Osler-9B against MedDecider-9B and JEV-9B; on
headline_seen and external_panel. The knowledge guard: Osler-4B against zero-shot Qwen3.5-4B, Osler-9B against zero-shot
Qwen3.5-9B, on knowledge. Accuracy is the template macro (G1's definition), Brier the item mean; both paired and bootstrapped
within templates with 1,000 resamples. A model's gate PASSes only when every primary comparison passes and the guard passes;
a missing baseline makes the verdict NOT MEASURED, and is never counted as a failure.

Osler rows are the both-order averages (the gate's input). The single-order rows are reported beside the gate, never in it.

Outputs:
  outputs/osler_v0/O11/gate.json   every number, with the item counts (gitignored)
  loops/osler_v0/gate_o1.md        the verdicts, every paired difference with its interval and item count (committed)

Usage:
    uv run --frozen python scripts/osler/o11_gate.py --osler-4b osler_4b_L --osler-9b osler_9b
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from meddecide.eval.gate import (
    KNOWLEDGE_GUARD_FLOOR,
    accuracy_difference,
    compare_rows,
    gate_verdict,
    knowledge_guard,
)
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data/bench/v0.3_ext/o2_items.jsonl"
O2_DIR = REPO / "outputs/osler_v0/O2"
O11_DIR = REPO / "outputs/osler_v0/O11"
REPORT = REPO / "loops/osler_v0/gate_o1.md"
# ADVISORY osler_v0 "Benchmark v0.2 unchanged" (held-out) and D16 (superseded v1 templates)
HELD_OUT_TEMPLATES = frozenset({"ct_phase_choice_v1", "fda_boxed_warning_noul_v1", "pubmed_humans_noul_v1",
                                "ct_arm_role_noul_v1"})
SUPERSEDED_TEMPLATES = frozenset({"pubmed_mesh_major_choice_v1", "fda_class_choice_v1"})
KNOWLEDGE_SETS = frozenset({"medqa", "medmcqa"})
SCOPES = ("headline_seen", "external_panel", "heldout_templates", "robustness", "knowledge")
PRIMARY_SCOPES = ("headline_seen", "external_panel")
BASELINES = {
    "4B": ("MedDecider-4B", "meddecider-4b", "zero-shot Qwen3.5-4B", "qwen35-4b"),
    "9B": ("MedDecider-9B", "meddecider-9b", "JEV-9B", "jev-9b"),
}
GUARD_BASELINE = {"4B": ("zero-shot Qwen3.5-4B", "qwen35-4b"), "9B": ("zero-shot Qwen3.5-9B", "qwen35-9b")}
N_RESAMPLES = 1000
SEED = 0


def scope_members(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    members: dict[str, list[dict[str, Any]]] = {s: [] for s in SCOPES}
    for row in rows:
        keep = {"item_id": row["item_id"], "template_id": row["template_id"]}
        if row["benchmark"] == "ext_panel":
            members["external_panel"].append(keep)
        elif row["benchmark"] == "robustness":
            members["robustness"].append(keep)
        elif row["benchmark"] == "v0.2":
            tier = row["meta"]["tier"]
            if tier == "fresh" and row["template_id"] in HELD_OUT_TEMPLATES:
                members["heldout_templates"].append(keep)
            elif tier == "fresh" and row["template_id"] not in SUPERSEDED_TEMPLATES:
                members["headline_seen"].append(keep)
            if tier == "established" and row["set_name"] in KNOWLEDGE_SETS:
                members["knowledge"].append(keep)
    return members


def by_item(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["item_id"]: row for row in iter_jsonl(path)}


def describe(comp: Any) -> dict[str, Any]:
    return {
        "scope": comp.comparison.scope, "baseline": comp.comparison.baseline, "n_scope": comp.n_scope,
        "n_paired": comp.n_paired, "n_baseline_skipped": comp.n_baseline_skipped,
        "n_baseline_missing": comp.n_baseline_missing, "n_osler_missing": comp.n_osler_missing,
        "accuracy": comp.comparison.accuracy.__dict__, "brier": comp.comparison.brier.__dict__,
        "passes": comp.comparison.passes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--osler-4b", required=True, help="the O11 output name of Osler-4B (the chosen arm)")
    parser.add_argument("--osler-9b", required=True, help="the O11 output name of Osler-9B")
    parser.add_argument("--items", type=Path, default=ITEMS, help=argparse.SUPPRESS)  # tests and smoke runs only
    parser.add_argument("--o2-root", type=Path, default=O2_DIR, help=argparse.SUPPRESS)
    parser.add_argument("--o11-root", type=Path, default=O11_DIR, help=argparse.SUPPRESS)
    parser.add_argument("--report", type=Path, default=REPORT, help=argparse.SUPPRESS)
    args = parser.parse_args()
    o2_dir, o11_dir, report_path = args.o2_root, args.o11_root, args.report

    rows = list(iter_jsonl(args.items))
    members = scope_members(rows)
    osler = {"4B": by_item(o11_dir / args.osler_4b / "both_orders.jsonl"),
             "9B": by_item(o11_dir / args.osler_9b / "both_orders.jsonl")}
    single = {"4B": by_item(o11_dir / args.osler_4b / "single_order.jsonl"),
              "9B": by_item(o11_dir / args.osler_9b / "single_order.jsonl")}
    scope_sizes = {s: len(members[s]) for s in SCOPES}

    results: dict[str, Any] = {"kind": "osler_v0_O11_gate_o1", "built_at_utc": utcnow(), "git_commit": git_commit(REPO),
                               "command": "scripts/osler/o11_gate.py", "scope_sizes": scope_sizes,
                               "osler": {"4B": args.osler_4b, "9B": args.osler_9b},
                               "knowledge_guard_floor": KNOWLEDGE_GUARD_FLOOR,
                               "n_resamples": N_RESAMPLES, "seed": SEED, "models": {}}
    lines: list[str] = []
    lines += ["# O11 — Gate O1 (PROGRAM D24)", "",
              "Osler rows: both-order averages for choice items; noul and score items are scored once (deviation 44).",
              "Accuracy is the template macro (G1's definition), bootstrapped within templates (1,000 resamples, seed 0);",
              "Brier is the item mean, bootstrapped the same way. An interval is shown as point [lower, upper].", ""]
    lines += ["| scope | items in scope |", "|---|---|"]
    lines += [f"| {s} | {scope_sizes[s]} |" for s in SCOPES] + [""]

    for size in ("4B", "9B"):
        guard_name, guard_key = GUARD_BASELINE[size]
        baseline_names = [(BASELINES[size][0], BASELINES[size][1]), (BASELINES[size][2], BASELINES[size][3])]
        model_record: dict[str, Any] = {"comparisons": [], "knowledge_guard": None, "single_order": [],
                                        "verdict": None, "verdict_reason": None}
        primary: list[Any] = []
        blocked_primary: list[str] = []  # missing primary comparisons or guard: the verdict is NOT MEASURED
        blocked_other: list[str] = []  # secondary scopes: reported, never decide the verdict
        lines += [f"## Osler-{size}", ""]
        lines += ["| scope | baseline | items (paired) | accuracy diff [lower, upper] | Brier diff [lower, upper] | pass | skipped / missing (baseline, Osler) |",
                  "|---|---|---|---|---|---|---|"]
        for scope in ("headline_seen", "external_panel", "heldout_templates", "robustness"):
            for name, key in baseline_names:
                base_rows = by_item(o2_dir / key / "predictions.jsonl")
                try:
                    comp = compare_rows(scope, name, members[scope], osler[size], base_rows,
                                        n_resamples=N_RESAMPLES, seed=SEED)
                except ValueError as exc:
                    (blocked_primary if scope in PRIMARY_SCOPES else blocked_other).append(str(exc))
                    lines.append(f"| {scope} | {name} | NOT MEASURED | | | | {exc} |")
                    model_record["comparisons"].append({"scope": scope, "baseline": name, "not_measured": str(exc)})
                    continue
                record = describe(comp)
                model_record["comparisons"].append(record)
                if scope in PRIMARY_SCOPES:
                    primary.append(comp.comparison)
                a, b = comp.comparison.accuracy, comp.comparison.brier
                lines.append(f"| {scope} | {name} | {comp.n_paired} | {a.point:+.4f} [{a.lo:+.4f}, {a.hi:+.4f}] | "
                             f"{b.point:+.4f} [{b.lo:+.4f}, {b.hi:+.4f}] | {'yes' if comp.comparison.passes else 'no'} | "
                             f"{comp.n_baseline_skipped} / {comp.n_baseline_missing}, {comp.n_osler_missing} |")
            lines.append("")
        # the knowledge guard: Osler minus the zero-shot base of the same size on MedQA and MedMCQA (micro accuracy)
        guard_rows = by_item(o2_dir / guard_key / "predictions.jsonl")
        try:
            guard_diff, n_guard = accuracy_difference(members["knowledge"], osler[size], guard_rows,
                                                      n_resamples=N_RESAMPLES, seed=SEED)
            guard_ok = knowledge_guard(guard_diff)
            model_record["knowledge_guard"] = {"baseline": guard_name, "n_paired": n_guard,
                                               "accuracy": guard_diff.__dict__, "passes": guard_ok}
            lines += [f"Knowledge guard (MedQA + MedMCQA, micro accuracy, vs {guard_name}): {n_guard} paired items, "
                      f"{guard_diff.point:+.4f} [{guard_diff.lo:+.4f}, {guard_diff.hi:+.4f}]; "
                      f"guard {'passes' if guard_ok else 'fails'} (lower bound must exceed {KNOWLEDGE_GUARD_FLOOR:+.2f}).", ""]
        except ValueError as exc:
            guard_ok = None
            blocked_primary.append(f"knowledge guard vs {guard_name}: {exc}")
            model_record["knowledge_guard"] = {"baseline": guard_name, "not_measured": str(exc)}
            lines += [f"Knowledge guard vs {guard_name}: NOT MEASURED ({exc}).", ""]

        # the single-order rows, reported beside the gate and not in it
        for scope in PRIMARY_SCOPES:
            for name, key in baseline_names:
                try:
                    comp = compare_rows(scope, name, members[scope], single[size], by_item(o2_dir / key / "predictions.jsonl"),
                                        n_resamples=N_RESAMPLES, seed=SEED)
                except ValueError as exc:
                    model_record["single_order"].append({"scope": scope, "baseline": name, "not_measured": str(exc)})
                    continue
                model_record["single_order"].append(describe(comp))

        if blocked_other:
            model_record["not_measured_secondary"] = blocked_other
        if blocked_primary:
            verdict, reason = "NOT MEASURED", "; ".join(blocked_primary)
        else:
            passed = gate_verdict(primary) and guard_ok is True
            verdict = "PASS" if passed else "FAIL"
            reason = ("every primary comparison passes and the knowledge guard passes" if passed
                      else "a primary comparison or the knowledge guard does not pass")
        model_record["verdict"] = verdict
        model_record["verdict_reason"] = reason
        results["models"][f"osler_{size}"] = model_record
        lines += [f"**Osler-{size} verdict: {verdict}.** {reason}.", ""]

    write_json(o11_dir / "gate.json", results)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for size in ("4B", "9B"):
        print(f"Osler-{size}: {results['models'][f'osler_{size}']['verdict']}", flush=True)
    print(f"wrote {report_path} and {o11_dir / 'gate.json'}", flush=True)


if __name__ == "__main__":
    main()
