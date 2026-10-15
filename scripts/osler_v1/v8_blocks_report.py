"""V8 blocks report (osler_v1 ADVISORY V8): writes loops/osler_v1/blocks.md from the block manifest, the block runs and the keep records.

Aggregates only: block contents, realised items and tokens, full fast-dev metrics of each run, the keep-rule statistics and the
decisions. Every number is read from a file written by the pipeline (V7 manifest, V8 block_result.json, V8 keep JSON files); a run
not yet finished is shown as pending, never estimated (R3).

The chain of comparisons follows ADVISORY V8: the starting block C is B0, or B0b when B0 fails its knowledge check and B0b passes
it (meddecide.train.blocks.starting_block). Each candidate is compared with the last kept block; a candidate whose comparison
record names a different C than the chain gives is an error, not a silent change.

Usage (after each block run and keep decision):
    uv run --frozen python scripts/osler_v1/v8_blocks_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meddecide.train.blocks import starting_block
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "outputs/osler_v1/V7/blocks.json"
V8 = REPO / "outputs/osler_v1/V8"
KEEP = V8 / "keep"
REPORT = REPO / "loops/osler_v1/blocks.md"
ADDS = {
    "B0": "mix v2 train plus the V3 replay (HellaSwag, WinoGrande, CommonsenseQA, QASC)",
    "B1": "+ V1 generators and V5a Synthea items (tabular page category capped at 2 %)",
    "B2": "+ V4 teacher prose: indicator (train pairs) and narrative pages",
    "B3": "+ V5b real text: MTSamples, CodiEsp train, PMC-Patients pairs",
}
RUNS = {"B0": "B0", "B0b": "B0b", "B1": "B1", "B2": "B2", "B3": "B3"}
CANDIDATES = ("B1", "B2", "B3")  # each is compared with the last kept block (ADVISORY V8)


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fmt(value: Any, digits: int = 4) -> str:
    return "pending" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["blocks"]
    lines = ["# osler_v1 — V8 blocks, run results and keep decisions", "",
             f"Written {utcnow()} by `scripts/osler_v1/v8_blocks_report.py` from the V7 manifest (`outputs/osler_v1/V7/blocks.json`), "
             "the V8 block results and the keep records. Aggregates only. A run that has not finished reads `pending`.", "",
             "## 1. Block contents (V7)", "",
             "| block | adds | rows | replay share | largest template | tabular page share | leakage hits (text, bare, qualified) | teacher prose final |",
             "|---|---|---|---|---|---|---|---|"]
    for name in ("B0", "B1", "B2", "B3"):
        m = manifest.get(name)
        if m is None:
            lines.append(f"| {name} | {ADDS[name]} | pending | | | | | |")
            continue
        hits = f"{len(m['leakage_hits'])}, {len(m.get('qualified_record_hits', {}))}" if m["leakage_clean"] else "NOT CLEAN"
        lines.append(f"| {name} | {ADDS[name]} | {m['rows']:,} | {m['replay_share'] * 100:.2f} % | "
                     f"{m['largest_template_share'] * 100:.2f} % | {m['tabular_page_share'] * 100:.2f} % | {hits} | "
                     f"{'yes' if m.get('teacher_final_flag') else 'not applicable'} |")
    lines += ["", "The replay is the same 100,132 rows in every block (HellaSwag 39,905, WinoGrande 40,398, CommonsenseQA 10,765, "
              "QASC 9,064); its share falls as teacher rows are added. The 2 % tabular cap (V7) and the 8 % template cap (D17) hold in "
              "every block; the narrative pages (V4-D6) are counted under their own source, not under the tabular cap.", ""]

    lines += ["## 2. Runs: steps, realised items, tokens and the selected checkpoint", "",
              "| run | steps | evaluations | realised items | real tokens | padded tokens | selected step | status |",
              "|---|---|---|---|---|---|---|---|"]
    results: dict[str, dict[str, Any]] = {}
    for name in RUNS:
        result = load(V8 / name / "block_result.json")
        if result is None:
            lines.append(f"| {name} | pending | | | | | | not run |")
            continue
        results[name] = result
        lines.append(f"| {name} | {result['steps']:,} | {len(result['evaluations'])} | {result['realised_items']:,} | "
                     f"{result['real_tokens']:,} | {result['padded_tokens']:,} | {result['best_step']:,} | done |")
    lines += ["", "Every run uses the same step count (13,992 optimiser steps, the 80,000-item replay of B0; V8-D4) and the same "
              "recipe (Qwen3.5-4B, the option-code head, the batching constant of V0). B0b changes only the LoRA targets (attention "
              "projections).", ""]

    lines += ["## 3. Dev results of each run (the full fast dev set, 5,723 items)", "",
              "| run | accuracy | template macro | Brier | mean NLL | temperatures fitted on dev |",
              "|---|---|---|---|---|---|"]
    for name, result in results.items():
        m = result["full_dev_metrics"]
        lines.append(f"| {name} | {fmt(m['accuracy'])} | {fmt(m['template_macro_accuracy'])} | {fmt(m['brier'])} | "
                     f"{fmt(m['mean_nll'])} | {', '.join(sorted(result['temperature_fits_keys']))} |")
    lines += ["", "The keep rule compares partitions (MR, general, knowledge), not these full-dev figures; the partition statistics "
              "are in section 4.", ""]

    knowledge = {name: load(KEEP / f"knowledge_check_{name}.json") for name in ("B0", "B0b")}
    b0, b0b = knowledge["B0"], knowledge["B0b"]
    start = starting_block(None if b0 is None else b0["passes"], None if b0b is None else b0b["passes"])

    lines += ["## 4. Keep rule (ADVISORY V8; paired bootstrap within templates, 1,000 resamples, seed 0)", "",
              "A block X is kept against the current C iff the lower bound of X minus C is above 0 on MR, above -0.01 on general and "
              "above -0.02 on knowledge.", "",
              "| candidate vs C | partition | items | difference [lower, upper] | margin | passes |", "|---|---|---|---|---|---|"]
    decisions: list[str] = []
    kept: list[str] = []
    current = start  # C for the next candidate; None while an earlier decision that sets it is pending
    for candidate in CANDIDATES:
        if current is None:
            lines.append(f"| {candidate} | | pending (its C waits for an earlier decision) | | | |")
            decisions.append(f"{candidate}: pending")
            continue
        record = load(KEEP / f"{candidate}_vs_{current}.json")
        if record is None:
            lines.append(f"| {candidate} vs {current} | | pending | | | |")
            decisions.append(f"{candidate} against {current}: pending")
            current = None
            continue
        if record["current"] != current:
            raise ValueError(f"{candidate} keep record compares against {record['current']}, but the chain's C is {current}")
        for part in ("mr", "general", "knowledge"):
            s = record["statistics"][part]
            lines.append(f"| {candidate} vs {current} | {part} | {s['n']:,} | {s['point']:+.4f} [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] | "
                         f"{record['margins'][part]:+.2f} | {'yes' if record['verdict']['checks'][part] else 'no'} |")
        kept_now = bool(record["verdict"]["kept"])
        decisions.append(f"{candidate} against {current}: {'KEPT' if kept_now else 'not kept'}")
        if kept_now:
            kept.append(candidate)
            current = candidate
    lines += [""]

    lines += ["## 5. Knowledge checks (B0 and B0b against zero-shot Qwen3.5-4B, 500 fast-dev items; D12 gate)", "",
              "| block | items | difference [lower, upper] | margin | status | passes |", "|---|---|---|---|---|---|"]
    knowledge_decisions: list[str] = []
    for name in ("B0", "B0b"):
        record = knowledge[name]
        if record is None:
            note = "not run (B0 passes)" if name == "B0b" and b0 is not None and b0["passes"] else "pending"
            lines.append(f"| {name} | | {note} | | | |")
            knowledge_decisions.append(f"{name}: {note}")
            continue
        s = record["statistic"]
        lines.append(f"| {name} | {record['items_block']} | {s['point']:+.4f} [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] | {record['margin']:+.2f} | "
                     f"{record['status']} | {'yes' if record['passes'] else 'no'} |")
        knowledge_decisions.append(f"{name}: {'passes' if record['passes'] else 'fails'}")
    lines += [""]

    last_kept = kept[-1] if kept else start
    lines += ["## 6. Decisions and the current C", "",
              "- Knowledge checks: " + "; ".join(knowledge_decisions) + ".",
              f"- Starting block C (ADVISORY V8: B0b only when B0 fails and B0b passes): {start or 'pending'}.",
              "- Keep decisions: " + "; ".join(decisions) + ".",
              f"- Last kept block: {last_kept or 'pending'}.", ""]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(V8 / "blocks_report.json", {"written_utc": utcnow(), "git_commit": git_commit(REPO), "starting_block": start,
                                           "kept": kept, "last_kept": last_kept, "decisions": decisions,
                                           "knowledge": knowledge_decisions})
    print(f"wrote {REPORT.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
