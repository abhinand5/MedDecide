"""d3_distill step 3 (CPU): Gate O7 status and the additional knowledge diagnostics.

Gate O7 (PLAN.md) needs four checks. On this machine only the public test splits rebuilt by
``build_eval_sets.py`` exist, so the gate checks are reported NOT MEASURED with their missing artifacts, and the
students' zero-shot vs final knowledge on those splits is reported as an **additional** diagnostic with a paired
bootstrap. The gate verdict is never manufactured from the diagnostics (R3/R4).

Usage:
    python scripts/d3_distill/step3_gate.py --out loops/d3_distill/GATE_O7.md
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STEP2 = REPO / "outputs" / "d3_distill" / "step2"
STEP3 = REPO / "outputs" / "d3_distill" / "step3"
STUDENTS = ("mini", "flash")
SETS = ("medmcqa_val", "medqa_test", "knowledge_subsample")
# The d3_bench knowledge-subsample rows (same 1,000 items, seed 0), from outputs/d3_bench/aggregate.json.
D3_BENCH_KNOWLEDGE = {"d3_ultra": 0.805, "d1_ultra": 0.823, "pplx_flash": 0.6773547094188377, "clef_flash": 0.697}
BOOTSTRAP = 1000
SEED = 0


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def paired_bootstrap(a: list[int], b: list[int]) -> dict:
    """Mean(a - b) and its 95 % percentile interval over paired item outcomes."""
    if len(a) != len(b) or not a:
        raise ValueError("paired bootstrap needs two equal, nonempty vectors")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    rng = random.Random(SEED)
    means = []
    for _ in range(BOOTSTRAP):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    point = sum(diffs) / n
    lo = means[int(0.025 * BOOTSTRAP)]
    hi = means[int(0.975 * BOOTSTRAP) - 1]
    return {"items": n, "delta": point, "lower": lo, "upper": hi}


def score_pair(student: str, set_name: str) -> dict | None:
    zero = STEP2 / student / "zero_shot_score" / f"{set_name}.jsonl"
    final = STEP2 / student / "final_score" / f"{set_name}.jsonl"
    if not (zero.exists() and final.exists()):
        return None
    zero_rows = {r["eval_id"]: r for r in read_jsonl(zero)}
    final_rows = {r["eval_id"]: r for r in read_jsonl(final)}
    ids = sorted(set(zero_rows) & set(final_rows))
    if not ids:
        return None
    a = [int(final_rows[i]["correct"]) for i in ids]
    b = [int(zero_rows[i]["correct"]) for i in ids]
    paired = paired_bootstrap(a, b)
    paired.update({"final_accuracy": sum(a) / len(a), "zero_shot_accuracy": sum(b) / len(b),
                   "final_correct": sum(a), "zero_shot_correct": sum(b), "items": len(ids)})
    return paired


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "loops" / "d3_distill" / "GATE_O7.md")
    args = ap.parse_args()
    STEP3.mkdir(parents=True, exist_ok=True)

    missing = {
        "y9_knowledge": not (REPO / "data/bench/osler_v4/y9/knowledge.jsonl").exists(),
        "y9_panel": not (REPO / "data/bench/v0.3_ext/panel.jsonl").exists(),
        "y9_mr_verdict": not (REPO / "data/bench/osler_v4/y9/mr_verdict.jsonl").exists(),
        "osler_v4_final_rows": not (REPO / "outputs/osler_v4").exists(),
        "mimic_tree": not (REPO / "data/physionet").exists(),
    }
    gate = {
        "panel": {"status": "NOT MEASURED", "reason": "the Y9 panel frozen file is absent (data/bench/v0.3_ext/panel.jsonl)"},
        "knowledge": {"status": "NOT MEASURED", "reason": "the Y9 knowledge frozen file is absent (data/bench/osler_v4/y9/knowledge.jsonl)"},
        "mr": {"status": "NOT MEASURED", "reason": "the osler_v4 final's mr_verdict rows are absent and its weights are not on this machine"},
        "mimic": {"status": "NOT MEASURED", "reason": "no MIMIC data on this machine (operator instruction: do not touch MIMIC)"},
    }
    diagnostics = {}
    for student in STUDENTS:
        for set_name in SETS:
            result = score_pair(student, set_name)
            if result is not None:
                diagnostics[f"{student}/{set_name}"] = result
    report = {
        "gate": gate,
        "gate_verdict": "BLOCKED — NOT MEASURED on this machine",
        "missing_artifacts": missing,
        "additional_diagnostics": diagnostics,
        "note": "The diagnostics are public test splits, not the frozen Y9 sets, and they are not a gate verdict.",
        "finished_utc": utcnow(),
    }
    (STEP3 / "gate_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    lines = [
        "# d3_distill — Gate O7 status",
        "",
        f"Written `{utcnow()}` by `scripts/d3_distill/step3_gate.py`. Aggregates only.",
        "",
        "## Verdict",
        "",
        "**BLOCKED — NOT MEASURED.** None of Gate O7's four checks can be computed on this machine: the frozen Y9",
        "knowledge/panel/mr_verdict files are absent (`data/` does not exist in this checkout), the osler_v4 final's",
        "predictions are absent, and MIMIC is out of scope by the operator's instruction. No gate verdict is issued.",
        "",
        "| check | status | missing artifact |",
        "|---|---|---|",
    ]
    for name, entry in gate.items():
        lines.append(f"| {name} | {entry['status']} | {entry['reason']} |")
    lines += [
        "",
        "## Additional diagnostics (not the gate)",
        "",
        "The public test splits rebuilt from their official sources: MedMCQA validation (4,183) and MedQA-USMLE-4-options",
        "test (1,273). Zero-shot = the frozen student; final = the selected final checkpoint. Both scored through the",
        "student's own runtime; paired bootstrap over items, 1,000 resamples, seed 0.",
        "",
        "| student / set | items | zero-shot | final | delta | 95 % interval |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key, d in sorted(diagnostics.items()):
        lines.append(
            f"| {key} | {d['items']} | {d['zero_shot_accuracy']:.4f} | {d['final_accuracy']:.4f} | "
            f"{d['delta']:+.4f} | [{d['lower']:+.4f}, {d['upper']:+.4f}] |"
        )
    if not diagnostics:
        lines.append("| — | — | — | — | — | no final scores yet |")
    lines += ["", "## Additional: the exact d3_bench knowledge subsample (rebuilt, all 1,000 ids matched)", "",
              "The frozen Y9 file is absent, but the item ids in `outputs/d3_bench/subsamples.json` are deterministic",
              "functions of the public records, so the 1,000 items were rebuilt exactly (759 MedMCQA + 241 MedQA; all ids",
              "matched). This is the same item set as the d3_bench rows below, but it is a subsample and is **not** the",
              "gate (loops/d3_bench/results.md).", "",
              "| model | items | accuracy |", "|---|---:|---:|"]
    subsample = {key: d for key, d in diagnostics.items() if key.endswith("/knowledge_subsample")}
    for key, d in sorted(subsample.items()):
        lines.append(f"| {key} | {d['items']} | {d['final_accuracy']:.4f} |")
    lines.append(f"| zero-shot pplx-decider-flash-v1.2 (d3_bench row) | 998 | {D3_BENCH_KNOWLEDGE['pplx_flash']:.4f} |")
    lines.append(f"| Clef-Flash (d3_bench row) | 1,000 | {D3_BENCH_KNOWLEDGE['clef_flash']:.4f} |")
    lines.append(f"| d3-ultra teacher (d3_bench row) | 1,000 | {D3_BENCH_KNOWLEDGE['d3_ultra']:.4f} |")
    lines.append(f"| d1-ultra teacher (d3_bench row) | 1,000 | {D3_BENCH_KNOWLEDGE['d1_ultra']:.4f} |")
    lines += ["", "## Why the panel and MR subsamples are not reconstructed", "",
              "The panel builder excludes every item whose text equals a bench v0.2 item, and that manifest is absent, so",
              "the accepted 4,353-item population (and thus the 500-item subsample) cannot be rebuilt. The MR verdict",
              "subsample is drawn from the frozen osler_v1 MR suite's patient ids, which are not in this checkout.",
              "MIMIC is out of scope by the operator's instruction.", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"gate_verdict": report["gate_verdict"], "diagnostics": diagnostics}, indent=1))


if __name__ == "__main__":
    main()
