#!/usr/bin/env python
"""osler_v7 T10 self-audit (R6, CPU): re-derive the gate from the raw prediction files in a fresh process.

Recomputes G-a/G-b/G-c for both models, checks the paired item counts against the frozen set sizes, verifies that the
candidate and base sides cover exactly the same items, and compares the result with `gate.json`.

Run (CPU): ``uv run --frozen python scripts/osler_v7/t10_audit.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.train.blocks import macro_difference, pair_by_item  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

T10 = ROOT / "outputs" / "osler_v7" / "T10"
EXPECTED = {"general": 5_988, "mr_verdict": 13_103, "mimic_suite": 4_253, "knowledge": 5_456}
THRESHOLDS = {"G-a": -0.01, "G-b": 0.0, "G-c": -0.02}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare(candidate: Path, base: Path) -> dict[str, Any]:
    cand, base_rows = read_jsonl(candidate), read_jsonl(base)
    cand_ids = {row["item_id"] for row in cand}
    base_ids = {row["item_id"] for row in base_rows}
    partial = cand_ids != base_ids
    if partial:
        cand = [row for row in cand if row["item_id"] in base_ids]
        base_rows = [row for row in base_rows if row["item_id"] in cand_ids]
    pairs = pair_by_item(cand, base_rows)
    diff = macro_difference(pairs).as_dict()
    return {"items": len(pairs), "point": round(diff["point"], 8), "ci_lo": round(diff["ci_lo"], 8),
            "ci_hi": round(diff["ci_hi"], 8), "rows_candidate": len(cand_ids), "rows_base": len(base_ids),
            "partial": partial}


def main() -> None:
    gate = json.loads((T10 / "gate.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    problems: list[str] = []

    def check(name: str, recorded: Any, recomputed: Any) -> None:
        match = recorded == recomputed
        checks.append({"check": name, "recorded": recorded, "recomputed": recomputed, "match": match})
        if not match:
            problems.append(f"{name}: recorded {recorded!r} != recomputed {recomputed!r}")

    for model, base_dir in (("4B", T10 / "baselines"), ("9B", T10)):
        prefix = f"{model.lower()}_cand"
        def cand(set_name: str, prefix: str = prefix) -> Path:
            return T10 / f"{prefix}_{set_name}" / "predictions.jsonl"
        def base(set_name: str, base_dir: Path = base_dir, model: str = model) -> Path:
            return base_dir / f"{model.lower()}_base_{set_name}" / "predictions.jsonl"

        general = compare(cand("general"), base("general"))
        check(f"{model}.G-a.items", EXPECTED["general"], general["items"])
        check(f"{model}.G-a.ci_lo", gate["models"][model]["checks"]["G-a"]["ci_lo"], general["ci_lo"])
        mr = compare(cand("mr_verdict"), base("mr_verdict"))
        mimic = compare(cand("mimic_suite"), base("mimic_suite"))
        check(f"{model}.audit.mr.items", EXPECTED["mr_verdict"], mr["items"])
        check(f"{model}.audit.mimic.items", EXPECTED["mimic_suite"], mimic["items"])
        pairs = pair_by_item(read_jsonl(cand("mr_verdict")), read_jsonl(base("mr_verdict"))) + \
            pair_by_item(read_jsonl(cand("mimic_suite")), read_jsonl(base("mimic_suite")))
        audit = macro_difference(pairs).as_dict()
        check(f"{model}.G-b.ci_lo", gate["models"][model]["checks"]["G-b"]["ci_lo"], round(audit["ci_lo"], 8))
        knowledge = compare(cand("knowledge"), base("knowledge"))
        check(f"{model}.G-c.items", gate["models"][model]["checks"]["G-c"]["items"], knowledge["items"])
        check(f"{model}.G-c.ci_lo", gate["models"][model]["checks"]["G-c"]["ci_lo"], knowledge["ci_lo"])
        partial = general["partial"] or mr["partial"] or mimic["partial"] or knowledge["partial"]
        passes = all(value > THRESHOLDS[key] for key, value in (("G-a", general["ci_lo"]), ("G-b", audit["ci_lo"]),
                                                               ("G-c", knowledge["ci_lo"])))
        verdict = "NOT MEASURED" if partial else ("PASS" if passes else "FAIL")
        check(f"{model}.verdict", gate["models"][model]["verdict"], verdict)
    audit = {
        "generated_utc": utcnow(), "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t10_audit.py",
        "checks": checks, "problems": problems, "status": "PASS" if not problems else "FAIL",
        "source_files": {"gate": "outputs/osler_v7/T10/gate.json",
                         "predictions": "outputs/osler_v7/T10/{4b,9b}_{cand,base}_*/predictions.jsonl"},
    }
    (T10 / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# T10 SELF_AUDIT — Gate v7 (osler_v7)",
        "",
        f"Generated {audit['generated_utc']} at `{audit['git_commit']}` in a fresh process; command `{audit['command']}`.",
        "",
        f"**Status: {audit['status']}** — {len(checks)} checks, {len(problems)} problems.",
        "",
        "| check | recorded | recomputed | match |",
        "|---|---|---|---|",
    ]
    for row in checks:
        lines.append(f"| {row['check']} | `{row['recorded']}` | `{row['recomputed']}` | "
                     f"{'yes' if row['match'] else '**NO**'} |")
    lines.extend(["", "## Problems", ""])
    lines.extend([f"- {problem}" for problem in problems] or ["None."])
    (T10 / "SELF_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"T10 audit {audit['status']}: {len(checks)} checks, {len(problems)} problems")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
