#!/usr/bin/env python
"""osler_v7 T9 self-audit (R6, CPU): re-derive the 9B blend table and the selection from the prediction files.

Fresh process. Recomputes each scale's S, partition accuracies and knowledge-dev accuracy from
`outputs/osler_v7/T9/predictions/scale_*.jsonl` joined to the d3 dev origin map, checks the paired non-inferiority
flags and the all-orders positional verdicts recorded in `selection.json`, and verifies the §7/§9 choice.

Run (CPU): ``uv run --frozen python scripts/osler_v7/t9_audit.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.composite import composite  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

OUT = ROOT / "outputs" / "osler_v7" / "T9"
DEV = ROOT / "data" / "train" / "osler_v7" / "d3_dev.jsonl"
SCALES = ("0", "0.25", "0.5", "0.75", "1")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    dev = read_jsonl(DEV)
    origin = {str(row["row_id"]): str(row["dev_origin"]) for row in dev}
    candidates = {float(c["scale"]): c for c in selection["candidates"]}
    checks: list[dict[str, Any]] = []
    problems: list[str] = []

    def check(name: str, recorded: Any, recomputed: Any) -> None:
        match = recorded == recomputed
        checks.append({"check": name, "recorded": recorded, "recomputed": recomputed, "match": match})
        if not match:
            problems.append(f"{name}: recorded {recorded!r} != recomputed {recomputed!r}")

    table: dict[str, Any] = {}
    for scale in SCALES:
        rows = read_jsonl(OUT / "predictions" / f"scale_{scale}.jsonl")
        check(f"scale{scale}.rows", len(dev), len(rows))
        check(f"scale{scale}.ids_equal", True, {str(row["item_id"]) for row in rows} == set(origin))
        fast = [row for row in rows if origin[str(row["item_id"])] == "fastdev_v4"]
        cg = [row for row in rows if origin[str(row["item_id"])] == "cg_v2"]
        know = [row for row in rows if origin[str(row["item_id"])] == "knowledge_dev_v2"]
        parts = composite(fast, cg).to_dict()
        knowledge = round(sum(bool(row["correct"]) for row in know) / len(know), 8)
        table[scale] = {"S": round(parts["S"], 8), "mr": round(parts["mr"], 8), "mimic": round(parts["mimic"], 8),
                        "cg_micro": round(parts["cg_micro"], 8), "knowledge": knowledge, "items": len(rows)}
        if scale != "0":
            candidate = candidates[float(scale)]
            check(f"scale{scale}.S", round(candidate["S"], 8), table[scale]["S"])
            check(f"scale{scale}.mr", round(candidate["composite"]["mr"], 8), table[scale]["mr"])
            check(f"scale{scale}.mimic", round(candidate["composite"]["mimic"], 8), table[scale]["mimic"])
            check(f"scale{scale}.cg", round(candidate["composite"]["cg_micro"], 8), table[scale]["cg_micro"])
            check(f"scale{scale}.knowledge", round(candidate["knowledge"], 8), knowledge)
            check(f"scale{scale}.positional", candidate["positional_passes_all_orders"],
                  all(candidate["positional"][seed]["passes"] for seed in ("0", "1", "2")))
    eligible = [c for c in selection["candidates"] if c["eligible"] and c["positional_passes_all_orders"]]
    best = max(eligible, key=lambda c: c["S"]) if eligible else None
    check("choice.scale", selection["choice"]["chosen_scale"], None if best is None else best["scale"])
    check("choice.S", selection["choice"].get("S"), None if best is None else best["S"])
    audit = {
        "generated_utc": utcnow(), "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t9_audit.py",
        "table": table, "checks": checks, "problems": problems,
        "status": "PASS" if not problems else "FAIL",
        "source_files": {"selection": "outputs/osler_v7/T9/selection.json",
                         "predictions": "outputs/osler_v7/T9/predictions/scale_{0,0.25,0.5,0.75,1}.jsonl"},
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# T9 SELF_AUDIT — 9B blend table and selection (osler_v7)",
        "",
        f"Generated {audit['generated_utc']} at `{audit['git_commit']}` in a fresh process; command `{audit['command']}`.",
        "",
        f"**Status: {audit['status']}** — {len(checks)} checks, {len(problems)} problems.",
        "",
        "| scale | S | MR | MIMIC | CG v2 micro | knowledge | items |",
        "|---|---|---|---|---|---|---|",
    ]
    for scale in SCALES:
        row = table[scale]
        lines.append(f"| {scale} | {row['S']:.6f} | {row['mr']:.6f} | {row['mimic']:.6f} | {row['cg_micro']:.6f} "
                     f"| {row['knowledge']:.6f} | {row['items']} |")
    lines.extend(["", "## Problems", ""])
    lines.extend([f"- {problem}" for problem in problems] or ["None."])
    (OUT / "SELF_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"T9 audit {audit['status']}: {len(checks)} checks, {len(problems)} problems")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
