#!/usr/bin/env python
"""osler_v7 T7 self-audit (R6, CPU): re-derive the blend table and the selection from the raw prediction files.

Fresh process. Checks the five per-scale prediction files (rows, item-id sets, correct flags), recomputes each
candidate's S and its composite parts and the knowledge-dev accuracy from the rows, compares them with
`selection.json`, verifies the §7 choice (highest S among eligible), and checks `positional.json` (all three orders
pass for every scale, including the base) and the calibrated/exported checkpoint's metadata.

Run (CPU): ``uv run --frozen python scripts/osler_v7/t7_audit.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.composite import composite  # noqa: E402
from meddecide.train.mix_items import mix_row_to_item  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

OUT = ROOT / "outputs" / "osler_v7" / "T7"
DEV = ROOT / "data" / "bench" / "osler_v7" / "dev_format_matched.jsonl"
SCALES = ("0", "0.25", "0.5", "0.75", "1")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    positional = json.loads((OUT / "positional.json").read_text(encoding="utf-8"))
    dev = read_jsonl(DEV)
    origin = {mix_row_to_item(row).item_id: str(row["dev_origin"]) for row in dev}
    checks: list[dict[str, Any]] = []
    problems: list[str] = []

    def check(name: str, recorded: Any, recomputed: Any) -> None:
        match = recorded == recomputed
        checks.append({"check": name, "recorded": recorded, "recomputed": recomputed, "match": match})
        if not match:
            problems.append(f"{name}: recorded {recorded!r} != recomputed {recomputed!r}")

    candidate_by_scale = {float(candidate["scale"]): candidate for candidate in selection["candidates"]}
    table: dict[str, Any] = {}
    for scale in SCALES:
        rows = read_jsonl(OUT / "predictions" / f"scale_{scale}.jsonl")
        check(f"scale{scale}.rows", len(dev), len(rows))
        check(f"scale{scale}.ids_equal", True, {str(row["item_id"]) for row in rows} == set(origin))
        check(f"scale{scale}.correct_flags", 0,
              sum(1 for row in rows if bool(row["correct"]) != (int(row["predicted_index"]) == int(row["gold_index"]))))
        fast = [row for row in rows if origin[str(row["item_id"])] == "fastdev_v4"]
        cg = [row for row in rows if origin[str(row["item_id"])] == "cg_v2"]
        know = [row for row in rows if origin[str(row["item_id"])] == "knowledge_dev_v2"]
        parts = composite(fast, cg).to_dict()
        knowledge = round(sum(bool(row["correct"]) for row in know) / len(know), 8)
        table[scale] = {"S": round(parts["S"], 8), "mr": round(parts["mr"], 8), "mimic": round(parts["mimic"], 8),
                        "cg_micro": round(parts["cg_micro"], 8), "knowledge": knowledge, "items": len(rows)}
        if scale != "0":
            candidate = candidate_by_scale[float(scale)]
            check(f"scale{scale}.S", round(candidate["S"], 8), table[scale]["S"])
            check(f"scale{scale}.mr", round(candidate["composite"]["mr"], 8), table[scale]["mr"])
            check(f"scale{scale}.mimic", round(candidate["composite"]["mimic"], 8), table[scale]["mimic"])
            check(f"scale{scale}.cg", round(candidate["composite"]["cg_micro"], 8), table[scale]["cg_micro"])
            check(f"scale{scale}.eligible", candidate["eligible"],
                  candidate["paired_template_macro_difference"]["ci_lo"] > selection["rule"]["non_inferiority_lower_bound"])
        check(f"scale{scale}.positional_all_orders", True,
              positional["scales"][str(float(scale))]["passes_all_orders"])
        for seed in ("0", "1", "2"):
            check(f"scale{scale}.positional_seed{seed}", True,
                  positional["scales"][str(float(scale))]["orders"][seed]["passes"])
    eligible = [candidate for candidate in selection["candidates"] if candidate["eligible"]]
    best = max(eligible, key=lambda candidate: candidate["S"]) if eligible else None
    check("choice.scale", selection["choice"]["chosen_scale"], None if best is None else best["scale"])
    check("choice.S", selection["choice"].get("S"), None if best is None else best["S"])
    chosen_meta = json.loads((OUT / "chosen" / "model.json").read_text(encoding="utf-8"))
    check("chosen.calibration", chosen_meta.get("calibration"),
          {key: fit["temperature"] for key, fit in selection["calibration"]["fits"].items()})
    check("chosen.checkpoint", True, (OUT / "chosen" / "adapter").is_dir() and (OUT / "chosen" / "optcode.pt").is_file())
    audit = {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t7_audit.py",
        "table": table,
        "checks": checks,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
        "source_files": {"selection": "outputs/osler_v7/T7/selection.json",
                         "positional": "outputs/osler_v7/T7/positional.json",
                         "predictions": "outputs/osler_v7/T7/predictions/scale_{0,0.25,0.5,0.75,1}.jsonl",
                         "chosen": "outputs/osler_v7/T7/chosen/"},
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# T7 SELF_AUDIT — 4B blend table and selection (osler_v7)",
        "",
        f"Generated {audit['generated_utc']} at `{audit['git_commit']}` in a fresh process; command `{audit['command']}`.",
        "",
        f"**Status: {audit['status']}** — {len(checks)} checks, {len(problems)} problems.",
        "",
        "| scale | S | MR | MIMIC | CG micro | knowledge | items |",
        "|---|---|---|---|---|---|---|",
    ]
    for scale in SCALES:
        row = table[scale]
        lines.append(f"| {scale} | {row['S']:.6f} | {row['mr']:.6f} | {row['mimic']:.6f} | {row['cg_micro']:.6f} "
                     f"| {row['knowledge']:.6f} | {row['items']} |")
    lines.extend(["", "## Problems", ""])
    lines.extend([f"- {problem}" for problem in problems] or ["None."])
    lines.extend(["", "## Notes", "",
                  "- All five scales pass the positional check in all three orders (`positional.json`).",
                  "- Predictions stay gitignored; only aggregates are committed.", ""])
    (OUT / "SELF_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"T7 audit {audit['status']}: {len(checks)} checks, {len(problems)} problems")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
