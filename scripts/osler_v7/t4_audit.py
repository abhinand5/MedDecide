#!/usr/bin/env python
"""osler_v7 T4 self-audit (R6, CPU): re-derive the grid's metrics from the per-cell prediction files.

Fresh process. Checks the nine cells and their order, the dev file's SHA-256, every prediction file's row count and
item-id set, and recomputes the headline metrics of every cell (overall, MR family, MIMIC templates, CG and knowledge
origins, qtype and option groups, two-letter accuracy and the positional check) from the raw prediction rows.
Writes `audit.json` and `SELF_AUDIT.md` (aggregates only).

Run: ``uv run --frozen python scripts/osler_v7/t4_audit.py`` (CPU; no GPU)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.eval.composite import positional_health, template_macro_accuracy  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

OUT = ROOT / "outputs" / "osler_v7" / "T4"
DEV = ROOT / "data" / "bench" / "osler_v7" / "dev_format_matched.jsonl"
MR_FAMILY = {
    "mr_icd_support_template_v2", "mr_indicator_evidence_v1", "mr_page_category_tabular_v1",
    "mtsamples_page_category_v1", "pmc_similar_patient_v1", "codiesp_icd_support_v1", "mkurman_icd_support_v1",
}
LORA_SCALES = (0.5, 0.75, 1.0)
READOUT_SCALES = (0.0, 0.5, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def option_group(n_options: int) -> str:
    if n_options <= 5:
        return str(n_options)
    if n_options <= 9:
        return "6_9"
    if n_options == 10:
        return "10"
    return "11_49"


def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"items": len(rows),
            "accuracy_item": round(sum(bool(r["correct"]) for r in rows) / len(rows), 8),
            "accuracy_template_macro": round(template_macro_accuracy(rows), 8)}


def recompute(rows: list[dict[str, Any]], meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_qtype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_qtype[str(row["qtype"])].append(row)
        by_group[option_group(int(row["n_options"]))].append(row)
    wide = [row for row in rows if int(row["n_options"]) > 26]
    wide_templates = {str(row["template_id"]) for row in wide}
    mr = [row for row in rows if str(row["template_id"]) in MR_FAMILY]
    mimic = [row for row in rows if str(row["template_id"]).startswith("mimic_")]
    cg = [row for row in rows if meta[str(row["item_id"])]["dev_origin"] == "cg_v2"]
    knowledge = [row for row in rows if meta[str(row["item_id"])]["dev_origin"] == "knowledge_dev_v2"]
    return {
        "overall": block(rows),
        "mr_family": block(mr) if mr else {"items": 0},
        "mimic_templates": block(mimic) if mimic else {"items": 0},
        "cg_origin_micro": {"items": len(cg),
                            "accuracy_item": round(sum(bool(r["correct"]) for r in cg) / len(cg), 8)} if cg else {"items": 0},
        "knowledge_origin": {"items": len(knowledge),
                             "accuracy_item": round(sum(bool(r["correct"]) for r in knowledge) / len(knowledge), 8)}
        if knowledge else {"items": 0},
        "by_qtype": {qtype: {"items": len(subset),
                             "accuracy_item": round(sum(bool(r["correct"]) for r in subset) / len(subset), 8)}
                     for qtype, subset in sorted(by_qtype.items())},
        "by_option_group": {group: {"items": len(subset),
                                    "accuracy_item": round(sum(bool(r["correct"]) for r in subset) / len(subset), 8)}
                            for group, subset in sorted(by_group.items())},
        "two_letter": {"items": len(wide),
                       "accuracy_item": round(sum(bool(r["correct"]) for r in wide) / len(wide), 8)} if wide else {
            "items": 0},
        "positional": positional_health([r for r in rows if str(r["template_id"]) in wide_templates]).to_dict()
        if wide else {"status": "NOT MEASURED — no >26-option dev row"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    grid = json.loads((args.out_dir / "grid.json").read_text(encoding="utf-8"))
    dev_manifest = json.loads((ROOT / "outputs" / "osler_v7" / "T2" / "dev_manifest.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    problems: list[str] = []

    def check(name: str, recorded: Any, recomputed: Any) -> None:
        match = recorded == recomputed
        checks.append({"check": name, "recorded": recorded, "recomputed": recomputed, "match": match})
        if not match:
            problems.append(f"{name}: recorded {recorded!r} != recomputed {recomputed!r}")

    dev_path = ROOT / grid["dev"]["path"]
    dev_rows = read_jsonl(dev_path)
    dev_ids = {str(row["item_id"]) for row in dev_rows}
    meta = {str(row["item_id"]): {"dev_origin": str(row["dev_origin"])} for row in dev_rows}
    check("dev.sha256", dev_manifest["sha256"], sha256_file(dev_path))
    check("dev.items", grid["dev"]["items"], len(dev_rows))
    expected_cells = [(lora, readout) for lora in LORA_SCALES for readout in READOUT_SCALES]
    check("grid.cells", len(expected_cells), len(grid["grid"]))
    for index, cell in enumerate(grid["grid"]):
        lora, readout = expected_cells[index]
        check(f"cell{index}.scales", (cell["lora_scale"], cell["readout_scale"]), (lora, readout))
        label = f"lora{lora:g}_readout{readout:g}"
        check(f"cell{index}.label", cell["label"], label)
        path = args.out_dir / "predictions" / f"{label}.jsonl"
        rows = read_jsonl(path)
        check(f"cell{index}.rows", cell["overall"]["items"], len(rows))
        check(f"cell{index}.ids", dev_ids, {str(row["item_id"]) for row in rows})
        check(f"cell{index}.correct_flags", 0,
              sum(1 for row in rows if bool(row["correct"]) != (int(row["predicted_index"]) == int(row["gold_index"]))))
        recomputed = recompute(rows, meta)
        for key in ("overall", "mr_family", "mimic_templates", "cg_origin_micro", "knowledge_origin", "by_qtype",
                    "by_option_group", "two_letter", "positional"):
            check(f"cell{index}.{key}", cell[key], recomputed[key])
    checkpoint = ROOT / grid["checkpoint"]
    check("checkpoint.exists", True, (checkpoint / "model.json").is_file())
    audit = {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t4_audit.py",
        "checkpoint_model_json_sha256": sha256_file(checkpoint / "model.json") if (checkpoint / "model.json").is_file()
        else None,
        "checks": checks,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# T4 SELF_AUDIT — Lever-3 grid on the v6 final (osler_v7)",
        "",
        f"Generated {audit['generated_utc']} at `{audit['git_commit']}` in a fresh process; command `{audit['command']}`.",
        "",
        f"**Status: {audit['status']}** — {len(checks)} checks, {len(problems)} problems.",
        "",
        "| check | recorded | re-derived | match |",
        "|---|---|---|---|",
    ]
    for row in checks:
        lines.append(
            f"| {row['check']} | `{row['recorded']}` | `{row['recomputed']}` | {'yes' if row['match'] else '**NO**'} |"
        )
    lines.extend(["", "## Problems", ""])
    lines.extend([f"- {problem}" for problem in problems] or ["None."])
    lines.extend(["", "## Notes", "",
                  "- The grid is a diagnostic (ADVISORY §5); no selection or gate reads it. Per-item predictions stay",
                  "  gitignored; only aggregates are reported.", ""])
    (args.out_dir / "SELF_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"T4 audit {audit['status']}: {len(checks)} checks, {len(problems)} problems")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
