#!/usr/bin/env python
"""osler_v7 T2 self-audit (R6, CPU): re-derive the frozen dev and confirmation manifests from the raw artifacts.

Fresh process. Recomputes the file hashes and row counts, the dev shape coverage, the confirmation general
composition and its date/split rules, the audit/knowledge hashes and counts, and the overlap of both sets with the
v6 training files (one fresh pass over the mix and the anchors). Differences are problems, not warnings. Writes
`audit.json` and `SELF_AUDIT.md` (aggregates only).

Run: ``uv run --frozen python scripts/osler_v7/t2_audit.py`` (CPU; re-scans the mix, a few minutes)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

OUT = ROOT / "outputs" / "osler_v7" / "T2"
MIX = ROOT / "data" / "train" / "osler_v6" / "mix_f2_distill.jsonl"
ANCHORS = ROOT / "data" / "train" / "osler_v6" / "z1_anchor_soft.jsonl"
WINDOW_START = date(2026, 10, 6)
WINDOW_END = date(2026, 11, 12)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise(text: Any) -> str:
    return " ".join(str(text or "").split())


def options_of(row: dict[str, Any]) -> list[Any]:
    options = row["options"]
    return options if isinstance(options, list) else json.loads(options)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def training_hashes() -> dict[str, Any]:
    item_ids: set[str] = set()
    item_texts: set[bytes] = set()
    record_texts: set[bytes] = set()
    rows = {}
    for name, path in (("mix", MIX), ("anchors", ANCHORS)):
        digest = hashlib.sha256()
        n = 0
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                row = json.loads(line)
                n += 1
                item_ids.add(str(row["item_id"]))
                state = normalise(row.get("state"))
                record_texts.add(hashlib.sha256(state.encode()).digest())
                item_texts.add(hashlib.sha256(f"{state}\x1f{normalise(row.get('question'))}".encode()).digest())
        rows[name] = {"rows": n, "sha256": digest.hexdigest()}
    return {"item_ids": item_ids, "item_texts": item_texts, "record_texts": record_texts, "rows": rows}


def overlap_counts(rows: list[dict[str, Any]], train: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if str(row["item_id"]) in train["item_ids"]:
            counts["item_id"] += 1
        state = normalise(row.get("state"))
        if hashlib.sha256(f"{state}\x1f{normalise(row.get('question'))}".encode()).digest() in train["item_texts"]:
            counts["item_text"] += 1
        if hashlib.sha256(state.encode()).digest() in train["record_texts"]:
            counts["record_text"] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    dev_manifest = json.loads((args.out_dir / "dev_manifest.json").read_text(encoding="utf-8"))
    conf_manifest = json.loads((args.out_dir / "confirmation_manifest.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    problems: list[str] = []

    def check(name: str, recorded: Any, recomputed: Any) -> None:
        match = recorded == recomputed
        checks.append({"check": name, "recorded": recorded, "recomputed": recomputed, "match": match})
        if not match:
            problems.append(f"{name}: recorded {recorded!r} != recomputed {recomputed!r}")

    dev_path = ROOT / dev_manifest["path"]
    dev_rows = read_jsonl(dev_path)
    check("dev.sha256", dev_manifest["sha256"], sha256_file(dev_path))
    check("dev.rows", dev_manifest["rows"], len(dev_rows))
    check("dev.qtypes", dev_manifest["composition"]["qtypes"],
          dict(sorted(Counter(str(r["qtype"]) for r in dev_rows).items())))
    check("dev.templates", dev_manifest["composition"]["templates"],
          dict(sorted(Counter(str(r["template_id"]) for r in dev_rows).items())))
    check("dev.option_counts", dev_manifest["composition"]["option_counts"],
          dict(sorted(Counter(str(len(options_of(r))) for r in dev_rows).items(), key=lambda kv: int(kv[0]))))
    shapes = {
        "many_option_10_50": sum(1 for r in dev_rows if 10 <= len(options_of(r)) <= 50),
        "two_option_pairs": sum(1 for r in dev_rows
                                if r["qtype"] == "noul" or (r["qtype"] == "choice" and len(options_of(r)) == 2)),
        "code_items": sum(1 for r in dev_rows if r["template_id"] in {
            "code_assignment_v1", "code_family_v1", "mimic_icd_support_v1", "mimic_icd_specificity_v1",
            "mimic_pcs_support_v1", "mr_icd_support_template_v2", "fda_class_choice_v2", "fda_route_choice_v1"}),
        "mcq_4_option": sum(1 for r in dev_rows if r["qtype"] == "choice" and len(options_of(r)) == 4),
        "mcq_10_option": sum(1 for r in dev_rows if r["qtype"] == "choice" and len(options_of(r)) == 10),
        "mr_verdict_family": sum(1 for r in dev_rows if r["template_id"] in {
            "mr_icd_support_template_v2", "mr_indicator_evidence_v1", "mr_page_category_tabular_v1",
            "mtsamples_page_category_v1", "pmc_similar_patient_v1", "codiesp_icd_support_v1",
            "mkurman_icd_support_v1"}),
        "mimic_audit_templates": sum(1 for r in dev_rows if str(r["template_id"]).startswith("mimic_")),
    }
    for name, value in shapes.items():
        check(f"dev.shape.{name}", dev_manifest["composition"]["shapes"][name]["items"], value)

    conf_general_path = ROOT / conf_manifest["general"]["path"]
    general = read_jsonl(conf_general_path)
    check("general.sha256", conf_manifest["general"]["sha256"], sha256_file(conf_general_path))
    check("general.rows", conf_manifest["general"]["rows"], len(general))
    check("general.rows_before_cap", conf_manifest["general"]["build"]["composition"]["rows_before_cap"],
          sum(conf_manifest["general"]["build"]["composition"]["templates_before_cap"].values()))
    check("general.all_test", True, all(str(r.get("split")) == "test" for r in general))
    check("general.all_in_window", True, all(
        WINDOW_START <= date.fromisoformat(str(r["record_date"])[:10]) <= WINDOW_END for r in general))
    check("general.qtypes", conf_manifest["general"]["build"]["composition"]["qtypes"],
          dict(sorted(Counter(str(r["qtype"]) for r in general).items())))
    check("general.templates", conf_manifest["general"]["build"]["composition"]["templates"],
          dict(sorted(Counter(str(r["template_id"]) for r in general).items())))
    dates = sorted(str(r["record_date"])[:10] for r in general)
    strict = sum(1 for d in dates if d >= "2026-09-10")
    check("general.strict_slice_rows", conf_manifest["general"]["build"]["composition"]["strict_slice_rows"], strict)
    check("general.strict_slice_share", conf_manifest["general"]["build"]["composition"]["strict_slice_share"],
          round(strict / len(dates), 8) if dates else None)
    check("general.review_flag", conf_manifest["review_required_before_first_use"], True)

    for name, component in conf_manifest["audit"].items():
        check(f"audit.{name}.rows", component["rows"], len(read_jsonl(ROOT / component["path"])))
        check(f"audit.{name}.sha256", component["sha256"], sha256_file(ROOT / component["path"]))
        rows = read_jsonl(ROOT / component["path"])
        check(f"audit.{name}.splits", component["splits"],
              dict(sorted(Counter(str(r.get("split", "")) for r in rows).items())))
    knowledge = conf_manifest["knowledge"]
    knowledge_rows = read_jsonl(ROOT / knowledge["path"])
    check("knowledge.rows", knowledge["rows"], len(knowledge_rows))
    check("knowledge.sha256", knowledge["sha256"], sha256_file(ROOT / knowledge["path"]))

    train = training_hashes()
    check("training.mix.rows", 1_456_438, train["rows"]["mix"]["rows"])
    check("training.mix.sha256", "91ff5362d56c7b14aafa1bb62e8d78fd31312c613ff5d310892f7e09c97137ce",
          train["rows"]["mix"]["sha256"])
    check("training.anchors.rows", 385_536, train["rows"]["anchors"]["rows"])
    check("training.anchors.sha256", "edd4c2070b7d16280cf388c2634910a408d59be5b97cbf51e007900fdd494743",
          train["rows"]["anchors"]["sha256"])
    check("dev.overlap_with_training", dev_manifest["overlap_with_training"], overlap_counts(dev_rows, train))
    check("general.overlap_with_training", conf_manifest["general"]["build"]["composition"]["overlap_with_training"],
          overlap_counts(general, train))
    for name, component in conf_manifest["audit"].items():
        rows = read_jsonl(ROOT / component["path"])
        check(f"audit.{name}.overlap_with_training", component["overlap_with_training"],
              overlap_counts(rows, train))
    check("knowledge.overlap_with_training", knowledge["overlap_with_training"],
          overlap_counts(knowledge_rows, train))

    audit = {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t2_audit.py",
        "checks": checks,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# T2 SELF_AUDIT — frozen dev and confirmation sets (osler_v7)",
        "",
        f"Generated {audit['generated_utc']} at `{audit['git_commit']}` in a fresh process; command "
        f"`{audit['command']}`.",
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
                  "- The confirmation set is not used for selection; the review flag is `review_required_before_first_use`",
                  "  and T10 must not run until the operator has reviewed `confirmation_manifest.json`.",
                  "- MIMIC item rows carry no subject id; the patient-level split is cited from the osler_v4 build,",
                  "  not re-derived here. The overlap check against the v6 training files is the re-derivable part.",
                  ""])
    (args.out_dir / "SELF_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"T2 audit {audit['status']}: {len(checks)} checks, {len(problems)} problems")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
