#!/usr/bin/env python
"""osler_v7 T10 — Gate v7 (ADVISORY §7) for the 4B and 9B chosen models against their own zero-shot bases.

G-a: confirmation-set general items, paired template-macro difference lower bound > -0.01.
G-b: audit items (MR verdict + MIMIC suite), lower bound > 0.
G-c: knowledge items, lower bound > -0.02.
Each check is computed from candidate and base prediction files paired by the trainer's item id; a missing set makes
the check NOT MEASURED and the model does not pass. Writes `gate.json` with every check's point and interval.

Run (CPU; after the scoring queue): ``uv run --frozen python scripts/osler_v7/t10_gate.py``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.train.blocks import macro_difference, pair_by_item  # noqa: E402
from meddecide.utils.io import write_json  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

T10 = ROOT / "outputs" / "osler_v7" / "T10"
SETS = {"general": "general", "mr_verdict": "mr_verdict", "mimic_suite": "mimic_suite", "knowledge": "knowledge"}
THRESHOLDS = {"G-a": -0.01, "G-b": 0.0, "G-c": -0.02}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare(candidate: Path, base: Path) -> dict[str, Any]:
    if not candidate.is_file() or not base.is_file():
        return {"status": "NOT MEASURED", "candidate": str(candidate), "base": str(base)}
    cand_rows, base_rows = read_jsonl(candidate), read_jsonl(base)
    cand_ids = {row["item_id"] for row in cand_rows}
    base_ids = {row["item_id"] for row in base_rows}
    status = "MEASURED" if cand_ids == base_ids else "PARTIAL"
    if status == "PARTIAL":
        # the zero-shot base path can refuse some items (e.g. repeated option labels); compare on the intersection and
        # keep the coverage explicit. A partial check never yields a PASS (ADVISORY §7: unmeasured does not pass).
        cand_rows = [row for row in cand_rows if row["item_id"] in base_ids]
        base_rows = [row for row in base_rows if row["item_id"] in cand_ids]
    pairs = pair_by_item(cand_rows, base_rows)
    diff = macro_difference(pairs).as_dict()
    return {"status": status, "items": diff["n"], "point": round(diff["point"], 8),
            "ci_lo": round(diff["ci_lo"], 8), "ci_hi": round(diff["ci_hi"], 8),
            "coverage": {"candidate_rows": len(cand_ids), "base_rows": len(base_ids),
                         "only_candidate": len(cand_ids - base_ids), "only_base": len(base_ids - cand_ids)}}


def gate_for(prefix: str, base_dir: Path) -> dict[str, Any]:
    def cand(set_name: str) -> Path:
        return T10 / f"{prefix}_{set_name}" / "predictions.jsonl"

    def base(set_name: str) -> Path:
        return base_dir / f"{prefix.replace('cand', 'base')}_{set_name}" / "predictions.jsonl"

    checks: dict[str, Any] = {
        "G-a": compare(cand(SETS["general"]), base(SETS["general"])),
    }
    mr = compare(cand(SETS["mr_verdict"]), base(SETS["mr_verdict"]))
    mimic = compare(cand(SETS["mimic_suite"]), base(SETS["mimic_suite"]))
    if mr["status"] == "MEASURED" and mimic["status"] == "MEASURED":
        # one paired template-macro difference over the audit items: concatenate the two independent pair sets
        pairs = pair_by_item(read_jsonl(cand(SETS["mr_verdict"])), read_jsonl(base(SETS["mr_verdict"]))) + \
            pair_by_item(read_jsonl(cand(SETS["mimic_suite"])), read_jsonl(base(SETS["mimic_suite"])))
        diff = macro_difference(pairs).as_dict()
        checks["G-b"] = {"status": "MEASURED", "items": diff["n"], "point": round(diff["point"], 8),
                         "ci_lo": round(diff["ci_lo"], 8), "ci_hi": round(diff["ci_hi"], 8),
                         "mr": mr, "mimic_suite": mimic}
    else:
        checks["G-b"] = {"status": "NOT MEASURED", "mr": mr, "mimic_suite": mimic}
    checks["G-c"] = compare(cand(SETS["knowledge"]), base(SETS["knowledge"]))
    for name, check in checks.items():
        if check["status"] in ("MEASURED", "PARTIAL"):
            check["threshold_lower_bound"] = THRESHOLDS[name]
            check["passes"] = bool(check["ci_lo"] > THRESHOLDS[name])
    verdict = "NOT MEASURED"
    if all(check["status"] == "MEASURED" for check in checks.values()):
        verdict = "PASS" if all(check["passes"] for check in checks.values()) else "FAIL"
    # partial coverage on any check keeps the gate NOT MEASURED
    return {"checks": checks, "verdict": verdict}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=T10 / "gate.json")
    args = parser.parse_args()
    result = {
        "generated_utc": utcnow(), "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t10_gate.py",
        "gate": "Gate v7 (ADVISORY §7): paired template-macro lower bounds -0.01 (general), 0 (audit), -0.02 (knowledge)",
        "sets": {"general": "data/bench/osler_v7/confirm/general.jsonl (5,988)",
                 "audit": "data/bench/osler_v4/y9/mr_verdict.jsonl (13,103) + mimic_suite.jsonl (4,253)",
                 "knowledge": "data/bench/osler_v4/y9/knowledge.jsonl (5,456)"},
        "models": {
            "4B": gate_for("4b_cand", T10 / "baselines"),
            "9B": gate_for("9b_cand", T10),
        },
    }
    write_json(args.out, result)
    print(json.dumps({model: data["verdict"] for model, data in result["models"].items()}, indent=1))
    for model, data in result["models"].items():
        print(model, {name: (check.get("ci_lo"), check.get("passes")) for name, check in data["checks"].items()})


if __name__ == "__main__":
    main()
