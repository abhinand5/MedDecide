"""osler_v6 (ADVISORY §2.2-2.5): the F2 training mix with the Z1 teacher distributions attached, built once and reused.

The mix is osler_v5's corrected F2 mix, composed by osler_v5's own driver code (osler_v4's Y8 composer, then the D56a
drop of `fda_route_choice_v1`, then the D52a option augmentation at probability 1.0, seed 0) — this script only imports
`scripts/osler_v5/run_arm.py` and calls its `compose_rows`, so the mix is the same rows osler_v5 F2 trained on. The
corrected composition hash is checked against the recorded `CORRECTED_COMPOSITION_SHA256` before anything else.

Then the Z1 soft labels are attached to rows by normalised text (`meddecide.distill.soft_labels.attach_teacher`: anchor
templates only, option keys the same and in the same order) and every status is counted. The output is private training
data under `data/train/osler_v6/` (gitignored); the manifest holds aggregates only.

Commands:
  build   compose + augment + attach; write `data/train/osler_v6/mix_f2_distill.jsonl` and its manifest
  check   re-derive the manifest's counts from the written file (exit 1 on a mismatch)
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.distill import soft_labels as sl
from meddecide.train.corrected_mix import CORRECTED_COMPOSITION_SHA256
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import utcnow

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "osler_v5" / "arm_F2.yaml"
SOFT = REPO_ROOT / "data" / "train" / "osler_v6" / "z1_anchor_soft.jsonl"
DATA_DIR = REPO_ROOT / "data" / "train" / "osler_v6"
OUT_DIR = REPO_ROOT / "outputs" / "osler_v6" / "Z2"
EXPECTED_COMPOSITION = CORRECTED_COMPOSITION_SHA256


def paths(seed: int) -> tuple[Path, Path]:
    """The mix file and its manifest; the seed names the D52a option-augmentation run (F2 seed 0, F2s1 seed 1)."""
    suffix = "" if seed == 0 else f"_s{seed}"
    return DATA_DIR / f"mix_f2_distill{suffix}.jsonl", OUT_DIR / f"mix_manifest{suffix}.json"


def load_module(name: str, path: Path) -> Any:
    """Load a script by path, registering it (its dataclasses resolve their module through sys.modules)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def teacher_map() -> dict[str, tuple[list[str], list[float]]]:
    """The Z1 soft labels keyed by normalised text: option keys and the teacher's distribution in that order."""
    if not SOFT.exists():
        raise SystemExit(f"{SOFT} is missing: run Z1's join first")
    soft: dict[str, tuple[list[str], list[float]]] = {}
    for row in iter_jsonl(SOFT):
        key = sl.overlap_keys(row)[0]
        if key in soft:
            raise SystemExit("two soft-label rows share a text key: the Z1 pool should hold one row per text: stop")
        soft[key] = ([str(option["key"]) for option in row["options"]], [float(p) for p in row["teacher_probs"]])
    return soft


def cmd_build(args: argparse.Namespace) -> None:
    from meddecide.train.config import load_config

    arm = load_module("osler_v5_run_arm", REPO_ROOT / "scripts" / "osler_v5" / "run_arm.py")
    config = dataclasses.replace(load_config(CONFIG), seed=int(args.seed))
    out, manifest_path = paths(int(args.seed))
    out_dir = OUT_DIR / "mix" / f"seed{int(args.seed)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, mix_info = arm.compose_rows(config, out_dir)
    if mix_info["composition_sha256"] != EXPECTED_COMPOSITION:
        raise SystemExit(
            f"composed mix hash {mix_info['composition_sha256']} is not the recorded {EXPECTED_COMPOSITION}: stop"
        )
    soft = teacher_map()
    statuses: Counter[str] = Counter()
    attached_by_qtype: Counter[str] = Counter()
    used: set[str] = set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            out_row, status = sl.attach_teacher(row, soft)
            statuses[status] += 1
            if status == "anchor: attached":
                attached_by_qtype[str(row["qtype"])] += 1
                used.add(sl.overlap_keys(row)[0])
            handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    total = sum(statuses.values())
    if total != len(rows):
        raise SystemExit(f"statuses {total} do not add up to rows {len(rows)}: stop")
    anchors = sum(count for status, count in statuses.items() if status.startswith("anchor"))
    domains = sum(count for status, count in statuses.items() if status.startswith("domain"))
    if anchors + domains != total:
        raise SystemExit("anchor + domain statuses do not add up: stop")
    attached = statuses.get("anchor: attached", 0)
    manifest = {
        "config": str(CONFIG.relative_to(REPO_ROOT)),
        "teacher": sl.TEACHER,
        "soft_label_file": str(SOFT.relative_to(REPO_ROOT)),
        "soft_label_file_sha256": file_sha256(SOFT),
        "soft_label_texts": len(soft),
        "soft_label_texts_used": len(used),
        "seed": int(args.seed),
        "output": str(out.relative_to(REPO_ROOT)),
        "output_sha256": file_sha256(out),
        "rows": total,
        "anchor_rows": anchors,
        "domain_rows": domains,
        "labelled_rows": attached,
        "labelled_share_of_rows": attached / total,
        "statuses": dict(sorted(statuses.items())),
        "attached_by_qtype": dict(sorted(attached_by_qtype.items())),
        "mix": mix_info,
        "generated_utc": utcnow(),
    }
    write_json(manifest_path, manifest)
    print(f"[{utcnow()}] mix: rows {total}; anchors {anchors}; attached {attached} ({attached / total:.4f}); "
          f"statuses {dict(sorted(statuses.items()))}")


def cmd_check(args: argparse.Namespace) -> None:
    out, manifest_path = paths(int(args.seed))
    if not manifest_path.exists() or not out.exists():
        raise SystemExit(f"{manifest_path} or {out} is missing: run the build first")
    manifest = json.loads(manifest_path.read_text())
    labelled = 0
    rows = 0
    for row in iter_jsonl(out):
        rows += 1
        if row.get("teacher_probs") is not None:
            labelled += 1
    problems = []
    if rows != manifest["rows"]:
        problems.append(f"rows {rows} != {manifest['rows']}")
    if labelled != manifest["labelled_rows"]:
        problems.append(f"labelled {labelled} != {manifest['labelled_rows']}")
    if file_sha256(out) != manifest["output_sha256"]:
        problems.append("output sha256 differs from the manifest")
    if problems:
        raise SystemExit("mix check failed: " + "; ".join(problems))
    print(f"[{utcnow()}] mix check ok: rows {rows}, labelled {labelled}, sha {manifest['output_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--seed", type=int, default=0, help="the D52a augmentation seed (F2 = 0, F2s1 = 1)")
    check = sub.add_parser("check")
    check.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"build": cmd_build, "check": cmd_check}[args.command](args)


if __name__ == "__main__":
    main()
