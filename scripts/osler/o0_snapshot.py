"""O0 snapshot: commit, benchmark and training-mix hashes, environment versions (AGENTS.md R7).

Every hash is recomputed here from the files on disk and compared with the value the builder
recorded in its manifest; a mismatch is reported, never silently accepted.

Usage:
    source scripts/pod_env.sh && uv run python scripts/osler/o0_snapshot.py
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from meddecide.bench.denominators import count_test_denominator
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, gpu_name, utcnow

REPO = Path(__file__).resolve().parents[2]


def _git(args: list[str]) -> str:
    out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _du_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0])


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            n += chunk.count(b"\n")
    return n


def _hash_entry(path: Path, expected: str | None) -> dict[str, Any]:
    actual = file_sha256(path)
    return {
        "path": str(path.relative_to(REPO)),
        "bytes": path.stat().st_size,
        "sha256": actual,
        "recorded_sha256": expected,
        "match": None if expected is None else actual == expected,
    }


def bench_block() -> dict[str, Any]:
    v02 = REPO / "data/bench/v0.2"
    manifest_path = v02 / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files: dict[str, Any] = {}
    mismatches: list[str] = []
    for name, rec in sorted(manifest["files"].items()):
        entry = _hash_entry(REPO / rec["path"], rec.get("sha256"))
        entry["n_rows_recorded"] = rec.get("n_rows")
        files[name] = entry
        if entry["match"] is False:
            mismatches.append(name)
    acceptance = json.loads((v02 / "acceptance.json").read_text())
    return {
        "manifest": _hash_entry(manifest_path, None),
        "manifest_n_items_total": manifest["n_items_total"],
        "acceptance_n_items_total": acceptance["n_items_total"],
        "manifest_totals": manifest["totals"],
        "denominator": v02_denominator_block(manifest),
        "window_start": manifest["window_start"],
        "window_end": manifest["window_end"],
        "strict_slice_start": manifest["strict_slice_start"],
        "files": files,
        "files_checked": len(files),
        "file_hash_mismatches": mismatches,
    }


def v02_denominator_block(manifest: dict[str, Any]) -> dict[str, Any]:
    """Test-split denominator of v0.2 from the files, using the screen's kept and dropped templates.

    The screen (data/bench/v0.2/screen.json) drops templates with fewer than 200 test items; that
    is the kept set student_v1 scored (37,289 items). Every drop is listed with its count.
    """
    v02 = REPO / "data/bench/v0.2"
    screen = json.loads((v02 / "screen.json").read_text())

    def rows() -> Iterator[dict[str, Any]]:
        for rec in manifest["files"].values():
            with (REPO / rec["path"]).open() as fh:
                for line in fh:
                    yield json.loads(line)

    counts = count_test_denominator(rows(), screen["templates"])
    dropped_meta = {t["template_id"]: {"tier": t["tier"], "n_test_screen": t["n_test"],
                                       "reasons": t["drop_reasons"]}
                    for t in screen["templates"] if t["drop"]}
    kept_test_screen = sum(t["n_test"] for t in screen["templates"] if not t["drop"])
    return {
        "raw_test_rows": counts.raw_test_rows,
        "screen_dropped_templates": {
            tid: {**dropped_meta[tid], "n_test_files": n} for tid, n in counts.dropped_by_template.items()
        },
        "screen_dropped_test_rows": counts.dropped_test_rows,
        "kept_test_rows_from_files": counts.kept_test_rows,
        "kept_test_rows_from_screen": kept_test_screen,
        "kept_test_rows_by_tier_from_files": {"fresh": counts.kept_fresh_rows,
                                              "tier1": counts.kept_tier1_rows},
        "kept_templates": sum(1 for t in screen["templates"] if not t["drop"]),
        "reference_student_v1_kept_test_items": 37289,
        "check_kept_files_equals_screen": counts.kept_test_rows == kept_test_screen,
    }


def train_block() -> dict[str, Any]:
    mix_dir = REPO / "data/train/student_v1"
    manifest = json.loads((mix_dir / "manifest.json").read_text())
    out: dict[str, Any] = {
        "student_v1_manifest": _hash_entry(mix_dir / "manifest.json", None),
        "student_v1_kind": manifest.get("kind"),
        "student_v1_held_out_templates": manifest.get("held_out_templates"),
        "student_v1_distinct_templates": manifest.get("distinct_training_templates_student_v1"),
        "student_v1_train": _hash_entry(mix_dir / "train.jsonl", None),
        "student_v1_dev": _hash_entry(mix_dir / "dev.jsonl", None),
        "student_v1_train_rows": _count_lines(mix_dir / "train.jsonl"),
        "student_v1_dev_rows": _count_lines(mix_dir / "dev.jsonl"),
    }
    base = manifest.get("base", {})
    base_path = REPO / base["path"]
    out["student_v0_base_train"] = _hash_entry(base_path, base.get("sha256"))
    out["student_v0_base_train_rows_recorded"] = base.get("rows")
    return out


def env_block() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for pkg in ["torch", "transformers", "peft", "accelerate", "datasets", "huggingface-hub",
                "flash-linear-attention", "causal-conv1d", "numpy", "pandas", "scipy"]:
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "gpu_name": gpu_name(),
        "nvidia_smi": smi.stdout.strip(),
        "root_uv_lock": _hash_entry(REPO / "uv.lock", None),
        "envs_unsloth_uv_lock": _hash_entry(REPO / "envs/unsloth/uv.lock", None),
        "pyproject": _hash_entry(REPO / "pyproject.toml", None),
    }


def disk_block() -> dict[str, Any]:
    """Bytes under the ADVISORY §3 budget (hf_home + outputs + data); envs and the uv cache are reported apart."""
    counted = {
        "hf_home": Path("/workspace/.hf_home"),
        "outputs": REPO / "outputs",
        "data": REPO / "data",
    }
    separate = {"envs": REPO / "envs", "uv_cache": Path("/workspace/.uv_cache")}
    sizes = {k: _du_bytes(p) for k, p in counted.items()}
    known = [v for v in sizes.values() if v is not None]
    return {"bytes": sizes, "total_bytes_counted": sum(known),
            "budget_bytes": 450 * 10**9,
            "outside_budget_bytes": {k: _du_bytes(p) for k, p in separate.items()},
            "note": "ADVISORY §3 budget: hf_home + outputs + data <= 450 GB; envs and uv cache are outside it"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "outputs/osler_v0/O0/snapshot.json")
    args = ap.parse_args()

    porcelain = _git(["status", "--porcelain"]).splitlines()
    payload = {
        "task": "O0",
        "utc": utcnow(),
        "git": {
            "commit": git_commit(REPO),
            "branch": _git(["branch", "--show-current"]),
            "porcelain_status": porcelain,
            "note": "untracked files listed above are not part of the commit being snapshot",
        },
        "bench_v0_2": bench_block(),
        "train_mix": train_block(),
        "environment": env_block(),
        "disk": disk_block(),
    }
    write_json(args.out, payload)
    b = payload["bench_v0_2"]
    t = payload["train_mix"]
    print(f"commit={payload['git']['commit']} branch={payload['git']['branch']}")
    print(f"v0.2 files checked={b['files_checked']} mismatches={b['file_hash_mismatches']} "
          f"manifest_items={b['manifest_n_items_total']} acceptance_items={b['acceptance_n_items_total']}")
    d = b["denominator"]
    print(f"v0.2 raw test rows={d['raw_test_rows']} screen-dropped={d['screen_dropped_test_rows']} "
          f"kept(files)={d['kept_test_rows_from_files']} kept(screen)={d['kept_test_rows_from_screen']} "
          f"check={d['check_kept_files_equals_screen']}")
    print(f"student_v1 train rows={t['student_v1_train_rows']} dev rows={t['student_v1_dev_rows']} "
          f"train sha={t['student_v1_train']['sha256'][:12]}")
    print(f"student_v0 base match={t['student_v0_base_train']['match']}")
    print(f"disk counted bytes={payload['disk']['total_bytes_counted']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
