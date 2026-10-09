#!/usr/bin/env python
"""S0 — orientation snapshot: what this loop starts from.

Records the git state, the v0.1 benchmark manifests (path, sha256, item counts) the
loop builds on, the screen's kept/dropped templates, the test count, and the kernel
measurements once they exist. Everything it writes is a hash or a count — no item
text, no predictions.

Usage:
    uv run python scripts/student/s0_snapshot.py --out outputs/student_v0/S0/snapshot.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from meddecide.utils.hashing import file_sha256

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "data" / "bench" / "v0.1"


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "UNKNOWN"


def manifest_summary(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path.relative_to(REPO)), "exists": False}
    data = json.loads(path.read_text())
    out = {
        "path": str(path.relative_to(REPO)),
        "exists": True,
        "sha256": file_sha256(path),
        "n_files": len(data.get("files", {})),
        "totals": data.get("totals", {}),
        "n_items": sum(data.get("totals", {}).values()),
        "built_at_utc": data.get("built_at_utc"),
    }
    for key in ("window_start", "window_end", "strict_slice_start", "window_source"):
        if key in data:
            out[key] = data[key]
    return out


def screen_summary(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path.relative_to(REPO)), "exists": False}
    data = json.loads(path.read_text())
    rows = data.get("templates", data.get("rows", []))
    kept, dropped = [], []
    for row in rows if isinstance(rows, list) else []:
        entry = {
            "template_id": row.get("template_id"),
            "n_test": row.get("n_test"),
            "drop_reasons": row.get("drop_reasons"),
        }
        (dropped if row.get("drop") else kept).append(entry)
    return {
        "path": str(path.relative_to(REPO)),
        "exists": True,
        "sha256": file_sha256(path),
        "n_templates": len(rows) if isinstance(rows, list) else None,
        "summary": data.get("summary"),
        "kept_ids": sorted(e["template_id"] for e in kept if e["template_id"]),
        "kept_n_test": {e["template_id"]: e["n_test"] for e in kept if e["template_id"]},
        "dropped": dropped,
    }


def pytest_summary(junit: Path) -> dict:
    if not junit.exists():
        return {"junit": str(junit.relative_to(REPO)), "exists": False}
    import xml.etree.ElementTree as ET

    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root[0]
    return {
        "junit": str(junit.relative_to(REPO)),
        "exists": True,
        "tests": int(suite.get("tests", 0)),
        "failures": int(suite.get("failures", 0)),
        "errors": int(suite.get("errors", 0)),
        "skipped": int(suite.get("skipped", 0)),
        "time_s": float(suite.get("time", 0.0)),
    }


def merge_kernels(before: Path, after: Path, out: Path) -> dict | None:
    """Combine the before/after measurements into one artifact, with the ratio.

    Only ``seconds_min`` (best of N reps after a warmup pass at the measured length) is
    compared: the first call at a new sequence length carries the Triton JIT compile, which
    is a property of the measurement protocol rather than of the kernel.
    """
    if not (before.exists() and after.exists()):
        return None
    b = json.loads(before.read_text())
    a = json.loads(after.read_text())
    by_len_before = {m["prompt_tokens"]: m for m in b.get("measurements", [])}
    by_len_after = {m["prompt_tokens"]: m for m in a.get("measurements", [])}
    rows = []
    for length in sorted(set(by_len_before) & set(by_len_after)):
        bm, am = by_len_before[length], by_len_after[length]
        rows.append(
            {
                "prompt_tokens": length,
                "before_tokens_per_s_min": bm["tokens_per_s_min"],
                "after_tokens_per_s_min": am["tokens_per_s_min"],
                "speedup": round(am["tokens_per_s_min"] / bm["tokens_per_s_min"], 3),
                "before_peak_allocated_bytes": bm["peak_allocated_bytes"],
                "after_peak_allocated_bytes": am["peak_allocated_bytes"],
                "before_peak_reserved_bytes": bm["peak_reserved_bytes"],
                "after_peak_reserved_bytes": am["peak_reserved_bytes"],
                "reps": am.get("reps"),
                "warmup_at_length": am.get("warmup_at_length"),
            }
        )
    merged = {
        "model": a.get("model"),
        "model_class": a.get("model_class"),
        "device": a.get("device"),
        "torch": a.get("torch"),
        "cuda": a.get("cuda"),
        "before": {"path": str(before.relative_to(REPO)), "kernels": b.get("kernels"),
                   "fallback_warnings": b.get("fallback_warnings")},
        "after": {"path": str(after.relative_to(REPO)), "kernels": a.get("kernels"),
                  "fallback_warnings": a.get("fallback_warnings"),
                  "resolved_implementations": a.get("resolved_implementations")},
        "comparison": rows,
        "protocol_note": (
            "before = the pre-install state reproduced by making the kernel packages "
            "unimportable in-process; after = packages installed and bound. Both runs warm "
            "up once at the measured length, then report the best of N timed passes."
        ),
    }
    out.write_text(json.dumps(merged, indent=2) + "\n")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "outputs/student_v0/S0/snapshot.json")
    parser.add_argument("--junit", type=Path, default=REPO / "outputs/student_v0/S0/pytest.xml")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.junit = args.junit.resolve()

    snapshot = {
        "taken_at_utc": utcnow(),
        "git": {
            "branch": git("branch", "--show-current"),
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "last_commit": git("log", "-1", "--format=%s"),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "v0_1_benchmark": {
            "tier1": manifest_summary(BENCH / "tier1" / "manifest.json"),
            "fresh": manifest_summary(BENCH / "fresh" / "manifest.json"),
            "screen": screen_summary(BENCH / "screen.json"),
        },
        "pytest": pytest_summary(args.junit),
    }

    for name in ("kernels_before", "kernels_after"):
        candidate = args.out.parent / f"{name}.json"
        if candidate.exists():
            data = json.loads(candidate.read_text())
            snapshot[name] = {
                "path": str(candidate.relative_to(REPO)),
                "kernels_mode": data.get("kernels_mode"),
                "kernels": data.get("kernels"),
                "measurements": data.get("measurements"),
                "errors": data.get("errors"),
            }

    merged = merge_kernels(
        args.out.parent / "kernels_before.json",
        args.out.parent / "kernels_after.json",
        args.out.parent / "kernels.json",
    )
    if merged is not None:
        snapshot["kernels_merged"] = {
            "path": str((args.out.parent / "kernels.json").relative_to(REPO)),
            "comparison": merged["comparison"],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps(snapshot, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
