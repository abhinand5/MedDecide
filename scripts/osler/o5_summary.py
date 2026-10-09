"""Summarise the O5 readout checks and the kernel-path controls (CPU; reads outputs/osler_v0/O5 only).

Prints tables and writes ``outputs/osler_v0/O5/o5_summary.json``:
  1. the O5 verdict: ``readout_checks.json`` (attempt 4; the adapter's lora_A was drawn from an unseeded RNG);
  2. seeded runs (``readout_checks_<precision>_seed<s>.json``): the as_run configuration re-run with a recorded seed,
     and the additional precision settings, labelled as additional analyses;
  3. seeded kernel-path controls (``kernel_control_<precision>_<tag>.json``) and, for each precision and seed run in two
     processes, the cross-process difference of the saved last-layer hidden states;
  4. the unseeded control records kept in ``unseeded_draws/``, with their cross-process differences.

Usage:
    uv run --frozen python scripts/osler/o5_summary.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "outputs/osler_v0/O5"
UNSEEDED_DIR = OUT_DIR / "unseeded_draws"
KEY_CHECKS = (
    "initialised option-code probabilities vs letter readout",
    "exported causal LM vs native readout, adapter separate",
    "exported causal LM vs native readout, adapter merged",
    "option-code padding invariance, causal",
    "option-code padding invariance, bidirectional",
    "pointer padding invariance",
)


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def print_checks(title: str, record: dict[str, Any]) -> dict[str, Any]:
    rows = [{"check": c["check"], "measured": c["measured"], "rule": c["rule"], "tolerance": c["tolerance"],
             "passed": c["passed"]} for c in record["checks"]]
    print(f"## {title}")
    print("| check | measured | rule | tolerance | passed |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['check']} | {r['measured']:.6g} | {r['rule']} | {r['tolerance']:.6g} | {r['passed']} |")
    print(f"all_passed = {record['all_passed']}\n")
    return {"all_passed": record["all_passed"], "checks": rows}


def control_row(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    sensitivity = record["sensitivity_max_abs_diff"]
    return {"file": path.name, "precision": record["precision"], "seed": record.get("seed", "unseeded"),
            "tag": record["tag"], "rep": record["rep_max_abs_diff"], "merge": record["merge_max_abs_diff"],
            "sens_1e-07": sensitivity["1e-07"], "sens_1e-05": sensitivity["1e-05"],
            "last_hidden_max_abs": record["last_hidden_max_abs"],
            "implementation": record["implementations"]["torch_chunk_gated_delta_rule"]}


def print_control_rows(rows: list[dict[str, Any]]) -> None:
    print("| file | precision | seed | rep | merge (last hidden) | sens 1e-7 | sens 1e-5 | last max abs h | implementation |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['file']} | {r['precision']} | {r['seed']} | {r['rep']:.6g} | {r['merge']:.6g} | "
              f"{r['sens_1e-07']:.6g} | {r['sens_1e-05']:.6g} | {r['last_hidden_max_abs']:.4g} | {r['implementation']} |")
    print()


def cross_process(directory: Path, precision: str, tags: list[str]) -> list[dict[str, Any]]:
    rows = []
    for a, b in combinations(sorted(tags), 2):
        x = torch.load(directory / f"kernel_control_{precision}_{a}_hidden.pt", weights_only=True)
        y = torch.load(directory / f"kernel_control_{precision}_{b}_hidden.pt", weights_only=True)
        rows.append({
            "precision": precision, "pair": f"{a} vs {b}",
            "separate_max_abs_diff": float((x["separate_last"] - y["separate_last"]).abs().max()),
            "merged_max_abs_diff": float((x["merged_last"] - y["merged_last"]).abs().max()),
        })
    return rows


def cross_precision(seed: int, tag: str = "run1") -> list[dict[str, Any]]:
    """Max abs difference of the saved last-layer hidden states between precision settings on the same adapter draw."""
    hidden = {p: torch.load(OUT_DIR / f"kernel_control_{p}_seed{seed}_{tag}_hidden.pt", weights_only=True)
              for p in ("as_run", "ieee", "reference")}
    rows = []
    for a, b in (("as_run", "reference"), ("ieee", "reference"), ("as_run", "ieee")):
        rows.append({
            "seed": seed, "pair": f"{a} vs {b}",
            "separate_max_abs_diff": float((hidden[a]["separate_last"] - hidden[b]["separate_last"]).abs().max()),
            "merged_max_abs_diff": float((hidden[a]["merged_last"] - hidden[b]["merged_last"]).abs().max()),
            "reference_last_hidden_max_abs": float(hidden["reference"]["separate_last"].abs().max()),
        })
    return rows


def print_cross(rows: list[dict[str, Any]]) -> None:
    print("| precision | pair | separate max abs diff | merged max abs diff |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['precision']} | {r['pair']} | {r['separate_max_abs_diff']:.6g} | {r['merged_max_abs_diff']:.6g} |")
    print()


def main() -> None:
    summary: dict[str, Any] = {}

    verdict = load(OUT_DIR / "readout_checks.json")
    if verdict is not None:
        summary["verdict_attempt4_unseeded"] = print_checks("O5 verdict (attempt 4, unseeded; readout_checks.json)",
                                                            verdict)

    summary["seeded_checks"] = {}
    for precision in ("as_run", "reference", "ieee"):
        record = load(OUT_DIR / f"readout_checks_{precision}_seed0.json")
        if record is None:
            print(f"## {precision} seed 0: not run\n")
            continue
        label = ("seeded re-run of the as_run configuration (not the verdict)" if precision == "as_run"
                 else f"{precision} precision (additional analysis, not the verdict)")
        summary["seeded_checks"][precision] = print_checks(f"{label}; seed 0", record)

    seeded: dict[tuple[str, int], list[Path]] = defaultdict(list)
    seeded_rows = []
    for path in sorted(OUT_DIR.glob("kernel_control_*.json")):
        row = control_row(path)
        seeded_rows.append(row)
        seeded[(row["precision"], row["seed"])].append(path)
    print("## kernel-path controls, seeded (diagnostic)")
    print_control_rows(seeded_rows)
    summary["kernel_controls_seeded"] = seeded_rows

    print("## cross-process reproducibility, seeded (same precision and seed, two processes)")
    cross_rows = []
    for (precision, _seed), paths in sorted(seeded.items()):
        if len(paths) >= 2:
            tags = [json.loads(p.read_text(encoding="utf-8"))["tag"] for p in paths]
            cross_rows.extend(cross_process(OUT_DIR, precision, tags))
    print_cross(cross_rows)
    summary["cross_process_seeded"] = cross_rows

    print("## cross-precision, same adapter draw (seed 0, run 1): last-layer hidden states")
    print("| pair | separate max abs diff | merged max abs diff | reference max abs h |")
    print("|---|---|---|---|")
    precision_rows = cross_precision(seed=0)
    for r in precision_rows:
        print(f"| {r['pair']} | {r['separate_max_abs_diff']:.6g} | {r['merged_max_abs_diff']:.6g} | "
              f"{r['reference_last_hidden_max_abs']:.4g} |")
    print()
    summary["cross_precision_seed0"] = precision_rows

    unseeded_paths = sorted(UNSEEDED_DIR.glob("kernel_control_*.json"))
    if unseeded_paths:
        print("## unseeded control records (kept for transparency; each adapter drew lora_A from an unseeded RNG)")
        unseeded_rows = [control_row(p) for p in unseeded_paths]
        print_control_rows(unseeded_rows)
        summary["kernel_controls_unseeded"] = unseeded_rows
        print("## cross-process differences, unseeded records")
        unseeded_cross = []
        for precision in ("as_run", "ieee"):
            tags = [r["tag"] for r in unseeded_rows if r["precision"] == precision]
            if len(tags) >= 2:
                unseeded_cross.extend(cross_process(UNSEEDED_DIR, precision, tags))
        print_cross(unseeded_cross)
        summary["cross_process_unseeded"] = unseeded_cross

    out = OUT_DIR / "o5_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
