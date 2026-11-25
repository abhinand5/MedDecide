#!/usr/bin/env python
"""osler_v7 T3 self-audit (R6, CPU): re-derive the port-check's headline numbers from the raw artifacts.

Fresh process; does not import the port-check's analysis code. Recomputes the parity maximum, the gate verdict, the
structural all-pass, the prepared-row counts, the readout shape and the snapshot manifest hashes, and compares each with
`outputs/osler_v7/T3/port_check.json`. Writes `audit.json` and `SELF_AUDIT.md` (aggregates only).

Run: ``uv run --frozen python scripts/osler_v7/t3_audit.py`` (CPU; hashes the small snapshot files only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

OUT = ROOT / "outputs" / "osler_v7" / "T3"
REVISION = "67e03e1bc157961c582ee7c6a35d9b15fada3dec"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--snapshot", type=Path, default=None)
    args = parser.parse_args()
    result = json.loads((args.out_dir / "port_check.json").read_text(encoding="utf-8"))
    snapshot = args.snapshot or Path(result["snapshot"]["snapshot"])
    checks: list[dict[str, Any]] = []
    problems: list[str] = []

    def check(name: str, recorded: Any, recomputed: Any) -> None:
        match = recorded == recomputed
        checks.append({"check": name, "recorded": recorded, "recomputed": recomputed, "match": match})
        if not match:
            problems.append(f"{name}: recorded {recorded!r} != recomputed {recomputed!r}")

    parity = result["parity"]
    diffs = [item["max_abs_diff"] for item in parity]
    max_diff = max(diffs) if diffs else None
    check("parity.items_measured", result["gate"]["items_measured"], len(parity))
    check("parity.items_skipped", result["gate"]["items_skipped"], len(result["parity_skipped"]))
    check("parity.max_abs_diff", result["gate"]["max_abs_diff"], max_diff)
    verdict = "PASS" if max_diff is not None and max_diff <= result["gate"]["bar"] else (
        "FAIL" if max_diff is not None else "NOT MEASURED"
    )
    check("gate.verdict", result["gate"]["verdict"], verdict)
    check("gate.bar", result["gate"]["bar"], 1e-4)
    check("structural.all_pass", result["structural"]["all_pass"],
          all(item["all_checks_pass"] for item in result["structural"]["items"]))
    check("structural.n_items", result["structural"]["n_items"], len(result["structural"]["items"]))
    check("prepare.items", len(result["prepare"]), result["selection"]["rows_selected"])
    prepare_errors = sum(1 for item in result["prepare"] if "prepare_error" in item)
    check("prepare.errors", prepare_errors, 0)
    check("runtime.readout_shape", result["runtime"]["readout_shape"], [255, 4096])
    check("snapshot.decision.codes", result["snapshot"]["decision_config"]["codes"], 255)
    check("snapshot.max_input_tokens", result["snapshot"]["max_input_tokens"], None)
    check("snapshot.revision_matches_pin", result["snapshot"]["revision_matches_pin"], snapshot.name == REVISION)
    check("runtime.device", result["runtime"]["device"], "cpu")

    # Re-verify the snapshot manifest hashes from the files themselves.
    manifest = json.loads((snapshot / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    hashes = manifest["files_sha256"]
    hash_results: dict[str, bool] = {}
    for name in ("config.json", "decision_config.json", "readout.safetensors", "tokenizer.json", "modeling_d3.py",
                 "d3_runtime.py", "chat_template.jinja"):
        hash_results[name] = sha256_file(snapshot / name) == hashes[name]
    check("snapshot.manifest_hashes_all_match", result["snapshot"]["manifest_hashes_all_match"], all(hash_results.values()))
    for name, ok in hash_results.items():
        check(f"snapshot.hash.{name}", True, ok)

    # The batch check's maximum, if it ran.
    batch = result["batch_check"]
    if batch.get("status") == "MEASURED":
        check("batch.max", batch["max_batch_vs_reference"],
              max(item["batch_vs_reference"] for item in batch["items"]))
    check("batch.items", len(batch.get("items", [])), 2 if batch.get("status") == "MEASURED" else len(batch.get("items", [])))

    audit = {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t3_audit.py",
        "checks": checks,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }
    with (args.out_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")

    gate = result["gate"]
    notes = []
    if result["parity_skipped"]:
        notes.append(
            f"- {len(result['parity_skipped'])} row(s) were prepared but their forward was NOT MEASURED: "
            + "; ".join(f"{item['tags']} ({item['prompt_tokens']} tokens): {item['reason']}" for item in result["parity_skipped"])
        )
    notes.append(
        "- The parity compares the training harness with the checkpoint's own runtime *on CPU in this session*; the "
        "d3_distill step-2 GPU parity (C024: max difference 0.0, 40 items) remains the GPU-side record and is not repeated here."
    )
    notes.append(
        "- The CPU environment has no flash-linear-attention / causal-conv1d kernels (the runtime's own guard refuses CPU "
        "otherwise), so both sides of the comparison run the same PyTorch reference path; this validates the harness "
        "against the runtime, not CPU-vs-GPU numerics."
    )
    lines = [
        "# T3 SELF_AUDIT — d3-flash port check for the 9B harness (osler_v7)",
        "",
        f"Generated {audit['generated_utc']} at `{audit['git_commit']}` in a fresh process; command "
        f"`{audit['command']}`.",
        "",
        f"**Status: {audit['status']}** — {len(checks)} checks, {len(problems)} problems.",
        "",
        f"Gate: {gate['verdict']} (max |harness - runtime| {gate['max_abs_diff']}, bar {gate['bar']}); "
        f"{gate['items_measured']} items measured, {gate['items_skipped']} skipped.",
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
    lines.extend(["", "## Notes", "", *notes, ""])
    (args.out_dir / "SELF_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"T3 audit {audit['status']}: {len(checks)} checks, {len(problems)} problems")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
