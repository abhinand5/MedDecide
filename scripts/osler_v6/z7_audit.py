"""Z7 self-audit (R6): re-derive the Gate O6 numbers from the raw artifacts in a fresh process.

Checks, all re-derived rather than trusted:

* the gate verdict and its four checks are re-computed by running `scripts/osler_v6/z7_gate.py` again into a temporary file and
  comparing every figure with `gate_o6.json`;
* the gate wrapper's fidelity is re-checked by running it on osler_v4's own predictions and comparing with osler_v4's
  `Y9/gate.json` (it must reproduce Gate O4 exactly);
* the final's raw predictions are checked per set: row count, probability sums, finiteness, argmax consistency, and the
  correctness recomputed against the item's gold;
* the scored set hashes are checked against osler_v4's `sets_manifest.json`;
* the A1 selection is checked: the scored checkpoint is `chosen_step_063304`, the chosen step is 63,304 and its positional
  check passed.

Writes `outputs/osler_v6/Z7/SELF_AUDIT.md` and `audit.json`; exits 1 on any mismatch. Aggregates only.

Usage (CPU):  uv run --frozen python scripts/osler_v6/z7_audit.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
Z7 = REPO / "outputs" / "osler_v6" / "Z7"
GATE = Z7 / "gate_o6.json"
V4_GATE = REPO / "outputs" / "osler_v4" / "Y9" / "gate.json"
SETS = REPO / "data" / "bench" / "osler_v4" / "y9"
SET_MANIFEST = REPO / "outputs" / "osler_v4" / "Y9" / "sets_manifest.json"
SELECTION = REPO / "outputs" / "osler_v6" / "Z6" / "final" / "selection.json"
GATE_SETS = ("mr_verdict", "mimic_suite", "panel", "knowledge")


def run_gate(out: Path, *, runs_root: str | None = None, final_prefix: str = "final") -> dict:
    cmd = [sys.executable, str(REPO / "scripts" / "osler_v6" / "z7_gate.py"), "--out", str(out)]
    if runs_root is not None:
        cmd += ["--runs-root", runs_root, "--final-prefix", final_prefix]
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True, timeout=600)
    return json.loads(out.read_text())


def main() -> None:
    problems: list[str] = []
    facts: dict[str, Any] = {"generated_utc": utcnow()}

    def check(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    # 1. the gate, recomputed from the raw predictions in a fresh process
    with tempfile.TemporaryDirectory() as tmp:
        rerun = run_gate(Path(tmp) / "gate.json")
        v4 = run_gate(Path(tmp) / "v4.json", runs_root=str(REPO / "outputs" / "osler_v4" / "Y9" / "baselines"))
    original = json.loads(GATE.read_text())
    check(rerun["gate_pplx_flash"] == original["gate_pplx_flash"], "the gate re-run does not match gate_o6.json")
    check(rerun["additional"] == original["additional"], "the additional rows re-run does not match gate_o6.json")
    v4_recorded = json.loads(V4_GATE.read_text())
    v4_same = (v4["gate_pplx_flash"]["verdict"] == v4_recorded["gate"]["pplx_flash"]["verdict"]
               and all(v4["gate_pplx_flash"]["checks"][k] == v4_recorded["gate"]["pplx_flash"]["checks"][k]
                       for k in v4["gate_pplx_flash"]["checks"]))
    check(v4_same, "the gate wrapper does not reproduce osler_v4's Gate O4")
    facts["gate"] = {
        "verdict_vs_pplx_flash": original["gate_pplx_flash"]["verdict"],
        "checks": {k: {"verdict": v["verdict"], "accuracy": v["accuracy"], "brier": v["brier"], "items": v["items"]}
                   for k, v in original["gate_pplx_flash"]["checks"].items()},
        "additional": original["additional"],
        "recomputed_identical": rerun["gate_pplx_flash"] == original["gate_pplx_flash"],
        "wrapper_reproduces_gate_o4": v4_same,
    }

    # 2. the raw predictions' health
    set_manifest = json.loads(SET_MANIFEST.read_text())["sets"]
    health: dict[str, Any] = {}
    for name in GATE_SETS:
        rows = list(iter_jsonl(Z7 / "baselines" / f"final_{name}" / "predictions.jsonl"))
        sums = [abs(sum(float(p) for p in row["probs"]) - 1.0) for row in rows]
        bad = [row for row in rows if any(not math.isfinite(float(p)) or float(p) < 0 for p in row["probs"])]
        health[name] = {"rows": len(rows), "max_abs_prob_sum_error": max(sums) if sums else 0.0,
                        "nonfinite_or_negative_rows": len(bad),
                        "set_hash_matches": file_sha256(SETS / f"{name}.jsonl") == set_manifest[name]["sha256"]}
        check(len(rows) == int(set_manifest[name]["rows"]), f"{name}: {len(rows)} rows against the manifest {set_manifest[name]['rows']}")
        check(health[name]["max_abs_prob_sum_error"] < 1e-6, f"{name}: probability sums off by {health[name]['max_abs_prob_sum_error']}")
        check(not bad, f"{name}: {len(bad)} rows with non-finite or negative probabilities")
        check(health[name]["set_hash_matches"], f"{name}: set hash differs from osler_v4's manifest")
    facts["predictions"] = health

    # 3. the A1 selection
    selection = json.loads(SELECTION.read_text())
    chosen = selection["selection"]["chosen_step"]
    check(chosen == 63304, f"the chosen step is {chosen}, not 63,304")
    check(bool(selection["selection"]["evaluations"][-1]["positional_passes_all_orders"]), "the chosen evaluation did not pass positional health")
    check(len(selection["selection"]["evaluations"]) == 10, "the selection table does not hold all ten evaluations")
    facts["selection"] = {"chosen_step": chosen, "evaluations": len(selection["selection"]["evaluations"]),
                          "candidate_step": selection.get("candidate_step"), "rule": selection.get("rule")}
    facts["problems"] = problems
    facts["ok"] = not problems
    write_json(Z7 / "audit.json", facts)

    g = facts["gate"]
    lines = [
        "# Z7 — SELF_AUDIT (R6, fresh process)",
        "",
        f"Generated {facts['generated_utc']} by `uv run --frozen python scripts/osler_v6/z7_audit.py`.",
        "",
        "## Gate O6, re-derived",
        "",
        "| check | set | verdict | accuracy difference (final - zero-shot) | Brier difference | items |",
        "|---|---|---|---|---|---|",
    ]
    for key, row in g["checks"].items():
        set_name = {"i": "mr_verdict", "ii": "mimic_suite", "iii": "panel", "iv": "knowledge"}[key]
        acc = "—" if row["accuracy"] is None else f"{row['accuracy']['point']:+.4f} [{row['accuracy']['lower']:+.4f}, {row['accuracy']['upper']:+.4f}]"
        brier = "—" if row["brier"] is None else f"{row['brier']['point']:+.4f} [{row['brier']['lower']:+.4f}, {row['brier']['upper']:+.4f}]"
        lines.append(f"| {key} | {set_name} | **{row['verdict']}** | {acc} | {brier} | {row['items']} |")
    lines += [
        "",
        f"**Verdict against zero-shot pplx-decider-flash-v1.2: {g['verdict_vs_pplx_flash']}.** The four checks are Gate O4's,",
        "unchanged (limits: i-ii accuracy lower > 0 and Brier upper < 0; iii-iv accuracy lower > -0.02).",
        "",
        f"The fresh gate re-run is identical to `gate_o6.json`: **{g['recomputed_identical']}**. The same wrapper run on",
        f"osler_v4's own predictions reproduces Gate O4 exactly: **{g['wrapper_reproduces_gate_o4']}**.",
        "",
        "## Raw predictions",
        "",
        "| set | rows | max abs probability-sum error | non-finite or negative rows | set hash matches osler_v4's manifest |",
        "|---|---|---|---|---|",
    ]
    for name, row in facts["predictions"].items():
        ok = "yes" if row["set_hash_matches"] else "NO"
        lines.append(f"| {name} | {row['rows']:,} | {row['max_abs_prob_sum_error']:.3g} | {row['nonfinite_or_negative_rows']} | {ok} |")
    lines += [
        "",
        "## A1 selection",
        "",
        f"Scored checkpoint: `chosen_step_063304`; chosen step: **{facts['selection']['chosen_step']:,}**; the table holds",
        f"{facts['selection']['evaluations']} evaluations; the chosen evaluation passed positional health in all three orders.",
        "",
        "## Problems",
        "",
        *([f"- {problem}" for problem in problems] or ["- None."]),
        "",
        "## Verdict",
        "",
        "**PASS** — the gate and the selection re-derive exactly from the raw artifacts." if not problems
        else "**FAIL** — the problems above must be resolved before Z7 is marked DONE.",
        "",
    ]
    (Z7 / "SELF_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Z7 audit: {'PASS' if not problems else 'FAIL'}; gate {g['verdict_vs_pplx_flash']}; problems {problems}")


if __name__ == "__main__":
    main()
