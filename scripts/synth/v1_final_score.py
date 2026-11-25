"""synth_v1 final: score the 63,304-step checkpoint on the frozen sets, then Gate O5 (D55 rules unchanged).

This is a wrapper. It reuses the osler_v5 Z5 code by path and redirects its output directory:
- ``score``: scripts/osler_v5/z5_score.py (GPU) on the frozen osler_v4 Y9 sets (MR verdict and reported, MR held-out, MIMIC audit
  suite and held-out, panel, knowledge, MedNLI test, pairs test). Writes outputs/synth_v1/final/<prefix>_<set>/predictions.jsonl.
- ``gate``: scripts/osler_v5/z5_gate.py (CPU) on those predictions. The gate is the osler_v4 Y9 gate (scripts/osler_v4/y9_gate.py),
  i.e. the D49 checks, which Gate O5 reuses under D55. Writes outputs/synth_v1/final/gate.json.
- ``validate``: the same gate on osler_v4's final predictions (CPU). It must reproduce osler_v4's published Gate O4 figures before
  the synth gate is read (wiring check). Writes outputs/synth_v1/final/validate_gate.json.
- ``summary``: aggregate-only lines from a gate file (verdicts and figures; no item text, identifiers or predictions).

The checkpoint must carry its calibration temperatures (z5_score refuses otherwise). The per-evaluation checkpoints do not carry
them; only final_step_063304 does, because the driver fits them after training.

Usage:
    uv run --frozen python scripts/synth/v1_final_score.py validate
    uv run --frozen python scripts/synth/v1_final_score.py gate --prefix v1final
    setsid nohup ... scripts/synth/v1_final_score.py score --checkpoint outputs/synth_v1/train/checkpoints/final_step_063304 --prefix v1final  (GPU)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[2]
OSLER_V5 = REPO / "scripts" / "osler_v5"
OUT = REPO / "outputs" / "synth_v1" / "final"
FINAL_CHECKPOINT = REPO / "outputs" / "synth_v1" / "train" / "checkpoints" / "final_step_063304"
# osler_v4's published Gate O4 figures (C220, C221), the wiring the validate mode must reproduce
VALIDATE_EXPECTED = {
    "pplx_flash": {"mr_verdict": 0.1387, "mimic_suite": 0.1654, "panel": -0.0421, "knowledge": -0.0067},
    "meddecider_4b": {"mr_verdict": 0.2449, "mimic_suite": 0.1651},
}
VALIDATE_VERDICTS = {("pplx_flash", "panel"): "FAIL", ("meddecider_4b", "panel"): "NOT MEASURED"}
VALIDATE_TOLERANCE = 0.0005


def load_module(name: str, relative: str) -> ModuleType:
    path = REPO / relative
    if not path.exists():
        raise SystemExit(f"{relative} is missing; this wrapper needs the osler_v5 tree (loop/osler_v5 at 23a0362)")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_score(checkpoint: Path, prefix: str) -> None:
    z5_score = load_module("z5_score", "scripts/osler_v5/z5_score.py")
    z5_score.OUT = OUT  # the predictions and the manifest go to the synth_v1 folder, not to outputs/osler_v5/Z5
    sys.argv = ["z5_score.py", "--checkpoint", str(checkpoint), "--prefix", prefix]
    z5_score.main()


def run_gate(prefix: str | None, validate: bool) -> None:
    z5_gate = load_module("z5_gate", "scripts/osler_v5/z5_gate.py")
    z5_gate.Z5 = OUT
    argv = ["z5_gate.py"] + (["--validate"] if validate else ["--prefix", prefix or "v1final"])
    sys.argv = argv
    z5_gate.main()


def summarise(gate: dict[str, Any]) -> dict[str, Any]:
    """Aggregate-only view of a gate file: the verdict per baseline and each check's verdict and figures. No item text."""
    out: dict[str, Any] = {"mode": gate.get("mode"), "sets_scored": gate.get("sets_scored", []), "baselines": {}}
    for base, result in gate.get("results", {}).items():
        out["baselines"][base] = {
            "verdict": result.get("verdict"),
            "checks": {key: {"check": c.get("check"), "verdict": c.get("verdict"), "reason": c.get("reason"),
                             "accuracy": c.get("accuracy"), "brier": c.get("brier"), "items": c.get("items")}
                       for key, c in result.get("checks", {}).items()},
        }
    return out


def wiring_mismatches(gate: dict[str, Any]) -> list[str]:
    """Each osler_v4 Gate O4 figure that the validate run does not reproduce (empty when it reproduces all of them)."""
    problems: list[str] = []
    for base, expected in VALIDATE_EXPECTED.items():
        checks = gate.get("results", {}).get(base, {}).get("checks", {})
        by_name = {c.get("check"): c for c in checks.values()}
        for set_name, value in expected.items():
            found = next((c for name, c in by_name.items() if name and name.endswith(set_name)), None)
            if found is None or found.get("accuracy") is None:
                problems.append(f"{base} {set_name}: no figure")
                continue
            got = found["accuracy"].get("point") if isinstance(found["accuracy"], dict) else None
            if got is None or abs(float(got) - value) > VALIDATE_TOLERANCE:
                problems.append(f"{base} {set_name}: {got} against {value}")
    for (base, set_name), verdict in VALIDATE_VERDICTS.items():
        checks = gate.get("results", {}).get(base, {}).get("checks", {})
        found = next((c for c in checks.values() if str(c.get("check", "")).endswith(set_name)), None)
        if found is None or found.get("verdict") != verdict:
            problems.append(f"{base} {set_name}: verdict {None if found is None else found.get('verdict')} against {verdict}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=["score", "gate", "validate", "summary"])
    parser.add_argument("--checkpoint", default=str(FINAL_CHECKPOINT))
    parser.add_argument("--prefix", default="v1final")
    parser.add_argument("--file", default=None, help="summary: the gate file to summarise")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "score":
        run_score(Path(args.checkpoint), args.prefix)
    elif args.mode == "gate":
        run_gate(args.prefix, validate=False)
    elif args.mode == "validate":
        run_gate(None, validate=True)
        gate = json.loads((OUT / "validate_gate.json").read_text(encoding="utf-8"))
        problems = wiring_mismatches(gate)
        print("wiring check:", "PASS, all osler_v4 Gate O4 figures reproduced" if not problems else "FAIL: " + "; ".join(problems),
              flush=True)
    else:
        path = Path(args.file) if args.file else OUT / "gate.json"
        print(json.dumps(summarise(json.loads(path.read_text(encoding="utf-8"))), indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
