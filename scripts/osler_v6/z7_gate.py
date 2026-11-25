"""Z7 (ADVISORY §2.8): Gate O6 — the four Gate O4 checks, unchanged, against zero-shot pplx-decider-flash-v1.2.

The final model's predictions on the frozen gate sets (`data/bench/osler_v4/y9/`) are read from
`outputs/osler_v6/Z7/baselines/final_<set>/predictions.jsonl`; the baselines are osler_v4 Y9's frozen runs. The comparison,
pairing, readout gate and bootstrap are osler_v4's own `meddecide.eval.gate_o4` code, called through osler_v4's `y9_gate`
loading and reporting functions (imported by path, unmodified), so the check is the same check that produced Gate O4:

    (i)   MR verdict      accuracy lower bound > 0 and Brier upper bound < 0
    (ii)  MIMIC suite     accuracy lower bound > 0 and Brier upper bound < 0
    (iii) external panel  accuracy lower bound > -0.02
    (iv)  knowledge       accuracy lower bound > -0.02

Additional rows (no verdict): the knowledge-test point estimates against MedDecider-4B and MedDecider-9B (ADVISORY §2.8),
and the same four checks against MedDecider-4B for information.

`--runs-root` and `--final-prefix` exist so this script can be validated against osler_v4's own artifact: with
`--runs-root outputs/osler_v4/Y9/baselines --final-prefix final --out /workspace/tmp/gate_validate.json` it must reproduce
`outputs/osler_v4/Y9/gate.json` (checked at Z7 and recorded in its SELF_AUDIT).

Usage (CPU; after the final's gate sets are scored):
    uv run --frozen python scripts/osler_v6/z7_gate.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
Y9_GATE = REPO / "scripts" / "osler_v4" / "y9_gate.py"
DEFAULT_RUNS = REPO / "outputs" / "osler_v6" / "Z7" / "baselines"
DEFAULT_OUT = REPO / "outputs" / "osler_v6" / "Z7" / "gate_o6.json"


def load_y9_gate() -> Any:
    if "y9_gate" in sys.modules:
        return sys.modules["y9_gate"]
    spec = importlib.util.spec_from_file_location("y9_gate", Y9_GATE)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {Y9_GATE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["y9_gate"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--final-prefix", default="final")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    runs_root = args.runs_root if args.runs_root.is_absolute() else REPO / args.runs_root
    y9 = load_y9_gate()
    original_paths = y9.predictions_path
    y9.MODELS["final"] = {"label": "Osler-4B (osler_v6 final)", "cap": None, "prefix": args.final_prefix, "y4": {}}

    def predictions_path(model: str, set_name: str) -> Path | None:
        """Our final from --runs-root; every frozen baseline from its osler_v4 run (read only)."""
        if model == "final":
            return runs_root / f"{args.final_prefix}_{set_name}" / "predictions.jsonl"
        return original_paths(model, set_name)

    y9.predictions_path = predictions_path
    items_by_set = {name: y9.load_items(name) for name in y9.SETS}
    scored: dict[str, dict[str, Any]] = {}
    for model in y9.MODELS:
        scored[model] = {}
        for set_name in y9.SETS:
            loaded = y9.load_scored(model, set_name, items_by_set[set_name])
            if loaded is not None:
                scored[model][set_name] = loaded
    final = scored["final"]
    if not final:
        raise SystemExit(f"no final-model predictions under {runs_root}: score the gate sets first")
    report: dict[str, Any] = {
        "task": "Z7",
        "gate": "Gate O6 (ADVISORY §2.8): Gate O4's four checks, unchanged, against zero-shot pplx-decider-flash-v1.2",
        "limits": {"mr_mimic": "accuracy lower > 0 and Brier upper < 0", "panel": "accuracy lower > -0.02",
                   "knowledge": "accuracy lower > -0.02"},
        "runs_root": str(runs_root.relative_to(REPO)) if runs_root.is_relative_to(REPO) else str(runs_root),
        "final_prefix": args.final_prefix,
        "gate_pplx_flash": {},
        "additional": {},
        "runs": {model: {name: (scored[model][name]["path"] if name in scored[model] else "NOT RUN")
                         for name in y9.SETS} for model in y9.MODELS},
        "generated_utc": utcnow(),
        "git_commit": git_commit(REPO),
    }
    for base in ("pplx_flash", "meddecider_4b"):
        report["gate_pplx_flash" if base == "pplx_flash" else "gate_meddecider_4b"] = y9.run_checks(
            "final", base, final, scored[base], items_by_set)
    # ADVISORY §2.8: knowledge-test point estimates against MedDecider-4B and -9B are additional rows (no verdict)
    for base in ("pplx_flash", "meddecider_4b", "meddecider_9b"):
        if "knowledge" not in final or "knowledge" not in scored[base]:
            report["additional"][f"knowledge_vs_{base}"] = {"status": "NOT RUN"}
            continue
        pairing = y9.pair_rows(final["knowledge"]["rows"], scored[base]["knowledge"]["rows"],
                               items_by_set["knowledge"], option_cap=y9.MODELS[base]["cap"])
        result = y9.comparison(pairing)
        report["additional"][f"knowledge_vs_{base}"] = {
            "items": result.items, "templates": list(result.templates),
            "accuracy": y9._diff(result.accuracy), "brier": y9._diff(result.brier),
            "unpaired": pairing.unpaired, "unpaired_share": round(pairing.unpaired_share, 6),
        }
    write_json(args.out if args.out.is_absolute() else REPO / args.out, report)
    pplx = report["gate_pplx_flash"]
    print(f"Gate O6 vs zero-shot pplx-flash: {pplx['verdict']}")
    for name, row in pplx["checks"].items():
        print(f"  {name}: {row['verdict']} — {row['reason']}; accuracy {row['accuracy']} Brier {row['brier']}")
    for key, row in report["additional"].items():
        print(f"  additional {key}: {row.get('accuracy')}")


if __name__ == "__main__":
    main()
