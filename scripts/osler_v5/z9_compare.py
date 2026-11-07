"""osler_v5 Z9 comparison (CPU; ADVISORY §10 Z9): the Z9 summary against Z1 (C032, C035) and against osler_v4's saved temperatures.

Reads outputs/osler_v5/Z9/summary.json, outputs/osler_v5/Z1/summary.json (the alpha 1 row and the readout split) and
outputs/osler_v4/Y8/checkpoints/best/model.json (the per-qtype temperatures osler_v4 saved for the final). Writes
outputs/osler_v5/Z9/comparison.json and prints the table. Nothing is recomputed on items here.

Usage (CPU):
    uv run --frozen python scripts/osler_v5/z9_compare.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meddecide.utils.io import read_json, write_json

REPO = Path(__file__).resolve().parents[2]
Z9_SUMMARY = REPO / "outputs" / "osler_v5" / "Z9" / "summary.json"
Z1_SUMMARY = REPO / "outputs" / "osler_v5" / "Z1" / "summary.json"
Y8_MODEL = REPO / "outputs" / "osler_v4" / "Y8" / "checkpoints" / "best" / "model.json"
OUT = REPO / "outputs" / "osler_v5" / "Z9" / "comparison.json"
Z1_ROW = {"base_readout": "split_trained_adapter_base_readout", "trained_readout": "alpha_1"}


def compare_setting(z9: dict[str, Any], z1: dict[str, Any]) -> dict[str, Any]:
    """Z9's figures against the Z1 row of the same readout configuration (C035 for the base readout, C032 for alpha 1)."""
    pairs = {
        "S": (z9["composite"]["S"], z1["composite"]["S"]),
        "MR macro": (z9["composite"]["mr"], z1["composite"]["mr"]),
        "MIMIC macro": (z9["composite"]["mimic"], z1["composite"]["mimic"]),
        "CG v2 micro": (z9["composite"]["cg_micro"], z1["composite"]["cg_micro"]),
        "knowledge dev": (z9["knowledge_dev_accuracy"], z1["knowledge_dev_micro"]),
        "DDX two-letter": (z9["ddx_two_letter_accuracy"], z1["ddx_two_letter_accuracy"]),
        "DDX single-letter": (z9["ddx_single_letter_accuracy"], z1["ddx_single_letter_accuracy"]),
    }
    out: dict[str, Any] = {label: {"z9": a, "z1": b, "abs_diff": abs(a - b)} for label, (a, b) in pairs.items()}
    out["positional_passes_all_orders"] = {"z9": z9["positional_passes_all_orders"], "z1": z1["positional_passes_all_orders"]}
    return out


def main() -> None:
    z9 = read_json(Z9_SUMMARY)
    z1 = read_json(Z1_SUMMARY)
    saved = read_json(Y8_MODEL)["calibration"]
    comparison: dict[str, Any] = {"z9_summary": str(Z9_SUMMARY.relative_to(REPO)), "against_z1": {}, "temperatures_vs_y8": {}}
    for name, key in Z1_ROW.items():
        comparison["against_z1"][name] = compare_setting(z9["settings"][name], z1[key])
    for qtype, value in sorted(saved.items()):
        refit = z9["settings"]["trained_readout"]["temperatures"][qtype]["temperature"]
        comparison["temperatures_vs_y8"][qtype] = {"z9_refit_trained_readout": refit, "y8_saved": value, "abs_diff": abs(refit - value)}
    write_json(OUT, comparison)

    print(f"{'setting':<16}{'figure':<20}{'Z9':>10}{'Z1 (C032/C035)':>17}{'abs diff':>12}")
    for name, rows in comparison["against_z1"].items():
        for label, value in rows.items():
            if label == "positional_passes_all_orders":
                print(f"{name:<16}{label:<20}{value['z9']!s:>10}{value['z1']!s:>17}{'':>12}")
            else:
                print(f"{name:<16}{label:<20}{value['z9']:>10.6f}{value['z1']:>17.6f}{value['abs_diff']:>12.2e}")
    print("temperatures (trained readout, refit on fast dev v4 against osler_v4's saved fit):")
    for qtype, row in comparison["temperatures_vs_y8"].items():
        print(f"  {qtype:<8} refit {row['z9_refit_trained_readout']:.6f}  saved {row['y8_saved']:.6f}  abs diff {row['abs_diff']:.2e}")
    for name, setting in z9["settings"].items():
        temps = {q: round(v["temperature"], 4) for q, v in setting["temperatures"].items()}
        paired = setting.get("knowledge_vs_zero_shot")
        paired_text = (f"{paired['point']:+.5f} [{paired['lo']:+.5f}, {paired['hi']:+.5f}]" if paired else "not computed")
        print(f"{name}: temperatures {temps}; knowledge vs zero-shot {paired_text}; positional all orders "
              f"{setting['positional_passes_all_orders']}; Brier calibrated CG micro {setting['brier_calibrated']['cg_v2']['micro']:.4f}"
              f" vs uncalibrated {setting['brier_uncalibrated']['cg_v2']['micro']:.4f}")


if __name__ == "__main__":
    main()
