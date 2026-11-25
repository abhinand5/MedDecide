"""d3_distill step 2: pick the final's seed from a student's two arms, by the pre-registered rule (ADVISORY §4).

Qualify an arm when its D57 band passed and its final dev accuracy is not below the zero-shot student by more than 2
points; among qualifiers pick the higher final dev accuracy, ties to seed 0. Writes the selection JSON.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

MIN_MARGIN = 0.02


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--student", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="arm directories (s0, s1)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    arms = []
    for directory in args.arms:
        path = Path(directory) / "arm_result.json"
        if not path.exists():
            raise SystemExit(f"{path} is missing")
        report = json.loads(path.read_text(encoding="utf-8"))
        final = report.get("final_eval") or {}
        zero = (report.get("zero_shot_dev") or {}).get("accuracy")
        band = report.get("band") or {}
        margin = None if not final or zero is None else final["accuracy"] - zero
        qualified = bool(band.get("passed")) and margin is not None and margin >= -MIN_MARGIN
        arms.append(
            {
                "dir": str(directory),
                "seed": report["cfg"]["seed"],
                "band_passed": band.get("passed"),
                "band_mean_share": band.get("mean_share"),
                "final_dev_accuracy": final.get("accuracy"),
                "final_dev_step": final.get("step"),
                "zero_shot_dev_accuracy": zero,
                "margin_over_zero_shot": margin,
                "qualified": qualified,
            }
        )
    qualified = [a for a in arms if a["qualified"]]
    if not qualified:
        selected = None
        outcome = "no arm qualifies: no final is trained"
    else:
        best = max(qualified, key=lambda a: (a["final_dev_accuracy"], -a["seed"]))
        selected = best
        outcome = f"seed {best['seed']} selected"
    result = {"student": args.student, "rule": "band pass and final dev >= zero-shot - 0.02; higher final dev; tie seed 0",
              "arms": arms, "selected": selected, "outcome": outcome, "finished_utc": utcnow()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({"student": args.student, "selected_seed": None if selected is None else selected["seed"],
                      "outcome": outcome, "arms": [{k: a[k] for k in ("seed", "band_passed",
                      "final_dev_accuracy", "zero_shot_dev_accuracy", "margin_over_zero_shot", "qualified")}
                      for a in arms]}, indent=1))


if __name__ == "__main__":
    main()
