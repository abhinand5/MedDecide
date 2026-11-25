"""Defect screen of one run (osler_v1 goal: defects are investigated first): the checks of meddecide.train.blocks.defect_screen on
the run's fast-dev predictions and its logged training losses. Writes <run-dir>/defect_screen.json (aggregates only: counts,
means, flags and the names of templates). Nothing is changed by a flag (R8).

Usage:
    uv run --frozen python scripts/osler_v1/v8_defect_screen.py --run-dir outputs/osler_v1/V8/B3
    uv run --frozen python scripts/osler_v1/v8_defect_screen.py --run-dir outputs/osler_v1/V9/osler_4b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from meddecide.train.blocks import defect_screen
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir if args.run_dir.is_absolute() else REPO / args.run_dir
    predictions = list(iter_jsonl(run / "dev_predictions.jsonl"))
    steps = [json.loads(line) for line in (run / "logs" / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    losses = [float(r["loss"]) for r in steps if r.get("event") == "step"]
    report = defect_screen(predictions, losses)
    report |= {"written_utc": utcnow(), "git_commit": git_commit(REPO), "dev_predictions": len(predictions)}
    write_json(run / "defect_screen.json", report)
    print(f"{run.name}: {report['items']} dev items, accuracy {report['accuracy']:.4f}, mean chance {report['mean_chance']:.4f}; "
          f"{report['logged_losses']} logged losses ({report['nonfinite_losses']} non-finite); flags {report['flags']}", flush=True)
    for key in ("constant_answer_templates", "too_good_templates"):
        if report[key]:
            print(f"  {key}: {', '.join(report[key])}", flush=True)
    print(f"wrote {(run / 'defect_screen.json').relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
