"""Z6 (ADVISORY §2.7): choose the final's checkpoint by S among the evaluations that pass positional health, calibrate it on
dev (fast dev v4 only) and record the choice.

The final's driver writes `evaluations.json` at every evaluation (the S composite, the positional-health result in all three
orders and the knowledge-dev accuracy) and saves a checkpoint per evaluation. This script applies the pre-registered rule
(`meddecide.eval.osler_v6_select`), then:

* verifies the chosen checkpoint's directory exists;
* loads it with `PplxDecisionModel.load`, fits its per-question-type temperatures on **dev fast dev v4 only** (never a test,
  panel, MIMIC audit-suite or held-out item) and saves it as `checkpoints/chosen_step_XXXXXX` with the calibration;
* writes `selection.json` (the full S/positional table, the chosen step, the temperatures and the command).

If no evaluation passes positional health, the script writes the table with a BLOCKED outcome and exits non-zero; no
checkpoint is chosen by any fallback, and Z7 then reports the final as BLOCKED.

Usage (GPU; one job; after the final's run):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v6/z6_select.py \
        --arm-dir outputs/osler_v6/Z6/final > outputs/osler_v6/Z6/final/logs/select.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from meddecide.eval.osler_v6_select import select_by_s
from meddecide.model.pplx import PplxDecisionModel
from meddecide.train.config import load_config
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.trainer import Trainer
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, gpu_name, utcnow

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "osler_v5" / "arm_F2.yaml"
DEV = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm-dir", required=True)
    parser.add_argument("--out", default=None, help="where selection.json goes (default: the arm directory)")
    args = parser.parse_args()
    arm_dir = (REPO / args.arm_dir) if not Path(args.arm_dir).is_absolute() else Path(args.arm_dir)
    out = (REPO / args.out) if args.out and not Path(args.out).is_absolute() else (Path(args.out) if args.out else arm_dir)
    evaluations_path = arm_dir / "evaluations.json"
    if not evaluations_path.exists():
        raise SystemExit(f"{evaluations_path} is missing: the final's run has not produced evaluations")
    evaluations = json.loads(evaluations_path.read_text())
    selection = select_by_s(evaluations)
    payload = {
        "task": "Z6",
        "rule": "ADVISORY §2.7: highest S among the evaluations that pass positional health in all three orders",
        "arm_dir": str(arm_dir.relative_to(REPO)) if arm_dir.is_relative_to(REPO) else str(arm_dir),
        "selection": selection.to_dict(),
        "generated_utc": utcnow(),
        "git_commit": git_commit(REPO),
    }
    out.mkdir(parents=True, exist_ok=True)
    if selection.chosen_step is None:
        write_json(out / "selection.json", payload)
        raise SystemExit(f"BLOCKED: {selection.outcome}")
    checkpoint = arm_dir / "checkpoints" / f"step_{selection.chosen_step:06d}"
    if not checkpoint.exists():
        raise SystemExit(f"the chosen checkpoint {checkpoint} does not exist: stop")
    # the config supplies the evaluation batch sizes and the temperature fit only; the distill fields are irrelevant here
    config = dataclasses.replace(load_config(CONFIG), backbone="pplx", device="cuda:0", dtype="bfloat16")
    started = utcnow()
    model = PplxDecisionModel.load(checkpoint, device=config.device, dtype=config.dtype)
    rows = list(iter_jsonl(DEV))
    items = [mix_row_to_item(row) for row in rows]
    fits = Trainer(model, config).fit_calibration(items)
    chosen = arm_dir / "checkpoints" / f"chosen_step_{selection.chosen_step:06d}"
    model.save(chosen)
    payload.update({
        "started_utc": started,
        "finished_utc": utcnow(),
        "checkpoint": str(checkpoint.relative_to(REPO)),
        "chosen_checkpoint": str(chosen.relative_to(REPO)),
        "temperatures": dict(model.calibration),
        "temperature_fits": {name: (fit.to_dict() if hasattr(fit, "to_dict") else str(fit)) for name, fit in fits.items()},
        "calibration_dev_items": len(items),
        "calibration_dev_file": str(DEV.relative_to(REPO)),
        "gpu": gpu_name(),
        "command": " ".join(["python", "scripts/osler_v6/z6_select.py", "--arm-dir", str(arm_dir)]),
    })
    write_json(out / "selection.json", payload)
    print(f"[{utcnow()}] {selection.outcome}; temperatures {dict(model.calibration)}; saved {chosen.relative_to(REPO)}")


if __name__ == "__main__":
    main()
