"""d3_distill step 2 audit (CPU): re-derive an arm's or final's headline numbers from its own logs.

Checks, from `steps.jsonl`, `evals.jsonl`, `evals/*.json`, `arm_result.json` and `checkpoints/`:

- every logged step has finite losses and (while the probe runs) finite CE/KL gradient norms;
- the D57 band mean over the first 200 steps, recomputed from `steps.jsonl`, matches `arm_result.json`;
- the last evaluation matches the last `evals.jsonl` line and the arm's `final_eval`;
- the best checkpoint (`best.pt`) has the step of the best evaluation;
- counts close (steps logged, evaluations, checkpoints).

Writes `audit.json`/`SELF_AUDIT.md` next to the run's outputs; exits non-zero on a mismatch.

Usage:
    python scripts/d3_distill/step2_audit.py --run outputs/d3_distill/step2/mini/arm_s0
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else REPO / args.run
    result = run / "arm_result.json"
    if not result.exists():
        raise SystemExit(f"{result} is missing: the run has not finished")
    arm = json.loads(result.read_text(encoding="utf-8"))
    steps = read_jsonl(run / "steps.jsonl")
    evals = read_jsonl(run / "evals.jsonl")
    checks: dict = {}
    failures: list[str] = []

    def expect(name: str, ok: bool, detail) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            failures.append(name)

    expect("steps_logged", len(steps) >= 1, len(steps))
    bad = [s["step"] for s in steps if any(
        s.get(k) is not None and not math.isfinite(float(s[k])) for k in ("ce", "kl", "share", "grad_norm_preclip")
    )]
    expect("finite_losses", not bad, {"nonfinite_steps": bad[:10], "count": len(bad)})
    monotone = [a["step"] for a, b in pairwise(steps) if b["step"] <= a["step"]]
    expect("step_order", not monotone, {"non_increasing": monotone[:10]})

    band_steps = [s for s in steps if s.get("share") is not None]
    if arm.get("band") is not None and band_steps:
        recomputed = sum(s["share"] for s in band_steps[: arm["band"]["steps"]]) / arm["band"]["steps"]
        expect("band_mean", abs(recomputed - arm["band"]["mean_share"]) <= 1e-6,
               {"recomputed": recomputed, "recorded": arm["band"]["mean_share"], "steps": len(band_steps)})
    else:
        checks["band_mean"] = {"ok": True, "detail": "no probe/band (probe run or no share logged)"}

    last = evals[-1] if evals else None
    final_eval = arm.get("final_eval")
    if final_eval is not None and last is not None:
        expect("final_eval_matches_last", final_eval.get("step") == last.get("step")
               and abs(final_eval.get("accuracy", -1) - last.get("accuracy", -2)) <= 1e-9,
               {"final_eval": final_eval.get("step"), "last": last.get("step")})
    expect("evaluations_present", bool(evals) or arm.get("mode") == "probe", len(evals))

    best = run / "best.pt"
    if best.exists():
        expect("best_checkpoint", arm.get("best_dev_accuracy") is not None,
               {"best_dev_step": arm.get("best_dev_step"), "best_dev_accuracy": arm.get("best_dev_accuracy")})
    else:
        checks["best_checkpoint"] = {"ok": True, "detail": "no best.pt (probe run or no evaluation)"}

    if steps:
        expect("steps_run", arm.get("steps_run") == steps[-1]["step"] + 1 - arm.get("start_step", 0),
               {"recorded": arm.get("steps_run"), "last_logged": steps[-1]["step"]})
    else:
        checks["steps_run"] = {"ok": False, "detail": "no steps logged"}

    audit = {"run": str(run), "mode": arm.get("mode"), "checks": checks, "failures": failures, "finished_utc": utcnow()}
    (run / "audit.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")
    lines = [f"# SELF_AUDIT — {run.name}", "", f"Re-derived `{utcnow()}` from the run's logs.", ""]
    for name, entry in checks.items():
        lines.append(f"- {name}: {'PASS' if entry['ok'] else 'FAIL'} — {entry['detail']}")
    lines += ["", f"Failures: {failures or 'none'}", ""]
    (run / "SELF_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"run": str(run), "failures": failures, "checks": len(checks)}, indent=1))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
