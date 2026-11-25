"""O9: head choice at 4B (ADVISORY O9), on dev only. Never test, never held-out.

Reads the dev predictions the three O6-O8 arms wrote (``<arm>/dev_predictions.jsonl``, one row per dev item), aligns them by
item id, applies the pre-registered rule (``meddecide.eval.head_choice.choose_head``), and writes:
    outputs/osler_v0/O9/head_choice.json     the raw record (gitignored)
    loops/osler_v0/head_choice.md            the committed report (aggregates only)

Refuses to decide when an arm is missing, when an arm's dev rows do not cover the same items, or when the gold labels of
a shared item differ between arms (no silent drops; every count is printed).

Usage:
    uv run --frozen python scripts/osler/o9_head_choice.py
    uv run --frozen python scripts/osler/o9_head_choice.py --root <dir>   # smoke test on another output root
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from meddecide.eval.head_choice import BASELINE, MARGIN, N_RESAMPLES, DevScores, choose_head
from meddecide.eval.metrics import template_macro_accuracy
from meddecide.eval.paired import item_brier
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
# Each arm's directory under outputs/osler_v0, as its config writes it (configs/osler_v0/arm_<arm>.yaml, output_dir).
# tests/test_o9_arm_paths.py keeps the two in step; the O6-O8 task labels are not directories.
ARM_DIRS = {"L": "O6/arm_L", "P": "O6/arm_P", "N": "O6/arm_N"}
SEED = 0


def load_arm(root: Path, name: str) -> dict[str, dict]:
    path = root / ARM_DIRS[name] / "dev_predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"arm {name} has no dev predictions at {path}; train and score it first")
    return {row["item_id"]: row for row in iter_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO / "outputs/osler_v0")
    parser.add_argument("--report", type=Path, default=REPO / "loops/osler_v0/head_choice.md")
    args = parser.parse_args()
    arms = {name: load_arm(args.root, name) for name in ARM_DIRS}
    counts = {name: len(rows) for name, rows in arms.items()}
    shared = sorted(set.intersection(*(set(rows) for rows in arms.values())))
    if any(len(rows) != len(shared) for rows in arms.values()):
        raise ValueError(f"the arms' dev rows do not cover the same items: rows {counts}, shared {len(shared)}")
    for item_id in shared:
        golds = {arms[name][item_id]["gold_key"] for name in ARM_DIRS}
        if len(golds) != 1:
            raise ValueError(f"item {item_id} has different gold labels across arms: {sorted(golds)}")

    scores: dict[str, DevScores] = {}
    summary: dict[str, dict] = {}
    gold_keys = [arms["L"][i]["gold_key"] for i in shared]
    templates = np.array([arms["L"][i]["template_id"] for i in shared], dtype=object)
    for name in ARM_DIRS:
        rows = [arms[name][i] for i in shared]
        correct = np.array([float(r["correct"]) for r in rows])
        scores[name] = DevScores(name=name, correct=correct, gold=np.array(gold_keys, dtype=object),
                                 templates=templates)
        briers = [item_brier(r["probs"], r["gold_index"]) for r in rows]
        summary[name] = {"items": len(rows), "macro_accuracy": template_macro_accuracy(correct, templates),
                         "accuracy": float(correct.mean()), "mean_brier": float(np.mean(briers))}

    choice = choose_head(scores[BASELINE], [scores["P"], scores["N"]], n_resamples=N_RESAMPLES, seed=SEED)
    record = {
        "kind": "osler_v0_O9_head_choice",
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "command": "scripts/osler/o9_head_choice.py",
        "items_shared": len(shared),
        "rule": {"baseline": BASELINE, "margin": MARGIN, "bootstrap_resamples": N_RESAMPLES, "seed": SEED,
                 "stratum": "template_id", "statistic": "macro accuracy (unweighted mean over templates of per-template accuracy; G1/D24)"},
        "arms": summary,
        "challenges": [{"arm": c.name, "point": c.difference.point, "ci_lo": c.difference.lo, "ci_hi": c.difference.hi,
                        "n": c.difference.n, "qualifies": c.qualifies} for c in choice.challenges],
        "chosen": choice.chosen,
        "reason": choice.reason,
    }
    (args.root / "O9").mkdir(parents=True, exist_ok=True)
    write_json(args.root / "O9/head_choice.json", record)

    lines = [
        "# O9 — head choice at 4B (dev only)",
        "",
        f"Rule (ADVISORY O9): start with L; switch to P or N only if its dev macro accuracy exceeds L's by at least "
        f"{MARGIN * 100:.1f} points with a paired bootstrap lower bound above 0 on dev ({N_RESAMPLES} resamples, items "
        f"resampled within templates, seed {SEED}). If both qualify, take the larger gain. Macro accuracy is the mean over "
        "templates of per-template accuracy (G1's and D24's statistic).",
        "",
        f"Dev items shared by all three arms: {len(shared)}.",
        "",
        "| arm | macro accuracy | accuracy | mean Brier |",
        "|---|---|---|---|",
    ]
    for name in ARM_DIRS:
        s = summary[name]
        lines.append(f"| {name} | {s['macro_accuracy']:.4f} | {s['accuracy']:.4f} | {s['mean_brier']:.4f} |")
    lines += ["", "| challenger minus L | point | 95 % lower | 95 % upper | qualifies |", "|---|---|---|---|---|"]
    for c in choice.challenges:
        lines.append(f"| {c.name} | {c.difference.point:+.4f} | {c.difference.lo:+.4f} | {c.difference.hi:+.4f} | "
                     f"{'yes' if c.qualifies else 'no'} |")
    lines += ["", f"**Chosen head: {choice.chosen}.** {choice.reason}.", ""]
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"O9: chosen {choice.chosen} ({choice.reason}); items {len(shared)}; wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
