"""O2 investigation: the pplx-decider v0.2 accuracy (0.861) split by tier, with the same metrics the scoreboard uses.

For every model that scored v0.2, the prediction rows are split by the item's tier (``meta.tier`` in the common item file:
``fresh`` or ``established``, the latter being the public tier-1 sets) and summarised with ``meddecide.eval.o2_metrics.summarise``
(accuracy with its bootstrap interval, template macro as G1 defines it, Brier, coverage). It also lists the templates where
the first model is most accurate, with every model's accuracy on those same templates, so shortcut-solvable templates are
visible.

Outputs:
    outputs/osler_v0/O2/tier_breakdown.json  (raw; gitignored)
    printed tables (copied into the O2 SELF_AUDIT)

Usage:
    uv run --frozen python scripts/osler/o2_tier_breakdown.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from meddecide.eval.o2_metrics import summarise
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "data/bench/v0.3_ext/o2_items.jsonl"
OUT_DIR = REPO / "outputs/osler_v0/O2"
MODELS = ("pplx-decider-v1.1-27b", "jev-27b", "clef_flash", "meddecider-4b", "meddecider-9b")
TIERS = ("fresh", "established")
TOP_N = 8
MIN_N = 100


def item_tiers() -> dict[str, str]:
    tiers: dict[str, str] = {}
    with ITEMS.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["benchmark"] == "v0.2":
                tiers[row["item_id"]] = row["meta"]["tier"]
    return tiers


def template_accuracies(v02_rows: list[dict], tiers: dict[str, str]) -> dict[tuple[str, str], tuple[float, int]]:
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in v02_rows:
        if r["status"] != "scored":
            continue
        pred = r["option_keys"][max(range(len(r["probs"])), key=lambda i: r["probs"][i])]
        cells[(tiers.get(r["item_id"], "?"), r["template_id"])].append(float(pred == r["gold"]))
    return {k: (sum(v) / len(v), len(v)) for k, v in cells.items()}


def main() -> None:
    tiers = item_tiers()
    record: dict = {"kind": "osler_v0_O2_tier_breakdown", "built_at_utc": utcnow(), "git_commit": git_commit(REPO),
                    "command": "scripts/osler/o2_tier_breakdown.py", "models": {}}
    per_model_templates: dict[str, dict[tuple[str, str], tuple[float, int]]] = {}
    print("| model | tier | n in scope | scored | accuracy (95 % CI) | template macro | Brier | coverage |")
    print("|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        path = OUT_DIR / model / "predictions.jsonl"
        if not path.exists():
            print(f"| {model} | — | not available | | | | | |")
            continue
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        v02 = [r for r in rows if r["benchmark"] == "v0.2"]
        record["models"][model] = {}
        per_model_templates[model] = template_accuracies(v02, tiers)
        for tier in TIERS:
            scoped = [r for r in v02 if tiers.get(r["item_id"]) == tier]
            s = summarise(scoped)
            record["models"][model][tier] = {
                "n_in_scope": s.n_in_scope, "n_scored": s.n_scored, "accuracy": s.accuracy,
                "accuracy_ci95": s.accuracy_ci95, "macro_accuracy": s.macro_accuracy, "brier": s.brier,
                "coverage": s.coverage, "n_templates": s.n_templates,
            }
            acc = f"{s.accuracy:.4f} ({s.accuracy_ci95[0]:.4f} to {s.accuracy_ci95[1]:.4f})" if s.accuracy is not None else "n/a"
            macro = f"{s.macro_accuracy:.4f}" if s.macro_accuracy is not None else "n/a"
            brier = f"{s.brier:.4f}" if s.brier is not None else "n/a"
            print(f"| {model} | {tier} | {s.n_in_scope} | {s.n_scored} | {acc} | {macro} | {brier} | {s.coverage:.4f} |")

    # the templates where the first model is most accurate, with every model's accuracy on the same templates
    lead = MODELS[0]
    if lead in per_model_templates:
        top = sorted(((k, acc, n) for k, (acc, n) in per_model_templates[lead].items() if n >= MIN_N),
                     key=lambda t: -t[1])[:TOP_N]
        record["top_templates_first_model"] = []
        print(f"\nTop v0.2 templates for {lead} (n >= {MIN_N}), with each model's accuracy on the same template:")
        print("| tier | template | n | " + " | ".join(m for m in per_model_templates) + " |")
        print("|---|---|---|" + "---|" * len(per_model_templates))
        for (tier, template), _, n in top:
            accs = {m: per_model_templates[m].get((tier, template), (float("nan"), 0))[0] for m in per_model_templates}
            record["top_templates_first_model"].append({"tier": tier, "template_id": template, "n": n,
                                                        "accuracy_by_model": accs})
            print(f"| {tier} | {template} | {n} | " + " | ".join(f"{accs[m]:.3f}" for m in per_model_templates) + " |")
    write_json(OUT_DIR / "tier_breakdown.json", record)
    print(f"\nwrote {(OUT_DIR / 'tier_breakdown.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
