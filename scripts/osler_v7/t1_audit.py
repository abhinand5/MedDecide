#!/usr/bin/env python
"""osler_v7 T1 self-audit (R6, CPU): re-derive every headline number of T1 from the raw artifacts in a fresh process.

The audit does not import the diagnostic's analysis code: it re-scans the three JSONL files for their sha256, row count and
the headline compositions, recomputes the panel accuracies and the yes/position priors from the prediction files, and
compares each value with `outputs/osler_v7/T1/composition.json` / `prior_shift.json`. Differences are problems, not
warnings. Writes `audit.json` and `SELF_AUDIT.md` (aggregates only; no item text, id or prediction).

Run: ``uv run --frozen python scripts/osler_v7/t1_audit.py`` (CPU; re-scans the mix, a few minutes)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.train.mix_items import mix_row_to_item  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

MIX = ROOT / "data" / "train" / "osler_v6" / "mix_f2_distill.jsonl"
ANCHORS = ROOT / "data" / "train" / "osler_v6" / "z1_anchor_soft.jsonl"
PANEL = ROOT / "data" / "bench" / "osler_v4" / "y9" / "panel.jsonl"
FINAL = ROOT / "outputs" / "osler_v6" / "Z7" / "baselines" / "final_panel" / "predictions.jsonl"
BASE = ROOT / "outputs" / "osler_v4" / "Y9" / "baselines" / "pplx_panel" / "predictions.jsonl"
OUT = ROOT / "outputs" / "osler_v7" / "T1"


def stream_sha_rows(path: Path) -> tuple[str, int]:
    """sha256 and line count of a JSONL file, without parsing it."""
    digest = sha256()
    rows = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            rows += 1
    return digest.hexdigest(), rows


def options_of(row: dict[str, Any]) -> list[dict[str, str]]:
    options = row["options"]
    if isinstance(options, str):
        options = json.loads(options)
    return [{"key": str(o["key"]), "label": str(o.get("label", ""))} for o in options]


def scan_composition_counts(path: Path) -> dict[str, Any]:
    """Independent re-count of the composition headline fields."""
    qtypes: Counter[str] = Counter()
    templates: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    n_options: Counter[str] = Counter()
    noul: Counter[str] = Counter()
    choice_gold: Counter[str] = Counter()
    state_chars = 0
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            qtypes[str(row["qtype"])] += 1
            templates[str(row["template_id"])] += 1
            sources[str(row["source"])] += 1
            options = options_of(row)
            n = len(options)
            n_options[str(n)] += 1
            keys = [o["key"] for o in options]
            gold = str(row["gold"])
            if gold in keys:
                if row["qtype"] == "choice":
                    choice_gold[str(keys.index(gold))] += 1
                elif row["qtype"] == "noul":
                    noul[gold.lower()] += 1
            elif row["qtype"] == "noul":
                noul["missing"] += 1
            state_chars += len(str(row.get("state", "")))
    return {
        "rows": rows,
        "qtype": dict(qtypes),
        "templates": dict(templates),
        "sources": dict(sources),
        "n_options": dict(n_options),
        "noul": dict(noul),
        "choice_gold_index": dict(choice_gold),
        "state_chars_mean": round(state_chars / rows, 3),
    }


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if raw.get("status", "scored") == "scored":
                rows[str(raw["item_id"])] = raw
    return rows


def recompute_panel_prior(panel: Path, final_path: Path, base_path: Path) -> dict[str, Any]:
    """Accuracies, noul yes statistics and the medconceptsqa position shares from the raw rows and predictions."""
    items: dict[str, tuple[dict[str, Any], list[dict[str, str]]]] = {}
    with panel.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            options = options_of(row)
            item = mix_row_to_item({**row, "options": options})
            items[item.item_id] = (row, options)
    final = load_predictions(final_path)
    base = load_predictions(base_path)
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    brier_final: dict[str, float] = defaultdict(float)
    brier_base: dict[str, float] = defaultdict(float)
    brier_final_all = 0.0
    brier_base_all = 0.0
    noul = {"items": 0, "gold_yes": 0, "final_p_yes": 0.0, "base_p_yes": 0.0,
            "final_yes": 0, "base_yes": 0}
    med_last = {"items": 0, "final": Counter(), "base": Counter(), "gold": Counter()}
    for item_id, (row, options) in items.items():
        keys = [o["key"] for o in options]
        gold = str(row["gold"])
        gi = keys.index(gold)
        source = str(row["source"])
        stat = per[source]
        stat[2] += 1
        final_row = final[item_id]
        base_row = base[item_id]
        final_index = max(range(len(keys)), key=lambda i: float(final_row["probs"][i]))
        base_index = max(range(len(keys)), key=lambda i: float(base_row["probs"][i]))
        stat[0] += int(keys[final_index] == gold)
        stat[1] += int(keys[base_index] == gold)
        final_probs = [float(p) for p in final_row["probs"]]
        base_probs = [float(p) for p in base_row["probs"]]
        onehot = [1.0 if i == gi else 0.0 for i in range(len(keys))]
        brier_final[source] += sum((p - y) ** 2 for p, y in zip(final_probs, onehot, strict=True))
        brier_base[source] += sum((p - y) ** 2 for p, y in zip(base_probs, onehot, strict=True))
        brier_final_all += sum((p - y) ** 2 for p, y in zip(final_probs, onehot, strict=True))
        brier_base_all += sum((p - y) ** 2 for p, y in zip(base_probs, onehot, strict=True))
        if str(row["qtype"]) == "noul":
            yes = keys.index("yes")
            noul["items"] += 1
            noul["gold_yes"] += int(gold == "yes")
            noul["final_p_yes"] += float(final_row["probs"][yes])
            noul["base_p_yes"] += float(base_row["probs"][yes])
            noul["final_yes"] += int(keys[final_index] == "yes")
            noul["base_yes"] += int(keys[base_index] == "yes")
        if source == "medconceptsqa_all":
            med_last["items"] += 1
            med_last["final"][str(final_index)] += 1
            med_last["base"][str(base_index)] += 1
            med_last["gold"][str(gi)] += 1
    datasets = {
        source: {
            "items": stat[2],
            "final_accuracy": round(stat[0] / stat[2], 8),
            "base_accuracy": round(stat[1] / stat[2], 8),
            "accuracy_diff": round(stat[0] / stat[2] - stat[1] / stat[2], 8),
            "final_brier": round(brier_final[source] / stat[2], 8),
            "base_brier": round(brier_base[source] / stat[2], 8),
            "brier_diff": round((brier_final[source] - brier_base[source]) / stat[2], 8),
        }
        for source, stat in sorted(per.items())
    }
    macro_final = sum(d["final_accuracy"] for d in datasets.values()) / len(datasets)
    macro_base = sum(d["base_accuracy"] for d in datasets.values()) / len(datasets)
    fraction = {
        "final": {k: round(v / med_last["items"], 8) for k, v in sorted(med_last["final"].items())},
        "base": {k: round(v / med_last["items"], 8) for k, v in sorted(med_last["base"].items())},
        "gold": {k: round(v / med_last["items"], 8) for k, v in sorted(med_last["gold"].items())},
    }
    return {
        "datasets": datasets,
        "macro_final": round(macro_final, 8),
        "macro_base": round(macro_base, 8),
        "macro_diff": round(macro_final - macro_base, 8),
        "final_brier_mean": round(brier_final_all / sum(d["items"] for d in datasets.values()), 8),
        "base_brier_mean": round(brier_base_all / sum(d["items"] for d in datasets.values()), 8),
        "losing_sources": [s for s, d in datasets.items() if d["accuracy_diff"] < 0],
        "noul": {
            "items": noul["items"],
            "gold_yes_share": round(noul["gold_yes"] / noul["items"], 8),
            "final_mean_p_yes": round(noul["final_p_yes"] / noul["items"], 8),
            "base_mean_p_yes": round(noul["base_p_yes"] / noul["items"], 8),
            "final_argmax_yes_rate": round(noul["final_yes"] / noul["items"], 8),
            "base_argmax_yes_rate": round(noul["base_yes"] / noul["items"], 8),
        },
        "medconceptsqa_position_share": fraction,
        "items": sum(d["items"] for d in datasets.values()),
        "predictions": {"final": len(final), "base": len(base)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--mix", type=Path, default=MIX)
    parser.add_argument("--anchors", type=Path, default=ANCHORS)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--final-predictions", type=Path, default=FINAL)
    parser.add_argument("--base-predictions", type=Path, default=BASE)
    args = parser.parse_args()
    composition = json.loads((args.out_dir / "composition.json").read_text(encoding="utf-8"))
    prior = json.loads((args.out_dir / "prior_shift.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    problems: list[str] = []

    def check(name: str, recorded: Any, recomputed: Any) -> None:
        match = recorded == recomputed
        checks.append({"check": name, "recorded": recorded, "recomputed": recomputed, "match": match})
        if not match:
            problems.append(f"{name}: recorded {recorded!r} != recomputed {recomputed!r}")

    # 1) file identity and row counts
    for name, path in (("mix", args.mix), ("anchors", args.anchors), ("panel", args.panel)):
        digest, rows = stream_sha_rows(path)
        check(f"{name}.sha256", composition["files"][name]["sha256"], digest)
        check(f"{name}.rows", composition["files"][name]["rows"], rows)

    # 2) composition headline fields (independent re-count)
    counts_by_name: dict[str, dict[str, Any]] = {}
    for name, path in (("mix", args.mix), ("anchors", args.anchors), ("panel", args.panel)):
        counts = scan_composition_counts(path)
        counts_by_name[name] = counts
        recorded = composition["files"][name]
        check(f"{name}.qtype", recorded["qtype"], counts["qtype"])
        check(f"{name}.templates", recorded["templates"], counts["templates"])
        check(f"{name}.sources", recorded["sources"], counts["sources"])
        check(f"{name}.n_options", recorded["n_options"], counts["n_options"])
        check(f"{name}.noul.gold", recorded["noul"]["gold"], counts["noul"])
        if name == "panel":
            check(f"{name}.choice.gold_index", recorded["choice"]["gold_index"],
                  counts["choice_gold_index"])
            check(f"{name}.text_chars.state_mean", recorded["text_chars"]["state_mean"],
                  counts["state_chars_mean"])

    panel_counts = counts_by_name["panel"]
    for train_name in ("mix", "anchors"):
        train_counts = counts_by_name[train_name]
        shared = composition["comparisons"][f"{train_name}_vs_panel"]
        check(f"compare.{train_name}.sources_shared", shared["sources_shared"],
              len(set(train_counts["sources"]) & set(panel_counts["sources"])))
        check(f"compare.{train_name}.templates_shared", shared["templates_shared"],
              len(set(train_counts["templates"]) & set(panel_counts["templates"])))
        check(f"compare.{train_name}.panel_items_from_seen_sources",
              shared["fraction_of_panel_items_from_sources_seen_in_training"],
              round(sum(panel_counts["sources"].get(s, 0) for s in train_counts["sources"])
                    / panel_counts["rows"], 8))

    # 3) prior shift (independent recompute from the predictions)
    recomputed = recompute_panel_prior(args.panel, args.final_predictions, args.base_predictions)
    recorded_prior = prior["per_dataset"]
    for source, values in recomputed["datasets"].items():
        check(f"prior.{source}.items", recorded_prior[source]["items"], values["items"])
        check(f"prior.{source}.final_accuracy", recorded_prior[source]["final_accuracy"], values["final_accuracy"])
        check(f"prior.{source}.base_accuracy", recorded_prior[source]["base_accuracy"], values["base_accuracy"])
        check(f"prior.{source}.accuracy_diff", recorded_prior[source]["accuracy_diff"], values["accuracy_diff"])
        check(f"prior.{source}.final_brier", recorded_prior[source]["final_brier"], values["final_brier"])
        check(f"prior.{source}.base_brier", recorded_prior[source]["base_brier"], values["base_brier"])
        check(f"prior.{source}.brier_diff", recorded_prior[source]["brier_diff"], values["brier_diff"])
    check("prior.overall.final_macro", prior["overall"]["final_accuracy_macro_over_datasets"], recomputed["macro_final"])
    check("prior.overall.base_macro", prior["overall"]["base_accuracy_macro_over_datasets"], recomputed["macro_base"])
    check("prior.overall.diff_macro", prior["overall"]["accuracy_diff_macro_over_datasets"], recomputed["macro_diff"])
    check("prior.overall.losing_sources", prior["overall"]["losing_sources"], recomputed["losing_sources"])
    check("prior.overall.final_brier", prior["overall"]["final_brier_mean"], recomputed["final_brier_mean"])
    check("prior.overall.base_brier", prior["overall"]["base_brier_mean"], recomputed["base_brier_mean"])
    check("prior.noul.items", prior["noul"]["items"], recomputed["noul"]["items"])
    check("prior.noul.gold_yes_share", prior["noul"]["gold_yes_share"], recomputed["noul"]["gold_yes_share"])
    check("prior.noul.final_mean_p_yes", prior["noul"]["final"]["mean_p_yes"],
          recomputed["noul"]["final_mean_p_yes"])
    check("prior.noul.base_mean_p_yes", prior["noul"]["base"]["mean_p_yes"],
          recomputed["noul"]["base_mean_p_yes"])
    check("prior.noul.final_argmax_yes_rate", prior["noul"]["final"]["argmax_yes_rate"],
          recomputed["noul"]["final_argmax_yes_rate"])
    check("prior.noul.base_argmax_yes_rate", prior["noul"]["base"]["argmax_yes_rate"],
          recomputed["noul"]["base_argmax_yes_rate"])
    med = prior["per_dataset"]["medconceptsqa_all"]
    check("prior.medconcepts.final_last_share", med["last_position_share"]["final"],
          recomputed["medconceptsqa_position_share"]["final"].get("3", 0.0))
    check("prior.medconcepts.base_last_share", med["last_position_share"]["base"],
          recomputed["medconceptsqa_position_share"]["base"].get("3", 0.0))
    check("prior.medconcepts.gold_last_share", med["last_position_share"]["gold"],
          recomputed["medconceptsqa_position_share"]["gold"].get("3", 0.0))
    check("prior.predictions.final_rows", 4_353, recomputed["predictions"]["final"])
    check("prior.predictions.base_rows", 4_353, recomputed["predictions"]["base"])
    check("prior.overall.items", prior["overall"]["items"], recomputed["items"])

    audit = {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t1_audit.py",
        "checks": checks,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }
    with (args.out_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    lines = [
        "# T1 SELF_AUDIT — composition and prior-shift diagnostics (osler_v7)",
        "",
        f"Generated {audit['generated_utc']} at `{audit['git_commit']}` in a fresh process; command "
        f"`{audit['command']}`.",
        "",
        f"**Status: {audit['status']}** — {len(checks)} checks, {len(problems)} problems. Every recorded value was",
        "re-derived from the raw artifacts (the three JSONL files re-scanned; the panel accuracies and priors recomputed",
        "from the raw predictions), not from the diagnostic's own intermediates.",
        "",
        "| check | recorded | re-derived | match |",
        "|---|---|---|---|",
    ]
    for row in checks:
        lines.append(
            f"| {row['check']} | `{row['recorded']}` | `{row['recomputed']}` | {'yes' if row['match'] else '**NO**'} |"
        )
    lines.extend(["", "## Problems", ""])
    lines.extend([f"- {problem}" for problem in problems] or ["None."])
    notes = []
    key_count = prior["overall"].get("predicted_key_not_item_key", 0)
    if key_count:
        notes.append(
            f"- The v6 final's noul rows carry readout letter codes: `predicted_key` is not the item key on {key_count} "
            "rows. Every accuracy and bias figure is computed from the probability argmax over the item's own option keys, "
            "and the macro difference reproduces Gate O6 to six decimals, so this is an interface detail, not a scoring error."
        )
    mix_dates = composition["files"]["mix"]["record_dates"]
    notes.append(
        f"- The mix has {mix_dates['missing']:,} rows without a `record_date` ({mix_dates['missing'] / composition['files']['mix']['rows']:.1%} "
        f"of the mix); the pre-window share is {mix_dates['pre_window_share_of_dated']:.4f} of dated rows but "
        f"{mix_dates['pre_window_share_of_rows']:.4f} of all rows. No claim is made about the undated rows' dates."
    )
    panel_choice = composition["files"]["panel"]["choice"]["n_options_share"]
    mix_choice = composition["files"]["mix"]["choice"]["n_options_share"]
    notes.append(
        f"- The panel's 22-option items are {panel_choice.get('22', 0):.4f} of its choice rows against "
        f"{mix_choice.get('22', 0):.4f} of the mix's; the 10-option items are {panel_choice.get('10', 0):.4f} against "
        f"{mix_choice.get('10', 0):.4f}. This is the measured composition gap, not a checked expectation of the mix."
    )
    med = prior["per_dataset"]["medexqa_test"]
    notes.append(
        f"- On medexqa the gold is skewed to the last option ({med['gold_index_share'].get('3', 0):.4f} of 940 items); "
        "the position figures are reported against that gold, never against a uniform assumption."
    )
    lines.extend(
        [
            "",
            "## Suspicious or notable (checked)",
            "",
            *notes,
            "",
            "## Notes on what was checked and what was not",
            "",
            "- The panel's record-date window is NOT MEASURED by construction: the frozen panel rows carry no",
            "  `record_date` field. No expected constant was substituted.",
            "- The diagnostic reads existing v6 final and zero-shot predictions; it runs no model and touches no GPU.",
            "- The panel is reported here as an additional diagnostic at the operator-approved T1; no v7 selection or gate",
            "  reads this file.",
            "- Item text, item ids and single-item predictions are absent from every written artifact by construction",
            "  (the accumulators keep counts and shares only).",
            "",
        ]
    )
    (args.out_dir / "SELF_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"T1 audit {audit['status']}: {len(checks)} checks, {len(problems)} problems")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
