#!/usr/bin/env python
"""osler_v7 T1 — composition and prior-shift diagnostics on the panel (CPU, aggregate only).

Two questions, as pre-registered in `loops/osler_v7/STATE.md` §1 for T1:

1. **Composition.** How do the shapes of the osler_v6 training mix (1,456,438 rows) and of the 385,536 anchor
   soft-label rows compare with the shapes of the frozen external panel (4,353 items)? Shapes are question type,
   option count, label balance (noul gold yes share, choice gold position, score level) and text length. The panel
   has no record dates, so the date-window comparison is reported as NOT MEASURED, never substituted.
2. **Prior shift.** On the panel, for each source (dataset): the osler_v6 final's and the zero-shot
   pplx-decider-flash-v1.2's accuracy on the same items, the predicted-yes rate and yes/no confusion on the noul
   items, and the predicted option-position distribution against the gold positions on the choice items.

This is a diagnostic, not a gate and not a selection input: it reports where the v6 model loses and whether the
losses line up with priors that the mix carries. Nothing here is used to choose a v7 arm, a checkpoint or a
temperature (SELECTION is on the format-matched dev set, ADVISORY §7). Only aggregate counts, shares and
per-dataset accuracies are written; no item text, id or prediction is written to the output or to git.

Reads: `data/train/osler_v6/mix_f2_distill.jsonl`, `data/train/osler_v6/z1_anchor_soft.jsonl`,
`data/bench/osler_v4/y9/panel.jsonl`, `outputs/osler_v6/Z7/baselines/final_panel/predictions.jsonl`,
`outputs/osler_v4/Y9/baselines/pplx_panel/predictions.jsonl`.
Writes: `outputs/osler_v7/T1/composition.json`, `prior_shift.json`, `run_provenance.json`.

Run: ``uv run --frozen python scripts/osler_v7/t1_diagnose.py`` (CPU; the mix scan takes a few minutes)
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import date
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

# Pinned expectations: failing one of these aborts the run rather than reporting a wrong number.
EXPECTED = {
    "mix": ("91ff5362d56c7b14aafa1bb62e8d78fd31312c613ff5d310892f7e09c97137ce", 1_456_438),
    "anchors": ("edd4c2070b7d16280cf388c2634910a408d59be5b97cbf51e007900fdd494743", 385_536),
    "panel": ("84836287043543c864527c9714780622282794be3fc210a30fd1e51aac30be99", 4_353),
}

PRE_WINDOW = date(2026, 3, 1)
TEXT_BUCKETS = (100, 250, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000)
REL_BINS = ("0_25", "25_50", "50_75", "75_100")


# ------------------------------------------------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_osler_v7_t1.py; no file access below this block until the readers)
# ------------------------------------------------------------------------------------------------------------------


def parse_options(row: dict[str, Any]) -> list[dict[str, str]]:
    """The row's options as a list of ``{"key", "label"}``; accepts the mix's JSON string and the panel's list."""
    options = row["options"]
    if isinstance(options, str):
        options = json.loads(options)
    return [{"key": str(o["key"]), "label": str(o.get("label", ""))} for o in options]


def gold_index(row: dict[str, Any], options: list[dict[str, str]]) -> int | None:
    """Index of the gold key in the row's option order, or None when the gold is not one of the options."""
    keys = [o["key"] for o in options]
    gold = str(row["gold"])
    return keys.index(gold) if gold in keys else None


def relative_position(index: int, n_options: int) -> float:
    """Where an option index sits in ``[0, 1]``: 0 is the first option, 1 the last."""
    return 0.0 if n_options <= 1 else index / (n_options - 1)


def relative_bin(relative: float) -> str:
    """The quartile of a relative position, with 1.0 falling in the last quartile."""
    for limit, label in ((0.25, REL_BINS[0]), (0.5, REL_BINS[1]), (0.75, REL_BINS[2])):
        if relative < limit:
            return label
    return REL_BINS[3]


def char_bucket(n_chars: int) -> str:
    """A coarse text-length bucket label for a character count."""
    lower = 0
    for upper in TEXT_BUCKETS:
        if n_chars < upper:
            return f"{lower}_{upper - 1}"
        lower = upper
    return f"ge{TEXT_BUCKETS[-1]}"


def total_variation(a: dict[str, float], b: dict[str, float]) -> float:
    """Total-variation distance between two categorical distributions over the union of their keys."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys)


def new_composition() -> dict[str, Any]:
    """An empty accumulator for ``absorb``."""
    return {
        "rows": 0,
        "qtype": Counter(),
        "templates": Counter(),
        "sources": Counter(),
        "splits": Counter(),
        "tiers": Counter(),
        "n_options": Counter(),
        "choice_n_options": Counter(),
        "choice_rows": 0,
        "choice_gold_index": Counter(),
        "choice_gold_missing": 0,
        "choice_gold_rel_bins": Counter(),
        "choice_gold_rel_sum": 0.0,
        "choice_gold_rel_n": 0,
        "noul_rows": 0,
        "noul_gold": Counter(),
        "noul_gold_missing": 0,
        "score_rows": 0,
        "score_gold_index": Counter(),
        "score_gold_missing": 0,
        "other_qtypes": Counter(),
        "labelled_rows": 0,
        "state_chars_sum": 0,
        "question_chars_sum": 0,
        "state_buckets": Counter(),
        "question_buckets": Counter(),
        "dated_rows": 0,
        "pre_window_rows": 0,
        "missing_dates": 0,
        "bad_dates": 0,
        "date_min": None,
        "date_max": None,
    }


def absorb(comp: dict[str, Any], row: dict[str, Any], *, with_dates: bool) -> None:
    """Add one row to the composition accumulator (never raises on a malformed optional field: it counts it)."""
    options = parse_options(row)
    n = len(options)
    qtype = str(row["qtype"])
    comp["rows"] += 1
    comp["qtype"][qtype] += 1
    comp["templates"][str(row["template_id"])] += 1
    comp["sources"][str(row["source"])] += 1
    comp["splits"][str(row.get("split", ""))] += 1
    comp["tiers"][str(row.get("tier", ""))] += 1
    if row.get("teacher_probs") is not None:
        comp["labelled_rows"] += 1
    comp["n_options"][str(n)] += 1
    gi = gold_index(row, options)
    if qtype == "choice":
        comp["choice_rows"] += 1
        comp["choice_n_options"][str(n)] += 1
        if gi is None:
            comp["choice_gold_missing"] += 1
        else:
            comp["choice_gold_index"][str(gi)] += 1
            relative = relative_position(gi, n)
            comp["choice_gold_rel_bins"][relative_bin(relative)] += 1
            comp["choice_gold_rel_sum"] += relative
            comp["choice_gold_rel_n"] += 1
    elif qtype == "noul":
        comp["noul_rows"] += 1
        if gi is None:
            comp["noul_gold_missing"] += 1
        else:
            comp["noul_gold"][str(row["gold"]).lower()] += 1
    elif qtype == "score":
        comp["score_rows"] += 1
        if gi is None:
            comp["score_gold_missing"] += 1
        else:
            comp["score_gold_index"][str(gi)] += 1
    else:
        comp["other_qtypes"][qtype] += 1
    state_len = len(str(row.get("state", "")))
    question_len = len(str(row.get("question", "")))
    comp["state_chars_sum"] += state_len
    comp["question_chars_sum"] += question_len
    comp["state_buckets"][char_bucket(state_len)] += 1
    comp["question_buckets"][char_bucket(question_len)] += 1
    if with_dates:
        raw = row.get("record_date")
        if not raw:
            comp["missing_dates"] += 1
        else:
            try:
                when = date.fromisoformat(str(raw)[:10])
            except ValueError:
                comp["bad_dates"] += 1
            else:
                comp["dated_rows"] += 1
                if when < PRE_WINDOW:
                    comp["pre_window_rows"] += 1
                if comp["date_min"] is None or when < comp["date_min"]:
                    comp["date_min"] = when
                if comp["date_max"] is None or when > comp["date_max"]:
                    comp["date_max"] = when


def _shares(counter: Counter[str], total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {k: round(v / total, 8) for k, v in sorted(counter.items())}


def finalise_composition(comp: dict[str, Any], *, with_dates: bool) -> dict[str, Any]:
    """Turn an accumulator into the JSON-safe composition block of one file."""
    rows = comp["rows"]
    noul_total = sum(comp["noul_gold"].values())
    out: dict[str, Any] = {
        "rows": rows,
        "qtype": dict(sorted(comp["qtype"].items())),
        "qtype_share": _shares(comp["qtype"], rows),
        "templates": dict(sorted(comp["templates"].items())),
        "sources": dict(sorted(comp["sources"].items())),
        "splits": dict(sorted(comp["splits"].items())),
        "tiers": dict(sorted(comp["tiers"].items())),
        "n_options": dict(sorted(comp["n_options"].items(), key=lambda kv: int(kv[0]))),
        "n_options_share": {
            k: round(v / rows, 8) for k, v in sorted(comp["n_options"].items(), key=lambda kv: int(kv[0]))
        },
        "choice": {
            "rows": comp["choice_rows"],
            "n_options": dict(sorted(comp["choice_n_options"].items(), key=lambda kv: int(kv[0]))),
            "n_options_share": _shares(comp["choice_n_options"], comp["choice_rows"]),
            "gold_index": dict(sorted(comp["choice_gold_index"].items(), key=lambda kv: int(kv[0]))),
            "gold_relative_mean": (
                round(comp["choice_gold_rel_sum"] / comp["choice_gold_rel_n"], 8)
                if comp["choice_gold_rel_n"]
                else None
            ),
            "gold_relative_bins": _shares(comp["choice_gold_rel_bins"], comp["choice_gold_rel_n"]),
        },
        "noul": {
            "rows": comp["noul_rows"],
            "gold": dict(sorted(comp["noul_gold"].items())),
            "gold_missing": comp["noul_gold_missing"],
            "yes_share": round(comp["noul_gold"].get("yes", 0) / noul_total, 8) if noul_total else None,
        },
        "score": {
            "rows": comp["score_rows"],
            "gold_index": dict(sorted(comp["score_gold_index"].items(), key=lambda kv: int(kv[0]))),
        },
        "other_qtypes": dict(sorted(comp["other_qtypes"].items())),
        "labelled_rows": comp["labelled_rows"],
        "labelled_share": round(comp["labelled_rows"] / rows, 8) if rows else None,
        "text_chars": {
            "state_mean": round(comp["state_chars_sum"] / rows, 3) if rows else None,
            "question_mean": round(comp["question_chars_sum"] / rows, 3) if rows else None,
            "state_buckets": _shares(comp["state_buckets"], rows),
            "question_buckets": _shares(comp["question_buckets"], rows),
        },
    }
    if with_dates:
        out["record_dates"] = {
            "dated_rows": comp["dated_rows"],
            "missing": comp["missing_dates"],
            "bad": comp["bad_dates"],
            "min": comp["date_min"].isoformat() if comp["date_min"] else None,
            "max": comp["date_max"].isoformat() if comp["date_max"] else None,
            "pre_window_share_of_dated": (
                round(comp["pre_window_rows"] / comp["dated_rows"], 8) if comp["dated_rows"] else None
            ),
            "pre_window_share_of_rows": round(comp["pre_window_rows"] / rows, 8) if rows else None,
        }
    else:
        out["record_dates"] = {
            "status": "NOT MEASURED — the panel rows carry no record_date field; no expected constant is substituted"
        }
    return out


def composition_compare(train: dict[str, Any], panel: dict[str, Any]) -> dict[str, Any]:
    """Share gaps and total-variation distances between one training file and the panel."""
    qtype_gaps = {
        k: round(train["qtype_share"].get(k, 0.0) - panel["qtype_share"].get(k, 0.0), 6)
        for k in sorted(set(train["qtype_share"]) | set(panel["qtype_share"]))
    }
    option_gaps = {
        k: round(train["choice"]["n_options_share"].get(k, 0.0) - panel["choice"]["n_options_share"].get(k, 0.0), 6)
        for k in sorted(
            set(train["choice"]["n_options_share"]) | set(panel["choice"]["n_options_share"]), key=int
        )
    }
    return {
        "qtype_tvd": round(total_variation(train["qtype_share"], panel["qtype_share"]), 6),
        "choice_option_count_tvd": round(
            total_variation(train["choice"]["n_options_share"], panel["choice"]["n_options_share"]), 6
        ),
        "qtype_share_gaps": qtype_gaps,
        "choice_option_count_gaps": option_gaps,
        "noul_yes_share_gap": (
            round(train["noul"]["yes_share"] - panel["noul"]["yes_share"], 6)
            if train["noul"]["yes_share"] is not None and panel["noul"]["yes_share"] is not None
            else None
        ),
        "choice_gold_relative_mean_gap": (
            round(train["choice"]["gold_relative_mean"] - panel["choice"]["gold_relative_mean"], 6)
            if train["choice"]["gold_relative_mean"] is not None and panel["choice"]["gold_relative_mean"] is not None
            else None
        ),
        "state_chars_mean_gap": round(train["text_chars"]["state_mean"] - panel["text_chars"]["state_mean"], 3),
        "question_chars_mean_gap": round(
            train["text_chars"]["question_mean"] - panel["text_chars"]["question_mean"], 3
        ),
        "templates_shared": len(set(train["templates"]) & set(panel["templates"])),
        "sources_shared": len(set(train["sources"]) & set(panel["sources"])),
        "fraction_of_panel_items_from_templates_seen_in_training": round(
            sum(
                panel["templates"].get(t, 0) for t in train["templates"]
            )
            / panel["rows"],
            8,
        )
        if panel["rows"]
        else None,
        "fraction_of_panel_items_from_sources_seen_in_training": round(
            sum(panel["sources"].get(s, 0) for s in train["sources"]) / panel["rows"], 8
        )
        if panel["rows"]
        else None,
    }


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Scored prediction rows keyed by item id; unscored rows are skipped and counted by the caller from the scan."""
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if raw.get("status", "scored") != "scored":
                continue
            rows[str(raw["item_id"])] = raw
    return rows


def load_panel_items(path: Path) -> list[dict[str, Any]]:
    """Panel rows with their options parsed and the trainer's recomputed item id (the id the predictions use)."""
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            options = parse_options(raw)
            item = mix_row_to_item({**raw, "options": options})
            items.append({**raw, "_options": options, "_item_id": item.item_id})
    if len({i["_item_id"] for i in items}) != len(items):
        raise ValueError("panel: recomputed item ids are not unique")
    return items


def _accuracy(correct: int, items: int) -> float | None:
    return round(correct / items, 8) if items else None


def analyse_panel(
    items: list[dict[str, Any]],
    final: dict[str, dict[str, Any]],
    base: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Per-source accuracy and yes/position priors for the final against the zero-shot base on the same items."""
    per: dict[str, dict[str, Any]] = {}
    missing = 0
    argmax_disagreements = 0
    overall = {"items": 0, "final_correct": 0, "base_correct": 0, "final_brier_sum": 0.0, "base_brier_sum": 0.0}
    noul = {
        "items": 0,
        "gold_yes": 0,
        "final": {"p_yes_sum": 0.0, "yes_argmax": 0, "correct": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "base": {"p_yes_sum": 0.0, "yes_argmax": 0, "correct": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0},
    }
    for item in items:
        item_id = item["_item_id"]
        if item_id not in final or item_id not in base:
            missing += 1
            continue
        keys = [o["key"] for o in item["_options"]]
        n = len(keys)
        gold = str(item["gold"])
        gi = keys.index(gold) if gold in keys else None
        if gi is None:
            raise ValueError(f"panel item {item_id}: gold key is not among the options")
        source = str(item["source"])
        stat = per.setdefault(
            source,
            {
                "qtype": str(item["qtype"]),
                "items": 0,
                "n_options_seen": Counter(),
                "final_correct": 0,
                "base_correct": 0,
                "final_brier_sum": 0.0,
                "base_brier_sum": 0.0,
                "gold_index": Counter(),
                "final_index": Counter(),
                "base_index": Counter(),
                "gold_rel_sum": 0.0,
                "final_rel_sum": 0.0,
                "base_rel_sum": 0.0,
                "gold_rel_bins": Counter(),
                "final_rel_bins": Counter(),
                "base_rel_bins": Counter(),
                "acc_by_gold_index": defaultdict(lambda: [0, 0, 0]),
            },
        )
        stat["items"] += 1
        stat["n_options_seen"][str(n)] += 1
        overall["items"] += 1
        final_row = final[item_id]
        base_row = base[item_id]
        predictions: dict[str, str] = {}
        for tag, row in (("final", final_row), ("base", base_row)):
            probs = [float(p) for p in row["probs"]]
            if len(probs) != n:
                raise ValueError(f"{tag} {item_id}: {len(probs)} probabilities for {n} options")
            index = int(max(range(n), key=probs.__getitem__))
            predicted = keys[index]
            if predicted != str(row["predicted_key"]):
                argmax_disagreements += 1
            correct = int(predicted == gold)
            brier = sum((prob - (1.0 if i == gi else 0.0)) ** 2 for i, prob in enumerate(probs))
            stat[f"{tag}_brier_sum"] += brier
            overall[f"{tag}_brier_sum"] += brier
            predictions[tag] = predicted
            stat[f"{tag}_correct"] += correct
            stat[f"{tag}_index"][str(index)] += 1
            relative = relative_position(index, n)
            stat[f"{tag}_rel_sum"] += relative
            stat[f"{tag}_rel_bins"][relative_bin(relative)] += 1
            overall[f"{tag}_correct"] += correct
        stat["gold_index"][str(gi)] += 1
        gold_relative = relative_position(gi, n)
        stat["gold_rel_sum"] += gold_relative
        stat["gold_rel_bins"][relative_bin(gold_relative)] += 1
        stat["acc_by_gold_index"][str(gi)][0] += int(predictions["final"] == gold)
        stat["acc_by_gold_index"][str(gi)][1] += int(predictions["base"] == gold)
        stat["acc_by_gold_index"][str(gi)][2] += 1
        if str(item["qtype"]) == "noul":
            yes = keys.index("yes")
            gold_yes = gold == "yes"
            noul["items"] += 1
            noul["gold_yes"] += int(gold_yes)
            for tag, row in (("final", final_row), ("base", base_row)):
                p_yes = float(row["probs"][yes])
                predicted_yes = predictions[tag] == "yes"
                noul[tag]["p_yes_sum"] += p_yes
                noul[tag]["yes_argmax"] += int(predicted_yes)
                noul[tag]["correct"] += int(predicted_yes == gold_yes)
                if predicted_yes and gold_yes:
                    noul[tag]["tp"] += 1
                elif predicted_yes and not gold_yes:
                    noul[tag]["fp"] += 1
                elif not predicted_yes and gold_yes:
                    noul[tag]["fn"] += 1
                else:
                    noul[tag]["tn"] += 1
    datasets = {}
    for source, stat in sorted(per.items()):
        items_n = stat["items"]
        dataset = {
            "qtype": stat["qtype"],
            "items": items_n,
            "n_options_seen": dict(sorted(stat["n_options_seen"].items(), key=lambda kv: int(kv[0]))),
            "final_accuracy": _accuracy(stat["final_correct"], items_n),
            "base_accuracy": _accuracy(stat["base_correct"], items_n),
            "accuracy_diff": round(stat["final_correct"] / items_n - stat["base_correct"] / items_n, 8),
            "final_brier": round(stat["final_brier_sum"] / items_n, 8),
            "base_brier": round(stat["base_brier_sum"] / items_n, 8),
            "brier_diff": round((stat["final_brier_sum"] - stat["base_brier_sum"]) / items_n, 8),
            "gold_relative_mean": round(stat["gold_rel_sum"] / items_n, 8),
            "final_relative_mean": round(stat["final_rel_sum"] / items_n, 8),
            "base_relative_mean": round(stat["base_rel_sum"] / items_n, 8),
            "gold_relative_bins": _shares(stat["gold_rel_bins"], items_n),
            "final_relative_bins": _shares(stat["final_rel_bins"], items_n),
            "base_relative_bins": _shares(stat["base_rel_bins"], items_n),
            "gold_index_share": _shares(stat["gold_index"], items_n),
            "final_index_share": _shares(stat["final_index"], items_n),
            "base_index_share": _shares(stat["base_index"], items_n),
            "accuracy_by_gold_index": [
                {
                    "index": int(index),
                    "items": counts[2],
                    "final": _accuracy(counts[0], counts[2]),
                    "base": _accuracy(counts[1], counts[2]),
                }
                for index, counts in sorted(stat["acc_by_gold_index"].items(), key=lambda kv: int(kv[0]))
            ],
        }
        dataset["position_tvd_final_gold"] = round(
            total_variation(dataset["final_index_share"], dataset["gold_index_share"]), 6
        )
        dataset["position_tvd_final_base"] = round(
            total_variation(dataset["final_index_share"], dataset["base_index_share"]), 6
        )
        if len(dataset["n_options_seen"]) == 1:
            n = next(iter(dataset["n_options_seen"]))
            last = str(int(n) - 1)
            dataset["last_position_share"] = {
                "gold": dataset["gold_index_share"].get(last, 0.0),
                "final": dataset["final_index_share"].get(last, 0.0),
                "base": dataset["base_index_share"].get(last, 0.0),
            }
        else:
            dataset["last_position_share"] = {
                "status": "NOT DEFINED — this source mixes option counts; the relative-position bins are reported"
            }
        datasets[source] = dataset
    accuracy_macro_final = sum(d["final_accuracy"] for d in datasets.values()) / len(datasets)
    accuracy_macro_base = sum(d["base_accuracy"] for d in datasets.values()) / len(datasets)
    noul_out = {
        "items": noul["items"],
        "gold_yes_share": round(noul["gold_yes"] / noul["items"], 8) if noul["items"] else None,
    }
    for tag in ("final", "base"):
        block = noul[tag]
        n = noul["items"]
        noul_out[tag] = {
            "mean_p_yes": round(block["p_yes_sum"] / n, 8) if n else None,
            "argmax_yes_rate": round(block["yes_argmax"] / n, 8) if n else None,
            "accuracy": round(block["correct"] / n, 8) if n else None,
            "tp": block["tp"],
            "fp": block["fp"],
            "tn": block["tn"],
            "fn": block["fn"],
        }
    losing = [source for source, d in datasets.items() if d["accuracy_diff"] < 0]
    return {
        "overall": {
            "items": overall["items"],
            "missing_paired": missing,
            "predicted_key_not_item_key": argmax_disagreements,
            "predicted_key_note": (
                "the v6 final writes readout letter codes (A/B) for its noul rows, so its predicted_key is not the item's "
                "yes/no key on exactly those rows; every accuracy and bias figure above is computed from the probability "
                "argmax over the item's own option keys, never from predicted_key"
            ),
            "final_accuracy_micro": _accuracy(overall["final_correct"], overall["items"]),
            "base_accuracy_micro": _accuracy(overall["base_correct"], overall["items"]),
            "final_brier_mean": round(overall["final_brier_sum"] / overall["items"], 8),
            "base_brier_mean": round(overall["base_brier_sum"] / overall["items"], 8),
            "brier_diff_mean": round(
                (overall["final_brier_sum"] - overall["base_brier_sum"]) / overall["items"], 8
            ),
            "final_accuracy_macro_over_datasets": round(accuracy_macro_final, 8),
            "base_accuracy_macro_over_datasets": round(accuracy_macro_base, 8),
            "accuracy_diff_macro_over_datasets": round(accuracy_macro_final - accuracy_macro_base, 8),
            "datasets": len(datasets),
            "losing_sources": losing,
            "losing_source_count": len(losing),
        },
        "per_dataset": datasets,
        "noul": noul_out,
        "summary": {
            "losing_sources": losing,
            "final_minus_base_mean_p_yes": (
                round(noul_out["final"]["mean_p_yes"] - noul_out["base"]["mean_p_yes"], 8)
                if noul_out["final"]["mean_p_yes"] is not None and noul_out["base"]["mean_p_yes"] is not None
                else None
            ),
            "final_yes_argmax_minus_gold_yes": (
                round(noul_out["final"]["argmax_yes_rate"] - noul_out["gold_yes_share"], 8)
                if noul_out["final"]["argmax_yes_rate"] is not None
                else None
            ),
            "max_last_position_excess_final_over_gold": max(
                (
                    d["last_position_share"]["final"] - d["last_position_share"]["gold"]
                    for d in datasets.values()
                    if "final" in d["last_position_share"]
                ),
                default=None,
            ),
        },
    }


# ------------------------------------------------------------------------------------------------------------------
# Readers
# ------------------------------------------------------------------------------------------------------------------


def scan_composition(path: Path, *, with_dates: bool, progress_label: str) -> tuple[dict[str, Any], str, int]:
    """One pass over a JSONL file: sha256, row count and the composition block."""
    comp = new_composition()
    digest = sha256()
    started = time.time()
    with path.open("rb") as handle:
        for count, line in enumerate(handle, start=1):
            digest.update(line)
            absorb(comp, json.loads(line), with_dates=with_dates)
            if count % 250_000 == 0:
                print(
                    f"[{progress_label}] {count:,} rows, {time.time() - started:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
    final = finalise_composition(comp, with_dates=with_dates)
    final["sha256"] = digest.hexdigest()
    return final, digest.hexdigest(), comp["rows"]


def sha256_file(path: Path) -> str:
    """sha256 of a file's bytes (small files only)."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", type=Path, default=MIX)
    parser.add_argument("--anchors", type=Path, default=ANCHORS)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--final-predictions", type=Path, default=FINAL)
    parser.add_argument("--base-predictions", type=Path, default=BASE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)

    mix, mix_sha, mix_rows = scan_composition(args.mix, with_dates=True, progress_label="mix")
    anchors, anchors_sha, anchor_rows = scan_composition(args.anchors, with_dates=True, progress_label="anchors")
    panel, panel_sha, panel_rows = scan_composition(args.panel, with_dates=False, progress_label="panel")
    for name, sha, rows in (
        ("mix", mix_sha, mix_rows),
        ("anchors", anchors_sha, anchor_rows),
        ("panel", panel_sha, panel_rows),
    ):
        expected_sha, expected_rows = EXPECTED[name]
        if sha != expected_sha or rows != expected_rows:
            raise SystemExit(
                f"{name}: expected {expected_rows} rows / sha {expected_sha}, got {rows} rows / sha {sha}"
            )
    composition = {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --frozen python scripts/osler_v7/t1_diagnose.py",
        "inputs": {
            "mix": str(args.mix.relative_to(ROOT)),
            "anchors": str(args.anchors.relative_to(ROOT)),
            "panel": str(args.panel.relative_to(ROOT)),
        },
        "files": {"mix": mix, "anchors": anchors, "panel": panel},
        "comparisons": {
            "mix_vs_panel": composition_compare(mix, panel),
            "anchors_vs_panel": composition_compare(anchors, panel),
        },
    }

    final_predictions = load_predictions(args.final_predictions)
    base_predictions = load_predictions(args.base_predictions)
    items = load_panel_items(args.panel)
    prior = analyse_panel(items, final_predictions, base_predictions)
    prior_shift = {
        "generated_utc": composition["generated_utc"],
        "git_commit": composition["git_commit"],
        "command": composition["command"],
        "inputs": {
            "panel": str(args.panel.relative_to(ROOT)),
            "final_predictions": str(args.final_predictions.relative_to(ROOT)),
            "base_predictions": str(args.base_predictions.relative_to(ROOT)),
            "final_predictions_sha256": sha256_file(args.final_predictions),
            "base_predictions_sha256": sha256_file(args.base_predictions),
            "final_model": "osler_v6 final, chosen_step_063304 (outputs/osler_v6/Z7/baselines/final_panel)",
            "base_model": "zero-shot perplexity-ai/pplx-decider-flash-v1.2 (osler_v4 Y9 baseline pplx_panel)",
        },
        "panel_noul_prior_reference": {
            "mix_noul_yes_share": mix["noul"]["yes_share"],
            "anchors_noul_yes_share": anchors["noul"]["yes_share"],
            "panel_noul_yes_share": prior["noul"]["gold_yes_share"],
        },
        **prior,
    }
    provenance = {
        "generated_utc": composition["generated_utc"],
        "git_commit": composition["git_commit"],
        "command": composition["command"],
        "host": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "wall_seconds": round(time.time() - started, 1),
        "argv": sys.argv,
        "inputs": composition["inputs"],
        "sha256": {"mix": mix_sha, "anchors": anchors_sha, "panel": panel_sha},
    }
    for name, payload in (
        ("composition.json", composition),
        ("prior_shift.json", prior_shift),
        ("run_provenance.json", provenance),
    ):
        with (args.out_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(f"wrote {args.out_dir}/composition.json, prior_shift.json, run_provenance.json in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
