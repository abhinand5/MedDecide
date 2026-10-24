"""Y7 self-audit (R6): re-derives the Stage B keep statistics, the MIMIC partition, the training records and the realised draws of
B1 to B4, in a fresh process (CPU).

Keep statistics, each block against B0 (the A0 predictions in outputs/osler_v4/Y6/A0). For each block: the item counts, the readout
gate's passing templates (meddecide.eval.readout_gate, the library's function), the five keep statistics as points computed here by
plain arithmetic on the prediction rows, and the keep verdict. The lower and upper bounds are the library's paired bootstrap run again
(same resamples, same seed), so they check that the bootstrap is reproducible, not that it is correct. Each value is compared with
the keep JSON, and every mismatch is written out.

MIMIC partition (additional, labelled, not in any verdict): the fast dev v4 items whose template id starts with mimic_, with the same
readout restriction, as a template macro against B0 with its bootstrap bounds.

Training records: the logged steps and evaluations of each block, the checkpoint that was scored (the block result's best step), and
the scored predictions' correct counts beside the in-training evaluation at that step.

Realised draws (unless --skip-replay): the block file's hash, its item count and MIMIC share, and the replay of its batches with
osler_v3's replay functions (scripts/osler_v3/x7_replay.py), compared with the stored replay (per-step sizes and realised items). The
MIMIC share of the realised items is the share of the items the trainer draws.

Writes outputs/osler_v4/Y7/rederive.json (counts and figures only). Usage (CPU; the replay reads the block files, about 25 GB):
    CUDA_VISIBLE_DEVICES= uv run --frozen python scripts/osler_v4/y7_rederive.py
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.eval.readout_gate import gate_by_template
from meddecide.train.blocks import KNOWLEDGE_SOURCES, MR_SOURCES, macro_difference, micro_difference
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import iter_jsonl, read_json, write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
Y6_A0 = REPO / "outputs" / "osler_v4" / "Y6" / "A0"
Y2 = REPO / "outputs" / "osler_v4" / "Y2"
Y7 = REPO / "outputs" / "osler_v4" / "Y7"
TRAIN = REPO / "data" / "train" / "osler_v4"
ARMS = ("B1", "B2", "B3", "B4")
MARGINS = {"own": 0.02, "mr": -0.015, "general": -0.01, "knowledge": -0.02, "cg_v2": -0.01}  # keep rule v2 (D47; ADVISORY Y7)
SHORTCUT_LIMIT = 0.90  # D35 and ADVISORY Y2: a Naive Bayes dev screen at or above this is a shortcut
SCREEN_FILES = ("counts.json", "counts_lab_med.json")
MIMIC = "mimic_"
STEPS = 13_992


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flag(row: dict[str, Any]) -> float:
    return 1.0 if row["correct"] else 0.0


def paired_in_candidate_order(cand: list[dict[str, Any]], base: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
    """Each candidate row with the base row of the same item, in the candidate file's order (the bootstrap depends on it)."""
    index = {str(r["item_id"]): r for r in base}
    ids = [str(r["item_id"]) for r in cand]
    if len(index) != len(base) or len(set(ids)) != len(ids) or set(ids) != set(index):
        raise ValueError("the two prediction sets do not cover the same items, once each")
    return [(r, index[str(r["item_id"])]) for r in cand]


def gate_passing(rows: list[dict[str, Any]]) -> set[str]:
    return {t for t, gate in gate_by_template(rows).items() if gate.passes}


def gate_failing(rows: list[dict[str, Any]]) -> list[str]:
    return sorted(t for t, gate in gate_by_template(rows).items() if not gate.passes)


def macro_point(pairs: list[tuple[dict, dict]]) -> float:
    """The template macro of the paired differences: the mean over templates of each template's mean difference."""
    groups: dict[str, list[float]] = defaultdict(list)
    for c, b in pairs:
        groups[str(c["template_id"])].append(flag(c) - flag(b))
    return sum(sum(v) / len(v) for v in groups.values()) / len(groups)


def micro_point(pairs: list[tuple[dict, dict]]) -> float:
    return sum(flag(c) - flag(b) for c, b in pairs) / len(pairs)


def macro_accuracy(rows: list[dict[str, Any]]) -> float:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        groups[str(r["template_id"])].append(flag(r))
    return sum(sum(v) / len(v) for v in groups.values()) / len(groups)


def shortcut_templates(templates: set[str]) -> set[str]:
    """The templates whose Naive Bayes dev screen (the Y2 counts) is at or above the shortcut limit."""
    out: set[str] = set()
    for name in SCREEN_FILES:
        for template, value in read_json(Y2 / name)["templates"].items():
            if template in templates and float(value["screens"]["nb_macro_accuracy_dev"]) >= SHORTCUT_LIMIT:
                out.add(template)
    return out


def derive_keep(arm: str, mismatches: list[str]) -> dict[str, Any]:
    keep = read_json(Y7 / f"keep_{arm}.json")
    new = set(keep["new_templates"])
    dev_c = list(iter_jsonl(Y7 / arm / "fastdev_v4_predictions.jsonl"))
    dev_b = list(iter_jsonl(Y6_A0 / "fastdev_v4_predictions.jsonl"))
    cg_c = list(iter_jsonl(Y7 / arm / "cg_v2_predictions.jsonl"))
    cg_b = list(iter_jsonl(Y6_A0 / "cg_v2_predictions.jsonl"))
    dev_pass = gate_passing(dev_c) & gate_passing(dev_b)
    cg_pass = gate_passing(cg_c) & gate_passing(cg_b)
    dev = [(c, b) for c, b in paired_in_candidate_order(dev_c, dev_b) if str(c["template_id"]) in dev_pass]
    cg = [(c, b) for c, b in paired_in_candidate_order(cg_c, cg_b) if str(c["template_id"]) in cg_pass]
    shortcut = shortcut_templates(new)
    if shortcut != set(keep["shortcut_templates_excluded_from_own"]):
        mismatches.append(f"{arm}: shortcut templates {sorted(shortcut)} against the keep JSON")
    own = [p for p in dev if str(p[0]["template_id"]) in new - shortcut]
    own_with = [p for p in dev if str(p[0]["template_id"]) in new]
    rest = [p for p in dev if str(p[0]["template_id"]) not in new]
    mr = [p for p in rest if str(p[0]["source"]) in MR_SOURCES]
    general = [p for p in rest if str(p[0]["source"]) not in MR_SOURCES]
    knowledge = [p for p in general if str(p[0]["source"]) in KNOWLEDGE_SOURCES]
    mimic = [p for p in dev if str(p[0]["template_id"]).startswith(MIMIC)]

    points = {"own": macro_point(own), "mr": macro_point(mr), "general": macro_point(general),
              "knowledge": micro_point(knowledge), "cg_v2": micro_point(cg)}
    lib = {"own": macro_difference(own), "mr": macro_difference(mr), "general": macro_difference(general),
           "knowledge": micro_difference(knowledge), "cg_v2": micro_difference(cg)}
    counts = {"own": len(own), "own_with_shortcut_templates": len(own_with), "mr": len(mr), "general": len(general),
              "knowledge": len(knowledge), "cg_v2": len(cg)}
    kept = all(lib[k].lo > MARGINS[k] for k in MARGINS)
    stored = keep["statistics"]
    for k in MARGINS:
        if round(points[k], 6) != stored[k]["point"]:
            mismatches.append(f"{arm} {k}: point {round(points[k], 6)} against the keep JSON {stored[k]['point']}")
        if round(lib[k].lo, 6) != stored[k]["lower"] or round(lib[k].hi, 6) != stored[k]["upper"]:
            mismatches.append(f"{arm} {k}: bounds {round(lib[k].lo, 6)}, {round(lib[k].hi, 6)} against the keep JSON")
        if counts[k] != keep["items"][k]:
            mismatches.append(f"{arm} {k}: {counts[k]} items against {keep['items'][k]}")
    if len(own_with) != keep["items"]["own_with_shortcut_templates"]:
        mismatches.append(f"{arm}: own with shortcut templates has {len(own_with)} items")
    additional = macro_difference(own_with)
    if round(additional.lo, 6) != keep["additional_not_in_verdict"]["own_with_shortcut_templates"]["lower"]:
        mismatches.append(f"{arm}: the additional own bound does not reproduce")
    if kept != keep["verdict"]["kept"]:
        mismatches.append(f"{arm}: verdict {kept} against the keep JSON {keep['verdict']['kept']}")
    for side, rows in (("candidate_fastdev_v4", dev_c), ("base_fastdev_v4", dev_b),
                       ("candidate_cg_v2", cg_c), ("base_cg_v2", cg_b)):
        if gate_failing(rows) != keep["failing_cells"][side]:
            mismatches.append(f"{arm} {side}: failing cells {gate_failing(rows)} against the keep JSON")
    if sorted(new - dev_pass) != keep["excluded_new_templates"]:
        mismatches.append(f"{arm}: excluded new templates do not reproduce")

    mimic_lib = macro_difference(mimic)
    mimic_templates = sorted({str(c["template_id"]) for c, _ in mimic})
    mimic_all = [str(r["template_id"]) for r in dev_c if str(r["template_id"]).startswith(MIMIC)]
    return {
        "keep": {
            "items": counts,
            "points": {k: round(points[k], 6) for k in MARGINS},
            "lower": {k: round(lib[k].lo, 6) for k in MARGINS},
            "upper": {k: round(lib[k].hi, 6) for k in MARGINS},
            "shortcut_templates": sorted(shortcut),
            "own_with_shortcut_templates": {"point": round(macro_point(own_with), 6),
                                            "lower": round(additional.lo, 6), "upper": round(additional.hi, 6)},
            "kept": kept,
            "scored_in_keep_json": keep["verdict"]["kept"],
            "excluded_new_templates": sorted(new - dev_pass),
            "failing_cells": {"candidate_fastdev_v4": gate_failing(dev_c), "base_fastdev_v4": gate_failing(dev_b),
                              "candidate_cg_v2": gate_failing(cg_c), "base_cg_v2": gate_failing(cg_b)},
        },
        "mimic": {
            "items_all": len(mimic_all),
            "items_after_readout_restriction": len(mimic),
            "templates": len(mimic_templates),
            "excluded_templates": sorted(set(mimic_all) - dev_pass),
            "template_macro_difference": {"point": round(macro_point(mimic), 6), "lower": round(mimic_lib.lo, 6),
                                          "upper": round(mimic_lib.hi, 6)},
            "template_macro_accuracy": {"candidate": round(macro_accuracy([c for c, _ in mimic]), 6),
                                        "base": round(macro_accuracy([b for _, b in mimic]), 6)},
            "scored_correct": {"fastdev": int(sum(flag(r) for r in dev_c)), "fastdev_n": len(dev_c),
                               "cg_v2": int(sum(flag(r) for r in cg_c)), "cg_v2_n": len(cg_c)},
        },
    }


def training_record(arm: str, mismatches: list[str]) -> dict[str, Any]:
    result = read_json(Y7 / arm / "block_result.json")
    rows = list(iter_jsonl(Y7 / arm / "logs" / "train.jsonl"))
    steps = [r for r in rows if r.get("event") == "step"]
    evals = {r["step"]: r["metrics"] for r in rows if r.get("event") == "eval"}
    ends = [r for r in rows if r.get("event") == "end"]
    selected = int(result["best_step"])
    scored = read_json(Y7 / arm / "score_result.json")
    if scored["checkpoint"] != f"outputs/osler_v4/Y7/{arm}/checkpoints/best":
        mismatches.append(f"{arm}: the scored checkpoint is {scored['checkpoint']}")
    if [e["step"] for e in result["evaluations"]] != sorted(evals):
        mismatches.append(f"{arm}: the evaluation steps in block_result differ from the log")
    if ends[0]["steps"] != STEPS or result["steps"] != STEPS:
        mismatches.append(f"{arm}: the step count is not {STEPS}")
    nonfinite = sum(1 for r in steps if not (math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"])))
    return {
        "steps": ends[0]["steps"],
        "logged_steps": len(steps),
        "last_logged_step": steps[-1]["step"],
        "nonfinite_logged_steps": nonfinite,
        "zero_loss_logged_steps": [(r["step"], r["n_items"]) for r in steps if r["loss"] == 0.0],
        "evaluation_steps": sorted(evals),
        "selected_step": selected,
        "in_training_at_selected": {"n_correct": evals[selected]["n_correct"], "n": evals[selected]["n"],
                                    "accuracy": round(evals[selected]["accuracy"], 6),
                                    "template_macro_accuracy": round(evals[selected]["template_macro_accuracy"], 6)},
        "in_training_template_macro_by_step": {s: round(evals[s]["template_macro_accuracy"], 6) for s in sorted(evals)},
        "validation_mismatches": result["validation"]["logged_steps"]["mismatches"],
    }


def realised(arm: str, mismatches: list[str]) -> dict[str, Any]:
    x7 = load_module("x7_replay", REPO / "scripts" / "osler_v3" / "x7_replay.py")
    rp = x7._w6()
    config = x7.load_config(x7.CONFIG)
    path = TRAIN / f"block_{arm}.jsonl"
    manifest = read_json(Y7 / f"block_{arm}.json")
    stored = read_json(Y7 / "replays" / f"{arm}_pplx.json")
    digest = file_sha256(path)
    items = rp.read_block(path)
    mix = Counter(str(it.template_id) for it in items)
    mix_mimic = sum(v for t, v in mix.items() if t.startswith(MIMIC))
    if digest != stored["train_file_sha256"] or digest != manifest["file_sha256"]:
        mismatches.append(f"{arm}: the block file hash differs from the replay or the manifest")
    if len(items) != manifest["rows"] or len(items) != stored["train_items"]:
        mismatches.append(f"{arm}: the block row count differs from the manifest or the replay")
    if round(mix_mimic / len(items), 4) != manifest["mimic_share"]:
        mismatches.append(f"{arm}: the MIMIC share {round(mix_mimic / len(items), 4)} against the manifest {manifest['mimic_share']}")
    encoder = rp.encoder_for("pplx", config)
    order = x7.shuffled_order(len(items), seed=config.seed, epoch=0)
    batches = x7.iter_batches(encoder, items, batch_size=config.batch_size, max_batch_tokens=config.max_batch_tokens,
                              max_prompt_tokens=config.max_prompt_tokens, order=order, augment_options=config.shuffle_options,
                              epoch=0, seed=config.seed)
    sizes: list[int] = []
    drawn: Counter[str] = Counter()
    for _, batch in itertools.islice(batches, STEPS):
        sizes.append(len(batch))
        drawn.update(str(e.template_id) for e in batch)
    batches.close()
    total = sum(sizes)
    if sizes != stored["sizes"]:
        mismatches.append(f"{arm}: the replayed per-step sizes differ from the stored replay")
    if total != stored["realised_items"]:
        mismatches.append(f"{arm}: realised items {total} against the stored {stored['realised_items']}")
    drawn_mimic = sum(v for t, v in drawn.items() if t.startswith(MIMIC))
    top_drawn = drawn.most_common(1)[0]
    return {
        "file_sha256": digest,
        "mix_rows": len(items),
        "mix_mimic_rows": mix_mimic,
        "mix_mimic_share": round(mix_mimic / len(items), 4),
        "mix_largest_template": manifest["largest_template"],
        "realised_items": total,
        "realised_items_per_step_mean": round(total / len(sizes), 4),
        "realised_mimic_items": drawn_mimic,
        "realised_mimic_share": round(drawn_mimic / total, 4),
        "realised_templates": len(drawn),
        "realised_largest_template": top_drawn[0],
        "realised_largest_template_share": round(top_drawn[1] / total, 4),
        "sizes_match_stored_replay": sizes == stored["sizes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-replay", action="store_true", help="skip the block files and the realised draws")
    args = parser.parse_args()
    mismatches: list[str] = []
    result: dict[str, Any] = {"generated_utc": utcnow(), "margins": MARGINS, "keep": {}, "mimic": {}, "training": {},
                              "realised": {}}
    for arm in ARMS:
        derived = derive_keep(arm, mismatches)
        result["keep"][arm] = derived["keep"]
        result["mimic"][arm] = derived["mimic"]
        result["training"][arm] = training_record(arm, mismatches)
        if not args.skip_replay:
            result["realised"][arm] = realised(arm, mismatches)
        print(f"{arm}: kept {derived['keep']['kept']} (keep JSON {derived['keep']['scored_in_keep_json']}); "
              f"mismatches so far {len(mismatches)}", flush=True)
    result["mismatches"] = mismatches
    write_json(Y7 / "rederive.json", result)
    print(f"mismatches: {len(mismatches)}", flush=True)
    for line in mismatches:
        print(f"  {line}", flush=True)


if __name__ == "__main__":
    main()
