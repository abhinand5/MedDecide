"""osler_v5 arm driver (ADVISORY Z1-Z3): train one recipe arm or the final on osler_v4's exact mix, and at every evaluation score the
dev sets only: fast dev v4 (MR, MIMIC and the rest), CG v2 (micro), knowledge dev v2, and the DDXPlus items of CG v2 under the original
order and two seeded option permutations (D52b). The per-evaluation checkpoints are kept, for the Z3 selection by S among the
evaluations that pass positional health. No test split, panel, MIMIC audit suite or held-out file is read here.

The mix is composed in memory with osler_v4's Y8 composer (composition 9d25e30f…, checked before any step). A config with
option_augment_probability > 0 rebuilds the closed-list rows of the TRAINING rows before they become items (D52a); the report of
the rebuild is written next to the arm. The run refuses to start when the arm has an arm_result.json (refuse-twice guard) and
resumes from the newest full checkpoint with --resume (the watchdog rule of ADVISORY §3.7).

Usage (GPU; one job; detached):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v5/run_arm.py \
        --config configs/osler_v5/arm_R0.yaml --out outputs/osler_v5/Z2/R0 --steps 13992 --evaluations 2 \
        > outputs/osler_v5/Z2/R0/logs/run.log 2>&1 < /dev/null &
    (after an interruption: the same command with --resume, appended to the same log)
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from meddecide.eval.composite import composite, permutation_for, positional_health
from meddecide.train.corrected_mix import (
    CORRECTED_COMPOSITION_SHA256,
    drop_template_rows,
    template_counts,
)
from meddecide.train.option_augment import augment_rows, build_pools

from meddecide.bench.schema import Item
from meddecide.model.meddecide_model import reorder_options
from meddecide.model.pplx import PplxDecisionModel
from meddecide.train.config import StudentConfig, load_config
from meddecide.train.final_mix import compose_final_mix, composition_sha256
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.osler_arm import dev_prediction_rows
from meddecide.train.resume import latest_full_checkpoint, truncate_log_after
from meddecide.train.trainer import Trainer, evaluation_grid
from meddecide.utils.hashing import stable_hash
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"
CG = REPO / "data" / "bench" / "osler_v4" / "cg_v2.jsonl"
KNOW = REPO / "data" / "bench" / "osler_v2" / "knowledge_dev_v2.jsonl"
GROUNDED_PATH: Path | None = None  # set from --grounded in main()
REBALANCE = True  # synth_v1 correction (coordinator, 2026-10-29): replay >= 20 %, MIMIC family <= 35 % of the final mix
REPLAY_ADD = 4_241  # 3,241 (the minimal top-up for the grounded-inclusive mix) plus a margin of 1,000
FAMILY_DROP = 11_500  # base MIMIC rows removed, water-filling from the largest MIMIC template; the grounded block is not dropped
FAMILY_TEMPLATES = frozenset({  # the 15 templates of the family blocks B1-B3 (checked against the corrected mix in code)
    "mimic_apr_mortality_v1", "mimic_apr_severity_v1", "mimic_icd_specificity_v1", "mimic_med_given_v1", "mimic_msdrg_v1",
    "mimic_pcs_support_v1", "mimic_principal_dx_v1", "mimic_rad_exam_v1", "mimic_ed_acuity_v1", "mimic_ed_disposition_v1",
    "mimic_drg_cc_level_v1", "mimic_lab_validation_v1", "mimic_icd_support_v1", "mimic_doc_support_v1", "mednli_v1"})
MIMIC_SOURCES = frozenset({"mimic_iv", "mimic_iv_note"})
EXPECTED_FAMILY_CORRECTED = 509_807  # osler_v5 Z0 corrected_mix.json family_kept


def rebalance(rows: list[dict[str, Any]], composed_ids: set[str], broad: Path, *, drop_n: int, add_n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """synth_v1: drop drop_n base MIMIC rows (water-filling from the largest MIMIC template, seeded by stable hash) and add add_n
    replay rows from the broad pool rows not already composed (seeded by stable hash). Never touches a held-out or test file: the
    family blocks are train-split by construction and only rows are removed; the pool is the broad replay file."""
    family_now = sum(1 for r in rows if str(r["template_id"]) in FAMILY_TEMPLATES)
    if family_now != EXPECTED_FAMILY_CORRECTED:
        raise SystemExit(f"the family rows of the corrected mix are {family_now}, not {EXPECTED_FAMILY_CORRECTED}")
    before = {"rows": len(rows), "replay_rows": sum(is_replay_row(r) for r in rows), "family_rows": family_now}
    mimic = [r for r in rows if str(r["source"]) in MIMIC_SOURCES]
    counts = Counter(str(r["template_id"]) for r in mimic)
    def removed_at(level: int) -> int:
        return sum(max(0, c - level) for c in counts.values())
    lo, hi = 0, max(counts.values())
    while lo < hi:  # the smallest level L with removed_at(L) <= drop_n: every template above L is cut to L
        mid = (lo + hi) // 2
        if removed_at(mid) <= drop_n:
            hi = mid
        else:
            lo = mid + 1
    level = lo
    quota_drop = {t: max(0, c - level) for t, c in counts.items()}
    short = drop_n - sum(quota_drop.values())
    if short < 0 or short > sum(1 for c in counts.values() if c >= level and level > 0):
        raise SystemExit(f"water-filling cannot place {drop_n} drops (short {short}, level {level})")
    for t in sorted(t for t, c in counts.items() if c >= level and level > 0)[:short]:  # one more row from the tied templates
        quota_drop[t] += 1
    drop_ids: set[str] = set()
    by_template: dict[str, list[str]] = defaultdict(list)
    for r in mimic:
        by_template[str(r["template_id"])].append(str(r["item_id"]))
    for t, ids in by_template.items():
        ids.sort(key=lambda i: stable_hash({"salt": "synth_v1_family_drop", "item": i}, length=16))
        drop_ids.update(ids[: quota_drop.get(t, 0)])
    if len(drop_ids) != drop_n:
        raise SystemExit(f"dropped {len(drop_ids)} rows, expected {drop_n}")
    kept = [r for r in rows if str(r["item_id"]) not in drop_ids]
    pool = [r for r in iter_jsonl(broad) if str(r["item_id"]) not in composed_ids]
    if add_n > len(pool):
        raise SystemExit(f"the replay addition needs {add_n} rows, the pool holds {len(pool)}")
    add = sorted(pool, key=lambda r: stable_hash({"salt": "synth_v1_replay_extra", "item": str(r["item_id"])}, length=16))[:add_n]
    add_ids = {str(r["item_id"]) for r in add}
    if add_ids & {str(r["item_id"]) for r in kept}:
        raise SystemExit("an added replay id repeats an id in the mix")
    out = kept + add
    after = {"rows": len(out), "replay_rows": sum(is_replay_row(r) for r in out),
             "family_rows": sum(1 for r in out if str(r["template_id"]) in FAMILY_TEMPLATES)}
    report = {"drop_mimic_rows": len(drop_ids), "drop_by_template": {t: quota_drop.get(t, 0) for t in sorted(quota_drop) if quota_drop.get(t, 0)},
              "water_level": level, "add_replay_rows": len(add), "replay_pool_rows": len(pool),
              "add_by_template": dict(sorted(Counter(str(r["template_id"]) for r in add).items())),
              "before": before, "after": after}
    return out, report


def is_replay_row(row: Mapping[str, Any]) -> bool:
    return bool(str(row["template_id"]).startswith("broad_") or str(row["source"]) in GENERAL_REPLAY_SOURCES)


GENERAL_REPLAY_SOURCES = frozenset({"snli", "winogrande", "hellaswag", "commonsenseqa", "boolq", "qasc", "arc_easy", "arc_challenge"})


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)
EXPECTED_COMPOSITION = "9d25e30f53d7823d051b9c71334c248660318c86ce904c26f855b9a9dbc618b8"
FULL_CHECKPOINT_EVERY = 2_500  # ADVISORY Z3: full checkpoints every 2,500 steps
FULL_CHECKPOINTS_KEPT = 2
DDX_PREFIX = "cg_v2_ddxplus"  # the CG v2 items that offer more than 26 options (49 options)
POSITIONAL_SEEDS = (1, 2)
TWO_LETTER_FROM = 26


def _module(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def compose_rows(config: StudentConfig, out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """osler_v4's final mix in memory (the Y8 composer), then the closed-list augmentation of its training rows if configured."""
    y8_mix = _module("y8_final_mix", "scripts/osler_v4/y8_final_mix.py")
    decisions = {name: y8_mix.entering(name) for name in ("B1", "B2", "B3", "B4")}
    family = {name: y8_mix.add_files(name) for name in y8_mix.FAMILY if decisions[name][0]}
    other = y8_mix.add_files("B4") if decisions["B4"][0] else []
    rows, composition = compose_final_mix(y8_mix.BASE, family, other, y8_mix.BROAD)
    if composition["composition_sha256"] != EXPECTED_COMPOSITION:
        raise SystemExit(f"the composed mix hash is {composition['composition_sha256']}, not osler_v4's {EXPECTED_COMPOSITION}")
    info: dict[str, Any] = {"composition_sha256": composition["composition_sha256"], "rows": len(rows)}
    composed_ids = {str(row["item_id"]) for row in rows}
    if config.drop_mix_templates:  # D56a: the corrected mix; its hash is the one recorded by build_corrected_mix.py
        rows, drop = drop_template_rows(rows, config.drop_mix_templates)
        corrected = composition_sha256(str(row["item_id"]) for row in rows)
        info["corrected_hash"] = corrected
        if corrected != CORRECTED_COMPOSITION_SHA256:
            raise SystemExit(f"the corrected mix hash is {corrected}, not the recorded {CORRECTED_COMPOSITION_SHA256}")
        info["composition_sha256"] = corrected
        info["rows"] = len(rows)
        info["drop"] = drop.to_dict()
    if REBALANCE:
        rows, rb = rebalance(list(rows), composed_ids, y8_mix.BROAD, drop_n=FAMILY_DROP, add_n=REPLAY_ADD)
        info["rebalance"] = rb
    if config.option_augment_probability > 0:
        pools = build_pools(rows, config.option_augment_templates)
        rows, report = augment_rows(rows, pools, seed=config.seed, probability=config.option_augment_probability)
        info["augmentation"] = {"probability": config.option_augment_probability, "templates": list(config.option_augment_templates),
                                "pool_sizes": {t: len(p.entries) for t, p in pools.items()},
                                "label_set_sizes": {t: p.label_set_size for t, p in pools.items()}, **report.to_dict()}
        write_json(out / "augmentation.json", info["augmentation"])
    if GROUNDED_PATH is not None:  # synth_v1: the MIMIC-grounded block, appended after the augmentation (caps checked in code)
        pre_rows = list(rows)
        before_counts = template_counts(pre_rows)
        grounded = [json.loads(line) for line in GROUNDED_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = pre_rows + grounded
        total = len(rows)
        after_counts = template_counts(rows)

        def shares(rs: list[dict[str, Any]]) -> dict[str, float]:
            n = len(rs)
            return {"rows": n,
                    "replay_share": sum(is_replay_row(r) for r in rs) / n,
                    "family_share": sum(1 for r in rs if str(r["template_id"]) in FAMILY_TEMPLATES) / n,
                    "grounded_share": sum(1 for r in rs if str(r.get("source")) == "synth_mimic_v1") / n,
                    "largest_template_share": max(template_counts(rs).values()) / n}

        before_shares = shares(pre_rows)
        after_shares = shares(rows)
        checks = {"grounded_at_most_20pct": after_shares["grounded_share"] <= 0.20,
                  "largest_template_at_most_8pct": after_shares["largest_template_share"] <= 0.08,
                  "replay_at_least_20pct": after_shares["replay_share"] >= 0.20,
                  "family_at_most_35pct": after_shares["family_share"] <= 0.35}
        if not all(checks.values()):
            raise SystemExit(f"caps failed: {checks} {after_shares}")
        grounded_templates = Counter(str(r["template_id"]) for r in grounded)
        if sum(grounded_templates.values()) != len(grounded) or any(str(r.get("source")) != "synth_mimic_v1" for r in grounded):
            raise SystemExit("grounded rows must all carry source synth_mimic_v1")
        mixed = composition_sha256(str(row["item_id"]) for row in rows)
        info["composition_sha256"] = mixed
        info["rows"] = total
        record = {"path": GROUNDED_PATH.name, "rows_grounded": len(grounded), "rows_before_grounded_append": len(pre_rows),
                  "rows_after": total, "composition_sha256_corrected": info.get("corrected_hash"),
                  "composition_sha256_final": mixed,
                  "rebalance": info.get("rebalance"),
                  "shares_before_grounded_append": before_shares, "shares_after": after_shares, "checks": checks,
                  "per_template_before": dict(sorted(before_counts.items())), "per_template_after": dict(sorted(after_counts.items())),
                  "grounded_per_template": dict(sorted(grounded_templates.items()))}
        info["grounded"] = record
        write_json(out / "composition.json", record)
        print(json.dumps({"composition": {k: v for k, v in record.items() if not k.startswith("per_template")}}, sort_keys=True, default=str), flush=True)
    return rows, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--evaluations", type=int, default=2, help="evaluations spread evenly over the steps, the last at the budget")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--grounded", default=None, help="synth_v1: the grounded training rows (JSONL) appended to the mix")
    parser.add_argument("--smoke", action="store_true",
                        help="a code-path check only: every dev set is cut to a few hundred items (the positional check keeps both groups)")
    parser.add_argument("--dry-run", action="store_true",
                        help="compose the mix, apply the augmentation and convert every item on the CPU, then stop before the model")
    args = parser.parse_args()
    global GROUNDED_PATH
    GROUNDED_PATH = Path(args.grounded) if args.grounded else None
    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    if (out / "arm_result.json").exists() and not args.dry_run:
        raise SystemExit(f"{out}: arm_result.json exists; the arm is not run twice (refuse-twice guard)")
    for directory in (out, out / "logs", out / "evals", out / "checkpoints"):
        directory.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    config = load_config(REPO / args.config)
    grid = evaluation_grid(args.steps, args.evaluations)
    config = dataclasses.replace(config, output_dir=str(out.relative_to(REPO)), eval_steps=grid, eval_every=args.steps // args.evaluations,
                                 backbone="pplx", kl_to_base=config.kl_to_base, distill_weight=0.0)

    rows, mix_info = compose_rows(config, out)
    train_items = [mix_row_to_item(row) for row in rows]
    del rows
    fast_rows, cg_rows, know_rows = read_jsonl(DEV), read_jsonl(CG), read_jsonl(KNOW)
    if args.smoke:
        fast_rows, cg_rows, know_rows = fast_rows[::50], cg_rows[::20], know_rows[::20]
    fast_items = [mix_row_to_item(r) for r in fast_rows]
    cg_items = [mix_row_to_item(r) for r in cg_rows]
    know_items = [mix_row_to_item(r) for r in know_rows]
    ddx_index = [i for i, item in enumerate(cg_items) if str(item.template_id).startswith(DDX_PREFIX)]
    if not ddx_index or any(cg_items[i].n_options <= TWO_LETTER_FROM for i in ddx_index):
        raise SystemExit("the positional check needs the DDXPlus items of CG v2 with more than 26 options")
    ddx_items = [cg_items[i] for i in ddx_index]
    write_json(out / "run_provenance.json", {
        "started_utc": started, "git_commit": git_commit(REPO), "command": " ".join(sys.argv), "config": config.to_dict(),
        "steps": args.steps, "evaluation_steps": list(grid), "mix": mix_info, "dev_items": len(fast_items), "cg_items": len(cg_items),
        "knowledge_items": len(know_items), "positional_items": len(ddx_items),
    })

    if args.dry_run:
        counts = {"train_items": len(train_items), "dev_items": len(fast_items), "cg_items": len(cg_items),
                  "knowledge_items": len(know_items), "positional_items": len(ddx_items), "grid": list(grid)}
        write_json(out / "dry_run.json", {"utc": utcnow(), "config": config.to_dict(), "mix": mix_info, "counts": counts})
        print(json.dumps({"dry_run": True, **counts, "augmentation": mix_info.get("augmentation", {}).get("seen")}, sort_keys=True), flush=True)
        return

    resume_from = None
    if args.resume:
        resume_from = latest_full_checkpoint(out / "full")
        if resume_from is None:
            raise SystemExit("--resume, but no complete full checkpoint exists")
        step = int(resume_from.name.split("_")[1])
        truncate_log_after(out / "logs" / "train.jsonl", step)
        with (out / "logs" / "resumes.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"resumed_from": str(resume_from.relative_to(REPO)), "step": step, "utc": utcnow()}) + "\n")

    torch.manual_seed(config.seed)  # the initialisation, as arm_run.run_arm seeds it (matched arms start from the same weights)
    model = PplxDecisionModel(lora=config.lora, dtype=config.dtype, device=config.device,
                              max_prompt_tokens=config.max_prompt_tokens)
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    trainer = Trainer(model, config, log_path=out / "logs" / "train.jsonl")
    evaluations: dict[int, dict[str, Any]] = {}

    def evaluate_dev(step: int, _metrics: dict[str, Any]) -> None:
        """Every dev set at one evaluation: rows saved, composite and positional health recorded; the model's checkpoint kept."""
        started_eval = time.time()
        # the trainer has just scored the full fast dev set for its tripwire (the same model, items and settings): reuse that result
        fast_scored = cached.pop("fast") if "fast" in cached else trainer.score(fast_items)
        cg_scored = trainer.score(cg_items)
        know_scored = trainer.score(know_items)
        fast_pred = dev_prediction_rows(fast_scored)
        cg_pred = dev_prediction_rows(cg_scored)
        know_pred = dev_prediction_rows(know_scored)
        perm_health: dict[str, Any] = {}
        ddx_pred_by_seed: dict[str, list[dict[str, Any]]] = {}
        for seed in POSITIONAL_SEEDS:
            permuted = [reorder_options(item, permutation_for(item.n_options, seed=seed, index=i))
                        for item, i in zip(ddx_items, ddx_index, strict=True)]
            pred = dev_prediction_rows(trainer.score(permuted))
            ddx_pred_by_seed[str(seed)] = pred
            perm_health[str(seed)] = positional_health(pred).to_dict()
        parts = composite(fast_pred, cg_pred)
        original = positional_health([r for r in cg_pred if r["template_id"].startswith(DDX_PREFIX)]).to_dict()
        eval_dir = out / "evals" / f"step_{step:06d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(eval_dir / "fast_dev.jsonl", fast_pred)
        write_jsonl(eval_dir / "cg_v2.jsonl", cg_pred)
        write_jsonl(eval_dir / "knowledge_dev.jsonl", know_pred)
        for seed, pred in ddx_pred_by_seed.items():
            write_jsonl(eval_dir / f"positional_seed{seed}.jsonl", pred)
        model.save(out / "checkpoints" / f"step_{step:06d}")
        knowledge = sum(bool(r["correct"]) for r in know_pred) / len(know_pred)
        record = {"step": step, "composite": parts.to_dict(), "knowledge_dev_accuracy": knowledge,
                  "positional_original": original, "positional_permuted": perm_health,
                  "positional_passes_all_orders": bool(original["passes"] and all(h["passes"] for h in perm_health.values())),
                  "seconds": round(time.time() - started_eval, 1)}
        evaluations[step] = record
        write_json(out / "evaluations.json", {str(k): v for k, v in sorted(evaluations.items())})
        print(f"eval step {step}: S {parts.s:.4f} (MR {parts.mr:.4f}, MIMIC {parts.mimic:.4f}, CG micro {parts.cg_micro:.4f}); "
              f"knowledge {knowledge:.4f}; positional original {original['passes']} seeds "
              f"{[h['passes'] for h in perm_health.values()]}", flush=True)

    trainer.eval_hook = evaluate_dev
    cached: dict[str, Any] = {}
    score_direct = trainer.score

    def score_and_keep_fast_dev(items: list[Item], *, batch_size: int | None = None) -> Any:
        scored = score_direct(items, batch_size=batch_size)
        if items is fast_items:
            cached["fast"] = scored
        return scored

    trainer.score = score_and_keep_fast_dev  # type: ignore[method-assign]
    # the trainer evaluates the full fast dev v4 at each evaluation step, as in osler_v4 Y8: the dev-macro tripwire reads that
    # measurement; the hook then scores the dev sets for the composite, the knowledge set and the positional check
    result = trainer.train(train_items, steps=args.steps, eval_items=fast_items, schedule_steps=args.steps,
                           full_checkpoint_dir=out / "full", full_checkpoint_every=FULL_CHECKPOINT_EVERY,
                           resume_from=resume_from, full_checkpoint_keep_last=FULL_CHECKPOINTS_KEPT)
    model.eval_mode()
    fits = Trainer(model, config).fit_calibration(fast_items)
    final_step = result.steps
    eval_steps = [int(s) for s in evaluations]
    final_scores = {
        "fast_dev": dev_prediction_rows(Trainer(model, config).score(fast_items)),
        "cg_v2": dev_prediction_rows(Trainer(model, config).score(cg_items)),
    }
    write_jsonl(out / "final_fast_dev.jsonl", final_scores["fast_dev"])
    write_jsonl(out / "final_cg_v2.jsonl", final_scores["cg_v2"])
    model.save(out / "checkpoints" / f"final_step_{final_step:06d}")
    write_json(out / "arm_result.json", {
        "steps": result.steps, "stopped_early": result.stopped_early, "evaluation_steps": eval_steps,
        "evaluations": {str(k): v for k, v in sorted(evaluations.items())},
        "temperature_fits": {k: v.to_dict() if hasattr(v, "to_dict") else str(v) for k, v in fits.items()},
        "final_step": final_step, "finished_utc": utcnow(), "config": config.to_dict(), "git_commit": git_commit(REPO),
    })
    print(f"done: {result.steps} steps; evaluations at {eval_steps}", flush=True)


if __name__ == "__main__":
    main()
