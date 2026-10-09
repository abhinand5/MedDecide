"""O4: build training mix v2 for osler_v0 (ADVISORY O4; D17, D20). Inputs are read-only.

Inputs: the student_v1 mix (data/train/student_v1, copied verbatim), the O3 seen generators (data/gen/osler_v0),
permissive replay and catalog sources from the Hugging Face cache (Parquet revisions), and the pre-window PubMed pool
(data/train/student_v1/sources/pubmed_prewindow.jsonl.gz) for PubMed-derived catalog records.

Steps: (1) pre-window dates for PubMed-derived catalog records; (2) convert replay and catalog records, counting every
drop by reason; (3) screen each new template on its dev split (gold-in-state, bag-of-words macro below 0.90), dropping
any template that fails; (4) count the student_v1 mix and measure how often each choice or score gold is visible in its
state; (5) write the train file (student_v1 rows verbatim, the new rows, gold-preserving augmentation within the 8 %
cap) and the dev file; (6) leakage counts against every protected set; (7) manifest and build record.

Outputs (gitignored): data/train/osler_v0/{train,dev}.jsonl and manifest.json; outputs/osler_v0/O4/build.json.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o4_build_mix.py
"""

from __future__ import annotations

import glob
import gzip
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from meddecide.gen.screen import gold_in_state, screen_generator
from meddecide.mix.augment import (
    TRANSFORMS,
    pad_state,
    plant_instruction,
    replace_gold_with_none,
    reverse_options,
    select,
)
from meddecide.mix.converters import (
    chemprot_rows,
    commonsense_row,
    evidence_row,
    generator_row,
    pubhealth_row,
    qasc_row,
)
from meddecide.mix.leakage import row_text_key
from meddecide.utils.hashing import file_sha256, stable_hash
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
HF_HUB = Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "hub"
STUDENT_DIR = REPO / "data/train/student_v1"
STUDENT_TRAIN = STUDENT_DIR / "train.jsonl"
STUDENT_DEV = STUDENT_DIR / "dev.jsonl"
POOL = STUDENT_DIR / "sources/pubmed_prewindow.jsonl.gz"
GEN_DIR = REPO / "data/gen/osler_v0"
V02_DIR = REPO / "data/bench/v0.2"
EXT_DIR = REPO / "data/bench/v0.3_ext"
OUT_DIR = REPO / "data/train/osler_v0"
OUT_JSON = REPO / "outputs/osler_v0/O4/build.json"

WINDOW = "2026-03-01"
CAP = 0.08
MIX_BUDGET = 200_000
AUG_MAX_SHARE = 0.15
AUG_FRACTION = {"reverse_options": 0.03, "pad_state": 0.03, "plant_instruction": 0.03, "replace_gold_with_none": 0.02}
CHEMPROT_MAX_PER_DOC = 3
SCREEN_THRESHOLD = 0.90
PAD_POOL_PER_SOURCE = 200
HELD_OUT_TEMPLATES = ("ct_arm_role_noul_v1", "ct_phase_choice_v1", "fda_boxed_warning_noul_v1", "pubmed_humans_noul_v1")
NEW_TEMPLATES = ("csqa_choice_v1", "qasc_science_choice_v1", "pubhealth_claim_true_noul_v1",
                 "chemprot_relation_choice_v1", "evidence_direction_choice_v1")
PMID_LINE = re.compile(r'\{"pmid": "(\d+)", "record_date": "([^"]*)"')
SEX_WORDS = re.compile(r"\b(sex|sexes|female|males?|women|men|gender)\b", re.I)
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def load_parquet(pattern: str) -> list[dict[str, Any]]:
    files = sorted(glob.glob(str(HF_HUB / pattern)))
    if not files:
        raise FileNotFoundError(f"no cached parquet files match {pattern}")
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def scan_pool(needed: set[str]) -> tuple[dict[str, str], dict[str, Any]]:
    """Record dates of the PMIDs we need, read from the pre-window pool, plus the pool's latest record date."""
    dates: dict[str, str] = {}
    scanned, max_date = 0, ""
    with gzip.open(POOL, "rt", encoding="utf-8") as fh:
        for line in fh:
            scanned += 1
            m = PMID_LINE.match(line)
            if m is None:
                rec = json.loads(line)
                pmid, record_date = str(rec["pmid"]), str(rec["record_date"])
            else:
                pmid, record_date = m.group(1), m.group(2)
            if record_date > max_date:
                max_date = record_date
            if pmid in needed:
                dates[pmid] = record_date
    return dates, {"lines_scanned": scanned, "max_record_date": max_date, "pmids_needed": len(needed),
                   "pmids_found": len(dates)}


def first_sentences(state: str, n: int = 2, limit: int = 300) -> str:
    text = " ".join(state.split())
    return " ".join(SENTENCE_BREAK.split(text)[:n])[:limit]


def pick_padding(item_id: str, source: str, record_id: str,
                 pool: dict[str, list[tuple[str, str]]]) -> str | None:
    """Two sentences from another record of the same source (never the same record)."""
    candidates = pool.get(source, [])
    if not candidates:
        return None
    start = int(stable_hash({"pad": item_id}, length=8), 16) % len(candidates)
    for step in range(len(candidates)):
        rec, text = candidates[(start + step) % len(candidates)]
        if rec != record_id:
            return text
    return None


def apply_transform(name: str, row: dict[str, Any], pool: dict[str, list[tuple[str, str]]]) -> dict[str, Any] | None:
    if name == "reverse_options":
        return reverse_options(row)
    if name == "pad_state":
        padding = pick_padding(row["item_id"], row["source"], row["source_record_id"], pool)
        return pad_state(row, [padding]) if padding else None
    if name == "plant_instruction":
        return plant_instruction(row)
    if name == "replace_gold_with_none":
        return replace_gold_with_none(row)
    raise ValueError(name)


def leak_add(total: Counter, row: dict[str, Any], protected_text: dict[str, set[str]],
             protected_records: dict[str, set[str]]) -> None:
    key = row_text_key(row)
    for name, keys in protected_text.items():
        if key in keys:
            total[f"text:{name}"] += 1
    rec = str(row["source_record_id"])
    for name, ids in protected_records.items():
        if rec in ids:
            total[f"record:{name}"] += 1


def with_pair(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "pair_id": row["item_id"]}


def convert_sources(pool_dates: dict[str, str], drops: dict[str, Counter]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    new_train: list[dict[str, Any]] = []
    new_dev: list[dict[str, Any]] = []

    def keep(row: dict[str, Any] | None, reason: str, source: str, split: str) -> None:
        drops[f"{source}/{split}"][reason] += 1
        if row is not None:
            (new_train if row["split"] == "train" else new_dev).append(row)

    for split_name, out_split in (("train", "train"), ("validation", "dev")):
        for rec in load_parquet(f"datasets--tau--commonsense_qa/snapshots/*/data/{split_name}-*.parquet"):
            row, reason = commonsense_row(rec, out_split)
            keep(row, reason, "commonsenseqa", out_split)
        for rec in load_parquet(f"datasets--allenai--qasc/snapshots/*/data/{split_name}-*.parquet"):
            row, reason = qasc_row(rec, out_split)
            keep(row, reason, "qasc", out_split)
        for rec in load_parquet(f"datasets--bigbio--pubhealth/snapshots/*/pubhealth_source/{split_name}/*.parquet"):
            row, reason = pubhealth_row(rec, out_split)
            keep(row, reason, "pubhealth", out_split)
        for rec in load_parquet(f"datasets--bigbio--chemprot/snapshots/*/chemprot_bigbio_kb/{split_name}/*.parquet"):
            pool_date = pool_dates.get(str(rec["document_id"]))
            if pool_date is None:
                drops[f"chemprot/{out_split}"]["document_not_in_prewindow_pool"] += 1
                continue
            rows, reasons = chemprot_rows(rec, pool_date=pool_date, split=out_split,
                                          max_per_doc=CHEMPROT_MAX_PER_DOC)
            for reason, count in reasons.items():
                if reason != "kept":
                    drops[f"chemprot/{out_split}"][reason] += count
            for row in rows:
                keep(row, "kept", "chemprot", out_split)
        for rec in load_parquet(f"datasets--bigbio--evidence_inference/snapshots/*/evidence-inference_bigbio_qa/{split_name}/*.parquet"):
            pool_date = pool_dates.get(str(rec["document_id"]))
            if pool_date is None:
                drops[f"evidence_inference/{out_split}"]["document_not_in_prewindow_pool"] += 1
                continue
            row, reason = evidence_row(rec, pool_date=pool_date, split=out_split)
            keep(row, reason, "evidence_inference", out_split)
    return new_train, new_dev


def load_generator_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    counts: Counter = Counter()
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for path, bucket in ((GEN_DIR / "train.jsonl", train), (GEN_DIR / "dev.jsonl", dev)):
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["held_out"]:
                counts[f"held_out_excluded/{rec['split']}"] += 1
                continue
            bucket.append(generator_row(rec))
    return train, dev, counts


def protected_sets() -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    """Text keys and record ids of the protected sets: v0.2 test, v0.2 dev, the external panel, the robustness pack,
    the held-out generators and the held-out templates."""
    text: dict[str, set[str]] = defaultdict(set)
    records: dict[str, set[str]] = defaultdict(set)
    for path in sorted((V02_DIR / "fresh").glob("*.jsonl")) + sorted((V02_DIR / "tier1").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["template_id"] in HELD_OUT_TEMPLATES:
                text["v02_heldout_templates"].add(row_text_key(rec))
            if rec["split"] in ("test", "dev"):
                name = f"v02_{rec['split']}"
                text[name].add(row_text_key(rec))
                records[name].add(str(rec["source_record_id"]))
    for path in (EXT_DIR / "panel.jsonl", EXT_DIR / "robustness.jsonl"):
        name = "panel" if path.name == "panel.jsonl" else "robustness"
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            text[name].add(row_text_key(rec))
            records[name].add(str(rec["source_record_id"]))
    for path in (GEN_DIR / "dev.jsonl", GEN_DIR / "test.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["held_out"]:
                text["heldout_generators"].add(row_text_key(generator_row(rec)))
    train_text = dict(text)
    dev_text = {k: v for k, v in text.items() if k != "v02_dev"}
    train_records = dict(records)
    dev_records = {k: v for k, v in records.items() if k != "v02_dev"}
    return train_text, dev_text, train_records, dev_records


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    drops: dict[str, Counter] = defaultdict(Counter)

    print("pool dates ...", flush=True)
    needed = set()
    for rec in load_parquet("datasets--bigbio--chemprot/snapshots/*/chemprot_bigbio_kb/*/*.parquet"):
        needed.add(str(rec["document_id"]))
    for rec in load_parquet("datasets--bigbio--evidence_inference/snapshots/*/evidence-inference_bigbio_qa/*/*.parquet"):
        needed.add(str(rec["document_id"]))
    pool_dates, pool_info = scan_pool(needed)
    print(f"  pool: {pool_info}", flush=True)

    print("converting replay and catalog sources ...", flush=True)
    new_train, new_dev = convert_sources(pool_dates, drops)
    gen_train, gen_dev, gen_counts = load_generator_rows()
    print(f"  new train rows {len(new_train)}, new dev rows {len(new_dev)}, generator train {len(gen_train)}, "
          f"generator dev {len(gen_dev)}", flush=True)

    print("screening new templates on dev ...", flush=True)
    screens: dict[str, dict[str, Any]] = {}
    passed: set[str] = set()
    for template in NEW_TEMPLATES:
        train_t = [with_pair(r) for r in new_train if r["template_id"] == template]
        dev_t = [with_pair(r) for r in new_dev if r["template_id"] == template]
        if not train_t or not dev_t:
            screens[template] = {"passes": False, "failures": ["no train or dev rows"], "n_train": len(train_t),
                                 "n_dev": len(dev_t)}
            continue
        rep = screen_generator(template, False, dev_t, train_t, "own train split", threshold=SCREEN_THRESHOLD)
        screens[template] = {**rep.as_dict(), "n_train": len(train_t), "n_dev": len(dev_t)}
        if rep.passes:
            passed.add(template)
        print(f"  {template}: n_train={len(train_t)} n_dev={len(dev_t)} gold_in_state={rep.gold_in_state_hits} "
              f"nb={rep.naive_bayes_macro} passes={rep.passes} {rep.failures}", flush=True)
    for template in NEW_TEMPLATES:
        if template in passed:
            continue
        for rows, split in ((new_train, "train"), (new_dev, "dev")):
            failed = [r for r in rows if r["template_id"] == template]
            if failed:
                drops[f"{template}/{split}"]["template_failed_screen"] += len(failed)
    new_train = [r for r in new_train if r["template_id"] in passed]
    new_dev = [r for r in new_dev if r["template_id"] in passed]

    print("protected sets ...", flush=True)
    train_text, dev_text, train_records, dev_records = protected_sets()

    print("counting the student_v1 mix ...", flush=True)
    base_counts: Counter = Counter()
    source_counts: Counter = Counter()
    visibility_n: Counter = Counter()
    visibility_hit: Counter = Counter()
    sex_seen = Counter()
    student_dates = Counter()
    pad_pool: dict[str, list[tuple[str, str]]] = defaultdict(list)
    n_student = 0
    with STUDENT_TRAIN.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            n_student += 1
            template = r["template_id"]
            base_counts[template] += 1
            source_counts[r["source"]] += 1
            if not r["record_date"]:
                student_dates["undated"] += 1
            elif r["record_date"] >= WINDOW:
                student_dates["in_window"] += 1
            if r["qtype"] != "noul":
                visibility_n[template] += 1
                if gold_in_state(r):
                    visibility_hit[template] += 1
            if template == "ct_eligibility_sex_choice_v1":
                sex_seen["n"] += 1
                if SEX_WORDS.search(r["state"]):
                    sex_seen["sex_word_in_state"] += 1
            if len(pad_pool[r["source"]]) < PAD_POOL_PER_SOURCE:
                snippet = first_sentences(r["state"])
                if snippet:
                    pad_pool[r["source"]].append((r["source_record_id"], snippet))
    for row in new_train + gen_train:
        if len(pad_pool[row["source"]]) < PAD_POOL_PER_SOURCE:
            snippet = first_sentences(row["state"])
            if snippet:
                pad_pool[row["source"]].append((row["source_record_id"], snippet))

    new_train_all = gen_train + new_train
    n_new = len(new_train_all)
    n_base = n_student + n_new
    allowance_total = math.floor(CAP * n_base)
    allowance = {t: max(0, allowance_total - base_counts[t] - sum(
        1 for r in new_train_all if r["template_id"] == t)) for t in set(base_counts) | {r["template_id"] for r in new_train_all}}

    print(f"writing train: student {n_student}, new {n_new}, per-template allowance from the 8 % cap "
          f"({allowance_total} rows)", flush=True)
    leak_train: Counter = Counter()
    aug_counts: Counter = Counter()
    aug_requested: Counter = Counter()
    aug_by_template: Counter = Counter()
    aug_not_applicable: Counter = Counter()
    aug_cap_skipped: Counter = Counter()
    final_template: Counter = Counter()
    final_source: Counter = Counter()
    final_qtype: Counter = Counter()
    train_path = OUT_DIR / "train.jsonl"
    written_base = 0
    written_aug = 0
    rows_iter = list(new_train_all)

    def write_row(out_fh, row: dict[str, Any], *, base: bool) -> None:
        nonlocal written_base, written_aug
        out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        final_template[row["template_id"]] += 1
        final_source[row["source"]] += 1
        final_qtype[row["qtype"]] += 1
        leak_add(leak_train, row, train_text, train_records)
        if base:
            written_base += 1
        else:
            written_aug += 1

    def augment(row: dict[str, Any], out_fh) -> None:
        for name in TRANSFORMS:
            if not select(row["item_id"], name, AUG_FRACTION[name]):
                continue
            aug_requested[name] += 1
            if allowance.get(row["template_id"], 0) <= aug_by_template[row["template_id"]]:
                aug_cap_skipped[name] += 1
                continue
            aug = apply_transform(name, row, pad_pool)
            if aug is None:
                aug_not_applicable[name] += 1
                continue
            aug_counts[name] += 1
            aug_by_template[row["template_id"]] += 1
            write_row(out_fh, aug, base=False)

    with STUDENT_TRAIN.open(encoding="utf-8") as src, train_path.open("w", encoding="utf-8") as out:
        for line in src:
            r = json.loads(line)
            out.write(line if line.endswith("\n") else line + "\n")
            written_base += 1
            final_template[r["template_id"]] += 1
            final_source[r["source"]] += 1
            final_qtype[r["qtype"]] += 1
            leak_add(leak_train, r, train_text, train_records)
            augment(r, out)
        for row in rows_iter:
            write_row(out, row, base=True)
            augment(row, out)
    final_n = written_base + written_aug
    print(f"  train rows written: base {written_base}, augmented {written_aug}, total {final_n}", flush=True)

    print("writing dev ...", flush=True)
    dev_path = OUT_DIR / "dev.jsonl"
    leak_dev: Counter = Counter()
    dev_counts: Counter = Counter()
    dev_template: Counter = Counter()
    dev_rows_student = 0
    with dev_path.open("w", encoding="utf-8") as out:
        with STUDENT_DEV.open(encoding="utf-8") as src:
            for line in src:
                r = json.loads(line)
                out.write(line if line.endswith("\n") else line + "\n")
                dev_rows_student += 1
                dev_template[r["template_id"]] += 1
                dev_counts[r["source"]] += 1
                leak_add(leak_dev, r, dev_text, dev_records)
        for row in gen_dev + new_dev:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            dev_template[row["template_id"]] += 1
            dev_counts[row["source"]] += 1
            leak_add(leak_dev, row, dev_text, dev_records)
    n_dev = dev_rows_student + len(gen_dev) + len(new_dev)

    shares = {t: n / final_n for t, n in final_template.items()}
    top = max(shares.items(), key=lambda kv: kv[1])
    aug_share = written_aug / final_n
    replay_base = sum(1 for r in new_train if r["source"] in ("commonsenseqa", "qasc"))
    budget = min(final_n, MIX_BUDGET)
    build = {
        "built_at_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "elapsed_s": round(time.time() - started, 1),
        "pool": pool_info,
        "drops_by_source_split": {k: dict(v) for k, v in sorted(drops.items())},
        "generator_counts": dict(gen_counts),
        "screens": screens,
        "templates_kept_after_screen": sorted(passed),
        "student_v1": {"train_rows": n_student, "dev_rows": dev_rows_student, "date_checks": dict(student_dates),
                       "template_counts": dict(base_counts), "source_counts": dict(source_counts)},
        "visibility_gold_in_state": {t: {"n": visibility_n[t], "hits": visibility_hit[t],
                                         "share": round(visibility_hit[t] / visibility_n[t], 4)}
                                     for t in sorted(visibility_n)},
        "ct_eligibility_sex": {"n": sex_seen["n"], "sex_word_in_state": sex_seen["sex_word_in_state"],
                               "share": round(sex_seen["sex_word_in_state"] / max(1, sex_seen["n"]), 4)},
        "augmentation": {"requested": dict(aug_requested), "applied": dict(aug_counts),
                         "not_applicable": dict(aug_not_applicable), "cap_skipped": dict(aug_cap_skipped),
                         "rows_written": written_aug, "share_of_train": round(aug_share, 4),
                         "max_share_allowed": AUG_MAX_SHARE},
        "cap": {"max_template_share": round(top[1], 4), "max_template": top[0], "limit": CAP,
                "ok": top[1] <= CAP},
        "leakage_train": dict(leak_train), "leakage_dev": dict(leak_dev),
        "train_rows": final_n, "dev_rows": n_dev, "budget_examples": budget,
        "replay_share_base": round(replay_base / final_n, 4),
        "final_template_counts": dict(final_template), "final_source_counts": dict(final_source),
        "final_qtype_counts": dict(final_qtype), "dev_template_counts": dict(dev_template),
        "dev_source_counts": dict(dev_counts),
        "inputs": {"student_train_sha256": file_sha256(STUDENT_TRAIN), "student_dev_sha256": file_sha256(STUDENT_DEV)},
        "outputs": {"train": {"rows": final_n, "sha256": file_sha256(train_path)},
                    "dev": {"rows": n_dev, "sha256": file_sha256(dev_path)}},
    }
    write_json(OUT_JSON, build)
    print(f"done in {build['elapsed_s']} s: train {final_n} (budget {budget}), dev {n_dev}; "
          f"leakage train {dict(leak_train)} dev {dict(leak_dev)}; cap ok {build['cap']['ok']}", flush=True)


if __name__ == "__main__":
    main()
