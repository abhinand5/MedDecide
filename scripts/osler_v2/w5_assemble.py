"""W5 step 1 (osler_v2 ADVISORY W5 and W6): fast dev v2, the CodiEsp dev carve-out, and the training blocks C0, C1, C2.

Everything is built from rows that already exist in mix format: osler_v1's blocks B1 (C0's base), B2 (teacher prose) and B3
(real text), osler_v1's fast dev (fastdev_mix.jsonl, the dev set of its block runs), the knowledge dev v2 and the EHR-Rel split
from this loop. Teacher dev rows and CodiEsp rows are converted with osler_v1's own converter (v7_blocks.mix_row_from_item),
so a row has the same shape in every file.

Fast dev v2 (ADVISORY W5 step 1): the osler_v1 fast dev (5,723) + knowledge dev v2 (4,000) + up to 400 dev items per candidate
source not already in it: indicator prose, narrative pages, EHR-Rel (dev) and CodiEsp (dev, from the carve-out). MTSamples and
PMC dev are already in the osler_v1 fast dev. The partitions are recorded (MR, general, knowledge).

CodiEsp carve-out: 15 % of the CodiEsp train cases, by a stable hash of the case id. All their rows leave training; up to 400
of their rows form CodiEsp dev. The carve-out is written as hashed case ids only.

Blocks (cumulative; the keep rule decides which one each block builds on, see W6):
- C0 = B1 + the fixed replay set (replay_v2_train.jsonl);
- C1 = C0 + real text: MTSamples train, CodiEsp train minus the carve-out, PMC train, EHR-Rel train;
- C2 = C1 + teacher prose (indicator and narrative, train rows of B2).

Writes data/bench/osler_v2/fastdev_v2.jsonl, data/bench/osler_v2/ehr_rel_test.jsonl (O2 format), data/train/osler_v2/block_C{0,1,2}.jsonl
and the parts, and outputs/osler_v2/W5/assembly.json (counts, shares and checks; no item text).

Usage: source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler_v2/w5_assemble.py > outputs/osler_v2/W5/logs/assemble.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.mix.leakage import row_text_key
from meddecide.mix.rows import make_row
from meddecide.utils.hashing import file_sha256, stable_hash
from meddecide.utils.io import write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
V1_BLOCKS = REPO / "data" / "train" / "osler_v1"
V2_TRAIN = REPO / "data" / "train" / "osler_v2"
V2_BENCH = REPO / "data" / "bench" / "osler_v2"
V2_GEN = REPO / "data" / "gen" / "osler_v2"
FASTDEV_V1 = REPO / "data" / "bench" / "osler_mr_v0" / "fastdev_mix.jsonl"
TEACHER = REPO / "data" / "gen" / "osler_v1" / "teacher"
KNOWLEDGE = V2_BENCH / "knowledge_dev_v2.jsonl"
REPLAY = V2_TRAIN / "replay_v2_train.jsonl"
OUT = REPO / "outputs" / "osler_v2" / "W5" / "assembly.json"
CODIESP_CARVE_PERCENT = 15
CAP_PER_SOURCE = 400
MR_TEMPLATES_PREFIX = ("mr_", "synthea_", "pmc_", "mtsamples_", "codiesp_", "mkurman_", "ehr_rel_")
KNOWLEDGE_TEMPLATES = {"medqa_usmle4_v1", "medmcqa_4opt_v1"}


def load_v7_converter():
    """osler_v1's mix_row_from_item, loaded from its script (the same conversion as B2 and B3)."""
    spec = importlib.util.spec_from_file_location("v7_blocks", REPO / "scripts" / "osler_v1" / "v7_blocks.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mix_row_from_item


def partition_of(template: str) -> str:
    if template in KNOWLEDGE_TEMPLATES:
        return "knowledge"
    if template.startswith(MR_TEMPLATES_PREFIX):
        return "mr"
    return "general"


def codiesp_case(row: dict[str, Any]) -> str:
    """The case id is the middle part of the CodiEsp source record id: codiesp_icd_support_v1:codiesp:<case>:<code>."""
    return row["source_record_id"].split(":")[2]


def carve_out_cases(cases: set[str]) -> set[str]:
    """Exactly CODIESP_CARVE_PERCENT of the cases: the lowest stable hashes (deterministic, no file order)."""
    ranked = sorted(cases, key=lambda c: stable_hash({"codiesp_case": c}, length=16))
    return set(ranked[:round(len(cases) * CODIESP_CARVE_PERCENT / 100)])


def to_o2(row: dict[str, Any]) -> dict[str, Any]:
    """A mix row as an O2 evaluation item (the runners' format: benchmark, set_name, template, options, gold)."""
    return {"item_id": row["item_id"], "benchmark": "mix_v2", "set_name": "osler_v2_dev_or_test",
            "template_id": row["template_id"], "qtype": row["qtype"], "state": row["state"],
            "question": row["question"], "options": row["options"], "gold": row["gold"], "split": row["split"],
            "held_out": False, "source": row["source"]}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    mix_row_from_item = load_v7_converter()
    report: dict[str, Any] = {"written_utc": utcnow(), "git_commit": git_commit(REPO), "checks": {}}

    # 1. the teacher dev rows (osler_v1's teacher files; dev split only for dev; train rows come from block B2)
    teacher_dev = []
    for name, source, skill in (("indicator.jsonl", "mr_teacher", "mr_audit"),
                                ("narrative.jsonl", "mr_teacher", "mr_narrative")):
        raw = read_jsonl(TEACHER / name)
        rows = [mix_row_from_item(r, source=source, skill=skill) for r in raw if r["split"] == "dev"]
        teacher_dev.extend(sorted(rows, key=lambda r: stable_hash({"id": r["item_id"]}, length=16))[:CAP_PER_SOURCE])
    report["teacher_dev_rows"] = dict(Counter(r["template_id"] for r in teacher_dev))

    # 2. CodiEsp: the train rows of B3 (mix format); the carve-out by case id
    b3_real: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prose_train: list[dict[str, Any]] = []
    with (V1_BLOCKS / "block_B3.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["template_id"] in ("mtsamples_page_category_v1", "codiesp_icd_support_v1", "pmc_similar_patient_v1"):
                b3_real[row["template_id"]].append(row)
    codiesp_all = b3_real["codiesp_icd_support_v1"]
    cases = {codiesp_case(r) for r in codiesp_all}
    carve_cases = carve_out_cases(cases)
    codiesp_dev_pool = [r for r in codiesp_all if codiesp_case(r) in carve_cases]
    codiesp_dev = sorted(codiesp_dev_pool, key=lambda r: stable_hash({"id": r["item_id"]}, length=16))[:CAP_PER_SOURCE]
    codiesp_train = [r for r in codiesp_all if codiesp_case(r) not in carve_cases]
    report["codiesp"] = {"train_rows_in_B3": len(codiesp_all), "cases": len(cases),
                         "carve_out_cases": len(carve_cases), "carve_out_percent_of_cases": round(
                             100 * len(carve_cases) / len(cases), 2),
                         "carve_out_rows": len(codiesp_dev_pool), "dev_rows_kept": len(codiesp_dev),
                         "train_rows_after_carve_out": len(codiesp_train)}

    # 3. the other real text: MTSamples and PMC train rows (B3), EHR-Rel train rows (this loop)
    ehr_train = read_jsonl(V2_TRAIN / "ehr_rel_train.jsonl")
    ehr_dev = read_jsonl(V2_GEN / "ehr_rel" / "ehr_rel_dev.jsonl")
    ehr_test = read_jsonl(V2_GEN / "ehr_rel" / "ehr_rel_test.jsonl")
    ehr_dev_sel = sorted(ehr_dev, key=lambda r: stable_hash({"id": r["item_id"]}, length=16))[:CAP_PER_SOURCE]
    realtext_train = b3_real["mtsamples_page_category_v1"] + codiesp_train + b3_real["pmc_similar_patient_v1"] + ehr_train

    # 4. teacher prose train rows: B2 minus B1 (rows whose template is a prose template)
    with (V1_BLOCKS / "block_B2.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["template_id"] in ("mr_indicator_prose_v1", "synthea_narrative_page_v1"):
                prose_train.append(row)

    # 5. fast dev v2
    fastdev_v1 = read_jsonl(FASTDEV_V1)
    knowledge = [make_row(source=o["source"], template_id=o["template_id"], skill="medical_knowledge", qtype="choice",
                          state=o["state"], question=o["question"],
                          options=[(x["key"], x["label"]) for x in o["options"]], gold=o["gold"],
                          record_id=o["item_id"], record_url="", licence="as the source dataset",
                          record_date="", split="dev", tier="knowledge_dev_v2", meta={"set": "knowledge_dev_v2"})
                 for o in read_jsonl(KNOWLEDGE)]
    candidates = fastdev_v1 + knowledge + teacher_dev + ehr_dev_sel + codiesp_dev
    fastdev_v2, seen_ids = [], set()
    for row in candidates:  # an item listed twice is counted once (the source repeated a record)
        if row["item_id"] not in seen_ids:
            seen_ids.add(row["item_id"])
            fastdev_v2.append(row)
    report["fastdev_v2_duplicate_item_ids_dropped"] = len(candidates) - len(fastdev_v2)
    report["checks"]["fastdev_v2_items"] = len(fastdev_v2)
    report["checks"]["fastdev_v2_unique_item_ids"] = len({r["item_id"] for r in fastdev_v2})
    report["fastdev_v2"] = {
        "osler_v1_fast_dev": len(fastdev_v1), "knowledge_dev_v2": len(knowledge), "teacher_dev": len(teacher_dev),
        "ehr_rel_dev": len(ehr_dev_sel), "codiesp_dev": len(codiesp_dev),
        "by_partition": dict(Counter(partition_of(r["template_id"]) for r in fastdev_v2)),
        "by_template": dict(Counter(r["template_id"] for r in fastdev_v2)),
    }
    write_jsonl(V2_BENCH / "fastdev_v2.jsonl", fastdev_v2)
    write_jsonl(V2_BENCH / "ehr_rel_test.jsonl", [to_o2(r) for r in ehr_test])
    report["ehr_rel_test_items"] = len(ehr_test)

    # 6. blocks: C0 = B1 + replay; C1 = C0 + real text; C2 = C1 + prose. Knowledge dev rows must not be in any block.
    knowledge_states = {(r["template_id"], row_text_key(r)) for r in knowledge}
    def clean(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
        kept = [r for r in rows if (r["template_id"], row_text_key(r)) not in knowledge_states]
        report["checks"][f"{name}_knowledge_dev_removed"] = len(rows) - len(kept)
        return kept

    replay = read_jsonl(REPLAY)
    b1 = read_jsonl_stream(V1_BLOCKS / "block_B1.jsonl")
    c0 = clean(b1 + replay, "C0")
    c1 = c0 + clean(realtext_train, "realtext")
    c2 = c1 + clean(prose_train, "prose")
    for name, rows in (("C0", c0), ("C1", c1), ("C2", c2)):
        path = V2_TRAIN / f"block_{name}.jsonl"
        write_jsonl(path, rows)
        templates = Counter(r["template_id"] for r in rows)
        report[f"block_{name}"] = {"rows": len(rows), "file": str(path.relative_to(REPO)),
                                   "sha256": file_sha256(path), "by_template_top": templates.most_common(12)}
    replay_rows = sum(1 for r in c0 if r["tier"] == "general_replay" and r["template_id"] in
                      {"arc_science_choice_v1", "boolq_passage_noul_v1", "snli_nli_choice_v1"})
    old_replay = sum(1 for r in c0 if r["template_id"] in
                     {"hellaswag_ending_choice_v1", "winogrande_blank_choice_v1", "csqa_choice_v1", "qasc_science_choice_v1"})
    shares = {}
    for name, rows in (("C0", c0), ("C1", c1), ("C2", c2)):
        rep = sum(1 for r in rows if r["template_id"] in {"arc_science_choice_v1", "boolq_passage_noul_v1",
                                                         "snli_nli_choice_v1", "hellaswag_ending_choice_v1",
                                                         "winogrande_blank_choice_v1", "csqa_choice_v1",
                                                         "qasc_science_choice_v1"})
        shares[name] = round(rep / len(rows), 4)
    report["replay_share"] = {"definition": "rows of the v1 replay templates plus the v2 share-alike templates",
                              "by_block": shares, "new_replay_rows": replay_rows, "v1_replay_rows": old_replay,
                              "target": 0.20}
    # the new dev extras must not share exact text with the largest training block (C2)
    c2_keys = {row_text_key(r) for r in c2}
    extras = teacher_dev + ehr_dev_sel + codiesp_dev
    report["checks"]["new_dev_rows_with_exact_text_in_C2"] = sum(row_text_key(r) in c2_keys for r in extras)
    report["checks"]["new_dev_rows"] = len(extras)
    write_json(OUT, report)
    print(json.dumps({"fastdev_v2": report["fastdev_v2"]["by_partition"], "blocks": {k: report[k]["rows"] for k in
                                                                                     ("block_C0", "block_C1", "block_C2")},
                      "replay_share": shares}), flush=True)


def read_jsonl_stream(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


if __name__ == "__main__":
    main()
