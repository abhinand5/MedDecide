"""d3_distill step 1 (CPU): build the 200,000-item label pool from MedMCQA train and the code-generated families.

Sources (PLAN.md step 1, with the operator's MIMIC constraint): the MedMCQA official train split, the
``meddecide.gen`` clinical-decision families that are not held out, and the ``meddecide.mr`` audit generators that need
no MIMIC and no teacher. The MIMIC-grounded synth block is excluded by the operator's "don't touch MIMIC data"
instruction (STATE.md, constraint 1). Gold is kept alongside every row.

Exclusions. The frozen benchmark files are not on this machine (`data/` does not exist), so the only test-side text
available is the MedMCQA official validation split, which is the knowledge test side (the bench's MedMCQA test set,
see src/meddecide/bench/tier1/mcq.py). The pool is checked against it by exact normalised text key, and the absent
files are recorded as NOT MEASURED with the reason; the d3_bench run files hold predictions but no item text, so they
cannot be used for a text-hash check either. Within-pool duplicates are dropped and counted.

Every filter returns counts and reasons. The pool is written to the gitignored training tree, the counts and the file
SHA-256 to the gitignored outputs manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from meddecide.gen.core import split_of  # noqa: E402
from meddecide.gen.families import FAMILIES  # noqa: E402
from meddecide.mr.generators import GENERATORS as MR_GENERATORS  # noqa: E402
from meddecide.mr.generators import is_held_out  # noqa: E402
from meddecide.utils.hashing import file_sha256, stable_hash  # noqa: E402

POOL = REPO / "data" / "train" / "d3_distill" / "pool_200k.jsonl"
MANIFEST = REPO / "outputs" / "d3_distill" / "step1" / "pool_manifest.json"
TARGET = 200_000
DEV_BUCKETS = 5  # 5 % of rows, by stable hash of row_id: the train-side dev carve for step 2 arm selection
MEDMCQA_QUESTION = "Which of the following is the best answer?"
SEED_SALT = "d3_distill/step1/pool/v1"


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def key_of(state: Any, question: Any, labels: list[str]) -> str:
    """Normalised text key; option labels are sorted so an option-order change cannot hide a duplicate."""
    body = "\n".join([norm(state), norm(question), "|".join(sorted(norm(label) for label in labels))])
    return stable_hash(body, length=32)


def state_key(state: Any) -> str:
    return stable_hash(norm(state), length=32)


def medmcqa_rows() -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The eligible MedMCQA train rows and the counts of every reason a row was dropped."""
    from datasets import load_dataset

    train = load_dataset("openlifescienceai/medmcqa", split="train")
    valid = load_dataset("openlifescienceai/medmcqa", split="validation")
    test_state_keys = {state_key(r["question"]) for r in valid}
    test_keys = {
        key_of(r["question"], MEDMCQA_QUESTION, [r["opa"], r["opb"], r["opc"], r["opd"]]) for r in valid
    }
    counts: Counter[str] = Counter()
    counts["medmcqa_train_rows"] = len(train)
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for r in train:
        labels = [r.get("opa"), r.get("opb"), r.get("opc"), r.get("opd")]
        cop = r.get("cop")
        if not all(isinstance(x, str) and x.strip() for x in labels) or cop not in (0, 1, 2, 3):
            counts["dropped_unparseable"] += 1
            continue
        state = str(r["question"]).strip()
        if not state:
            counts["dropped_empty_state"] += 1
            continue
        if state_key(state) in test_state_keys or key_of(state, MEDMCQA_QUESTION, labels) in test_keys:
            counts["dropped_in_medmcqa_validation"] += 1
            continue
        key = key_of(state, MEDMCQA_QUESTION, labels)
        if key in seen:
            counts["dropped_duplicate_text"] += 1
            continue
        seen.add(key)
        options = [{"key": k, "label": str(lab).strip()} for k, lab in zip("ABCD", labels, strict=True)]
        kept.append(
            {
                "row_id": stable_hash(
                    {"source": "medmcqa", "record": str(r.get("id")), "template": "medmcqa_4opt_v1"},
                    length=16,
                ),
                "source": "medmcqa",
                "template_id": "medmcqa_4opt_v1",
                "qtype": "choice",
                "state": state,
                "question": MEDMCQA_QUESTION,
                "options": options,
                "gold": "ABCD"[int(cop)],
                "meta": {
                    "subject_name": str(r.get("subject_name", "")),
                    "topic_name": str(r.get("topic_name", "")),
                    "choice_type": str(r.get("choice_type", "")),
                },
            }
        )
    counts["medmcqa_kept"] = len(kept)
    return kept, dict(sorted(counts.items()))


def synthetic_sources() -> list[tuple[str, Any]]:
    """The non-held-out code-generated families, ``(source_name, generator)``."""
    sources = [("gen", f) for f in FAMILIES if not f.held_out]
    sources += [("mr", g) for g in MR_GENERATORS]
    return sources


def synthetic_rows(needed: int, exclusions: dict[str, set[str]], counts: Counter[str]) -> list[dict[str, Any]]:
    """Generate round-robin over the non-held-out families until ``needed`` rows pass every check.

    Only patient ids whose ``split_of`` bucket is train are used (twins share a patient and a split), and the
    namespace is ``d3distill-<generator>-<k>``, fresh against osler's generator ids, so no generated item is a copy of
    a benchmark or osler training item.
    """
    rows: list[dict[str, Any]] = []
    sources = synthetic_sources()
    counts["synthetic_sources"] = len(sources)
    k = 0
    per_source = Counter()
    while len(rows) < needed:
        progressed = False
        for _kind, gen in sources:
            if len(rows) >= needed:
                break
            while split_of(pid := f"d3distill-{gen.name}-{k:07d}") != "train":
                counts["synthetic_pid_not_train"] += 1
                k += 1
            counts["synthetic_pids_used"] += 1
            for item in gen.make(pid):
                progressed = True
                if is_held_out(item.meta) or getattr(gen, "held_out", False):
                    counts["synthetic_dropped_held_out"] += 1
                    continue
                row_dict = item.as_dict()
                options = [{"key": str(o["key"]), "label": str(o["label"])} for o in row_dict["options"]]
                key = key_of(row_dict["state"], row_dict["question"], [o["label"] for o in options])
                if state_key(row_dict["state"]) in exclusions["state_keys"] or key in exclusions["keys"]:
                    counts["synthetic_dropped_benchmark_text"] += 1
                    continue
                options_keys = [o["key"] for o in options]
                if (
                    row_dict["qtype"] not in ("choice", "noul", "score")
                    or not 2 <= len(options) <= 255
                    or row_dict["gold"] not in options_keys
                    or len(set(options_keys)) != len(options_keys)
                    or not str(row_dict["state"]).strip()
                ):
                    counts["synthetic_dropped_malformed"] += 1
                    continue
                rows.append(
                    {
                        "row_id": str(row_dict["item_id"]),
                        "source": f"{_kind}:{gen.name}",
                        "template_id": str(row_dict["generator"]),
                        "qtype": str(row_dict["qtype"]),
                        "state": str(row_dict["state"]),
                        "question": str(row_dict["question"]),
                        "options": options,
                        "gold": str(row_dict["gold"]),
                        "meta": {"family": row_dict["family"], "role": row_dict["role"], "pair_id": row_dict["pair_id"]},
                    }
                )
                per_source[f"{_kind}:{gen.name}"] += 1
            k += 1
        if not progressed:
            raise SystemExit("the synthetic generators produced nothing: stop")
    counts.update({f"synthetic_kept::{name}": n for name, n in sorted(per_source.items())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    args = ap.parse_args()

    counts: Counter[str] = Counter()
    medmcqa, medmcqa_counts = medmcqa_rows()
    counts.update(medmcqa_counts)
    counts["target"] = args.target
    if len(medmcqa) > args.target:
        raise SystemExit(f"{len(medmcqa)} eligible MedMCQA rows exceed the target; add a selection rule first")

    # Test-side text available on this machine: MedMCQA validation only. The synthetic rows are checked against it too.
    from datasets import load_dataset

    valid = load_dataset("openlifescienceai/medmcqa", split="validation")
    exclusions = {
        "state_keys": {state_key(r["question"]) for r in valid},
        "keys": {key_of(r["question"], MEDMCQA_QUESTION, [r["opa"], r["opb"], r["opc"], r["opd"]]) for r in valid},
    }
    counts["benchmark_text_checks_available"] = 1  # medmcqa validation

    # Generate a small margin: a cross-source duplicate is dropped after generation, and the pool must still reach
    # the target exactly. The truncation back to the target is counted, never silent.
    needed = args.target - len(medmcqa) + max(64, args.target // 200)
    counts["synthetic_needed"] = needed
    synthetic = synthetic_rows(needed, exclusions, counts)

    # One duplicate check across both sources, then a deterministic mixed order.
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in medmcqa + synthetic:
        key = key_of(row["state"], row["question"], [o["label"] for o in row["options"]])
        if key in seen:
            counts["dropped_pool_duplicate"] += 1
            continue
        seen.add(key)
        row["split"] = "dev" if int(stable_hash(row["row_id"], length=8), 16) % 100 < DEV_BUCKETS else "train"
        rows.append(row)
    if len(rows) < args.target:
        raise SystemExit(f"pool has {len(rows)} rows, short of the {args.target} target: stop")
    if len(rows) > args.target:
        # Only duplicates could remove rows, and they are counted; keep the deterministic first `target` if any remain.
        rows = rows[: args.target]
        counts["truncated_to_target"] += 1
    rows.sort(key=lambda r: stable_hash({"row_id": r["row_id"], "salt": SEED_SALT}, length=16))
    for order, row in enumerate(rows):
        row["order"] = order

    args.pool.parent.mkdir(parents=True, exist_ok=True)
    with args.pool.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_source = Counter(r["source"] for r in rows)
    per_template = Counter(r["template_id"] for r in rows)
    per_qtype = Counter(r["qtype"] for r in rows)
    per_split = Counter(r["split"] for r in rows)
    manifest = {
        "pool": str(args.pool),
        "pool_sha256": file_sha256(args.pool),
        "rows": len(rows),
        "target": args.target,
        "per_source": dict(sorted(per_source.items())),
        "per_template": dict(sorted(per_template.items())),
        "per_qtype": dict(sorted(per_qtype.items())),
        "per_split": dict(sorted(per_split.items())),
        "counts": dict(sorted(counts.items())),
        "dev_rule": f"stable_hash(row_id) % 100 < {DEV_BUCKETS}; train-side only",
        "exclusions": {
            "enforced": ["medmcqa official validation split (knowledge test side), exact normalised text key",
                         "within-pool duplicate text", "held-out generator families and held-out meta families",
                         "generator patients not in the train split"],
            "not_measured": {
                "y9_knowledge_full": "data/bench/osler_v4/y9/knowledge.jsonl is absent; the d3_bench run rows store no item text",
                "y9_panel": "data/bench/v0.3_ext/panel.jsonl is absent",
                "mr_suites": "data/bench/osler_v4/y9/mr_*.jsonl absent; only the d3_bench mr_verdict predictions are here (no text)",
                "mimic_audit_suite": "no MIMIC data on this machine; excluded by operator instruction",
                "mednli_test": "data/physionet/mednli is absent",
                "cg_v2_dev_proxy": "data/bench/osler_v4/cg_v2.jsonl and the osler_v1 B0-B3 blocks are absent",
            },
        },
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "per_source", "per_split", "per_qtype")}, indent=1))
    print(json.dumps(manifest["counts"], indent=1))


if __name__ == "__main__":
    main()
