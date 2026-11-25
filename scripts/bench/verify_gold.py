#!/usr/bin/env python
"""F1: verify every tier-1 item's gold against its raw source record, independently.

For each of the eight tier-1 sources this script re-loads the raw HF dataset at the pinned
revision, recomputes the gold option *text* with a minimal mapping written from the dataset
card (in this file — deliberately not the loader code), and compares it with the built item.
It also compares the gold-class distribution with the raw answer-field distribution.

Usage:
    uv run python scripts/bench/verify_gold.py --tier1 data/bench/tier1 \
        --out outputs/bench_v0_fix0/F1/gold_verification.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.bench.schema import Item
from meddecide.bench.verify import VerificationReport, check_items_against_raw, normalize_label
from meddecide.utils.io import read_jsonl
from meddecide.utils.provenance import Provenance, utcnow

# Pinned dataset revisions (from data/bench/tier1/manifest.json, bench_v0 T3).
REVISIONS = {
    "medqa": "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18",
    "medmcqa": "91c6572c454088bf71b679ad90aa8dffcd0d5868",
    "pubmedqa": "9001f2853fb87cab8d220904e0de81ac6973b318",
    "mmlu_medical": "c30699e8356da336a370243923dbaf21066bb9fe",
    "medquad": "84ea67f83cec9692ad254eaa02c9731b24ecfe4c",
    "trec_covid": "7e16fde3016c639c7f856e803f4bab92645562c4",
    "nfcorpus": "b5026a0e96e8a7ac4f95f482a596389289d46269",
    "scifact": "b3b5335604bf5ee3c4447671af975ea25143d4f5",
}

# ---------------------------------------------------------------------------
# Independent raw -> gold-text mappings (written from the dataset cards)
# ---------------------------------------------------------------------------
def medqa_gold_text(raw: dict[str, Any]) -> str | None:
    key = str(raw.get("answer_idx", "")).strip()
    options = raw.get("options") or {}
    if key and key in options:
        return str(options[key])
    answer = raw.get("answer")
    return str(answer) if answer else None


def medmcqa_gold_text(raw: dict[str, Any]) -> str | None:
    """`cop` is the 0-based index into opa..opd (proven against `exp` in verify_gold --prove-cop)."""
    try:
        index = int(raw["cop"])
    except (KeyError, TypeError, ValueError):
        return None
    options = [raw.get("opa"), raw.get("opb"), raw.get("opc"), raw.get("opd")]
    if not 0 <= index < 4 or not options[index]:
        return None
    return str(options[index])


def pubmedqa_gold_text(raw: dict[str, Any]) -> str | None:
    decision = str(raw.get("final_decision", "")).strip().lower()
    return decision or None


def mmlu_gold_text(raw: dict[str, Any]) -> str | None:
    choices = raw.get("choices") or []
    try:
        index = int(raw["answer"])
    except (KeyError, TypeError, ValueError):
        return None
    if 0 <= index < len(choices):
        return str(choices[index])
    return None


def medquad_gold_text(raw: dict[str, Any]) -> str | None:
    text = str(raw.get("question_type", "")).strip().lower()
    return text or None


def relevance_gold_text(kind: str):
    """Relevance sources: gold is qrels membership (and grade for score items)."""

    def _noul(item: Item, judgements: dict[str, int]) -> str | None:
        grade = judgements.get(str(item.meta.get("corpus_id")))
        if grade is None:
            return None
        return "yes" if grade > 0 else "no"

    def _score(item: Item, judgements: dict[str, int]) -> str | None:
        grade = judgements.get(str(item.meta.get("corpus_id")))
        if grade is None:
            return None
        return str(grade)

    return _noul if kind == "noul" else _score


def _raw_index(dataset, key_field: str) -> dict[str, Any]:
    return {str(row[key_field]): row for row in dataset}


def _qrels_index(qrels) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in qrels:
        out.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    return out


def prove_cop_semantics() -> dict[str, Any]:
    """Independent proof that `cop` indexes the correct option, using the record's own `exp`.

    Not part of the loader path: the explanation text of each MedMCQA record is scanned for the
    option texts at index `cop` and `cop+1`. If `cop` were 1-based, option[cop+1] would be named
    as the answer far more often than option[cop].
    """
    import re

    from datasets import load_dataset

    dataset = load_dataset("openlifescienceai/medmcqa", split="validation")
    hits_cop = hits_next = usable = 0
    for row in dataset:
        exp = str(row.get("exp") or "").strip()
        if len(exp) < 20:
            continue
        try:
            index = int(row["cop"])
        except (TypeError, ValueError):
            continue
        options = [row.get("opa"), row.get("opb"), row.get("opc"), row.get("opd")]
        usable += 1

        def mentions(text: Any, explanation: str = exp) -> bool:
            text = str(text or "").strip()
            return (
                len(text) >= 4
                and re.search(re.escape(text), explanation, re.IGNORECASE) is not None
            )

        if 0 <= index < 4 and mentions(options[index]):
            hits_cop += 1
        if index + 1 < 4 and mentions(options[index + 1]):
            hits_next += 1
    return {
        "split": "openlifescienceai/medmcqa:validation",
        "rows_with_usable_explanation": usable,
        "naming_option_at_cop": hits_cop,
        "naming_option_at_cop_plus_1": hits_next,
        "share_naming_cop": hits_cop / usable if usable else None,
        "share_naming_cop_plus_1": hits_next / usable if usable else None,
        "conclusion": (
            "cop is 0-based: the explanation names option[cop] far more often than option[cop+1]"
            if hits_cop > hits_next
            else "INCONCLUSIVE: option[cop+1] is named at least as often as option[cop]"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier1", type=Path, default=Path("data/bench/tier1"))
    parser.add_argument("--out", type=Path, default=Path("outputs/bench_v0_fix0/F1/gold_verification.json"))
    parser.add_argument("--prove-cop", action="store_true", help="also re-run the cop proof")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    wanted = set(args.only) if args.only else None

    def want(name: str) -> bool:
        return wanted is None or name in wanted

    reports: dict[str, Any] = {}
    raw_distributions: dict[str, Any] = {}

    def load_items(source: str) -> list[Item]:
        path = args.tier1 / f"{source}.jsonl"
        rows, report = read_jsonl(path, Item)
        report.check_closes()
        return rows

    # ---- MedQA -----------------------------------------------------------
    if want("medqa"):
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test", revision=REVISIONS["medqa"])
        items = [i for i in load_items("medqa") if str(i.split) == "test"]
        # the loader hashes the question into the record id, so join on the question text itself
        by_question = {normalize_label(row["question"]): row for row in ds}

        def medqa_key(row: dict[str, Any], item: Item) -> str | None:
            options = row.get("options") or {}
            answer_idx = str(row.get("answer_idx", "")).strip()
            if answer_idx not in options:
                return None
            position = sorted(options).index(answer_idx)   # the loader orders options by key
            return item.option_keys[position] if position < item.n_options else None

        def medqa_text(row: dict[str, Any], _item: Item) -> str | None:
            return str((row.get("options") or {}).get(str(row.get("answer_idx", "")).strip(), "")) or None

        rep = VerificationReport(source="medqa", n_items=len(items))
        for item in items:
            rep.gold_class_counts[normalize_label(item.gold)] += 1
            row = by_question.get(normalize_label(item.state))
            if row is None:
                rep.n_unverifiable += 1
                continue
            want_key = medqa_key(row, item)
            if want_key is None:
                rep.n_unverifiable += 1
                continue
            rep.raw_class_counts[normalize_label(want_key)] += 1
            rep.n_checked += 1
            if normalize_label(item.gold) != normalize_label(want_key):
                rep.add_mismatch(item.item_id, str(row.get("meta_info")), want_key, item.gold)
            elif normalize_label(item.options[item.gold_index].label) != normalize_label(
                medqa_text(row, item) or ""
            ):
                rep.add_mismatch(item.item_id, str(row.get("meta_info")),
                                 f"text:{medqa_text(row, item)}",
                                 item.options[item.gold_index].label)
        reports["medqa"] = rep.to_dict()
        raw_distributions["medqa"] = dict(
            sorted(Counter(str(r["answer_idx"]) for r in ds).items())
        )

    # ---- MedMCQA ---------------------------------------------------------
    if want("medmcqa"):
        val = load_dataset("openlifescienceai/medmcqa", split="validation", revision=REVISIONS["medmcqa"])
        raw = _raw_index(val, "id")

        def medmcqa_key(row: dict[str, Any], item: Item) -> str | None:
            try:
                index = int(row["cop"])
            except (KeyError, TypeError, ValueError):
                return None
            # items store the option *key* (A/B/C/D), not the raw index
            return item.option_keys[index] if 0 <= index < item.n_options else None

        def medmcqa_text(row: dict[str, Any], _item: Item) -> str | None:
            options = [row.get("opa"), row.get("opb"), row.get("opc"), row.get("opd")]
            index = int(row["cop"])
            return str(options[index]) if options[index] else None

        rep = check_items_against_raw(
            "medmcqa", [i for i in load_items("medmcqa") if str(i.split) == "test"], raw,
            medmcqa_key, expected_text=medmcqa_text,
        )
        reports["medmcqa"] = rep.to_dict()
        raw_distributions["medmcqa"] = {
            "cop_counts_validation": dict(sorted(Counter(str(r["cop"]) for r in val).items())),
            "n_validation_rows": len(val),
        }

    # ---- PubMedQA --------------------------------------------------------
    if want("pubmedqa"):
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", revision=REVISIONS["pubmedqa"])
        raw = _raw_index(ds, "pubid")

        def pubmedqa_key(row: dict[str, Any], item: Item) -> str | None:
            decision = str(row.get("final_decision", "")).strip().lower()
            for key, option in zip(item.option_keys, item.options, strict=True):
                if normalize_label(option.label) == decision:
                    return key
            return None

        rep = check_items_against_raw(
            "pubmedqa", load_items("pubmedqa"), raw, pubmedqa_key,
            expected_text=lambda row, _item: str(row.get("final_decision", "")).strip().lower() or None,
        )
        reports["pubmedqa"] = rep.to_dict()
        raw_distributions["pubmedqa"] = dict(
            sorted(Counter(str(r["final_decision"]) for r in ds).items())
        )

    # ---- MMLU ------------------------------------------------------------
    if want("mmlu_medical"):
        subjects = ["anatomy", "clinical_knowledge", "college_medicine", "college_biology",
                    "medical_genetics", "professional_medicine"]
        rows = []
        for subject in subjects:
            for split in ("test", "validation"):
                ds = load_dataset("cais/mmlu", subject, split=split, revision=REVISIONS["mmlu_medical"])
                for row in ds:
                    rows.append({**row, "_subject": subject, "_split": split})
        from meddecide.utils.hashing import stable_hash

        by_key = {}
        for row in rows:
            key = stable_hash(
                {"subject": row["_subject"], "q": str(row["question"]).strip(),
                 "choices": [str(c).strip() for c in row["choices"]]},
                length=20,
            )
            by_key[key] = row

        def mmlu_key(row: dict[str, Any], item: Item) -> str | None:
            try:
                index = int(row["answer"])
            except (KeyError, TypeError, ValueError):
                return None
            return item.option_keys[index] if 0 <= index < item.n_options else None

        def mmlu_text(row: dict[str, Any], _item: Item) -> str | None:
            choices = row.get("choices") or []
            index = int(row["answer"])
            return str(choices[index]) if 0 <= index < len(choices) else None

        rep = check_items_against_raw(
            "mmlu_medical", load_items("mmlu_medical"), by_key, mmlu_key, expected_text=mmlu_text
        )
        reports["mmlu_medical"] = rep.to_dict()
        raw_distributions["mmlu_medical"] = dict(sorted(Counter(str(r["answer"]) for r in rows).items()))

    # ---- MedQuAD ---------------------------------------------------------
    if want("medquad"):
        ds = load_dataset("lavita/MedQuAD", split="train", revision=REVISIONS["medquad"])
        by_question = {normalize_label(r["question"]): r for r in ds}
        items = load_items("medquad")
        rep = VerificationReport(source="medquad", n_items=len(items))
        for item in items:
            rep.gold_class_counts[normalize_label(item.gold)] += 1
            question = normalize_label(str(item.state).replace("Consumer health question:", ""))
            row = by_question.get(question)
            if row is None:
                rep.n_unverifiable += 1
                continue
            want_label = normalize_label(row.get("question_type", ""))
            position = next(
                (i for i, o in enumerate(item.options) if normalize_label(o.label) == want_label), None
            )
            if position is None:
                rep.n_unverifiable += 1
                continue
            want_key = item.option_keys[position]
            rep.raw_class_counts[normalize_label(want_key)] += 1
            rep.n_checked += 1
            if normalize_label(item.gold) != normalize_label(want_key):
                rep.add_mismatch(item.item_id, str(row.get("question_id")), want_key, item.gold)
        rep.notes.append(
            "MedQuAD gold is the option whose *label* equals the record's question_type; the "
            "loader shuffles option order with a seeded permutation, so verification matches on "
            "label and returns the corresponding key"
        )
        reports["medquad"] = rep.to_dict()
        raw_distributions["medquad"] = dict(
            sorted(Counter(str(r["question_type"]).strip().lower() for r in ds).items())
        )

    relevance_specs = [
        ("trec_covid", "BeIR/trec-covid", "BeIR/trec-covid-qrels", ["test"]),
        ("nfcorpus", "BeIR/nfcorpus", "BeIR/nfcorpus-qrels", ["train", "test"]),
        ("scifact", "BeIR/scifact", "BeIR/scifact-qrels", ["test"]),
    ]
    for source, _corpus_id, qrels_id, qrels_splits in relevance_specs:
        if not want(source):
            continue
        # qrels live in their own datasets with their own revision history, so the corpus
        # revision cannot be applied to them; resolve and record the revision actually used.
        qrels_revision = HfApi().dataset_info(qrels_id).sha
        REVISIONS[qrels_id] = qrels_revision
        # Each qrels *split* defines its own judgements, and the builder used different qrels
        # files for dev (train qrels) and test. A judgement is therefore looked up in the qrels
        # split that produced the item's own split; pooling the files created 432 + 150 false
        # mismatches on the first run of this checker.
        judgements_by_split = {
            split: _qrels_index(list(load_dataset(qrels_id, split=split, revision=qrels_revision)))
            for split in qrels_splits
        }
        if len(judgements_by_split) == 1:
            only = next(iter(judgements_by_split.values()))
            judgements_by_split = {"train": only, "test": only}
        items = load_items(source)
        rep = VerificationReport(source=source, n_items=len(items))
        for item in items:
            rep.gold_class_counts[normalize_label(item.gold)] += 1
            query = str(item.meta.get("query_id"))
            doc = str(item.meta.get("corpus_id"))
            judgements = judgements_by_split.get(str(item.split))
            grade = (judgements or {}).get(query, {}).get(doc)
            if grade is None:
                rep.n_unverifiable += 1
                continue
            # the loader stores the *key*: for score items the level key is the grade + 1
            # (level 1 = grade 0), for noul items only the three graded levels exist
            # score options are keyed "1".."N" lowest-first and level k corresponds to qrels
            # grade k-1, so the expected key is the key at *position* grade; noul options are
            # keyed yes/no and a grade > 0 means "yes".
            if item.qtype == "score":
                want_key = (
                    item.option_keys[grade] if 0 <= grade < item.n_options else None
                )
            else:
                want_key = "yes" if grade > 0 else "no"
            if want_key is None:
                rep.n_unverifiable += 1
                continue
            rep.raw_class_counts[normalize_label(want_key)] += 1
            rep.n_checked += 1
            if normalize_label(item.gold) != normalize_label(want_key):
                rep.add_mismatch(item.item_id, f"{query}:{doc}", want_key, item.gold)
        rep.notes.append(
            "relevance gold is qrels membership: score items use level key = grade + 1 "
            "(level 1 = grade 0), noul items use yes for grade > 0"
        )
        reports[source] = rep.to_dict()
        grade_counts: Counter[str] = Counter()
        for split in qrels_splits:
            for row in load_dataset(qrels_id, split=split, revision=qrels_revision):
                grade_counts[str(row["score"])] += 1
        raw_distributions[source] = {
            "qrels_splits": qrels_splits,
            "grade_counts": dict(sorted(grade_counts.items())),
        }

    payload: dict[str, Any] = {
        "generated_at_utc": utcnow(),
        "tier1_dir": str(args.tier1),
        "dataset_revisions": REVISIONS,
        "sources": reports,
        "raw_distributions": raw_distributions,
        "summary": {
            "n_sources": len(reports),
            "total_items": sum(r["n_items"] for r in reports.values()),
            "total_checked": sum(r["n_checked"] for r in reports.values()),
            "total_mismatched": sum(r["n_mismatched"] for r in reports.values()),
            "total_unverifiable": sum(r["n_unverifiable"] for r in reports.values()),
            "all_ok": all(r["ok"] for r in reports.values()),
        },
    }
    if args.prove_cop:
        payload["cop_semantics_proof"] = prove_cop_semantics()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    prov = Provenance(
        run_name="F1_verify_gold",
        command=" ".join([sys.executable, *sys.argv]),
        config={"tier1": str(args.tier1)},
    )
    prov.datasets = [{"id": k, "revision": v} for k, v in REVISIONS.items()]
    prov.finish().write(args.out.with_name("gold_verification_provenance.json"))

    print(json.dumps(payload["summary"], indent=2))
    for name, rep in sorted(reports.items()):
        flag = "OK " if rep["ok"] else "MISMATCH"
        print(f"  {flag} {name}: items={rep['n_items']} checked={rep['n_checked']} "
              f"mismatched={rep['n_mismatched']} unverifiable={rep['n_unverifiable']}")
    return 0 if payload["summary"]["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
