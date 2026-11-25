"""Y4 step 2 (ADVISORY Y4 §2): CG v2 = osler_v3's CG partition plus general-clinical dev items from non-panel sources with
verified licences. Fixed before any Stage A run; the external panel never enters (13-gram check against all seven panel sets).

Sources (licence and origin checked in the Hub metadata on 2026-10-21):
- cg_v3 (osler_v3, 4,865 items): kept unchanged.
- HeadQA dev, English (dvilares/head_qa, MIT; HEAD_EN.json restricted to the dev qids of dev_HEAD.json): 1,366 items.
- PubMedQA labelled (qiaojin/PubMedQA, MIT; pqa_labeled, 1,000 items): yes, no or maybe; pubids already in cg_v3 removed.
- MedNLI dev (PhysioNet credentialed; local): the dev pairs NOT used by fast dev v4's mednli partition (disjoint by design).
- DDXPlus validation (aai530-group6/ddxplus, CC BY 4.0): a stable sample of 1,000 cases rendered from the dataset's own
  evidence dictionary; 49 conditions as options.

Output: data/bench/osler_v4/cg_v2.jsonl (gitignored; item text) and outputs/osler_v4/Y4/cg_v2_manifest.json (counts only).
Fails (exit code 1) if fewer than 6,000 items survive: the partition is then not fixed (ADVISORY §2).
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from meddecide.panel import ngram
from meddecide.panel import training_overlap as to
from meddecide.panel.overlap import text_key
from meddecide.utils.hashing import stable_hash
from meddecide.utils.io import write_json

CG_V3 = Path("data/bench/osler_v3/cg_v3.jsonl")
PANEL = Path("data/bench/v0.3_ext/panel.jsonl")
FASTDEV_V4 = Path("data/bench/osler_v4/fastdev_v4.jsonl")
HEADQA_DIR = Path("data/bench/osler_v4/raw/headqa_extract")      # extracted JSON members only (untrusted data)
MEDNLI_DEV = Path("data/physionet/mednli/1.0.0/mli_dev_v1.jsonl")
OUT = Path("data/bench/osler_v4/cg_v2.jsonl")
MANIFEST = Path("outputs/osler_v4/Y4/cg_v2_manifest.json")
MIN_ITEMS = 6000
# training input = every .jsonl under data/train/ except the dev.jsonl files (osler_v0/dev.jsonl is osler_v0's dev split, a
# dev input, not training; the first version of this check included it and reported dev-against-dev overlap as training)
TRAINING_GLOB = "data/train/**/*.jsonl"
DDX_SAMPLE = 1000
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _item(source: str, record_id: str, state: str, question: str, options: list[dict[str, str]], gold: str,
          template: str) -> dict[str, Any]:
    keys = [o["key"] for o in options]
    if gold not in keys or len(set(keys)) != len(keys) or not state.strip():
        raise ValueError(f"malformed CG v2 item {source}/{record_id}")
    return {"benchmark": "cg_v2", "cg_partition": source, "gold": gold, "held_out": False,
            "item_id": stable_hash({"source": source, "record": record_id}, length=16), "options": options,
            "qtype": "choice", "question": question, "set_name": source, "source": source, "split": "dev",
            "state": state, "template_id": template, "source_record_id": record_id}


def headqa_items(base: Path) -> tuple[list[dict[str, Any]], Counter]:
    counts: Counter[str] = Counter()
    dev = json.loads((base / "dev_HEAD.json").read_text(encoding="utf-8"))
    en = json.loads((base / "HEAD_EN.json").read_text(encoding="utf-8"))
    dev_keys = {(exam, q["qid"]) for exam, e in dev["exams"].items() for q in e["data"]}
    out = []
    for exam, e in en["exams"].items():
        for q in e["data"]:
            if (exam, q["qid"]) not in dev_keys:
                continue
            answers = q["answers"]
            right = [a for a in answers if int(a["aid"]) == int(q["ra"])]
            if len(right) != 1:
                counts["headqa: no unique right answer"] += 1
                continue
            options = [{"key": LETTERS[i], "label": str(a["atext"])} for i, a in enumerate(answers)]
            gold = LETTERS[[int(a["aid"]) for a in answers].index(int(q["ra"]))]
            out.append(_item("headqa_dev_en", f"{exam}:{q['qid']}", str(q["qtext"]),
                             "Which of the following answers is correct?", options, gold, "cg_v2_headqa_dev_v1"))
    counts["headqa dev items"] = len(out)
    return out, counts


def pubmedqa_items(path: Path, exclude_questions: set[str]) -> tuple[list[dict[str, Any]], Counter]:
    counts: Counter[str] = Counter()
    df = pd.read_parquet(path)
    options = [{"key": "A", "label": "yes"}, {"key": "B", "label": "no"}, {"key": "C", "label": "maybe"}]
    gold_of = {"yes": "A", "no": "B", "maybe": "C"}
    out = []
    for row in df.to_dict("records"):
        pubid = str(row["pubid"])
        if str(row["question"]) in exclude_questions:
            counts["pubmedqa: question already in cg_v3"] += 1
            continue
        context = row["context"]
        contexts = context.get("contexts", []) if isinstance(context, dict) else list(context)
        state = " ".join(str(c) for c in contexts).strip()
        out.append(_item("pubmedqa_labelled", pubid, state, str(row["question"]), options,
                         gold_of[str(row["final_decision"])], "cg_v2_pubmedqa_labelled_v1"))
    counts["pubmedqa items"] = len(out)
    return out, counts


def mednli_items(excluded_pairs: set[str]) -> tuple[list[dict[str, Any]], Counter]:
    counts: Counter[str] = Counter()
    keys = (("A", "entailment"), ("B", "contradiction"), ("C", "neutral"))
    gold_of = {label: key for key, label in keys}
    out = []
    for line in MEDNLI_DEV.read_text().splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        if str(pair["pairID"]) in excluded_pairs:
            counts["mednli: used by fast dev v4"] += 1
            continue
        label = str(pair.get("gold_label", ""))
        if label not in gold_of:
            counts["mednli: gold_label not in the three classes"] += 1
            continue
        out.append(_item("mednli_dev_rest", str(pair["pairID"]), str(pair["sentence1"]).strip(),
                         f"Hypothesis: {str(pair['sentence2']).strip()} Given the premise, is the hypothesis an entailment, "
                         "a contradiction, or neutral?", [{"key": k, "label": lab} for k, lab in keys], gold_of[label],
                         "cg_v2_mednli_dev_v1"))
    counts["mednli dev rest items"] = len(out)
    return out, counts


def ddxplus_items(base: Path) -> tuple[list[dict[str, Any]], Counter]:
    counts: Counter[str] = Counter()
    conditions = json.loads((base / "release_conditions.json").read_text())
    evidences = json.loads((base / "release_evidences.json").read_text())
    names = [str(c["condition_name"]) for c in conditions.values()]
    options = [{"key": str(i + 1), "label": name} for i, name in enumerate(names)]
    key_of = {name: str(i + 1) for i, name in enumerate(names)}
    rows = []
    with (base / "validate.csv").open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            rows.append((stable_hash({"ddx": index}, length=16), index, row))
    rows.sort(key=lambda r: r[0])
    out = []
    seen_states: set[str] = set()
    for _, index, row in rows:
        if len(out) >= DDX_SAMPLE:
            break
        gold = key_of.get(str(row["PATHOLOGY"]))
        if gold is None:
            counts["ddxplus: pathology not in the condition list"] += 1
            continue
        findings = []
        for code in ast.literal_eval(row["EVIDENCES"]):
            if "_@_" in code:
                name, value = code.split("_@_", 1)
                ev = evidences.get(name)
                if ev is None:
                    counts["ddxplus: evidence code not in dictionary"] += 1
                    continue
                findings.append(f"{ev['question_en']} {ev.get('value_meaning', {}).get(value, {}).get('en', value)}")
            else:
                ev = evidences.get(code)
                if ev is None:
                    counts["ddxplus: evidence code not in dictionary"] += 1
                    continue
                findings.append(f"{ev['question_en']} yes")
        if not findings:
            counts["ddxplus: no findings"] += 1
            continue
        sex = "male" if str(row["SEX"]).upper() == "M" else "female"
        state = f"A synthetic patient, {row['AGE']} years old, {sex}. Reported findings: " + "; ".join(findings) + "."
        if state in seen_states:
            counts["ddxplus: duplicate presentation"] += 1
            continue
        seen_states.add(state)
        out.append(_item("ddxplus_validate", f"row:{index}", state, "Which condition best explains these findings?",
                         options, gold, "cg_v2_ddxplus_validate_v1"))
    counts["ddxplus items"] = len(out)
    return out, counts


def training_overlap(candidates: list[dict[str, Any]]) -> tuple[set[str], Counter]:
    """item ids of CG v2 candidates that match training input by text or by source record (meddecide.panel.training_overlap),
    counted per source and match kind. The dev files are not training input."""
    found = to.match_training(candidates, to.training_files())
    source_of = {c["item_id"]: c.get("source", c.get("cg_partition", "?")) for c in candidates}
    counts: Counter[str] = Counter()
    for item_id, match in found.items():
        kind = "text and record" if match.text and match.record else "text" if match.text else "record"
        counts[f"training match ({kind}): {source_of[item_id]}"] += 1
    return set(found), counts

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    counts: Counter[str] = Counter()
    cg3 = [json.loads(line) for line in CG_V3.read_text().splitlines() if line.strip()]
    counts["cg_v3 items"] = len(cg3)
    cg3_pubmed_questions = {str(r["question"]) for r in cg3 if r.get("source") == "pubmedqa"}
    counts["cg_v3 pubmedqa questions"] = len(cg3_pubmed_questions)
    fastdev = [json.loads(line) for line in FASTDEV_V4.read_text().splitlines() if line.strip()]
    used_pairs = {str(r["source_record_id"]).split("pair:", 1)[1] for r in fastdev
                  if r.get("fastdev_partition") == "mednli" and str(r.get("source_record_id", "")).startswith("pair:")}
    counts["fast dev v4 mednli pairs excluded"] = len(used_pairs)

    headqa, c1 = headqa_items(HEADQA_DIR)
    pubmed_path = Path(hf_hub_download("qiaojin/PubMedQA", "pqa_labeled/train-00000-of-00001.parquet", repo_type="dataset"))
    pubmed, c2 = pubmedqa_items(pubmed_path, cg3_pubmed_questions)
    mednli, c3 = mednli_items(used_pairs)
    ddx_dir = Path(hf_hub_download("aai530-group6/ddxplus", "release_conditions.json", repo_type="dataset")).parent
    ddx, c4 = ddxplus_items(ddx_dir)
    counts.update(c1), counts.update(c2), counts.update(c3), counts.update(c4)

    panel = [json.loads(line) for line in PANEL.read_text().splitlines() if line.strip()]
    panel_index = ngram.build_index(panel)
    counts["panel items indexed"] = len(panel)
    candidates = cg3 + headqa + pubmed + mednli + ddx
    train_matched, train_counts = training_overlap(candidates)
    counts.update(train_counts)
    counts["training text overlap: dropped"] = len(train_matched)
    candidates = [c for c in candidates if c["item_id"] not in train_matched]

    kept: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    near_dup = Counter()
    duplicate = Counter()
    for row in candidates:
        if ngram.hits(row, panel_index) > 0:
            near_dup[row.get("source", row.get("cg_partition", "?"))] += 1
            continue
        key = text_key(row["state"], row["question"], [o["label"] for o in row["options"]])
        if key in seen_keys:
            duplicate[row.get("source", "?")] += 1
            continue
        seen_keys.add(key)
        kept.append(row)
    counts["near-duplicate of the panel (13-gram): dropped"] = sum(near_dup.values())
    counts["duplicate text within CG v2: dropped"] = sum(duplicate.values())
    by_source = Counter(r.get("source") for r in kept)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as handle:
        for row in kept:
            handle.write(json.dumps(row) + "\n")
    manifest = {
        "total_items": len(kept), "minimum_required": MIN_ITEMS, "items_by_source": dict(by_source),
        "near_duplicate_drops_by_source": dict(near_dup), "duplicate_drops_by_source": dict(duplicate),
        "counts": dict(counts), "ddxplus_sample": DDX_SAMPLE, "panel_datasets": 7,
    }
    write_json(MANIFEST, manifest)
    print(json.dumps({k: manifest[k] for k in ("total_items", "items_by_source", "near_duplicate_drops_by_source")}, indent=2))
    if len(kept) < MIN_ITEMS:
        raise SystemExit(f"CG v2 has {len(kept)} items, fewer than {MIN_ITEMS}: not fixed (ADVISORY §2)")


if __name__ == "__main__":
    main()
