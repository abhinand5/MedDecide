"""RAGTruth (MIT; human span annotations of RAG responses) as osler_v3 items (D42; ADVISORY X1).

Source: the original release (github.com/ParticleMedia/RAGTruth, MIT, pinned by commit), its dataset/response.jsonl and
dataset/source_info.jsonl. Labels come only from the human span annotations in ``labels``: a labelled span is a hallucination
(RAGTruth's own convention: a response with any span is hallucinated). The ``meta`` field of a span holds annotator notes and
is never read into an item. ``implicit_true`` and ``due_to_null`` flags are counted, not used.

Splits: the official split is per source context (train or test, never both), and is kept. Ten per cent of the official
train contexts, by stable hash of the source id, form dev. One context never crosses splits.

Templates (ADVISORY X1), all binary or choice, labels computed here from the spans and nothing else:
- ``ragtruth_response_grounded_v1`` (noul): does the response contain no unsupported or contradicted information?
- ``ragtruth_sentence_supported_v1`` (noul): does this sentence of the response contain no such information? At most 20
  sentences per response (a stable sample).
- ``ragtruth_halluc_type_v1`` (choice): which kind of unsupported information does the response contain, if any? Responses
  whose spans have two or more types are excluded (their gold is not one type).

Each binary template is balanced 1:1 and each type class is balanced to the smallest class, by a stable-hash order, in each
split. Every drop and every balancing step is counted (no silent drops).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from meddecide.mix.rows import make_row
from meddecide.mix.sentences import sentence_spans
from meddecide.utils.hashing import stable_hash

SOURCE = "ragtruth"
SKILL = "hallucination_detection"
LICENCE = "MIT"
COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
REPO_URL = f"https://github.com/ParticleMedia/RAGTruth/tree/{COMMIT}"
TIER = "osler_v3_source"
DEV_MODULUS = 10  # 10 % of the official train contexts form dev
SENTENCE_CAP = 20  # ADVISORY X1: at most 20 sentence items per response

GROUNDED = "ragtruth_response_grounded_v1"
SENTENCE = "ragtruth_sentence_supported_v1"
HALLUC_TYPE = "ragtruth_halluc_type_v1"

# RAGTruth's four annotated types, in a fixed order (letters A-D), and the option for a response with none (E)
TYPE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Evident Baseless Info", "Evident baseless information"),
    ("Subtle Baseless Info", "Subtle baseless information"),
    ("Evident Conflict", "Evident conflict with the source"),
    ("Subtle Conflict", "Subtle conflict with the source"),
)
NONE_LABEL = "No unsupported information"
TYPE_KEYS = {name: letter for (name, _), letter in zip(TYPE_OPTIONS, "ABCD", strict=True)}
NONE_KEY = "E"


def context_split(source_id: str, official: str) -> str:
    """``test`` for the official test contexts; ``dev`` or ``train`` for the official train contexts (stable hash)."""
    if official == "test":
        return "test"
    if official != "train":
        raise ValueError(f"unknown official split {official!r}")
    return "dev" if int(stable_hash({"ragtruth_context": str(source_id)}, length=8), 16) % DEV_MODULUS == 0 else "train"



def overlaps(span: tuple[int, int], label: Mapping[str, Any]) -> bool:
    """True when a character span of the response shares at least one character with an annotated span."""
    return span[0] < int(label["end"]) and int(label["start"]) < span[1]



def label_types(labels: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(label["label_type"]) for label in labels}



def response_type_gold(labels: Sequence[Mapping[str, Any]]) -> tuple[str | None, str]:
    """The type key of a response (``E`` for no span, else the letter of its one annotated type) and the reason when there is
    none: ``unknown_type`` for a type outside RAGTruth's four, ``mixed_types`` for two or more types."""
    if not labels:
        return NONE_KEY, "kept"
    types = label_types(labels)
    if types - set(TYPE_KEYS):
        return None, "unknown_type"
    if len(types) > 1:
        return None, "mixed_types"
    (name,) = types
    return TYPE_KEYS[name], "kept"



def stable_order(keys: Iterable[str], *, salt: str) -> list[str]:
    """The keys in a stable pseudo-random order (by hash), for deterministic sampling and balancing."""
    return sorted(keys, key=lambda k: stable_hash({"salt": salt, "key": k}, length=16))



def balance(rows: Sequence[dict[str, Any]], *, group_of: Mapping[str, str], salt: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep the same number of rows for every group (the smallest group's size), chosen by a stable-hash order.

    ``group_of`` maps each row's item id to its gold group. Returns the kept rows (in input order) and the kept count per group.
    """
    by_group: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_group[group_of[row["item_id"]]].append(row["item_id"])
    if not by_group:
        return [], {}
    size = min(len(ids) for ids in by_group.values())
    keep: set[str] = set()
    kept_per_group: dict[str, int] = {}
    for group, ids in sorted(by_group.items()):
        chosen = stable_order(ids, salt=f"{salt}:{group}")[:size]
        keep.update(chosen)
        kept_per_group[group] = len(chosen)
    return [row for row in rows if row["item_id"] in keep], kept_per_group



def build(responses: Sequence[Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Items by template and split, with every count. ``sources`` maps a source id to its source_info row."""
    counts: Counter[str] = Counter()
    raw: dict[str, list[dict[str, Any]]] = defaultdict(list)  # template -> rows before balancing
    groups: dict[str, dict[str, str]] = defaultdict(dict)  # template -> item id -> balancing group
    for response in responses:
        counts["responses"] += 1
        if str(response["quality"]) != "good":
            counts[f"dropped_quality_{response['quality']}"] += 1
            continue
        source = sources[str(response["source_id"])]
        split = context_split(str(response["source_id"]), str(response["split"]))
        counts[f"responses_{split}"] += 1
        labels = list(response["labels"])
        text = str(response["response"])
        meta = {"task_type": str(source["task_type"]), "dataset": str(source["source"]), "model": str(response["model"]),
                "official_split": str(response["split"]), "spans": len(labels),
                "implicit_true_spans": sum(bool(lab["implicit_true"]) for lab in labels),
                "due_to_null_spans": sum(bool(lab["due_to_null"]) for lab in labels)}
        state = f"Instruction and source given to the writer:\n{source['prompt']}\n\nResponse:\n{text}"
        rid = f"{source['source_id']}:{response['id']}"
        # 1. response level: grounded?
        grounded_gold = "no" if labels else "yes"
        row = _row(GROUNDED, rid, split, "noul", state,
                   "Does the response contain no unsupported or contradicted information, relative to the source?",
                   [("yes", "Yes"), ("no", "No")], grounded_gold, meta)
        raw[GROUNDED].append(row)
        groups[GROUNDED][row["item_id"]] = grounded_gold
        # 2. sentence level: a sentence overlapping no span is supported
        spans = sentence_spans(text)
        if not spans:
            counts["responses_without_sentences"] += 1
        chosen = stable_order(range(len(spans)), salt=f"sent:{rid}")[:SENTENCE_CAP]
        counts["sentences_seen"] += len(spans)
        counts["sentences_dropped_by_cap"] += len(spans) - len(chosen)
        for index in sorted(chosen):
            start, end = spans[index]
            hit = any(overlaps((start, end), label) for label in labels)
            gold = "no" if hit else "yes"
            sentence = text[start:end]
            row = _row(SENTENCE, f"{rid}:s{index}", split, "noul", state,
                       "Does this sentence of the response contain no unsupported or contradicted information? "
                       f"Sentence: {sentence}",
                       [("yes", "Yes"), ("no", "No")], gold, {**meta, "sentence_index": index})
            raw[SENTENCE].append(row)
            groups[SENTENCE][row["item_id"]] = gold
        # 3. response type: the single annotated type, or none
        type_gold, reason = response_type_gold(labels)
        if type_gold is None:
            counts[f"dropped_{reason}_for_type_template"] += 1
            continue
        options = [(TYPE_KEYS[name], label) for name, label in TYPE_OPTIONS] + [(NONE_KEY, NONE_LABEL)]
        row = _row(HALLUC_TYPE, rid, split, "choice", state,
                   "Which kind of unsupported or contradicted information does the response contain, if any?",
                   options, type_gold, {**meta, "types": sorted(label_types(labels))})
        raw[HALLUC_TYPE].append(row)
        groups[HALLUC_TYPE][row["item_id"]] = type_gold
    out: dict[str, list[dict[str, Any]]] = {}
    report: dict[str, Any] = {"counts": dict(counts), "templates": {}}
    for template in (GROUNDED, SENTENCE, HALLUC_TYPE):
        rows_by_split: dict[str, list[dict[str, Any]]] = {}
        balance_report: dict[str, Any] = {}
        for split in ("train", "dev", "test"):
            part = [r for r in raw[template] if r["split"] == split]
            before = Counter(groups[template][r["item_id"]] for r in part)
            kept, per_group = balance(part, group_of=groups[template], salt=f"{template}:{split}")
            rows_by_split[split] = kept
            balance_report[split] = {"before": dict(sorted(before.items())), "after": dict(sorted(per_group.items())),
                                     "items": len(kept)}
        out[template] = [r for split in ("train", "dev", "test") for r in rows_by_split[split]]
        report["templates"][template] = balance_report
    return out, report



def _row(template: str, record_id: str, split: str, qtype: str, state: str, question: str,
         options: list[tuple[str, str]], gold: str, meta: Mapping[str, Any]) -> dict[str, Any]:
    row = make_row(source=SOURCE, template_id=template, skill=SKILL, qtype=qtype, state=state, question=question,
                   options=options, gold=gold, record_id=record_id, record_url=REPO_URL, licence=LICENCE,
                   record_date="", split=split, tier=TIER, meta=dict(meta))
    return row
