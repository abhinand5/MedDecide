"""Broad general replay for osler_v3 (ADVISORY X2; D36): human-labelled general multiple-choice and yes/no training items.

Each converter takes the records of one train split and returns (rows in the mix schema, counts of what was kept and what was
dropped and why). The labels come from the source's answer key, or the score of its one correct option; the rationale,
annotation, formula and explanation fields of a source are never read into an item. Train splits only (the caller passes
them). Every drop is counted under a reason.

The licence of each source is recorded in ``SOURCES``, from its Hub card or its README (scripts/osler_v3/x2_fetch_replay.py).
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from meddecide.mix.rows import make_row

TIER = "osler_v3_source"
SKILL = "general_replay"
LETTERS = "ABCDEFGH"
YES_NO = (("A", "Yes"), ("B", "No"))
MATHQA_OPTION = re.compile(r"(?:^|,\s*)([a-e])\s*\)\s*(.*?)(?=\s*,\s*[a-e]\s*\)|\s*$)")
AQUA_OPTION = re.compile(r"^\s*([A-E])\)\s*(.*?)\s*$")
ANSWER_PROMPT = "Which option is the correct answer?"

# source -> (template id, licence as verified, dataset page); a row carries these
SOURCES: dict[str, tuple[str, str, str]] = {
    "strategyqa": ("broad_strategyqa_v1", "MIT", "https://huggingface.co/datasets/ChilleD/StrategyQA"),
    "quartz": ("broad_quartz_v1", "CC BY 4.0", "https://huggingface.co/datasets/allenai/quartz"),
    "social_iqa": ("broad_social_iqa_v1", "CC BY 4.0", "https://leaderboard.allenai.org/socialiqa"),
    "cosmos_qa": ("broad_cosmos_qa_v1", "CC BY 4.0", "https://wilburone.github.io/cosmos/"),
    "math_qa": ("broad_math_qa_v1", "Apache-2.0", "https://math-qa.github.io/"),
    "aqua_rat": ("broad_aqua_rat_v1", "Apache-2.0", "https://huggingface.co/datasets/deepmind/aqua_rat"),
    "bigbench_snarks": ("broad_bigbench_snarks_v1", "Apache-2.0", "https://github.com/google/BIG-bench"),
    "bigbench_ruin_names": ("broad_bigbench_ruin_names_v1", "Apache-2.0", "https://github.com/google/BIG-bench"),
    "bigbench_sports_understanding": ("broad_bigbench_sports_v1", "Apache-2.0", "https://github.com/google/BIG-bench"),
    "bigbench_causal_judgment": ("broad_bigbench_causal_v1", "Apache-2.0", "https://github.com/google/BIG-bench"),
    "bigbench_date_understanding": ("broad_bigbench_date_v1", "Apache-2.0", "https://github.com/google/BIG-bench"),
}


def _row(source: str, record_id: str, state: str, question: str, options: Sequence[tuple[str, str]], gold: str,
         meta: Mapping[str, Any]) -> dict[str, Any]:
    template, licence, page = SOURCES[source]
    return make_row(source=source, template_id=template, skill=SKILL, qtype="choice", state=state, question=question,
                    options=list(options), gold=gold, record_id=record_id, record_url=page, licence=licence,
                    record_date="", split="train", tier=TIER, meta=dict(meta))


def parse_options(text: str) -> list[tuple[str, str]] | None:
    """MathQA's option string, "a ) 5 , b ) 8 , ...", as (letter, label) pairs with upper-case keys; None when it does not parse."""
    found = MATHQA_OPTION.findall(text)
    letters = [letter for letter, _ in found]
    if len(found) < 2 or letters != list("abcde")[: len(found)]:
        return None
    return [(letter.upper(), label.strip()) for letter, label in found]


def strategyqa_rows(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    counts: Counter[str] = Counter()
    rows = []
    for rec in records:
        counts["read"] += 1
        answer = str(rec["answer"]).strip().lower()
        if answer not in ("true", "false"):
            counts["dropped_answer_not_yes_or_no"] += 1
            continue
        term = str(rec.get("term") or "").strip()
        description = str(rec.get("description") or "").strip()
        state = f"Topic: {term}" + (f" ({description})" if description else "") + "."
        gold = "A" if answer == "true" else "B"
        rows.append(_row("strategyqa", str(rec["qid"]), state, str(rec["question"]).strip(), YES_NO, gold, {}))
        counts["kept"] += 1
    return rows, counts


def quartz_rows(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    counts: Counter[str] = Counter()
    rows = []
    for rec in records:
        counts["read"] += 1
        choices = rec["choices"]
        labels = [str(x) for x in choices["label"]]
        texts = [str(x).strip() for x in choices["text"]]
        if str(rec["answerKey"]) not in labels or len(set(labels)) != len(labels) or len(labels) < 2:
            counts["dropped_answer_not_offered"] += 1
            continue
        options = list(zip(labels, texts, strict=True))
        rows.append(_row("quartz", str(rec["id"]), str(rec["para"]).strip(), str(rec["question"]).strip(), options,
                         str(rec["answerKey"]), {}))
        counts["kept"] += 1
    return rows, counts


def problem_key(text: str) -> str:
    """A problem's text with case, spacing and punctuation removed. MathQA's tokenised copy of an AQuA-RAT problem has the same key."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def drop_rows_with_problem_in(rows: Sequence[dict[str, Any]], keys: frozenset[str]) -> tuple[list[dict[str, Any]], int]:
    """The rows whose state's problem key is not in ``keys``, and the number dropped (each one is the same problem as a kept row)."""
    kept = [r for r in rows if problem_key(str(r["state"])) not in keys]
    return kept, len(rows) - len(kept)


def aqua_rat_rows(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    """AQuA-RAT: the problem as the state; ``rationale`` is never read."""
    counts: Counter[str] = Counter()
    rows = []
    for index, rec in enumerate(records):
        counts["read"] += 1
        problem = str(rec["question"]).strip()
        raw = rec["options"]
        parsed = [AQUA_OPTION.match(str(x)) for x in (ast.literal_eval(raw) if isinstance(raw, str) else list(raw))]
        if not parsed or any(m is None for m in parsed):
            counts["dropped_options_unparsed"] += 1
            continue
        options = [(m.group(1), m.group(2)) for m in parsed if m is not None]
        if str(rec["correct"]) not in [k for k, _ in options]:
            counts["dropped_answer_not_offered"] += 1
            continue
        rows.append(_row("aqua_rat", f"aqua_rat:{index}", problem, ANSWER_PROMPT, options, str(rec["correct"]), {}))
        counts["kept"] += 1
    return rows, counts


def math_qa_rows(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    counts: Counter[str] = Counter()
    rows = []
    for index, rec in enumerate(records):
        counts["read"] += 1
        options = parse_options(str(rec["options"]))
        if options is None:
            counts["dropped_options_unparsed"] += 1
            continue
        gold = str(rec["correct"]).strip().upper()
        if gold not in [k for k, _ in options]:
            counts["dropped_answer_not_offered"] += 1
            continue
        rows.append(_row("math_qa", f"math_qa:{index}", str(rec["Problem"]).strip(), ANSWER_PROMPT, options, gold,
                         {"category": str(rec.get("category") or "")}))
        counts["kept"] += 1
    return rows, counts


def social_iqa_rows(records: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> tuple[list[dict[str, Any]], Counter[str]]:
    counts: Counter[str] = Counter()
    rows = []
    if len(labels) != len(records):
        raise ValueError(f"{len(labels)} labels for {len(records)} records")
    for index, (rec, label) in enumerate(zip(records, labels, strict=True)):
        counts["read"] += 1
        if label not in ("1", "2", "3"):
            counts["dropped_label_not_1_to_3"] += 1
            continue
        options = [("A", str(rec["answerA"]).strip()), ("B", str(rec["answerB"]).strip()), ("C", str(rec["answerC"]).strip())]
        rows.append(_row("social_iqa", f"social_iqa:{index}", str(rec["context"]).strip(), str(rec["question"]).strip(),
                         options, LETTERS[int(label) - 1], {}))
        counts["kept"] += 1
    return rows, counts


def cosmos_qa_rows(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    counts: Counter[str] = Counter()
    rows = []
    for rec in records:
        counts["read"] += 1
        label = str(rec["label"]).strip()
        if label not in ("0", "1", "2", "3"):
            counts["dropped_label_not_0_to_3"] += 1
            continue
        options = [(LETTERS[i], str(rec[f"answer{i}"]).strip()) for i in range(4)]
        rows.append(_row("cosmos_qa", f"cosmos_qa:{rec['id']}", str(rec["context"]).strip(), str(rec["question"]).strip(),
                         options, LETTERS[int(label)], {}))
        counts["kept"] += 1
    return rows, counts


def bigbench_rows(task: str, records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    """One BIG-bench subtask (``task`` is its key in SOURCES): the inputs as the state; the gold is the one option scored 1."""
    source = f"bigbench_{task}"
    counts: Counter[str] = Counter()
    rows = []
    for rec in records:
        counts["read"] += 1
        targets = ast.literal_eval(str(rec["multiple_choice_targets"]))
        scores = ast.literal_eval(str(rec["multiple_choice_scores"]))
        if len(targets) != len(scores) or len(targets) < 2 or len(targets) > len(LETTERS):
            counts["dropped_options_unparsed"] += 1
            continue
        winners = [i for i, s in enumerate(scores) if int(s) == 1]
        if len(winners) != 1:
            counts["dropped_not_exactly_one_correct"] += 1
            continue
        options = [(LETTERS[i], str(t).strip()) for i, t in enumerate(targets)]
        rows.append(_row(source, f"{source}:{rec['idx']}", str(rec["inputs"]).strip(),
                         "Which of the offered options is correct?", options, LETTERS[winners[0]], {}))
        counts["kept"] += 1
    return rows, counts


def drop_duplicates(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the first row of each (state, question, options) text, by item id; returns the rows and the number dropped."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for row in rows:
        key = f"{row['state']}\x1f{row['question']}\x1f{[o['label'] for o in row['options']]}"
        if key in seen:
            continue
        seen.add(key)
        kept.append(dict(row))
    return kept, len(rows) - len(kept)
