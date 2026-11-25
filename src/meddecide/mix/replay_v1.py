"""General human-labelled replay for osler_v1 (ADVISORY V3; D20(b) and D27).

Only sources whose licence is permissive are converted (D20: "permissive licence, recorded per source"). Each candidate is
listed with the licence the source states and the reason it was kept or dropped, so the choice can be audited without
reopening the Hub. The converters reuse the mix's multiple-choice path, so a replay row has the same shape as a mix v2
row. Train splits only (ADVISORY V3); test labels are never read.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from meddecide.mix.converters import multiple_choice_row

LETTERS = "ABCD"

# candidate -> (decision, licence as stated, where the licence is stated). The ADVISORY V3 list is the starting point; the
# decision applies D20's permissive rule.
CANDIDATES: dict[str, tuple[str, str, str]] = {
    "hellaswag": ("kept", "MIT", "Hub README links the MIT LICENSE of rowanz/hellaswag"),
    "winogrande": ("kept", "CC-BY (dataset); Apache-2.0 (codebase)",
                   "allenai/winogrande README, Licensing Information: the dataset is licensed under CC-BY"),
    "openbookqa": ("dropped", "unknown (Hub card); Apache-2.0 for the repository LICENSE only",
                   "the dataset's own licence is not stated; the ADVISORY says drop an unclear licence"),
    "ai2_arc": ("dropped", "CC BY-SA 4.0", "Hub card; share-alike, not permissive (D20)"),
    "boolq": ("dropped", "CC BY-SA 3.0", "Hub card; share-alike, not permissive (D20; osler_v0 o4 review)"),
    "snli": ("dropped", "CC BY-SA 4.0", "Hub card; share-alike, not permissive (D20; osler_v0 o4 review)"),
}


def hellaswag_row(rec: Mapping[str, Any], split: str) -> tuple[dict[str, Any] | None, str]:
    """HellaSwag: the context is the state; the four endings are the options; the label is the plausible ending."""
    endings = list(rec["endings"])
    if len(endings) != 4:
        return None, "not_four_endings"
    label = str(rec["label"]).strip()
    if label not in {"0", "1", "2", "3"}:
        return None, "label_missing"
    derived = {
        "id": f"{rec.get('split', split)}-{rec['ind']}",
        "question": str(rec["ctx"]).strip(),
        "choices": {"label": list(LETTERS), "text": endings},
        "answerKey": LETTERS[int(label)],
    }
    return multiple_choice_row(
        derived, split=split, source="hellaswag", template="hellaswag_ending_choice_v1", skill="commonsense_continuation",
        record_url="https://huggingface.co/datasets/Rowan/hellaswag", licence="mit (source README)",
        labels=list(LETTERS), texts=endings, answer=derived["answerKey"],
        meta={"activity_label": rec.get("activity_label"), "source_split": rec.get("split", split)})


def winogrande_row(rec: Mapping[str, Any], split: str) -> tuple[dict[str, Any] | None, str]:
    """WinoGrande: the sentence with its blank is the state; the two fillers are the options; answer "1" or "2"."""
    answer = str(rec.get("answer", "")).strip()
    if answer not in {"1", "2"}:
        return None, "answer_missing"
    sentence = str(rec["sentence"]).strip()
    if "_" not in sentence:
        return None, "no_blank"
    stem = f"{sentence}\nWhich option fills the blank (_)?"
    derived = {"id": str(rec.get("qID", rec.get("id", ""))), "question": stem,
               "choices": {"label": ["A", "B"], "text": [str(rec["option1"]), str(rec["option2"])]},
               "answerKey": "A" if answer == "1" else "B"}
    return multiple_choice_row(
        derived, split=split, source="winogrande", template="winogrande_blank_choice_v1", skill="commonsense_coreference",
        record_url="https://huggingface.co/datasets/allenai/winogrande", licence="cc-by (dataset card)",
        labels=["A", "B"], texts=[str(rec["option1"]), str(rec["option2"])], answer=derived["answerKey"],
        meta={"source_split": split})
