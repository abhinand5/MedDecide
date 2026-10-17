"""EHR-Rel as a score item (osler_v2 ADVISORY W2 step 2; template ``ehr_rel_relatedness_v1``).

EHR-Rel (Schulz et al., COLING 2020; Apache-2.0; github.com/babylonhealth/EHR-Rel, the source of bigbio/ehr_rel): concept
pairs sampled from EHRs, each rated by three (RelB) or five (RelA) clinicians on the paper's 0-3 relatedness scale (Table 1):

- 0 Unrelated: completely unrelated concepts.
- 1 Marginally related: a correlation exists, but an established link might not.
- 2 Related: strongly related medically, one leads to the other, or an established link.
- 3 Extremely related: the concepts always occur together, or one cannot happen without the other.

Gold = the mean rating rounded half-up to the nearest level (ADVISORY W2: the rounded mean; the bins are the scale's own
integers). The split is by concept pair (stable hash), 80 / 10 / 10. Concept names are the dataset's own published SNOMED
preferred terms (osler_v1 operator answer, Q11), used only in local training and evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from meddecide.mix.rows import make_row
from meddecide.utils.hashing import stable_hash

TEMPLATE = "ehr_rel_relatedness_v1"
LICENCE = "apache-2.0 (EHR-Rel repository; bigbio/ehr_rel Hub card)"
RECORD_URL = "https://github.com/babylonhealth/EHR-Rel"
LEVELS = (
    ("0", "Unrelated"),
    ("1", "Marginally related"),
    ("2", "Related"),
    ("3", "Extremely related"),
)
QUESTION = "How related are these two clinical concepts, as they occur together in patient records?"
SPLIT_BUCKETS = {"train": range(0, 8), "dev": range(8, 9), "test": range(9, 10)}  # 80 / 10 / 10 of 10 buckets


def gold_level(mean_rating: float) -> str:
    """Round the mean rating half-up (not banker's rounding) and keep it on the scale."""
    if not 0.0 <= mean_rating <= 3.0:
        raise ValueError(f"mean rating {mean_rating} is outside the 0-3 scale")
    return str(min(3, math.floor(mean_rating + 0.5)))


def pair_key(snomed_a: str, snomed_b: str) -> str:
    """Order-free key for an unordered concept pair."""
    a, b = sorted((snomed_a.strip(), snomed_b.strip()))
    return f"{a}|{b}"


def split_of(key: str) -> str:
    bucket = int(stable_hash({"ehr_rel_pair": key}, length=8), 16) % 10
    for name, buckets in SPLIT_BUCKETS.items():
        if bucket in buckets:
            return name
    raise AssertionError("every bucket belongs to a split")


def ehr_rel_row(rec: Mapping[str, str], subset: str) -> tuple[dict[str, Any] | None, str]:
    """One EHR-Rel line as a score row. ``rec`` holds the TSV columns; ``subset`` is ``RelA`` or ``RelB``."""
    label_a = (rec.get("snomed_label_1") or "").strip()
    label_b = (rec.get("snomed_label_2") or "").strip()
    id_a = (rec.get("snomed_id_1") or "").strip()
    id_b = (rec.get("snomed_id_2") or "").strip()
    if not (label_a and label_b and id_a and id_b):
        return None, "blank_concept"
    if id_a == id_b:
        return None, "same_concept"
    mean_text = (rec.get("mean_rating") or "").strip()
    if not mean_text:
        return None, "no_mean_rating"
    try:
        mean_rating = float(mean_text)
    except ValueError:
        return None, "bad_mean_rating"
    if not 0.0 <= mean_rating <= 3.0:
        return None, "rating_off_scale"
    key = pair_key(id_a, id_b)
    row = make_row(source="ehr_rel", template_id=TEMPLATE, skill="concept_relatedness", qtype="score",
                   state=f"Concept A: {label_a}\nConcept B: {label_b}", question=QUESTION,
                   options=list(LEVELS), gold=gold_level(mean_rating), record_id=f"{subset}|{key}",
                   record_url=RECORD_URL, licence=LICENCE, record_date="", split=split_of(key), tier="catalog",
                   meta={"subset": subset, "mean_rating": round(mean_rating, 4)})
    return row, "kept"
