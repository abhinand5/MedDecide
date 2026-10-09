"""Training rows of mix v2 (osler_v0 O4) as validated items for the trainer (O6-O8).

The benchmark's ``Item`` is strict: a tier from the benchmark enum, an http(s) source URL, and a dated record. Mix v2 rows come
from sources that do not all carry these: generated items have no record, and the replay and catalog sources are outside the
benchmark tiers. The training path reads the state, the question, the options and the gold, and never the tier, the source
URL or the record date (the trainer's loss and the prompt do not use them). So those fields are filled with the placeholders
below. The placeholders are a training-only convention; the benchmark code never reads a training row.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from meddecide.bench.schema import Item, Option, Tier, make_item
from meddecide.utils.hashing import stable_hash

PLACEHOLDER_URL = "https://example.org/meddecide-training-only"
PLACEHOLDER_DATE = date(1900, 1, 1)
TRAINING_TIER = Tier.ESTABLISHED
UNRECORDED_LICENCE = "unrecorded (training row)"


def _url(value: Any) -> str:
    text = str(value or "")
    return text if text.startswith(("http://", "https://")) else PLACEHOLDER_URL


def _date(value: Any) -> date:
    text = str(value or "")
    return date.fromisoformat(text) if text else PLACEHOLDER_DATE


def mix_row_to_item(row: Mapping[str, Any]) -> Item:
    """One mix v2 row as a validated training item. Raises ``ValueError`` (pydantic) for any row the schema rejects."""
    tier = Tier.FRESH if row.get("tier") == "fresh" else TRAINING_TIER
    seed = int(stable_hash({"order": row["item_id"]}, length=8), 16)
    return make_item(
        tier=tier,
        source=str(row["source"]),
        source_record_id=str(row.get("source_record_id") or "unrecorded"),
        source_url=_url(row.get("source_url")),
        source_license=str(row.get("source_license") or UNRECORDED_LICENCE),
        record_date=_date(row.get("record_date")),
        split=str(row["split"]),
        template_id=str(row["template_id"]),
        skill=str(row.get("skill") or "training"),
        qtype=str(row["qtype"]),
        state=str(row["state"]),
        question=str(row["question"]),
        options=[Option(key=str(o["key"]), label=str(o["label"])) for o in row["options"]],
        gold=str(row["gold"]),
        option_order_seed=seed,
    )
