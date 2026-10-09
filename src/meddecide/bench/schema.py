"""MedDecide-Bench item schema (task T2; documented in ``docs/benchmark/schema.md``).

One item format is spoken by every source, the harness, and every model. Gold always
comes from a structured source field — never from a language model.
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from meddecide.utils.hashing import stable_hash

ITEM_ID_LENGTH = 16
NOUL_YES = "yes"
NOUL_NO = "no"


class QuestionType(StrEnum):
    """The three typed question shapes MedDecide answers."""

    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


class Tier(StrEnum):
    """Benchmark tier: established public test sets, or fresh post-cutoff items."""

    ESTABLISHED = "established"
    FRESH = "fresh"


class Split(StrEnum):
    """Official split. Test splits are evaluation-only — nothing is ever fitted on them."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class Option(BaseModel):
    """One allowed answer.

    ``key`` is the stable identifier used for scoring (``A``, ``B``, … for choice items,
    ``1``…``N`` for score levels, ``yes``/``no`` for noul). ``label`` is what the model
    sees.
    """

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=8)
    label: str = Field(min_length=1)
    description: str | None = None

    @field_validator("key")
    @classmethod
    def _key_is_token_like(cls, value: str) -> str:
        key = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9]{1,8}", key):
            raise ValueError(f"option key must be alphanumeric, got {value!r}")
        return key


class Item(BaseModel):
    """A single benchmark item.

    Invariants enforced here (they are checks, not conveniences):
    ``gold`` is one of the option keys; a ``score`` item has 2-10 levels; a ``choice``
    item has 2-255 options; option keys are unique; ``source_url`` is an http(s) URL.
    """

    model_config = ConfigDict(extra="forbid")

    # identity
    item_id: str = Field(min_length=ITEM_ID_LENGTH, max_length=ITEM_ID_LENGTH)
    tier: Tier
    source: str = Field(min_length=1)
    source_record_id: str = Field(min_length=1)
    source_url: str
    source_license: str = Field(min_length=1)
    record_date: date
    split: Split
    template_id: str = Field(min_length=1)
    skill: str = Field(min_length=1)
    qtype: QuestionType

    # content
    state: str = Field(min_length=1)
    question: str = Field(min_length=1)
    # NOTE: whitespace-only text is rejected below; pydantic's min_length counts
    # characters, so " " would otherwise pass as a non-empty state.
    options: list[Option] = Field(min_length=2, max_length=255)
    gold: str = Field(min_length=1)

    # free-form, but never item text that would leak into a committed file
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_url")
    @classmethod
    def _url_is_http(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"source_url must be http(s), got {value!r}")
        return value

    @field_validator("gold")
    @classmethod
    def _gold_normalised(cls, value: str) -> str:
        return value.strip()

    @field_validator("state", "question", "skill", "template_id", "source")
    @classmethod
    def _not_blank(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @model_validator(mode="after")
    def _check(self) -> Item:
        keys = [opt.key for opt in self.options]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate option keys: {keys}")

        if self.qtype is QuestionType.SCORE:
            if not 2 <= len(self.options) <= 10:
                raise ValueError(f"score item needs 2-10 levels, got {len(self.options)}")
            if keys != [str(i) for i in range(1, len(self.options) + 1)]:
                raise ValueError(f"score level keys must be 1..N in ascending order, got {keys}")
            if self.gold not in keys:
                raise ValueError(f"score gold {self.gold!r} is not one of {keys}")
        elif self.qtype is QuestionType.NOUL:
            if keys != [NOUL_YES, NOUL_NO]:
                raise ValueError(f"noul option keys must be ['yes', 'no'], got {keys}")
            if self.gold not in keys:
                raise ValueError(f"noul gold must be yes|no, got {self.gold!r}")
        elif self.gold not in keys:
            raise ValueError(f"gold {self.gold!r} is not one of the option keys {keys}")

        if len(self.state) + len(self.question) + sum(len(o.label) for o in self.options) == 0:
            raise ValueError("empty item")
        return self

    # ---- helpers -------------------------------------------------------------
    @property
    def option_keys(self) -> list[str]:
        return [opt.key for opt in self.options]

    @property
    def gold_index(self) -> int:
        """0-based position of the gold option."""
        return self.option_keys.index(self.gold)

    @property
    def n_options(self) -> int:
        return len(self.options)

    def content_signature(self) -> str:
        """Signature of the item *content* (state, question, option labels, gold).

        Used by the option-shuffle probe, which permutes option order and asks whether
        the argmax *content* changed.
        """
        return stable_hash(
            {
                "state": self.state,
                "question": self.question,
                "labels": sorted(o.label for o in self.options),
                "gold_label": self.options[self.gold_index].label,
            }
        )


def compute_item_id(
    source: str,
    source_record_id: str,
    template_id: str,
    option_order_seed: int,
    split: str = "none",
) -> str:
    """Deterministic item id: stable hash of source + record + template + seed + split.

    Same logical item → same id on any machine, in any process, in any dict order. The
    **split is part of the identity**: it guarantees id uniqueness across dev/test, and
    makes an id used as a prediction key unambiguous about which split produced it.
    """
    return stable_hash(
        {
            "source": source,
            "source_record_id": str(source_record_id),
            "template_id": template_id,
            "option_order_seed": int(option_order_seed),
            "split": str(split),
        },
        length=ITEM_ID_LENGTH,
    )


def make_item(
    *,
    tier: Tier | str,
    source: str,
    source_record_id: str,
    source_url: str,
    source_license: str,
    record_date: date | str,
    split: Split | str,
    template_id: str,
    skill: str,
    qtype: QuestionType | str,
    state: str,
    question: str,
    options: list[Option | dict[str, Any]],
    gold: str,
    option_order_seed: int,
    meta: dict[str, Any] | None = None,
) -> Item:
    """Build a validated :class:`Item`, deriving ``item_id`` from the identity fields.

    This is the only supported constructor: it guarantees the id matches the content
    that produced it.
    """
    split_value = Split(split).value if not isinstance(split, Split) else split.value
    item_id = compute_item_id(source, source_record_id, template_id, option_order_seed, split_value)
    return Item(
        item_id=item_id,
        tier=tier,
        source=source,
        source_record_id=str(source_record_id),
        source_url=source_url,
        source_license=source_license,
        record_date=record_date,
        split=split,
        template_id=template_id,
        skill=skill,
        qtype=qtype,
        state=state,
        question=question,
        options=options,
        gold=gold,
        meta=meta or {},
    )


def split_by_record_hash(source_record_id: str, *, dev_fraction: float = 0.2, salt: str = "") -> Split:
    """Deterministic dev/test assignment by record-id hash (fresh tier: 20 % dev).

    Assigning by *record* (not by item) keeps every item derived from one record in the
    same split, so templates cannot leak across the dev/test boundary.
    """
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be in (0, 1)")
    digest = stable_hash({"record": str(source_record_id), "salt": salt}, length=8)
    bucket = int(digest, 16) / float(16**8)
    return Split.DEV if bucket < dev_fraction else Split.TEST


AnswerKey = Literal["yes", "no"]
