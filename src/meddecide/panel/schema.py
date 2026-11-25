"""Evaluation item for the osler_v0 external panel and robustness pack (O1).

One schema serves three sets: the external panel (public sets converted to typed decisions),
the robustness pack (perturbed copies of v0.2 fresh items and of panel items), and v0.2 items
when they are used as robustness bases. It deliberately drops benchmark-only fields (record
dates, templates' skill tags) that external sets do not carry; ``meta`` keeps any extra detail.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from meddecide.bench.schema import Option, QuestionType
from meddecide.utils.hashing import normalize_text

Benchmark = Literal["v0.2", "ext_panel", "robustness"]


class EvalItem(BaseModel):
    """One typed decision for evaluation. ``gold`` is an option key, never a label."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=16, max_length=16)
    benchmark: Benchmark
    set_name: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    qtype: QuestionType
    state: str = Field(min_length=1)
    question: str = Field(min_length=1)
    options: list[Option] = Field(min_length=2, max_length=255)
    gold: str = Field(min_length=1)
    source_record_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    licence: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    split: str = Field(min_length=1)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("state", "question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not normalize_text(value):
            raise ValueError("text must not be blank")
        return value

    @model_validator(mode="after")
    def _gold_is_an_offered_key(self) -> EvalItem:
        keys = [option.key for option in self.options]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate option keys: {keys}")
        if self.gold not in keys:
            raise ValueError(f"gold {self.gold!r} is not an offered key {keys}")
        return self

    def gold_index(self) -> int:
        return [option.key for option in self.options].index(self.gold)

    def label_of(self, key: str) -> str:
        for option in self.options:
            if option.key == key:
                return option.label
        raise KeyError(key)
