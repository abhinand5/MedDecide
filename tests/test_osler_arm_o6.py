"""O6-O8 recipe checks (no GPU): the budget arithmetic, the dev subset, the tier-1 selector, and that the three arm configs
match the ADVISORY recipe and differ only in readout, the non-causal flag and the output directory."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from meddecide.bench.schema import Option, QuestionType, Tier, make_item
from meddecide.train.config import load_config
from meddecide.train.osler_arm import (
    example_budget,
    step_count,
    stratified_subset,
    tier1_items,
)

REPO = Path(__file__).resolve().parents[1]


def _item(seed: int, source: str, template: str):
    return make_item(
        tier=Tier.FRESH, source=source, source_record_id=f"r{seed}", source_url="https://example.org/x",
        source_license="test", record_date=date(2026, 1, 1), split="dev", template_id=template, skill="test",
        qtype=QuestionType.CHOICE, state=f"Stem {seed}.", question="Which?",
        options=[Option(key="A", label="one"), Option(key="B", label="two")], gold="A", option_order_seed=seed)


def test_budget_is_one_pass_or_the_cap_whichever_is_smaller() -> None:
    assert example_budget(573_611) == 200_000
    assert example_budget(150_000) == 150_000
    with pytest.raises(ValueError):
        example_budget(0)


def test_step_count_divides_the_budget_by_the_batch_size_exactly() -> None:
    assert step_count(200_000, 8) == 25_000
    with pytest.raises(ValueError):
        step_count(200_001, 8)


def test_stratified_subset_has_the_requested_size_is_deterministic_and_keeps_small_groups_whole() -> None:
    items = [_item(i, "pubmed", "big") for i in range(1000)] + [_item(1000 + i, "medqa", "small") for i in range(10)] \
        + [_item(2000 + i, "openfda", "tiny") for i in range(5)]
    first = stratified_subset(items, key="template_id", size=100, seed=0)
    again = stratified_subset(items, key="template_id", size=100, seed=0)
    assert len(first) == 100
    assert [i.item_id for i in first] == [i.item_id for i in again]
    assert sum(1 for i in first if i.template_id == "small") == 10  # a small group is kept whole
    assert sum(1 for i in first if i.template_id == "tiny") == 5
    assert sum(1 for i in first if i.template_id == "big") == 85
    with pytest.raises(ValueError):
        stratified_subset(items[:10], key="template_id", size=100, seed=0)


def test_tier1_selector_takes_only_medqa_and_medmcqa() -> None:
    items = [_item(1, "medqa", "a"), _item(2, "medmcqa", "b"), _item(3, "pubmed", "c")]
    assert [i.source for i in tier1_items(items)] == ["medqa", "medmcqa"]


def test_the_three_arm_configs_match_the_adr_recipe_and_differ_only_where_the_design_says() -> None:
    arms = {arm: load_config(REPO / f"configs/osler_v0/arm_{arm}.yaml") for arm in "LPN"}
    assert {a: c.readout for a, c in arms.items()} == {"L": "option_code", "P": "pointer", "N": "option_code"}
    assert {a: c.bidirectional_full_attention for a, c in arms.items()} == {"L": False, "P": False, "N": True}
    for cfg in arms.values():
        assert (cfg.lora.r, cfg.lora.alpha) == (32, 32)
        assert (cfg.lr, cfg.lora_lr) == (1e-3, 1e-4)
        assert (cfg.lambda_brier, cfg.batch_size, cfg.max_prompt_tokens) == (1.0, 8, 16384)
        assert (cfg.lr_schedule, cfg.warmup_fraction, cfg.lr_floor) == ("cosine", 0.03, 0.1)
        # operator decision 2026-10-10 (STATE Q19/Q20): eval every 2,500 steps; select on the template macro
        assert (cfg.eval_every, cfg.eval_items, cfg.seed, cfg.shuffle_options) == (2500, 6000, 0, True)
        assert cfg.selection_metric == "template_macro_accuracy"
        assert (cfg.tripwire_grad_norm, cfg.tripwire_macro_floor, cfg.tripwire_consecutive_evals) == (5000.0, 0.5, 2)
        assert cfg.train_path == "data/train/osler_v0/train.jsonl" and cfg.dev_path == "data/train/osler_v0/dev.jsonl"
    common = {k: v for k, v in arms["L"].to_dict().items() if k not in {"output_dir", "readout",
                                                                         "bidirectional_full_attention"}}
    for arm in "PN":
        other = {k: v for k, v in arms[arm].to_dict().items() if k not in {"output_dir", "readout",
                                                                           "bidirectional_full_attention"}}
        assert other == common, arm
    assert len({c.output_dir for c in arms.values()}) == 3
