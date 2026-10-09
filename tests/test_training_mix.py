"""Unit tests for the S6 training-mix builder (``scripts/bench/build_training_mix.py``).

Acceptance criteria exercised here, with no network and no downloaded data:

1. the **leakage remover** — a constructed collision of each of the four kinds (record id,
   normalised state hash, record date, held-out template id) is removed and counted, and the
   re-check on the kept rows reports zero;
2. the **held-out-template exclusion** — at build time (through the ClinicalTrials.gov path
   with an injected builder) and when assembling the dev set;
3. the **train/dev disjointness check** — every shared key kind is detected, and the written
   files from a synthetic end-to-end mix share nothing;

plus the supporting rules the mix depends on: the ``split=train`` rewrite with the item-id
recomputation, the per-template balance + cap, the near-miss hub rule shared with
``scripts/bench/build_v0_2.py``, and the mix's arithmetic closing.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from datetime import date, timedelta
from pathlib import Path

from meddecide.bench.schema import Item, Split, compute_item_id, make_item
from meddecide.utils.hashing import normalize_text, stable_hash
from meddecide.utils.io import read_jsonl, write_json, write_jsonl

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "build_training_mix.py"
BUILD_V0_2 = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "build_v0_2.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


btm = _load(SCRIPT, "build_training_mix")
b02 = _load(BUILD_V0_2, "build_v0_2")

CFG = {"seed": 0, "caps": {"fresh_balance_min_class": 200, "fda_class_pool_size": 8}}


def _item(
    *,
    source: str = "clinicaltrials",
    record_id: str = "NCT00000001",
    state: str = "a state",
    question: str = "q?",
    template: str = "ct_randomised_noul_v1",
    split: Split | str = Split.TRAIN,
    record_date: date = date(2025, 1, 1),
    gold: str | None = None,
    option_order_seed: int = 0,
    tier: str = "fresh",
    n_options: int = 2,
) -> Item:
    if "noul" in template:
        options = [{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}]
        gold = gold if gold in ("yes", "no") else "yes"
    else:
        keys = [chr(ord("A") + i) for i in range(n_options)]
        options = [{"key": k, "label": f"option {k}"} for k in keys]
        gold = gold if gold in keys else keys[0]
    return make_item(
        tier=tier, source=source, source_record_id=record_id,
        source_url="https://example.org/record", source_license="public-domain",
        record_date=record_date, split=split, template_id=template,
        skill="trial_design", qtype="noul" if "noul" in template else "choice",
        state=state, question=question, options=options, gold=gold,
        option_order_seed=option_order_seed,
    )


def _pairs(template: str, *, record_prefix: str, state_prefix: str) -> list[Item]:
    """Two items of one template with different golds (so no class is degenerate)."""
    golds = ("yes", "no") if "noul" in template else ("A", "B")
    return [
        _item(template=template, record_id=f"{record_prefix}-{g}", state=f"{state_prefix} {g}", gold=g)
        for g in golds
    ]


# ---------------------------------------------------------------------------
# 1. Leakage: index, remover, verifier
# ---------------------------------------------------------------------------
def test_leakage_index_reads_only_test_and_dev(tmp_path: Path) -> None:
    bench = tmp_path / "v0.2"
    write_jsonl(bench / "tier1" / "medqa.jsonl", [
        _item(source="medqa", record_id="dev-1", state="dev stem", split=Split.DEV),
        _item(source="medqa", record_id="test-1", state="test stem", split=Split.TEST),
        _item(source="medqa", record_id="train-1", state="train stem", split=Split.TRAIN),
    ])
    write_jsonl(bench / "fresh" / "openfda.jsonl", [
        _item(source="openfda", record_id="fda-1", state="fresh test", split=Split.TEST),
    ])
    write_jsonl(bench / "supplementary" / "hle_med.jsonl", [
        _item(source="hle", record_id="hle-1", state="hle test", split=Split.TEST),
    ])
    index = btm.build_leakage_index(bench)

    assert index.n_items == 4  # the tier-1 train item is not a reference
    assert index.by_split == {"dev": 1, "test": 3}
    assert index.by_source == {"hle": 1, "medqa": 2, "openfda": 1}
    assert index.files == {
        "fresh/openfda.jsonl": 1,
        "supplementary/hle_med.jsonl": 1,
        "tier1/medqa.jsonl": 2,
    }
    assert len(index.record_keys) == 4


def test_leakage_remover_counts_each_of_the_four_kinds(tmp_path: Path) -> None:
    bench = tmp_path / "v0.2"
    write_jsonl(bench / "tier1" / "medqa.jsonl", [
        _item(record_id="ref-dev", state="shared dev stem", question="dev question?",
              split=Split.DEV),
        _item(record_id="ref-test", state="shared test stem", question="test question?",
              split=Split.TEST),
        _item(record_id="ref-test2", state="second test stem", question="other question?",
              split=Split.TEST, template="ct_healthy_volunteers_noul_v1"),
    ])
    index = btm.build_leakage_index(bench)

    record_hit = _item(record_id="ref-dev", state="unseen stem A", question="q-a")
    state_hit = _item(record_id="other-1", state="shared dev stem", question="q-b")
    # same state *and* question as a test item: the S5 content-hash variant fires as well
    content_hit = _item(
        record_id="other-2", state="shared test stem", question="test question?"
    )
    date_hit = _item(record_id="other-3", state="dated stem", question="q-c",
                     record_date=date(2026, 3, 1))
    held_out_hit = _item(record_id="other-4", state="held out stem", question="q-d",
                         template="ct_phase_choice_v1")
    clean = _item(record_id="clean-1", state="unseen stem B", question="q-e")

    kept, report = btm.remove_leaks(
        [record_hit, state_hit, content_hit, date_hit, held_out_hit, clean],
        index,
        source_origin={
            record_hit.item_id: "prewindow_structured",
            state_hit.item_id: "prewindow_structured",
            content_hit.item_id: "prewindow_consistency",
            date_hit.item_id: "prewindow_structured",
            held_out_hit.item_id: "prewindow_structured",
            clean.item_id: "tier1_train",
        },
    )

    assert [i.item_id for i in kept] == [clean.item_id]
    assert report["n_input"] == 6 and report["n_kept"] == 1 and report["n_removed_total"] == 5
    # the four kinds, counted separately
    assert report["n_removed_source_record_id"] == 1
    assert report["n_removed_state_hash"] == 2  # state_hit + content_hit
    assert report["n_removed_state_question_hash"] == 1  # content_hit
    assert report["n_removed_date"] == 1
    assert report["n_removed_held_out_template"] == 1
    assert report["by_reason"][btm.R_RECORD_ID] == 1
    assert report["by_source"] == {"clinicaltrials": 5}
    assert report["by_origin_reason"]["prewindow_structured|held_out_template_id"] == 1
    assert report["by_origin_reason"]["prewindow_consistency|normalised_state_hash_in_v0_2_test_or_dev"] == 1
    assert report["examples"][0]["reasons"] == [btm.R_RECORD_ID]

    # the re-check on the kept rows is clean
    assert btm.verify_no_leaks(kept, index)["ok"] is True
    assert btm.verify_no_leaks(kept, index)["n_failing_total"] == 0


def test_one_item_can_fail_several_kinds_and_is_counted_in_each() -> None:
    index = btm.LeakageIndex()
    ref = _item(record_id="ref-1", state="the same stem", question="q?", split=Split.TEST)
    index.record_keys[btm.record_key(ref)] = ref.item_id
    index.state_hashes[btm.state_hash(ref)] = ref.item_id
    index.content_hashes[btm.state_question_hash(ref)] = ref.item_id

    both = _item(
        record_id="ref-1", state="  the   same   stem ", question="q?",
        template="ct_phase_choice_v1", record_date=date(2026, 5, 1),
    )
    kept, report = btm.remove_leaks([both], index)

    assert kept == []
    assert report["n_removed_total"] == 1
    assert report["by_reason"] == {
        btm.R_RECORD_ID: 1,
        btm.R_STATE_HASH: 1,
        btm.R_STATE_QUESTION_HASH: 1,
        btm.R_DATE: 1,
        btm.R_HELD_OUT: 1,
    }


def test_state_hash_normalises_whitespace_and_excludes_the_question() -> None:
    a = _item(state="Query:  a\nPassage: b", question="q1")
    b = _item(state="Query: a Passage: b", question="q2")
    assert btm.state_hash(a) == btm.state_hash(b)
    assert btm.state_question_hash(a) != btm.state_question_hash(b)


def test_window_rule_is_strictly_before_2026_03_01() -> None:
    index = btm.LeakageIndex()
    assert btm.leakage_reasons(_item(record_date=date(2026, 2, 28)), index) == []
    assert btm.leakage_reasons(_item(record_date=date(2026, 3, 1)), index) == [btm.R_DATE]
    assert btm.leakage_reasons(_item(record_date=date(2026, 3, 2)), index) == [btm.R_DATE]


def test_held_out_list_is_the_d14_list() -> None:
    assert set(btm.HELD_OUT_TEMPLATES) == {
        "ct_phase_choice_v1",
        "fda_boxed_warning_noul_v1",
        "pubmed_humans_noul_v1",
        "ct_arm_role_noul_v1",
    }


# ---------------------------------------------------------------------------
# 2. Held-out-template exclusion
# ---------------------------------------------------------------------------
def test_clinicaltrials_build_excludes_held_out_and_rewrites_split(tmp_path: Path) -> None:
    """An injected builder returns items for every CT template, including the held-out one."""
    items = [
        *_pairs("ct_randomised_noul_v1", record_prefix="NCT1", state_prefix="state rand"),
        *_pairs("ct_healthy_volunteers_noul_v1", record_prefix="NCT2", state_prefix="state hv"),
        *_pairs("ct_intervention_type_choice_v1", record_prefix="NCT3", state_prefix="state int"),
        *_pairs("ct_primary_purpose_choice_v1", record_prefix="NCT4", state_prefix="state pp"),
        _item(template="ct_phase_choice_v1", record_id="NCT5", state="state phase"),
    ]
    for row in items:
        row.split = Split.DEV  # every built row starts outside train

    def fake_fetch(**_kwargs):
        return [{"nctId": "NCT1"}]

    def fake_builder(_studies, **_kwargs):
        return list(items), {"some_drop": 2}, ["note"]

    rows, report = btm.build_prewindow_clinicaltrials(
        cfg=CFG, cache_dir=tmp_path / "cache", fetch=fake_fetch, builder=fake_builder,
        slice_months=1,
    )

    assert report["n_items_built"] == 9
    assert report["n_held_out_items_excluded"] == 1
    assert report["held_out_items_excluded_by_template"] == {"ct_phase_choice_v1": 1}
    assert report["n_items_after_held_out_exclusion"] == 8
    assert report["n_split_rewritten_to_train"] == 8
    assert report["build_drops"] == {"some_drop": 2}
    assert report["build_notes"] == ["note"]
    assert {r.template_id for r in rows} == {
        "ct_randomised_noul_v1",
        "ct_healthy_volunteers_noul_v1",
        "ct_intervention_type_choice_v1",
        "ct_primary_purpose_choice_v1",
    }
    assert all(str(r.split) == "train" for r in rows)
    assert all(r.template_id not in btm.HELD_OUT_TEMPLATES for r in rows)
    assert all(btm.leakage_reasons(r, btm.LeakageIndex()) == [] for r in rows)
    # one monthly fetch slice per month of the window, each cached
    assert report["n_slices"] == 38  # Jan 2023 .. Feb 2026 inclusive
    assert report["n_slices_downloaded"] == 38 and report["n_slices_cached"] == 0
    assert len(list((tmp_path / "cache" / "ct_raw").glob("*.jsonl.gz"))) == 38
    # balanced within each template: 2 items, one per gold class
    assert {r.gold for r in rows if r.template_id == "ct_randomised_noul_v1"} == {"yes", "no"}


def test_dev_builder_removes_held_out_and_keeps_only_dev(tmp_path: Path) -> None:
    bench = tmp_path / "v0.2"
    write_jsonl(bench / "tier1" / "medqa.jsonl", [
        _item(source="medqa", record_id="t-dev", state="d1", split=Split.DEV),
        _item(source="medqa", record_id="t-test", state="t1", split=Split.TEST),
        _item(source="medqa", record_id="t-train", state="r1", split=Split.TRAIN),
    ])
    write_jsonl(bench / "fresh" / "clinicaltrials.jsonl", [
        _item(record_id="NCT-dev", state="d2", split=Split.DEV),
        _item(record_id="NCT-held", state="d3", split=Split.DEV, template="ct_phase_choice_v1"),
        _item(record_id="NCT-test", state="t2", split=Split.TEST),
    ])
    dev, report = btm.build_dev(bench)

    assert [r.state for r in dev] == ["d1", "d2"]
    assert all(str(r.split) == "dev" for r in dev)
    assert report["n_input_dev_items"] == 3
    assert report["n_held_out_removed"] == 1
    assert report["held_out_removed_by_template"] == {"ct_phase_choice_v1": 1}
    assert report["by_file"] == {"fresh/clinicaltrials.jsonl": 2, "tier1/medqa.jsonl": 1}


# ---------------------------------------------------------------------------
# 3. Split rewrite, balance/cap, disjointness
# ---------------------------------------------------------------------------
def test_as_train_rewrites_split_and_recomputes_item_id() -> None:
    dev_item = _item(split=Split.DEV, option_order_seed=7)
    moved, changed = btm.as_train(dev_item, option_order_seed=7)

    assert changed is True
    assert str(moved.split) == "train"
    assert moved.item_id == compute_item_id(
        dev_item.source, dev_item.source_record_id, dev_item.template_id, 7, "train"
    )
    assert moved.item_id != dev_item.item_id
    assert btm.item_id_matches(moved, option_order_seed=7) is True
    assert btm.item_id_matches(dev_item, option_order_seed=7) is True
    # the wrong seed does not silently produce a "valid-looking" row
    wrong, _ = btm.as_train(dev_item, option_order_seed=0)
    assert wrong.item_id != moved.item_id
    assert moved.model_dump(exclude={"item_id", "split"}) == dev_item.model_dump(
        exclude={"item_id", "split"}
    )
    # a row already in train is returned unchanged
    train_item = _item(split=Split.TRAIN)
    same, changed_again = btm.as_train(train_item)
    assert same is train_item and changed_again is False


def test_balance_and_cap_balances_then_caps() -> None:
    rows = [
        _item(record_id=f"a{i}", state=f"a {i}", question=f"qa{i}", gold="yes") for i in range(5_000)
    ] + [
        _item(record_id=f"b{i}", state=f"b {i}", question=f"qb{i}", gold="no") for i in range(3_000)
    ]
    kept, report = btm.balance_and_cap(rows, cfg=CFG, salt="t", cap_per_template=6_000)

    tmpl = report["templates"]["ct_randomised_noul_v1"]
    assert tmpl["target_per_class"] == 3_000  # min(cap // 2, smallest class)
    assert len(kept) == 6_000
    assert report["templates_over_cap"] == {}
    assert sum(1 for r in kept if r.gold == "yes") == 3_000
    assert sum(1 for r in kept if r.gold == "no") == 3_000

    # three gold classes, ample supply: the cap binds and every class gets cap // n_classes
    rows3 = [
        _item(record_id=f"{cls}-{i}", state=f"{cls} {i}", question=f"q{cls}{i}",
              template="ct_intervention_type_choice_v1", gold=cls, n_options=3)
        for cls in ("A", "B", "C")
        for i in range(3_000)
    ]
    kept3, report3 = btm.balance_and_cap(rows3, cfg=CFG, salt="t", cap_per_template=6_000)
    assert len(kept3) == 6_000
    assert report3["templates"]["ct_intervention_type_choice_v1"]["target_per_class"] == 2_000
    assert {r.gold for r in kept3} == {"A", "B", "C"}
    assert all(sum(1 for r in kept3 if r.gold == g) == 2_000 for g in ("A", "B", "C"))

    # a single-class template cannot produce a discriminating accuracy: dropped, with reason
    single = [_item(record_id=f"s{i}", state=f"s {i}", question=f"qs{i}", gold="yes") for i in range(10)]
    kept_single, report_single = btm.balance_and_cap(single, cfg=CFG, salt="t", cap_per_template=6_000)
    assert kept_single == []
    assert report_single["templates"]["ct_randomised_noul_v1"]["balance"]["dropped_groups"] == [
        "ct_randomised_noul_v1|train"
    ]


def test_check_disjointness_detects_every_shared_kind() -> None:
    train = [
        _item(record_id="shared-record", state="shared state", question="shared question"),
        _item(record_id="train-only", state="train state", question="train question"),
    ]
    # the same row in both sets: item id, state hash, content hash and record key all collide
    dev = [train[0], _item(record_id="dev-only", state="dev state", question="dev question")]
    report = btm.check_disjointness(train, dev)

    assert report["ok"] is False
    assert report["n_shared_items"] == 1
    assert report["n_shared_by_kind"] == {
        "item_id": 1,
        "state_hash": 1,
        "state_question_hash": 1,
        "record_key": 1,
    }
    assert report["examples_shared_records"] == ["clinicaltrials|shared-record"]

    disjoint = btm.check_disjointness(
        [train[0]], [_item(record_id="other", state="other state", question="other question")]
    )
    assert disjoint["ok"] is True and disjoint["n_shared_items"] == 0
    # a state shared while the record differs is still a collision: the state IS the item
    state_only = btm.check_disjointness(
        [train[0]], [_item(record_id="other", state="shared state", question="other question")]
    )
    assert state_only["ok"] is False and state_only["n_shared_by_kind"]["state_hash"] == 1
    assert state_only["n_shared_by_kind"]["item_id"] == 0


def test_date_slices_cover_the_window_without_gaps_or_overlap() -> None:
    slices = btm._date_slices(date(2023, 1, 1), date(2026, 2, 28), months=3)
    assert len(slices) == 13
    assert slices[0] == (date(2023, 1, 1), date(2023, 3, 31))
    assert slices[-1][1] == date(2026, 2, 28)
    for (_, prev_end), (next_start, _) in itertools.pairwise(slices):
        assert next_start == prev_end + timedelta(days=1)
    months = btm._date_slices(date(2023, 1, 1), date(2026, 2, 28), months=1)
    assert len(months) == 38  # Jan 2023 .. Feb 2026 inclusive
    assert all(start <= end for start, end in months)
    assert months[0][0] == date(2023, 1, 1) and months[-1][1] == date(2026, 2, 28)


# ---------------------------------------------------------------------------
# 4. The near-miss rule shared with build_v0_2
# ---------------------------------------------------------------------------
def _class_index() -> dict:
    return {
        "meta": {"n_classes": 5},
        "classes": {
            "Gold [EPC]": {"moa": ["Mechanism A [MoA]", "Hub Mechanism [MoA]"], "pe": [],
                           "routes": ["ORAL"], "n_labels": 3},
            "NearMoA [EPC]": {"moa": ["Mechanism A [MoA]"], "pe": [], "routes": ["TOPICAL"],
                              "n_labels": 2},
            "HubOnly [EPC]": {"moa": ["Hub Mechanism [MoA]"], "pe": [], "routes": ["ORAL"],
                              "n_labels": 9},
            "RouteOnly [EPC]": {"moa": [], "pe": [], "routes": ["ORAL"], "n_labels": 4},
            "Gold2 [EPC]": {"moa": ["Hub Mechanism [MoA]"], "pe": [], "routes": ["ORAL"],
                            "n_labels": 1},
        },
    }


def test_hub_class_values_matches_build_v0_2() -> None:
    index = _class_index()
    # Mechanism A is on 2 classes, Hub Mechanism on 3: the two implementations must agree
    for max_df in (0, 1, 2, 20):
        assert btm.hub_class_values(index, max_df=max_df) == b02._hub_class_values(
            index, max_df=max_df
        )
    assert btm.hub_class_values(index, max_df=1) == {"Hub Mechanism [MoA]", "Mechanism A [MoA]"}
    assert btm.hub_class_values(index, max_df=2) == {"Hub Mechanism [MoA]"}
    assert btm.hub_class_values(index, max_df=20) == set()


def test_near_miss_keeps_only_mechanistic_non_hub_candidates() -> None:
    index = _class_index()
    near_miss = btm.make_near_miss(index, max_value_df=2)  # Hub Mechanism (df 3) is a hub
    rules = {c["class"]: (c["rule"], c["shared"]) for c in near_miss("Gold [EPC]", set())}
    assert "RouteOnly [EPC]" not in rules  # same_route is excluded by rule
    assert "HubOnly [EPC]" not in rules  # its only shared mechanism is a hub
    assert rules["NearMoA [EPC]"] == ("shared_moa", ["Mechanism A [MoA]"])

    # at max_df=1 every shared value is a hub, so no near miss is specific enough
    assert btm.make_near_miss(index, max_value_df=1)("Gold [EPC]", set()) == []

    rules_hi = {
        c["class"]: c["rule"] for c in btm.make_near_miss(index, max_value_df=20)("Gold [EPC]", set())
    }
    assert rules_hi["NearMoA [EPC]"] == "shared_moa"
    assert rules_hi["HubOnly [EPC]"] == "shared_moa"  # not a hub at df <= 20
    assert "RouteOnly [EPC]" not in rules_hi


# ---------------------------------------------------------------------------
# 5. End-to-end assembly on synthetic inputs (no network)
# ---------------------------------------------------------------------------
def test_assemble_mix_writes_train_dev_and_a_closing_manifest(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    bench = tmp_path / "v0.2"

    write_jsonl(train_dir / "tier1_train.jsonl", [
        _item(source="medqa", record_id="t1", state="tier1 state", question="tier1 q",
              template="medqa_usmle4_v1", tier="established"),
        _item(source="medqa", record_id="t2", state="tier1 state 2", question="tier1 q2",
              template="medqa_usmle4_v1", tier="established"),
        # collides with a v0.2 test item by record id, state and question
        _item(source="medqa", record_id="collide-test", state="tier1 state 3", question="tier1 q3",
              template="medqa_usmle4_v1", tier="established"),
    ])
    write_jsonl(train_dir / "prewindow_consistency.jsonl", [
        _item(record_id="NCT-cons-1", state="consistency state", question="cons q1",
              split=Split.TEST, template="ct_claim_set_choice_v1"),
        _item(record_id="NCT-cons-2", state="consistency state 2", question="cons q2",
              split=Split.DEV, template="ct_claim_set_choice_v1", gold="B"),
    ])
    write_jsonl(train_dir / "prewindow_structured.jsonl", [
        _item(record_id="NCT-pre-1", state="prewindow state", question="pre q"),
        # collides with a v0.2 fresh dev record
        _item(record_id="NCT-dev-1", state="prewindow state 2", question="pre q2"),
        # held out (D14): must never enter training
        _item(record_id="NCT-pre-3", state="prewindow state 3", question="pre q3",
              template="ct_phase_choice_v1"),
    ])
    write_jsonl(bench / "tier1" / "medqa.jsonl", [
        _item(source="medqa", record_id="collide-test", state="tier1 state 3", question="tier1 q3",
              template="medqa_usmle4_v1", split=Split.TEST, tier="established"),
        _item(source="medqa", record_id="dev-1", state="dev state", question="dev q",
              template="medqa_usmle4_v1", split=Split.DEV, tier="established"),
    ])
    write_jsonl(bench / "fresh" / "clinicaltrials.jsonl", [
        _item(record_id="NCT-dev-1", state="dev fresh state", question="dev fresh q",
              split=Split.DEV),
        _item(record_id="NCT-held-1", state="held state", question="held q",
              template="ct_phase_choice_v1", split=Split.DEV),
        _item(record_id="NCT-test-1", state="test state", question="test q", split=Split.TEST),
    ])

    manifest = btm.assemble_mix(train_dir=train_dir, bench_dir=bench, cfg=CFG, cap_per_template=100)

    train_rows, train_report = read_jsonl(train_dir / "train.jsonl", Item)
    dev_rows, dev_report = read_jsonl(train_dir / "dev.jsonl", Item)
    assert train_report.n_dropped == 0 and dev_report.n_dropped == 0

    removals = manifest["checks"]["leakage_pass1_removals"]
    assert removals["n_removed_total"] == 3
    assert removals["by_reason"] == {
        btm.R_RECORD_ID: 2,  # collide-test (tier1) + NCT-dev-1 (prewindow)
        btm.R_STATE_HASH: 1,
        btm.R_STATE_QUESTION_HASH: 1,
        btm.R_HELD_OUT: 1,
    }
    assert removals["by_origin_reason"]["prewindow_structured|held_out_template_id"] == 1
    assert removals["by_origin_reason"]["tier1_train|source_record_id_in_v0_2_test_or_dev"] == 1

    recheck = manifest["checks"]["leakage_recheck_on_written_train"]
    assert recheck["ok"] is True and recheck["n_failing_total"] == 0

    assert all(str(r.split) == "train" for r in train_rows)
    assert {r.template_id for r in train_rows}.isdisjoint(btm.HELD_OUT_TEMPLATES)
    # the reused consistency input is untouched on disk, but balanced+capped on the way in
    consistency_raw, _ = read_jsonl(train_dir / "prewindow_consistency.jsonl", Item)
    assert len(consistency_raw) == 2
    assert str(consistency_raw[0].split) == "test"
    assert manifest["mixture"]["n_used_from_each_component"] == {
        "prewindow_consistency": 2,
        "prewindow_structured": 1,
        "tier1_train": 2,
    }
    counts = manifest["checks"]["counts_close"]
    assert counts["closes"] is True
    assert counts["train_n_rows"] == len(train_rows) == 5

    # dev: v0.2 dev only, held-out removed, disjoint from the written train file
    assert {r.template_id for r in dev_rows} == {"medqa_usmle4_v1", "ct_randomised_noul_v1"}
    dev_checks = manifest["checks"]["dev_checks"]
    assert dev_checks["ok"] is True
    assert dev_checks["train_dev_disjoint"]["n_shared_items"] == 0
    assert dev_checks["train_dev_disjoint"]["n_shared_by_kind"] == {
        "item_id": 0, "state_hash": 0, "state_question_hash": 0, "record_key": 0,
    }
    assert dev_checks["n_dev_held_out_template_items"] == 0
    assert manifest["dev"]["n_held_out_removed"] == 1

    # the manifest's headline counts re-derive from the written files
    on_disk, _ = read_jsonl(train_dir / "train.jsonl", Item)
    assert len(on_disk) == manifest["totals"]["train"] == manifest["files"]["train"]["n_rows"]
    assert manifest["by_source_template_qtype"]["train"]["medqa"]["medqa_usmle4_v1"] == {"choice": 2}
    assert manifest["window_rule"]["window_start"] == "2026-03-01"
    assert manifest["held_out_templates"] == list(btm.HELD_OUT_TEMPLATES)
    assert manifest["seed"] == 0 and manifest["caps"]["cap_per_template"] == 100
    assert json.loads((train_dir / "manifest.json").read_text())["totals"]["dev"] == len(dev_rows)


def test_manifest_payload_is_json_serialisable(tmp_path: Path) -> None:
    """No Counter / date objects may survive into the manifest (it must round-trip)."""
    payload = btm.LeakageIndex().to_dict()
    write_json(tmp_path / "x.json", payload)
    assert json.loads((tmp_path / "x.json").read_text()) == payload
    assert normalize_text("  a\n b ") == "a b"
    assert stable_hash({"a": "x"}) == stable_hash({"a": " x "})
