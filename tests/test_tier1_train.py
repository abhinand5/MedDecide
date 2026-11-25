"""Unit tests for the S5 tier-1 training builder (``scripts/bench/build_tier1_train.py``).

Three behaviours are acceptance criteria and are tested directly, with no network and no
downloaded data: the leakage remover, the MedMCQA dev-carve exclusion, and the "offer only
the score levels present in the pool" rule (the S1 `_v2` fix) as it applies to a train pool.
Two smaller rules the builder relies on are covered as well: the MedQA contradictory-stem
drop and the template-id rewrite.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

from meddecide.bench.schema import Item, Split, make_item
from meddecide.bench.tier1 import relevance
from meddecide.bench.tier1.common import split_train_into_dev_train
from meddecide.utils.io import write_jsonl

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "build_tier1_train.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_tier1_train", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


btt = _load_builder()

CFG = {"seed": 0, "caps": {"tier1_max_test_items_per_source": 5000,
                           "tier1_max_dev_items_per_source": 2000}}


def _item(
    *,
    source: str = "medqa",
    record_id: str = "r1",
    state: str = "a state",
    question: str = "which of the following?",
    template: str = "medqa_usmle4_v1",
    split: Split | str = Split.TRAIN,
    gold: str = "A",
    labels: tuple[str, ...] = ("alpha", "beta"),
    qtype: str = "choice",
) -> Item:
    from meddecide.bench.schema import QuestionType

    if qtype == "noul":
        options = [{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}]
        gold = "yes"
    else:
        options = [{"key": chr(ord("A") + i), "label": text} for i, text in enumerate(labels)]
    return make_item(
        tier="established", source=source, source_record_id=record_id,
        source_url="https://example.org/dataset", source_license="cc0",
        record_date=date(2021, 1, 1), split=split, template_id=template,
        skill="medical_knowledge", qtype=QuestionType(qtype), state=state, question=question,
        options=options, gold=gold, option_order_seed=0,
    )


# ---------------------------------------------------------------------------
# 1. Leakage remover
# ---------------------------------------------------------------------------
def test_leakage_index_reads_only_test_and_dev(tmp_path: Path) -> None:
    bench = tmp_path / "v0.2"
    write_jsonl(bench / "tier1" / "medqa.jsonl", [
        _item(split=Split.TEST, state="shared stem", question="q"),
        _item(split=Split.DEV, record_id="dev-1", state="dev stem", question="q"),
        _item(split=Split.TRAIN, record_id="train-1", state="train stem", question="q"),
    ])
    write_jsonl(bench / "fresh" / "openfda.jsonl", [
        _item(source="openfda", split=Split.TEST, record_id="f1", state="fresh test", question="q"),
    ])
    index = btt.build_leakage_index(bench)
    assert index.n_items == 3  # the tier-1 train item is not a reference
    assert index.by_split == {"dev": 1, "test": 2}
    assert index.files == {"fresh/openfda.jsonl": 1, "tier1/medqa.jsonl": 2}


def test_leakage_filter_removes_content_hash_and_record_id_collisions(tmp_path: Path) -> None:
    bench = tmp_path / "v0.2"
    write_jsonl(bench / "tier1" / "medqa.jsonl", [
        _item(split=Split.TEST, record_id="ref-1", state="the same stem", question="the question"),
    ])
    index = btt.build_leakage_index(bench)

    content_hit = _item(record_id="other", state="  the   same stem ", question="the question")
    record_hit = _item(record_id="ref-1", state="different stem", question="different question")
    both_hit = _item(record_id="ref-1", state="the same stem", question="the question")
    clean = _item(record_id="clean", state="unseen stem", question="unseen question")
    kept, report = btt.apply_leakage_filter([content_hit, record_hit, both_hit, clean], index)

    assert [i.item_id for i in kept] == [clean.item_id]
    assert report["n_input"] == 4
    assert report["n_removed_total"] == 3
    assert report["n_removed_content_hash"] == 2  # content_hit + both_hit
    assert report["n_removed_record_id"] == 2  # record_hit + both_hit
    assert report["by_source"] == {"medqa": 3}
    assert report["by_source_reason"]["medqa|content_hash_in_v0_2_test_or_dev"] == 2
    assert report["by_source_reason"]["medqa|source_record_id_in_v0_2_test_or_dev"] == 2
    assert report["n_kept"] == 1
    assert {tuple(e["reasons"]) for e in report["examples"]} == {
        ("content_hash_in_v0_2_test_or_dev",),
        ("source_record_id_in_v0_2_test_or_dev",),
        ("content_hash_in_v0_2_test_or_dev", "source_record_id_in_v0_2_test_or_dev"),
    }


def test_leakage_hash_normalises_whitespace_and_ignores_split() -> None:
    a = btt.leakage_content_hash("Query:  a\nPassage: b", "Is it relevant?")
    b = btt.leakage_content_hash("Query: a Passage: b", "Is  it relevant?")
    assert a == b
    assert btt.leakage_content_hash("x", "y") != btt.leakage_content_hash("y", "x")


# ---------------------------------------------------------------------------
# 2. MedMCQA dev-carve exclusion
# ---------------------------------------------------------------------------
def test_medmcqa_carve_excludes_exactly_the_benchmark_dev_rows() -> None:
    rows = [{"id": str(i), "question": f"question {i}"} for i in range(500)]
    kept, dev_indices, over = btt.select_medmcqa_train_indices(
        len(rows), dev_cap=100, cap=200, seed=0
    )
    dev_rows, train_rows, _ = split_train_into_dev_train(rows, dev_cap=100, seed=0, salt="medmcqa")

    # the carve is the benchmark's carve, row for row
    assert [rows[i]["id"] for i in dev_indices] == [r["id"] for r in dev_rows]
    assert len(dev_indices) == 100
    # exclusion is total and disjoint
    assert set(kept).isdisjoint(dev_indices)
    assert {rows[i]["id"] for i in kept} <= {r["id"] for r in train_rows}
    # the cap is applied to what is left, and the drop is counted
    assert len(kept) == 200
    assert over == len(train_rows) - 200
    # deterministic
    assert btt.select_medmcqa_train_indices(len(rows), dev_cap=100, cap=200, seed=0) == (
        kept, dev_indices, over,
    )
    # no cap: everything the dev set did not use
    kept_all, _, over_all = btt.select_medmcqa_train_indices(len(rows), dev_cap=100, cap=10**6, seed=0)
    assert len(kept_all) == len(train_rows)
    assert over_all == 0


def test_medmcqa_carve_reproduces_v0_1_seed_and_salt() -> None:
    """The carve must be the *same* stream the v0.1 loader used (seed=0, salt='medmcqa')."""
    import random

    rows = [{"id": str(i)} for i in range(200_000)]
    _, dev_indices, _ = btt.select_medmcqa_train_indices(len(rows), dev_cap=2_000, cap=60_000, seed=0)
    rng = random.Random("0:medmcqa:dev")
    assert dev_indices == sorted(rng.sample(range(len(rows)), 2_000))


# ---------------------------------------------------------------------------
# 3. "Only the levels present" score rule
# ---------------------------------------------------------------------------
def test_score_levels_present_uses_the_pool_and_the_grade_mapping() -> None:
    corpus = {"d1", "d2", "d3"}
    assert btt.score_levels_present({"q1": {"d1": 1, "d2": 2}}, ["q1"], corpus) == [1, 2]
    assert btt.score_levels_present({"q1": {"d1": 1}}, ["q1"], corpus) == [1]  # binary pool
    assert btt.score_levels_present({"q1": {"d1": 0, "d2": 2}}, ["q1"], corpus) == [0, 2]
    assert btt.score_levels_present({"q1": {"d9": 2}}, ["q1"], corpus) == []  # passage not in corpus
    assert btt.score_levels_present({"q1": {"d1": -1}}, ["q1"], corpus) == [0]  # negative -> lowest
    assert btt.score_levels_present({"q1": {"d1": 2, "d2": 2}}, ["q1"], corpus) == [2]
    # queries outside the pool passed in contribute nothing
    assert btt.score_levels_present({"q1": {"d1": 1}, "q2": {"d2": 2}}, ["q1"], corpus) == [1]


def _beir(game: dict[str, int], *, graded: bool):
    queries = [{"_id": "q1", "text": "query one"}]
    corpus = [
        {"_id": cid, "title": f"title {cid}", "text": f"body {cid}"} for cid in ("d1", "d2", "d3")
    ]
    qrels = [{"query-id": "q1", "corpus-id": cid, "score": g} for cid, g in game.items()]
    return relevance.load_beir_relevance(
        source="nfcorpus", dataset_id=relevance.NFCORPUS_ID, queries=queries, corpus=corpus,
        qrels=qrels, cfg=CFG, revision="test-revision", split=Split.TRAIN,
        record_date=date(2015, 1, 1), graded=graded, score_levels=None,
        score_template_id=btt.NFCORPUS_TRAIN_SCORE_TEMPLATE,
    )


def test_score_items_offer_exactly_the_levels_present_and_every_level_is_a_gold() -> None:
    levels = btt.score_levels_present({"q1": {"d1": 1, "d2": 2}}, ["q1"], {"d1", "d2", "d3"})
    assert levels == [1, 2]
    res = _beir({"d1": 1, "d2": 2}, graded=True)
    score_items = [i for i in res.rows if str(i.qtype) == "score"]
    assert len(score_items) == 2
    assert all(i.template_id == btt.NFCORPUS_TRAIN_SCORE_TEMPLATE for i in score_items)
    assert all(i.meta["offered_levels"] == levels for i in score_items)
    assert all(
        [o.label for o in i.options] == [relevance.GRADE_LEVELS[k] for k in levels]
        for i in score_items
    )
    # every offered level is the gold of at least one built item
    assert {i.gold for i in score_items} == {str(k + 1) for k in range(len(levels))}
    # and the gold of each item is this pair's own grade
    by_corpus = {i.meta["corpus_id"]: i for i in score_items}
    assert by_corpus["d1"].gold == "1" and by_corpus["d2"].gold == "2"


def test_binary_pool_yields_one_level_so_no_score_item_is_built() -> None:
    """The NFCorpus train qrels case: one level present -> the caller must not build score items."""
    levels = btt.score_levels_present({"q1": {"d1": 1, "d2": 1}}, ["q1"], {"d1", "d2", "d3"})
    assert levels == [1]
    res = _beir({"d1": 1, "d2": 1}, graded=False)
    assert [i for i in res.rows if str(i.qtype) == "score"] == []
    noul = [i for i in res.rows if str(i.qtype) == "noul"]
    assert {i.gold for i in noul} == {"yes", "no"}  # the noul items are still built


# ---------------------------------------------------------------------------
# Smaller rules the builder relies on
# ---------------------------------------------------------------------------
def test_medqa_conflicting_stems_finds_only_contradictions() -> None:
    same = {"question": "please refer to the summary above", "options": {"A": "x", "B": "y"},
            "answer_idx": "A"}
    other = {"question": "please refer to the summary above", "options": {"A": "x", "B": "z"},
             "answer_idx": "B"}
    identical = dict(same)
    conflicts = btt.medqa_conflicting_stems([same, other])
    assert len(conflicts) == 1
    detail = next(iter(conflicts.values()))
    assert detail["n_rows_with_this_stem"] == 2
    assert detail["n_distinct_option_sets"] == 2
    assert detail["n_distinct_correct_answers"] == 2
    # identical repeats are a dedupe problem, not an ambiguity
    assert btt.medqa_conflicting_stems([same, identical]) == {}
    # a differing answer with an identical option set is still a contradiction
    flipped = {**same, "answer_idx": "B"}
    assert len(btt.medqa_conflicting_stems([same, flipped])) == 1


def test_retemplate_changes_the_id_and_nothing_else() -> None:
    item = _item(record_id="r9", state="stem", question="q")
    renamed = btt.retemplate(item, "scifact_relevant_noul_train_v1", option_order_seed=0)
    assert renamed.template_id == "scifact_relevant_noul_train_v1"
    assert renamed.item_id != item.item_id
    assert (renamed.state, renamed.question, renamed.gold, renamed.split) == (
        item.state, item.question, item.gold, item.split,
    )
    assert [o.label for o in renamed.options] == [o.label for o in item.options]
    assert renamed.meta == item.meta


def test_build_medquad_train_items_forces_train_split_and_excludes_used_rows() -> None:
    rows = [
        {"question_id": "1", "document_id": "d1", "question": "what is x?",
         "question_type": "treatment", "document_source": "s", "question_focus": "f"},
        {"question_id": "2", "document_id": "d2", "question": "what is y?",
         "question_type": "symptoms", "document_source": "s", "question_focus": "f"},
        {"question_id": "2", "document_id": "d3", "question": "what is z?",
         "question_type": "symptoms", "document_source": "s", "question_focus": "f"},
    ] + [
        {"question_id": str(100 + i), "document_id": f"d{100 + i}", "question": f"filler {i}?",
         "question_type": kind, "document_source": "s", "question_focus": "f"}
        for i, kind in enumerate(["treatment"] * 25 + ["symptoms"] * 25)
    ]
    items, report = btt.build_medquad_train_items(
        rows, seed=0, license_="UNKNOWN", used_question_ids={"2"}
    )
    assert all(i.split is Split.TRAIN for i in items)
    assert report["dropped_by_reason"]["question_id_used_by_v0_2_tier1_split"] == 2
    states = " ".join(i.state for i in items)
    assert "what is y?" not in states and "what is z?" not in states
    assert "what is x?" in states
    # every option is a question type the source uses often enough to be eligible, and the
    # gold option is the row's own question_type
    eligible = set(report["option_types"])
    assert all(o.label in eligible for i in items for o in i.options)
    by_question = {
        i.state.replace("Consumer health question: ", ""): i.options[i.gold_index].label
        for i in items
    }
    assert by_question["what is x?"] == "treatment"
    assert by_question["filler 0?"] == "treatment"
    assert by_question["filler 25?"] == "symptoms"
