"""Loader tests on synthetic in-memory rows (no network, no real item text)."""

from __future__ import annotations

from datetime import date

from meddecide.bench.schema import QuestionType, Split
from meddecide.bench.tier1 import mcq, medquad, relevance
from meddecide.utils.io import check_no_record_crosses_splits

CFG = {"seed": 0, "caps": {"tier1_max_test_items_per_source": 100, "tier1_max_dev_items_per_source": 100}}
SMALL_CFG = {"seed": 0, "caps": {"tier1_max_test_items_per_source": 1, "tier1_max_dev_items_per_source": 1}}


def _medqa_rows(n=3, offset=0):
    """Synthetic MedQA rows; ``offset`` keeps test and train stems distinct."""
    return [
        {
            "question": f"Synthetic USMLE question {offset + i}?",
            "answer": "Nitrofurantoin",
            "options": {"A": "Ampicillin", "B": "Ceftriaxone", "C": "Doxycycline", "D": "Nitrofurantoin"},
            "meta_info": "step2&3",
            "answer_idx": "D",
        }
        for i in range(n)
    ]


def test_medqa_loader_gold_and_splits() -> None:
    res = mcq.load_medqa(_medqa_rows(3), _medqa_rows(2, offset=100), cfg=CFG, revision="rev")
    assert res.source == "medqa"
    assert res.n_items == 5
    test_items = [i for i in res.items if i.split is Split.TEST]
    dev_items = [i for i in res.items if i.split is Split.DEV]
    assert len(test_items) == 3 and len(dev_items) == 2
    for item in res.items:
        assert item.gold == "D"
        assert item.options[item.gold_index].label == "Nitrofurantoin"
        assert item.qtype is QuestionType.CHOICE
        assert item.tier == "established"


def test_medqa_cap_drops_are_counted() -> None:
    res = mcq.load_medqa(_medqa_rows(5), _medqa_rows(4, offset=100), cfg=SMALL_CFG, revision="rev")
    assert res.n_items == 2  # 1 test + 1 dev
    # caps are per (template, split), so the drop reason names the template
    assert res.dropped["test_over_cap:medqa_usmle4_v1"] == 4
    assert res.dropped["dev_over_cap:medqa_usmle4_v1"] == 3


def test_medqa_records_unparseable_rows() -> None:
    bad = {"question": "x", "options": {"A": "a"}, "answer_idx": "Z"}
    res = mcq.load_medqa([bad], [], cfg=CFG, revision="rev")
    assert res.n_items == 0
    assert res.dropped["unparseable_options"] == 1


def _medmcqa_rows(n=3, offset=0):
    return [
        {
            "id": f"id-{offset + i}",
            "question": f"Synthetic MedMCQA question {offset + i}?",
            "opa": "Hyperplasia",
            "opb": "Hypertrophy",
            "opc": "Atrophy",
            "opd": "Dysplasia",
            "cop": "2",
            "subject_name": "Anatomy",
            "topic_name": "Urinary tract",
            "choice_type": "single",
        }
        for i in range(n)
    ]


def test_medmcqa_uses_validation_as_test_and_carves_dev() -> None:
    res = mcq.load_medmcqa(_medmcqa_rows(3), _medmcqa_rows(4, offset=100), cfg=CFG, revision="rev")
    assert res.n_items == 7
    assert {i.split for i in res.items} == {Split.TEST, Split.DEV}
    # `cop` is 0-based, so cop="2" selects the third option ("C"). bench_v0 read it as 1-based
    # and produced "B" here; this assertion is the regression test for that off-by-one.
    assert all(i.gold == "C" for i in res.items)
    assert "validation" in res.split_map["test"]
    assert any("labels are not public" in n for n in res.notes)


def test_medmcqa_bad_cop_is_dropped() -> None:
    rows = _medmcqa_rows(1)
    rows[0]["cop"] = "9"
    res = mcq.load_medmcqa(rows, [], cfg=CFG, revision="rev")
    assert res.n_items == 0 and res.dropped["unparseable_record"] == 1


def _pubmedqa_rows(n=6, offset=0):
    return [
        {
            "pubid": str(1000 + offset + i),
            "question": f"Synthetic PubMedQA question {offset + i}?",
            "context": {"contexts": [f"Synthetic abstract context {i}."], "labels": [], "meshes": []},
            "long_answer": "Synthetic long answer.",
            "final_decision": ["yes", "no", "maybe"][i % 3],
        }
        for i in range(n)
    ]


def test_pubmedqa_three_way_gold_and_hash_split() -> None:
    rows = _pubmedqa_rows(30)
    res = mcq.load_pubmedqa(rows, cfg=CFG, revision="rev")
    assert res.n_items == 30
    golds = {i.gold for i in res.items}
    assert golds <= {"A", "B", "C"}
    assert {i.split for i in res.items} <= {Split.TEST, Split.DEV}
    # a record never appears in both splits
    assert check_no_record_crosses_splits(res.items)["ok"] is True
    # re-running gives the same split assignment
    again = mcq.load_pubmedqa(rows, cfg=CFG, revision="rev")
    assert [i.item_id for i in res.items] == [i.item_id for i in again.items]
    assert [i.split for i in res.items] == [i.split for i in again.items]


def test_pubmedqa_decision_maps_to_label_not_position() -> None:
    rows = _pubmedqa_rows(3)
    res = mcq.load_pubmedqa(rows, cfg=CFG, revision="rev")
    by_record = {i.source_record_id: i for i in res.items}
    assert by_record["1000"].options[by_record["1000"].gold_index].label == "yes"
    assert by_record["1001"].options[by_record["1001"].gold_index].label == "no"
    assert by_record["1002"].options[by_record["1002"].gold_index].label == "maybe"


def _mmlu_rows(n=2, n_choices=4, offset=0):
    """Synthetic MMLU rows. ``offset`` keeps test and validation stems distinct."""
    return [
        {
            "question": f"Synthetic MMLU question {offset + i}?",
            "choices": [f"choice {j}" for j in range(n_choices)],
            "answer": 1,
        }
        for i in range(n)
    ]


def test_mmlu_loader_uses_test_and_validation() -> None:
    by_subject = {
        "anatomy": {"test": _mmlu_rows(3), "validation": _mmlu_rows(2, offset=100)},
        "medical_genetics": {"test": _mmlu_rows(1, offset=200), "validation": _mmlu_rows(1, offset=300)},
    }
    res = mcq.load_mmlu(by_subject, cfg=CFG, revision="rev")
    assert res.n_items == 7
    assert all(i.gold == "B" for i in res.items)
    assert sum(1 for i in res.items if i.split is Split.TEST) == 4
    assert sum(1 for i in res.items if i.split is Split.DEV) == 3
    assert res.items[0].meta["subject"] in {"anatomy", "medical_genetics"}


def _medquad_rows(n=100, offset=0):
    types = ["symptoms", "treatment", "diagnosis", "prevention"]
    return [
        {
            "question_id": f"{offset + i}-1",
            "question": f"Synthetic MedQuAD question {offset + i}?",
            "question_type": types[i % len(types)],
            "question_focus": "synthetic focus",
            "document_source": "GHR",
            "document_url": "https://example.org/doc",
            "umls_cui": "C0000000",
            "umls_semantic_types": "T047",
            "umls_semantic_group": "Disorders",
            "synonyms": "SYN",
            "answer": "Synthetic answer text.",
        }
        for i in range(n)
    ]


def test_medquad_builds_routing_items_without_umls_fields() -> None:
    res = medquad.load_medquad(_medquad_rows(100), cfg=CFG, revision="rev")
    assert res.n_items == 100
    for item in res.items:
        # the UMLS-only fields must not survive into an item, in any form
        blob = item.model_dump_json()
        for field in medquad.UMLS_FIELDS:
            assert field not in blob
        assert "C0000000" not in blob and "T047" not in blob
        assert item.gold in item.option_keys
        assert len(item.options) == 4  # 3 distractors + gold
        assert item.qtype is QuestionType.CHOICE
    # the answer text must never appear in the state (the task is routing, not answering)
    assert all("Synthetic answer text." not in i.state for i in res.items)


def test_medquad_drops_types_below_frequency_threshold() -> None:
    rows = _medquad_rows(100)
    rows[0]["question_type"] = "rare-type"
    res = medquad.load_medquad(rows, cfg=CFG, revision="rev")
    assert res.n_items == 99
    assert res.dropped["question_type_below_frequency_threshold"] == 1


def test_medquad_requires_two_eligible_types() -> None:
    rows = _medquad_rows(100)
    for row in rows:
        row["question_type"] = "only-type"
    res = medquad.load_medquad(rows, cfg=CFG, revision="rev")
    assert res.n_items == 0
    assert res.dropped["fewer_than_two_eligible_types"] == 100


# ---------------------------------------------------------------------------
# relevance
# ---------------------------------------------------------------------------
def _beir_fixture(n_queries=10, n_docs=40):
    queries = [{"_id": str(i), "text": f"Synthetic query {i}"} for i in range(n_queries)]
    corpus = [
        {"_id": str(j), "title": f"Synthetic title {j}", "text": f"Synthetic passage body {j}."}
        for j in range(n_docs)
    ]
    qrels = []
    for i in range(n_queries):
        qrels.append({"query-id": str(i), "corpus-id": str(i), "score": 2})
        qrels.append({"query-id": str(i), "corpus-id": str(n_queries + i), "score": 1})
        qrels.append({"query-id": str(i), "corpus-id": str(2 * n_queries + i), "score": 0})
    return queries, corpus, qrels


def test_relevance_noul_has_one_positive_one_negative_per_query() -> None:
    queries, corpus, qrels = _beir_fixture()
    res = relevance.load_beir_relevance(
        source="synthetic_rel", dataset_id="x/y", queries=queries, corpus=corpus, qrels=qrels,
        cfg=CFG, revision="rev", split=Split.TEST, record_date=date(2020, 1, 1),
        graded=False,
    )
    assert res.n_items == 20
    assert all(i.qtype is QuestionType.NOUL for i in res.items)
    golds = [i.gold for i in res.items]
    assert golds.count("yes") == 10 and golds.count("no") == 10
    # the negative passage is one the qrels mark 0, never a positive
    for item in res.items:
        if item.gold == "no":
            assert item.meta["qrel_grade"] == 0
        else:
            assert item.meta["qrel_grade"] > 0


def test_relevance_graded_score_items_use_all_three_levels() -> None:
    queries, corpus, qrels = _beir_fixture()
    res = relevance.load_beir_relevance(
        source="synthetic_rel", dataset_id="x/y", queries=queries, corpus=corpus, qrels=qrels,
        cfg=CFG, revision="rev", split=Split.TEST, record_date=date(2020, 1, 1),
        graded=True,
    )
    score_items = [i for i in res.items if i.qtype is QuestionType.SCORE]
    assert len(score_items) == 30  # 10 queries x 3 levels
    assert {i.gold for i in score_items} == {"1", "2", "3"}
    for item in score_items:
        assert item.options[item.gold_index].label == relevance.GRADE_LEVELS[int(item.gold) - 1]


def test_relevance_gold_level_matches_grade() -> None:
    queries, corpus, qrels = _beir_fixture(1, 5)
    res = relevance.load_beir_relevance(
        source="synthetic_rel", dataset_id="x/y", queries=queries, corpus=corpus, qrels=qrels,
        cfg=CFG, revision="rev", split=Split.TEST, record_date=date(2020, 1, 1),
        graded=True,
    )
    by_grade = {i.meta["qrel_grade"]: i for i in res.items if i.qtype is QuestionType.SCORE}
    assert by_grade[2].gold == "3" and by_grade[2].options[2].label == "Highly relevant"
    assert by_grade[1].gold == "2" and by_grade[1].options[1].label == "Relevant"
    assert by_grade[0].gold == "1" and by_grade[0].options[0].label == "Not relevant"


def test_relevance_sampling_is_deterministic() -> None:
    queries, corpus, qrels = _beir_fixture()
    kwargs = {
        "source": "synthetic_rel", "dataset_id": "x/y", "queries": queries, "corpus": corpus,
        "qrels": qrels, "cfg": CFG, "revision": "rev", "split": Split.TEST,
        "record_date": date(2020, 1, 1), "graded": True,
    }
    first = relevance.load_beir_relevance(**kwargs)
    second = relevance.load_beir_relevance(**kwargs)
    assert [i.item_id for i in first.items] == [i.item_id for i in second.items]
    assert [i.meta["corpus_id"] for i in first.items] == [i.meta["corpus_id"] for i in second.items]


def test_relevance_query_without_judged_passage_is_dropped_not_invented() -> None:
    queries, corpus, qrels = _beir_fixture(2, 10)
    # remove query 1's judged passages: it still has a query text but no judged passage
    qrels = [q for q in qrels if q["query-id"] != "1"]
    res = relevance.load_beir_relevance(
        source="synthetic_rel", dataset_id="x/y", queries=queries, corpus=corpus, qrels=qrels,
        cfg=CFG, revision="rev", split=Split.TEST, record_date=date(2020, 1, 1),
        graded=False,
    )
    assert res.n_items == 2  # only query 0
    assert res.dropped["query_without_text_or_judged_passage"] == 1


def test_mmlu_identical_stem_on_both_sides_is_forced_to_test() -> None:
    """MMLU really does repeat a stem with different answer sets across splits."""
    by_subject = {
        "anatomy": {
            "test": [{"question": "Shared stem?", "choices": ["a", "b", "c", "d"], "answer": 0}],
            "validation": [{"question": "Shared stem?", "choices": ["w", "x", "y", "z"], "answer": 2}],
        }
    }
    res = mcq.load_mmlu(by_subject, cfg=CFG, revision="rev")
    assert res.n_items == 2
    assert {i.split for i in res.items} == {Split.TEST}, "no stem may sit on both sides"
    assert res.dropped.get("content_in_two_splits_moved_to_test") == 1
    assert any("identical question text" in n for n in res.notes)


def test_medquad_same_question_in_two_documents_lands_in_one_split() -> None:
    """MedQuAD repeats a question across documents; the split follows the question text."""
    from meddecide.bench.tier1 import medquad as mq

    rows = mq_rows_with_shared_question()
    res = mq.load_medquad(rows, cfg={"seed": 0, "caps": {}}, revision="rev")
    shared = [i for i in res.rows if i.state.endswith("What is shared condition ?")]
    assert len(shared) == 2
    assert len({str(i.split) for i in shared}) == 1, "one question must not straddle dev/test"


def mq_rows_with_shared_question():
    types = ["symptoms", "treatment", "diagnosis", "prevention"]
    rows = []
    for i in range(100):
        rows.append(
            {
                "question_id": f"q{i}-1",
                "document_id": f"doc{i}",
                "question": f"Synthetic MedQuAD question {i}?",
                "question_type": types[i % len(types)],
                "question_focus": "focus",
                "document_source": "GHR",
                "document_url": "https://example.org/doc",
                "umls_cui": "C0000000",
                "answer": "answer text",
            }
        )
    for j, doc in enumerate(["docA", "docB"]):
        rows.append(
            {
                "question_id": f"shared-{j}",
                "document_id": doc,
                "question": "What is shared condition ?",
                "question_type": "symptoms",
                "question_focus": "shared condition",
                "document_source": "GHR",
                "document_url": "https://example.org/doc",
                "umls_cui": "C0000000",
                "answer": "answer text",
            }
        )
    return rows


def test_medmcqa_cop_is_zero_based() -> None:
    """cop=0 must select option A, and cop=3 must select D (F1 regression test).

    Evidence for the 0-based reading is in scripts/bench/verify_gold.py --prove-cop: the record's
    own `exp` explanation names option[cop] in 66.6 % of rows versus 11.5 % for option[cop+1].
    """
    from meddecide.bench.tier1 import mcq

    for cop, expected in ((0, "A"), (1, "B"), (2, "C"), (3, "D")):
        rows = _medmcqa_rows(1)
        rows[0]["cop"] = str(cop)
        res = mcq.load_medmcqa(rows, [], cfg=CFG, revision="rev")
        assert res.n_items == 1, f"cop={cop} was dropped instead of mapped"
        assert res.rows[0].gold == expected, f"cop={cop} mapped to {res.rows[0].gold}, want {expected}"
        # the gold option's text must be the cop-indexed option text
        labels = [rows[0]["opa"], rows[0]["opb"], rows[0]["opc"], rows[0]["opd"]]
        assert res.rows[0].options[res.rows[0].gold_index].label == labels[cop]
    # an out-of-range cop is dropped, with a reason
    bad = _medmcqa_rows(1)
    bad[0]["cop"] = "4"
    res = mcq.load_medmcqa(bad, [], cfg=CFG, revision="rev")
    assert res.n_items == 0 and res.dropped["unparseable_record"] == 1
