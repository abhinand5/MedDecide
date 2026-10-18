"""osler_v3 X1 source converters (CPU, synthetic records shaped like the real ones): sentences, RAGTruth and DrugProt.

The records here are invented for the tests. The real RAGTruth and DrugProt rows are read only by scripts/osler_v3/x1_real_sources.py.
"""

from __future__ import annotations

from collections import Counter

import pytest

from meddecide.mix import drugprot, ragtruth
from meddecide.mix.sentences import sentence_index, sentence_spans


def _span(text: str, piece: str) -> tuple[int, int]:
    start = text.index(piece)
    return start, start + len(piece)


# ---- sentences ----------------------------------------------------------------------------------------------------

def test_sentence_spans_split_after_terminal_punctuation_and_newlines() -> None:
    text = "First one. Second one! Third?\nFourth line  "
    spans = sentence_spans(text)
    assert [text[a:b] for a, b in spans] == ["First one.", "Second one!", "Third?", "Fourth line"]


def test_sentence_spans_of_empty_text_is_empty() -> None:
    assert sentence_spans("") == []
    assert sentence_spans("  \n  ") == []


def test_sentence_index_finds_the_containing_sentence_or_none() -> None:
    text = "Alpha beta. Gamma delta."
    spans = sentence_spans(text)
    assert sentence_index(spans, text.index("Gamma")) == 1
    assert sentence_index(spans, 0) == 0
    assert sentence_index(spans, len(text) + 5) is None


# ---- RAGTruth -----------------------------------------------------------------------------------------------------

def test_ragtruth_official_test_contexts_stay_test() -> None:
    assert ragtruth.context_split("15592", "test") == "test"


def test_ragtruth_train_contexts_split_by_a_stable_hash_into_train_or_dev() -> None:
    got = {ragtruth.context_split(str(i), "train") for i in range(200)}
    assert got == {"train", "dev"}
    assert ragtruth.context_split("15592", "train") == ragtruth.context_split("15592", "train")
    with pytest.raises(ValueError):
        ragtruth.context_split("1", "validation")


@pytest.mark.parametrize(("labels", "gold", "reason"), [
    ([], ragtruth.NONE_KEY, "kept"),
    ([{"label_type": "Subtle Conflict"}], "D", "kept"),
    ([{"label_type": "Evident Baseless Info"}, {"label_type": "Subtle Baseless Info"}], None, "mixed_types"),
    ([{"label_type": "Made Up Type"}], None, "unknown_type"),
])
def test_ragtruth_type_gold(labels: list, gold: str | None, reason: str) -> None:
    assert ragtruth.response_type_gold(labels) == (gold, reason)


def _rt_source(source_id: str, prompt: str) -> dict:
    return {"source_id": source_id, "prompt": prompt, "source": "CNN/DM", "task_type": "Summary", "source_info": prompt}


def _rt_response(rid: str, source_id: str, text: str, labels: list[dict], split: str = "test",
                 quality: str = "good") -> dict:
    return {"id": rid, "source_id": source_id, "model": "gpt-4-0613", "temperature": 0.7, "labels": labels,
            "split": split, "quality": quality, "response": text}


def _label(text: str, piece: str, label_type: str) -> dict:
    start, end = _span(text, piece)
    return {"start": start, "end": end, "text": piece, "label_type": label_type, "implicit_true": False,
            "due_to_null": False, "meta": "ANNOTATOR NOTE THAT MUST NOT ENTER AN ITEM"}


def test_ragtruth_build_labels_come_from_spans_and_balance_holds() -> None:
    sources = {"s1": _rt_source("s1", "Summarize: article one."), "s2": _rt_source("s2", "Summarize: article two.")}
    grounded_text = "The sky is blue. It is sunny."
    bad_text = "The sky is green. It is sunny."
    mixed_text = "The sky is blue. It rains a lot. Mars is red."
    responses = [
        _rt_response("0", "s1", grounded_text, []),
        _rt_response("1", "s1", bad_text, [_label(bad_text, "green", "Evident Conflict")]),
        _rt_response("2", "s2", mixed_text, [_label(mixed_text, "It rains a lot.", "Subtle Baseless Info"),
                                              _label(mixed_text, "Mars is red.", "Evident Baseless Info")]),
        _rt_response("3", "s2", grounded_text, [], quality="truncated"),
    ]
    out, report = ragtruth.build(responses, sources)
    # response level: r0 has no span (yes); r1 and r2 have spans (no). Balanced 1:1 keeps one of each.
    assert Counter(r["gold"] for r in out[ragtruth.GROUNDED]) == {"yes": 1, "no": 1}
    for row in out[ragtruth.GROUNDED]:
        assert "ANNOTATOR NOTE" not in row["state"] and "ANNOTATOR NOTE" not in row["question"]
        assert row["meta"]["dataset"] == "CNN/DM"
    # sentence level: a sentence is "no" exactly when it overlaps an annotated span; 4 yes and 3 no, balanced to 3 and 3
    hallucinated = {"The sky is green.", "It rains a lot.", "Mars is red."}
    for row in out[ragtruth.SENTENCE]:
        sentence = row["question"].split("Sentence: ", 1)[1]
        assert row["gold"] == ("no" if sentence in hallucinated else "yes"), sentence
    assert Counter(r["gold"] for r in out[ragtruth.SENTENCE]) == {"yes": 3, "no": 3}
    # type level: r0 is none (E), r1 is one type (C), r2 has two types and is excluded from this template
    assert Counter(r["gold"] for r in out[ragtruth.HALLUC_TYPE]) == {ragtruth.NONE_KEY: 1, "C": 1}
    assert all(r["options"][-1]["key"] == ragtruth.NONE_KEY for r in out[ragtruth.HALLUC_TYPE])
    assert report["counts"]["dropped_quality_truncated"] == 1
    assert report["counts"]["dropped_mixed_types_for_type_template"] == 1


# ---- DrugProt -----------------------------------------------------------------------------------------------------

def _dp_entity(eid: str, etype: str, text: str, abstract: str, occurrence: int = 0) -> dict:
    start = -1
    for _ in range(occurrence + 1):
        start = abstract.index(text, start + 1)
    return {"id": eid, "type": etype, "text": text, "start": start, "end": start + len(text)}


def _dp_record(pmid: str, abstract: str, entities: list[dict], relations: list[dict]) -> dict:
    return {"pmid": pmid, "title": "", "abstract": abstract, "text": abstract, "entities": entities, "relations": relations}


def test_drugprot_option_keys_are_letters_with_no_relation_last() -> None:
    keys = drugprot.option_keys(["ACTIVATOR", "INHIBITOR"])
    assert keys == {"ACTIVATOR": "A", "INHIBITOR": "B", drugprot.NO_RELATION: "C"}


def test_drugprot_pmid_split_is_stable_and_keeps_train_whole() -> None:
    assert drugprot.pmid_split("17512723", "train") == "train"
    assert drugprot.pmid_split("17651117", "validation") == drugprot.pmid_split("17651117", "validation")
    got = {drugprot.pmid_split(str(p), "validation") for p in range(200)}
    assert got == {"dev", "test"}


ABSTRACT = "Aspirin inhibits COX1 in platelets. Retinol activates RDH12 in the eye. Ibuprofen is studied alone."


def test_drugprot_pairs_and_same_sentence_candidates() -> None:
    entities = [
        _dp_entity("T1", "CHEMICAL", "Aspirin", ABSTRACT), _dp_entity("T2", "GENE-N", "COX1", ABSTRACT),
        _dp_entity("T3", "CHEMICAL", "Retinol", ABSTRACT), _dp_entity("T4", "GENE-Y", "RDH12", ABSTRACT),
        _dp_entity("T5", "CHEMICAL", "Ibuprofen", ABSTRACT),
    ]
    relations = [{"arg1": "T1", "arg2": "T2", "type": "INHIBITOR"},
                 {"arg1": "T4", "arg2": "T3", "type": "ACTIVATOR"},
                 {"arg1": "T1", "arg2": "T2", "type": "SUBSTRATE"}]
    record = _dp_record("1", ABSTRACT, entities, relations)
    pairs, counts = drugprot.chem_gene_pairs(record)
    assert pairs == {("T1", "T2"): {"INHIBITOR", "SUBSTRATE"}, ("T3", "T4"): {"ACTIVATOR"}}
    assert counts == Counter()
    assert drugprot.co_sentence_pairs(record) == {("T1", "T2"), ("T3", "T4")}


def test_drugprot_build_has_one_negative_per_positive_and_offers_no_relation() -> None:
    other = "Warfarin and CYP2C9 are linked. Warfarin is not mentioned with CYP3A4."
    entities_a = [
        _dp_entity("T1", "CHEMICAL", "Aspirin", ABSTRACT), _dp_entity("T2", "GENE-N", "COX1", ABSTRACT),
        _dp_entity("T3", "CHEMICAL", "Retinol", ABSTRACT), _dp_entity("T4", "GENE-Y", "RDH12", ABSTRACT),
        _dp_entity("T5", "CHEMICAL", "Ibuprofen", ABSTRACT),
    ]
    record_a = _dp_record("11", ABSTRACT, entities_a, [{"arg1": "T1", "arg2": "T2", "type": "INHIBITOR"},
                                                       {"arg1": "T4", "arg2": "T3", "type": "ACTIVATOR"}])
    entities_b = [
        _dp_entity("U1", "CHEMICAL", "Warfarin", other), _dp_entity("U2", "GENE-N", "CYP2C9", other),
        _dp_entity("U3", "GENE-N", "CYP3A4", other),
    ]
    record_b = _dp_record("12", other, entities_b, [])
    options = drugprot.relation_types([record_a, record_b])
    rows, report = drugprot.build([("train", record_a), ("train", record_b)], options)
    train = rows["train"]
    positives = [r for r in train if r["gold"] != drugprot.option_keys(options)[drugprot.NO_RELATION]]
    negatives = [r for r in train if r["gold"] == drugprot.option_keys(options)[drugprot.NO_RELATION]]
    assert len(positives) == 2
    assert len(negatives) <= len(positives)
    assert all(r["options"][-1]["label"] == drugprot.NO_RELATION for r in train)
    assert report["counts"]["positives_train"] == 2
    assert report["counts"]["negative_candidates_train"] >= len(negatives)
