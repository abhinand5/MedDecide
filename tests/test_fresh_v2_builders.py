"""Unit tests for the v0.2 near-miss builders (S1).

Both builders take their distractor lookup as an injected callable, so these tests need no
network, no MeSH file and no openFDA index: the *rule* is what is being tested — that the
distractors come from the sibling / mechanism pool, and that no other correct answer for the
same record can be offered as a distractor.
"""

from __future__ import annotations

from datetime import date

from meddecide.bench.fresh import openfda, pubmed
from meddecide.bench.schema import QuestionType

CFG = {"seed": 0, "caps": {"tier1_max_test_items_per_source": 5000,
                           "tier1_max_dev_items_per_source": 2000}}
WINDOW = (date(2026, 3, 1), date(2026, 10, 5))


def _record(pmid: str, topics: list[str]) -> pubmed.PubmedRecord:
    return pubmed.PubmedRecord(
        pmid=pmid,
        entrez_date=date(2026, 4, 1),
        title=f"Title {pmid}",
        abstract="Abstract body text for the synthetic record.",
        pub_types=["Journal Article"],
        mesh_major_topics=topics,
        check_tags=["Humans"],
        journal="Synthetic Journal",
    )


SIBLINGS = {
    "Hypertension": ["Blood Pressure", "Prehypertension", "Masked Hypertension", "White Coat Hypertension"],
    "Blood Pressure": ["Hypotension", "Orthostatic Hypotension"],
    "Prehypertension": ["Hypertension", "Blood Pressure"],
    "Diabetes Mellitus, Type 2": ["Diabetes Mellitus, Type 1"],
}


def test_mesh_v2_distractors_are_tree_siblings() -> None:
    records = [_record("1", ["Hypertension"])]
    rows, drops, notes = pubmed.build_mesh_major_v2_items(
        records, cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1],
        mesh_siblings=lambda name: SIBLINGS.get(name, []),
        mesh_meta={"year": 2026, "sha256": "deadbeef"},
    )
    assert len(rows) == 1
    assert drops.get("mesh_index_unavailable", 0) == 0
    item = rows[0]
    assert item.template_id == "pubmed_mesh_major_choice_v2"
    assert item.n_options == 4
    assert item.options[item.gold_index].label == "Hypertension"
    distractors = [o.label for o in item.options if o.label != "Hypertension"]
    assert set(distractors) <= set(SIBLINGS["Hypertension"])
    assert item.meta["distractor_rule"] == "mesh_sibling"
    assert item.meta["mesh"]["year"] == 2026
    assert "sibling" in notes[0]


def test_mesh_v2_never_offers_another_correct_topic_as_a_distractor() -> None:
    """A record with two major topics: *both* are correct, so neither may be a distractor."""
    records = [_record("2", ["Hypertension", "Blood Pressure"])]
    rows, _drops, _notes = pubmed.build_mesh_major_v2_items(
        records, cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1],
        mesh_siblings=lambda name: SIBLINGS.get(name, []),
    )
    item = rows[0]
    offered = {o.label for o in item.options}
    assert "Blood Pressure" not in offered  # the record's second correct topic
    assert item.meta["gold_topic"] == "Hypertension"
    # and no sibling of that second correct topic either
    assert not (offered & set(SIBLINGS["Blood Pressure"]) - {"Hypertension"})


def test_mesh_v2_falls_back_to_the_window_pool_and_records_the_rule() -> None:
    records = [_record("3", ["Diabetes Mellitus, Type 2"])]
    # a window pool big enough to fill the two missing distractors
    pool_records = [_record(str(100 + i), ["Topic A" if i < 5 else "Topic B"]) for i in range(10)]
    rows, _drops, _notes = pubmed.build_mesh_major_v2_items(
        [*records, *pool_records], cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1],
        mesh_siblings=lambda name: SIBLINGS.get(name, []),
    )
    item = next(r for r in rows if r.source_record_id == "3")
    assert item.meta["distractor_rule"] == "mesh_sibling_partial+window_pool_fill"
    assert item.meta["n_siblings_available"] == 1
    assert item.options[item.gold_index].label == "Diabetes Mellitus, Type 2"


def test_mesh_v2_without_an_index_drops_every_record_with_a_reason() -> None:
    rows, drops, _notes = pubmed.build_mesh_major_v2_items(
        [_record("4", ["Hypertension"])], cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1],
        mesh_siblings=None,
    )
    assert rows == []
    assert drops["mesh_index_unavailable"] == 1


def test_mesh_v2_is_deterministic() -> None:
    records = [_record(str(i), ["Hypertension"]) for i in range(6)]
    kwargs = {
        "cfg": CFG, "window_start": WINDOW[0], "window_end": WINDOW[1],
        "mesh_siblings": lambda name: SIBLINGS.get(name, []),
    }
    first, _d1, _n1 = pubmed.build_mesh_major_v2_items(records, **kwargs)
    second, _d2, _n2 = pubmed.build_mesh_major_v2_items(records, **kwargs)
    assert [i.item_id for i in first] == [i.item_id for i in second]
    assert [i.gold for i in first] == [i.gold for i in second]
    assert [[o.label for o in i.options] for i in first] == [
        [o.label for o in i.options] for i in second
    ]


# ---------------------------------------------------------------------------
# openFDA
# ---------------------------------------------------------------------------
def _label(set_id: str, classes: list[str]) -> dict:
    return {
        "set_id": set_id,
        "effective_time": "20260415",
        "openfda": {"pharm_class_epc": classes, "brand_name": ["Synthetic"]},
        "indications_and_usage": [
            "Used for the synthetic indication in adults with the synthetic condition, once daily."
        ],
        "mechanism_of_action": [
            "It inhibits the synthetic target, reducing the synthetic mediator in a dose-dependent way."
        ],
    }


def _near_miss(candidates: list[dict]):
    def lookup(gold_class: str, exclude: set[str]) -> list[dict]:
        return [c for c in candidates if c["class"] not in exclude and c["class"] != gold_class]

    return lookup


def test_class_v2_distractors_come_from_the_near_miss_lookup() -> None:
    labels = [_label("s1", ["Penicillin-class Antibacterial [EPC]"])]
    candidates = [
        {"class": "Cephalosporin-class Antibacterial [EPC]", "rule": "shared_moa", "shared": ["m"]},
        {"class": "Beta-Lactamase Inhibitors [EPC]", "rule": "shared_pe", "shared": ["p"]},
        {"class": "Aminoglycoside-class Antibacterial [EPC]", "rule": "same_route", "shared": ["ORAL"]},
        {"class": "Unrelated Class [EPC]", "rule": "same_route", "shared": ["ORAL"]},
    ]
    rows, drops, _notes = openfda.build_class_v2_items(
        labels, cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1],
        near_miss=_near_miss(candidates),
    )
    assert len(rows) == 1
    assert drops.get("mesh_index_unavailable", 0) == 0
    item = rows[0]
    assert item.template_id == "fda_class_choice_v2"
    assert item.options[item.gold_index].label == "Penicillin-class Antibacterial [EPC]"
    assert item.meta["distractor_rule"] == "near_miss"
    assert item.meta["distractor_rules"] == ["shared_moa", "shared_pe", "same_route"]
    assert all(o.label != "Unrelated Class [EPC]" for o in item.options)


def test_class_v2_never_offers_another_class_of_the_same_label() -> None:
    labels = [
        _label("s2", ["Penicillin-class Antibacterial [EPC]", "Beta-Lactamase Inhibitors [EPC]"])
    ]
    candidates = [
        {"class": "Beta-Lactamase Inhibitors [EPC]", "rule": "shared_moa", "shared": ["m"]},
        {"class": "Cephalosporin-class Antibacterial [EPC]", "rule": "shared_moa", "shared": ["m"]},
        {"class": "Other Antibacterial [EPC]", "rule": "shared_pe", "shared": ["p"]},
        {"class": "Third Antibacterial [EPC]", "rule": "same_route", "shared": ["ORAL"]},
    ]
    rows, _drops, _notes = openfda.build_class_v2_items(
        labels, cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1],
        near_miss=_near_miss(candidates),
    )
    offered = {o.label for o in rows[0].options}
    assert "Beta-Lactamase Inhibitors [EPC]" not in offered  # also correct for this label


def test_class_v2_falls_back_to_the_pool_and_records_it() -> None:
    labels = [_label("s3", ["Only Class [EPC]"])]
    rows, _drops, _notes = openfda.build_class_v2_items(
        labels, cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1],
        near_miss=_near_miss([{"class": "Sibling [EPC]", "rule": "shared_moa", "shared": ["m"]}]),
        fallback_pool=["Pool A [EPC]", "Pool B [EPC]", "Pool C [EPC]"],
    )
    item = rows[0]
    assert item.meta["distractor_rule"] == "near_miss+pool_fill"
    assert item.meta["distractor_rules"] == ["shared_moa", "openfda_pool_fallback",
                                            "openfda_pool_fallback"]


def test_class_v2_drops_a_label_with_too_few_safe_distractors() -> None:
    labels = [_label("s4", ["Lonely Class [EPC]"])]
    rows, drops, _notes = openfda.build_class_v2_items(
        labels, cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1],
        near_miss=_near_miss([]),
    )
    assert rows == []
    assert drops["fewer_than_3_near_miss_distractors"] == 1


def test_class_v2_skips_labels_whose_set_id_predates_the_window() -> None:
    labels = [_label("old", ["Class [EPC]"])]
    rows, drops, _notes = openfda.build_class_v2_items(
        labels, cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1],
        near_miss=_near_miss([]), set_ids_seen_before={"old"},
    )
    assert rows == [] and drops["set_id_existed_before_window"] == 1


def test_class_v2_items_are_choice_with_four_options() -> None:
    labels = [_label(f"s{i}", ["Penicillin-class Antibacterial [EPC]"]) for i in range(5)]
    candidates = [
        {"class": f"Near {i} [EPC]", "rule": "shared_moa", "shared": ["m"]} for i in range(4)
    ]
    rows, _drops, _notes = openfda.build_class_v2_items(
        labels, cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1],
        near_miss=_near_miss(candidates),
    )
    assert len(rows) == 5
    for item in rows:
        assert item.qtype is QuestionType.CHOICE
        assert item.n_options == 4
        assert {o.key for o in item.options} == {"A", "B", "C", "D"}
