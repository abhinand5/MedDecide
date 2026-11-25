"""Unit tests for the osler_v0 O3 clinical generators (D20).

Labels are recomputed from the structured parts, each twin flips the gold label, and the screen's checks pass or fail
as intended on small hand-made cases. The split tests read the built data and are skipped when it is not built.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meddecide.gen.core import LABS, Lab, Patient, Problem, render_note, rng_for, split_of
from meddecide.gen.families import (
    CHOICE_LETTERS,
    FAMILIES,
    GENERATORS,
    HELD_OUT,
    _condition_code,
    _triage_level,
)
from meddecide.gen.screen import (
    NaiveBayes,
    gold_in_state,
    macro_accuracy,
    record_text,
    screen_generator,
    string_presence_predict,
)

DATA = Path(__file__).resolve().parents[1] / "data/gen/osler_v0"
RANGES = {name: (lo, hi) for name, _unit, lo, hi, _span in LABS}
ALPHA_BETA = [{"key": "A", "label": "Alpha"}, {"key": "B", "label": "Beta"}]


def _items(name: str) -> list:
    return [it for k in range(40) for it in GENERATORS[name].make(f"{name}-{k:06d}")]


def _rows(split: str) -> list[dict]:
    path = DATA / f"{split}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} is not built")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_registry_has_nine_generators_and_the_fixed_held_out_pair() -> None:
    assert len(FAMILIES) == 9
    assert HELD_OUT == ("note_lab_range_v1", "policy_triage_v1")
    assert sorted(f.name for f in FAMILIES if f.held_out) == sorted(HELD_OUT)


@pytest.mark.parametrize("name", list(GENERATORS))
def test_each_patient_gives_a_base_and_a_twin_whose_gold_differs(name: str) -> None:
    items = _items(name)
    assert len(items) == 80
    assert len({it.item_id for it in items}) == 80
    for base, twin in zip(items[::2], items[1::2], strict=True):
        assert (base.role, twin.role) == ("base", "twin")
        assert base.pair_id == twin.pair_id and base.split == twin.split
        assert base.gold != twin.gold, (name, base.pair_id)


def test_negation_gold_comes_from_the_affirmed_flag() -> None:
    for it in _items("note_negation_v1"):
        assert it.gold == ("yes" if it.meta["affirmed"] else "no")


def test_subject_gold_comes_from_the_subject() -> None:
    for it in _items("note_subject_v1"):
        assert it.gold == ("yes" if it.meta["subject"] == "patient" else "no")


def test_timing_gold_is_the_two_year_rule_on_the_recorded_years() -> None:
    for it in _items("note_timing_v2"):
        gap = it.meta["visit"] - it.meta["episode"]
        assert it.meta["active"] == (gap <= 2)
        assert it.gold == ("yes" if gap <= 2 else "no")


def test_allergy_gold_is_whether_the_drug_is_on_the_medication_list() -> None:
    for it in _items("note_allergy_med_v1"):
        assert it.gold == ("yes" if it.meta["on_medication_list"] else "no")
        assert it.meta["drug"] in it.state.lower()  # the allergy template capitalises the drug at line start


def test_lab_range_gold_follows_the_value_and_the_reference_range() -> None:
    letter = {"within": "A", "below": "B", "above": "C"}
    for it in _items("note_lab_range_v1"):
        lo, hi = RANGES[it.meta["lab"]]
        value = it.meta["value"]
        side = "within" if lo <= value <= hi else ("below" if value < lo else "above")
        assert it.meta["side"] == side
        assert it.gold == letter[side]
        assert f"{value:g}" in it.state


def test_criterion_gold_is_met_not_met_or_not_enough_information() -> None:
    for it in _items("criterion_eligibility_v1"):
        value, thr = it.meta["value"], it.meta["threshold"]
        assert it.gold == ("C" if value is None else ("A" if value >= thr else "B"))
        assert "Trial criterion:" in it.state


def test_triage_levels_follow_the_written_policy() -> None:
    assert _triage_level(83, chest_pain=False, sweating=False) == 3   # below 90
    assert _triage_level(95, chest_pain=True, sweating=True) == 3     # chest pain with sweating
    assert _triage_level(120, chest_pain=True, sweating=False) == 2   # 90 to 139
    assert _triage_level(185, chest_pain=False, sweating=False) == 2  # 180 or above
    assert _triage_level(155, chest_pain=False, sweating=False) == 1  # otherwise
    for it in _items("policy_triage_v1"):
        m = it.meta
        assert it.gold == str(_triage_level(m["systolic_bp"], m["chest_pain"], m["sweating"]))


@pytest.mark.parametrize("name", ["code_assignment_v1", "code_family_v1"])
def test_coding_gold_is_the_target_code_or_none_of_these(name: str) -> None:
    for it in _items(name):
        labels = dict(it.options)
        present = it.meta["affirmed"] if name == "code_assignment_v1" else it.meta["subject"] == "patient"
        if present:
            assert labels[it.gold] == _condition_code(it.meta["target"])
        else:
            assert it.gold == "E" and labels["E"] == "none of these"


@pytest.mark.parametrize("name", ["code_assignment_v1", "code_family_v1"])
def test_coding_items_list_four_codes_then_none_of_these(name: str) -> None:
    for it in _items(name):
        assert [k for k, _ in it.options] == [*CHOICE_LETTERS[:4], "E"]
        assert it.options[-1][1] == "none of these"
        assert it.state.startswith("Code list: ")


def test_built_splits_are_patient_disjoint_and_keep_held_out_generators_out_of_training() -> None:
    rows = {s: _rows(s) for s in ("train", "dev", "test")}
    patients = {s: {r["patient_id"] for r in rows[s]} for s in rows}
    assert not patients["train"] & patients["dev"]
    assert not patients["train"] & patients["test"]
    assert not patients["dev"] & patients["test"]
    for s, split_rows in rows.items():
        assert all(r["split"] == s and split_of(r["patient_id"]) == s for r in split_rows)
    assert not any(r["held_out"] or r["generator"] in HELD_OUT for r in rows["train"])
    assert set(HELD_OUT) <= {r["generator"] for r in rows["dev"]}
    assert set(HELD_OUT) <= {r["generator"] for r in rows["test"]}


def test_render_note_is_deterministic_and_has_the_sections() -> None:
    p = Patient(pid="x", age=60, sex="female", problems=(Problem("asthma", "J45", affirmed=False),),
                labs=(Lab("haemoglobin", "g/dL", 12.0, 16.0, 13.5),))
    a = render_note(p, rng_for("t", "x", "render"))
    assert a == render_note(p, rng_for("t", "x", "render"))
    assert a.startswith("HPI:") and "Past medical history:" in a and "Results:" in a and "13.5" in a


def test_record_text_drops_fixed_blocks_and_gold_in_state_flags_a_leak() -> None:
    state = "Code list: A) E11 type 2 diabetes mellitus; B) I10 essential hypertension.\n\nHPI: patient seen."
    text = record_text(state)
    assert "e11" not in text and "patient seen" in text
    leaky = {"qtype": "choice", "state": "Summary: the answer is Meets the criterion.", "gold": "A",
             "question": "q", "options": [{"key": "A", "label": "Meets the criterion"},
                                           {"key": "B", "label": "Does not meet"}]}
    assert gold_in_state(leaky)
    assert not gold_in_state({**leaky, "state": "Summary: values reviewed."})
    yes_no = {"qtype": "noul", "state": "no known allergies", "gold": "no", "question": "q",
              "options": [{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}]}
    assert not gold_in_state(yes_no)


def test_string_presence_picks_the_one_offered_label_that_occurs_in_the_record() -> None:
    item = {"state": "Results: the note says level 2 applies.", "question": "q", "qtype": "score",
            "options": [{"key": "1", "label": "level 1"}, {"key": "2", "label": "level 2"},
                        {"key": "3", "label": "level 3"}]}
    assert string_presence_predict(item, majority="level 1") == "level 2"


def test_naive_bayes_separates_a_token_cue_and_macro_is_per_class() -> None:
    def row(state: str, gold: str) -> dict:
        return {"state": state, "question": "q", "qtype": "choice", "gold": gold, "options": ALPHA_BETA}

    train = [row("alpha alpha signal", "A")] * 5 + [row("beta beta signal", "B")] * 5
    nb = NaiveBayes().fit(train)
    assert nb.predict(row("alpha signal", "A")) == "Alpha"
    assert nb.predict(row("beta signal", "B")) == "Beta"
    assert macro_accuracy([("a", "a"), ("a", "b"), ("b", "b")]) == pytest.approx(0.75)


def test_screen_fails_a_generator_whose_gold_label_is_in_the_record() -> None:
    def row(pid: str, state: str, gold: str) -> dict:
        return {"pair_id": pid, "state": state, "question": "Which?", "qtype": "choice", "gold": gold,
                "options": ALPHA_BETA}

    dev = [row("p1", "Note: alpha finding.", "A"), row("p2", "Note: beta finding.", "B")]
    train = [row("t1", "Note: alpha finding.", "A"), row("t2", "Note: beta finding.", "B")]
    rep = screen_generator("toy", False, dev, train, "own train split")
    assert rep.gold_in_state_hits == 2
    assert not rep.passes
    assert any("gold label appears" in f for f in rep.failures)


def test_screen_passes_a_generator_with_no_label_text_and_no_separable_tokens() -> None:
    rows = [{"pair_id": f"p{k}", "state": "Note: finding recorded.", "question": "Which?", "qtype": "choice",
             "gold": "A" if k % 2 == 0 else "B", "options": ALPHA_BETA} for k in range(20)]
    rep = screen_generator("toy", False, rows, rows, "own train split")
    assert rep.gold_in_state_hits == 0
    assert rep.naive_bayes_macro == pytest.approx(0.5)
    assert rep.passes, rep.failures
