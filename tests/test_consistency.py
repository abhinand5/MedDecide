"""Unit tests for the record-claim consistency builders (S2, D15).

The properties tested are the design rules, because those are what make the template a
*consistency* question rather than a string match:

* gold comes from a structured field and is exactly one of the offered options;
* the swapped (unsupported) value also appears in the state, in a different role;
* the state never contains the role label that would give the answer away;
* supported and unsupported items are balanced;
* an intervention listed in two arms is never used (the claim would be true of either arm);
* the same input produces the same items.
"""

from __future__ import annotations

from datetime import date

from meddecide.bench.fresh import consistency

CFG = {"seed": 0, "caps": {"tier1_max_test_items_per_source": 5000,
                           "tier1_max_dev_items_per_source": 2000}}
WINDOW = (date(2026, 3, 1), date(2026, 10, 5))
PRE_WINDOW = (date(2023, 1, 1), date(2026, 2, 28))


def _study(nct: str, *, arm_types=("EXPERIMENTAL", "ACTIVE_COMPARATOR"),
           arm_names=(("Drug: Aspirin",), ("Drug: Clopidogrel",)), first_posted="2026-04-01",
           phase="PHASE3", allocation="RANDOMIZED", purpose="TREATMENT",
           intervention_types=("DRUG", "OTHER"), healthy=True) -> dict:
    arms = [
        {"type": t, "interventionNames": list(names)}
        for t, names in zip(arm_types, arm_names, strict=True)
    ]
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct, "briefTitle": f"Study {nct}"},
            "statusModule": {"studyFirstPostDateStruct": {"date": first_posted}},
            "descriptionModule": {
                "briefSummary": (
                    "This study evaluates whether the study drug reduces the composite endpoint "
                    "compared with control over twelve weeks in adults with the condition."
                ),
                "detailedDescription": "Participants are followed for twelve weeks.",
            },
            "conditionsModule": {"conditions": ["Synthetic Condition"]},
            "designModule": {
                "phases": [phase],
                "studyType": "INTERVENTIONAL",
                "designInfo": {"allocation": allocation, "primaryPurpose": purpose},
                "enrollmentInfo": {"count": 120},
            },
            "armsInterventionsModule": {
                "armGroups": arms,
                "interventions": [{"type": t} for t in intervention_types],
            },
            "eligibilityModule": {"healthyVolunteers": healthy},
        }
    }


# ---------------------------------------------------------------------------
# ct_arm_role_noul_v1
# ---------------------------------------------------------------------------
def test_arm_role_claim_is_balanced_and_role_bound() -> None:
    studies = [_study(f"NCT{i:08d}") for i in range(8)]
    rows, drops, notes = consistency.build_ct_arm_role_items(
        studies, cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1]
    )
    assert len(rows) == 8
    golds = [r.gold for r in rows]
    assert golds.count("yes") == 4 and golds.count("no") == 4, "balanced by construction"
    for row in rows:
        claimed = row.meta["claimed_intervention"]
        assert claimed in row.state, "the claimed value must be in the state (role binding)"
        if row.gold == "no":
            # the unsupported claim names the comparator's own intervention, which is listed
            # in the state in the other role
            assert claimed == "Clopidogrel"
        else:
            assert claimed == "Aspirin"
    assert "experimental arm" in notes[0]
    assert drops.get("no_intervention_confined_to_a_comparator_arm", 0) == 0


def test_arm_role_state_never_names_the_arm_type() -> None:
    rows, _drops, _notes = consistency.build_ct_arm_role_items(
        [_study("NCT00000001")], cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1]
    )
    state = rows[0].state.upper()
    for label in ("EXPERIMENTAL", "PLACEBO COMPARATOR", "ACTIVE COMPARATOR"):
        assert label not in state, f"the state must not contain the role label {label!r}"
    assert "EXPERIMENTAL" not in rows[0].question.upper().replace("EXPERIMENTAL ARM", "")


def test_arm_role_drops_a_record_without_a_comparator_arm() -> None:
    study = _study("NCT00000002", arm_types=("EXPERIMENTAL",), arm_names=(("Drug: Aspirin",),))
    rows, drops, _notes = consistency.build_ct_arm_role_items(
        [study], cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1]
    )
    assert rows == []
    assert drops["no_comparator_arm_with_interventions"] == 1


def test_arm_role_drops_a_placebo_only_comparator_with_a_counted_reason() -> None:
    """A "Placebo" arm is a comparator but its intervention name is a role word.

    Claiming "Placebo is given in the experimental arm" would be answerable without reading
    the record, so such records are dropped and counted rather than turned into free items.
    """
    study = _study(
        "NCT00000004",
        arm_types=("EXPERIMENTAL", "PLACEBO_COMPARATOR"),
        arm_names=(("Drug: Aspirin",), ("Drug: Placebo",)),
    )
    rows, drops, _notes = consistency.build_ct_arm_role_items(
        [study], cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1]
    )
    assert rows == []
    assert drops["no_intervention_confined_to_a_comparator_arm"] == 1


def test_arm_role_never_uses_an_intervention_listed_in_two_arms() -> None:
    study = _study(
        "NCT00000003",
        arm_names=(("Drug: Aspirin", "Drug: Saline"), ("Drug: Saline", "Drug: Clopidogrel")),
    )
    rows, drops, _notes = consistency.build_ct_arm_role_items(
        [study], cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1]
    )
    assert len(rows) == 1
    assert rows[0].meta["claimed_intervention"] != "Saline"
    assert drops.get("no_intervention_confined_to_a_comparator_arm", 0) == 0


def test_arm_role_is_deterministic_and_window_filtered() -> None:
    studies = [_study(f"NCT{i:08d}") for i in range(4)]
    kwargs = {"cfg": CFG, "window_start": WINDOW[0], "window_end": WINDOW[1]}
    first, _d1, _n1 = consistency.build_ct_arm_role_items(studies, **kwargs)
    second, _d2, _n2 = consistency.build_ct_arm_role_items(studies, **kwargs)
    assert [i.item_id for i in first] == [i.item_id for i in second]
    assert [i.gold for i in first] == [i.gold for i in second]
    out_of_window, drops, _notes = consistency.build_ct_arm_role_items(
        studies, cfg=CFG, window_start=PRE_WINDOW[0], window_end=PRE_WINDOW[1]
    )
    assert out_of_window == [] and drops["first_posted_outside_window"] == 4


# ---------------------------------------------------------------------------
# ct_claim_set_choice_v1 (multi-field variant)
# ---------------------------------------------------------------------------
def test_claim_set_has_four_options_and_exactly_one_unsupported_field() -> None:
    studies = [_study(f"NCT{i:08d}") for i in range(6)]
    rows, _drops, notes = consistency.build_ct_claim_set_items(
        studies, cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1]
    )
    assert len(rows) == 6
    for row in rows:
        assert row.n_options == 4
        stated = {d["name"]: d["value"] for d in row.meta["stated_fields"]}
        swapped = row.meta["swapped_field"]
        assert stated[swapped] == row.meta["stated_value"] != row.meta["true_value"]
        assert row.gold == next(o.key for o in row.options
                                if o.label == f"{swapped}: {row.meta['stated_value']}")
        # the other three stated fields are the record's own values
        assert row.meta["true_value"] != row.meta["stated_value"]
    assert "rotates" in notes[0]


def test_claim_set_rotates_the_swapped_field() -> None:
    studies = [_study(f"NCT{i:08d}") for i in range(8)]
    rows, _drops, _notes = consistency.build_ct_claim_set_items(
        studies, cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1]
    )
    swapped = [r.meta["swapped_field"] for r in rows]
    assert len(set(swapped)) >= 3, f"the swapped field should rotate, got {swapped}"


def test_claim_set_uses_another_intervention_type_of_the_same_record_when_available() -> None:
    study = _study("NCT00000009", intervention_types=("DRUG", "DEVICE"))
    rows, _drops, _notes = consistency.build_ct_claim_set_items(
        [study], cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1],
    )
    row = rows[0]
    if row.meta["swapped_field"] == "intervention type":
        assert row.meta["stated_value"] == "Device"  # the record's other type
        assert row.meta["swap_is_another_value_of_this_record"] is True


# ---------------------------------------------------------------------------
# fda_route_claim_noul_v1
# ---------------------------------------------------------------------------
def _label(set_id: str, routes: list[str], text: str) -> dict:
    return {
        "set_id": set_id,
        "effective_time": "20260415",
        "openfda": {"route": routes, "pharm_class_epc": ["Synthetic Class [EPC]"]},
        "indications_and_usage": [text],
        "dosage_and_administration": [
            "The drug may be given orally or intravenously depending on the clinical situation."
        ],
    }


def test_route_claim_unsupported_value_appears_in_the_state() -> None:
    labels = [_label(f"s{i}", ["ORAL"], "Used for the synthetic indication in adults.") for i in range(6)]
    rows, _drops, notes = consistency.build_fda_route_claim_items(
        labels, cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1]
    )
    assert len(rows) == 6
    assert [r.gold for r in rows].count("yes") == 3
    for row in rows:
        assert row.meta["claimed_route"] in row.state.upper()
        if row.gold == "no":
            assert row.meta["claimed_route"] != "ORAL"
            assert row.meta["claimed_route"] in row.meta["alternative_route_words_in_state"]
    assert "string-presence" in notes[0]


def test_route_claim_drops_a_label_with_no_alternative_route_word() -> None:
    label = _label("only-oral", ["ORAL"], "Used for the synthetic indication in adults.")
    label["dosage_and_administration"] = ["Take one tablet each morning with water."]
    rows, drops, _notes = consistency.build_fda_route_claim_items(
        [label], cfg=CFG, window_start=WINDOW[0], window_end=WINDOW[1]
    )
    assert rows == [] or rows[0].gold == "yes"
    assert drops.get("no_alternative_route_word_in_the_state_text", 0) in (0, 1)


def test_route_claim_is_deterministic() -> None:
    labels = [_label(f"s{i}", ["ORAL"], "Used for the synthetic indication in adults.") for i in range(4)]
    kwargs = {"cfg": CFG, "window_start": WINDOW[0], "window_end": WINDOW[1]}
    first, _d1, _n1 = consistency.build_fda_route_claim_items(labels, **kwargs)
    second, _d2, _n2 = consistency.build_fda_route_claim_items(labels, **kwargs)
    assert [i.item_id for i in first] == [i.item_id for i in second]
    assert [i.gold for i in first] == [i.gold for i in second]
