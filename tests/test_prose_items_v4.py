"""The teacher-prose items of osler_v1 (ADVISORY V4): the decision is the generator's, the page is verified, and a faithful
teacher and a wrong one are told apart. A stub stands in for the endpoint; nothing here touches the network."""

from __future__ import annotations

import json
import random

import pytest

from meddecide.mr import icd
from meddecide.mr.criteria import supports
from meddecide.mr.generators import indicator_pair, is_held_out, mr_icd_support_template_v2
from meddecide.mr.prose_build import bucket, process_icd, process_indicator
from meddecide.mr.prose_items import (
    ICD_PAGE_CLASSES,
    dx_from_meta,
    has_code_string,
    icd_page_class,
    icd_prose_item,
    icd_request,
    indicator_prose_item,
    indicator_request,
)
from meddecide.mr.teacher_prose import WRITE_SYSTEM, ProseRequest, generate


class StubTeacher:
    """Writes a page that carries no code, then extracts the facts back: faithfully, or with one field corrupted."""

    local = False

    def __init__(self, corrupt: str | None = None) -> None:
        self.corrupt = corrupt
        self.last: dict | None = None
        self.calls = 0

    def chat(self, messages, *, temperature, max_tokens):
        self.calls += 1
        if messages[0]["content"] == WRITE_SYSTEM:
            self.last = json.loads(messages[1]["content"].split("FACTS:\n", 1)[1])
            return "Assessment and plan, as documented on the day of the encounter."
        facts = self.last
        assert facts is not None
        if "diagnoses" in facts:  # several diagnoses on one page: every entry comes back, the first one corrupted on request
            entries = []
            for i, d in enumerate(facts["diagnoses"]):
                certainty = d["certainty"]
                if self.corrupt == "certainty" and i == 0:
                    certainty = "probable" if certainty == "confirmed" else "confirmed"
                entries.append({"condition": d["condition"], "attributes": d["attributes"], "certainty": certainty,
                                "subject": d["subject"], "status": d["status"]})
            return json.dumps({"diagnoses": entries})
        if "condition" in facts:  # a diagnosis
            certainty = facts["certainty"]
            if self.corrupt == "certainty":
                certainty = "probable" if certainty == "confirmed" else "confirmed"
            return json.dumps({"condition": facts["condition"], "attributes": facts["attributes"],
                               "certainty": certainty, "subject": facts["subject"], "status": facts["status"]})
        observations = [{"name": o["name"], "value": o["value"], "encounter": o["encounter"],
                         "baseline": o["baseline"], "on_room_air": o["on_room_air"]} for o in facts["observations"]]
        if self.corrupt == "value" and observations:
            observations[0]["value"] = observations[0]["value"] * 1.5
        meds = [{"drug": m["drug"], "status": m["status"], "encounter": m["encounter"]} for m in facts["medications"]]
        return json.dumps({"observations": observations, "medications": meds})


def _first_pid(prefix: str, pick, *, start: int = 0) -> str:
    for k in range(start, start + 500):
        pid = f"{prefix}-{k:07d}"
        if pick(pid):
            return pid
    raise AssertionError("no suitable patient id in 500 tries")


def test_code_strings_are_caught_and_prose_is_not() -> None:
    assert has_code_string("Impression: acute respiratory failure with hypoxia, J96.01.")
    assert has_code_string("Coded as E11.65 on this admission")
    assert not has_code_string("Assessment: acute on chronic kidney disease, improving. SpO2 92% on room air.")


def test_icd_page_classes_follow_the_setting() -> None:
    rng = random.Random(0)
    outpatient = {icd_page_class(rng, "outpatient") for _ in range(200)}
    inpatient = {icd_page_class(rng, "inpatient") for _ in range(200)}
    assert outpatient <= {"progress_note", "consult_note"}
    assert inpatient <= set(ICD_PAGE_CLASSES["inpatient"]) and inpatient == set(ICD_PAGE_CLASSES["inpatient"])
    with pytest.raises(ValueError):
        icd_page_class(rng, "emergency")


def test_the_diagnosis_rebuilt_from_a_template_item_gives_the_template_gold() -> None:
    checked = 0
    for k in range(40):
        for item in mr_icd_support_template_v2(f"round-trip-{k:04d}"):
            dx = dx_from_meta(item.meta)
            assert icd.supported_code(dx, item.meta["setting"]) == item.meta["code"]
            checked += 1
    assert checked == 80


def test_a_prose_icd_item_keeps_the_template_decision() -> None:
    template = mr_icd_support_template_v2("keep-decision-0000001")[0]
    item = icd_prose_item(template, "mr_icd_support_prose_v1-x", "Discharge summary text.", "discharge_summary")
    assert item.generator == "mr_icd_support_prose_v1"
    assert item.options == template.options and item.gold == template.gold and item.question == template.question
    assert item.state == "Discharge summary text."
    assert item.meta["source"] == "teacher_prose" and item.meta["template_item"] == template.item_id
    assert item.meta["code"] == template.meta["code"]


def test_an_indicator_prose_item_is_labelled_by_the_criteria_rule() -> None:
    pid = "indicator-gold-0000001"
    pair = indicator_pair(pid)
    for role, facts in pair.order:
        item = indicator_prose_item(pid, role, facts, pair.indicator, pair.trap, pair.question, "Progress note text.")
        assert item.gold == ("yes" if supports(pair.indicator, facts) else "no")
        assert item.generator == "mr_indicator_prose_v1" and item.qtype == "noul"


def test_the_request_builders_describe_the_facts_and_the_setting() -> None:
    template = mr_icd_support_template_v2("request-0000001")[0]
    dx = dx_from_meta(template.meta)
    req = icd_request(dx, "outpatient", "progress_note")
    assert isinstance(req, ProseRequest)
    assert req.kind == "diagnosis" and req.setting == "outpatient" and req.page_class == "progress_note"
    assert req.facts["family"] == dx.family and req.facts["certainty"] == dx.certainty
    pair = indicator_pair("request-indicator-0000001")
    chunk = indicator_request(pair.order[0][1])
    assert chunk.kind == "chunk" and chunk.page_class == "progress_note"


def test_a_faithful_teacher_passes_both_twins_of_an_icd_pair() -> None:
    pid = _first_pid("mr_icd_support_prose_v1", lambda p: not is_held_out(mr_icd_support_template_v2(p)[0].meta))
    stub = StubTeacher()
    outcome = process_icd(pid, stub)
    assert outcome.verdict == "accepted", outcome.reasons
    assert len(outcome.items) == 2 and {i.role for i in outcome.items} == {"base", "twin"}
    assert stub.calls == 4  # a write and an extraction for each twin
    assert all(i.state == "Assessment and plan, as documented on the day of the encounter." for i in outcome.items)


def test_a_teacher_that_changes_a_certainty_is_rejected_with_its_reason() -> None:
    pid = _first_pid("mr_icd_support_prose_v1", lambda p: not is_held_out(mr_icd_support_template_v2(p)[0].meta))
    outcome = process_icd(pid, StubTeacher(corrupt="certainty"))
    assert outcome.verdict == "rejected" and outcome.items == []
    assert outcome.reasons == ["certainty"]


def test_held_out_families_are_skipped_without_a_teacher_call() -> None:
    pid = _first_pid("mr_icd_support_prose_v1", lambda p: is_held_out(mr_icd_support_template_v2(p)[0].meta))
    stub = StubTeacher()
    outcome = process_icd(pid, stub)
    assert outcome.verdict == "skipped" and stub.calls == 0


def test_a_faithful_teacher_passes_an_indicator_pair_and_a_changed_value_fails() -> None:
    pid = _first_pid("mr_indicator_prose_v1", lambda p: indicator_pair(p).indicator != "aki_creatinine_rise")
    ok = process_indicator(pid, StubTeacher())
    assert ok.verdict == "accepted", ok.reasons
    assert {i.gold for i in ok.items} == {"yes", "no"}  # a flipping twin: the two golds differ
    bad = process_indicator(pid, StubTeacher(corrupt="value"))
    assert bad.verdict == "rejected"
    assert any(r in {"observation_value", "observation_count"} for r in bad.reasons), bad.reasons


def test_reason_buckets_hide_the_text_of_each_problem() -> None:
    assert bucket("attribute acuity: 'acute' != 'chronic'") == "attribute"
    assert bucket("certainty: 'probable' != 'confirmed'") == "certainty"
    assert bucket("missing or changed observation lactate=4 (current, baseline=False, room_air=True)") == "observation_value"
    assert bucket("write failed: HTTPStatusError") == "write failed"
    assert bucket("condition not reported: pneumonia") == "condition_not_reported"


def test_generate_refuses_credentialed_facts_for_a_hosted_teacher() -> None:
    req = ProseRequest(kind="chunk", page_class="progress_note", facts={"observations": [], "medications": []},
                       source="mimic-iii")
    with pytest.raises(PermissionError):
        generate(StubTeacher(), req)
