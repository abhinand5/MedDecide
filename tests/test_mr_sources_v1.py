"""osler_v1 pre-MIMIC sources: Synthea encounters, MTSamples, CodiEsp, PMC-Patients, and whole-page teacher prose
(CPU, tiny fixtures; no network)."""

from __future__ import annotations

import csv

from meddecide.mr import criteria, realtext, synthea
from meddecide.mr.icd import OrderRow
from meddecide.mr.teacher_prose import ProseRequest, verify


def _encounter(**obs_values) -> synthea.SyntheaEncounter:
    enc = {"Id": "e1", "PATIENT": "p1", "ENCOUNTERCLASS": "emergency", "START": "2020-01-01T10:00:00Z",
           "DESCRIPTION": "Emergency room admission (procedure)", "REASONDESCRIPTION": "Sepsis (disorder)",
           "BASE_ENCOUNTER_COST": 120.0}
    patient = {"BIRTHDATE": "1950-06-01", "GENDER": "F"}
    codes = {"lactate": "2524-7", "sodium": "2947-0", "heart_rate": "8867-4", "temperature": "8310-5"}
    units = {"lactate": "mmol/L", "sodium": "mmol/L", "heart_rate": "/min", "temperature": "Cel"}
    obs = [{"DATE": "2020-01-01T11:00:00Z", "CODE": codes[k], "VALUE": str(v), "UNITS": units[k]}
           for k, v in obs_values.items()]
    obs += [{"DATE": "2020-01-01T11:00:00Z", "CODE": "8480-6", "VALUE": "80", "UNITS": "mm[Hg]"},
            {"DATE": "2020-01-01T11:00:00Z", "CODE": "8462-4", "VALUE": "50", "UNITS": "mm[Hg]"},
            {"DATE": "2020-01-01T11:00:00Z", "CODE": "2708-6", "VALUE": "85", "UNITS": "%"},     # SaO2: not mapped
            {"DATE": "2020-01-01T11:00:00Z", "CODE": "2947-0", "VALUE": "130", "UNITS": "mg/dL"}]  # wrong unit
    meds = [{"START": "2020-01-01T12:00:00Z", "CODE": 1, "DESCRIPTION": "norepinephrine 1 MG/ML Injectable Solution",
             "BASE_COST": 10.0, "DISPENSES": 1}]
    procs = [{"START": "2020-01-01T10:30:00Z", "CODE": 123456, "DESCRIPTION": "Triage (procedure)", "BASE_COST": 50.0}]
    conds = [{"DESCRIPTION": "Sepsis (disorder)"}]
    counts: dict[str, int] = {}
    se = synthea.build_encounter(enc, patient, obs, meds, procs, conds, counts)
    se.meta["counts"] = counts
    return se


def test_synthea_encounter_maps_codes_units_and_map() -> None:
    se = _encounter(lactate=3.1, sodium=131, heart_rate=110, temperature=38.6)
    names = sorted(o.name for o in se.observations)
    assert names == ["heart_rate", "lactate", "map", "sodium", "temperature"]
    mapo = next(o for o in se.observations if o.name == "map")
    assert abs(mapo.value - 60.0) < 1e-9 and mapo.time_h == 1.0
    assert se.meta["counts"]["skipped: sodium unit 'mg/dL'"] == 1
    assert se.age == 69 and se.sex == "female" and se.conditions == ("Sepsis",)
    assert se.meds[0].drug == "norepinephrine" and len(se.charges) == 3


def test_synthea_indicator_labels_use_only_the_values_on_the_page() -> None:
    se = _encounter(lactate=3.1, sodium=137, heart_rate=80, temperature=37.0)
    items = {it.meta["indicator"]: it for it in synthea.indicator_items(se)}
    assert items["lactate_gt_2"].gold == "yes" and items["sodium_lt_135"].gold == "no"
    assert items["map_lt_65"].gold == "yes" and items["vasopressor_given"].gold == "yes"
    assert items["lactate_gt_2"].meta["page"] == "laboratory" and items["map_lt_65"].meta["page"] == "flowsheet"
    for it in items.values():
        facts = synthea.page_facts(se, it.meta["page"])
        assert it.gold == ("yes" if criteria.supports(it.meta["indicator"], facts) else "no")
    assert "aki_creatinine_rise" not in items  # one creatinine value is not enough to ask
    held = synthea.indicator_items(se, held_out=frozenset({"lactate_gt_2"}))
    assert all(it.meta["indicator"] != "lactate_gt_2" for it in held)


def test_synthea_page_category_items_cover_renderable_pages() -> None:
    se = _encounter(lactate=1.0, sodium=140, heart_rate=80, temperature=37.0)
    pages = {it.meta["page_class"] for it in synthea.page_category_items(se)}
    assert pages == {"laboratory", "flowsheet", "mar", "bill", "emergency_department"}


def test_mtsamples_mapping_dedup_and_conflicts(tmp_path) -> None:
    path = tmp_path / "mtsamples.csv"
    long = "SUBJECTIVE: x " * 30
    rows = [
        {"medical_specialty": " Discharge Summary", "sample_name": "Discharge Summary - A", "transcription": long + "a"},
        {"medical_specialty": " Cardiology", "sample_name": "Discharge Summary - A", "transcription": long + "a"},
        {"medical_specialty": " Consult - History and Phy.", "sample_name": "Cardiology Consult", "transcription": long + "b"},
        {"medical_specialty": " Consult - History and Phy.", "sample_name": "Chest pain", "transcription": long + "c"},
        {"medical_specialty": " Radiology", "sample_name": "CT", "transcription": long + "d"},
        {"medical_specialty": " SOAP / Chart / Progress Notes", "sample_name": "x", "transcription": long + "d"},
        {"medical_specialty": " Radiology", "sample_name": "short", "transcription": "too short"},
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["description", "medical_specialty", "sample_name", "transcription", "keywords"])
        w.writeheader()
        for r in rows:
            w.writerow({"description": "", "keywords": "", **r})
    counts: dict[str, int] = {}
    pages = realtext.read_mtsamples(path, counts)
    assert sorted(p.page_class for p in pages) == ["consult_note", "discharge_summary"]
    assert counts["dropped: conflicting classes for one transcription"] == 1
    item = realtext.page_item(pages[0], "mtsamples_page_category_v1")
    assert item.gold in [k for k, _ in item.options]


def _order() -> dict[str, OrderRow]:
    rows = {"N39.0": True, "N39.3": True, "N39.4": False, "N39.41": True, "I10": True, "E11.9": True, "E11.65": True}
    return {c: OrderRow(c, b, c, f"title {c}") for c, b in rows.items()}


def test_codiesp_items(tmp_path) -> None:
    d = tmp_path / "train"
    (d / "text_files_en").mkdir(parents=True)
    (d / "trainD.tsv").write_text("c1\tn39.0\nc1\ti10\nc1\te11.9\nc1\te11.65\nc2\tz99.9\n", encoding="utf-8")
    (d / "text_files_en" / "c1.txt").write_text("A woman with a urinary tract infection.", encoding="utf-8")
    cases = realtext.read_codiesp_split(tmp_path, "train")
    assert cases[0][0] == "c1" and "N39.0" in cases[0][2]
    counts: dict[str, int] = {}
    items = realtext.codiesp_items(*cases[0], _order(), counts)
    golds = {it.meta["code"] for it in items}
    assert golds == {"N39.0"}  # I10 has no sibling; E11 has two coded codes (ambiguous)
    assert counts["dropped: several coded codes in one category"] == 1
    assert counts["dropped: category has a single billable code"] == 1
    it = items[0]
    labels = dict(it.options)
    assert labels[it.gold].startswith("N39.0") and "N39.4 " not in " ".join(labels.values())


def test_pmc_similar_pairs() -> None:
    pts = [realtext.PmcPatient("1-1", "case one", {"2-1": 1}, "comm"),
           realtext.PmcPatient("2-1", "case two", {"1-1": 1}, "noncomm")] + [
        realtext.PmcPatient(f"{i}-1", f"case {i}", {}, "comm") for i in range(3, 12)]
    items = realtext.pmc_similar_items(pts, seed=1)
    assert len(items) == 4 and {it.gold for it in items} == {"yes", "no"}
    for it in items:
        a, b = it.meta["pair_uids"]
        linked = b in {"1-1": {"2-1"}, "2-1": {"1-1"}}.get(a, set())
        assert (it.gold == "yes") == linked


def test_page_prose_verification() -> None:
    req = ProseRequest("page", "progress_note", {"conditions": ["Acute bronchitis", "Hypertension"]})
    ok = {"page_type": "progress_note", "conditions_mentioned": ["acute bronchitis", "essential hypertension"]}
    assert verify(req, ok) == []
    assert verify(req, {**ok, "page_type": "consult_note"})
    assert verify(req, {**ok, "conditions_mentioned": ["hypertension"]})


def test_teacher_client_falls_back_to_openai_variables(monkeypatch) -> None:
    from meddecide.teacher.client import TeacherClient

    for v in ("TEACHER_BASE_URL", "TEACHER_API_KEY", "TEACHER_MODEL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "https://example.org/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "m")
    c = TeacherClient.from_env()
    assert (c.base_url, c.api_key, c.model, c.local) == ("https://example.org/v1", "k", "m", False)
    monkeypatch.setenv("TEACHER_BASE_URL", "https://teacher.example/v1")
    assert TeacherClient.from_env().base_url == "https://teacher.example/v1"


def test_usage_meter_prices_cache_and_peak_hours() -> None:
    from datetime import datetime

    from meddecide.teacher.client import UsageMeter, is_peak

    assert is_peak(datetime(2026, 10, 13, 7, 0)) and not is_peak(datetime(2026, 10, 13, 12, 0))
    assert not is_peak(datetime(2026, 10, 17, 7, 0))  # Saturday
    m = UsageMeter()
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000,
             "prompt_tokens_details": {"cached_tokens": 500_000}, "completion_tokens_details": {"reasoning_tokens": 0}}
    m.add(usage, when=datetime(2026, 10, 13, 12, 0))
    assert abs(m.cost_usd - (0.5 * 0.15 + 0.5 * 0.003 + 0.60)) < 1e-9
    m.add(usage, when=datetime(2026, 10, 13, 7, 0))
    assert abs(m.cost_usd - 2 * (0.5 * 0.15 + 0.5 * 0.003 + 0.60) - (0.5 * 0.15 + 0.5 * 0.003 + 0.60)) < 1e-9


def test_chat_payload_turns_thinking_off_by_default(monkeypatch) -> None:
    from meddecide.teacher.client import TeacherClient

    sent = {}

    class Resp:
        def raise_for_status(self) -> None: ...
        def json(self):
            return {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}

    c = TeacherClient(base_url="https://x/v1", model="m")
    monkeypatch.setattr(c, "post", lambda path, payload: sent.update(payload) or Resp())
    assert c.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert sent["thinking"] == {"type": "disabled"} and "reasoning_effort" not in sent
    c.thinking = "high"
    c.chat([{"role": "user", "content": "hi"}])
    assert sent["reasoning_effort"] == "high" and c.usage.calls == 2
