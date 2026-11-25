"""osler_v2 W3b: whole synthetic charts with a planted evidence page (CPU; a fake local teacher)."""

from __future__ import annotations

import json
import re

from test_mr_sources_v1 import _encounter

from meddecide.mr import criteria, icd
from meddecide.mr.charts import build_chart
from meddecide.mr.pages import PAGE_KEYS
from meddecide.mr.teacher_prose import EXTRACT_SYSTEM, WRITE_SYSTEM


class FakeLocalTeacher:
    """Writes a page that restates its facts in words (no numbers) and answers each extraction faithfully."""

    local = True

    def __init__(self, leak_numbers: bool = False) -> None:
        self.pages: dict[str, dict] = {}
        self.leak = leak_numbers
        self.calls = 0

    def chat(self, messages, *, temperature, max_tokens):
        self.calls += 1
        system, user = messages[0]["content"], messages[1]["content"]
        if system == WRITE_SYSTEM:
            page_type = re.search(r"PAGE TYPE: (.+)", user).group(1).replace(" ", "_")
            facts = json.loads(user.split("FACTS:\n", 1)[1])
            text = f"[{len(self.pages)}] {page_type} note for {facts['patient_name']}." + (" HR 120." if self.leak else "")
            self.pages[text] = {"page_type": page_type, "facts": facts}
            return text
        page = self.pages[user.split("PAGE:\n", 1)[1]]
        if system == EXTRACT_SYSTEM["page"]:
            return json.dumps({"page_type": page["page_type"], "conditions_mentioned": page["facts"]["conditions"]})
        if system == EXTRACT_SYSTEM["chunk"]:
            obs = [{"name": "heart_rate", "value": 120}] if self.leak else []
            return json.dumps({"observations": obs, "medications": []})
        if system == EXTRACT_SYSTEM["diagnosis"]:
            dx = page["facts"]["final_diagnosis"]
            return json.dumps({k: dx[k] for k in ("condition", "attributes", "certainty", "subject", "status")})
        raise AssertionError(system[:60])


def test_chart_items_labels_and_twins() -> None:
    enc = _encounter(lactate=1.0, sodium=140, heart_rate=80, temperature=37.0)
    kinds: dict[str, int] = {}
    for i in range(12):
        teacher = FakeLocalTeacher()
        res = build_chart(teacher, enc, f"chart-{i}")
        assert res.items, res.reasons
        for it in res.items:
            kinds[it.generator] = kinds.get(it.generator, 0) + 1
            assert it.gold in [k for k, _ in it.options]
            pages = re.findall(r"=== Page (\d+) of (\d+) ===", it.state)
            assert pages and all(n == pages[0][1] for _, n in pages)
            assert len(set(re.findall(r"note for ([^.]+)\.", it.state))) == 1  # one patient per chart
        twins = [it for it in res.items if it.generator == "chart_indicator_evidence_v1"]
        if twins:
            assert {t.gold for t in twins} == {"yes", "no"} and twins[0].meta["indicator"] != "aki_creatinine_rise"
        icd_item = next(it for it in res.items if it.generator == "chart_icd_support_v1")
        assert icd_item.meta["family"] != "heart_failure"
        dx_gold = icd_item.meta["code"]
        assert dx_gold is None or dx_gold in icd.FAMILY_BY_KEY[icd_item.meta["family"]].codes
    assert kinds.get("chart_indicator_evidence_v1", 0) >= 12 and kinds.get("chart_evidence_page_v1", 0) >= 6


def test_evidence_page_label_points_at_the_planted_page() -> None:
    enc = _encounter(lactate=1.0, sodium=140, heart_rate=80, temperature=37.0)
    for i in range(10):
        res = build_chart(FakeLocalTeacher(), enc, f"chart-{i}")
        for it in res.items:
            if it.generator != "chart_evidence_page_v1":
                continue
            page_no = it.meta["page"]
            assert dict(it.options)[it.gold] == f"Page {page_no}"
            body = it.state.split(f"=== Page {page_no} of")[1].split("=== Page")[0]
            assert "note for" not in body  # a code-rendered page, never a narrative one


def test_numbers_in_narrative_pages_drop_the_chart() -> None:
    enc = _encounter(lactate=1.0, sodium=140, heart_rate=80, temperature=37.0)
    res = build_chart(FakeLocalTeacher(leak_numbers=True), enc, "chart-leak")
    assert not res.items and any("states heart_rate" in r for r in res.reasons)


def test_page_classes_exist() -> None:
    assert {"admission", "progress_note", "discharge_summary"} <= set(PAGE_KEYS)
    assert criteria.INDICATOR_BY_KEY
