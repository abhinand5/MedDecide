"""Tests for the generation driver: loopback guard, keep/reject accounting with a fake client, manifest content.

Synthetic families only. The fake client returns canned strings; no network call is made.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from meddecide.synth.client import (
    Completion,
    LoopbackViolation,
    NemotronClient,
    ServerError,
    require_loopback_url,
)
from meddecide.synth.driver import (
    REASON_REQUEST_ERROR,
    generate_one,
    kept_records,
    run_plan,
    verdicts_of,
)
from meddecide.synth.families import FAMILIES, PlannedCase, build_plan
from meddecide.synth.manifest import build_manifest, write_manifest
from meddecide.synth.render import render_request
from meddecide.synth.verify import (
    REASON_BANNED_LABEL,
    REASON_BANNED_META,
    REASON_EMPTY,
    REASON_MISSING_FACT,
)

MARKER = "ZQXMARKERQ"
FACT_LINE = re.compile(r"^- [^|]+ \| [^:]+: (.*)$", re.MULTILINE)
WORDS = re.compile(r"Target length: (\d+) words")

BEHAVIOURS = ("good", "missing", "label", "meta", "empty", "error")


class CannedClient:
    """Returns a canned text chosen per user prompt. Behaviour is keyed by the plan position, so it is deterministic."""

    def __init__(self, behaviour_by_user: dict[str, str], texts_by_user: dict[str, str]) -> None:
        self.behaviour_by_user = behaviour_by_user
        self.texts_by_user = texts_by_user
        self.calls = 0

    def complete(self, system: str, user: str) -> Completion:
        self.calls += 1
        behaviour = self.behaviour_by_user[user]
        if behaviour == "error":
            raise ServerError("request failed: canned")
        return Completion(text=self.texts_by_user[user], output_tokens=7, seconds=0.0)


def _fact_values(user: str) -> list[str]:
    return [m.group(1).strip() for m in FACT_LINE.finditer(user)]


def _filler(words: int) -> str:
    return " ".join(["The patient remained stable overnight."] * (words // 7 + 1)).split()[:words]


def _canned(planned: PlannedCase, user: str, behaviour: str) -> str:
    values = _fact_values(user)
    target = int(WORDS.search(user).group(1))
    if behaviour == "empty":
        return ""
    if behaviour == "meta":
        body = [*values, "As an AI, this is a synthetic note."]
    elif behaviour == "missing":
        body = values[1:]
    elif behaviour == "label":
        body = [*values, f"Conclusion: {planned.case.label}."]
    else:
        body = values
    body_text = " ".join(body)
    pad = max(0, target - len(body_text.split()))
    return f"{body_text} {' '.join(_filler(pad))} {MARKER}".strip()


def _run(plan: list[PlannedCase]) -> tuple[list, CannedClient]:
    behaviour_by_user: dict[str, str] = {}
    texts: dict[str, str] = {}
    for position, planned in enumerate(plan):
        user = render_request(
            planned.case, target_words=planned.target_words, style=planned.style, spec=planned.spec
        ).user
        behaviour = BEHAVIOURS[position % len(BEHAVIOURS)]
        behaviour_by_user[user] = behaviour
        texts[user] = _canned(planned, user, behaviour) if behaviour != "error" else ""
    client = CannedClient(behaviour_by_user, texts)
    results = run_plan(plan, client, concurrency=1, retries=1, backoff_s=0.0)
    return results, client


# ---- loopback guard -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8011/v1",
        "http://127.0.0.1:8001/v1/chat/completions",
    ],
)
def test_loopback_guard_accepts_loopback_http(url: str) -> None:
    assert require_loopback_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8011/v1",
        "http://0.0.0.0:8011/v1",
        "http://10.0.0.5:8011/v1",
        "http://example.com:8011/v1",
        "https://127.0.0.1:8011/v1",
        "http://127.0.0.2:8011/v1",
        "http://user:pass@127.0.0.1:8011/v1",
        "http://127.0.0.1/v1",
        "http://[::1]:8011/v1",
        "ftp://127.0.0.1:8011/v1",
        "",
    ],
)
def test_loopback_guard_refuses_other_hosts_and_schemes(url: str) -> None:
    with pytest.raises(LoopbackViolation):
        require_loopback_url(url)


def test_client_constructor_refuses_non_loopback_base_url() -> None:
    with pytest.raises(LoopbackViolation):
        NemotronClient("http://203.0.113.7:8011/v1")


def test_loopback_error_does_not_echo_the_url() -> None:
    with pytest.raises(LoopbackViolation) as info:
        require_loopback_url("http://secret-host.invalid:8011/v1?key=ABC123")
    assert "ABC123" not in str(info.value)
    assert "secret-host" not in str(info.value)


# ---- keep and reject accounting -----------------------------------------------------------------------------


def test_keep_reject_accounting_with_canned_client() -> None:
    plan = build_plan(42, seed=7)
    results, client = _run(plan)

    assert len(results) == len(plan) == 42
    assert client.calls >= len(plan)  # errors are retried, so there are at least as many calls as cases

    expected_by_position = [BEHAVIOURS[i % len(BEHAVIOURS)] for i in range(len(plan))]
    kept_positions = [i for i, b in enumerate(expected_by_position) if b == "good"]
    assert [r.planned.index for r in results if r.kept] == [plan[i].index for i in kept_positions]

    primary = Counter(r.verdict.primary_reason for r in results if not r.kept)
    assert primary[REASON_MISSING_FACT] == expected_by_position.count("missing")
    assert primary[REASON_BANNED_LABEL] == expected_by_position.count("label")
    assert primary[REASON_BANNED_META] == expected_by_position.count("meta")
    assert primary[REASON_EMPTY] == expected_by_position.count("empty")
    assert primary[REASON_REQUEST_ERROR] == expected_by_position.count("error")
    assert sum(primary.values()) == len(plan) - len(kept_positions)  # no silent drops (R5)


def test_kept_text_is_the_only_text_returned_and_rejections_have_none() -> None:
    plan = build_plan(12, seed=7)
    results, _ = _run(plan)
    for result in results:
        if result.kept:
            assert result.text is not None and result.text.endswith(MARKER)
        else:
            assert result.text is None
    records = kept_records(results)
    assert len(records) == sum(r.kept for r in results)
    for record in records:
        assert record["label"] in record["options"]


def test_retries_then_request_error_is_recorded() -> None:
    planned = build_plan(3, seed=1)[0]
    user = render_request(planned.case, target_words=planned.target_words, style=planned.style, spec=planned.spec).user
    client = CannedClient({user: "error"}, {})
    generated = generate_one(planned, client, retries=2, backoff_s=0.0)
    assert not generated.kept
    assert generated.verdict.primary_reason == REASON_REQUEST_ERROR
    assert client.calls == 3


def test_length_out_of_range_is_a_reason_not_a_silent_drop() -> None:
    planned = build_plan(3, seed=1)[0]
    user = render_request(planned.case, target_words=planned.target_words, style=planned.style, spec=planned.spec).user
    facts = _fact_values(user)
    client = CannedClient({user: "good"}, {user: " ".join(facts)})
    generated = generate_one(planned, client, retries=0)
    assert not generated.kept
    assert generated.verdict.primary_reason == "length_out_of_range"


def test_real_families_only_and_three_of_them() -> None:
    plan = build_plan(200, seed=20261029)
    assert len(plan) == 200
    assert {p.case.family for p in plan} == set(FAMILIES)
    for planned in plan:
        assert 3 <= len(planned.spec.label_set) <= 5
        assert planned.spec.label_match == "token"
        assert planned.case.label in planned.spec.label_set


# ---- manifest content ---------------------------------------------------------------------------------------


def test_manifest_has_counts_rates_reasons_and_no_item_text(tmp_path: Path) -> None:
    plan = build_plan(42, seed=7)
    results, _ = _run(plan)
    manifest = build_manifest(
        verdicts_of(results),
        run_name="unit",
        config={"seed": 7, "timing": {"wall_clock_s": 1.0, "output_tokens": 0, "output_tokens_per_s": 0.0}},
    )
    out = write_manifest(tmp_path / "manifest.json", manifest)
    text = out.read_text(encoding="utf-8")
    loaded = json.loads(text)

    assert loaded["item_text_included"] is False
    assert loaded["totals"]["generated"] == 42
    assert loaded["totals"]["kept"] + loaded["totals"]["rejected"] == 42
    assert set(loaded["families"]) == set(FAMILIES)
    for summary in loaded["families"].values():
        assert summary["kept"] + summary["rejected"] == summary["generated"]
        assert sum(summary["rejected_by_primary_reason"].values()) == summary["rejected"]
        assert summary["keep_rate"] == pytest.approx(summary["kept"] / summary["generated"])

    # No item text and no fact value appears anywhere in the manifest file.
    assert MARKER not in text
    for planned in plan:
        for fact in planned.case.facts:
            assert fact.value not in text
