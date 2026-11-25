"""The MIMIC loopback guard (osler_v4 Y0; AGENTS.md; D28, D37). A request flagged ``contains_mimic=True`` is refused before
any network call unless the teacher's base URL is loopback. ``post`` is replaced by a fake, so nothing leaves the test."""

from __future__ import annotations

from typing import Any

import pytest

from meddecide.teacher.client import TeacherClient, is_loopback
from meddecide.teacher.guarded import GuardedChat, Halt

MESSAGES = [{"role": "user", "content": "synthetic probe"}]


class _Response:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


class _CountingLimiter:
    def __init__(self) -> None:
        self.grants = 0

    def acquire(self) -> float:
        self.grants += 1
        return 0.0


def _client(base_url: str, calls: list[str]) -> TeacherClient:
    client = TeacherClient(base_url=base_url, model="nemotron", local=True)

    def fake_post(path: str, payload: dict[str, Any], **kwargs: Any) -> _Response:
        calls.append(path)
        return _Response()

    client.post = fake_post  # the guard must stop a flagged request before this is reached
    return client


@pytest.mark.parametrize("url", ["http://127.0.0.1:8011/v1", "http://localhost:8011/v1", "http://[::1]:8011/v1"])
def test_loopback_addresses_are_accepted(url: str) -> None:
    assert is_loopback(url)


@pytest.mark.parametrize("url", ["https://api.example.com/v1", "http://0.0.0.0:8011/v1", "http://127.0.0.1.example.com/v1",
                                 "http://10.0.0.5:8011/v1", "", "not a url"])
def test_other_hosts_and_empty_or_unparseable_urls_are_not_loopback(url: str) -> None:
    assert not is_loopback(url)


def test_a_flagged_request_to_a_hosted_endpoint_is_refused_before_any_call() -> None:
    calls: list[str] = []
    client = _client("https://api.example.com/v1", calls)
    with pytest.raises(PermissionError, match="loopback"):
        client.chat(MESSAGES, temperature=0.0, max_tokens=5, contains_mimic=True)
    assert calls == []


def test_a_flagged_request_to_a_loopback_teacher_is_sent() -> None:
    calls: list[str] = []
    client = _client("http://127.0.0.1:8011/v1", calls)
    assert client.chat(MESSAGES, temperature=0.0, max_tokens=5, contains_mimic=True) == "ok"
    assert calls == ["chat/completions"]


def test_an_unflagged_request_to_a_hosted_endpoint_is_not_refused_by_the_mimic_guard() -> None:
    calls: list[str] = []
    client = _client("https://api.example.com/v1", calls)
    assert client.chat(MESSAGES, temperature=0.0, max_tokens=5) == "ok"
    assert calls == ["chat/completions"]


def test_the_openai_fallback_environment_cannot_carry_a_flagged_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEACHER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.example.com/v1")
    client = TeacherClient.from_env()
    calls: list[str] = []
    client.post = lambda path, payload, **kwargs: calls.append(path) or _Response()  # type: ignore[method-assign,return-value]
    with pytest.raises(PermissionError):
        client.chat(MESSAGES, temperature=0.0, max_tokens=5, contains_mimic=True)
    assert calls == []


def test_the_guarded_path_refuses_a_flagged_request_before_it_takes_a_rate_slot() -> None:
    calls: list[str] = []
    limiter = _CountingLimiter()
    guarded = GuardedChat(_client("https://api.example.com/v1", calls), limiter, Halt())  # type: ignore[arg-type]
    with pytest.raises(PermissionError):
        guarded.chat(MESSAGES, temperature=0.0, max_tokens=5, contains_mimic=True)
    assert limiter.grants == 0
    assert calls == []


def test_the_guarded_path_sends_a_flagged_request_to_a_loopback_teacher() -> None:
    calls: list[str] = []
    limiter = _CountingLimiter()
    guarded = GuardedChat(_client("http://127.0.0.1:8011/v1", calls), limiter, Halt())  # type: ignore[arg-type]
    assert guarded.chat(MESSAGES, temperature=0.0, max_tokens=5, contains_mimic=True) == "ok"
    assert limiter.grants == 1
    assert calls == ["chat/completions"]
