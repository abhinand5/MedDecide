"""Chat client for the local Nemotron server, restricted to loopback.

The guard checks the URL before any request is made, and again on each request. Only the ``http`` scheme and the
host ``127.0.0.1`` are accepted. Proxy variables are ignored (an empty ``ProxyHandler``), so a proxy in the
environment cannot route a request off the machine. Reasoning is off: every request sends
``chat_template_kwargs={"enable_thinking": False}``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_BASE_URL = "http://127.0.0.1:8011/v1"


class LoopbackViolation(ValueError):
    """The URL does not name the loopback address over plain http. Nothing is sent."""


class ServerError(RuntimeError):
    """A request failed (connection, HTTP status, timeout or a malformed response)."""


def require_loopback_url(url: str) -> str:
    """Return ``url`` if it is an ``http`` URL on 127.0.0.1 with a port, else raise ``LoopbackViolation``.

    The message never echoes the URL, so a mistyped secret in a URL is not printed.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "http":
        raise LoopbackViolation("generator refuses a non-http endpoint")
    if parts.hostname != LOOPBACK_HOST:
        raise LoopbackViolation(f"generator refuses any host other than {LOOPBACK_HOST}")
    if parts.username is not None or parts.password is not None:
        raise LoopbackViolation("generator refuses URLs with credentials")
    try:
        port = parts.port
    except ValueError as exc:
        raise LoopbackViolation("generator refuses a URL with an invalid port") from exc
    if port is None:
        raise LoopbackViolation("generator needs an explicit port on the loopback URL")
    return url


@dataclass(frozen=True)
class Completion:
    text: str
    output_tokens: int
    seconds: float


class NemotronClient:
    """OpenAI-compatible ``/chat/completions`` client on the loopback address."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        model: str = "nemotron",
        timeout_s: float = 600.0,
        temperature: float = 0.7,
        max_tokens: int = 1200,
    ) -> None:
        self.base_url = require_loopback_url(base_url).rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _body(self, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def complete(self, system: str, user: str) -> Completion:
        require_loopback_url(self.base_url + "/chat/completions")
        data = json.dumps(self._body(system, user)).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ServerError(f"request failed: {type(exc).__name__}") from exc
        except json.JSONDecodeError as exc:
            raise ServerError("response was not JSON") from exc
        seconds = time.monotonic() - started
        try:
            text = payload["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ServerError("response has no message content") from exc
        usage = payload.get("usage") or {}
        output_tokens = int(usage.get("completion_tokens") or 0)
        return Completion(text=text, output_tokens=output_tokens, seconds=seconds)
