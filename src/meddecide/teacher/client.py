"""Teacher client for the self-hosted DeepSeek-V4.1-Flash endpoint (tasks T6/T10).

The endpoint is an OpenAI-compatible server operated by the operator. Its exact logprob
surface is *not* assumed: :meth:`TeacherClient.probe_shapes` tries the documented
candidate shapes and records which one answers, so T10 can report the API shape it used
rather than guessing (ADVISORY T10).

Secrets are read from the environment only (``TEACHER_BASE_URL``, ``TEACHER_API_KEY``)
and are never written to disk or logged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

DEFAULT_TIMEOUT_S = 120.0

# MIMIC text, values and identifiers go only to a teacher on this machine (AGENTS.md; D28, D37). A request flagged
# contains_mimic=True is refused, before any network call, unless the base URL's host is a loopback address.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def is_loopback(base_url: str) -> bool:
    """True when the base URL's host is 127.0.0.1, localhost or ::1. An empty or unparseable URL is not loopback."""
    return (urlsplit(base_url).hostname or "") in LOOPBACK_HOSTS


def require_loopback_for_mimic(base_url: str) -> None:
    """Raise PermissionError unless ``base_url`` is loopback; called before any request that carries MIMIC content."""
    if not is_loopback(base_url):
        raise PermissionError("request carries MIMIC content but the teacher is not on loopback (AGENTS.md; D28, D37)")

# OpenCode Go list prices for DeepSeek V4.1 Flash, USD per 1M tokens (2026-10-13): (uncached input, output, cached read)
OFF_PEAK = (0.15, 0.60, 0.003)
PEAK = (0.30, 1.20, 0.006)


def is_peak(when: datetime | None = None) -> bool:
    """OpenCode Go peak hours: 01:00-04:00 and 06:00-10:00 UTC, Monday to Friday."""
    t = when or datetime.now(UTC)
    return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)


@dataclass
class UsageMeter:
    """Running token counts and an estimated cost at list price (peak or off-peak by the time of each call)."""

    priced: bool = True            # False for a local teacher: tokens are counted, cost stays 0
    calls: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, usage: dict[str, Any], when: datetime | None = None) -> None:
        prompt = int(usage.get("prompt_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        out = int(usage.get("completion_tokens") or 0)
        rate_in, rate_out, rate_cached = PEAK if is_peak(when) else OFF_PEAK
        self.calls += 1
        self.prompt_tokens += prompt
        self.cached_tokens += cached
        self.completion_tokens += out
        self.reasoning_tokens += int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        if self.priced:
            self.cost_usd += ((prompt - cached) * rate_in + cached * rate_cached + out * rate_out) / 1e6

    def as_dict(self) -> dict[str, Any]:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens, "cached_tokens": self.cached_tokens,
                "completion_tokens": self.completion_tokens, "reasoning_tokens": self.reasoning_tokens,
                "cost_usd": round(self.cost_usd, 4)}


@dataclass
class TeacherClient:
    """Minimal OpenAI-compatible client for the teacher endpoint."""

    base_url: str = ""
    api_key: str = ""
    timeout_s: float = DEFAULT_TIMEOUT_S
    model: str = ""
    # True only for a teacher on an operator-controlled machine; credentialed data (MIMIC) may go only to a local teacher
    local: bool = False
    extra_headers: dict[str, str] = field(default_factory=dict)
    # thinking is off by default: measured on 2026-10-13, DeepSeek V4.1 Flash on OpenCode Go spent 3,000-6,000 reasoning
    # tokens on a 150-word note (default and "max" effort used the whole budget and returned no text), and reasoning is
    # billed as output. "disabled" sends thinking={"type": "disabled"}; "none" sends no thinking field (for servers such as
    # vLLM, where the switch is a chat-template argument passed through extra_body); any other value is reasoning_effort.
    thinking: str = "disabled"
    # merged into every request body, e.g. {"chat_template_kwargs": {"enable_thinking": false}} for Gemma 4 on vLLM
    extra_body: dict[str, Any] = field(default_factory=dict)
    usage: UsageMeter = field(default_factory=lambda: UsageMeter())

    def __post_init__(self) -> None:
        self.usage.priced = not self.local
    _client: httpx.Client | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> TeacherClient:
        """Build a client from ``TEACHER_BASE_URL`` / ``TEACHER_API_KEY`` (falling back to ``OPENAI_API_BASE`` or
        ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``), plus ``TEACHER_MODEL`` (or ``OPENAI_MODEL``), ``TEACHER_LOCAL``
        ("1" for an operator-hosted endpoint) and ``TEACHER_EXTRA_HEADERS`` (a JSON object, e.g. the
        ``x-opencode-session`` header a hosted gateway asks for; never a secret)."""
        extra = os.environ.get("TEACHER_EXTRA_HEADERS", "")
        env = os.environ.get
        return cls(
            # the standard OpenAI-client variables are the fallback, so an existing setup works unchanged
            base_url=(env("TEACHER_BASE_URL") or env("OPENAI_API_BASE") or env("OPENAI_BASE_URL") or "").rstrip("/"),
            api_key=env("TEACHER_API_KEY") or env("OPENAI_API_KEY") or "",
            timeout_s=timeout_s,
            model=env("TEACHER_MODEL") or env("OPENAI_MODEL") or "",
            local=os.environ.get("TEACHER_LOCAL", "") == "1",
            extra_headers={"User-Agent": "osler-datagen/1.0", "x-opencode-session": env("TEACHER_SESSION") or "osler-v1",
                           **(dict(json.loads(extra)) if extra else {})},
            thinking=env("TEACHER_THINKING") or "disabled",
            extra_body=dict(json.loads(env("TEACHER_EXTRA_BODY") or "{}")),
        )

    # ---- plumbing ------------------------------------------------------------
    @property
    def configured(self) -> bool:
        """True when a base URL is present (the API key may legitimately be empty)."""
        return bool(self.base_url)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s, headers=self._headers())
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> TeacherClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _url(self, path: str) -> str:
        if not self.configured:
            raise RuntimeError("TEACHER_BASE_URL is not set")
        return f"{self.base_url}/{path.lstrip('/')}"

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.client().get(self._url(path), **kwargs)

    def post(self, path: str, payload: dict[str, Any], **kwargs: Any) -> httpx.Response:
        return self.client().post(self._url(path), json=payload, **kwargs)

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.7, max_tokens: int = 1024,
             contains_mimic: bool = False) -> str:
        """One ``/chat/completions`` call; returns the assistant text (raises on an HTTP error). Token usage, including
        cached prompt tokens and reasoning tokens, is added to ``self.usage``. With ``contains_mimic=True`` the call is
        refused (PermissionError, nothing sent) unless the base URL is loopback."""
        if contains_mimic:
            require_loopback_for_mimic(self.base_url)
        if not self.model:
            raise RuntimeError("TEACHER_MODEL is not set")
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature,
                                   "max_tokens": max_tokens}
        if self.thinking == "disabled":
            payload["thinking"] = {"type": "disabled"}
        elif self.thinking and self.thinking != "none":
            payload["reasoning_effort"] = self.thinking
        payload.update(self.extra_body)
        response = self.post("chat/completions", payload)
        response.raise_for_status()
        body = response.json()
        self.usage.add(body.get("usage") or {})
        return str(body["choices"][0]["message"]["content"] or "")

    # ---- endpoint checks -----------------------------------------------------
    def list_models(self) -> dict[str, Any]:
        """``GET /models`` — the reachability check T0/T10 use. Returns status + parsed body."""
        response = self.get("models")
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text[:500]}
        return {"status_code": response.status_code, "ok": response.status_code == 200, "body": body}

    def probe_shapes(self, prompt: str = "The capital of France is") -> dict[str, Any]:
        """Try the candidate logprob API shapes; report which ones answer and how.

        Returns a dict with one entry per shape: ``status_code``, ``ok``, ``n_top_logprobs``
        and, on success, whether the returned token list is non-empty. No prompt text from
        benchmark items is ever passed here — the default probe prompt is a fixed string.
        """
        shapes: dict[str, Any] = {}

        def record(name: str, response: httpx.Response) -> None:
            entry: dict[str, Any] = {"status_code": response.status_code, "ok": response.status_code == 200}
            if response.status_code == 200:
                try:
                    body = response.json()
                except ValueError:
                    entry["ok"] = False
                    entry["error"] = "non-JSON body"
                    shapes[name] = entry
                    return
                choices = body.get("choices") or []
                lp = (choices[0].get("logprobs") if choices else None) or {}
                entry["n_top_logprobs"] = len(lp.get("top_logprobs") or [])
                entry["has_token_logprobs"] = bool(lp.get("tokens"))
                entry["tokens_returned"] = len(lp.get("tokens") or [])
            else:
                entry["error"] = response.text[:200]
            shapes[name] = entry

        for name, path in (("v1_completions", "v1/completions"), ("completions", "completions")):
            try:
                record(
                    name,
                    self.post(
                        path,
                        {
                            "model": "default",
                            "prompt": prompt,
                            "max_tokens": 1,
                            "temperature": 0,
                            "logprobs": 20,
                            "echo": False,
                        },
                    ),
                )
            except (httpx.HTTPError, RuntimeError) as exc:
                shapes[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        for name, path in (("v1_chat_completions", "v1/chat/completions"), ("chat_completions", "chat/completions")):
            try:
                record(
                    name,
                    self.post(
                        path,
                        {
                            "model": "default",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 1,
                            "temperature": 0,
                            "logprobs": True,
                            "top_logprobs": 20,
                        },
                    ),
                )
            except (httpx.HTTPError, RuntimeError) as exc:
                shapes[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        working = [name for name, entry in shapes.items() if entry.get("ok")]
        return {
            "base_url_configured": self.configured,
            "shapes": shapes,
            "working_shapes": working,
            "recommended": working[0] if working else None,
        }


def write_probe(path: Path, payload: dict[str, Any]) -> Path:
    """Persist a probe result (never contains prompts or secrets)."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
