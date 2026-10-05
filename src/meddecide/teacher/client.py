"""Teacher client for the self-hosted DeepSeek-V4.1-Flash endpoint (tasks T6/T10).

The endpoint is an OpenAI-compatible server operated by the operator. Its exact logprob
surface is *not* assumed: :meth:`TeacherClient.probe_shapes` tries the documented
candidate shapes and records which one answers, so T10 can report the API shape it used
rather than guessing (ADVISORY T10).

Secrets are read from the environment only (``TEACHER_BASE_URL``, ``TEACHER_API_KEY``)
and are never written to disk or logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_TIMEOUT_S = 120.0


@dataclass
class TeacherClient:
    """Minimal OpenAI-compatible client for the teacher endpoint."""

    base_url: str = ""
    api_key: str = ""
    timeout_s: float = DEFAULT_TIMEOUT_S
    _client: httpx.Client | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> TeacherClient:
        """Build a client from ``TEACHER_BASE_URL`` / ``TEACHER_API_KEY``."""
        return cls(
            base_url=os.environ.get("TEACHER_BASE_URL", "").rstrip("/"),
            api_key=os.environ.get("TEACHER_API_KEY", ""),
            timeout_s=timeout_s,
        )

    # ---- plumbing ------------------------------------------------------------
    @property
    def configured(self) -> bool:
        """True when a base URL is present (the API key may legitimately be empty)."""
        return bool(self.base_url)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
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
