"""The chat call the teacher build uses: the shared request rate, retries on 429 and 5xx, and a halt on refused access.

``GuardedChat`` has the shape ``meddecide.mr.teacher_prose.generate`` expects (``local`` and ``chat``). It never retries a
4xx other than 429, and it sets the shared ``Halt`` on 401 or 403 so that every worker stops sending requests.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import httpx

from meddecide.teacher.budget import RateLimiter
from meddecide.teacher.client import TeacherClient, require_loopback_for_mimic

RETRIES = 5


class Halt:
    """Set when the endpoint refuses access or fails repeatedly; every later request then fails at once."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason = ""

    def set(self, reason: str) -> None:
        if not self._event.is_set():
            self.reason = reason
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()


class GuardedChat:
    def __init__(self, client: TeacherClient, limiter: RateLimiter, halt: Halt,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._client = client
        self._limiter = limiter
        self._halt = halt
        self._sleep = sleep
        self.local = client.local

    def chat(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int,
             contains_mimic: bool = False) -> str:
        if contains_mimic:
            require_loopback_for_mimic(self._client.base_url)
        attempt = 0
        while True:
            if self._halt.is_set():
                raise RuntimeError("halted")
            self._limiter.acquire()
            try:
                return self._client.chat(messages, temperature=temperature, max_tokens=max_tokens,
                                         contains_mimic=contains_mimic)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in (401, 403):
                    self._halt.set(f"HTTP {status} from the teacher endpoint")
                    raise
                if status == 429 or 500 <= status < 600:
                    attempt += 1
                    if attempt > RETRIES:
                        raise
                    self._sleep(2.0**attempt)
                    continue
                raise
