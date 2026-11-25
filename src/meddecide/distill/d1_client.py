"""HTTP layer for the hosted Liquid ``d1-ultra`` teacher (distill_v0 D1/D2).

Split from ``d1_teacher`` so the request/validation/freeze logic stays pure and testable: this
module owns the session, the retry policy and the credit-balance probe. Concurrency is a thread
pool of keep-alive sessions (the pre-loop rate measurement used the same shape: 23-32 calls/s at
24-32 threads; 64 threads were slower).
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import requests

from meddecide.distill.d1_teacher import (
    ENDPOINT,
    CostLedger,
    TeacherAnswerError,
    TeacherRequest,
    failure_row,
    freeze_row,
    parse_response,
)

MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = 1.5
CREDITS_URL = "https://openrouter.ai/api/v1/credits"
_THREAD_LOCAL = threading.local()


def session() -> requests.Session:
    """One pooled session per worker thread (keep-alive keeps the measured throughput)."""
    current = getattr(_THREAD_LOCAL, "session", None)
    if current is None:
        current = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=0)
        current.mount("https://", adapter)
        _THREAD_LOCAL.session = current
    return current


def credits(api_key: str) -> dict[str, Any] | None:
    """The OpenRouter balance (total credits, usage, remaining); ``None``-free dict either way."""
    try:
        response = requests.get(CREDITS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        if response.status_code != 200:
            return {"status": response.status_code, "body": response.text[:200]}
        data = response.json()["data"]
        total = float(data["total_credits"])
        used = float(data["total_usage"])
        return {"status": 200, "total_credits": total, "total_usage": used, "remaining": round(total - used, 4)}
    except Exception as exc:  # the balance is context, not a reason to fail the run
        return {"error": repr(exc)[:200]}


def call_once(api_key: str, request: TeacherRequest) -> tuple[list[float], dict[str, Any]]:
    """One HTTP call, one attempt. Raises ``TeacherAnswerError`` or ``requests.RequestException``."""
    response = session().post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=request.body_bytes(),
        timeout=180,
    )
    if response.status_code != 200:
        raise TeacherAnswerError(f"http_{response.status_code}")
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise TeacherAnswerError("bad_json") from exc
    probs = parse_response(payload, request)
    usage = payload.get("usage") or {}
    return probs, {"cost": float(usage.get("cost") or 0.0), "input_tokens": int(usage.get("input_tokens") or 0),
                   "output_tokens": int(usage.get("output_tokens") or 0), "provider": payload.get("provider")}


def label_one(api_key: str, request: TeacherRequest, ledger: CostLedger, fatal: threading.Event,
              *, text_key: str | None = None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Up to ``MAX_ATTEMPTS`` attempts; returns ``(freeze row, None)`` or ``(None, failure row)``."""
    reason = "unattempted"
    attempts = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempts = attempt
        if fatal.is_set() or ledger.exhausted:
            reason = "stopped"
            break
        try:
            probs, usage = call_once(api_key, request)
        except TeacherAnswerError as exc:
            reason = exc.reason
            if reason in {"http_401", "http_402"}:
                fatal.set()
                break
            time.sleep(RETRY_BACKOFF_S * attempt)
            continue
        except requests.RequestException as exc:
            reason = f"transport:{type(exc).__name__}"
            time.sleep(RETRY_BACKOFF_S * attempt)
            continue
        ledger.charge(usage["cost"], usage["input_tokens"])
        return freeze_row(request, probs, cost=usage["cost"], input_tokens=usage["input_tokens"],
                          text_key=text_key), None
    ledger.note_failure()
    return None, failure_row(request, reason, attempts=attempts)
