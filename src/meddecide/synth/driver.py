"""Generation driver: render each planned case, ask a client for text, verify it, keep or reject with a reason.

A client is any object with ``complete(system, user) -> Completion`` (``NemotronClient`` in production, a fake in
tests). Every planned case gets exactly one ``Generated`` result: kept, or rejected with a reason code. A failed
request is retried and then recorded as ``request_error``; a refused render is recorded as ``render_refused``.
Nothing is dropped silently.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from meddecide.synth.client import Completion, ServerError
from meddecide.synth.families import PlannedCase
from meddecide.synth.render import LabelLeakError, render_request
from meddecide.synth.verify import Verdict, verify_text

REASON_REQUEST_ERROR = "request_error"
REASON_RENDER_REFUSED = "render_refused"


class TextClient(Protocol):
    def complete(self, system: str, user: str) -> Completion: ...


@dataclass(frozen=True)
class Generated:
    """Outcome of one planned case. ``text`` is None unless the item was kept (only kept text is stored)."""

    planned: PlannedCase
    verdict: Verdict
    text: str | None
    output_tokens: int

    @property
    def kept(self) -> bool:
        return self.verdict.kept


def _rejected(planned: PlannedCase, reason: str) -> Verdict:
    return Verdict(
        family=planned.case.family,
        case_id=planned.case.case_id,
        reasons=(reason,),
        primary_reason=reason,
        missing_facts=(),
        banned_hits=(),
        word_count=0,
    )


def generate_one(planned: PlannedCase, client: TextClient, *, retries: int = 2, backoff_s: float = 2.0) -> Generated:
    """Render, call, verify one case. Keeps text only when verified; retries request errors up to ``retries``."""
    try:
        request = render_request(
            planned.case,
            target_words=planned.target_words,
            style=planned.style,
            spec=planned.spec,
        )
    except LabelLeakError:
        return Generated(planned, _rejected(planned, REASON_RENDER_REFUSED), None, 0)

    completion: Completion | None = None
    for attempt in range(retries + 1):
        try:
            completion = client.complete(request.system, request.user)
            break
        except ServerError:
            if attempt == retries:
                return Generated(planned, _rejected(planned, REASON_REQUEST_ERROR), None, 0)
            time.sleep(backoff_s * (attempt + 1))
    assert completion is not None

    verdict = verify_text(
        completion.text,
        planned.case,
        word_range=planned.word_range,
        spec=planned.spec,
    )
    text = completion.text if verdict.kept else None
    return Generated(planned, verdict, text, completion.output_tokens)


def run_plan(
    plan: Sequence[PlannedCase],
    client: TextClient,
    *,
    concurrency: int = 32,
    retries: int = 2,
    backoff_s: float = 2.0,
) -> list[Generated]:
    """Run every planned case with ``concurrency`` workers. Results come back in plan order."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(
            pool.map(lambda p: generate_one(p, client, retries=retries, backoff_s=backoff_s), plan)
        )
    if len(results) != len(plan):
        raise RuntimeError("driver lost cases: result count does not match plan")
    return results


def verdicts_of(results: Iterable[Generated]) -> list[Verdict]:
    return [result.verdict for result in results]


def kept_records(results: Iterable[Generated]) -> list[dict[str, Any]]:
    """JSON-ready records of kept items only. These go to the gitignored outputs directory, never to a manifest."""
    records: list[dict[str, Any]] = []
    for result in results:
        if not result.kept or result.text is None:
            continue
        planned = result.planned
        records.append(
            {
                "index": planned.index,
                "case_id": planned.case.case_id,
                "family": planned.case.family,
                "label": planned.case.label,
                "options": list(planned.spec.options_for(planned.case)),
                "style": planned.style.name,
                "target_words": planned.target_words,
                "word_count": result.verdict.word_count,
                "text": result.text,
            }
        )
    return records
