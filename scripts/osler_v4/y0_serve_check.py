"""Y0 serving check for the local Nemotron teacher (ADVISORY Y0 §2-3). Synthetic prompts only.

Steps: three short prompts with reasoning off (no reasoning field may come back), one with reasoning on (it must), then two
throughput batches of 200 requests at concurrency 64: a ~3,000-token prompt with a 600-token output, and the same prompt
with a 60-token output (the Y3 selection shape). Every request has its own synthetic prompt. Writes counts and rates only
to ``--out`` (under the gitignored outputs/ tree).

Run after ``scripts/osler_v4/serve_nemotron.sh`` reports ready on 127.0.0.1:8011, then stop the server.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from meddecide.teacher.client import TeacherClient
from meddecide.teacher.serving import reasoning_text, run_stats, synthetic_record

MODEL = "nemotron"
CONCURRENCY = 64
REQUESTS = 200
PROMPT_CHARS = 13_500          # about 3,000 tokens of synthetic text; the measured prompt tokens are reported
INSTRUCTION = "Write one short summary per numbered line of the synthetic records below.\n\n"
SHORT_PROMPTS = (
    "In one sentence, what does a serum sodium below 135 mmol/L indicate?",
    "Answer yes or no: is hypokalaemia a disturbance of potassium?",
    "List three causes of thrombocytopenia, one per line.",
)


def _body(prompt: str, *, max_tokens: int, thinking: bool, ignore_eos: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if ignore_eos:
        body["ignore_eos"] = True
    return body


def _post(client: TeacherClient, body: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = client.post("chat/completions", body)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None


def _batch(client: TeacherClient, prompts: list[str], *, max_tokens: int) -> tuple[list[dict[str, Any] | None], float]:
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        bodies = list(pool.map(lambda p: _post(client, _body(p, max_tokens=max_tokens, thinking=False, ignore_eos=True)),
                               prompts))
    return bodies, time.monotonic() - start


def _sample_gpu(stop: list[bool], samples: list[float]) -> None:
    """Poll nvidia-smi every 2 s until ``stop`` is set (run in a thread)."""
    while not stop[0]:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=False).stdout.strip()
        if out:
            samples.append(float(out.splitlines()[0]))
        time.sleep(2.0)


def _start_sampler(stop: list[bool], samples: list[float]) -> threading.Thread:
    thread = threading.Thread(target=_sample_gpu, args=(stop, samples), daemon=True)
    thread.start()
    return thread


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--out", type=Path, default=Path("outputs/osler_v4/Y0/serve_check.json"))
    args = parser.parse_args()

    client = TeacherClient(base_url=args.base_url, model=MODEL, local=True, thinking="none", timeout_s=1800.0)
    result: dict[str, Any] = {"base_url": args.base_url, "concurrency": CONCURRENCY, "requests_per_batch": REQUESTS,
                              "prompt_chars": PROMPT_CHARS}

    models = client.get("models")
    result["models_status"] = models.status_code

    short: list[dict[str, Any]] = []
    for prompt in SHORT_PROMPTS:
        body = _post(client, _body(prompt, max_tokens=256, thinking=False))
        if body is None:
            short.append({"ok": False})
            continue
        message = body["choices"][0]["message"]
        short.append({"ok": True, "finish_reason": body["choices"][0].get("finish_reason"),
                      "completion_tokens": body["usage"]["completion_tokens"],
                      "reasoning_chars": len(reasoning_text(message)),
                      "content_chars": len(str(message.get("content") or ""))})
    result["reasoning_off"] = short

    on = _post(client, _body(SHORT_PROMPTS[0], max_tokens=2048, thinking=True))
    if on is None:
        result["reasoning_on"] = {"ok": False}
    else:
        message = on["choices"][0]["message"]
        result["reasoning_on"] = {"ok": True, "finish_reason": on["choices"][0].get("finish_reason"),
                                  "completion_tokens": on["usage"]["completion_tokens"],
                                  "reasoning_chars": len(reasoning_text(message)),
                                  "content_chars": len(str(message.get("content") or ""))}

    shapes = {"A_600_token_output": (13_000, 600), "B_60_token_output": (17_000, 60)}
    for name, (seed_base, max_tokens) in shapes.items():
        prompts = [INSTRUCTION + synthetic_record(PROMPT_CHARS, seed=seed_base + i) for i in range(REQUESTS)]
        samples: list[float] = []
        stop = [False]
        sampler = _start_sampler(stop, samples)
        bodies, wall = _batch(client, prompts, max_tokens=max_tokens)
        stop[0] = True
        sampler.join()
        usages = [b["usage"] for b in bodies if b is not None]
        finish = Counter(str(b["choices"][0].get("finish_reason")) for b in bodies if b is not None)
        reasoning_seen = sum(1 for b in bodies if b is not None and reasoning_text(b["choices"][0]["message"]))
        stats = run_stats(usages, failed=sum(1 for b in bodies if b is None), wall_s=wall)
        result[name] = {**stats.as_dict(), "finish_reasons": dict(finish), "responses_with_reasoning": reasoning_seen,
                        "gpu_peak_mib": max(samples, default=0.0),
                        "mean_prompt_tokens": round(stats.prompt_tokens / max(len(usages), 1), 1)}
        print(name, json.dumps(result[name]), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
