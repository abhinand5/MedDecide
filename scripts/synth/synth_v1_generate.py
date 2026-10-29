"""Thin CLI: plan synthetic cases, generate with the local Nemotron server, verify, write the manifest.

Kept item text goes to the gitignored outputs directory (``--items``). The manifest (``--manifest``) holds counts,
keep rates, reason codes, the config and timing only: it never contains item text or fact values.

Example (after the server answers on 127.0.0.1:8011):
    uv run python scripts/synth/synth_v1_generate.py --n-cases 200 --concurrency 32 --run-name smoke \
        --items outputs/synth_v1/items_smoke.jsonl --manifest outputs/synth_v1/manifest_smoke.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from meddecide.synth.client import DEFAULT_BASE_URL, NemotronClient
from meddecide.synth.driver import kept_records, run_plan, verdicts_of
from meddecide.synth.families import build_plan
from meddecide.synth.manifest import build_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="nemotron")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--run-name", default="smoke")
    parser.add_argument("--items", type=Path, required=True, help="kept items (JSONL, gitignored outputs only)")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    plan = build_plan(args.n_cases, seed=args.seed)
    client = NemotronClient(
        args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    started = time.monotonic()
    results = run_plan(plan, client, concurrency=args.concurrency)
    wall_s = time.monotonic() - started

    output_tokens = sum(result.output_tokens for result in results)
    config = {
        "seed": args.seed,
        "n_cases": args.n_cases,
        "concurrency": args.concurrency,
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "timing": {
            "wall_clock_s": round(wall_s, 2),
            "output_tokens": output_tokens,
            "output_tokens_per_s": round(output_tokens / wall_s, 2) if wall_s > 0 else None,
        },
    }
    manifest = build_manifest(verdicts_of(results), run_name=args.run_name, config=config)
    write_manifest(args.manifest, manifest)

    args.items.parent.mkdir(parents=True, exist_ok=True)
    with args.items.open("w", encoding="utf-8") as handle:
        for record in kept_records(results):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    totals = manifest["totals"]
    print(f"generated={totals['generated']} kept={totals['kept']} rejected={totals['rejected']}")
    print(f"wall_clock_s={wall_s:.1f} output_tokens={output_tokens} "
          f"output_tokens_per_s={config['timing']['output_tokens_per_s']}")
    print(f"manifest={args.manifest}")


if __name__ == "__main__":
    main()
