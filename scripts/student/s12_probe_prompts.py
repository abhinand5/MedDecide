"""S12 probe: describe the fixed 20-prompt set used by the byte-identity check.

The prompt set is the one S7's acceptance test uses: the dev slice
``read_items(dev.jsonl, limit=64, stride=11)``, sorted by ``item_id``, first 20 items,
each rendered through :func:`meddecide.eval.readout.render_prompt` with the shipped
`bare` letter variant. This script only measures it (item ids, token counts); the
generation check lives in ``s12_byte_identity.py``.

    uv run python scripts/student/s12_probe_prompts.py --out outputs/student_v0/S12/prompts.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from meddecide.eval.readout import canonicalise_options, render_prompt
from meddecide.train.data import read_items

REPO = Path(__file__).resolve().parents[2]
DEV_PATH = REPO / "data" / "train" / "student_v0" / "dev.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, default=DEV_PATH)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--stride", type=int, default=11)
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--out", type=Path, default=REPO / "outputs/student_v0/S12/prompts.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    items = read_items(args.dev, limit=args.limit, stride=args.stride)
    items.sort(key=lambda i: i.item_id)
    prompt_items = items[: args.n_prompts]
    rows = []
    for item in prompt_items:
        canonical, _ = canonicalise_options(item)
        prompt = render_prompt(canonical, tokenizer, "bare")
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        rows.append(
            {
                "item_id": item.item_id,
                "template_id": str(item.template_id),
                "qtype": str(item.qtype),
                "n_options": canonical.n_options,
                "prompt_chars": len(prompt),
                "prompt_tokens": len(token_ids),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
    payload = {
        "source": str(args.dev.relative_to(REPO)),
        "sampling": f"read_items(limit={args.limit}, stride={args.stride}); sort by item_id; first {args.n_prompts}",
        "tokenizer": args.tokenizer,
        "n_prompts": len(rows),
        "prompt_tokens_min": min(r["prompt_tokens"] for r in rows),
        "prompt_tokens_max": max(r["prompt_tokens"] for r in rows),
        "prompt_tokens_sum": sum(r["prompt_tokens"] for r in rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    for row in rows:
        print(f"{row['item_id']:>40} {row['template_id']:<32} {row['prompt_tokens']:>6} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
