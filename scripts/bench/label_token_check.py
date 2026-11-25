#!/usr/bin/env python
"""T6 acceptance: check option-letter tokenisation for every model in the ladder.

Loads only tokenizers (no weights), so it runs in seconds and does not need the GPU. For each
model it tries the ``" A"`` (leading space) and ``"A"`` variants for option keys ``A``..``AF``
(32 options, the harness ``max_options``) plus the numeric keys ``1``..``10`` used by ``score``
items, and records which variant is single-token. A model with no single-token variant cannot
be scored by this harness and is recorded as such rather than silently mis-scored.

Writes ``outputs/bench_v0/T6/label_token_check.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from meddecide.eval.readout import check_label_tokens, pick_variant

LADDER = [
    "LiquidAI/LFM2.5-350M",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-0.8B-Base",
    "google/medgemma-1.5-4b-it",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
]
# single-GPU decision models whose published inference path maps onto our item format; a few
# are encoder-style and are checked with their own tokenizer only
OTHERS = [
    "rAVEUK/open-jev-deberta-v3-large",
    "junma/MedJev-Qwen3.5-0.8B",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/bench_v0/T6/label_token_check.json"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--n-options", type=int, default=32)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    models = args.models or [*LADDER, *OTHERS]
    report: dict[str, Any] = {"n_options_checked": args.n_options, "models": {}}
    for model_id in models:
        entry: dict[str, Any] = {"model_id": model_id}
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            checks = check_label_tokens(
                tokenizer, model_id, n_options=args.n_options, variants=("space", "bare")
            )
            entry["variants"] = {c.variant: c.to_dict() for c in checks}
            try:
                chosen = pick_variant(checks)
                entry["chosen_variant"] = chosen.variant
                entry["usable"] = True
                # Which variant the *model* emits is detected empirically when weights are
                # loaded (meddecide.eval.readout.detect_variant); only tokenizers are loaded
                # here, so this field is informational and is overwritten by the run summary.
                entry["variant_used_at_run_time"] = "detected at run time (see run summary)"
            except ValueError as exc:
                entry["chosen_variant"] = None
                entry["usable"] = False
                entry["reason"] = str(exc)
            # numeric keys used by score items
            numeric = {}
            for level in range(1, 11):
                for variant in ("space", "bare"):
                    text = f" {level}" if variant == "space" else str(level)
                    numeric.setdefault(str(level), {})[variant] = len(
                        tokenizer.encode(text, add_special_tokens=False)
                    )
            entry["numeric_key_tokens"] = numeric
        except Exception as exc:  # pragma: no cover - network/model dependent
            entry["usable"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report["models"][model_id] = entry

    report["summary"] = {
        "n_models": len(models),
        "n_usable": sum(1 for e in report["models"].values() if e.get("usable")),
        "usable": sorted(m for m, e in report["models"].items() if e.get("usable")),
        "not_usable": sorted(m for m, e in report["models"].items() if not e.get("usable")),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
