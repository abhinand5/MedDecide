"""osler_v2 (D39): zero-shot pplx-decider-flash-v1.2 scored through its authors' code (the reference path), in the O2 format.

pplx is the candidate base and, if chosen, the substitute baseline for zero-shot Qwen3.5-4B in Gate O3 (D39). Its own code
(`pplx_decider`, from the model repository) reads one question per request, with the item's option labels as criteria in
their order (noul items as Yes / No, as the O2 runners render them). The returned logits are turned into probabilities over
the offered options by a softmax, and written with the O2 prediction shape (probabilities aligned with the item's own keys).

Run with torchvision and pillow added for this run only (pplx's processor needs them; never added to the project):
    source scripts/pod_env.sh && uv run --frozen --with pillow --with "torchvision==0.29.*" \
        python scripts/osler_v2/o2_run_pplx_zeroshot.py --scope mix_v2 --items data/bench/osler_v2/knowledge_dev_v2.jsonl \
        --out-root outputs/osler_v2/W5/zeroshot_pplx
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from meddecide.eval.o2_io import (
    MIX_V2,
    MR_V0,
    PANEL_ROBUSTNESS,
    Scope,
    completed_ids,
    iter_items,
    open_predictions,
    prediction_row,
    write_row,
)
from meddecide.model.pplx import REPO, checkpoint_dir
from meddecide.utils.io import write_json
from meddecide.utils.provenance import gpu_name, utcnow

SCOPES = {"mr_v0": MR_V0, "mix_v2": MIX_V2, "panel_robustness": PANEL_ROBUSTNESS}


def request_for(row: dict) -> tuple[dict | None, list[str]]:
    """The reference request and the option labels in order (noul yes/no rendered Yes / No, as the O2 runners do)."""
    original = [o["key"] for o in row["options"]]
    labels = [o["label"] for o in row["options"]]
    if row["qtype"] == "noul" and original == ["yes", "no"]:
        labels = ["Yes", "No"]
    if len(set(labels)) != len(labels):  # criteria are keyed by label: a repeated label cannot be scored by the reference
        return None, original
    request = {"state": row["state"], "questions": {"q": {"type": "choice", "instructions": row["question"],
                                                          "criteria": dict.fromkeys(labels)}}}
    return request, original


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scope", choices=sorted(SCOPES), default=None)
    ap.add_argument("--benchmarks", nargs="+", default=None, help="benchmarks to score (e.g. ext_panel, or v0.2 with --sets)")
    ap.add_argument("--sets", nargs="+", default=None, help="keep only these set names (e.g. medqa medmcqa)")
    ap.add_argument("--items", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if (args.scope is None) == (args.benchmarks is None):
        raise SystemExit("give --scope or --benchmarks (not both)")
    scope = SCOPES[args.scope] if args.scope else Scope("custom", tuple(args.benchmarks))
    keep_sets = set(args.sets) if args.sets else None
    out_dir = Path(args.out_root) / "pplx-decider-flash-v1.2"
    out_path = out_dir / "predictions.jsonl"

    ckpt = checkpoint_dir(REPO)
    sys.path.insert(0, str(ckpt / "source" / "src"))
    from pplx_decider import Decider  # the authors' code, from the checkpoint

    ref = Decider.from_pretrained(ckpt, device="cuda")
    write_json(out_dir / "run.json", {"model": REPO, "checkpoint_dir": str(ckpt), "reference": "pplx_decider.Decider",
                                      "criteria": "option labels in order; softmax over returned logits",
                                      "scope": scope.name, "items": args.items, "gpu": gpu_name(), "utc": utcnow()})
    done = completed_ids(out_path)
    print(f"[{utcnow()}] pplx reference on {gpu_name()}; resume_rows={len(done)} -> {out_path}", flush=True)
    n_new = 0
    n_skipped = 0
    t0 = time.time()
    with open_predictions(out_path) as fh:
        for row in iter_items(Path(args.items), scope):
            if row["item_id"] in done or (keep_sets is not None and row.get("set_name") not in keep_sets):
                continue
            if args.limit and n_new >= args.limit:
                break
            request, original = request_for(row)
            if request is None:
                write_row(fh, prediction_row(row, status="skipped", reason="duplicate_option_labels"))
                n_skipped += 1
                continue
            t_item = time.time()
            logits = [float(x) for x in ref.logits([request])[0]["q"]]
            if len(logits) != len(original):
                raise RuntimeError(f"{row['item_id']}: {len(logits)} logits for {len(original)} options")
            write_row(fh, prediction_row(
                row, status="scored", probs=softmax(logits), latency_s=round(time.time() - t_item, 4),
                prompt_tokens=None, extras={"readout": "reference logits, softmax over offered options",
                                            "original_keys": original}))
            n_new += 1
            if n_new % 500 == 0:
                print(f"[{utcnow()}] scored {n_new} new rows ({n_new / (time.time() - t0):.1f} items/s)", flush=True)
    print(f"[{utcnow()}] done: new rows={n_new}; skipped (duplicate option labels)={n_skipped}; "
          f"total rows={len(completed_ids(out_path))}", flush=True)


if __name__ == "__main__":
    main()
