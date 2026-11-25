"""d3_distill step 0, batched: teacher throughput and answer validity on 1,000 MedMCQA training items.

The path is the model's own code, batched over requests: ``d3_runtime.D3.prepare`` renders and tokenizes one
one-question request per item (the card's one-question-per-request shape), then ``D3.probabilities`` runs the
model's own ``prefix_bidir`` block mask, forward, FP32 readout and softmax on a left-padded batch of those
requests (``meddecide.distill.d3_batch``). The single-request ``system_one`` API is used only for the 40-item
parity check, never for the gate.

Gate (PLAN.md step 0): >= 25 items/s on 1,000 items, with the probability check (sum to 1, one entry per
option, non-constant) passing for every item. Writes aggregates plus the per-item probabilities (gitignored).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# Keep Triton autotune results on disk (d3_engine.py does the same), so warm-up results persist across processes.
os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")

import torch
from transformers import AutoModel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from meddecide.distill.d3_batch import (  # noqa: E402
    argmax_index,
    distribution_ok,
    length_batches,
    non_constant,
    padded_tokens,
    prepare_choice,
    score_batch,
)

INSTRUCTIONS = "Which option is the correct answer?"


def utcnow() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def medmcqa_items(n: int) -> list[dict]:
    """The first ``n`` MedMCQA training rows, in dataset order (the same rows as step0_throughput.py)."""
    from datasets import load_dataset

    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    items = []
    for i in range(n):
        r = ds[i]
        items.append(
            {
                "id": f"mcq{i}",
                "stem": r["question"],
                "options": {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]},
                "gold": "ABCD"[r["cop"]],
            }
        )
    return items


def environment(snapshot: Path) -> dict:
    import transformers

    try:
        import fla

        fla_version = getattr(fla, "__version__", "?")
    except Exception as exc:  # reported, not fatal
        fla_version = f"missing ({type(exc).__name__})"
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {
        "git_commit": commit,
        "snapshot": str(snapshot),
        "revision": snapshot.name,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "fla": fla_version,
        "gpu": torch.cuda.get_device_name(0),
        "offline": os.environ.get("HF_HUB_OFFLINE"),
    }


def load(snapshot: Path):
    t0 = time.time()
    model = AutoModel.from_pretrained(
        str(snapshot), trust_remote_code=True, dtype=torch.bfloat16, device="cuda:0"
    )
    return model, time.time() - t0


def batched_probs(model, items: list[dict], *, max_rows: int, max_tokens: int):
    """Prepare every one-question request, then score it in length-sorted padded batches."""
    runtime = model.runtime
    t0 = time.time()
    prepared = [
        prepare_choice(runtime, i, it["stem"], INSTRUCTIONS, it["options"])
        for i, it in enumerate(items)
    ]
    prepare_s = time.time() - t0
    batches = length_batches(prepared, max_rows=max_rows, max_tokens=max_tokens)
    probs: list[list[float] | None] = [None] * len(items)
    batch_rows = []
    forward_s = 0.0
    with runtime.request_temperature(1):  # every request holds one question
        for batch in batches:
            torch.cuda.synchronize()
            t1 = time.time()
            values = score_batch(runtime, batch)
            torch.cuda.synchronize()
            elapsed = time.time() - t1
            forward_s += elapsed
            batch_rows.append(
                {
                    "rows": len(batch),
                    "padded_tokens": padded_tokens(batch),
                    "longest": max(r.length for r in batch),
                    "seconds": round(elapsed, 4),
                }
            )
            for row, value in zip(batch, values, strict=True):
                probs[row.index] = value
    return prepared, batches, probs, prepare_s, forward_s, batch_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--max-rows", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--warmup-n", type=int, default=24)
    ap.add_argument("--parity-n", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows-out", type=Path, default=None)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    items = medmcqa_items(args.n)
    model, load_s = load(args.snapshot.resolve())
    env = environment(args.snapshot.resolve())

    # Warm-up: the same path on a few rows pays Triton autotune and kernel-cache costs once. Labelling 200k
    # items sees the steady state, so the timed gate run starts after this.
    warm = batched_probs(
        model, items[: args.warmup_n], max_rows=args.max_rows, max_tokens=args.max_tokens
    )
    warm_s = warm[3] + warm[4]
    torch.cuda.reset_peak_memory_stats()
    print(f"[{utcnow()}] warm-up: {warm_s:.2f}s over {args.warmup_n} rows", flush=True)

    # The gate run: 1,000 items, timed end to end (prepare + batched forward).
    t_gate = time.time()
    prepared, batches, probs, prepare_s, forward_s, batch_rows = batched_probs(
        model, items, max_rows=args.max_rows, max_tokens=args.max_tokens
    )
    end_to_end_s = time.time() - t_gate
    assert [p.index for p in prepared] == list(range(len(items)))

    invalid, constant, correct, sums = 0, 0, 0, []
    rows = []
    for it, row, value in zip(items, prepared, probs, strict=True):
        ok = distribution_ok(value, row.count) and non_constant(value)
        invalid += int(not distribution_ok(value, row.count))
        constant += int(not non_constant(value))
        sums.append(sum(value) if value else float("nan"))
        best = argmax_index(value)
        gold_index = list(it["options"]).index(it["gold"])
        correct += int(best == gold_index)
        rows.append(
            {
                "id": it["id"],
                "gold": it["gold"],
                "option_keys": list(it["options"]),
                "probs": value,
                "argmax": list(it["options"])[best],
                "ok": ok,
                "input_tokens": row.length,
            }
        )
    rows_out = args.rows_out or args.out.parent / "batched_probs.jsonl"
    rows_out.parent.mkdir(parents=True, exist_ok=True)
    rows_out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )

    # Parity (after the gate, so the timed run is untouched): the batched reference against the card's
    # single-request path on the same 40 items.
    parity_n = min(args.parity_n, len(items))
    sub = items[:parity_n]
    _, _, parity_probs, _, _, _ = batched_probs(
        model, sub, max_rows=args.max_rows, max_tokens=args.max_tokens
    )
    diffs, agree = [], 0
    for it, batched in zip(sub, parity_probs, strict=True):
        answer = model.system_one(
            state=it["stem"],
            questions={
                "q": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": it["options"]}
            },
        )["answers"]["q"]
        single = [float(answer["probabilities"][k]) for k in it["options"]]
        diffs.append(max(abs(a - b) for a, b in zip(batched, single, strict=True)))
        agree += int(argmax_index(batched) == argmax_index(single))
    parity = {
        "n": parity_n,
        "max_abs_diff": max(diffs),
        "mean_abs_diff": sum(diffs) / len(diffs),
        "argmax_agreement": agree,
        "within_1e-4": max(diffs) <= 1e-4,
    }
    print(f"[{utcnow()}] parity: {json.dumps(parity)}", flush=True)

    rate_e2e = args.n / end_to_end_s
    rate_forward = args.n / forward_s
    result = {
        "n": args.n,
        "path": "D3.prepare one request per item; D3.probabilities over left-padded batches "
        "(the runtime's own prefix_bidir mask, forward, readout and softmax)",
        "max_rows": args.max_rows,
        "max_tokens": args.max_tokens,
        "parity_with_system_one": parity,
        "warmup_n": args.warmup_n,
        "warmup_s": round(warm_s, 3),
        "prepare_s": round(prepare_s, 3),
        "forward_s": round(forward_s, 3),
        "end_to_end_s": round(end_to_end_s, 3),
        "items_per_s_end_to_end": round(rate_e2e, 3),
        "items_per_s_forward": round(rate_forward, 3),
        "batches": len(batches),
        "mean_rows_per_batch": round(sum(b["rows"] for b in batch_rows) / len(batch_rows), 2),
        "mean_padded_tokens_per_batch": round(
            sum(b["padded_tokens"] for b in batch_rows) / len(batch_rows), 1
        ),
        "padded_tokens_per_row": round(sum(b["padded_tokens"] for b in batch_rows) / args.n, 1),
        "invalid_distributions": invalid,
        "constant_distributions": constant,
        "sum_min": round(min(sums), 8),
        "sum_max": round(max(sums), 8),
        "dev_accuracy_diagnostic": round(correct / args.n, 4),
        "load_s": round(load_s, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "rows_file": str(rows_out),
        "finished_utc": utcnow(),
        "gate_25_items_per_s": "PASS"
        if rate_e2e >= 25 and invalid == 0 and constant == 0
        else "FAIL",
        "gate_basis": "items_per_s_end_to_end includes prepare; the forward-only rate is reported separately",
        **env,
    }
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "n",
                    "items_per_s_end_to_end",
                    "items_per_s_forward",
                    "invalid_distributions",
                    "constant_distributions",
                    "dev_accuracy_diagnostic",
                    "gate_25_items_per_s",
                    "peak_gpu_gb",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
