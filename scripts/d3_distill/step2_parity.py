"""d3_distill step 2, port parity: a student's trainable harness must reproduce its own decision probabilities.

The student is loaded through the card's ``AutoModel`` path. For 40 pool items (one one-question request per item, as
the card requires) we compare:

  reference  the student runtime's own ``D3.probabilities`` on each item (one request, one row — its inference path);
  harness    ``D3Student.forward_logits`` — the same ``D3.encoder`` and readout, built by the training harness, one row.

The gate (ADVISORY §1) is max |harness - reference| ≤ 1e-4 at the reference's own one-row shape. Additional diagnostics:
the same comparison with 8 rows in one pass (bf16 batch-composition numerics, the same phenomenon accepted for the
teacher) and against ``system_one``. Writes a JSON with the per-item differences.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from meddecide.distill.d3_train import D3Student, TrainRow, batch_tensors, masked_logits  # noqa: E402

POOL = REPO / "data" / "train" / "d3_distill" / "pool_200k.jsonl"
PARITY_N = 40


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def criteria(row: dict) -> dict:
    if row["qtype"] == "choice":
        return {"type": "choice", "instructions": row["question"],
                "criteria": {o["key"]: o["label"] for o in row["options"]}}
    if row["qtype"] == "noul":
        return {"type": "noul", "instructions": row["question"]}
    return {"type": "score", "instructions": row["question"], "criteria": [o["label"] for o in row["options"]]}


def system_answer(model, row: dict) -> list[float]:
    answer = model.system_one(state=row["state"], questions={"q": criteria(row)})["answers"]["q"]
    options = [(o["key"], o["label"]) for o in row["options"]]
    if row["qtype"] == "choice":
        return [float(answer["probabilities"][o["key"]]) for o in row["options"]]
    if row["qtype"] == "noul":
        # The harness probabilities are in the readout's rendered order: d3 renders noul as false (No), true (Yes).
        p_yes = float(answer["noul"])
        return [1.0 - p_yes, p_yes]
    return [float(answer["probabilities"][str(i)]) for i in range(len(options))]


def harness_probs(student: D3Student, rows: list[TrainRow]) -> list[list[float]]:
    ids, mask, counts, _ = batch_tensors(rows, student.pad_id, student.device)
    with torch.inference_mode():
        logits = masked_logits(student.forward_logits(ids, mask), counts) / student.temperature
        values = torch.softmax(logits, dim=-1).cpu().tolist()
    return [value[:count] for value, count in zip(values, counts, strict=True)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--n", type=int, default=PARITY_N)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.pool)[: args.n]
    model = AutoModel.from_pretrained(
        str(args.snapshot.resolve()), trust_remote_code=True, dtype=torch.bfloat16, device="cuda:0"
    )
    runtime = model.runtime
    student = D3Student(runtime, lora_r=32, lora_alpha=32).to(runtime.device)
    runtime.readout = student.readout.detach()

    prepared = []
    for row in rows:
        options = [(o["key"], o["label"]) for o in row["options"]]
        ready = runtime.prepare(row["state"], {"q": criteria(row)})
        if "q" in ready.errors:
            raise SystemExit(f"{row['row_id']}: {ready.errors['q']['message']}")
        prepared.append(
            TrainRow(
                row_id=row["row_id"],
                sequence=tuple(ready.sequences["q"]),
                count=len(options),
                gold=0,
                teacher=None,
                split=row["split"],
                source=row["source"],
            )
        )

    # The reference call is one request = one row; compare the harness the same way (a batch changes bf16 numerics).
    single = [harness_probs(student, [row])[0] for row in prepared]
    diffs, diffs_system, argmax_agree = [], [], 0
    for i, (row, harness) in enumerate(zip(rows, single, strict=True)):
        reference = runtime.probabilities([list(prepared[i].sequence)], [prepared[i].count])[0]
        diffs.append(max(abs(a - b) for a, b in zip(harness, reference, strict=True)))
        system = system_answer(model, row)
        diffs_system.append(max(abs(a - b) for a, b in zip(harness, system, strict=True)))
        argmax_agree += int(max(range(len(harness)), key=harness.__getitem__) == max(range(len(reference)), key=reference.__getitem__))

    batch = harness_probs(student, prepared[:8])
    diffs_batch8 = []
    for i, harness in enumerate(batch):
        reference = runtime.probabilities([list(prepared[i].sequence)], [prepared[i].count])[0]
        diffs_batch8.append(max(abs(a - b) for a, b in zip(harness, reference, strict=True)))

    result = {
        "student": str(args.snapshot),
        "revision": args.snapshot.name,
        "items": len(prepared),
        "temperature": student.temperature,
        "attention_mode": runtime.attention_mode,
        "rows": rows,
        "single_row": {
            "max_abs_diff": max(diffs),
            "mean_abs_diff": sum(diffs) / len(diffs),
            "argmax_agreement": argmax_agree,
            "per_item": diffs,
        },
        "batch8": {"max_abs_diff": max(diffs_batch8), "per_item": diffs_batch8},
        "system_one_diagnostic": {"max_abs_diff": max(diffs_system)},
        "gate_within_1e-4": max(diffs) <= 1e-4,
        "finished_utc": utcnow(),
        "git_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("items", "gate_within_1e-4", "single_row", "batch8",
                                             "system_one_diagnostic", "peak_gpu_gb")}, indent=1)[:3000], flush=True)


if __name__ == "__main__":
    main()
