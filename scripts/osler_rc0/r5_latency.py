#!/usr/bin/env python
"""osler_rc0 R5 — standardised latency, throughput and footprint benchmark (GPU, one model at a time).

A fixed, deterministic 500-item sample (250 `panel` + 250 `mr_verdict`, stratified by template, seed 0) is scored on a
single GPU in bfloat16. Ten warm-up items are excluded; every forward is surrounded by ``torch.cuda.synchronize()``.
Batch 1 measures online single-request serving (p50/p95/mean ms per item, items/s); batch 16 measures batched throughput.
Peak allocated VRAM and the total parameter count are recorded per model.

Models reachable through the two native decision-model paths are measured directly: the pplx option-code path
(`PplxDecisionModel`: pplx-flash, pplx-27B, Osler-4B-rc0 with its LoRA checkpoint) and the d3 path (`D3Student`:
d3-flash, Osler-9B-rc0 with `step_063304.pt`). A competitor whose own runner needs custom wiring beyond the ADVISORY's
5-minute per-model timebox is recorded ``NOT MEASURED — runner timebox`` and never imputed.

Run (GPU; one model per invocation, detached)::

    source scripts/pod_env.sh && uv run --frozen python scripts/osler_rc0/r5_latency.py \
        --model Osler-4B-rc0 --sample outputs/osler_rc0/R5/latency_sample_manifest.json \
        --out outputs/osler_rc0/R5/parts/Osler-4B-rc0.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.train.mix_items import mix_row_to_item  # noqa: E402
from meddecide.utils.provenance import gpu_name, utcnow  # noqa: E402

SETS = {
    "panel": ROOT / "data/bench/osler_v4/y9/panel.jsonl",
    "mr_verdict": ROOT / "data/bench/osler_v4/y9/mr_verdict.jsonl",
}
SAMPLE_PER_SET = 250
WARMUP = 10
SEED = 0

PPLX_REV = "6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d"
PPLX27_REV = "cce0f3f80033e9d89d80831d5281fd72124aacce"
D3_REV = "67e03e1bc157961c582ee7c6a35d9b15fada3dec"
HF = Path("/workspace/.hf_home/hub")

# the ADVISORY's 8-model latency roster; the three whose authors' runners expose only a single-item loop are recorded
# NOT MEASURED rather than wrapped beyond the timebox
UNWIRED = {
    "MedDecider-4B": "NOT MEASURED — runner timebox (authors' two-order single-item loop; no batching path wired)",
    "MedDecider-9B": "NOT MEASURED — runner timebox (authors' two-order single-item loop; no batching path wired)",
    "Clef-Flash": "NOT MEASURED — runner timebox (authors' isolated-venv systemone loop; no batching path wired)",
}

MODELS: dict[str, dict[str, Any]] = {
    "Osler-4B-rc0": {"backend": "pplx", "checkpoint": ROOT / "outputs/osler_rc0/R1/arm/checkpoints/final_step_063304"},
    "Osler-9B-rc0": {"backend": "d3", "checkpoint": ROOT / "outputs/osler_v7/T11/final/checkpoints/step_063304.pt"},
    "pplx-flash": {"backend": "pplx", "checkpoint": HF / "models--perplexity-ai--pplx-decider-flash-v1.2/snapshots" / PPLX_REV},
    "d3-flash": {"backend": "d3", "checkpoint": None},
    "pplx-27B": {"backend": "pplx", "checkpoint": HF / "models--perplexity-ai--pplx-decider-v1.2-27b/snapshots" / PPLX27_REV},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_sample(out: Path) -> dict[str, Any]:
    """250 items per set, stratified by template with proportional allocation, seed 0; no text is written."""
    rng = np.random.default_rng(SEED)
    manifest: dict[str, Any] = {"seed": SEED, "per_set": SAMPLE_PER_SET, "sets": {}, "items": []}
    for set_name, path in SETS.items():
        rows = read_jsonl(path)
        groups: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            groups.setdefault(str(row["template_id"]), []).append(index)
        total = len(rows)
        allocation: dict[str, int] = {t: max(1, round(SAMPLE_PER_SET * len(v) / total)) for t, v in groups.items()}
        # correct rounding drift on the largest groups
        diff = SAMPLE_PER_SET - sum(allocation.values())
        order = sorted(groups, key=lambda t: -len(groups[t]))
        for template in order:
            if diff == 0:
                break
            step = 1 if diff > 0 else -1
            if allocation[template] + step >= 1:
                allocation[template] += step
                diff -= step
        chosen: list[int] = []
        for template, count in sorted(allocation.items()):
            members = np.asarray(groups[template], dtype=np.int64)
            picked = rng.choice(members, size=min(count, members.size), replace=False)
            chosen.extend(int(i) for i in sorted(picked.tolist()))
        manifest["sets"][set_name] = {"universe": total, "sampled": len(chosen),
                                      "templates": {t: allocation[t] for t in sorted(allocation)}}
        for index in chosen:
            manifest["items"].append({"set": set_name, "index": index, "item_id": str(rows[index]["item_id"]),
                                      "template_id": str(rows[index]["template_id"])})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def summarise(times_ms: list[float], n_items: int, total_s: float) -> dict[str, Any]:
    values = np.asarray(times_ms, dtype=np.float64)
    return {
        "n_items": n_items,
        "n_forwards": len(times_ms),
        "p50_ms_per_item": round(float(np.percentile(values, 50)), 3),
        "p95_ms_per_item": round(float(np.percentile(values, 95)), 3),
        "mean_ms_per_item": round(float(values.mean()), 3),
        "items_per_s": round(n_items / total_s, 3) if total_s > 0 else None,
        "total_seconds": round(total_s, 3),
    }


def param_count(module: Any) -> int | None:
    try:
        return int(sum(p.numel() for p in module.parameters()))
    except Exception:
        return None


class PplxBackend:
    kind = "pplx"

    def __init__(self, spec: dict[str, Any]) -> None:
        from meddecide.model.pplx import PplxDecisionModel
        from meddecide.train.config import load_config

        config = load_config(ROOT / "configs/osler_v7/arm_4b.yaml")
        checkpoint = Path(spec["checkpoint"])
        if (checkpoint / "model.json").is_file():  # a saved Osler checkpoint directory (adapter + optcode + temperatures)
            self.model = PplxDecisionModel.load(str(checkpoint), device="cuda:0", dtype="bfloat16")
        else:
            kwargs: dict[str, Any] = {"dtype": "bfloat16", "device": "cuda:0"}
            if spec.get("lora"):
                kwargs["lora"] = config.lora
            self.model = PplxDecisionModel(str(checkpoint), **kwargs)
        self.model.eval_mode()
        self.module = self.model.peft_model if getattr(self.model, "peft_model", None) is not None else self.model.base
        self.items = None

    def load_items(self, rows: list[dict[str, Any]]) -> None:
        self.items = [mix_row_to_item(row) for row in rows]

    def forward(self, batch_size: int) -> None:
        self.model.score_items(self.items, batch_size=batch_size, max_batch_tokens=32768)


class D3Backend:
    kind = "d3"

    def __init__(self, spec: dict[str, Any]) -> None:
        from transformers import AutoModel

        from meddecide.distill.d3_train import D3Student

        snapshot = HF / "models--vllm-sr--d3-flash/snapshots" / D3_REV
        model = AutoModel.from_pretrained(str(snapshot), trust_remote_code=True, dtype=torch.bfloat16, device="cuda:0")
        self.runtime = model.runtime
        self.student = D3Student(self.runtime, lora_r=32, lora_alpha=32).to(self.runtime.device)
        base_readout = self.runtime.readout.detach().clone()
        if spec.get("checkpoint") is None:
            self.student.readout.data.copy_(base_readout)
            for adapter in self.student.adapters.values():
                adapter.scaling = 0.0
        else:
            payload = torch.load(spec["checkpoint"], map_location=self.student.device, weights_only=False)
            self.student.load_checkpoint_state(payload["model"])
        self.student.eval()
        self.module = self.student
        self.prepared = None

    def load_items(self, rows: list[dict[str, Any]]) -> None:
        from meddecide.distill.d3_train import prepare_rows

        def to_d3_row(row: dict[str, Any]) -> dict[str, Any]:
            options = row["options"] if isinstance(row["options"], list) else json.loads(row["options"])
            return {"row_id": mix_row_to_item(row).item_id, "source": str(row["source"]),
                    "template_id": str(row["template_id"]), "qtype": str(row["qtype"]), "state": str(row["state"]),
                    "question": str(row["question"]), "gold": str(row["gold"]), "split": "eval",
                    "options": [{"key": str(o["key"]), "label": str(o["label"])} for o in options]}

        self.prepared = prepare_rows(self.runtime, [to_d3_row(row) for row in rows])

    def forward(self, batch_size: int) -> None:
        self.student.probabilities([list(row.sequence) for row in self.prepared], [row.count for row in self.prepared],
                                   batch_size=batch_size, max_tokens=32768)


def measure(backend: Any, n_items: int, batch_size: int) -> dict[str, Any]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times: list[float] = []
    started = time.perf_counter()
    for start in range(WARMUP, n_items, batch_size):
        stop = min(start + batch_size, n_items)
        if backend.kind == "pplx":
            backend.items = backend.all_items[start:stop]
        else:
            backend.prepared = backend.all_prepared[start:stop]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        backend.forward(batch_size)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed / (stop - start))
    total = time.perf_counter() - started
    result = summarise(times, n_items - WARMUP, total)
    result["batch_size"] = batch_size
    result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    return result


def merge_parts(parts_dir: Path, out: Path) -> None:
    """Merge the per-model part files into the single latency.json the leaderboard reads."""
    models: dict[str, Any] = {}
    not_measured: dict[str, str] = {}
    for path in sorted(parts_dir.glob("*.json")):
        entry = json.loads(path.read_text())
        name = str(entry.get("model"))
        if entry.get("status") == "MEASURED":
            models[name] = entry
        else:
            not_measured[name] = str(entry.get("status"))
    for name in MODELS:
        if name not in models and name not in not_measured:
            not_measured[name] = "NOT MEASURED — runner timebox (authors' single-item runner; no batching path wired)"
    for name, reason in UNWIRED.items():
        if name not in models:
            not_measured.setdefault(name, reason)
    payload = {"generated_utc": utcnow(), "gpu": gpu_name(), "sample": str(parts_dir.parent / "latency_sample_manifest.json"),
               "dtype": "bfloat16", "models": models, "not_measured": not_measured}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"measured": sorted(models), "not_measured": not_measured}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=False, choices=sorted(MODELS))
    parser.add_argument("--merge", action="store_true", help="merge outputs/osler_rc0/R5/parts/*.json into latency.json")
    parser.add_argument("--sample", type=Path, default=ROOT / "outputs/osler_rc0/R5/latency_sample_manifest.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.merge:
        merge_parts(ROOT / "outputs/osler_rc0/R5/parts", args.out or ROOT / "outputs/osler_rc0/R5/latency.json")
        return
    if not args.model:
        raise SystemExit("--model is required unless --merge is given")
    sample = args.sample if args.sample.is_file() else None
    manifest = json.loads(sample.read_text()) if sample else build_sample(ROOT / "outputs/osler_rc0/R5/latency_sample_manifest.json")
    rows: list[dict[str, Any]] = []
    per_set_rows = {name: read_jsonl(path) for name, path in SETS.items()}
    for entry in manifest["items"]:
        rows.append(per_set_rows[entry["set"]][entry["index"]])
    spec = MODELS[args.model]
    out = args.out or ROOT / f"outputs/osler_rc0/R5/parts/{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result: dict[str, Any] = {"model": args.model, "utc": utcnow(), "gpu": gpu_name(),
                              "dtype": "bfloat16", "sample_items": len(rows), "warmup_items": WARMUP,
                              "sample_manifest": str(args.sample)}
    try:
        backend = PplxBackend(spec) if spec["backend"] == "pplx" else D3Backend(spec)
        result["load_seconds"] = round(time.perf_counter() - started, 2)
        backend.load_items(rows)
        if backend.kind == "pplx":
            backend.all_items = backend.items
            backend.items = backend.all_items[:WARMUP]
        else:
            backend.all_prepared = backend.prepared
            backend.prepared = backend.all_prepared[:WARMUP]
        # warm-up on the first WARMUP items (excluded from every reported number)
        backend.forward(1)
        for batch_size in (1, 16):
            result[f"batch{batch_size}"] = measure(backend, len(rows), batch_size)
        result["params_total"] = param_count(backend.module)
        result["status"] = "MEASURED"
    except Exception as error:
        result["status"] = f"NOT MEASURED — {type(error).__name__}: {error}"
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
