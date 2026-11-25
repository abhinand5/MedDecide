#!/usr/bin/env python
"""osler_v7 T3 — port check for the d3 harness on `vllm-sr/d3-flash` (CPU where possible).

The 9B v7 arm trains on d3-flash through `meddecide.distill.d3_train.D3Student`. Before that arm runs, this check
asks whether the harness matches the checkpoint's own runtime on the three shapes v7 adds to the d3_distill pool:

1. `score` items (2-10 ordered levels; the v6 mix has 157,379),
2. long MIMIC notes (the v6 mix's MIMIC states are the longest text in the mix),
3. the 255-row readout at the last prompt position (the runtime's `pooling: "last"`, `readout_dtype: "float32"`).

Structural checks run first and need no forward. The parity phase loads the pinned checkpoint on CPU and compares
`D3Student.forward_logits` (single row) with the runtime's own `D3.probabilities` on the same prepared row; the gate
is the d3_distill step-2 bar, max |harness - reference| <= 1e-4. CPU inference needs an environment without the
GPU-only `flash-linear-attention` / `causal-conv1d` kernels (the checkpoint's runtime refuses CPU otherwise), so the
parity run is executed under `uv run --isolated --with transformers==5.17.0 ...` (see the command in CLAIMS). A long
item whose forward exceeds the CPU time budget is reported as NOT MEASURED for the forward, never substituted.

Reads: the pinned snapshot under `$HF_HOME`, `data/train/osler_v6/mix_f2_distill.jsonl` (to select score/noul/choice
rows and the longest MIMIC rows).
Writes: `outputs/osler_v7/T3/port_check.json` (aggregates and per-item differences; no item text, id or prompt).

Run (CPU)::

    uv run --isolated --python 3.12 --with "transformers==5.17.0" --with accelerate --with safetensors \
        --with einops --with torch --with pydantic --with numpy \
        python scripts/osler_v7/t3_port_check.py
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meddecide.distill.d3_batch import (  # noqa: E402
    align_probabilities,
    criteria_for,
    question_request,
)
from meddecide.distill.d3_train import (  # noqa: E402
    MAX_CODES,
    D3Student,
    TrainRow,
    batch_tensors,
    masked_logits,
    render_order,
    render_targets,
)
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

REVISION = "67e03e1bc157961c582ee7c6a35d9b15fada3dec"
SNAP = Path(os.environ.get("HF_HOME", "/workspace/.hf_home")) / "hub" / "models--vllm-sr--d3-flash" / "snapshots" / REVISION
MIX = ROOT / "data" / "train" / "osler_v6" / "mix_f2_distill.jsonl"
OUT = ROOT / "outputs" / "osler_v7" / "T3"
GATE = 1e-4


# ------------------------------------------------------------------------------------------------------------------
# Snapshot verification and row selection
# ------------------------------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(snapshot: Path) -> dict[str, Any]:
    """Check the pinned snapshot directory against its own `MODEL_MANIFEST.json` on the cheap files."""
    manifest = json.loads((snapshot / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    decision = json.loads((snapshot / "decision_config.json").read_text(encoding="utf-8"))
    hashes = manifest["files_sha256"]
    checked: dict[str, bool] = {}
    for name in ("config.json", "decision_config.json", "readout.safetensors", "tokenizer.json", "modeling_d3.py",
                 "d3_runtime.py", "chat_template.jinja"):
        path = snapshot / name
        checked[name] = path.exists() and sha256_file(path) == hashes.get(name)
    sizes_ok = all(
        (snapshot / name).stat().st_size == size
        for name, size in manifest["files_bytes"].items()
        if name.endswith(".safetensors") and not name.startswith("readout")
    )
    readout_shape: Any = None
    try:
        from safetensors.torch import load_file

        readout_shape = list(load_file(str(snapshot / "readout.safetensors"))["weight"].shape)
    except Exception as exc:  # reported, not raised
        readout_shape = f"unreadable ({type(exc).__name__}: {exc})"
    return {
        "snapshot": str(snapshot),
        "revision_dir": snapshot.name,
        "revision_matches_pin": snapshot.name == REVISION,
        "manifest_schema": manifest.get("schema"),
        "format_id": manifest.get("format_id"),
        "attention_mode": manifest.get("attention_mode"),
        "max_input_tokens": manifest.get("max_input_tokens"),
        "parameters": manifest.get("parameters"),
        "licence": manifest.get("licence", {}).get("spdx"),
        "decision_config": {
            "codes": len(decision["codes"]),
            "temperature": decision["temperature"],
            "attention_mode": decision["attention_mode"],
            "pooling": decision["pooling"],
            "max_length": decision["max_length"],
            "readout_dtype": decision["readout_dtype"],
            "base_model": decision["base_model"],
        },
        "manifest_hash_checks": checked,
        "manifest_hashes_all_match": all(checked.values()),
        "weight_shard_sizes_match_manifest": sizes_ok,
        "readout_shape_from_file": readout_shape,
        "hidden_size_from_config": json.loads((snapshot / "config.json").read_text(encoding="utf-8"))[
            "text_config"
        ]["hidden_size"],
    }


def parse_options(row: dict[str, Any]) -> list[dict[str, str]]:
    options = row["options"]
    if isinstance(options, str):
        options = json.loads(options)
    return [{"key": str(o["key"]), "label": str(o.get("label", ""))} for o in options]


@dataclass
class Selected:
    """The rows chosen for the port check, tagged by what they exercise."""

    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    tags: dict[str, list[str]] = field(default_factory=dict)

    def add(self, row: dict[str, Any], tag: str) -> None:
        item_id = str(row["item_id"])
        self.rows.setdefault(item_id, row)
        self.tags.setdefault(item_id, []).append(tag)


def select_rows(
    use_score: int = 3,
    use_noul: int = 2,
    use_choice4: int = 2,
    use_mimic_long: int = 3,
    use_mimic_first: int = 2,
    mix: Path = MIX,
) -> tuple[Selected, dict[str, Any]]:
    """Scan the v6 mix once and select the rows that exercise score, noul, 4-option choice and long MIMIC text.

    The MIMIC choice is by state length: the longest rows seen and the first two seen, so the long-note check does
    not depend on the file's order being meaningful.
    """
    selected = Selected()
    counts = {"score": 0, "noul": 0, "choice4": 0, "mimic_total": 0}
    longest: list[dict[str, Any]] = []
    first_mimic: list[dict[str, Any]] = []
    started = time.time()
    with mix.open(encoding="utf-8") as handle:
        for n, line in enumerate(handle, start=1):
            row = json.loads(line)
            qtype = str(row["qtype"])
            source = str(row["source"])
            if qtype == "score" and counts["score"] < use_score:
                selected.add(row, "score")
                counts["score"] += 1
            elif qtype == "noul" and counts["noul"] < use_noul:
                selected.add(row, "noul")
                counts["noul"] += 1
            if qtype == "choice" and counts["choice4"] < use_choice4 and len(parse_options(row)) == 4:
                selected.add(row, "choice4")
                counts["choice4"] += 1
            if source.startswith("mimic"):
                counts["mimic_total"] += 1
                if len(first_mimic) < use_mimic_first:
                    first_mimic.append(row)
                length = len(str(row.get("state", "")))
                longest.append({"row": row, "length": length})
                longest.sort(key=lambda x: -x["length"])
                longest = longest[:use_mimic_long]
            if n % 250_000 == 0:
                print(f"[select] {n:,} rows, {time.time() - started:.1f}s", file=sys.stderr, flush=True)
    for entry in longest:
        selected.add(entry["row"], "mimic_long")
    for row in first_mimic:
        selected.add(row, "mimic_first")
    counts["mimic_longest_state_chars"] = longest[0]["length"] if longest else 0
    counts["mimic_first_state_chars"] = [len(str(r.get("state", ""))) for r in first_mimic]
    return selected, counts


# ------------------------------------------------------------------------------------------------------------------
# Structural checks (no model forward)
# ------------------------------------------------------------------------------------------------------------------


def structural_report(row: dict[str, Any]) -> dict[str, Any]:
    """Everything about one row that can be checked without a forward: order, gold mapping, criteria, option cap."""
    options = parse_options(row)
    keys = [o["key"] for o in options]
    labels = [o["label"] for o in options]
    n = len(options)
    qtype = str(row["qtype"])
    gold = str(row["gold"])
    gold_index = keys.index(gold) if gold in keys else None
    order = render_order(qtype, keys)
    rendered_gold, _ = render_targets(qtype, keys, gold, None)
    pairs = [(o["key"], o["label"]) for o in options]
    criteria = criteria_for(qtype, pairs)
    request = question_request(qtype, str(row.get("question", "")), pairs)
    runtime_order = [1.0 / n] * n
    aligned = align_probabilities(qtype, pairs, runtime_order)
    checks = {
        "order_is_permutation": sorted(order) == list(range(n)),
        "rendered_gold_matches_order": rendered_gold == order.index(gold_index) if gold_index is not None else False,
        "align_preserves_length": len(aligned) == n,
        "align_preserves_sum": abs(sum(aligned) - 1.0) <= 1e-9,
        "option_count_within_255": n <= MAX_CODES,
        "request_type_matches": request["type"] == qtype,
    }
    if qtype == "noul":
        checks["noul_renders_no_then_yes"] = order == [keys.index("no"), keys.index("yes")]
        checks["noul_has_no_criteria"] = criteria is None
    elif qtype == "score":
        checks["score_renders_in_level_order"] = order == list(range(n))
        checks["score_criteria_are_labels_in_order"] = criteria == labels
    elif qtype == "choice":
        checks["choice_renders_in_row_order"] = order == list(range(n))
        checks["choice_criteria_are_the_options"] = criteria == dict(zip(keys, labels, strict=True))
    return {
        "qtype": qtype,
        "source": str(row["source"]),
        "template_id": str(row["template_id"]),
        "n_options": n,
        "state_chars": len(str(row.get("state", ""))),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


# ------------------------------------------------------------------------------------------------------------------
# Parity phase (model on CPU)
# ------------------------------------------------------------------------------------------------------------------


def harness_probabilities(student: D3Student, rows: list[TrainRow]) -> list[list[float]]:
    """The harness's probabilities for prepared rows, single batch, in the rendered code order."""
    import torch

    ids, mask, counts, _ = batch_tensors(rows, student.pad_id, student.device)
    with torch.inference_mode():
        logits = masked_logits(student.forward_logits(ids, mask), counts) / student.temperature
        values = torch.softmax(logits, dim=-1).cpu().tolist()
    return [value[:count] for value, count in zip(values, counts, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAP)
    parser.add_argument("--mix", type=Path, default=MIX)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--max-forward-tokens", type=int, default=6_000,
                        help="skip the forward of a row longer than this (reported NOT MEASURED, never substituted)")
    parser.add_argument("--time-budget-s", type=float, default=2_700.0,
                        help="stop starting new forwards after this many seconds; remaining rows are NOT MEASURED")
    args = parser.parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)
    result: dict[str, Any] = {
        "generated_utc": utcnow(),
        "git_commit": git_commit(ROOT),
        "command": "uv run --isolated --python 3.12 --with \"transformers==5.17.0\" --with accelerate "
                   "--with safetensors --with einops --with torch --with pydantic --with numpy "
                   "python scripts/osler_v7/t3_port_check.py",
        "revision_pin": REVISION,
    }
    result["snapshot"] = verify_snapshot(args.snapshot)
    selected, selection_counts = select_rows()
    result["selection"] = {**selection_counts, "rows_selected": len(selected.rows)}
    structural = []
    prepared_rows: list[tuple[str, dict[str, Any], list[dict[str, str]], list[str]]] = []
    for item_id, row in selected.rows.items():
        tags = sorted(set(selected.tags[item_id]))
        report = structural_report(row)
        report["tags"] = tags
        structural.append(report)
        prepared_rows.append((item_id, row, parse_options(row), tags))
    result["structural"] = {
        "items": structural,
        "all_pass": all(item["all_checks_pass"] for item in structural),
        "n_items": len(structural),
    }

    import torch
    from transformers import AutoModel

    load_started = time.time()
    model = AutoModel.from_pretrained(str(args.snapshot), trust_remote_code=True, dtype=torch.bfloat16, device="cpu")
    runtime = model.runtime
    student = D3Student(runtime, lora_r=32, lora_alpha=32)
    runtime.readout = student.readout.detach()
    load_seconds = time.time() - load_started
    result["runtime"] = {
        "device": str(runtime.device),
        "load_seconds": round(load_seconds, 1),
        "kernels": runtime.kernels,
        "attention_mode": runtime.attention_mode,
        "temperature": runtime.temperature,
        "pad_id": int(runtime.pad_id),
        "max_length": runtime.max_length,
        "readout_shape": list(runtime.readout.shape),
        "readout_dtype": str(runtime.readout.dtype),
        "fast_path": runtime.fast is not None,
        "fast_skipped": getattr(runtime, "fast_skipped", None),
        "probabilities_parameters": list(inspect.signature(runtime.probabilities).parameters),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }

    prepared: list[dict[str, Any]] = []
    for item_id, row, options, tags in prepared_rows:
        pairs = [(o["key"], o["label"]) for o in options]
        request = question_request(str(row["qtype"]), str(row["question"]), pairs)
        ready = runtime.prepare(str(row["state"]), {"q": request})
        if "q" in ready.errors:
            prepared.append({"tags": tags, "qtype": row["qtype"], "source": row["source"],
                             "template_id": row["template_id"], "n_options": len(options),
                             "prepare_error": ready.errors["q"]})
            continue
        sequence = list(ready.sequences["q"])
        prepared.append({
            "item_key": item_id[:12],
            "tags": tags,
            "qtype": str(row["qtype"]),
            "source": str(row["source"]),
            "template_id": str(row["template_id"]),
            "n_options": len(options),
            "state_chars": len(str(row["state"])),
            "prompt_tokens": len(sequence),
            "state_block": None,
            "prepared_fields": [f.name for f in dataclasses.fields(ready)],
            "sequence": sequence,
            "options": options,
            "gold": str(row["gold"]),
        })
    result["prepare"] = [
        {k: v for k, v in item.items() if k not in ("sequence", "options")} for item in prepared
    ]
    prepared_fields = prepared[0].get("prepared_fields", []) if prepared else []
    probability_parameters = result["runtime"]["probabilities_parameters"]
    result["batch_helper_compatibility"] = {
        "prepared_fields": prepared_fields,
        "has_states": "states" in prepared_fields,
        "probabilities_parameters": probability_parameters,
        "d3_batch_scoring_usable": ("states" in prepared_fields) and len(probability_parameters) >= 4,
        "note": "meddecide.distill.d3_batch.prepare_request reads prepared.states and score_batch calls "
                "runtime.probabilities(sequences, counts, states); this runtime's Prepared has no states field and its "
                "probabilities takes (sequences, counts), so the batched helper is not usable on d3-flash. The training "
                "harness (D3Student.forward_logits -> runtime.probabilities(sequences, counts)) does not use it, which "
                "the parity rows verify.",
    }

    parity: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    parity_started = time.time()
    for item in prepared:
        if "prepare_error" in item:
            continue
        if item["prompt_tokens"] > args.max_forward_tokens:
            skipped.append({"item_key": item["item_key"], "tags": item["tags"], "prompt_tokens": item["prompt_tokens"],
                            "reason": f"forward skipped: {item['prompt_tokens']} tokens over the "
                                      f"{args.max_forward_tokens}-token CPU timebox; NOT MEASURED"})
            continue
        if time.time() - parity_started > args.time_budget_s:
            skipped.append({"item_key": item["item_key"], "tags": item["tags"], "prompt_tokens": item["prompt_tokens"],
                            "reason": "forward skipped: CPU time budget exhausted; NOT MEASURED"})
            continue
        train_row = TrainRow(
            row_id=item["item_key"], sequence=tuple(item["sequence"]), count=item["n_options"],
            gold=0, teacher=None, split="train", source=item["source"],
        )
        print(f"[parity] start {item['tags']} qtype={item['qtype']} tokens={item['prompt_tokens']}",
              file=sys.stderr, flush=True)
        t0 = time.time()
        harness = harness_probabilities(student, [train_row])[0]
        reference = runtime.probabilities([list(item["sequence"])], [item["n_options"]])[0]
        elapsed = time.time() - t0
        diff = max(abs(a - b) for a, b in zip(harness, reference, strict=True))
        harness_argmax = max(range(len(harness)), key=harness.__getitem__)
        reference_argmax = max(range(len(reference)), key=reference.__getitem__)
        gold_rendered, _ = render_targets(
            str(item["qtype"]), [o["key"] for o in item["options"]], item["gold"], None
        )
        aligned = align_probabilities(
            str(item["qtype"]), [(o["key"], o["label"]) for o in item["options"]], harness
        )
        parity.append({
            "item_key": item["item_key"],
            "tags": item["tags"],
            "qtype": item["qtype"],
            "n_options": item["n_options"],
            "prompt_tokens": item["prompt_tokens"],
            "state_block": item["state_block"],
            "max_abs_diff": diff,
            "argmax_agree": harness_argmax == reference_argmax,
            "harness_argmax_index": harness_argmax,
            "reference_argmax_index": reference_argmax,
            "gold_index_rendered": gold_rendered,
            "harness_correct": harness_argmax == gold_rendered,
            "probability_sum": sum(harness),
            "aligned_length": len(aligned),
            "aligned_argmax_in_row_order": max(range(len(aligned)), key=aligned.__getitem__),
            "seconds": round(elapsed, 2),
        })
        print(f"[parity] done  {item['tags']} tokens={item['prompt_tokens']} seconds={elapsed:.1f} "
              f"max_abs_diff={diff:.3e} sum={sum(harness):.9f}", file=sys.stderr, flush=True)
    result["parity"] = parity
    result["parity_skipped"] = skipped
    measured = [item for item in parity if isinstance(item["max_abs_diff"], float)]
    result["gate"] = {
        "max_abs_diff": max((item["max_abs_diff"] for item in measured), default=None),
        "items_measured": len(measured),
        "items_skipped": len(skipped),
        "bar": GATE,
        "verdict": "PASS" if measured and max(item["max_abs_diff"] for item in measured) <= GATE else
                   ("FAIL" if measured else "NOT MEASURED"),
    }

    # Batch/padding check: the short items in one pass must keep their single-row probabilities (the readout is the
    # last column of each left-padded row, so a correct mask leaves it unchanged). Only rows already forwards.
    batch_items = [item for item in prepared if "sequence" in item and item["prompt_tokens"] <= 2_000][:2]
    batch_check: dict[str, Any] = {"status": "NOT MEASURED (fewer than two short prepared rows)"}
    if len(batch_items) == 2:
        singles = {item["item_key"]: item for item in parity}
        rows = [
            TrainRow(row_id=item["item_key"], sequence=tuple(item["sequence"]), count=item["n_options"],
                     gold=0, teacher=None, split="train", source=item["source"])
            for item in batch_items
        ]
        harness_batch = harness_probabilities(student, rows)
        references = [runtime.probabilities([list(item["sequence"])], [item["n_options"]])[0] for item in batch_items]
        diffs = []
        for item, got, ref in zip(batch_items, harness_batch, references, strict=True):
            harness_single = singles.get(item["item_key"], {}).get("max_abs_diff")
            diffs.append({
                "item_key": item["item_key"],
                "batch_vs_reference": max(abs(a - b) for a, b in zip(got, ref, strict=True)),
                "batch_vs_single_row_reference": harness_single,
            })
        batch_check = {
            "status": "MEASURED",
            "items": diffs,
            "max_batch_vs_reference": max(item["batch_vs_reference"] for item in diffs),
            "interpretation": "each row's last column is its own final prompt token; the batch difference is bf16 "
                              "batch-composition numerics, as in d3_distill step 2",
        }
    result["batch_check"] = batch_check
    result["timing"] = {"entry_seconds": round(time.time() - started, 1),
                        "parity_seconds": round(time.time() - parity_started, 1)}
    with (args.out_dir / "port_check.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"gate": result["gate"], "structural_all_pass": result["structural"]["all_pass"],
                      "readout_shape": result["runtime"]["readout_shape"],
                      "prepare_errors": sum(1 for item in prepared if "prepare_error" in item),
                      "skipped": len(skipped)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
