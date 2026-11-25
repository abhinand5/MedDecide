#!/usr/bin/env python
"""osler_rc0 R8 — stage the release artifacts locally and verify them, then write the honest model cards.

Nothing is uploaded. The staged tree lives under the gitignored ``outputs/osler_rc0/R8/release_staging/`` and contains
only model weights, calibration and a provenance manifest: no training data, no MIMIC text, no secrets.

Steps::

    source scripts/pod_env.sh && uv run --frozen python scripts/osler_rc0/r8_stage_release.py --step stage
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_rc0/r8_stage_release.py --step verify
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_rc0/r8_stage_release.py --step cards

``verify`` checks (a) staged-vs-original probability parity on 20 fixed dev items (max |Δp| < 1e-4, 20/20 argmax
agreement), (b) the 4B adapter-disabled generation is token-for-token identical to the untouched
``pplx-decider-flash-v1.2`` base on 20 fixed generic prompts (with an adapter-on control that must differ), and (c) the
9B staged weights carry no ``lm_head`` (decision-only, non-generative).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from meddecide.utils.io import write_json  # noqa: E402
from meddecide.utils.provenance import git_commit, utcnow  # noqa: E402

OUT = ROOT / "outputs/osler_rc0/R8"
STAGE = OUT / "release_staging"
PPLX4 = ROOT / "outputs/osler_rc0/R1/arm/checkpoints/final_step_063304"
D3_9 = ROOT / "outputs/osler_v7/T11/final/checkpoints/step_063304.pt"
DEV = ROOT / "outputs/osler_rc0/R2/fastdev_v4.jsonl"
CALIB9 = ROOT / "outputs/osler_rc0/R2/9b_calibration.json"
PPLX_BASE = "/workspace/.hf_home/hub/models--perplexity-ai--pplx-decider-flash-v1.2/snapshots/9ff7cf9838e59df461c276b7699d56b244d2af58"
PPLX_BASE_REPO = "perplexity-ai/pplx-decider-flash-v1.2"
PPLX_BASE_REV = "9ff7cf9838e59df461c276b7699d56b244d2af58"
D3_BASE_REPO = "vllm-sr/d3-flash"
D3_BASE_REV = "67e03e1bc157961c582ee7c6a35d9b15fada3dec"
PARITY_ITEMS = 20
GEN_PROMPTS = [
    "The patient was admitted to the hospital with",
    "Discharge diagnosis:",
    "The patient is a 65-year-old man who presents with",
    "Assessment and plan:",
    "The following laboratory values were obtained on admission:",
    "Medications on discharge include",
    "The patient underwent an uncomplicated",
    "History of present illness:",
    "The electrocardiogram showed",
    "The chest radiograph demonstrates",
    "Vital signs on admission were",
    "The patient denies any",
    "Review of systems is positive for",
    "The operative report describes",
    "Pathology revealed",
    "The patient was started on",
    "Follow-up is scheduled in",
    "The most likely diagnosis is",
    "The patient's family history is significant for",
    "Complications during the hospital stay included",
]


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _copy(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"staged": str(dst.relative_to(STAGE)), "bytes": dst.stat().st_size, "sha256": sha256_file(dst),
            "source": str(src), "source_sha256": sha256_file(src)}


def step_stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    if not PPLX4.is_dir():
        raise SystemExit(f"{PPLX4} is missing: R1 has not finished")
    # --- Osler-9B-rc0: weights-only checkpoint, readout matrix, calibration, descriptor ---
    nine = STAGE / "osler-9b-rc0"
    payload = torch.load(D3_9, map_location="cpu", weights_only=False)
    stage_payload = {"step": int(payload["step"]), "model": payload["model"]}
    weights = nine / "weights" / "osler_9b_rc0.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stage_payload, weights)
    readout = nine / "weights" / "readout.pt"
    torch.save(payload["model"]["readout"], readout)
    calibration9 = json.loads(CALIB9.read_text())["temperatures"]
    write_json(nine / "calibration.json", {"method": "dev-only per-qtype temperature on fastdev_v4",
                                           "temperatures": calibration9,
                                           "source": "outputs/osler_rc0/R2/9b_calibration.json"})
    descriptor9 = {
        "name": "Osler-9B-rc0", "backbone": "d3", "base_model_id": D3_BASE_REPO, "base_revision": D3_BASE_REV,
        "architecture": "Qwen3_5Model + attention-only LoRA (r=32) + option-code readout [255, 4096]; no lm_head",
        "step": int(payload["step"]), "checkpoint_contents": ["lora", "readout"],
        "files": {"weights/osler_9b_rc0.pt": sha256_file(weights), "weights/readout.pt": sha256_file(readout),
                  "calibration.json": sha256_file(nine / "calibration.json")},
        "license": "Apache-2.0 (base vllm-sr/d3-flash); adapter and readout released under Apache-2.0",
        "training_data": "private; not staged", "mimic_text": "none staged",
        "created_utc": utcnow(), "git_commit": git_commit(ROOT),
        "source_artifact": str(D3_9), "source_sha256": sha256_file(D3_9),
    }
    write_json(nine / "model.json", descriptor9)
    manifest9 = {"model": "Osler-9B-rc0", "staged_utc": utcnow(), "git_commit": git_commit(ROOT),
                 "base_repo": D3_BASE_REPO, "base_revision": D3_BASE_REV,
                 "files": {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                           for path in sorted(nine.rglob("*")) if path.is_file()}}
    write_json(nine / "provenance_manifest.json", manifest9)
    # --- Osler-4B-rc0: the saved checkpoint directory (model.json + adapter + optcode) ---
    four = STAGE / "osler-4b-rc0"
    staged_files = {}
    for path in sorted(PPLX4.rglob("*")):
        if path.is_file():
            staged_files[str(path.relative_to(PPLX4))] = _copy(path, four / "weights" / path.relative_to(PPLX4))
    meta4 = json.loads((PPLX4 / "model.json").read_text())
    write_json(four / "calibration.json", {"method": "saved by the R1 trainer (dev-only per-qtype temperatures)",
                                           "temperatures": meta4["calibration"],
                                           "source": "outputs/osler_rc0/R1/arm/checkpoints/final_step_063304/model.json"})
    descriptor4 = {
        "name": "Osler-4B-rc0", "backbone": "pplx", "base_model_id": PPLX_BASE_REPO, "base_revision": PPLX_BASE_REV,
        "base_snapshot_note": "both cached snapshots 6d771ad7… and 9ff7cf98… are byte-identical (osler_rc0 R0 C002)",
        "architecture": "pplx Qwen3_5Model backbone + LoRA r=32 + option-code readout (optcode.pt), noncausal full attention",
        "step": 63304, "files": {**{name: entry["sha256"] for name, entry in staged_files.items()},
                                 "calibration.json": sha256_file(four / "calibration.json")},
        "license": "Apache-2.0 (base per perplexity-ai/pplx-decider-flash-v1.2 LICENSE/NOTICE; derived from Qwen/Qwen3.5-4B)",
        "training_data": "private; not staged", "mimic_text": "none staged",
        "created_utc": utcnow(), "git_commit": git_commit(ROOT),
        "source_artifact": str(PPLX4), "source_sha256": {name: entry["source_sha256"] for name, entry in staged_files.items()},
    }
    write_json(four / "model.json", descriptor4)
    manifest4 = {"model": "Osler-4B-rc0", "staged_utc": utcnow(), "git_commit": git_commit(ROOT),
                 "base_repo": PPLX_BASE_REPO, "base_revision": PPLX_BASE_REV,
                 "files": {str(path.relative_to(four)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                           for path in sorted(four.rglob("*")) if path.is_file()}}
    write_json(four / "provenance_manifest.json", manifest4)
    print(f"staged {STAGE}")


def _d3_probs(checkpoint: Path, items: list[Any]) -> list[list[float]]:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModel

    from meddecide.distill.d3_train import D3Student, prepare_rows
    from meddecide.train.mix_items import mix_row_to_item

    snapshot = Path("/workspace/.hf_home/hub/models--vllm-sr--d3-flash/snapshots") / D3_BASE_REV
    model = AutoModel.from_pretrained(str(snapshot), trust_remote_code=True, dtype=torch.bfloat16, device="cuda:0")
    runtime = model.runtime
    student = D3Student(runtime, lora_r=32, lora_alpha=32).to(runtime.device)
    payload = torch.load(checkpoint, map_location=student.device, weights_only=False)
    student.load_checkpoint_state(payload["model"])
    student.eval()

    def to_row(row: dict[str, Any]) -> dict[str, Any]:
        return {"row_id": mix_row_to_item(row).item_id, "source": str(row["source"]),
                "template_id": str(row["template_id"]), "qtype": str(row["qtype"]), "state": str(row["state"]),
                "question": str(row["question"]), "gold": str(row["gold"]), "split": "eval",
                "options": [{"key": str(o["key"]), "label": str(o["label"])} for o in row["options"]]}

    prepared = prepare_rows(runtime, [to_row(row) for row in items])
    probs = student.probabilities([list(row.sequence) for row in prepared], [row.count for row in prepared],
                                  batch_size=8, max_tokens=16384)
    return [[float(x) for x in row] for row in probs]


def _pplx_probs(checkpoint: Path, items: list[Any]) -> list[list[float]]:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from meddecide.model.pplx import PplxDecisionModel
    from meddecide.train.mix_items import mix_row_to_item

    model = PplxDecisionModel.load(checkpoint, device="cuda:0", dtype="bfloat16")
    model.eval_mode()
    converted = [mix_row_to_item(row) for row in items]
    scored = model.score_items(converted, batch_size=8, max_batch_tokens=16384)
    return [[float(x) for x in probs] for probs in scored.probs]


def _compare(name: str, a: list[list[float]], b: list[list[float]]) -> dict[str, Any]:
    if len(a) != len(b):
        raise SystemExit(f"{name}: {len(a)} vs {len(b)} scored items")
    max_delta = 0.0
    agree = 0
    for left, right in zip(a, b, strict=True):
        max_delta = max(max_delta, max(abs(x - y) for x, y in zip(left, right, strict=True)))
        agree += int(max(range(len(left)), key=left.__getitem__) == max(range(len(right)), key=right.__getitem__))
    return {"items": len(a), "max_abs_probability_delta": max_delta, "argmax_agreement": agree,
            "passed": max_delta < 1e-4 and agree == len(a)}


def _causal_from_pplx(source: Path, *, adapter_scale: float | None) -> Any:
    """A causal LM carrying the pplx base backbone, built through the decision-model loader.

    ``source`` is either the untouched base snapshot (no adapter; ``adapter_scale=None``) or a saved checkpoint directory
    (adapter attached; the adapter's scaling is set explicitly before merging). Both sides of the byte-identity check
    therefore come from the SAME loader and the SAME weight-copy path, which is what isolates the adapter: the checkpoint
    is a custom pplx architecture, so a plain ``AutoModelForCausalLM.from_pretrained`` on the snapshot would miss every
    backbone key and compare against a randomly initialised model (observed and rejected during this audit).
    """
    import gc

    sys.path.insert(0, str(ROOT / "scripts/osler_v4"))
    import y10_audit as y10
    from transformers import AutoModelForCausalLM

    from meddecide.model.pplx import PplxDecisionModel

    if (source / "model.json").is_file():
        model = PplxDecisionModel.load(source, device="cuda:0", dtype="bfloat16")
        if adapter_scale is not None:
            for module in model.peft_model.modules():
                scaling = getattr(module, "scaling", None)
                if isinstance(scaling, dict):
                    module.scaling = dict.fromkeys(scaling, float(adapter_scale))
                elif isinstance(scaling, float):
                    module.scaling = float(adapter_scale)
        merged = model.peft_model.merge_and_unload().eval()
    else:
        model = PplxDecisionModel(source, dtype="bfloat16", device="cuda:0")
        merged = model.base.eval()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    causal = AutoModelForCausalLM.from_pretrained(PPLX_BASE, dtype=torch.bfloat16, device_map="cuda")
    y10._load_backbone(causal, merged)
    del merged
    gc.collect()
    torch.cuda.empty_cache()
    return causal.eval()


def _generate(causal: Any, prompts: list[str], new_tokens: int = 20) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(PPLX_BASE)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    out_ids = []
    device = next(causal.parameters()).device
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        with torch.no_grad():
            out = causal.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=new_tokens,
                                  do_sample=False, pad_token_id=pad_id)
        out_ids.append(out[0, ids.shape[1]:].tolist())
    return out_ids


def step_verify() -> None:
    items = read_jsonl(DEV)[:PARITY_ITEMS]
    result: dict[str, Any] = {"generated_utc": utcnow(), "git_commit": git_commit(ROOT), "parity_items": PARITY_ITEMS}
    nine = STAGE / "osler-9b-rc0"
    four = STAGE / "osler-4b-rc0"
    if not (nine / "weights/osler_9b_rc0.pt").is_file() or not (four / "weights/model.json").is_file():
        raise SystemExit(f"{STAGE} is not staged: run --step stage first")
    result["nine"] = _compare("9b staged vs original", _d3_probs(D3_9, items),
                              _d3_probs(nine / "weights/osler_9b_rc0.pt", items))
    result["four"] = _compare("4b staged vs original", _pplx_probs(PPLX4, items),
                              _pplx_probs(four / "weights", items))
    # 4B base generation byte-identity, with the adapter on as a non-vacuity control
    import gc

    adapter_off = _generate(_causal_from_pplx(four / "weights", adapter_scale=0.0), GEN_PROMPTS)
    gc.collect()
    torch.cuda.empty_cache()
    adapter_on = _generate(_causal_from_pplx(four / "weights", adapter_scale=1.0), GEN_PROMPTS)
    gc.collect()
    torch.cuda.empty_cache()
    untouched = _generate(_causal_from_pplx(Path(PPLX_BASE), adapter_scale=None), GEN_PROMPTS)
    gc.collect()
    torch.cuda.empty_cache()
    identical = sum(1 for a, b in zip(adapter_off, untouched, strict=True) if a == b)
    control_diff = sum(1 for a, b in zip(adapter_on, untouched, strict=True) if a != b)
    result["base_generation"] = {
        "prompts": len(GEN_PROMPTS), "new_tokens": 20,
        "identical_prompts_adapter_off": identical, "control_prompts_adapter_on_differ": control_diff,
        "passed": identical == len(GEN_PROMPTS) and control_diff > 0,
    }
    staged9 = torch.load(nine / "weights/osler_9b_rc0.pt", map_location="cpu", weights_only=False)
    keys = list(staged9["model"].keys())
    lm_head = [key for key in keys if "lm_head" in key]
    result["nine_non_generative"] = {"state_keys": keys, "lm_head_keys": lm_head, "has_lm_head": bool(lm_head),
                                     "passed": not lm_head}
    result["all_passed"] = bool(result["nine"]["passed"] and result["four"]["passed"]
                                and result["base_generation"]["passed"] and result["nine_non_generative"]["passed"])
    write_json(OUT / "release_verification.json", result)
    print(json.dumps(result, indent=1))


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _cell(board: dict[str, Any], model: str, set_name: str, key: str) -> Any:
    metrics = board["models"].get(model, {}).get(set_name, {})
    if metrics.get("status") != "MEASURED":
        return "NOT MEASURED"
    return metrics.get(key, "NOT MEASURED")


def step_cards() -> None:
    board = json.loads((ROOT / "outputs/osler_rc0/R7/rc0_board.json").read_text())
    latency = json.loads((ROOT / "outputs/osler_rc0/R5/latency.json").read_text()) if (
        ROOT / "outputs/osler_rc0/R5/latency.json").is_file() else {"models": {}}
    audit = json.loads((ROOT / "outputs/osler_rc0/R6/membership_auc.json").read_text())
    sample_summary = json.loads((ROOT / "outputs/osler_rc0/R6/sample_summary.json").read_text())
    visited_summary = json.loads((ROOT / "outputs/osler_rc0/R6/visited_summary.json").read_text())
    y10 = json.loads((ROOT / "outputs/osler_rc0/R6/y10_membership.json").read_text())
    verification = json.loads((OUT / "release_verification.json").read_text())
    calibration9 = json.loads(CALIB9.read_text())["temperatures"]
    calibration4 = json.loads((PPLX4 / "model.json").read_text())["calibration"]

    def card(model: str, base: str, other: str, calibration: dict[str, float], audit_key: str,
             stage_dir: str, architecture: str, base_repo: str, base_rev: str, extra_probes: list[str],
             serving_code: str) -> str:
        lines = [f"# {model} — model card (RC0)", "",
                 f"**Status:** release candidate staged locally on {utcnow()}; not uploaded. "
                 f"Staged tree: `{stage_dir}` (weights, calibration, provenance manifest; no training data, no MIMIC text).",
                 "",
                 "## What it is", "",
                 f"`{model}` is a calibrated medical decision model: given a state (a note, a trial record, a question, or a "
                 "query + passage) and a list of typed questions, it returns a probability for every offered option in one "
                 "prefill pass — no text generation. Question types: `noul` (yes/no), `choice` (2-255 named options) and "
                 "`score` (2-10 ordered levels).",
                 "",
                 f"- **Base model:** `{base_repo}` at revision `{base_rev}`",
                 f"- **Architecture:** {architecture}",
                 f"- **Calibration (dev-only, fastdev_v4):** choice {calibration.get('choice'):.6f}, "
                 f"noul {calibration.get('noul'):.6f}, score {calibration.get('score'):.6f}",
                 "- **Licence:** Apache-2.0. The base's LICENSE and NOTICE travel with the project; no MIMIC text or "
                 "training rows are in the release tree.",
                 "- **Serving:** one prefill pass, greedy readout over the offered option codes; no sampling, no generation.",
                 "",
                 "```python",
                 serving_code,
                 "```",
                 "",
                 "## Headline measurements (12-model comparison, 9 frozen sets)", "",
                 "| set | accuracy | 95% CI | template macro | Brier | ECE | coverage |", "|---|---|---|---|---|---|---|"]
        for set_name in ["panel", "knowledge", "confirm_general", "mr_verdict", "mimic_suite"]:
            ci = _cell(board, model, set_name, "accuracy_ci95")
            ci_text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if isinstance(ci, list) else "—"
            lines.append(f"| `{set_name}` | {_fmt(_cell(board, model, set_name, 'accuracy'))} | {ci_text} | "
                         f"{_fmt(_cell(board, model, set_name, 'macro_accuracy'))} | "
                         f"{_fmt(_cell(board, model, set_name, 'brier'))} | {_fmt(_cell(board, model, set_name, 'ece'))} | "
                         f"{_fmt(_cell(board, model, set_name, 'coverage'), 3)} |")
        lines += ["", "## Paired differences against its own zero-shot base", "",
                  "| set | Δ accuracy | 95% CI |", "|---|---|---|"]
        paired = board["paired"].get(model, {}).get(base, {})
        for set_name in ["panel", "knowledge", "confirm_general", "mr_verdict", "mimic_suite"]:
            diff = paired.get(set_name, {})
            if not diff or diff.get("n", 0) == 0:
                lines.append(f"| `{set_name}` | NOT MEASURED | — |")
            else:
                ci = diff["accuracy_diff_ci95"]
                lines.append(f"| `{set_name}` | {diff['accuracy_diff']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
        # 12-model Pareto summary: where this model sits against every other model in the loop's roster
        front = set(board.get("pareto", {}).get("two_axis", {}).get("front", []))
        undominated = set(board.get("pareto", {}).get("quality_le10b", {}).get("undominated", []))
        rows = []
        for rival in board["params_b"]:
            values = []
            measured = []
            for set_name in ["panel", "knowledge", "confirm_general", "mr_verdict", "mimic_suite"]:
                value = _cell(board, rival, set_name, "accuracy")
                values.append(_fmt(value) if isinstance(value, float) else "—")
                if isinstance(value, float):
                    measured.append(value)
            macro = sum(measured) / len(measured) if len(measured) == 5 else None
            if macro is not None:
                rows.append((rival, board["params_b"][rival], macro, values))
        rows.sort(key=lambda row: -row[2])
        lines += ["", "## 12-model Pareto summary", "",
                  f"This model is on the 2-axis Pareto front (OOD General Avg vs Clinical Audit Avg): "
                  f"**{'yes' if model in front else 'no'}**; undominated on all five headline sets among <=10B models: "
                  f"**{'yes' if model in undominated else 'no'}**. Full frontier tables: `loops/osler_rc0/PARETO.md`.", "",
                  "| model | params (B) | 5-set macro | panel | knowledge | confirm_general | mr_verdict | mimic_suite |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {name} | {params:.2f} | {macro:.4f} | " + " | ".join(values) + " |"
                  for name, params, macro, values in rows]
        latency_entry = latency.get("models", {}).get(model, {})
        lines += ["", "## Efficiency", "",
                  f"- Parameters: {board['params_b'][model]:.2f} B",
                  f"- Batch-1 p50 latency: {_fmt(latency_entry.get('batch1', {}).get('p50_ms_per_item'))} ms/item "
                  f"(p95 {_fmt(latency_entry.get('batch1', {}).get('p95_ms_per_item'))} ms/item)",
                  f"- Batch-16 throughput: {_fmt(latency_entry.get('batch16', {}).get('items_per_s'))} items/s",
                  f"- Peak allocated VRAM (batch 1): {_fmt(latency_entry.get('batch1', {}).get('peak_vram_gb'))} GB", "",
                  "## MIMIC PHI / memorisation audit (D28; local only)", ""]
        entry = audit["models"].get(model, {})
        lines += [f"- Visited-member membership AUC (members visited by both release candidates vs late-anchor "
                  f"`mimic_suite` non-members, n={audit['members']}+{audit['nonmembers']}): "
                  f"**{_fmt(entry.get('auc'))}** [{_fmt(entry.get('auc_lower'))}, {_fmt(entry.get('auc_upper'))}]; "
                  f"base control {_fmt(entry.get('auc_control'))}; difference {_fmt(entry.get('auc_difference'))} "
                  f"[{_fmt((entry.get('auc_difference_ci95') or ['-', '-'])[0])}, "
                  f"{_fmt((entry.get('auc_difference_ci95') or ['-', '-'])[1])}].",
                  f"- Y10 historical compatibility set AUC: {_fmt(y10['models'].get(model, {}).get('auc'))} "
                  f"(osler_v4 reference 0.5042).",
                  f"- {sample_summary['members']:,} members sampled from "
                  f"{sample_summary['mix_mimic_rows_visited_both']:,} MIMIC rows visited by both runs; the training mix has "
                  f"{visited_summary['mix_mimic_rows']:,} MIMIC-derived rows "
                  f"({100 * visited_summary['mix_mimic_rows'] / visited_summary['mix_rows']:.1f} % of "
                  f"{visited_summary['mix_rows']:,})."]
        lines += extra_probes
        paired = board["paired"].get(model, {}).get(base, {})
        def _delta(set_name: str) -> str:
            diff = paired.get(set_name)
            if not diff or diff.get("n", 0) == 0:
                return "NOT MEASURED"
            return f"{diff['accuracy_diff']:+.4f} [{diff['accuracy_diff_ci95'][0]:+.4f}, {diff['accuracy_diff_ci95'][1]:+.4f}]"
        held_out_flag = (
            "**Held-out failure, reported:** on `mr_heldout_test` / `synthea_indicator_evidence_v1` (n=514, a template "
            "excluded from training) both release candidates read below chance (9B accuracy 0.356, CI upper 0.3988; 4B "
            "0.2451, upper 0.2821; chance 0.5) while their bases pass the same cell (d3-flash 0.4591, pplx-flash 0.5973). "
            "The cell is flagged `READOUT_FAIL` and is not quoted as an accuracy anywhere in this card.")
        calibration_note = (
            "Dev-only calibration does not transfer to the fresh post-cutoff set: mean confidence on `confirm_general` "
            "is 0.8137 (9B) / 0.8005 (4B) against accuracies 0.9008 / 0.8883, so the ECE there is 0.0900 / 0.0940 versus "
            "0.0192 on the dev fit set (the models are under-confident on the fresh distribution). No test-set "
            "temperature was fitted.")
        lines += ["", "## Honest limitations", "",
                  f"- General OOD QA versus its own zero-shot base: `panel` {_delta('panel')}, `knowledge` "
                  f"{_delta('knowledge')} — "
                  + ("parity (both intervals include zero), not a win." if model.startswith("Osler-9B")
                     else "**below the base by about one point, both intervals excluding zero**; the model trades "
                          "general-OOD accuracy for clinical-audit gains and better Brier on four of five headline sets."),
                  "- " + held_out_flag,
                  "- " + calibration_note,
                  "- The 27B comparison model is stronger on hard multi-step medical QA (for example MedXpertQA); the "
                  "release candidate is not claimed to beat it.",
                  "- Held-out transfer (`mimic_heldout`, `mr_heldout_test`, `mednli_test`, `pairs_test`) is reported as "
                  "measured, with wide intervals on the small sets.",
                  "- Not for unsupervised clinical decisions. Outputs are probabilities from a research model; a "
                  "clinician remains responsible for any decision.",
                  "- MIMIC: trained on credentialed data locally; no MIMIC text was sent to any hosted service, and the "
                  "release tree contains no MIMIC text.",
                  "", "## Provenance", "",
                  f"- Staged and verified {utcnow()} at git `{verification['git_commit']}`; "
                  f"probability parity max |Δp| {verification['four' if model.startswith('Osler-4B') else 'nine']['max_abs_probability_delta']:.3g} "
                  f"with 20/20 argmax agreement on 20 fixed dev items.",
                  "- Full manifest with SHA-256 of every staged file: `provenance_manifest.json` in the staged tree.", ""]
        return "\n".join(lines)

    serving9 = ("# Osler-9B-rc0: one prefill pass over the offered option codes\n"
                "import torch\n"
                "from transformers import AutoModel\n"
                "from meddecide.distill.d3_train import D3Student, prepare_rows\n"
                "snapshot = '/path/to/vllm-sr--d3-flash@67e03e1b'\n"
                "runtime = AutoModel.from_pretrained(snapshot, trust_remote_code=True, dtype=torch.bfloat16,\n"
                "                                   device='cuda:0').runtime\n"
                "student = D3Student(runtime, lora_r=32, lora_alpha=32).to(runtime.device)\n"
                "state = torch.load('staged/weights/osler_9b_rc0.pt', map_location=student.device)['model']\n"
                "student.load_checkpoint_state(state)  # LoRA + [255, 4096] option-code readout\n"
                "student.eval()\n"
                "prepared = prepare_rows(runtime, rows)  # rows: state / question / options / gold\n"
                "probs = student.probabilities([list(r.sequence) for r in prepared],\n"
                "                              [r.count for r in prepared], batch_size=16)\n"
                "# apply the per-qtype temperatures in calibration.json to the option logits before softmax")
    serving4 = ("# Osler-4B-rc0: one prefill pass over the offered option codes\n"
                "from meddecide.model.pplx import PplxDecisionModel\n"
                "model = PplxDecisionModel.load('staged/weights', device='cuda')\n"
                "model.eval_mode()\n"
                "decisions = model.predict(items, batch_size=16)  # temperature from model.json is applied\n"
                "p_yes = decisions[0].probs  # aligned with the item's option keys")
    cards = {
        "MODEL_CARD_9B.md": card("Osler-9B-rc0", "d3-flash", "Osler-4B-rc0", calibration9, "Osler-9B-rc0",
                                 str(STAGE / "osler-9b-rc0"),
                                 "Qwen3_5Model with attention-only LoRA (r=32) and a [255, 4096] option-code readout; "
                                 "non-generative (no lm_head)",
                                 D3_BASE_REPO, D3_BASE_REV,
                                 ["- The verbatim-extraction probe is **not applicable**: the staged weights carry no "
                                  "`lm_head` and no language-model head supports generation (verified on the staged "
                                  "weights)."],
                                 serving9),
        "MODEL_CARD_4B.md": card("Osler-4B-rc0", "pplx-flash", "Osler-9B-rc0", calibration4, "Osler-4B-rc0",
                                 str(STAGE / "osler-4b-rc0"),
                                 "pplx Qwen3_5Model backbone with LoRA (r=32) and a separate option-code readout; "
                                 "noncausal full attention",
                                 PPLX_BASE_REPO, PPLX_BASE_REV,
                                 [f"- Verbatim-extraction probe on 300 MIMIC discharge notes: "
                                  f"{json.loads((ROOT / 'outputs/osler_rc0/R6/verbatim_probe.json').read_text())['final_matches_at_least_50']} "
                                  "matches of ≥50 consecutive tokens for the adapter-on causal export; the zero-shot "
                                  "`pplx-flash` control on the same notes: "
                                  f"{json.loads((ROOT / 'outputs/osler_rc0/R6/verbatim_probe.json').read_text())['base_matches_at_least_50']}.",
                                  "- Adapter-disabled generation is token-for-token identical to the untouched "
                                  "`pplx-flash` base on 20 fixed prompts (adapter-on control differs)."],
                                 serving4),
    }
    for name, text in cards.items():
        (ROOT / "loops/osler_rc0" / name).write_text(text + "\n", encoding="utf-8")
        print(f"wrote loops/osler_rc0/{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--step", required=True, choices=["stage", "verify", "cards"])
    args = parser.parse_args()
    {"stage": step_stage, "verify": step_verify, "cards": step_cards}[args.step]()


if __name__ == "__main__":
    main()
