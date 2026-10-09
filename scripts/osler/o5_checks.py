"""O5 readout checks on the pinned Qwen3.5-4B (GPU, fp32): the acceptance run for the option-code readout.

Runs the checks of ``meddecide.model.readout_checks`` on the real base model and writes
``outputs/osler_v0/O5/readout_checks.json``: initialisation equality with the zero-shot letter readout (1e-4), the
exported causal LM against the native readout with the adapter separate and merged (1e-3, 50 items), the causal path
unchanged with the non-causal flag off and the change reaching earlier positions with it on, padding invariance on the
option-code path (causal and bidirectional) and on the pointer path (1e-3), and greedy generation with the adapter off
byte-identical to an untouched base (with a control). One GPU job at a time: run this only when no other GPU job is
running. Logs go to ``outputs/osler_v0/O5/logs/``.

``--precision as_run`` (the default) is the O5 verdict. ``--precision reference`` and ``--precision ieee`` run the same
checks under the other settings of ``meddecide.precision``; they write ``readout_checks_<precision>.json`` and are
additional analyses, not the verdict.

Usage:
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler/o5_checks.py \
        > outputs/osler_v0/O5/logs/o5_checks.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import gc
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from meddecide.precision import PRECISIONS, prepare_precision
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEV = REPO / "data/train/student_v1/dev.jsonl"
OUT = REPO / "outputs/osler_v0/O5/readout_checks.json"
LORA_SETTINGS = {"r": 32, "alpha": 32}  # the O6 recipe's adapter shape (ADVISORY O6-O8)
TOL = {"init_logits": 1e-4, "init_probs": 1e-4, "export": 1e-3, "padding": 1e-3, "causal_off": 1e-5,
       "causal_on_min": 1e-4, "control_min": 1e-4}


def versions() -> dict[str, str]:
    import peft
    import transformers

    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
            "cuda": torch.version.cuda or "none", "gpu": torch.cuda.get_device_name(0)}


def check(name: str, value: float, tolerance: float, *, at_most: bool = True) -> dict[str, Any]:
    passed = value <= tolerance if at_most else value >= tolerance
    return {"check": name, "measured": value, "tolerance": tolerance,
            "rule": "<=" if at_most else ">=", "passed": bool(passed)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=50)
    parser.add_argument("--precision", choices=PRECISIONS, default="as_run",
                        help="as_run is the O5 configuration; reference and ieee are additional analyses")
    parser.add_argument("--seed", type=int, default=0, help="seeds the global RNG before the adapter is built")
    args = parser.parse_args()
    # Before transformers or fla is imported: both read the environment and the module table at import time.
    prepare_precision(args.precision)

    from transformers import AutoModelForCausalLM

    from meddecide.model.meddecide_model import LoraSettings, MedDecideModel
    from meddecide.model.readout_checks import (
        CHUNK_ITEMS,
        causal_flag_max_abs_diff,
        export_check,
        export_plan,
        generation_identity,
        init_max_abs_diff,
        padding_max_abs_diff,
        prompts_for,
        randomise_lora,
    )
    from meddecide.precision import apply_after_import
    from meddecide.train.data import read_items

    apply_after_import(args.precision)
    lora = LoraSettings(**LORA_SETTINGS)
    out = OUT.with_name(f"readout_checks_{args.precision}_seed{args.seed}.json")
    started = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    items = read_items(DEV, limit=args.items, stride=max(1, 5619 // args.items))
    short = items[: min(3, len(items))]
    results: list[dict[str, Any]] = []
    record: dict[str, Any] = {"kind": "osler_v0_O5_readout_checks", "built_at_utc": utcnow(),
                              "git_commit": git_commit(REPO), "command": "scripts/osler/o5_checks.py "
                              f"--precision {args.precision} --seed {args.seed}", "precision": args.precision,
                              "seed": args.seed,
                              "role": ("seeded run of the as_run configuration; the O5 verdict is the unseeded attempt-4 "
                                       "record" if args.precision == "as_run"
                                       else "additional analysis (not the O5 verdict)"),
                              "base_model": BASE_MODEL, "base_revision": BASE_REVISION, "dtype": "float32",
                              "device": "cuda:0", "lora": lora.to_dict(), "items_used": len(items),
                              "items_source": str(DEV.relative_to(REPO)), "versions": versions(),
                              "tolerances": TOL, "chunk_items": CHUNK_ITEMS}

    # PEFT draws lora_A from the global RNG; without this seed every process builds a different adapter.
    torch.manual_seed(args.seed)
    model = MedDecideModel(BASE_MODEL, revision=BASE_REVISION, lora=lora, device="cuda:0", dtype="float32",
                           readout="option_code", variant="bare")
    randomise_lora(model)
    init = init_max_abs_diff(model, items)
    results.append(check("initialised option-code logits vs zero-shot letter logits", init["logits_max_abs_diff"],
                         TOL["init_logits"]))
    results.append(check("initialised option-code probabilities vs letter readout", init["probs_max_abs_diff"],
                         TOL["init_probs"]))
    record["init"] = init

    # a stand-in for training moves the head off its initialisation, so the export checks test a changed head
    with torch.no_grad():
        model.optcode.weight.add_(0.05 * torch.randn_like(model.optcode.weight))

    off = max(causal_flag_max_abs_diff(model, item, bidirectional=False) for item in short)
    on = max(causal_flag_max_abs_diff(model, item, bidirectional=True) for item in short)
    results.append(check("causal path unchanged with the non-causal flag off (max change before the edit)", off,
                         TOL["causal_off"]))
    results.append(check("non-causal flag on: the edit reaches earlier positions (max change)", on,
                         TOL["causal_on_min"], at_most=False))
    record["causal_flag"] = {"off_max_change": off, "on_max_change": on, "items": len(short)}

    for bidirectional in (False, True):
        worst = max(padding_max_abs_diff(model, items[i], items[i + 1], bidirectional=bidirectional)
                    for i in range(0, min(len(items) - 1, 8), 2))
        name = "bidirectional" if bidirectional else "causal"
        results.append(check(f"option-code padding invariance, {name} (batch 1 vs padded)", worst, TOL["padding"]))
        record.setdefault("padding", {})[name] = worst

    prompts = prompts_for(model, items[:2])
    plain = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION, dtype=torch.float32,
                                                 device_map="cuda:0").eval()
    identity = generation_identity(model, plain, prompts)
    results.append(check("greedy generation, adapter off, byte identical to the untouched base",
                         0.0 if identity["byte_identical"] else 1.0, 0.0))
    results.append(check("control: adapter on changes the next-token logits",
                         identity["control_adapter_logits_max_abs_diff"], TOL["control_min"], at_most=False))
    record["generation"] = identity
    del plain
    gc.collect()
    torch.cuda.empty_cache()

    # the export plan is computed on the live model; the live model is then released before the exported causal LM is
    # loaded, because two fp32 copies of the 4B model do not fit on the GPU together with the activations
    with tempfile.TemporaryDirectory(prefix="o5_export_", dir=str(OUT.parent)) as tmp:
        plan = export_plan(model, items[:50], Path(tmp))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        separate = export_check(plan, merged=False)
        results.append(check("exported causal LM vs native readout, adapter separate (50 items)",
                             separate["probs_max_abs_diff"], TOL["export"]))
        merged = export_check(plan, merged=True)
        results.append(check("exported causal LM vs native readout, adapter merged (50 items)",
                             merged["probs_max_abs_diff"], TOL["export"]))
        del plan
    record["export"] = {"separate": separate, "merged": merged}
    gc.collect()
    torch.cuda.empty_cache()

    pointer = MedDecideModel(BASE_MODEL, revision=BASE_REVISION, lora=lora, device="cuda:0", dtype="float32",
                             readout="pointer", variant="bare")
    pointer_worst = max(padding_max_abs_diff(pointer, items[i], items[i + 1], bidirectional=False)
                        for i in range(0, min(len(items) - 1, 8), 2))
    results.append(check("pointer padding invariance (batch 1 vs padded)", pointer_worst, TOL["padding"]))
    record["padding"]["pointer"] = pointer_worst

    record["checks"] = results
    record["all_passed"] = all(r["passed"] for r in results)
    record["wall_clock_s"] = round(time.time() - started, 1)
    write_json(out, record)
    failed = [r["check"] for r in results if not r["passed"]]
    print(f"o5 checks ({args.precision}): {len(results) - len(failed)} of {len(results)} passed; "
          f"wall clock {record['wall_clock_s']} s", flush=True)
    for name in failed:
        print(f"  FAILED: {name}", flush=True)


if __name__ == "__main__":
    main()
