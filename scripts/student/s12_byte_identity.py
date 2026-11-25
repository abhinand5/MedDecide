"""S12 — generation byte-identity with the adapter disabled, on shipped checkpoints.

ADVISORY S12: *re-run S7's byte-identity test on the trained S9 adapter (adapter off ⇒ base
generation unchanged on 20 fixed prompts)*. This script does exactly that, against the
checkpoints that were actually shipped (S9 run 3's ``best/`` and S10's ``best/``), rather
than against a freshly initialised adapter as ``tests/test_model_pointer.py`` does.

The comparison, for every prompt in a fixed set, greedy with ``max_new_tokens`` and no
sampling:

* ``reference``   — a **separately loaded, untouched** ``AutoModelForCausalLM`` built from the
  checkpoint's ``base_model_id`` with the same dtype/device. Never wrapped by peft.
* ``adapter_off`` — the trained checkpoint loaded through :meth:`MedDecideModel.load` and
  generated inside :meth:`MedDecideModel.adapter_disabled`.
* ``adapter_on``  — the same trained checkpoint with its **trained** adapter active. Control 1:
  if this equals the reference the adapter has no effect and the identity result would be
  vacuous, so the flag is reported rather than assumed.
* ``perturbed``   — adapter active but every ``lora_B`` re-drawn from ``N(0, 0.05)`` (restored
  exactly afterwards). Control 2, the same one S7's test uses: a changed adapter *must* change
  generation or the comparison cannot detect anything.

**Kernel path on a CPU-only box.** ``Qwen3_5GatedDeltaNet`` binds ``causal_conv1d`` and ``fla``
at *import* time if the packages are importable, and both are CUDA-only, so a CPU forward pass
dies inside the CUDA kernel. ``--torch-kernels auto`` (the default on a non-CUDA device)
installs empty placeholder modules for ``causal_conv1d`` and ``fla`` in ``sys.modules`` before
the modelling module is imported; transformers' ``use_kernel_func_from_hub_with_fallback``
then resolves to its reference PyTorch implementations. Both sides of the comparison use the
same path, so this is a valid adapter-bypass test — but it is a *different kernel path* from
the shipped GPU configuration, and the artifact records that.

Prompts contain item text; they are never written out — only item ids, token counts and the
sha256 of the rendered prompt. Generated token ids are model outputs and live in ``outputs/``
(gitignored) only.

    uv run python scripts/student/s12_byte_identity.py \
        --checkpoint outputs/student_v0/S9_run3/best \
        --out outputs/student_v0/S12/byte_identity_S9_run3_best.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch

from meddecide.eval.readout import canonicalise_options, render_prompt
from meddecide.model.meddecide_model import MedDecideModel
from meddecide.train.data import read_items

REPO = Path(__file__).resolve().parents[2]
DEV_PATH = REPO / "data" / "train" / "student_v0" / "dev.jsonl"
S7_SAMPLING = "read_items(dev.jsonl, limit=64, stride=11); sort by item_id; first 20"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git always present in this repo
        return None


def force_reference_kernels(packages: tuple[str, ...] = ("causal_conv1d", "fla")) -> list[str]:
    """Make transformers fall back to its reference PyTorch kernels.

    The modelling module binds these packages at import time; both ship CUDA-only kernels, so a
    CPU forward pass would otherwise fail inside the kernel. An empty placeholder module makes
    ``resolve_internal_import`` return ``None`` and the decorator keep its torch implementation.
    """
    shadowed = []
    for name in packages:
        existing = sys.modules.get(name)
        if existing is None:
            sys.modules[name] = types.ModuleType(name)
            shadowed.append(name)
    return shadowed


def _prompt_row(item: Any, tokenizer: Any) -> dict[str, Any]:
    canonical, _ = canonicalise_options(item)
    prompt = render_prompt(canonical, tokenizer, "bare")
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return {
        "item_id": item.item_id,
        "template_id": str(item.template_id),
        "qtype": str(item.qtype),
        "n_options": canonical.n_options,
        "prompt_tokens": len(token_ids),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt": prompt,
    }


def build_s7_prompts(tokenizer: Any, dev_path: Path, n: int) -> list[dict[str, Any]]:
    """The exact prompt set S7's acceptance test uses."""
    items = read_items(dev_path, limit=64, stride=11)
    items.sort(key=lambda i: i.item_id)
    return [_prompt_row(item, tokenizer) for item in items[:n]]


def build_template_spread_prompts(
    tokenizer: Any,
    dev_path: Path,
    *,
    limit: int,
    stride: int,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """An **additional** set: the shortest-prompt item of every dev template seen.

    Not part of S12's acceptance criterion; it widens the check from one template to all
    templates the dev file covers, at bounded prompt length so a CPU run stays cheap.
    """
    items = read_items(dev_path, limit=limit, stride=stride)
    by_template: dict[str, dict[str, Any]] = {}
    for item in items:
        row = _prompt_row(item, tokenizer)
        if row["prompt_tokens"] > max_tokens:
            continue
        key = row["template_id"]
        current = by_template.get(key)
        if current is None or (row["prompt_tokens"], row["item_id"]) < (
            current["prompt_tokens"],
            current["item_id"],
        ):
            by_template[key] = row
    covered = sorted(by_template)
    rows = [by_template[key] for key in covered]
    return rows, covered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dev", type=Path, default=DEV_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default=None, help="default: the checkpoint's stored dtype")
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0, help="0 keeps torch's default")
    parser.add_argument(
        "--torch-kernels",
        choices=("auto", "force", "off"),
        default="auto",
        help="auto = force the reference PyTorch kernels on a non-CUDA device (see module docstring)",
    )
    parser.add_argument(
        "--template-spread",
        action="store_true",
        help="also run the additional one-short-item-per-template set",
    )
    parser.add_argument("--spread-limit", type=int, default=2000)
    parser.add_argument("--spread-stride", type=int, default=8)
    parser.add_argument("--spread-max-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    use_reference_kernels = args.torch_kernels == "force" or (
        args.torch_kernels == "auto" and not str(args.device).startswith("cuda")
    )
    shadowed = force_reference_kernels() if use_reference_kernels else []
    kernel_path = (
        "reference PyTorch implementations (causal_conv1d / fla shadowed in sys.modules)"
        if use_reference_kernels
        else "as shipped (fused kernels bound)"
    )
    print(f"[kernels] {kernel_path}; shadowed={shadowed}")

    checkpoint = args.checkpoint
    meta = json.loads((checkpoint / "model.json").read_text(encoding="utf-8"))
    base_id = meta["base_model_id"]
    dtype_name = args.dtype or meta.get("dtype", "bfloat16")
    torch_dtype = getattr(torch, dtype_name)

    started = time.time()
    base = MedDecideModel.load(checkpoint, device=args.device, dtype=dtype_name)
    tokenizer = base.tokenizer

    s7_rows = build_s7_prompts(tokenizer, args.dev, args.n_prompts)
    prompt_sets: dict[str, list[dict[str, Any]]] = {"s7_fixed_20": s7_rows}
    spread_meta: dict[str, Any] = {}
    if args.template_spread:
        spread_rows, covered = build_template_spread_prompts(
            tokenizer,
            args.dev,
            limit=args.spread_limit,
            stride=args.spread_stride,
            max_tokens=args.spread_max_tokens,
        )
        prompt_sets["template_spread"] = spread_rows
        spread_meta = {
            "sampling": (
                f"read_items(limit={args.spread_limit}, stride={args.spread_stride}); "
                f"shortest prompt per template with <= {args.spread_max_tokens} tokens"
            ),
            "templates_covered": covered,
        }

    # The reference is loaded separately and never wrapped by peft.
    from transformers import AutoModelForCausalLM

    pristine = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=torch_dtype, device_map=args.device
    )
    pristine.eval()

    sets_out: dict[str, Any] = {}
    for name, rows in prompt_sets.items():
        prompts = [row["prompt"] for row in rows]
        reference = generate_greedy_raw(pristine, tokenizer, prompts, args)
        base.eval_mode()
        with base.adapter_disabled():
            adapter_off = base.generate_greedy(
                prompts, max_new_tokens=args.max_new_tokens, seed=args.seed
            )
        adapter_on = base.generate_greedy(
            prompts, max_new_tokens=args.max_new_tokens, seed=args.seed
        )
        perturbed = generate_perturbed(base, prompts, args.max_new_tokens, args.seed)
        sets_out[name] = {
            "n_prompts": len(rows),
            "prompt_tokens_sum": sum(r["prompt_tokens"] for r in rows),
            "prompt_tokens_max": max(r["prompt_tokens"] for r in rows),
            "templates": sorted({r["template_id"] for r in rows}),
            "rows": [
                {
                    "item_id": row["item_id"],
                    "template_id": row["template_id"],
                    "prompt_tokens": row["prompt_tokens"],
                    "prompt_sha256": row["prompt_sha256"],
                    "reference_tokens": ref,
                    "adapter_off_tokens": off,
                    "adapter_on_tokens": on,
                    "perturbed_tokens": pert,
                    "adapter_off_equals_reference": off == ref,
                    "adapter_off_decodes_equal": base.decode(off) == tokenizer.decode(ref),
                    "adapter_on_differs_from_reference": on != ref,
                    "perturbed_differs_from_reference": pert != ref,
                }
                for row, ref, off, on, pert in zip(
                    rows, reference, adapter_off, adapter_on, perturbed, strict=True
                )
            ],
        }
        agg = aggregate(sets_out[name]["rows"])
        sets_out[name].update(agg)
        print(
            f"[{name}] n={len(rows)} identity={agg['n_identity']}/{len(rows)} "
            f"decoded_equal={agg['n_decoded_equal']}/{len(rows)} "
            f"adapter_on_differs={agg['n_adapter_on_differs']}/{len(rows)} "
            f"perturbed_differs={agg['n_perturbed_differs']}/{len(rows)}"
        )
        del prompts

    del pristine
    del base

    payload = {
        "task": "S12",
        "check": "generation byte-identity with the adapter disabled vs an untouched base",
        "checkpoint": str(checkpoint),
        "checkpoint_files": {
            "adapter_model.safetensors": sha256_file(checkpoint / "adapter/adapter_model.safetensors"),
            "head.pt": sha256_file(checkpoint / "head.pt"),
            "model.json": sha256_file(checkpoint / "model.json"),
        },
        "base_model_id": base_id,
        "dtype": dtype_name,
        "device": args.device,
        "kernel_path": kernel_path,
        "kernel_modules_shadowed": shadowed,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "seed": args.seed,
        "sampling_note": "greedy, one prompt per forward pass (the shipped generate_greedy path)",
        "s7_prompt_set": S7_SAMPLING,
        "prompt_sets": sets_out,
        "template_spread_meta": spread_meta,
        "wall_seconds": round(time.time() - started, 1),
        "torch": torch.__version__,
        "transformers": _version("transformers"),
        "peft": _version("peft"),
        "python": platform.python_version(),
        "git_commit": git_commit(),
        "torch_threads": torch.get_num_threads(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} in {payload['wall_seconds']}s")
    return 0


def _version(name: str) -> str | None:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", None))
    except Exception:  # pragma: no cover
        return None


def generate_greedy_raw(
    model: Any, tokenizer: Any, prompts: list[str], args: argparse.Namespace
) -> list[list[int]]:
    """The untouched base, generated with plain ``transformers`` (S7's reference path)."""
    torch.manual_seed(args.seed)
    out: list[list[int]] = []
    with torch.inference_mode():
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(args.device) for k, v in encoded.items()}
            generated = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens, do_sample=False
            )
            out.append([int(t) for t in generated[0, encoded["input_ids"].shape[1] :]])
    return out


def generate_perturbed(
    model: MedDecideModel, prompts: list[str], max_new_tokens: int, seed: int
) -> list[list[int]]:
    """Control: a changed adapter must change generation (S7's perturbation control)."""
    lora_b = [
        (name, param)
        for name, param in model.peft_model.named_parameters()
        if name.endswith("lora_B.default.weight")
    ]
    if not lora_b:
        return []
    saved = [param.detach().clone() for _, param in lora_b]
    with torch.no_grad():
        for _, param in lora_b:
            param.normal_(0.0, 0.05)
    try:
        return model.generate_greedy(prompts, max_new_tokens=max_new_tokens, seed=seed)
    finally:
        with torch.no_grad():
            for (_, param), before in zip(lora_b, saved, strict=True):
                param.copy_(before)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_prompts": len(rows),
        "n_identity": sum(r["adapter_off_equals_reference"] for r in rows),
        "n_decoded_equal": sum(r["adapter_off_decodes_equal"] for r in rows),
        "n_adapter_on_differs": sum(r["adapter_on_differs_from_reference"] for r in rows),
        "n_perturbed_differs": sum(r["perturbed_differs_from_reference"] for r in rows),
        "identity_all": all(r["adapter_off_equals_reference"] for r in rows),
        "decoded_all_equal": all(r["adapter_off_decodes_equal"] for r in rows),
        "adapter_on_differs_all": all(r["adapter_on_differs_from_reference"] for r in rows),
        "perturbed_differs_all": all(r["perturbed_differs_from_reference"] for r in rows),
    }


if __name__ == "__main__":
    raise SystemExit(main())
