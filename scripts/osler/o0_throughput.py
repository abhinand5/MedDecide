"""O0 throughput: training and dev-eval speed for one Osler arm at a given base size (ADVISORY O0).

Recipe under measurement (matches O6's recipe where it can be measured at O0):
  - base Qwen/Qwen3.5-{4B,9B} in bf16, base frozen; LoRA r=32, alpha=32 on every language-model
    linear layer; a 255-row readout initialised from the base's lm_head rows of the code tokens
    (here random: only the shape and cost matter for timing); CE over the offered options at the
    answer position only (no full-vocabulary logits);
  - batch of 8 sequences at the median prompt length of the student_v1 mix, 50 steps, timing the
    last 45 (the first 5 warm up kernels and the allocator);
  - forward-only batches of 8 at the same length, for the dev-eval projection.

The median length is measured on a deterministic sample of the student_v1 training rows
(every 120th line; the mix v2 does not exist yet), rendered through the same chat template and
prompt wording as the scoring harness (meddecide.eval.readout.render_prompt).

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o0_throughput.py \
        --model Qwen/Qwen3.5-4B --out outputs/osler_v0/O0/throughput_4b.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from meddecide.eval.prompt_lengths import summarize_lengths
from meddecide.eval.readout import render_prompt
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, gpu_name, utcnow

REPO = Path(__file__).resolve().parents[2]
MIX = REPO / "data/train/student_v1/train.jsonl"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]


def median_prompt_tokens(tokenizer: Any, every: int = 120) -> dict[str, Any]:
    """Prompt-length summary over every ``every``-th line of the student_v1 mix (deterministic)."""
    lengths: list[int] = []
    with MIX.open() as fh:
        for i, line in enumerate(fh):
            if i % every:
                continue
            r = json.loads(line)
            options = [SimpleNamespace(key=o["key"], label=o["label"]) for o in r["options"]]
            item = SimpleNamespace(state=r["state"], question=r["question"], options=options)
            lengths.append(len(tokenizer.encode(render_prompt(item, tokenizer), add_special_tokens=False)))
    summary = summarize_lengths(lengths).as_dict()
    summary["sampling"] = f"every {every}th line of {MIX.relative_to(REPO)}"
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--eval_batches", type=int, default=20)
    ap.add_argument("--eval_batch", type=int, default=32)
    ap.add_argument("--budget_examples", type=int, default=200_000)
    ap.add_argument("--dev_items", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    t_all = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    lens = median_prompt_tokens(tok)
    L = lens["median"]
    print(f"[{utcnow()}] median prompt tokens={L} (mean {lens['mean']}, p95 {lens['p95']}, max {lens['max']})",
          flush=True)

    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    load_s = time.time() - t0
    n_base = sum(p.numel() for p in base.parameters())
    cfg = base.config.get_text_config() if hasattr(base.config, "get_text_config") else base.config
    hidden = cfg.hidden_size
    full_attn = [i for i, t in enumerate(getattr(cfg, "layer_types", [])) if t == "full_attention"]

    lcfg = LoraConfig(r=args.rank, lora_alpha=args.rank, lora_dropout=0.0, bias="none",
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg)
    for p in model.parameters():
        p.requires_grad_(False)
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    for p in lora_params:
        p.requires_grad_(True)
    n_lora = sum(p.numel() for p in lora_params)
    readout = torch.nn.Linear(hidden, 255, bias=True, dtype=torch.float32).to("cuda")
    opt = torch.optim.AdamW([{"params": lora_params, "lr": 1e-4},
                             {"params": readout.parameters(), "lr": 1e-3}],
                            weight_decay=0.0, fused=True)
    inner = model.base_model.model.model  # text backbone; LoRA layers are injected in place

    vocab = cfg.vocab_size
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    ids = torch.randint(0, vocab, (args.batch, L), device="cuda", generator=gen)
    n_opts = torch.full((args.batch,), 4, device="cuda")
    target = torch.zeros(args.batch, dtype=torch.long, device="cuda")
    mask = torch.arange(255, device="cuda")[None, :] >= n_opts[:, None]

    def step() -> tuple[float, float]:
        torch.cuda.synchronize()
        s = time.time()
        h = inner(input_ids=ids, use_cache=False).last_hidden_state[:, -1].float()
        logits = readout(h).masked_fill(mask, -1e9)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params + list(readout.parameters()), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return time.time() - s, float(loss.detach())

    step_times = []
    loss_values = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.steps):
        dt, loss_value = step()
        step_times.append(dt)
        loss_values.append(loss_value)
    timed = step_times[args.warmup:]
    sec_per_step = statistics.fmean(timed)
    peak_train_gb = torch.cuda.max_memory_allocated() / 1e9

    def eval_rate(batch: int) -> float:
        """Forward-only items per second at this length and batch size (dev-eval cost)."""
        x = torch.randint(0, vocab, (batch, L), device="cuda", generator=gen)
        with torch.no_grad():
            for _ in range(3):
                inner(input_ids=x, use_cache=False)
            torch.cuda.synchronize()
            s = time.time()
            for _ in range(args.eval_batches):
                h = inner(input_ids=x, use_cache=False).last_hidden_state[:, -1].float()
                readout(h)
            torch.cuda.synchronize()
        return args.eval_batches * batch / (time.time() - s)

    model.eval()
    eval_items_per_s = eval_rate(args.batch)
    eval_items_per_s_b32 = eval_rate(args.eval_batch)
    peak_all_gb = torch.cuda.max_memory_allocated() / 1e9

    budget_steps = args.budget_examples // args.batch
    n_evals = args.budget_examples // 2000
    tokens_per_s = args.batch * L / sec_per_step
    mean_len = lens["mean"]
    # Length-proportional projection: bucketed batches (the planned batch order) pad little, so the
    # cost of an example is roughly its own token count. The median-length projection is a lower bound.
    projected_train_s = budget_steps * sec_per_step
    projected_train_s_bucketed = args.budget_examples * mean_len / tokens_per_s
    projected_eval_s = n_evals * args.dev_items / eval_items_per_s
    best_eval_rate = max(eval_items_per_s, eval_items_per_s_b32)  # items/s at median length
    projected_eval_s_bucketed = n_evals * args.dev_items * mean_len / (best_eval_rate * L)
    payload = {
        "task": "O0",
        "utc": utcnow(),
        "git_commit": git_commit(REPO),
        "model": args.model,
        "gpu": gpu_name(),
        "recipe": {"lora_rank": args.rank, "lora_alpha": args.rank, "lora_targets": LORA_TARGETS,
                   "dtype": "bf16 base, fp32 adapters and readout", "batch": args.batch,
                   "sequence_length": L, "steps": args.steps, "warmup_steps_excluded": args.warmup,
                   "loss": "CE over 4 offered codes at the answer position", "clip_grad_norm": 1.0,
                   "optimizer": "AdamW fused, lr 1e-4 LoRA / 1e-3 readout"},
        "base": {"n_parameters": n_base, "hidden_size": hidden, "vocab_size": vocab,
                 "n_layers": cfg.num_hidden_layers, "full_attention_layers": full_attn},
        "lora": {"n_trainable_parameters": n_lora},
        "prompt_length_tokens": lens,
        "timing": {
            "load_s": round(load_s, 1),
            "sec_per_step_mean_timed": round(sec_per_step, 4),
            "sec_per_step_p50": round(statistics.median(timed), 4),
            "sec_per_step_max": round(max(timed), 4),
            "tokens_per_s_train": round(args.batch * L / sec_per_step, 1),
            "examples_per_s_train": round(args.batch / sec_per_step, 2),
            "eval_items_per_s_forward_b8": round(eval_items_per_s, 2),
            "eval_items_per_s_forward_b32": round(eval_items_per_s_b32, 2),
            "eval_batch": args.eval_batch,
            "wall_clock_total_s": round(time.time() - t_all, 1),
        },
        "memory_gb": {"peak_train": round(peak_train_gb, 2), "peak_incl_eval": round(peak_all_gb, 2)},
        "projection_one_arm": {
            "assumption": "example budget 200,000 (one pass over mix v2 is larger than 200,000); dev eval "
                          "every 2,000 examples (every 250 steps at batch 8) over dev_items, forward only. "
                          "Headline = length-proportional (mean prompt length, bucketed batches); "
                          "median-length figures are lower bounds",
            "budget_examples": args.budget_examples,
            "steps": budget_steps,
            "dev_items": args.dev_items,
            "n_dev_evals": n_evals,
            "mean_prompt_tokens": mean_len,
            "projected_train_s_bucketed": round(projected_train_s_bucketed),
            "projected_eval_s_bucketed_faster_batch": round(projected_eval_s_bucketed),
            "eval_rate_used_items_per_s_median": round(best_eval_rate, 2),
            "projected_total_h_bucketed": round((projected_train_s_bucketed + projected_eval_s_bucketed) / 3600, 2),
            "lower_bound_train_s_median_b8": round(projected_train_s),
            "lower_bound_eval_s_median_b8": round(projected_eval_s),
            "lower_bound_total_h_median": round((projected_train_s + projected_eval_s) / 3600, 2),
            "mean_to_median_length_ratio": round(mean_len / lens["median"], 3),
        },
        "step_times_s": [round(t, 4) for t in step_times],
        "loss_per_step": [round(v, 4) for v in loss_values],
        "fla_importable": importlib.util.find_spec("fla") is not None,
        "causal_conv1d_importable": importlib.util.find_spec("causal_conv1d") is not None,
        "torch": torch.__version__,
    }
    write_json(args.out, payload)
    print(f"[{utcnow()}] {args.model}: {sec_per_step:.3f} s/step at batch {args.batch} x {L} tokens; "
          f"eval {eval_items_per_s:.1f} items/s; peak {peak_all_gb:.1f} GB; "
          f"projected arm {payload['projection_one_arm']['projected_total_h_bucketed']} h (bucketed)",
          flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
