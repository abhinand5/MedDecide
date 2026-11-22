# 4B full run (matched to the 9B final) — instructions for DeepSeek

Status: operator-requested, not part of the closed osler_v7 loop. Read it in full before you start.

## Why

The 9B final was trained for 63,304 steps. The 4B arm was trained for only 13,992. The panel comparison is therefore not
like-for-like. The 4B must run the same 63,304 steps with the same batching as the 9B final. Nothing else changes.

## Frozen (do not change)

- Mix: `data/train/osler_v7/mix_v7.jsonl` (manifest `outputs/osler_v7/T6/mix_manifest.json`; 1,526,405 rows).
- Config: `configs/osler_v7/arm_4b.yaml` (lr 0.001 head / 0.0001 LoRA, cosine, floor 0.1, LoRA r=32 alpha=32 on q/k/v/o, seed 0).
- λ_teacher = 3.243598; min labelled per batch = 3; kl_to_base = 0.1 on `v7_c*` rows only.
- Base: pplx-decider-flash-v1.2 (revision 851bf6e8…).
- Panel (`data/bench/osler_v4/y9/panel.jsonl`), knowledge, MR and MIMIC test splits: evaluation only. Never train, select
  or calibrate on them.

## Step 1 — port the 9B batching to the 4B path (STOP after this step and report)

The 9B final used token-capped batches and activation checkpointing. The 4B path (`scripts/osler_v7/t6_run_arm.py`) does
not have them. Reference commits: `965053a` (activation checkpointing, inference-tensor fix), `e763a02` (token-capped
batches, prepared-row cache, zero-KL guard), `8e606cc`. Port the same logic to the pplx 4B path; do not change the
mix, the loss, λ, the learning-rate schedule or the LoRA settings.

Parity check (run both, each 500 steps, same seed, one GPU job at a time):
1. old 4B batching and new token-capped batching both finish without OOM;
2. log per-step items, real and padded tokens, the fraction of steps with KL-to-base active, and mean CE/KL;
3. report the two sets side by side. The new batching should show KL-to-base active on about the same small share of steps as the 9B final (about 3%), not the old 40%.

Write the report to `outputs/osler_v7/4B_FULL/parity/`. **Stop and wait for the operator** before Step 2. The operator decides
on the batching change.

## Step 2 — the full run (only after the operator approves Step 1)

- Steps: 63,304. Evaluations at 13,992, 27,984, 41,976, 55,968 and 63,304 (the same points as the 9B final).
- Full checkpoints every 2,500 steps, with `--resume` available. Keep the step checkpoints needed for the final.
- Run detached: `source scripts/pod_env.sh && setsid nohup uv run --frozen --with pillow==12.3.0 --with "torchvision==0.29.0"
  python scripts/osler_v7/<new 4B script> --out outputs/osler_v7/4B_FULL/arm --steps 63304 > .../logs/arm.log 2>&1 < /dev/null &`
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (an OOM happened on a 4B run at step 2,525 before this).
- Outputs go under `outputs/osler_v7/4B_FULL/`. Check the outputs themselves, not only the exit code.

## Step 3 — score once, then stop

After the run finishes, score the frozen checkpoint once on each set, with the same scripts used for the 9B final:
- panel (`data/bench/osler_v4/y9/panel.jsonl`), knowledge, mr_verdict, mimic_suite (`data/bench/osler_v4/y9/*.jsonl`);
- use `scripts/osler_v7/score_items.py` (it writes probabilities, needed for Brier and ECE).

Output to `outputs/eval0/4b_full/`. Then run `scripts/eval0/leaderboard.py` and `scripts/eval0/category_table.py` after
adding the new model to the registry. Do not rescore or pick a different checkpoint after seeing the numbers.

## Guardrails

- Commit code changes only on `loop/osler_v7` with the message `loop(osler_v7): 4B-full <step> <DONE|BLOCKED> — <headline>`
  and the co-author line `Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>`. Never push to `main` or `dev`.
- Run `uv run ruff check .` and `uv run pytest` before any code commit.
- Never commit raw predictions, labels, checkpoints, keys, or MIMIC-derived text.
- Do not touch the 9B outputs or the eval0 panel predictions already written.
- Report wall-clock per step at the start of Step 2. If the projected time for 63,304 steps is over 72 hours, stop and report.
