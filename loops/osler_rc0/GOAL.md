# GOAL — osler_rc0 (Release Candidate 0: Matched 4B + 9B, 12-Model Pareto Frontier, Calibration & PHI Audit)

Follow `loops/osler_rc0/ADVISORY.md` exactly. This loop produces release-ready **Osler-9B-rc0** and **Osler-4B-rc0** (matched
at 63,304 steps with identical mix and token-capped batching), establishes the ≤10B medical decision Pareto frontier against
10 carefully chosen comparison models across 9 frozen evaluation sets with full calibration (Brier/ECE) and latency, completes
the mandatory local MIMIC PHI/memorisation audit (D28), and stages verified release artifacts + model cards.

Nothing in `ADVISORY.md` changes after R0 starts. Any change requires an operator instruction recorded in `STATE.md §2`.

## Targets
- **Osler-9B-rc0** — frozen 63,304-step checkpoint `outputs/osler_v7/T11/final/checkpoints/step_063304.pt` on
  `vllm-sr/d3-flash@67e03e1bc157961c582ee7c6a35d9b15fada3dec` (8.39B). No retraining; dev-only temperature calibration + full
  probability scoring + PHI audit.
- **Osler-4B-rc0** — matched 63,304-step training run on `perplexity-ai/pplx-decider-flash-v1.2@9ff7cf9838e59df461c276b7699d56b244d2af58`
  (4.0B) using `scripts/osler_v7/4b_full_run.py --batching token` (Option A, no D57 quota, matching 9B's batching and KL dose),
  dev-only temperature calibration, full probability scoring + PHI audit.

## Read first, in this order
1. `AGENTS.md` (Current loop section, R1–R10, data & environment rules, guardrails).
2. `loops/osler_rc0/ADVISORY.md` (exact verified paths, pinned revisions, ID-bijection rules, CLI templates, checks, timeboxes).
3. `loops/osler_rc0/STATE.md` (task board R0–R9, operator approvals in §2, active checklists).

## Each iteration
1. Verify `git branch --show-current` is `loop/osler_rc0`. Never commit to `main`, `dev`, or `loop/osler_v7`.
2. Read `STATE.md`. Take the first `PENDING` task whose dependencies are `DONE` and whose approval is recorded in `STATE.md §2`.
3. Mark it `IN_PROGRESS` with `date -u +%FT%TZ` before starting work. Copy its acceptance checks into the task checklist.
4. Execute, verify output completeness (row counts, finite probabilities, hashes — never exit code alone), and write
   `outputs/osler_rc0/<task>/SELF_AUDIT.md` in a fresh process (R6).
5. Append `CLAIMS.md` rows for every reported number (R1; `NOT MEASURED — <reason>` and `BLOCKED — <reason>` are valid).
6. Mark `DONE` or `BLOCKED — <reason>`, append a concise log entry in `STATE.md §4`, run `uv run ruff check .` and
   `uv run pytest` before any commit that touches `src/`, `scripts/`, `tests/`, or `configs/` (loops-only commits skip pytest),
   and commit `loop(osler_rc0): R<n> <DONE|BLOCKED> — <headline>`.
   - **Do NOT `git push` unless `STATE.md §2` explicitly authorizes pushing.**

## Operational rules (no surprises)
- **Always** run `source scripts/pod_env.sh` before any GPU or `uv` job. Never write models/datasets/caches to `/root` or `/tmp`.
- **One GPU job at a time.** Run every GPU job detached (`setsid nohup ... > outputs/osler_rc0/<task>/logs/<name>.log 2>&1 < /dev/null &`).
- Check disk headroom (2 GB write test in `/workspace/tmp` and delete) before R1 and R8.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on R1 (4B 63,304-step training).
- Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` on all evaluation and audit scripts (all 12 models are already cached locally).
- Never overwrite a running shell script; if replacing a script, write a new filename and kill the old PID by explicit PID (never `pkill -f`).

## Do not
- **NEVER publish or upload** anything (no Hugging Face upload, no making repos public, no external posts). R8 stages locally under gitignored `outputs/osler_rc0/R8/release_staging/` only.
- **NEVER send MIMIC text** to any hosted API or commit MIMIC text, items, notes, predictions, soft labels, checkpoints, or secrets.
- **NEVER use test sets** (`panel`, `knowledge`, `confirm_general`, `mr_verdict`, `mimic_suite`, `mednli_test`, `mimic_heldout`, `mr_heldout_test`, `pairs_test`) for training, checkpoint selection, or temperature calibration. Calibration uses `fastdev_v4` dev items only.
- **NEVER re-select a checkpoint** after seeing test numbers. Both RCs are frozen before test scoring (`step_063304.pt` for 9B, `final_step_063304` for 4B).
- **NEVER edit a threshold, gold label, or test** to make a check pass (R8). Note: `tests/test_padding_trained.py` is a known pre-existing strict-xfail that XPASSes in some full-suite runs; if it fails for that exact reason, record it in `STATE.md` rather than modifying the test.
- **NEVER touch** `outputs/osler_v7/`, `outputs/osler_v4/`, `outputs/osler_v5/`, or `outputs/eval0/` raw predictions (R9: immutable). Read them; write all new outputs under `outputs/osler_rc0/`.
- **NEVER use the word "gate"** in user-facing summaries or model cards (say "evaluation check / comparison / release bar").

## Done when
Every task R0–R9 is `DONE` (or `BLOCKED` with a documented reason); no task is `IN_PROGRESS`; committed reports exist (`FINDINGS.md`, `LEADERBOARD.md`, `PER_CATEGORY.md`, `PARETO.md`, `AUDIT.md`, `MODEL_CARD_9B.md`, `MODEL_CARD_4B.md`); 5-claim fresh-process spot-check passes in `outputs/osler_rc0/R9/SELF_AUDIT.md`; `Loop status: STOPPED` is at the top of `STATE.md`; committed on `loop/osler_rc0`. Then stop and report: release readiness of 9B and 4B, headline Pareto numbers, PHI audit summary, and any open flags.
