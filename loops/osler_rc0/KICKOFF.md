# KICKOFF — osler_rc0

Paste the block below into a fresh DeepSeek goal-loop session on `/workspace/MedDecide` once `STATE.md §2` has your GO + GPU approval.

---

You are executing loop `osler_rc0` in `/workspace/MedDecide` end-to-end from R0 through R9.

1. Confirm `git branch --show-current` is `loop/osler_rc0` (if not, `git switch loop/osler_rc0`). Do NOT `git pull` (we are on the same machine). Never commit to `main`, `dev`, or `loop/osler_v7`.
2. Read, in exact order:
   - `AGENTS.md` (Current loop + universal rules R1–R10 + data/environment rules)
   - `loops/osler_rc0/GOAL.md`
   - `loops/osler_rc0/ADVISORY.md` (every verified path, pinned revision, ID-bijection rule, CLI template, and timebox is in here — follow it verbatim)
   - `loops/osler_rc0/STATE.md` (STATE.md always wins over memory)
3. Blanket Loop GO, blanket GPU approval (R1–R6, R8), Option A for R1, and `git push -u origin loop/osler_rc0` per task are **already GIVEN** in `STATE.md §2`. Run autonomously end-to-end from R0 through R9 without pausing to ask the operator unless a hard guardrail trips.
4. Execute R0 → R9 in strict order, one GPU job at a time:
   - Always `source scripts/pod_env.sh` first. Run every GPU job detached with `setsid nohup ... > outputs/osler_rc0/<task>/logs/<name>.log 2>&1 < /dev/null &` and poll conservatively (every 15–20 min during R1). If R1 is interrupted or OOMs, resume cleanly with `--resume`.
   - Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for R1 (`4b_full_run.py --batching token --steps 63304`).
   - Use `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for all scoring and audit jobs (all 12 models are already cached locally).
   - Never touch or overwrite raw predictions in `outputs/osler_v4/`, `outputs/osler_v5/`, `outputs/osler_v7/`, or `outputs/eval0/`.
   - Never use test/panel/MIMIC-suite/held-out sets for training, checkpoint selection, or calibration. Fit 9B temperatures on `fastdev_v4` dev only.
   - Never send MIMIC text to any hosted API and never print or commit MIMIC text, raw items, predictions, checkpoints, or secrets.
   - Never edit a check, threshold, or test to make it pass. Apply the 30-min timebox on optional competitor wiring (`JEV-9B` on `confirm_general`, latency wrappers) and report `NOT MEASURED — <reason>` if exceeded.
   - Never publish or upload anything to Hugging Face or GitHub releases.
5. After each task: verify output completeness (row counts, finite probs, hashes), write `outputs/osler_rc0/<task>/SELF_AUDIT.md` in a fresh process, append `CLAIMS.md` rows, update `STATE.md`, run `uv run ruff check .` and `uv run pytest` whenever code changed (`src/`, `scripts/`, `tests/`, `configs/`), and commit `loop(osler_rc0): R<n> <DONE|BLOCKED> — <headline>` (respect `STATE.md §2` on whether to push).
6. Stop at R9 when `FINDINGS.md`, `LEADERBOARD.md`, `PER_CATEGORY.md`, `PARETO.md`, `AUDIT.md`, `MODEL_CARD_9B.md`, and `MODEL_CARD_4B.md` are committed, the 5-claim fresh-process spot-check passes, and `Loop status: STOPPED` is written at the top of `STATE.md`. Report the release-readiness summary concisely without using the word "gate".
