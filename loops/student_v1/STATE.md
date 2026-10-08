# student_v1 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/student_v1/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (V10), or when no PENDING task can proceed without the operator -->
Run started (UTC): `2026-10-08T03:59:12Z`
Last updated (UTC): `2026-10-08T04:29:35Z`
Iterations so far: `1`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| V0 | Orientation, snapshot, Unsloth environment | smoke | — | DONE | 2026-10-08T03:59:12Z | 2026-10-08T04:17:00Z |
| V1 | JEV-9B on v0.2 + G1 recomputed per D16 | yes | V0 | PENDING | | |
| V2 | Padding fix + measured effect | small | V0 | PENDING | | |
| V3 | Checkpoint trajectory (seen vs held-out) | yes | V2 | PENDING | | |
| V4 | Training data v1 (diversity) | no | V0 | PENDING | | |
| V5 | Unsloth validation + converter + evaluator | small | V0, V4 | PENDING | | |
| V6 | Arm A: pointer head | yes | V2, V4 | PENDING | | |
| V7 | Arm B: letter-readout LoRA | yes | V4 | PENDING | | |
| V8 | Arm C: Clef-style head via Unsloth | yes | V5 | PENDING | | |
| V9 | Evaluation + G1 per arm + arm-vs-arm | yes | V1, V6, V7, V8 | PENDING | | |
| V10 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (exception in ADVISORY §6:
V4 may run on CPU while V1–V3 use the GPU). Never run two GPU jobs at once. A `BLOCKED` task
does not block unrelated work — record it and move on. No row may be left `IN_PROGRESS` at
closure.

---

## 2. Key values discovered during the run

Fill these in as they are measured; later tasks read them from here rather than
recomputing or guessing.

| key | value | source | task |
|---|---|---|---|
| unsloth / torch / transformers versions (unsloth env; main env) | unsloth env: unsloth 2026.10.2, torch 2.14.1+cu130, transformers 5.17.0, peft 0.21.2; main env: torch 2.14.1+cu130, transformers 5.18.0 | `outputs/student_v1/V0/envs.json` | V0 |
| G1 for student_v0 per D16 (seen / held-out) | | `loops/student_v1/g1_student_v0_rerun.md` | V1 |
| JEV-9B v0.2 coverage | | | V1 |
| padding effect on student_v0 test accuracy (max per-template Δ) | | `outputs/student_v1/V2/effect.json` | V2 |
| Spearman(step, held-out dev accuracy) with CI | | `loops/student_v1/trajectory.md` | V3 |
| training mix v1: items, distinct templates (vs student_v0), max template share | | `data/train/student_v1/manifest.json` | V4 |
| leakage check result | | | V4 |
| Unsloth typed-decisions validation (before → after) | | | V5 |
| arm A / B / C: selected step, dev macro, wall-clock, grad p95 / max | | | V6–V8 |
| G1 per arm (seen / held-out) | | `loops/student_v1/g1.md` | V9 |

---

## 3. Current task checklist

**Fill this in before starting any task**, by copying that task's concrete steps out of
the plan as unticked boxes. Tick each one the moment it is finished, not at the end.
This is the only record of progress *within* a task — without it, a compaction mid-task
leaves you unable to tell what you already did.

Clear this section and write the new task's checklist when you start the next task; the
completed checklist goes into the iteration-log entry.

```
Task in flight: none (V0 DONE 2026-10-08T04:20:00Z; V1 next)
Working dir:    outputs/student_v1/V1/

- [x] fill "Run started" in STATE and "Started" in AGENTS.md (V0 exception)
- [x] snapshot: git commit, v0.2 manifest + training-mix hashes, test count -> outputs/student_v1/V0/snapshot.json
- [x] create envs/unsloth/ (own uv project, pinned unsloth lockfile)
- [x] verify FastDecisionModel.from_pretrained Qwen3.5-0.8B loads bf16 (load_in_4bit=False) on GPU
- [x] record torch/transformers/unsloth versions in both envs -> outputs/student_v1/V0/envs.json
- [x] acceptance check run and passed
- [x] self-audit (R6) written to SELF_AUDIT.md
- [x] CLAIMS.md rows appended (V001-V007)
- [x] STATE updated, committed, pushed
```

---

## 4. Iteration log (append only, newest last)

<!-- Template for each entry:
### <task-id> — <DONE|BLOCKED> — <UTC timestamp>
- What ran: <command or script>
- Output: <path>
- Headline: <one number or one sentence, with its CLAIMS id>
- Surprises: <anything unexpected, or "none">
- Next: <what this unblocks>
-->

<!-- The Headline line matters beyond this file: these headlines are the raw material
     FINDINGS.md's Summary section is written from at closure. A headline that only
     makes sense with full context ("done, see report") starves the summary. Write
     each one so a reader who has not seen the task can repeat it: what was measured,
     what came out, with what denominator. -->

### V0 — 2026-10-08T04:17:00Z — DONE — 2026-10-08T04:17:00Z
- What ran: `scripts/student/v1_snapshot.py` (main + `--env unsloth`), `scripts/student/v1_unsloth_load.py` (in `envs/unsloth/.venv`); `uv sync --project envs/unsloth` (123 packages).
- Output: `outputs/student_v1/V0/{snapshot,envs,unsloth_load}.json`, `SELF_AUDIT.md`.
- Headline: Unsloth 2026.10.2 (first release with `FastDecisionModel`) loads Qwen3.5-0.8B in bf16 with `load_in_4bit=False`: 344 bf16 weight tensors; 251 norm/head tensors fp32; `predict` probabilities finite, sums 1.0 / 0.9999 (V001–V007).
- Surprises: the loader keeps RMSNorm weights in fp32 (arm C precision differs from A/B on norms, see Deviations); `causal_conv1d` is not installed in either env, so the reference kernel is used; the main env has transformers 5.18.0 vs 5.17.0 in the Unsloth env.
- Next: V1 (JEV-9B on v0.2 + D16 G1 rerun). Needs an F7-to-harness row converter and a G1 change that restricts verdict baselines to zeroshot + jev9b.

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|

---

## 6. Questions for the operator

Anything you could not resolve without a human. Be specific enough to answer without
re-reading the run: state the ambiguity, the options, and which you would pick.

---

## 7. Deviations from the plan

- **V0 (norm precision, decision deferred to V8):** Unsloth's `FastDecisionModel` loads the 251 RMSNorm/LayerNorm tensors in fp32; arms A/B load them in bf16. ADVISORY §3 asks for matched bf16 arms. Unresolved; V8 will either cast the norms to bf16 to match or record the difference in the per-arm provenance table.

Any place you departed from GOAL/ADVISORY, with the reason. An empty section is the
expected outcome. Editing code or a check to make it pass is never an acceptable
deviation — that is a `BLOCKED`.

---

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| | | | |
