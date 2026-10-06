# student_v0 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/student_v0/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (S14), or when no PENDING task can proceed without the operator -->
Run started (UTC): `2026-10-06T18:06:22Z`
Last updated (UTC): `2026-10-06T20:37:00Z`
Iterations so far: `1`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| S0 | Orientation, snapshot, fused kernels | smoke | — | DONE | 2026-10-06T18:06:22Z | 2026-10-06T18:25:39Z |
| S1 | Benchmark v0.2 fixes | small | S0 | DONE | 2026-10-06T18:25:57Z | 2026-10-06T20:37:00Z |
| S2 | Record–claim consistency templates | small | S1 | PENDING | | |
| S3 | Long-record slice + shared-prefix measurement | yes | S2 | PENDING | | |
| S4 | HLE medical subset (supplementary test) | small | S1 | PENDING | | |
| S5 | Tier-1 train-split builders | no | S1 | PENDING | | |
| S6 | Pre-window structured-gold data + training mix + leakage check | no | S2, S5 | PENDING | | |
| S7 | Training code: LoRA + pointer head | smoke | S0 | PENDING | | |
| S8 | Evaluation path for trained models | yes | S7 | PENDING | | |
| S9 | Train MedDecide-0.8B (instruct) | yes | S6, S8 | PENDING | | |
| S10 | Ablation: MedDecide-0.8B from Base | yes | S9 | PENDING | | |
| S11 | Gate G1 evaluation | yes | S9, S3, S4 | PENDING | | |
| S12 | Byte-identity + audit read-back | small | S9 | PENDING | | |
| S13 | Candidate dataset catalog | no | S0 | DONE | 2026-10-06T18:42:26Z | 2026-10-06T18:57:22Z |
| S14 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (exceptions in
ADVISORY §6: S13 may run while a GPU job runs; S11 may precede S10 when time is short).
Never run two GPU jobs at once. A `BLOCKED` task does not block
unrelated work — record it and move on.

---

## 2. Key values discovered during the run

Fill these in as they are measured; later tasks read them from here rather than
recomputing or guessing.

| key | value | source | task |
|---|---|---|---|
| fused kernels installed (yes/no, versions) | **yes** — `causal_conv1d` 1.7.0, `flash-linear-attention` 0.5.2 (triton 3.8.0); both bound by transformers 5.18 (`fallback_warnings` 2 → 0) | `outputs/student_v0/S0/kernels.json` | S0 |
| Qwen3.5-0.8B tokens/s at 8k / 16k prompt, before → after kernels | **40,809 → 124,526** (8k) and **36,399 → 106,697** (16k), best of 5 after warmup at length | `outputs/student_v0/S0/kernels.json` | S0 |
| v0.2 manifest sha256 | `a81f2a0377fdec7fc352f6be37608561384bfcc5c20057f12d4d8179ed0db60b` (45,009 items: 35,263 carried + 9,746 new) | `data/bench/v0.2/manifest.json` | S1 |
| consistency templates built (ids) + the held-out one | | `docs/benchmark/consistency_templates.md` | S2 |
| long-record slice size (items, templates) | | | S3 |
| HLE-med item count | | | S4 |
| training items per source (tier1_train / prewindow) | | `data/train/student_v0/manifest.json` | S5/S6 |
| leakage check result (counts removed / found) | | | S6 |
| training throughput (tokens/s, items/s) | | | S7 |
| S9 steps, tokens seen, wall-clock, best dev step | | | S9 |
| G1 verdict (seen / held-out) | | `loops/student_v0/g1.md` | S11 |

---

## 3. Current task checklist

**Fill this in before starting any task**, by copying that task's concrete steps out of
the plan as unticked boxes. Tick each one the moment it is finished, not at the end.
This is the only record of progress *within* a task — without it, a compaction mid-task
leaves you unable to tell what you already did.

Clear this section and write the new task's checklist when you start the next task; the
completed checklist goes into the iteration-log entry.

```
Task in flight: **S1 — benchmark v0.2 fixes** (started 2026-10-06T18:25:57Z)
Working dir:    outputs/student_v0/S1/

- [x] 1. nfcorpus_graded_score_v2: offer only levels present in the template pool; unit test (every offered level has >=1 item with that gold in test)
- [x] 2a. pubmed_mesh_major_choice_v2: MeSH tree-sibling distractors (built, screened; saturated -> recorded)
- [x] 2b. fda_class_choice_v2: moa/pe-sharing class distractors (built, screened, 9B 0.8705 <= 0.90)
- [x] 3. constant-answer check in eval/health.py + unit test (LFM2.5-350M pattern)
- [x] 4. prediction files carry run_id; readers dedupe by (run_id, item_id) and report raw vs unique counts
- [x] 5. data/bench/v0.2/ built with manifest; v1 versions marked superseded; unchanged templates keep identical item_ids
- [x] 6. three _v2 templates pass the screen (gold-in-state, BoW macro < 0.90, n_test >= 200)
- [~] 7. zero-shot done (5 cells reported, 1 D12-withheld); saturation FAILS for the MeSH template (0.9830) — recorded, fix investigated and proposed
- [x] 8. acceptance check run: 5 of 6 criteria PASS, the no-saturation criterion FAILS for one template (recorded)
- [x] 9. self-audit (R6) written to SELF_AUDIT.md
- [x] 10. CLAIMS.md rows appended (S014-S026)
- [x] 11. STATE updated, committed, pushed
```

---

## 4. Iteration log (append only, newest last)

### S0 — DONE — 2026-10-06T18:25:39Z
- What ran: `uv pip install causal-conv1d flash-linear-attention` (TORCH_CUDA_ARCH_LIST=12.0, built from source ~13 min); `scripts/student/s0_kernels.py --kernels off|auto --lengths 8192 16384 --reps 5 --warmup-at-length`; `scripts/student/s0_snapshot.py`
- Output: `outputs/student_v0/S0/{snapshot.json,kernels.json,kernels_before.json,kernels_after.json,SELF_AUDIT.md,pytest.xml}`
- Headline: fused kernels are installed **and bound** (transformers fallback warnings 2 → 0), and Qwen3.5-0.8B prefill goes **40,809 → 124,526 tok/s at 8k (3.05×)** and **36,399 → 106,697 tok/s at 16k (2.93×)**, peak allocated 2.33 → 1.97 GiB at 8k (S001–S006).
- Surprises: a single-shot measurement said the kernels made it **36× slower**; the cause was Triton JIT-compiling on the first call at each new sequence length (~7.5 s) landing inside the timed region — re-measured with a warmup pass at length (S007). Also, the first binding check read `func.__module__`, which `functools.wraps` copies from the *torch* function, so it reported a fallback that was not happening.
- Next: S1 (benchmark v0.2 fixes) is unblocked.

### S13 — DONE — 2026-10-06T18:57:22Z
- What ran: `scripts/bench/catalog_datasets.py` (HfApi metadata for 74 curated ids over 49 search angles); committed catalog regenerated offline with `--from-json` for verification
- Output: `docs/benchmark/dataset_catalog.md` (committed), `scripts/bench/catalog_datasets.py`, `outputs/student_v0/S13/SELF_AUDIT.md`; scratch `/workspace/tmp/s13/`
- Headline: **62 candidate datasets** catalogued — 14 `train-candidate`, 20 `eval-candidate`, 28 `reject`; provenance 37 human / 15 structured / 6 llm / 4 unknown (S010–S013). Documentation only: no rows downloaded, nothing trained.
- Surprises: `bigbio/mednli` turns out to be under a PhysioNet licence (credentialed data) — rejected, and not previously flagged in this loop; 18 rows carry no licence in the card field, 16 of which stay `UNKNOWN` and are demoted to eval-only.
- Next: nothing in this catalog may be trained on in this loop (ADVISORY section 7); it is input for the next review.

### S1 — DONE (one acceptance criterion failed and recorded) — 2026-10-06T20:37:00Z
- What ran: `build_v0_2.py` (672 s), `verify_v0_2.py`, `screen_v0_1.py` on v0.2, `run_s1_baselines.sh` (6 GPU cells), `report_baselines_v0_1.py`, `mesh_options_experiment.py` + 2 GPU cells
- Output: `data/bench/v0.2/` (gitignored), `loops/student_v0/{bench_v0_2.md,template_screen_v0_2.md,bench_v0_2_baselines.md}`, `outputs/student_v0/S1/`
- Headline: v0.2 is **45,009 items = 35,263 carried identical to v0.1 + 9,746 new `_v2`**; the three repairs are the score level set (offered {1,2}, every offered level a gold in test), MeSH tree-sibling distractors, and mechanism-sharing FDA class distractors. 0.8B/9B: score 0.5966 / `READOUT_FAIL`, MeSH **0.9580 / 0.9830**, FDA class **0.7803 / 0.8705** (S014-S025).
- Surprises: the MeSH template is **still saturated** (9B 0.9830 vs a <= 0.90 criterion) and it is not the copy shortcut (0.9491 without it) nor the option count (0.9614 at 8 options, paired on 1,838 records) — a measured negative result, recorded with a proposal to retire the template rather than loosen the rule. The 9B score cell is withheld by D12 (greedy 0.880). A first build failed acceptance on **1 record of 43,621 crossing splits** (v0.1 had moved its items to test); fixed by joining the record's existing split.
- Next: S2 (record-claim consistency templates) is unblocked; S3/S5 too.

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|

---

## 6. Questions for the operator

Anything you could not resolve without a human. Be specific enough to answer without
re-reading the run: state the ambiguity, the options, and which you would pick.

---

## 7. Deviations from the plan

Any place you departed from GOAL/ADVISORY, with the reason. An empty section is the
expected outcome. Editing code or a check to make it pass is never an acceptable
deviation — that is a `BLOCKED`.

1. **S0 — a timestamp was written from memory instead of `date -u`.** The first version of
   the S0 row recorded `18:27:30Z` as the finish time; the real time was `18:25:39Z`
   (commit `50064f6`). Caught and corrected in the same session, before any report used it.
   No other timestamp in this file was affected; every other one is copied from the output
   of `date -u +%FT%TZ` or from `git log --format=%cI`.
2. **S0 — lint ran behind a pipe.** The commit command chained `uv run ruff check . | tail -5`,
   so ruff's non-zero exit was masked by `tail` and a commit with 6 lint errors landed
   (`50064f6`). Fixed in `af65aba` (unused `noqa` directives removed) rather than by
   rewriting the pushed commit. Later commits run ruff without a pipe so its exit status
   decides the chain.
3. **S0 — measurement protocol changed after the first result.** The first "after kernels"
   measurement was single-shot and reported a 36× slowdown; it was re-measured with a warmup
   pass at the measured length to remove the JIT cost. The single-shot artifact is kept
   under `logs/`, not deleted, and the change is recorded in SELF_AUDIT §3 and CLAIMS S007.

---

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| | | | |
