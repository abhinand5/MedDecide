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
Last updated (UTC): `2026-10-08T01:56:21Z`
Iterations so far: `1`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| S0 | Orientation, snapshot, fused kernels | smoke | — | DONE | 2026-10-06T18:06:22Z | 2026-10-06T18:25:39Z |
| S1 | Benchmark v0.2 fixes | small | S0 | DONE | 2026-10-06T18:25:57Z | 2026-10-06T20:37:00Z |
| S2 | Record–claim consistency templates | small | S1 | DONE | 2026-10-06T20:38:06Z | 2026-10-06T21:47:41Z |
| S3 | Long-record slice + shared-prefix measurement | yes | S2 | DONE | 2026-10-06T21:48:05Z | 2026-10-06T22:40:00Z |
| S4 | HLE medical subset (supplementary test) | small | S1 | DONE | 2026-10-06T22:39:33Z | 2026-10-07T00:04:50Z |
| S5 | Tier-1 train-split builders | no | S1 | DONE | 2026-10-06T22:42:40Z | 2026-10-06T23:00:21Z |
| S6 | Pre-window structured-gold data + training mix + leakage check | no | S2, S5 | DONE | 2026-10-06T23:19:43Z | 2026-10-07T00:04:50Z |
| S7 | Training code: LoRA + pointer head | smoke | S0 | DONE | 2026-10-07T00:05:13Z | 2026-10-07T00:58:42Z |
| S8 | Evaluation path for trained models | yes | S7 | DONE | 2026-10-07T01:15:06Z | 2026-10-07T02:06:26Z |
| S9 | Train MedDecide-0.8B (instruct) | yes | S6, S8 | IN_PROGRESS | 2026-10-07T02:06:36Z | |
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
Task in flight: **S9 — train MedDecide-0.8B (instruct)** (started 2026-10-07T02:06:36Z)
Working dir:    outputs/student_v0/S9/

Sizing (measured in S7, not estimated): 21.7 items/s over a real full pass -> ~163 min per pass over
212,481 items, GPU peak 32.1 GB. The 4 h box therefore fits ONE pass + dev evals + the temperature
fit (~15 min). The training set has no `score` items at all, so that qtype's temperature is not
identifiable (S8 recorded `NOT FITTED`).

- [ ] 1. training entry point (thin CLI over meddecide.train.Trainer) with the S9 recipe: LoRA r=16, lr 2e-4, batch 8, 8k tokens, one pass, dev eval every 500 steps, best checkpoint by dev Brier, then the dev temperature fit
- [ ] 2. launch detached with a pidfile + log under outputs/student_v0/S9/logs/, wall-clock budget inside the 4 h box
- [ ] 3. per-step log (loss, lr, tokens/s, GPU memory) and every dev eval recorded
- [ ] 4. best checkpoint saved with its config and tokenizer; eval with run_student.py
- [ ] 5. acceptance check run and passed
- [ ] 6. self-audit (R6) written to SELF_AUDIT.md
- [ ] 7. CLAIMS.md rows appended
- [ ] 8. STATE updated, committed, pushed
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

### S2 — DONE — 2026-10-06T21:47:41Z
- What ran: `build_consistency.py --window fresh` (243 s) and `--window prewindow` (828 s); `screen_v0_1.py` on v0.2; `run_s2_baselines.sh` (6 GPU cells); `report_baselines_v0_1.py`
- Output: 3 templates in `data/bench/v0.2/fresh/`, `data/train/student_v0/prewindow_consistency.jsonl`, `docs/benchmark/consistency_templates.md`, `loops/student_v0/consistency_baselines.md`, `outputs/student_v0/S2/`
- Headline: three record-claim templates built and measured — string-presence baseline **0.4985 / 0.3138 / 0.4750** (cap 0.60), all kept by the screen, zero-shot 0.8B/9B **0.6950/0.8780** (arm role, held out), **0.2517/0.7778** (multi-field claim set), **0.8875/0.9020** (route claim); pre-window training file **117,972 items** with the held-out template excluded (S027-S034).
- Surprises: the claim-set template was dropped by the screen in its first build (gold-in-state 1.000) because the stated claims were rendered into the state — fixed by stating them in the question; the route-claim's unsupported half leaked a construction cue that a BoW baseline read at 0.715 macro — fixed by requiring the same cue for both classes (0.530). Also found openFDA's `skip` cap of 25,000 (HTTP 400 at skip=25100) and made long windows fetch in month slices.
- Next: S3 (long-record slice + shared prefix) and S5 (tier-1 train splits) are unblocked.

### S3 — DONE (one task step failed and is explained) — 2026-10-06T22:40:00Z
- What ran: `tag_long_records.py`; `report_long_slice.py`; `measure_prefix.py` (3 versions: two bug fixes); `run_s3_0p8b.sh` (3 GPU cells)
- Output: `data/bench/v0.2/{long_record.json,long_record_items.json}` + manifest block; `loops/student_v0/long_record_v0_2.md`; `outputs/student_v0/S3/{prefix.json,long_slice.json}`
- Headline: the long-record slice is **2,224 of 24,263** fresh test items (>8,192 tokens; max 75,319); on it the 0.8B scores route claim **0.9915**, class **0.9451**, boxed warning **0.4296** against a long-subset majority of 0.835 (the one below-majority subset). Shared-prefix reuse is **slower** (24.14 vs 26.93 questions/s) and agrees with the plain path on **0.9697** of questions, both disagreements being ties (S035-S041).
- Surprises: (a) the harness was **truncating the question away** for the 584 items over the 16,384-token cap — fixed (keep the tail) and re-measured, +0.030 and +0.065 on the two affected templates, which corrects S020/S031; (b) my first shared-prefix implementation appended each question to the previous question's cache (a 0.73-margin flip), fixed by branching from a copy of the prefix cache; (c) a draft audit quoted "431 of 3,236" over-cap class items — the verified count is **137**, corrected in all three places it appeared.
- Next: S5 (tier-1 train-split builders) is unblocked; the 9B re-measurement continues in the background.

### S5 — started while S4's GPU cells are queued — 2026-10-06T22:42:40Z
- Why now: S5 is CPU/network-only and on the critical path (S6 depends on it); the ADVISORY sanctions CPU-only work while a GPU job runs. S4's remaining work is the queued 0.8B/9B cells, so S4 stays IN_PROGRESS and S5 starts in parallel (recorded as a benign parallel start, not a dependency shortcut).
- What ran: build_tier1_train.py (MedQA train, MedMCQA train capped 60k with the v0.1 dev carve excluded, SciFact train qrels, NFCorpus train qrels with the `_v2` pool rule, MedQuAD rows unused by tier 1) + leakage check vs every v0.2 test/dev split + independent gold verification.
- Output: data/train/student_v0/tier1_train.jsonl + manifest (gitignored); scripts/bench/build_tier1_train.py; tests/test_tier1_train.py

### S5 — DONE — 2026-10-06T23:00:21Z
- What ran: `build_tier1_train.py` (MedQA/MedMCQA/SciFact/NFCorpus/MedQuAD official train splits, 219 s, deterministic), leakage filter vs every v0.2 test/dev item, independent gold verification
- Output: `data/train/student_v0/tier1_train.jsonl` (106,184 items, 100 MB, sha256 `4ed0ae4e…`) + manifest; `scripts/bench/build_tier1_train.py`; `tests/test_tier1_train.py`
- Headline: **106,184 gold-only training items** (choice 99,573 + noul 6,611), gold verified **0 mismatches of 106,184**, leakage **2,188 removed / 0 collisions** against 57,007 v0.2 test+dev items (S043–S048).
- Surprises: the official NFCorpus **train** qrels are binary, so **no `score` items are constructible** — the student will face the score template without score-shaped training data (recorded, S046); MedQuAD over-exclusion by id (~7.6k rows) is deliberate and counted; the MedQA source ships two contradictory placeholder stems (dropped).
- Next: S6 (pre-window structured-gold mix + leakage check) is unblocked and now has both inputs (this file and the consistency pre-window file).

### S6 — started in parallel (CPU/network only) — 2026-10-06T23:19:43Z
- Why now: S6's deps (S2, S5) are DONE and it is CPU/network-only; the GPU is running S3's 9B re-measurement and S4's queued HLE cells. Same sanctioned parallel pattern as S5.
- What runs: pre-window ClinicalTrials.gov / openFDA / PubMed items (2023-01-01 -> 2026-02-28, balanced, cap 20k/template, held-out templates excluded) + the training mix (tier1_train + prefwindow_structured, dev = v0.2 tier-1 dev + fresh dev on non-held-out templates) + the four-part mechanical leakage check on the written files.
- Inputs reused: data/train/student_v0/{tier1_train.jsonl,prewindow_consistency.jsonl}.
- Output: data/train/student_v0/{train.jsonl,dev.jsonl,manifest.json} (gitignored); scripts/bench/build_training_mix.py; tests/test_training_mix.py

### S4 — DONE — 2026-10-07T00:04:50Z
- What ran: `build_hle_med.py`; two zero-shot cells (0.8B, 9B) after a harness fix (`ece_from_confidence` for mixed option counts — the runner crashed on 5-16-option items)
- Output: `data/bench/v0.2/supplementary/hle_med.jsonl` + manifest; `outputs/student_v0/S4/{model_*.json,results.json,SELF_AUDIT.md}`
- Headline: **141 HLE Biology/Medicine multiple-choice items** (no images, licence mit, revision `5a81a4c7`); zero-shot **0.1560 (22/141) for both 0.8B and 9B** — **at or below chance** (mean per-item 1/n_options **0.1706**, majority 0.2128), so **no skill claim** (S049-S052).
- Surprises: the D12 gate reads `chance = 0.0625` because it uses the template's *maximum* option count, so a mixed-count cell can be below its real chance and still pass — recorded as a gate gap, thresholds untouched (S051). Both models hitting exactly 22/141 is a coincidence of different error patterns (their picked-letter profiles differ).
- Next: S9 has its training mix.

### S6 — DONE — 2026-10-07T00:04:50Z
- What ran: `build_training_mix.py --stage all` (pre-window CT.gov + openFDA builds, mix, independent audit), CPU/network only
- Output: `data/train/student_v0/{train.jsonl,dev.jsonl,manifest.json,prewindow_*}` (gitignored); `scripts/bench/build_training_mix.py`; `tests/test_training_mix.py`
- Headline: **train 212,481 items** (tier1 106,184 + consistency 35,655 + pre-window structured 70,642; sha256 `2535d46d…`) and **dev 16,454** (sha256 `351cdeb0…`); the four-part leakage check is **0 on all four kinds** (7,711 removals counted), train ∩ dev = 0 (S053-S059).
- Surprises: **PubMed pre-window = `NOT MEASURED`** (annual baseline: 1,334 files / 31.4 GB / ~1.7 h, over the 60-min box; a subset would be a silent subsample); **no `score` items in training at all** (dev has 33) — the student will face the score template untrained on that shape; S2's consistency file had 2,495 items leaking into v0.2 dev/test, caught here.
- Next: S7 (training code: LoRA + pointer head) is unblocked.

### S7 — DONE — 2026-10-07T00:55:37Z
- What ran: the S7 implementation agent wrote `src/meddecide/model/{head,markers,meddecide_model}.py`, `src/meddecide/train/{config,data,losses,smoke,temperature}.py`, `configs/student_v0.yaml`, `tests/test_model_pointer.py`, and a 200-step smoke run (`outputs/student_v0/S7/smoke.json` + `checkpoint/`).
- Smoke (**corrected**): fixed-64-item reference loss 2.413 -> 1.401 (accuracy 0.25 -> 0.61); **21.7 items/s / 16.9k tokens/s** over a full slice pass, peak 32.1 GB; 11.6M trainable params; dev slice accuracy **0.611** (333/545, majority 0.378) - **smoke only, not a result**. The agent's interim report replaced my first numbers (73.1 items/s, 48.4 min/pass, dev 0.666): they came from the short end of the length-sorted batch plan.
- Anomalies recorded for investigation: (a) the loss **bounces** after step 100 rather than falling monotonically (`loss_curve_decreased: True` in the artifact is too generous - the curve's own numbers are quoted here); (b) the `score` temperature fit hit the search bound (49.9999) on 33 dev items with NLL 1.105 -> 1.099, i.e. it did not fit - the score qtype has only 33 dev items and no training items at all.
- Next: I must independently run the four acceptance tests (distribution-only, permutation, adapter-disabled byte-identity, 64-item overfit >= 0.95) and check the byte-identity test really compares against the untouched base, then write SELF_AUDIT + CLAIMS.

### S7 — follow-up running — 2026-10-07T01:14:52Z
- The S7 implementation agent is **still running** (session `14b8a3dc-bb6d-4b92-9264-783dcd23f866`) finishing its test suites and full report. Its code is already in the tree and committed (`e994a8d`, `c089537`); `uv run ruff check .` is clean and `uv run pytest -q tests/test_model_pointer.py` is **17 passed** at those commits.
- If it edits `src/meddecide/{model,train}/` again: re-run ruff + that test file, then re-read `outputs/student_v0/S7/smoke.json` before S9 sizes its step count from it. Nothing else in the loop depends on its report.

### S8 — DONE — 2026-10-07T01:45:44Z
- What exists: `scripts/bench/run_student.py`, `tests/test_run_student.py` (**18 tests pass in my run**), `outputs/student_v0/S8/{model_*.json,preds_*.jsonl,preds_*__dev.jsonl,recheck_test.json,scratch/}`. `uv run ruff check .` clean.
- Smoke-checkpoint numbers (test split, **a 200-step smoke checkpoint, not a result**): medmcqa 0.2950, medqa 0.3433, medquad 0.9660, mmlu 0.4152, nfcorpus_graded_score_v2 0.5000 -> **`READOUT_FAIL - constant_answer`**, nfcorpus_relevant_noul_v1 0.5518; tiers: tier 1 8 cells (6 pass / 2 fail), fresh 11 cells (10 pass / 1 fail), coverage 1.000 both.
- Verified and closed: temperature fit written (`outputs/student_v0/S7/checkpoint/temperature.json`): `choice` 0.9189 (n=11,170), `noul` 0.0921 (n=7,676), **`score` NOT FITTED (0 dev items)**; the inapplicable D12 checks are recorded per cell as `NOT APPLICABLE - pointer head reads the option states directly`, with an **additional** permutation check (0.260, n=200) labelled as additional. The S8 agent was still polishing when this closed; if it edits further, re-run ruff + `tests/test_run_student.py` and re-read the two model JSONs.

### S9 — running — 2026-10-07T02:06:44Z
- The S9 training agent is **running** (session `887043c8-04e7-469f-9e13-cbc8116f1da9`): it writes `scripts/bench/train_student.py`, launches the real ~2.7 h pass detached with a pidfile + log under `outputs/student_v0/S9/logs/`, and maintains **`outputs/student_v0/S9/RUN_NOTES.md`** as the handover document (exact command, pidfile, log path, current step).
- **To resume:** read `outputs/student_v0/S9/RUN_NOTES.md` first; check the pidfile with `kill -0`; if the process is gone, check the step log's completeness and whether `outputs/student_v0/S9/best/` and `best.json` exist **before** relaunching anything. One GPU job at a time - do not start S10/S11 work on the GPU while this runs.
- Expected artifacts: `outputs/student_v0/S9/{logs/*.jsonl,best/,best.json,temperature.json,model_*.json,SELF_AUDIT.md}`.
- **GPU serialisation (2026-10-07T02:07:02Z).** The S8 agent was still re-running its own acceptance evaluation (`scripts/bench/run_student.py --split test`, PID 112270, ~17 GB) when S9 started. The S9 agent has been told to finish CPU-only work, wait for that PID to exit, and only then launch the training pass - one GPU job at a time. If a future round finds two GPU jobs running, kill the redundant one and record it as a deviation (bench_v0_fix0's lesson: concurrent GPU jobs cost ~1 h of OOM re-runs).
- **GPU contention, measured (2026-10-07T02:22:41Z):** `nvidia-smi --query-compute-apps` shows **two** GPU processes and **no training run**: PID 112270 `run_student.py --split test` (S8 agent's redundant re-check of its post-edit code, 23,488 MiB, 18 min in) and PID 116048 `pytest -q --ignore=tests/test_train_student.py` (53,464 MiB - the repo's suite includes GPU-marked model tests, so running it *is* a GPU job). Total 77 GB of 96 GB. `outputs/student_v0/S9/RUN_NOTES.md` confirms **S9's training had not been launched** and the agent was waiting, so the timebox is intact but the "one GPU job at a time" rule is being bent by two *transient* jobs. Not killed: both are verification runs whose output is wanted, and neither is the training pass; recorded here as a deviation rather than hidden. If S9's launch ever overlaps a live job, kill the redundant one first.
- **GPU reserved for S9 (2026-10-07T02:32:57Z to ~05:30Z).** S8's re-check finished (wall 1342.7 s) and S9 has asked for the card exclusively. I have told the g1 agent (S11 machinery) and the S8 agent to stop all GPU work and use `pytest -m "not gpu"` from now on. S9 then runs one full `pytest -q` (~5 min) and launches the training pass: pidfile `outputs/student_v0/S9/train.pid`, log `outputs/student_v0/S9/logs/train_stdout.log`, wall-clock deadline **05:30Z** so the temperature fit and the post-training `run_student.py --split test` finish inside the 06:06Z box. **No other agent or job may touch the GPU until S9 reports it has exited.**
- **Card cleared (2026-10-07T02:33:35Z):** at 02:33Z the GPU was **full - 96,464 MiB of 96 GB** across two pytest runs (PID 117267 53,464 MiB and PID 118173 43,564 MiB `--ignore=tests/test_train_student.py`), which blocked S9's launch. I killed the redundant one (**118173**; its owner was already told to stop GPU work) and left 117267, asking S9 to confirm ownership: if it is not S9's, it gets killed too rather than starting a ~32 GB pass on a 53 GB card. Deviation recorded: the "one GPU job at a time" rule was broken twice today by *test suites* (a full `pytest` run loads models), so from here on non-owner agents use `pytest -m "not gpu"`.
- **S9 training is LIVE (2026-10-07T02:41:10Z).** Launched 02:34:02Z: `setsid nohup uv run python scripts/bench/train_student.py --max-seconds 10565 --eval-every 500 --out outputs/student_v0/S9`; **training PID 119600** (`outputs/student_v0/S9/train.pid`), logs `logs/train_stdout.log`, `logs/train_steps.jsonl`, `logs/dev_evals.jsonl`; deadline 05:30Z (box ends 06:06Z). Dev evals so far: step 2500 acc 0.4697, step 3000 acc 0.4844 (both improved and checkpointed). The foreign GPU processes (117267, 118794) have **exited**; the card is S9's alone at 9 GB and growing toward the 32 GB peak. **Do not start any other GPU job until S9 reports exit.**
- **S9 watch item (2026-10-07T03:01:34Z, measured - not a conclusion).** Step **13,500** of ~39,736 (about 34 % of the pass), 31 dev evals written. Recent dev evals on the 545-item stratified dev sample: step 12500 acc **0.4275** / macro 0.3003 / brier 0.6982, step 13000 **0.4110** / 0.3076 / 0.6644, step 13500 **0.4477** / 0.5298 / 0.6507 - i.e. **noisy and flat in the 0.41-0.49 band, below the 200-step smoke model's 0.611 on its own dev slice** (smoke: 2,048-item slice, mostly short CT/MedMCQA items; this run: the full 212,481-item mix including ~56k long CT/openFDA records, same 545-item dev sample). Hypotheses (prose, no numbers): the full-mix batches are longer and the fixed lr 2e-4 may be too high for the mix, or the length-sorted plan's early steps are dominated by easy long-template items. **Do not tune anything mid-run** - the recipe is fixed; S9's agent reports the whole trajectory, and S14 records the observation either way. GPU 13,463 MiB and the run is alive.
- **S9 watch item, resolved by the run (2026-10-07T03:21:42Z).** The flat 0.41-0.49 band was a **mid-run dip, not a failure**: at step **21,000** of ~39,736 (53 %, GPU 34,597 MiB, alive) the dev evals read step 20000 acc **0.5688** / macro 0.4070 / brier 0.4997, step 20500 **0.5688** / 0.4106 / 0.5246, step 21000 **0.5339** / 0.3870 / 0.5335. That is back in line with the smoke model's 0.611 on the same 545-item sample, with Brier down from ~0.65-0.70 to ~0.50-0.53. Both directions are recorded (the dip and the recovery); no tuning was done. On this pace the pass ends ~04:40-05:00Z, leaving the `score`-fit (on the S6 dev mix) and the post-training test evaluation inside the 06:06Z box.
- **S9 progress (2026-10-07T03:46:49Z):** step **25,000** (~63 %), alive, GPU ~34 GB; recent dev evals step 24500 acc **0.5248** / brier 0.5807, step 25000 **0.5174** / 0.6004 - holding in the 0.51-0.57 band, above the 0.378 majority and below the smoke model's 0.611. No intervention; the pass should end ~04:30-05:00Z and S9's agent will report the trajectory, the best step, the fits and the post-training test cells.
- **S9 progress (2026-10-07T04:12:00Z):** step **30,000** (~75 %), alive; dev evals step 29500 acc 0.4954 / brier 0.8019 / nll 2.162, step 30000 **0.4972** / 0.7503 / 1.864 - the band is noisy (0.43 -> 0.57 -> 0.50 across the pass), which is exactly why the recipe selects the checkpoint by **dev Brier** and not by the last step. `outputs/student_v0/S9/{best/,best.json,run_started.json,dev_eval_ids.json}` now exist, so the selection is being materialised as the run goes. Still no intervention.
- **S9 best checkpoint SELECTED (2026-10-07T04:37:13Z): step 18,000.** `outputs/student_v0/S9/best.json` records `selection_rule = "lowest dev Brier on the fixed dev evaluation sample; ties (|dBrier| <= 1e-12) broken by higher dev macro accuracy, then by the earlier step"` - the rule is materialised with the artifact, not asserted in prose. The selection is doing real work: at step **35,000** (88 %, still alive) the dev evals read acc 0.4532 / brier 0.7321, 0.4771 / 0.7699, 0.4752 / **0.7918** - i.e. the model got **worse** after step 18,000, so a last-step checkpoint would have been a mis-selection. For S14: the dev curve is non-monotone with a late decline (hypotheses: overfitting on the mix, or lr 2e-4 too high - unproven).
- **S11 machinery done (CPU-only):** `scripts/bench/g1.py` + `tests/test_g1.py` (23 tests - I ran them: pass in 0.73 s; ruff clean), dry run `outputs/student_v0/S11/g1_dryrun.{md,json}`, independent re-derivation `audit_g1.py` (ALL CHECKS PASS). The **real** verdict still needs (a) S9's checkpoint scored over all v0.2 test items, (b) zero-shot 0.8B v0.2 predictions, (c) JEV-9B v0.2 predictions (does not exist yet - a GPU job for after S9). `loops/student_v0/g1.md` is deliberately unwritten.
- **Handover items from S8's final report (for S11/S14):** (1) `ct_claim_set_choice_v1` 0.9165 on the smoke checkpoint needs the S11 construction-cue check (seen template, 19,997 pre-window training items); (2) the strict slice reads `strict_slice_start` from `data/bench/v0.2/fresh/manifest.json`, **not** `acceptance.json` - g1.py walks acceptance -> fresh manifest -> bench manifest and records the source; (3) pointer-head rows carry `label_mass = 1.0` as a **convention, not a measurement** (the softmax lies on the offered options), so the mass check is never evaluated - flag if a different convention is wanted; (4) the checkpoint's stored `noul` T=0.039 **degrades** test calibration vs T=1, so S8 reports an additional `uncalibrated` (T=1) block per cell; (5) **S9 must fit `score` on the S6 dev mix (33 items) or ship it uncalibrated** - the screened dev split has no score item; (6) S7's report calls `noul` T=0.039 a "search bound" hit, but the bounds are (0.02, 50) - a documentation error to correct in S14.
- Nothing else in the loop is blocked by it: S10/S11/S12 depend on S9, S14 depends on all.

### S9 run 2 — LAUNCHED with the operator's fixes — 2026-10-07T05:32:56Z
- **Run 2 progress (2026-10-07T05:58:07Z):** step **5,500** of 44,152 (~12 %), alive, 13 dev evals. Dev evals on the **new 2,019-item stratified sample** (n=2019, eval 44.5 s each): step 5000 acc **0.6315** / macro 0.4065 / brier 0.4266 / nll 0.7714, step 5500 **0.5835** / 0.3494 / 0.5134 / 1.0064. **Not comparable to run 1's or the smoke model's dev numbers**: those used a 545-item sample, this uses 2,019 stratified by template - any run-1-vs-run-2 dev comparison is invalid and must not be made.
- **Run 2 progress (2026-10-07T06:28:17Z):** step **11,000** of 44,152 (25 %), alive. Dev evals step 10500 acc **0.6275** / macro 0.4331 / brier 0.4279, step 11000 **0.6369** / 0.4089 / 0.4428 - **stable ~0.63 with no sign of run 1's monotone decline**, which is the outcome the curriculum fix was meant to produce (still a single interval's evidence, not a trend claim).
- **Run 2 progress (2026-10-07T06:58:24Z):** step **17,000** of 44,152 (39 %), alive. step 16500 acc **0.6132** / macro 0.4125 / brier 0.4392, step 17000 **0.6305** / macro **0.5102** / brier 0.4381. **This is the step region where run 1 peaked and then degraded (~17.5k)** - run 2 is holding ~0.61-0.63 with macro still improving, i.e. the curriculum fix's intended effect is visible at exactly the point run 1 failed.
- **Started 05:22:33Z**, training pid **126700**, `outputs/student_v0/S9/train.pid`; log `outputs/student_v0/S9/logs/train_stdout.log`, per-step `logs/train_steps.jsonl`, per-eval `logs/dev_evals.jsonl`; **deadline 09:10Z** (fresh 4 h box, GPU empty at launch). GPU ~45 GB (higher than run 1's 32 GB peak: eval batches are now 16 items / 32k tokens).
- **All five operator requirements are in the code, not just in prose:** (1) `iter_batches` cuts the epoch into chunks of 100 x batch_size, length-buckets **inside** a chunk, and shuffles the batch order per (seed, epoch) - bucketing decides membership (padding) only, never order; module docstring records the bug. (2) opt-in linear **warmup 3 % + cosine decay to zero on both** param groups, per-step `lr_head`/`lr_lora` logged, schedule in `run.json:training.schedule`. (3) selection = ADVISORY's **macro accuracy first, Brier tie-break, then earlier step**, verbatim in `best.json`. (4) dev-eval sample **2,019 items over 19 templates** (choice 1286 / noul 700 / score 33), sha256 `d22d84fcd928757f`, and the sampler **raises below 2,000**. (5) **the CLI refuses to start if Spearman |rho| >= 0.1** - run 2's planned epoch reports **rho = -0.0002** over 44,152 batches / 201,539,105 tokens, and the regression test shows the same statistic is ~+1 on a length-ordered epoch, so it can detect the bug it guards against.
- **Verification:** ruff clean; `pytest -o addopts=""` **279 passed** (the 23 new tests incl. the curriculum regression test, plus the 4 GPU tests); a 30-step GPU smoke of the full CLI path showed the warmup/cosine ramp (lr_head 5e-4 -> 1e-3 -> 5.8e-4) and macro-first selection.
- **Expected outcome, stated in advance by the agent:** the pass needs ~260 min against a 221 min step budget, so the run should stop at **~83 % of the epoch** and keep the best dev checkpoint (the ADVISORY's "steps that fit the box" case). `score` will read **`NOT FITTED - too few dev items`** (33 < 50) and ship **uncalibrated (T=1)** with the 33-item diagnostic recorded; `choice`/`noul` are fitted. S14 must say plainly whether the run completed or hit its budget.
- **To resume:** read `outputs/student_v0/S9/RUN_NOTES.md`; `kill -0 126700`; **no other GPU job until it exits** (one agent at a time on GPU work). `outputs/student_v0/S9_run1_sorted/` remains the preserved failed run 1.

### DEVIATION — S9 run 1 stopped by operator instruction: failed by pipeline, not by recipe — 2026-10-07T04:47:29Z
- **Fix progress (2026-10-07T05:12:43Z):** the code changes have landed in the working tree - `src/meddecide/train/{data,config,trainer}.py`, `scripts/bench/train_student.py`, `configs/student_v0.yaml`, `tests/test_train_student.py` (`data.py` now carries the chunked length bucketing + the Spearman regression test). The agent is running a **30-step sanity pass on 512 items** (`--train-limit 512 --steps 30 --eval-every 15`) on the GPU before committing the box - the correct order. The fresh S9 box has **not** started yet; its start time will be recorded when it does, and `outputs/student_v0/S9_run1_sorted/` stays untouched.
- **Instruction (operator, advisor-reviewed):** S9 run 1 is **invalid due to a pipeline bug**. Stop it, keep its artifacts under `outputs/student_v0/S9_run1_sorted/`, and record it here as failed-by-pipeline, **not as evidence about the recipe**.
- **Done:** killed training PID 119600 and launchers 119582/119592 (SIGTERM; all three stopped, no `train_student` process remains, GPU back to 0 MiB); moved `outputs/student_v0/S9/` -> **`outputs/student_v0/S9_run1_sorted/`** (RUN_NOTES.md, best/, best.json, run_started.json, dev_eval_ids.json, logs/, train.pid all preserved).
- **Root cause (operator's diagnosis):** `src/meddecide/train/data.py::iter_batches()` sorts the whole epoch by `char_proxy` **after** shuffling, so training ran shortest -> longest - an **accidental length curriculum**. The dev curve peaked ~step 17.5k and then degraded as batches got long. My own recorded watch items (the flat 0.41-0.49 band, the late decline to brier 0.79) were symptoms of this, and my "lr too high / overfitting" hypotheses were **wrong in mechanism** - the operator's diagnosis supersedes them.
- **Required fixes before any re-run:** (1) bucket by length **within shuffled chunks** (e.g. chunks of 100 x batch_size), form batches inside each chunk, then shuffle the **batch order** per epoch; unit test: Spearman |rho| between batch index and mean batch length **< 0.1** over one planned epoch. (2) **Linear warmup (3 % of steps) + cosine decay** on both param groups. (3) Selection rule per ADVISORY S9: **dev macro accuracy first, Brier as tiebreak** - run 1's `best.json` used Brier first, which I accepted without checking it against the ADVISORY; that is a miss on my part and is corrected here. (4) Enlarge the fixed dev-eval sample to **>= 2,000 items stratified by template**. (5) Re-run with a **fresh 4 h box** and record the new start time.
- **Process rule reaffirmed:** one agent at a time on GPU work; no parallel sub-agents for GPU tasks (CPU-only sub-tasks are fine).
- **Also my error (same class as an earlier one):** `pkill -TERM -f "train_student.py"` matched **my own shell's command line** and killed the shell, so the artifact move had to be re-run. Use explicit PIDs or `pgrep -f "[t]rain_student"`, never `pkill -f <literal>`.

### QUESTION FOR THE OPERATOR — ANSWERED: run 2 diverged; the selection rule was NOT the problem — 2026-10-07T15:41:17Z
Raised by the S9 agent at 07:08Z, at step 18,864/44,152 (42.7 %; items % == tokens %, i.e. the curriculum bug is gone), GPU peak 37.7 GB, projected finish ~09:05-09:17 against the 09:10Z deadline.
- **Measured dev trajectory (2,019-item template-stratified sample):** step 500 macro 0.608 / acc 0.677 / Brier 0.362; **step 1000 macro 0.7006 / acc 0.7132 / Brier 0.4601 <- the current best under the ADVISORY's macro-first rule**; steps 1,500-18,500 macro oscillates **0.28-0.57**, micro accuracy 0.59-0.64, Brier improving to ~0.43.
- **The finding:** the macro-first rule selects the **least-specialised** checkpoint (~2.5 % of a pass). A **Brier-first rule would have selected step 500** - the least-trained model of all - which is independent confirmation that run 1's Brier-first rule was wrong. Later checkpoints are better calibrated and similar on micro accuracy but have much worse **rare-class recall on the equal-weight-per-template sample**. Mechanism (hypothesis, prose): CE on the imbalanced training mix drives the model to specialise onto frequent answers, which micro accuracy tolerates and macro accuracy punishes.
- **Why it is time-sensitive:** only checkpoints that improved under the rule were saved, so a different selection key **cannot be applied retroactively** - it needs either a fresh run or checkpoints saved from now on. The run ends ~09:10Z.
- **What I will NOT do:** change the selection rule mid-run to get a better-looking number - that is editing a check to pass it (R8). The agent is holding the rule and `best.json` will carry the full trajectory plus the Brier-first alternative.
- **Options put to the operator:** (a) keep the ADVISORY rule and report the trajectory finding; (b) keep the run but save every dev eval from now on so a dev-based reselection is possible later; (c) change the key and re-run fresh (another 4 h box); (d) stop and re-plan.
- **Consequence if (a):** `outputs/student_v0/S9/best/` is a step-1000 snapshot and the post-training test cells (and therefore G1) describe that snapshot - a **selection** outcome, not a capability verdict on the recipe, and S14 must say so in exactly those terms.

### DEVIATION — S9 run 2 DIVERGED (stopped 07:21:43Z) — operator ruling — 2026-10-07T15:41:32Z
- **Operator ruling:** the selection rule is **not** the problem. Run 2 **diverged**: train loss never falls meaningfully (0.91 at steps 0-2k, best 0.84, 1.78 at 20k), **pre-clip grad norm averages 20-95 with clip 1.0 and spikes to ~25,000 at step 18k**, and dev collapses to acc 0.34 / macro 0.11 by step 21.5k. Keep the ADVISORY selection rule unchanged. **Record run 2 here as diverged - not a result about the recipe.**
- **My earlier reading was wrong:** I framed it as a selection-metric problem (macro-first picking a step-1000 snapshot). The metric observation is real but secondary; the run was diverging. Superseded.
- **It was not the curriculum:** planned epoch rho = **-0.00023** over 44,152 batches, and items % equalled tokens % to 0.1 pt for the whole run.
- **Run 2 chronology:** launched 05:22:33Z (pid 126700, deadline 09:10Z) -> stopped **07:21:43Z at step 21,574 / 44,152 (104,134 items, 98.5 M of 201.5 M tokens, 48.9 %)** by the agent's stop-and-report rule (the rule I put in its brief: stop rather than burn the box when the dev pattern looks wrong). Throughput **22.69 items/s / 21,474 tokens/s**, GPU peak 37.72 GB; the planned epoch held **201.5 M tokens** (S7 extrapolated 165 M from a stride slice), so the 4 h box could never have held a full pass.
- **Artifacts finalised (they describe a diverged run's step-1000 checkpoint, NOT the recipe):** `temperature.json` choice 4.2743 (n=10,967), noul 11.6125 (n=5,454), **score NOT FITTED (n=33 < 50) -> shipped uncalibrated**; `dev_final.json` full S6 dev 16,454 items acc 0.7283 / macro 0.7056 / Brier 0.3360; test cells tier 1 **0.6175** (13,521 items, 7/8 PASS) and fresh **0.8108** (23,768, 9/11 PASS + 1 additional-check fail), 3 cells `READOUT_FAIL - constant_answer`. Leakage control: above the zero-shot letter baseline on same-template fresh cells, **below it on all four D14 held-out templates** -> readout effect + same-template training, not leakage.
- **Next per the operator: S9-diag** (GPU timebox **3 h**): 2,000-step runs on a fixed 20k-item subset, one change at a time, each reporting train-loss slope, grad-norm distribution and dev macro on the same dev sample - (a) head LR 1e-4 / LoRA 5e-5; (b) `max_prompt_tokens` 2048 with per-batch grad norm logged against batch max length and templates; (c) **padding check**: the same item alone vs inside a left-padded batch must give the same head logits within 1e-3 (Qwen3.5 linear-attention + left padding); (d) LoRA rank 8. Then the most stable config -> **S9 run 3 saving a checkpoint at every dev eval** -> S10 -> S11. **If no config is stable in the timebox: skip S10, run S11 on the best available checkpoint, and close the loop with the diagnosis as the main finding (ADVISORY section 2, "diagnose the recipe at 0.8B").**
- Run 1's artifacts remain in `outputs/student_v0/S9_run1_sorted/`; run 2's in `outputs/student_v0/S9/`.

### S9-diag — running (GPU timebox 3 h from ~15:45Z) — 2026-10-07T16:02:00Z
- **Padding check (operator's probe (c)) — RESULT: it is a bf16 precision effect, not a positional bug.** Same item scored alone vs inside a left-padded batch, 8 items, tolerance 1e-3: **bf16 max |delta| = 0.0080 / mean 0.0041 -> FAILS the tolerance**; **fp32 max |delta| = 0.00040 / mean 0.00018 -> within tolerance**. So left padding with Qwen3.5's linear-attention layers is *not* semantically broken; bf16 accumulation across a padded batch perturbs head logits at the ~1e-2 scale. Artifacts: `padding_check_fresh.json`, `padding_check_fresh_fp32.json`.
  - **Why it matters for the divergence (hypothesis, prose, unproven):** a forward pass whose logits depend on batch composition at the 1e-2 scale means gradients inherit that batch-dependent noise, which is a plausible contributor to the grad-norm spikes; it is *not* established as the cause.
- **Arms so far:** `arms/baseline/` and `arms/a_low_lr/` exist (2,000 steps each, `--eval-every 50`); a further arm is on the GPU now (2.1 GB footprint). Each arm reports train-loss slope, grad-norm distribution (p50/p95/max, pre-clip) and dev macro on the same fixed sample; `diag.json`/`diag.md` will hold the comparison table. `RUN_NOTES.md` is the handover doc.
- **Not yet answered:** which batches spike and against which max length/templates (arm b), the LoRA-rank-8 arm (d), and the chosen config. Then either **run 3** (checkpoint at every dev eval, per the operator) or, if nothing is stable inside the timebox, **stop and report** so the ADVISORY section 2 fallback applies (skip S10, S11 on the best available checkpoint, diagnosis as the loop's main finding).

### S9-diag — KEY FINDING: the trained head's logits depend on batch composition (padding), far beyond precision noise — 2026-10-07T16:27:22Z
- **Padding check at the *trained* checkpoint (run 2's), tolerance 1e-3:** fresh/untrained head - bf16 **8.00e-03 FAIL**, fp32 **3.97e-04 PASS** (precision-level, as recorded earlier). **Trained head - bf16 9.45e-02 FAIL, fp32 2.51e-02 FAIL**, and the uniform-batch control is 1.40e-03 in fp32 (also above tolerance). Artifacts: `padding_check_fresh{,_fp32}.json`, `padding_check_trained{,_fp32}.json`.
- **What it means:** for the untrained head this is a bf16 precision effect; for the **trained** head it is a **real batch-composition dependence in fp32 too** (25x the tolerance), i.e. the same item scored alone and inside a left-padded batch gives *different head logits*, and training **amplified** the discrepancy by ~60x. Mechanism not yet established (candidates: the `key_end` marker positions under left padding, or the head exploiting padded positions as a degenerate shortcut). Either way it is a **correctness bug in the padding path, not merely a numerical one**, and a plausible contributor to the divergence - stated as a hypothesis, with the numbers above as the evidence.
- **Arm baseline (run-2 recipe on the fixed 20k subset), 2,000 steps:** loss 1.053 -> 0.954 over the last 500 steps, slope **-0.2756 +/- 0.1558** per 1k steps (decreasing, slowly), pre-clip grad **p50 6.76 / p95 36.25 / max 971.9**, **8 steps with grad > 100**, dev macro 0.599 / 0.600. Arms `baseline`, `a_low_lr`, `b_cap2048`, `d_rank8` have all run; one more job is in flight.
- **Spiking steps are longer batches:** mean batch max-len **3,760 on spike steps vs 2,190 overall**, top templates in spikes ct_randomised_noul_v1 21 %, fda_class_choice_v1 18 %, ct_healthy_volunteers_noul_v1 18 %, medquad_routing_v1 18 % - i.e. the instability tracks **length**, which is consistent with the padding bug above.
- **Consequence for the loop:** a stable recipe cannot be chosen while the head's logits depend on batch composition; the padding path must be fixed (or the head made padding-invariant) before run 3 is worth 4 h. The full arm table and the diag agent's verdict are still coming; `outputs/student_v0/S9_diag/{diag.md,diag.json,RUN_NOTES.md}`.

### S9-diag — DONE (inside the 3 h box): config picked, run 3 launched — 2026-10-07T16:48:12Z
- **Run 3 progress (2026-10-07T17:18:26Z):** step **5,500**, alive, and **a checkpoint at every dev eval confirmed on disk** (`checkpoints/{step_500,...,step_5500}`). Dev evals on the **same 2,019-item sample as run 2** (sha256 d22d84fcd928757f, so this comparison is legitimate): step 5000 acc **0.7231** / macro **0.6545** / brier 0.3377 / nll 0.6068; step 5500 acc **0.7276** / macro **0.6632** / brier 0.3482. Run 2 on this sample at the same step count was around acc 0.63 / macro 0.4 and already sliding toward divergence; run 3 is **higher and stable 11x further into training than run 2's best-ever step (1000, macro 0.7006)** - still one window's evidence, and the deadline (20:45Z) plus the full trajectory decide the outcome.
- **Run 3 progress (2026-10-07T17:48:38Z):** step **11,000** (of ~44k), alive. step 10500 acc **0.7261** / macro 0.5588 / brier 0.3416 / nll 0.6480, step 11000 acc **0.7335** / macro 0.6079 / brier 0.3435. Same dev sample as run 2, and run 2 at this step count was acc ~0.63 / macro ~0.41 / brier ~0.44 and heading for divergence at ~18k - run 3 is higher, better calibrated (brier ~0.34 vs ~0.44) and stable. **The decisive window is ~18k, where run 2 broke**; macro is the noisier of the two series (0.56-0.66 here).
- **Run 3 at the decisive window (2026-10-07T18:18:49Z):** step **17,000** (run 2 broke at ~18k), alive. step 16000 acc 0.7127 / macro 0.5560 / brier 0.3583; step 16500 acc **0.7316** / macro 0.5391 / brier **0.3314**; step 17000 acc 0.7142 / macro 0.5839 / brier 0.3573. Run 2 at 16.5-17k was acc ~0.61-0.63 / brier ~0.44 / macro sliding through 0.28-0.41 on its way to divergence. **Run 3 is stable where run 2 failed** - the next evals (18k-22k) decide whether `d_rank8` + warmup/cosine actually fixed it or only delayed it. Macro remains the noisy series (0.54-0.58 here vs .66 earlier).
Fixed 20k-item subset (`limit=20000, stride=10`), fixed 2,019-item template-stratified dev sample (sha256 `d22d84fcd928757f`), 2,000 steps each, eval every 500, warmup 3 % + cosine, pre-clip grad norms, sequential on the GPU. Artifacts `outputs/student_v0/S9_diag/{diag.md,diag.json,arms/,RUN_NOTES.md}`.

| arm | change | loss (first->last 500) | slope/1k (CI) | grad p50 / p95 / max | #>100 | dev macro @500/1000/1500/2000 |
|---|---|---|---|---|---|---|
| baseline | run-2 recipe | 1.053 -> 0.797 | -0.192 +/- 0.061 | 5.79 / 31.07 / **971.9** | 14 | .599/.600/.613/.570 |
| a_low_lr | head 1e-4, LoRA 5e-5 | 1.006 -> 0.695 | -0.229 +/- 0.055 | 10.00 / 33.35 / 615.2 | 10 | .556/.627/.592/.582 |
| b_cap2048 | prompt cap 2048 | 1.252 -> 0.939 | -0.217 +/- 0.053 | 8.65 / 31.90 / **325.4** | 10 | .519/.585/.622/.650 (different rendering) |
| **d_rank8 <- PICKED** | **LoRA r=8, alpha 16** | 1.080 -> 0.735 | -0.250 +/- 0.061 | 6.10 / **29.47** / 457.8 | 10 | **.649/.655/.681/.666** |
| e_rank8_lowlr | r=8 + low LRs | 1.027 -> 0.687 | -0.247 +/- 0.054 | 9.65 / 32.86 / **310.0** | **7** | .569/.594/.605/.569 |

- **Why d_rank8:** best dev macro at *every* eval among cap-comparable arms (peak .681 / final .666 vs baseline .613/.570), lowest grad p95, worst-case grad less than half the baseline's, loss decreasing with a CI excluding zero. Rejected: `e_rank8_lowlr` (lowest max grad but clearly worse dev macro), `b_cap2048` (truncating to 2048 would remove the long-record capability S3/S11 measure), `a_low_lr` (best subset fit but no dev/tail win).
- **Spiking steps are length-driven (arm b's question answered):** baseline spike steps carry mean batch max length **5,126 vs 2,217 overall (2.3x)**; `fda_class_choice_v1` is 6.3 % of batches but **22.2 % of spikes** (44.8 % in arm b), `fda_class_choice_v2` 4.3 % -> 11.1 %, while short `medmcqa_4opt_v1` (29.8 % of batches) is only 8.3 % of spikes. Capping at 2048 removes the worst tail (max 971.9 -> 325.4) without moving the bulk (p95 31.07 -> 31.90).
- **Padding check verdict (arm c):** fresh/untrained - bf16 FAIL 8.00e-3 but **fp32 PASS 3.97e-4** (so position/mask handling is sound); **trained - bf16 9.45e-2, fp32 2.51e-2, i.e. 17.9x the no-padding control -> a real train/eval batch-composition inconsistency** (linear-attention + pads). The agent's calibrated reading: magnitude is small next to a trained head's logit spread, so it is a **contributing noise source, not on its own the gradient-explosion mechanism**. Recorded as such - the earlier stronger phrasing is superseded.
- **Caveats recorded with the pick:** the subset averages 653 tokens/item vs the full mix's 948, so the arms **understate** length pressure; and 2,000 steps is 4.5 % of a pass, so "stable" means *no spiking tail and a clean slope in that window*, not proof for a full pass.
- **S9 run 3 LAUNCHED 16:42:39Z:** `train_student.py --max-seconds 14549 --eval-every 500 --save-every-eval --lora-rank 8 --out outputs/student_v0/S9_run3`, **pid 135436**, pidfile `outputs/student_v0/S9_run3/train.pid`, log `S9_run3/logs/train_stdout.log`, **a checkpoint at every dev eval** in `S9_run3/checkpoints/step_<n>/` (the operator's requirement, recorded in `run.json:dev_evals.checkpoints_per_eval`), **deadline 20:45Z** (4 h box), then the CLI does the temperature fit and the per-template dev report. Handover: `S9_run3/RUN_NOTES.md`. No other GPU job may run until it exits.

### S9 run 3 — PAST THE DIVERGENCE WINDOW, stable — 2026-10-07T18:49:01Z
- **Step 22,500, alive**, i.e. **past 18k-21.5k where run 2 collapsed**. Dev evals on the same 2,019-item sample: step 21500 acc 0.7276 / macro 0.6587 / brier 0.3643; step 22000 acc **0.7330** / macro **0.6670** / brier 0.3417; step 22500 acc 0.7320 / macro 0.5812 / brier **0.3337**.
- **The contrast that carries the finding:** when run 2 was stopped at step 21,574 its dev had collapsed to acc ~0.34 / macro 0.11 with pre-clip grad spikes to ~25,000; run 3 at step 22,500 reads acc ~0.73 / macro ~0.58-0.67 / brier ~0.33-0.36.
- **What this does and does not establish:** the combination of (i) the chunked length bucketing (curriculum fix), (ii) warmup 3 % + cosine on both groups, and (iii) **LoRA r=8** (the diag's pick) is **stable 25 % further into training than the recipe that diverged** - at the same dev sample and the same batch plan. It does **not** yet establish a full pass (deadline 20:45Z), nor which single change mattered: the diag's arms were run one at a time on a 20k subset that understates length pressure, so the attribution between schedule and rank is **not measured**.
- **Run 3 progress (2026-10-07T19:19:09Z):** step **28,500** (~65 %), alive, still stable - step 27500 acc 0.7251 / macro 0.5788 / brier 0.3498, step 28000 0.7301 / 0.5755 / 0.3263, step 28500 acc **0.7370** / macro 0.5695 / brier 0.3378. No sign of the run-2 tail; deadline 20:45Z, then the CLI's temperature fit + per-template dev report, then the test evaluation.
- **Run 3 progress (2026-10-07T19:49:17Z):** step **34,000** (~77 %), alive, and the best numbers so far - step 33000 acc 0.7395 / macro 0.6129 / brier 0.3275, step 33500 acc **0.7464** / macro **0.6607** / brier **0.3241**, step 34000 acc 0.7311 / macro 0.6574 / brier 0.3290. `best/` + `best.json` exist (selection tracking under the ADVISORY macro-first rule). Deadline 20:45Z (~55 min), after which the CLI writes the temperature fit and the per-template dev report.
- **Run 3 near the end of the pass (2026-10-07T20:24:25Z):** step **41,000 of 44,152 (93 %)**, alive, and the **best and most stable readings of the whole loop** - step 39500 acc **0.7548** / macro 0.6652 / brier 0.3176, step 40000 0.7494 / 0.6633 / 0.3169, step 40500 0.7499 / 0.6634 / 0.3182, step 41000 0.7479 / 0.6631 / 0.3182. For contrast, run 2 (diverged) read acc ~0.34-0.45 / macro 0.11-0.18 in this region. Deadline 20:45Z. `best.json`/`best/` present; the temperature fit + per-template dev report are written by the CLI at the end of the run.
- Remaining risk recorded earlier still applies: the length-driven grad tail in the long openFDA/CT records (arm b localised it; `--max-prompt-tokens 2048` is the untried-in-full-run fallback).
- Checkpoints exist at **every** dev eval (`S9_run3/checkpoints/step_<n>/`), so the ADVISORY's macro-first selection can be re-examined on dev without another 4 h run.

### S9 run 3 — COMPLETED THE FULL PASS (no early stop) — 2026-10-07T20:44:35Z
- **`[S9] training done: steps=44152 epochs=1 stopped_early=None wall=13867s items=212481 items/s=22.64`** - the **entire planned epoch** (44,152 batches / 212,481 items), **not** a budget cut, in **3 h 51 m** against the 14,549 s (4 h) budget. Measured throughput 22.64 items/s (the S7 estimate was 21.7).
- **Final dev trajectory (2,019-item sample):** converged and flat at the end - step 39500 acc **0.7548** / macro 0.6652 / brier 0.3176, step 40000 0.7494 / 0.6633 / 0.3169, step 40500 0.7499 / 0.6634 / 0.3182, step 41000 0.7479 / 0.6631 / 0.3182. Run 2 (diverged) read acc 0.34-0.45 / macro 0.11-0.18 in the same region.
- **This is the loop's core artifact:** a stable, complete full-pass 0.8B run under the fixed pipeline (chunked bucketing, warmup 3 % + cosine, LoRA r=8, macro-first selection, a checkpoint per dev eval). Post-training the CLI is writing the temperature fit and the per-template dev report (weights loading at 2026-10-07T20:44:35Z); the test evaluation follows, then S10/S11.
- **Finalisation written (2026-10-07T20:59:46Z):** `temperature.json` -> **choice T=1.2190** (n=10,967, FITTED), **noul T=2.8959** (n=5,454, FITTED), **score `NOT FITTED - too few dev items (n=33 < 50)`** with the value beside it labelled a diagnostic on that small sample (as required - the qtype has no training items either). Note the contrast with run 2's fits (choice 4.27 / noul 11.61): run 3 needs **far less sharpening**, consistent with its much lower Brier. Also written: `dev_final.json`, `run.json`. `best.json` -> step **1000**, rule: highest dev macro accuracy on the fixed dev evaluation sample; ties (|dmacro| <= 1e-12) broken by lower dev Brier, then .
- **Test evaluation running** (`run_student.py --checkpoint outputs/student_v0/S9_run3/...`): the tier-1/fresh cells with coverage and gate status land next, then S10.
- **Selection-metric outcome, recorded not worked around (2026-10-07T20:59:56Z):** `best.json` selects **step 1000** again - the ADVISORY's macro-first rule picks an early snapshot because macro accuracy on the balanced dev sample peaks early (run 2's step-1000 macro was 0.7006, higher than anything later). The rule is **not** changed (operator ruling, R8), and the official test evaluation runs on `best/` = step 1000. Because run 3 saved **a checkpoint at every dev eval**, I have asked the agent to additionally evaluate the **converged** checkpoint (~step 41,000) on **identical item sets** and report it as an explicitly labelled **additional** analysis (R4) - never substituted for the official model. It will also quote the step-1000 vs best-late dev rows so the report shows *why* the rule chose what it chose.
- **Attribution caveat still stands:** which single change (schedule vs rank) was decisive is **not measured** - the diag ran arms one at a time on a subset that understates length pressure. S14 states it that way.

### S9 run 3 — OFFICIAL TEST CELLS (selected step 1000) — 2026-10-07T21:18:10Z
- **Run facts:** full pass, `steps=44152 stopped_early=None wall=13867s items=212481 items/s=22.64 tokens/s=21478`, GPU peak 37.98 GB, **88 dev evals / 88 checkpoints**. Selection rule verbatim: *"highest dev macro accuracy on the fixed dev evaluation sample; ties (|dmacro| <= 1e-12) broken by lower dev Brier, then by the earlier step"*. Full-dev (16,454 items, calibrated): **acc 0.7497 / macro 0.6653 / Brier 0.3215 / NLL 0.5597**.
- **Why step 1000, in two rows (measured):** step 1000 n=2019 acc 0.742942 / **macro 0.679832** / brier 0.349875; step 39,500 n=2019 acc **0.754829** / macro 0.665220 / brier **0.317572**. The rule picks step 1000 because 0.6798 is the **maximum macro of all 88 evals**; the converged checkpoints win on micro accuracy (+0.8 pt) and Brier (-0.032) but sit below on macro. **No rule change, no substitution.**
- **Tier 1: 13,521 items, coverage 1.00, 7/8 gates PASS** - medmcqa 0.3619, medqa 0.3928, medquad_routing **0.9978**, mmlu 0.5095, nfcorpus_relevant_noul 0.6507, pubmedqa 0.6373 (macro 0.4849), scifact 0.6433; the single failure is `nfcorpus_graded_score_v2` = **`READOUT_FAIL - constant_answer`** (so no accuracy is presented for it).
- **Fresh: 23,768 items, coverage 1.00, 11/11 gates PASS, 0 additional-check failures.** Held-out (D14): ct_arm_role 0.6165, **ct_phase 0.2555**, fda_boxed_warning 0.6215, pubmed_humans 0.7242. Seen: ct_claim_set 0.9417, ct_randomised 0.8500, ct_healthy 0.6540, fda_class_v2 0.9734, fda_route_claim 0.9595, pubmed_mesh_v2 0.9685, pubmed_observational 0.8786.
- **Looks-too-good flags for S11/S14 (not gains yet):** ct_claim_set 0.9417 vs zero-shot 0.2517, fda_class_v2 0.9734 vs 0.8106, pubmed_mesh_v2 0.9685, **medquad_routing 0.9978** vs family 0.9346-0.9894. The agent's explanation (readout effect + same-template training, with the four held-out cells as the leakage control) must be **re-derived in the run-3 audit**, not asserted - and note the held-out cells are *not* uniformly improved (ct_phase 0.2555 is below its 0.3455 zero-shot), which is what a leakage-free result looks like.
- **Instability detail recorded:** one **196,688** pre-clip grad spike at step 16,995 (a single 7,794-token `fda_route_claim_noul_v1` item, loss 15.96); clipping bounded it and the run did **not** diverge (run 2's failure was a *sustained* rise: p50 10.5 -> 40.8 after 18k; run 3's p50 fell to ~2.6). The length-driven tail is still real, just survivable under this recipe.
- **Next:** the converged checkpoint (step 44,000) evaluation as a labelled additional analysis (running, `--tag converged`, `ADDITIONAL_ANALYSIS_note.json`), then **S10** Base ablation (Qwen3.5-0.8B-Base, same recipe, `--save-every-eval`, same dev sample, 4 h box).

### S9 run 3 — ADDITIONAL ANALYSIS: the rule's pick BEATS the converged checkpoint, and my earlier framing was wrong — 2026-10-07T21:38:28Z
- **Official (rule's pick, step 1000) vs converged (step 44,000), identical item sets, coverage 1.00, same filters/metrics:** tier 1 **0.6505 vs 0.6141** (gates 7/8 vs 5/8, converged has 3 `READOUT_FAIL`); fresh **0.8227 vs 0.7767** (gates **11/11 vs 7/11**, converged has 3 constant-answer + 1 flip failure). The additional run is labelled in `S9_run3/ADDITIONAL_ANALYSIS_note.json`; the official model remains the rule's selection.
- **What the converged model actually does:** it is *better* on the trained structured-gold templates (ct_claim_set 0.9417 -> 0.9995, fda_route_claim 0.9595 -> 0.9975, scifact 0.6433 -> 0.9167, ct_healthy 0.6540 -> 0.8340, fda_class_v2 0.9734 -> 0.9898) and *much worse* on knowledge and held-out cells (medqa 0.3928 -> 0.2922, mmlu 0.5095 -> 0.3046, pubmedqa 0.6373 -> 0.5430, **fda_boxed_warning 0.6215 -> 0.2068 (below chance, held out)**, **pubmed_humans 0.7242 -> 0.4673 (held out)**, ct_arm_role 0.6165 -> 0.5020, pubmed_mesh_v2 0.9685 -> 0.8558). It is the more **training-mix-specialised** model; its dev accuracy/Brier win comes from the mix's frequent templates.
- **Correction to my own earlier reasoning (important):** I twice framed the macro-first rule as "selecting an undertrained snapshot" and treated the converged model as the likely-better one. **The test battery contradicts that**: the rule's pick is better on *both* tiers and passes 4 more gates on fresh. The operator's ruling ("the selection rule is not the problem") is **confirmed on evidence**, and the ADVISORY's macro-first rule is doing exactly what it is for - finding the better-*balanced* model rather than the more mix-specialised one. My framing is superseded here, not quietly dropped.
- **S10 (Base ablation) LAUNCHED 21:33:32Z:** `train_student.py --base-model Qwen/Qwen3.5-0.8B-Base --model-id meddecide-0.8b-base-lora-pointer --max-seconds 14798 --eval-every 500 --save-every-eval --lora-rank 8 --out outputs/student_v0/S10`, **pid 139222**, deadline **01:40Z**, checkpoints per dev eval, monitor writes `logs/poll.log`, and `run_student.py --split test --tag test` runs automatically on `best/` when training exits. Handover: `S10/RUN_NOTES.md`. New `--base-model` CLI flag; ruff clean, 279 non-GPU tests pass.

### S10 (Base ablation) — running — 2026-10-07T22:08:50Z
- **Handover (2026-10-07T22:19:08Z), S9/S10 agent's context exhausted - all state is on disk.** S10 at 22:03Z: step 4,671 (11 %), 21.6k tokens/s, loss (last 500) **0.663 falling**, grad p50 **4.57** / p95 38.4 / max 1,357 (one step > 1000) - the run-3 stability regime. **9 evals, dev macro best 0.7109 at step 2000** (instruct run 3's peak was 0.6798 at step 1000) - the agent flagged, correctly, that this may mean the Base ablation beats the instruct model on the balanced dev sample; it is also the same early-peak shape, so the converged checkpoint will likely differ again. **Nothing claimed from a flag.** Automatic from here: at the 01:40Z deadline the CLI writes `run.json`/`temperature.json`/`dev_final.json`, then the monitor `/workspace/tmp/s10_monitor.sh` (log `S10/logs/poll.log`) sees the exit and runs `run_student.py --split test --checkpoint outputs/student_v0/S10/best --tag test` (~17 min) -> artifacts expected by ~02:10Z.
- **A fresh agent now owns**: S10's wrap-up (confirm completed-vs-budget from `run.json:training.stopped_early`, quote the rule's selected step and the dev rows for it vs the late evals, report tier-1/fresh cells with coverage + gate status, run the labelled `--tag converged` comparison on the last checkpoint - the same side-by-side that mattered for run 3 - and write `S10/SELF_AUDIT.md`), **then the missing S11 input: the full-v0.2 zero-shot Qwen3.5-0.8B run** (command in the S11 plan above), one GPU job at a time. It must **not** run `g1.py` itself.
- **S10 progress (2026-10-07T22:59:29Z):** step **15,000** (34 %), alive; dev eval acc 0.6924 / macro 0.6206 / brier 0.3827. Run 3 (instruct) at the same step was acc ~0.71-0.73 / brier ~0.33-0.36, so the Base ablation now sits **below** the instruct run - consistent with the handover flag that its dev-macro advantage was an early peak (0.7109 at step 2,000) rather than a sustained lead. Not a claim: the run has 66 % to go and the rule selects on the whole trajectory.
- **S10 progress (2026-10-07T23:39:40Z):** step **23,000** (52 %), alive; dev eval acc 0.6756 / macro 0.4173 / brier 0.3886. Run 3 (instruct) at the same step was acc ~0.73 / macro ~0.6 / brier ~0.33, so the ablation is now clearly behind on all three - the expected direction for a Base-vs-instruct comparison. It is **not** diverging (no collapse, unlike run 2 at this step), so the fixed recipe is holding on the Base model too.
- **S10 progress (2026-10-08T00:19:51Z):** step **30,500** (69 %), alive, and it **recovered from the mid-run dip** - step 30000 acc 0.7197 / macro 0.5911 / brier 0.3529, step 30500 acc 0.7211 / macro 0.5785 / brier 0.3503 (was acc 0.6756 / brier 0.3886 at step 23,000). Same non-monotone shape as run 3, and now close to the instruct run again (run 3 at 30k: acc ~0.73 / brier ~0.32-0.33). The Base-vs-instruct gap is therefore **step-dependent**, which is exactly why the final comparison has to be on test cells at each run's rule-selected checkpoint rather than on mid-run dev readings.
`--base-model Qwen/Qwen3.5-0.8B-Base`, pid 139222, deadline 01:40Z, checkpoints per dev eval. Early dev evals on the **same 2,019-item sample**: step 5000 acc 0.6929 / macro 0.6225 / brier 0.3730, step 5500 acc 0.7207 / macro 0.5754 / brier 0.3552. Run 3 (instruct) at the same steps read acc 0.7231-0.7276 / macro 0.6545-0.6632 / brier 0.3377-0.3482 - so the Base ablation tracks slightly below at this point, which is the expected shape for an ablation that isolates the recipe from the instruct tuning. No intervention; its test cells land automatically on exit.

### S11 — execution plan (what the real G1 run still needs) — 2026-10-07T21:38:40Z
- **Inputs that exist:** MedDecide official = `outputs/student_v0/S9_run3/preds_meddecide-0p8b-lora-pointer__test.jsonl` (run `meddecide-0p8b-lora-pointer:2026-10-07T20:58:27Z`, 13,521 tier-1 + 23,768 fresh, coverage 1.00); Base ablation = S10's preds when it lands (`outputs/student_v0/S10/`, training now, deadline 01:40Z, its own auto test eval).
- **Input that does NOT exist: a full-v0.2 zero-shot Qwen3.5-0.8B run.** S1/S3 hold only **per-template** logs (`outputs/student_v0/S1/preds_qwen3p5-0p8b__<template>.jsonl`, `S3/...`), so G1 cannot pair them against a model scored over the whole test set. **One GPU job is needed** after S10 exits: `uv run python scripts/bench/run_baselines_v0_1.py --model Qwen/Qwen3.5-0.8B --tier1 data/bench/v0.2/tier1 --fresh data/bench/v0.2/fresh --keep-screen data/bench/v0.2/screen.json --out outputs/student_v0/S11 --run-id s11_zeroshot0p8b` (same filters as the trained-model runs so the item sets match).
- **JEV-9B on v0.2 is `NOT MEASURED`** unless commissioned: F7 measured it on **v0.1**, not v0.2. The ADVISORY's G1 list says "JEV-9B **where measured**", so G1 may legitimately run without it - but the report must say so explicitly rather than implying the comparison exists. If time permits, the same `run_baselines_v0_1.py` invocation with `--model` pointing at the JEV-9B checkpoint would fill it in (one more GPU job, ~30-40 min).
- **Then:** `uv run python scripts/bench/g1.py --models meddecide=<S9 preds> --models base=<S10 preds> --models zeroshot=<S11 preds> [--models jev9b=...] --tier1 data/bench/v0.2/tier1 --seed 0 --resamples 1000 --out loops/student_v0/g1.md --json outputs/student_v0/S11/g1.json` (23 tests already pass; the strict slice reads `data/bench/v0.2/fresh/manifest.json`, the D14 held-out list is fixed, D12-failing cells are excluded from the verdict).
- **Then S12** (byte-identity + audit read-back) and **S14** (findings, CLAIMS roll-up, `Loop status: STOPPED`).

### S10 (Base ablation) — COMPLETED THE FULL PASS — 2026-10-08T01:40:34Z
- **`training done: steps=44152 epochs=1 stopped_early=None wall=13760s items=212481 items/s=22.80`** - the **entire epoch**, no budget cut, in 3 h 49 m of a 4 h box, at 22.80 items/s (run 3: 44,152 steps / 13,867 s / 22.64 items/s - the two runs are a **matched pair**: same recipe, same data, same dev sample, same step count, one difference = instruct vs Base).
- Artifacts on disk: `best/`, `best.json` (**step 17000**, rule: highest dev macro accuracy on the fixed dev evaluation sample; ties (|dmacro| <= 1e-12) broken by lower dev Br), `dev_final.json`{, `temperature.json` fits {"choice": {"temperature": 4.138178516811879, "n_items": 10967, "status": "FITTED"}, "noul": {"temperature": 3.63202836193281, "n_items": 5454, "status": "FITTED"}, "score": {"temperature": 17.0324787}, `run.json`, and `preds_medecide-0p8b-lora-pointer__test.jsonl` (the monitor's automatic test evaluation has run).
- **For S11:** MedDecide-instruct (S9 run 3) and Base-ablation (S10) predictions now both exist on the **same item sets**; the only missing G1 input is the full-v0.2 **zero-shot Qwen3.5-0.8B** run, which the fresh agent is queued to do next. JEV-9B on v0.2 stays `NOT MEASURED` (F7 measured v0.1).
- The fresh agent (`40cff25f`) owns the wrap-up: cells with gate status, the labelled `--tag converged` side-by-side, `SELF_AUDIT.md`, then the zero-shot job.
- **Status (2026-10-08T01:40:52Z):** S10's test evaluation is running (pid 143607; `preds_..._test.jsonl` is being written, the summary `model_...__test.json` lands when it finishes ~17 min). No cells are quoted until that file exists - predictions alone are not a result. Then the agent's wrap-up (cells + `--tag converged` side-by-side + `SELF_AUDIT.md`) and the **zero-shot 0.8B** job that G1 still needs.
- **S10 test cells (2026-10-08T01:56:21Z) - the ablation separates sharply on readout health:** rule-selected checkpoint (step 17000): tier 1 13,521 items **5/8 gates PASS (3 fail)**, fresh 23,768 items **6/11 gates PASS (5 fail)**. Run 3 (instruct, same recipe/data/steps) scored **7/8** and **11/11**. So the Base ablation is healthy on fewer than two-thirds of the fresh templates while the instruct model is healthy on all of them - i.e. **the instruct tuning is doing real work for the pointer-head readout**, not just for accuracy. Per-template accuracies are suppressed in the failing cells (D12: no accuracy is presented for a cell that fails its readout check), so the honest headline is the gate counts, not a mean accuracy. The agent's wrap-up will give the per-template detail, the labelled converged comparison, and why each cell failed.

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|
| — | — | nothing is blocked | — |



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

### S7 — DONE — 2026-10-07T00:58:42Z
- What ran: the implementation agent's code + a 200-step smoke; I ran the test suite myself (`17 passed`) and read the two decisive tests
- Output: `src/meddecide/model/{head,markers,meddecide_model}.py`, `src/meddecide/train/{config,data,losses,smoke,temperature}.py`, `configs/student_v0.yaml`, `tests/test_model_pointer.py`, `outputs/student_v0/S7/{smoke.json,checkpoint/,SELF_AUDIT.md}`
- Headline: the MedDecide architecture is implemented and tested — frozen base + LoRA (11.6M trainable params) + per-option pointer head, CE + 1.0 Brier, per-qtype temperature; **21.7 items/s -> ~2.7 h per pass**, so S9 fits one pass (S061-S068).
- Surprises: (a) the agent's interim report **corrected my smoke numbers** — my 48.4 min/pass came from the short end of the length-sorted plan; the real figure is **2.7 h/pass**; (b) the raw per-step loss is non-monotone by construction, so the fixed-64-item reference set is the series to read; (c) the head **cannot learn with the brief's literal key-token state** (19/64) and ships `key_end` (64/64) — a recorded departure, flagged for the operator; (d) two latent bugs fixed that would have hit S9 (temperature fitting crashed on mixed option counts; the marker locator matched "C. " inside "C. difficile colitis" and scored the wrong token); (e) the `noul` temperature hit its search bound (0.039) and `score` has no training items at all.
- Next: S8 (evaluation path for the trained model) is unblocked.

