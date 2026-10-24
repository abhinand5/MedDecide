# osler_v4 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/osler_v4/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (Y11), or when no PENDING task can proceed without the operator -->
Run started (UTC): `2026-10-21T18:26:40Z`
Last updated (UTC): `2026-10-24T05:58:28Z`
Iterations so far: `1`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| Y0 | Orientation, Nemotron serving, MIMIC guard | small | — | DONE | 2026-10-21T18:26:40Z | 2026-10-21T18:53:35Z |
| Y1 | MIMIC foundation (tables, sections, splits) | no | Y0 | DONE | 2026-10-21T18:54:18Z | 2026-10-21T19:44:00Z |
| Y2 | MIMIC structured-gold templates (+ Y2b late sources when present) | no | Y1 | DONE (mimic_page_category_v1 NOT MEASURED: present but not built before B2's decision (missed re-check)) | 2026-10-21T19:44:00Z | 2026-10-21T20:40:16Z |
| Y3 | Nemotron-crafted minimal pairs (`mimic_doc_support_v1`) | yes | Y0, Y1 | DONE (dev 250 and test 250 pairs; train 10,125 pairs of a 40,000 target: the 8 GPU-hour limit ended the build; cue AUC dev 0.5645, gate < 0.75 passes) | 2026-10-21T19:25:21Z | 2026-10-22T03:58Z |
| Y4 | Fast dev v4, CG v2, MIMIC audit suite v1, baselines | yes | Y2, Y3 | DONE (14 baseline runs complete and audited in a fresh process, C104; pplx-flash 0.624 on the frozen suite with every cell passing; MedDecider-4B has no set-level result under D12, STATE §6 question 6; full pytest with the GPU visible, C105) | 2026-10-21T21:26:55Z | 2026-10-22T12:12Z |
| Y5 | pplx 27B soft labels (only if D48 confirmed) | yes | Y0 | DONE (parity 40 of 40 against the single-request reference; dev diagnostics recorded; 112,128 anchor soft labels scored in the 5 h budget; overlap with benchmark rows 0) | 2026-10-21T21:33:00Z | 2026-10-22T09:20Z |
| Y6 | Stage A: A4, A5 + re-scored A0/A1/A3 + choice | yes | Y4, Y5 | DONE (the pre-registered choice is A0, the control; C118, C120; A4 qualifies by 0.0001 and trails A0 on CG v2; no readout cell fails for any arm; full pytest with the GPU visible, C122; loops/osler_v4/stage_a.md) | 2026-10-22T12:11:12Z | 2026-10-23T00:35Z |
| Y7 | Stage B: blocks B1–B4 under keep rule v2 | yes | Y6 | DONE (no block kept under keep rule v2; B1, B2 and B3 enter the final mix under operator ruling item 1; B4 not kept; fresh-process re-derivation 0 mismatches, C168–C172) | 2026-10-23T00:35Z | 2026-10-24T05:58:28Z |
| Y8 | Final Osler-4B (63,304 steps) | yes | Y7 | IN_PROGRESS (launched 2026-10-24T02:34:17Z; PID 1472063 chain, 1472069 driver; resumable, full checkpoints every 5,000 steps) | 2026-10-24T02:34:17Z | |
| Y9 | Evaluation, Gate O4, size-class scoreboard | yes | Y8 | PENDING | | |
| Y10 | Memorisation and PHI audit | small | Y8 | PENDING | | |
| Y11 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (order exceptions and the GPU sequence in ADVISORY §5).
Never run two GPU jobs at once (the Nemotron server counts). A `BLOCKED` task does not block unrelated work. No row may be
left `IN_PROGRESS` at closure.

---

## 2. Key values discovered during the run

**Operator confirmations** (filled by the operator before kickoff; the loop never edits these two lines):
- D48 (pplx 27B soft-label distillation on general items): `yes` (operator, 2026-10-21, session message during Y0)
- D46, D47, D49, D50 (MIMIC design, keep rule v2, Gate O4, final length): `yes` (operator, 2026-10-21, same message)

| key | value | source | task |
|---|---|---|---|
| suite counts (passed / skipped / deselected) | baseline 840 / 1 / 2 (exit 0); after Y0 860 / 1 / 2 (exit 0); ruff clean | `outputs/osler_v4/Y0/logs/pytest_final.log`; CLAIMS C001–C004 | Y0 |
| Nemotron serving: flags that worked, out tok/s, requests/h, peak memory | committed flags, no fallback; ready 6 min 43 s after launch; KV cache 382,713 tokens; A (600-token output, 64-way, 200 req): 1,509.4 out tok/s, 9,056.7 req/h; B (60-token): 20,751 req/h; GPU peak 96,971 MiB of 97,887 | `outputs/osler_v4/Y0/serve_check.json`; CLAIMS C006–C013 | Y0 |
| MIMIC: patients / admissions / notes per split; ICD-10 share; token p50/p95 | | `loops/osler_v4/mimic.md` | Y1 |
| MIMIC templates: train/dev/test items, NB screen, majority rate per template | 13 MIMIC templates and MedNLI built. MIMIC items train 507,374 / dev 6,257 / test 6,253; held-out 110 (test patients only). NB dev: all below 0.90 except mimic_rad_exam_v1 at 0.9000 (shortcut, excluded from the verdict). Cross-split leakage 0 after the state-text rule (lab 106 dev / 92 test, ED 2+6 and 1+3 removed). | `loops/osler_v4/mimic.md` §2; `outputs/osler_v4/Y2/counts*.json`; CLAIMS C031–C049 | Y2 |
| minimal pairs: pilot keep rate per mode, chosen mode, cue AUC, pairs built | pilot keep rate: off 0.2806 (55 of 196), on 0.1735 (34 of 196); no mode reached 40 %; chosen mode off (deviation, §7). Build (mode off): dev 250 of 1,016 attempts; test 250 of 1,023; train 10,125 of 34,515 (target 40,000 not reached). Pairs: dev 250, test 250, train 10,125 (21,250 rows; malformed 0; impure edits 0; over 30 % 0). Cue AUC dev 0.5645 (gate < 0.75). | `loops/osler_v4/crafted.md`; `outputs/osler_v4/Y3/audit.json`; CLAIMS C025–C030 and C066–C071 | Y3 |
| fast dev v4 partitions; CG v2 size and sources; MIMIC suite v1 templates and items | Fast dev v4: 17,532 items; its MIMIC partition 5,562 items (C082). CG v2: 8,307 items (C051). MIMIC suite v1 (frozen 09:33Z): 4,253 items in 9 templates (icd_support 400, icd_specificity 488, principal_dx 500, msdrg 500, drg_cc_level 498, ed_acuity 494, ed_disposition 497, lab_validation 376, med_given 500); excluded 4 (apr_severity, apr_mortality, pcs_support: readout gate failed, so the ceiling is not measured; rad_exam: Naive Bayes 0.90); held-out 110 (not in the suite); training overlap 0 (C083–C085). | `outputs/osler_v4/Y4/mimic_audit_suite_v1_manifest.json`; `outputs/osler_v4/Y4/ceiling_qwen35_4b_mimic_dev.json`; CLAIMS C051, C080–C085 | Y4 |
| pplx 27B: revision, licence, parity, anchor items scored, overlap check (must be 0) | revision cce0f3f80033e9d89d80831d5281fd72124aacce; licence apache-2.0 (C073). Parity (C072): reference_batched argmax 40 of 40 against the single-request reference (max abs diff 0.0353); the batched path is the scoring path; the fast path also 40 of 40 and is not used. Dev diagnostics (the teacher's own accuracy; never used for a choice): 0.8516 on 9,122 CG v2 and knowledge dev items (C074); no constant answer on choice items; noul predicts no on 78.5 % of its 195 items against 54.4 % gold (C075). Anchor items scored: 112,128 (the stable-order prefix of the pool), 0 errors (C076). Overlap check: 0 (C077). Distributions valid (C078). Teacher accuracy on the scored training items (diagnostic): choice 0.8366, noul 0.81, score 0.6677 (C079). | `outputs/osler_v4/Y5/parity.json`; CLAIMS C072–C079 | Y5 |
| Stage A: MR, knowledge, CG v2 per run; the chosen recipe | | `loops/osler_v4/stage_a.md` | Y6 |
| Stage B: kept blocks; own / MR / general / knowledge / CG v2 per block; MIMIC share | no block kept under keep rule v2 (B1–B4 NOT KEPT: C129, C139, C152, C162). B1, B2 and B3 enter the final mix under ruling item 1: own lower +0.1581, +0.0475, +0.1625 against +0.02; knowledge lower −0.0134, −0.0163, −0.0082 against −0.02. B4 not in the final mix (CG v2 lower −0.0272 against −0.010). MIMIC partition (additional): +0.1535, +0.0201, +0.0741, +0.0314 for B1–B4. MIMIC share of the mix 0.3157, 0.0238, 0.0857, 0.0000; realised 0.3135, 0.0242, 0.0866, 0.0000. Fresh-process re-derivation: 0 mismatches (C168–C172) | `loops/osler_v4/blocks.md`; `outputs/osler_v4/Y7/SELF_AUDIT.md` | Y7 |
| final: steps, selected step, CG v2 and MIMIC trend | | | Y8 |
| Gate O4 verdict (4B) | | `loops/osler_v4/gate_o4.md` | Y9 |
| audit: membership AUC; verbatim matches (model vs base) | | `loops/osler_v4/audit.md` | Y10 |
| Y4 fast dev v4: total, partitions, training-overlap removals | 17,532 items: fastdev_v2 9,590 (1,333 removed: 854 text and record, 479 text only), new_template_dev 1,980, mimic 5,562, mednli 400; 30 training files checked (every .jsonl under data/train except dev.jsonl) | `outputs/osler_v4/Y4/fastdev_v4_manifest.json`; CLAIMS C050–C051 | Y4 |
| Y4 CG v2: total, sources, removals | 8,307 items (≥ 6,000): cg_v3 items kept except near-duplicates of the panel (458) and training text (460 MedMCQA); HeadQA dev (en) 1,342; MedNLI dev rest 995; PubMedQA labelled 1,000; DDXPlus validation sample 1,000 | `outputs/osler_v4/Y4/cg_v2_manifest.json`; CLAIMS C051 | Y4 |
| Y4 verdict-set audit against training (text) | V10 rows 30,621: 0 text matches for MR test (19,294), MR held-out (1,518), the seven panel sets (4,353), MedQA (1,273) and MedMCQA (4,183). Record-level overlap for these sets is not checked (V10 rows carry no source id). | CLAIMS C052 | Y4 |
| Y5 anchor pool: rows, exclusions, eligible, written | 992,873 rows (C1 830,076; replay 162,797); 101,725 domain-template rows; 891,148 anchors; excluded for benchmark overlap 7,287; cannot be sent exactly 7,564; same text as an earlier row 6,136; eligible 870,161; written 400,000 (the cap). Teacher scores: not yet (GPU). | `outputs/osler_v4/Y5/anchor_pool_manifest.json`; CLAIMS C061 | Y5 |

---

## 3. Current task checklist

**Fill this in before starting any task**, by copying that task's concrete steps out of
the plan as unticked boxes. Tick each one the moment it is finished, not at the end.
This is the only record of progress *within* a task — without it, a compaction mid-task
leaves you unable to tell what you already did.

Clear this section and write the new task's checklist when you start the next task; the
completed checklist goes into the iteration-log entry.

```
Task:           Y8 — Final Osler-4B, 63,304 steps (GPU; resumable). IN_PROGRESS since 2026-10-24T02:34:17Z. Writes outputs/osler_v4/Y8/ and the Y8 rows of §1, §2 and CLAIMS.
Recipe:         A0 (STATE §7; loops/osler_v4/stage_a.md): control, KL 0, osler_v2 C1 base plus the entering MIMIC-family blocks (B1, B2, B3 by operator ruling item 1; B4 not kept). Ten evaluations at 6,330 … 63,304 (CG v2 and the MIMIC partition at each; selection by fast dev v4 template macro, unchanged).
Mix:            composed in memory (C164): 1,456,593 rows; composition SHA-256 9d25e30f…; family share 0.3500, replay share 0.2000, largest template 0.0480. Replay in process: 63,304 steps, 242,255 items realised (C166).
Running:        chain PID 1472063, driver PID 1472069 (`outputs/osler_v4/Y8/logs/y8_chain.sh`); log outputs/osler_v4/Y8/logs/final_train.log; events outputs/osler_v4/Y8/logs/train.jsonl; chain exit code in outputs/osler_v4/Y8/logs/y8_chain.log. Full checkpoints every 5,000 steps under outputs/osler_v4/Y8/full (newest two kept). After an interruption: `bash outputs/osler_v4/Y8/logs/y8_chain.sh --resume` (identical data order; STATE §7).
Progress:       2026-10-24T05:58Z: step 11,800 of 63,304 (last logged step); no non-finite logged values; full checkpoints 5,000 and 10,000 complete (C167). The evaluation at 6,330 is C173 (fast dev v4 13,306 of 17,532, template macro 0.7782; CG v2 and MIMIC partition recorded in partitions.jsonl at the same step). The pace has varied between 56 and 77 steps per minute; with 51,504 steps and nine evaluations of about 18 minutes left, the training end is expected between about 2026-10-24T19:45Z and 2026-10-25T00:00Z (the 56 to 77 steps-per-minute range, plus nine evaluations).
Working rule:   one GPU job at a time; the Y7 closure work is CPU only and does not touch the GPU.
Checklist:
- [x] final mix composed in memory and checked (C164; scripts/osler_v4/y8_compose_check.py)
- [x] replay in process (C166)
- [x] full suite with the GPU visible before launch (C165); launch 2026-10-24T02:34:17Z
- [x] first full checkpoints written and complete: 5,000 (C167), 10,000
- [x] evaluation at 6,330 recorded in CLAIMS (C173)
- [ ] training finished 63,304 steps: logged steps, non-finite check, evaluation steps on the grid, realised items 242,255 (validation as in Y7)
- [ ] selection by fast dev v4 template macro over the ten evaluations; scored on fast dev v4 and CG v2; readout gate; CG v2 and MIMIC trend at every evaluation
- [ ] Y8 DONE: §2 final row, §4 entry, SELF_AUDIT, CLAIMS rows, commit and push


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

### Y0 — DONE — 2026-10-21T18:53:35Z
- What ran: `uv run ruff check .` (pass); `uv run pytest` with the two ADVISORY deselects, before (baseline) and after the Y0
  changes; `scripts/osler_v4/serve_nemotron.sh` launched detached 18:39:03Z (vLLM PID 420612, EngineCore 421021), then
  `scripts/osler_v4/y0_serve_check.py`; server stopped by SIGINT (GPU 0 MiB at 18:48:23Z); `./dm status` read only.
- Output: `outputs/osler_v4/Y0/` (serve_check.json, SELF_AUDIT.md, logs/; gitignored).
- Headline: Nemotron 3.5 Super serves on this pod with the committed flags, no fallback (ready 6 min 43 s after launch; KV cache
  382,713 tokens). 200 requests, 2,897 prompt tokens on average, forced 600-token output, concurrency 64, reasoning off:
  1,509.4 output tokens/s and 9,056.7 requests/h, 0 failed (C006–C013). The MIMIC loopback guard is in the teacher client
  (C004); pytest 860 passed, 1 skipped, 2 deselected after the changes (C002; baseline 840).
- Surprises: (1) the advisor's 1,234 tok/s and about 8,100 requests/h were exceeded; not investigated, and no decision depends
  on it. (2) GPU peak 96,971 MiB of 97,887: the server fills the card, so nothing else can run beside it. (3) Two pytest runs
  printed no summary: `-q` with the pyproject `addopts = "-q"` gives `-qq`; the reruns used the configured level. (4) The
  operator's data-update steer and the D48/D46/D47/D49/D50 confirmations came in mid-run (§7, §2).
- Checklist (completed): ruff and pytest counts; server up; three short prompts (reasoning off) and one reasoning-on prompt;
  throughput A and B; MIMIC guard, its tests and the OPENAI_ grep; `./dm status`; acceptance; SELF_AUDIT; CLAIMS C001–C014;
  STATE; commit.
- Next: Y1 (CPU, MIMIC foundation). Y2 (CPU, the templates, including the operator-steered ED and MedNLI templates) runs while
  Y3's teacher works. Y3 needs the server restarted. D48 is yes, so Y5 (27B) is no longer blocked; it follows Y3 in the GPU order.

---

### Y4 — preparation (CPU) — 2026-10-21T21:26:55Z
- What ran: `scripts/osler_v4/y4_fastdev.py` (fast dev v4 with the training check); `scripts/osler_v4/y4_cg_v2.py` (CG v2 with the training check and the 13-gram panel check); `scripts/osler_v4/y4_mimic_suite.py` (written; not run: it needs the GPU ceiling file and refuses to freeze without it); `src/meddecide/panel/ngram.py`, `src/meddecide/panel/training_overlap.py` with 7 tests.
- Output: `data/bench/osler_v4/fastdev_v4.jsonl` and `cg_v2.jsonl` (private, gitignored); `outputs/osler_v4/Y4/*_manifest.json` (counts only).
- Headline: fast dev v4 17,532 items; CG v2 8,307 items (≥ 6,000); no V10 evaluation item (MR test, held-out, panel, MedQA, MedMCQA) occurs as text in any training file (CLAIMS C050–C052).
- Surprises: earlier fast dev v2 items (ct, pubmed, fda, criterion, medmcqa) are in the Osler training mixes; the dev proxies are contaminated, the verdict sets are not by text (see §6 question 4). A first training check counted osler_v0's dev file as training; corrected.
- Next: when Y3 is DONE, Y4's GPU steps: the zero-shot ceiling screen on the MIMIC dev items, the suite freeze, then the baselines.

### Y2 — DONE — 2026-10-21T20:40:16Z
- What ran: `scripts/osler_v4/y2_templates.py` (stage 1: 12 templates including ED acuity, ED disposition and MedNLI; reruns 20:23–20:37Z) and `scripts/osler_v4/y2_lab_med.py` (stage 2: labevents, eMAR and prescriptions; 20:23–20:35Z); `scripts/osler_v4/y2_report.py` (mimic.md §2, leakage columns; 20:39Z); tests: 71 passed on CPU (CUDA hidden).
- Output: `data/mimic_build/items/` (private JSONL, gitignored); `outputs/osler_v4/Y2/counts.json`, `counts_lab_med.json`, `SELF_AUDIT.md`; `loops/osler_v4/mimic.md` §2.
- Headline: 13 MIMIC templates plus MedNLI. MIMIC items: train 507,374, dev 6,257, test 6,253; held-out 110. Dev Naive Bayes is below 0.90 for every template except mimic_rad_exam_v1 (0.9000, a shortcut: excluded from the verdict). Cross-split leakage is 0 after the state-text rule (CLAIMS C031–C049).
- Surprises: the first stage-1 and stage-2 runs had cross-split duplicate lab pages (96 and 82) and ED records; the rule removes them (after the screens were re-run). The lab dev majority rate moved from 0.50 to 0.60 because the removed duplicates were mostly one class. A first lab run used all Sodium and Glucose rows regardless of fluid; it was restarted with blood only (two urine glucose and one other-fluid potassium were excluded).
- Checklist (completed): schemas and drivers; stage-1 builders and tests; stage-2 builders and tests; labevents and eMAR runs; dedupe rule and reruns; report with leakage columns; SELF_AUDIT; CLAIMS C031–C049; STATE; commit.
- Next: Y3 build (GPU) is running; Y4 needs Y2 and Y3 done (Y4 starts after the build and the cue check).

### Y1 — DONE — 2026-10-21T19:44:00Z
- What ran: `scripts/osler_v4/y1_mimic_base.py` on the official files (CPU; run 2 from 19:10:55Z to 19:20:33Z; the first run's log is kept as y1_mimic_base_run1.log); `tests/test_mimic_sections_v4.py` (12); full-suite acceptance with the GPU free (C024).
- Output: `data/mimic_build/` (private parquet, gitignored); `outputs/osler_v4/Y1/aggregates.json`; `loops/osler_v4/mimic.md` (generated; aggregates only).
- Headline: MIMIC-IV 3.1 hosp holds 364,627 patients and 546,028 admissions (train/dev/test admissions 436,063 / 55,580 / 54,385; 203,429 late-anchor). 331,793 discharge notes, 331,732 linked to an admission; token p50 3,046 and p95 5,631 with the pplx tokenizer; ICD-10 share 0.4663 of coded admissions (C015–C023).
- Surprises: the first run's split bookkeeping was wrong (fixed; see §7); 208,046 radiology notes are for subjects with no hospital admission; 1,749 discharge notes exceed the 8,192-token cap (cut by section priority in Y2 and Y3).
- Checklist (completed): schemas and tokenizer; `sections` with unit tests; admission table; private parquet; script run; `mimic.md` §1; acceptance; SELF_AUDIT; CLAIMS C015–C024; STATE; commit.
- Next: Y3 build (GPU) is running; Y2 templates (CPU) are next, including the ED and MedNLI templates from the operator steer.

---

### Y3 — DONE — 2026-10-22T03:58Z
- What ran: `scripts/osler_v4/y3_pairs.py build --mode off --hours 8` under `outputs/osler_v4/Y3/logs/build_chain.sh` (PIDs: chain 509602, Nemotron server 509604, build 511361; build finished 03:53:08Z, chain done 03:53:39Z); `y3_pairs.py cue --build build_off` (CPU); `y3_pairs.py report --mode off`; `scripts/osler_v4/y3_audit.py` (fresh process); the full suite with the card visible (C070) and `tests/test_pplx_parity.py` with torchvision for that run (C071).
- Output: data/mimic_build/pairs/build_off/rows_{dev,test,train}.jsonl (private); outputs/osler_v4/Y3/{audit.json,SELF_AUDIT.md,logs/}; loops/osler_v4/crafted.md.
- Headline: mode-off Nemotron pairs: dev 250 and test 250 complete; train 10,125 pairs from 34,515 attempts, short of its 40,000 target because the 8 GPU-hour limit ended the build (C066). The fresh-process audit: every edit is a pure deletion, no pair selects over 30 %, every kept pair sits in its stage's split, and the test pairs are all late-anchor (C067). The Naive Bayes cue AUC on dev is 0.5645 (gate < 0.75; C068).
- Surprises: the train target was not reached; the largest rejection is the string rule (11,553 of 34,515 train attempts); the reasoning-on pilot's 76 invalid lists are unexplained (logged only); no mode reached the pre-registered 40 % keep rate (a recorded deviation).
- Checklist (completed): build; cue; report; audit; full-suite acceptance (C070) and the pplx parity test (C071); SELF_AUDIT; CLAIMS C066–C071; STATE.
- Next: Y5 (27B soft labels) on the GPU (parity running); then Y4 (ceiling screen, suite freeze, baselines).

---

### Y5 — DONE — 2026-10-22T09:20Z
- What ran: `scripts/osler_v4/y5_soft_labels.py` parity (GPU), dev, anchor (5 h budget, GPU), join and report, in `outputs/osler_v4/Y5/logs/y5_chain.sh` (PID 783889; the chain ended 09:15:24Z); `scripts/osler_v4/y5_duplicate_check.py` (CPU; the item-id census); `scripts/osler_v4/y5_audit.py` (fresh process).
- Output: data/train/osler_v4/y5_anchor_soft.jsonl (112,128 rows, private); outputs/osler_v4/Y5/{parity.json, dev_accuracy.json, anchor_probs.jsonl, join_manifest.json, audit.json, SELF_AUDIT.md}; loops/osler_v4/teacher27b.md (aggregates).
- Headline: the pplx-decider-v1.2-27b teacher (Apache-2.0, pinned cce0f3f8…) agrees with the authors' single-request reference on 40 of 40 argmax decisions, so its batched path is used (C072). Its distributions are valid, the scored items are the stable-order prefix of the pool (112,128 of 870,161 eligible), and they overlap no benchmark row (C076, C077). Its accuracy on the dev diagnostics is 0.8516 over 9,122 items (C074).
- Surprises: about 35–39 % of choice and noul targets are above 0.99 (C078); the dev noul set (195) is predicted no on 78.5 % (C075), the training pool is balanced (C079); the knowledge dev accuracy (0.86) may reflect the teacher's own training exposure (unverifiable, diagnostic only). The first pool build dropped 22,106 Winogrande twins by item id; corrected before any teacher output (CLAIMS C057–C061).
- Checklist (completed): pool; parity; dev; anchor; join (overlap 0); report; duplicate check; audit; SELF_AUDIT; CLAIMS C057–C079; STATE.
- Next: Y4 (zero-shot ceiling screen running, PID 809816); then the suite freeze and the baselines; then Y6.

### Y4 — DONE — 2026-10-22T12:12Z
- What ran: `scripts/osler_v4/y4_mimic_suite.py` (the freeze, 09:33Z); `y4_pplx_flash.py`, `y4_meddecider.py` (MedDecider-4B and -9B) and `y4_checkpoint_score.py` (osler_v2) in `outputs/osler_v4/Y4/logs/y4_chain_{baselines,suite,rest}.sh`, one GPU job at a time; `scripts/osler_v4/y4_baselines_report.py`; `scripts/osler_v4/y4_rederive.py` (fresh process, 14 runs); `ruff check .` and the full pytest with the GPU visible (C105).
- Output: `outputs/osler_v4/Y4/` (baselines_report.json, baselines/<run>/, SELF_AUDIT.md, rederive_final.txt, mimic_audit_suite_v1_manifest.json; gitignored); `loops/osler_v4/baselines.md` (committed aggregates).
- Headline: on the frozen MIMIC audit suite (4,253 items, 9 templates; C083) zero-shot pplx-flash scores micro 0.624 with every cell passing (C087); osler_v2 0.5551, below its base on 8 of 9 templates (C090); MedDecider-9B 0.6189 (C089, extra). MedDecider-4B has no set-level result under D12: its drg_cc_level cell is a constant answer (C088, C093). On CG v2, pplx-flash's set-level figure fails the gate on one cell (C101); osler_v2 scores 0.7565 over 13 templates (C102). On the MIMIC partition of fast dev v4 (a development proxy): pplx-flash 0.6638, osler_v2 0.5951, MedDecider-9B 0.6352 (C098–C100, C103).
- Surprises: (1) my first report's macro and set-level status were wrong; corrected before any gate used them (STATE §7; six claim rows corrected in place). (2) The trainer's keys for options beyond 26 are base-26 (BA for 26); the audit checks every row (STATE §7). (3) ed_acuity is below the majority rate for every baseline; the suite has no level-5 gold while the models answer 5 on 22–96 items (C094). (4) rad_exam (Naive Bayes at the 0.90 limit) scores 0.94–0.97 on the partition for every model (C098–C100, C103): flagged for the Y7 decision.
- Checklist (completed): CPU preparation (fast dev v4, CG v2, the MIMIC partition, the zero-shot ceiling screen, the APR check, the suite freeze); the baselines (C087–C103); the report and the fresh-process audit (C104); parity (C092, C103); pytest (C105); SELF_AUDIT; STATE.
- Next: Y6 (Stage A) on the GPU, started 2026-10-22T12:11:12Z: A0, A1 and A3 re-scored, then A4 and A5 trained. The choice needs the D12 gating recorded as a deviation first (STATE §6, question 7).

### Y6 — DONE — 2026-10-23T00:35Z
- What ran: `scripts/osler_v4/y6_stage_a.py` (score A0, A1, A3, A4, A5; train A4 and A5; choose) in `outputs/osler_v4/Y6/logs/y6_chain.sh` (PID 873094; one GPU job at a time; ended 00:25:10Z), with `y6_cpu_prep.sh` (the A5 mix and replay, CPU); `y6_readout_gate.py` (the gate per arm); `y6_reproduce_a0.py` (osler_v3's decisions, A0, A1, A3); `y6_batch_check.py` (the GPU batch-dependence check, after the chain); `y6_rederive.py` (fresh process: choice inputs and probability checks); `meddecide.eval.readout_gate` (6 tests) and `meddecide.train.stage_a_choice` (the restriction, 4 tests); the full pytest with the GPU visible (C122).
- Output: `outputs/osler_v4/Y6/` (A0 to A5 directories, choice.json, batch_check_A3.json, a5_mix_manifest.json, replays/, SELF_AUDIT.md; gitignored); `loops/osler_v4/stage_a.md` (committed).
- Headline: the pre-registered choice is A0, the control (C118). A4 (KL 0.1) qualifies with a 0.0001 margin on its MR lower bound and trails A0 on CG v2 by 0.0043; A1, A3 and A5 fail a lower bound. On fast dev v4 the arms score A0 0.7675, A4 0.7805, A5 0.7743; on CG v2 A0 0.7879, A4 0.7836, A5 0.7791 (C106, C115, C117). The distillation arm A5 is below A4 on both partitions and on CG v2. No readout cell fails for any arm (C107, C111, C113, C115, C117).
- Surprises: (1) bfloat16 scores depend on batch composition: one probability moves by 0.22 and a decision changes; float32 moves by at most 0.001 on the same items (C119). The loop's scores are bfloat16; Y8 and Y9 must decide the precision. (2) The in-training and full-dev accuracies of the same checkpoint differ by two items (A4) and ten (A5); the cause is not checked (C114, C116). (3) A3's re-score differs from osler_v3's record on three CG items, consistent with the batch effect (C112, C119). (4) The CPU attempt at the batch check failed (a CUDA-only kernel); it ran on the GPU.
- Checklist (completed): the osler_v3 re-scores (C106, C110, C112) and their gates (C107, C111, C113); the A4 and A5 trainings (C114, C116) and scores (C115, C117); the A5 mix and replay (C108); the choice (C118) and its fresh-process audit (C120); the probability checks (C121); the batch check (C119); stage_a.md; SELF_AUDIT; the closure pytest (C122).
- Next: Y7 (Stage B) on the chosen recipe A0: B0 is A0 (no new run); B1 (the Y2 MIMIC templates), B2 (the Y3 pairs), B3 (the Y2b late sources; NOT MEASURED for MIMIC-III) and B4 (osler_v3's X1 real sources), each judged under keep rule v2.

### Y7 — DONE — 2026-10-24T05:58:28Z
- What ran: `scripts/osler_v4/y7_build_block.py` (the B1 to B4 mixes; C123 and the B2 to B4 mix claims); `scripts/osler_v4/y7_replay.py` (the replays); `scripts/osler_v4/y7_block.py train|score` (chains in `outputs/osler_v4/Y7/logs/y7_b*_chain*.sh`; one GPU job at a time); `scripts/osler_v4/y7_x1_screen.py` (the B4 pre-check); `scripts/osler_v4/y7_readout_gate.py`; `scripts/osler_v4/y7_keep.py` (`keep_B1` to `keep_B4.json`); `scripts/osler_v4/y7_rederive.py` (the fresh-process re-derivation, CPU); tests C163 and C174 (995 passed with the GPU hidden).
- Output: `outputs/osler_v4/Y7/` (`keep_B1` to `keep_B4.json`, `rederive.json`, `SELF_AUDIT.md`, the arm directories and logs; gitignored); `loops/osler_v4/blocks.md` (committed).
- Headline: under keep rule v2 no block is kept. B1 fails MR, general and CG v2; B2 fails MR and CG v2; B3 fails MR and CG v2; B4 fails CG v2 (C129, C139, C152, C162). Under operator ruling item 1, B1, B2 and B3 enter the final mix (own lower bounds +0.1581, +0.0475, +0.1625 against +0.02; knowledge lower bounds −0.0134, −0.0163, −0.0082 against −0.02); B4 does not. A fresh-process re-derivation reproduces every keep statistic, count, verdict and replay draw with 0 mismatches (C168–C171).
- Surprises: (1) B4 was scored at its selected checkpoint (step 10,494), not the final step; the working notes' "post-training" label was wrong (C160, C170). (2) Scored and in-training counts on the same checkpoint differ by 2 to 7 items (C170). (3) B3's readout cell `ragtruth_sentence_supported_v1` is a constant answer and is excluded from its comparisons (C150, C151). (4) B3's first attempt stopped silently after step 10,450 (C143); B4's first attempt stopped at the disk-quota incident (C154). (5) B2's near-zero loss at step 1,650 was described as exactly zero; the logged values are 5.96e-07 and 4.6e-05, and the note is corrected (C172). (6) The realised MIMIC shares match the mix shares within 0.0022 (C171); B1 draws fewer items per step because its MIMIC items are long.
- Checklist (completed): the keep rule v2 margins as a library function (`src/meddecide/train/keep_v2.py`); the B1 to B4 mixes and replays; the four trainings and their validation; the four scorings and readout gates; the four keep decisions (C129, C139, C152, C162); `blocks.md`; the final-mix rule and its checks (C164, C166); resumable training and its tests (C155 to C157); the fresh-process re-derivation (C168 to C172); SELF_AUDIT; the closure tests (C174); CLAIMS; STATE.
- Next: Y8 (running; checklist in §3). Y9 (Gate O4) and Y10 (the audit) follow Y8.

---

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|
| Y2 mimic_page_category_v1 | MIMIC-III page-category template (B3 part) | NOT MEASURED — present but not built before B2's decision (missed re-check); MIMIC-III 1.4 is on disk (28 files; NOTEEVENTS.csv.gz modified 2026-10-22T10:02:24Z; C142) | none: no block is added for it (operator ruling 2026-10-23, item 4) |

---

## 6. Questions for the operator

Anything you could not resolve without a human. Be specific enough to answer without
re-reading the run: state the ambiguity, the options, and which you would pick.

1. **Y3 mode when no mode reaches the 40 % keep rate** (ADVISORY Y3 §7 is silent). Options: (a) reasoning off, the higher
   keep rate (0.2806) and 1,302.6 kept per GPU-hour, which is what the build uses; (b) reasoning on (0.1735; 178.1 per
   GPU-hour); (c) block Y3 and run B2 as NOT MEASURED. I picked (a) and recorded it under §7. Please confirm or redirect.
2. **Spot-reads of MIMIC pairs** (ADVISORY Y3 §7). The loop checks structure only (sentence counts, deleted numbers, the rule
   that dropped each attempt) and prints no note text, because this agent session runs on a hosted model. Options: (a) keep
   the structural checks; (b) you read 20 kept and 20 rejected pairs on the pod yourself and record the verdict in §6 or §7.
   I picked (a) until you say otherwise.
4. **Dev-proxy contamination in earlier loops** (found 2026-10-21 in Y4; counts in CLAIMS C053–C055). The fast dev v2 templates ct_* (clinical trials), pubmed_* (PubMed), fda_* (openFDA), criterion_eligibility and medmcqa_4opt occur as exact text and as source records in the Osler training mixes: osler_v0 train, osler_v1 blocks B0–B3, osler_v2 C0–C2, osler_v3 A3 and B1–B3, and in the student_v0 and student_v1 training files. 1,333 of the 10,923 fastdev_v2 items match training. cg_v3's knowledge items are in student training files (460 MedMCQA items match text). Consequences: the dev proxies used for osler_v2's base choice (D39, fast dev v2 MR partition) and osler_v3's Stage A (cg_v3 and fast dev v3) are not clean. The verdict sets are clean by text: V10 MR test, MR held-out, the external panel and the MedQA and MedMCQA items have no text match in any training file (record-level not checked). Options: (a) this loop removes matched items from its own proxies (done) and records the caveat; (b) earlier choices are re-run on clean proxies in a new loop. I pick (a). Option (b) is the operator's decision.
3. **Reasoning-on selections with no valid list** (76 of 200 attempts in the on pilot). Cause not investigated; likely the
   4,096-token cap is reached before the list (not verified). Not needed for the chosen mode; logged only.
5. **Item ids are not unique in some sources** (found 2026-10-21 in Y5; CLAIMS C059, C060). The `item_id` hash in
   `src/meddecide/mix/rows.py` leaves `state` out. In block C1, 22,108 anchor rows repeat an id. All of them are Winogrande
   twins of their first row (same source record), and 22,106 have different text. In the benchmark files no single file
   repeats an id, but 297 rows carry an id that appears in an earlier benchmark file with different text. Checked: the
   paired-comparison code (`train/blocks.py`, `pair_by_item`) raises on any duplicate id, so no paired comparison is silently
   mis-paired; the trainer uses `item_id` for an ordering seed and a length statistic only (`train/mix_items.py`,
   `train/data.py`). The training scan (CLAIMS C062) shows the same twins repeat in 12 Osler mixes: osler_v1 block_B0–B3 and
   replay_v1_train, osler_v2 block_C0–C2, osler_v3 block_A3 and block_B1–B3, 22,108 repeats in each file (22,106 with different
   text). The other 18 training files have none, and no RAGTruth item repeats an id. Both twins of every pair are in the files,
   so no training row is lost by id in the code paths found; a dict keyed by id is used only for the pairing check (which
   raises) and the statistics. Options: (a) leave `rows.py` as it is for this loop and record it; (b) change the hash in a
   later loop, which changes every item id of those mixes and so every paired comparison built on ids. I pick (a).
   Operator's decision: whether the id scheme is fixed, and when.
6. **Gate O4's MedDecider-4B comparison under D12 (for Y9).** MedDecider-4B's `mimic_drg_cc_level_v1` cell fails the readout gate (a constant answer, A, on 0.952 of 498 items; CLAIMS C088, C093), so its suite-level accuracy is not a result, and D49 names MedDecider-4B as a Gate O4 baseline. Options: (a) the MedDecider-4B comparison is NOT MEASURED at the set level and Gate O4 is read against zero-shot pplx-flash alone (the D12 reading, with no new rule); (b) every arm and the baseline are compared on the eight templates that pass (a rule change the operator must approve, since the suite was frozen with nine); (c) the template is dropped from the suite (not allowed: the suite is frozen). I pick (a). **Answered by the operator (2026-10-22): (b)**, pre-registered before any final result; recorded in §7 (Y9 entry).
7. **The Stage A choice and the CG v2 figures under D12 (for Y6).** ADVISORY Y6 ranks the qualifying runs by their CG v2 accuracy. Under D12 a set-level accuracy is a result only when every cell in it passes, and two CG v2 figures may not: zero-shot pplx-flash's (its `nfcorpus_relevant_noul_v1` cell is a constant `no` on 0.908 of 87 items; C093) and any arm's own cells. Options: (a) the Y6 choice is BLOCKED until the operator states how CG v2 is scored under D12; (b) the ranking uses the CG v2 templates that pass the gate for every qualifying arm (the same items for each arm, labelled as such). The zero-shot offset cancels in the ranking, so the zero-shot figure is needed only for the report, where it stays NOT MEASURED at set level, with its passing-cell figure as an additional row; (c) the failing template stays in the ranking (not allowed by D12). I pick (b) and will record it as a deviation before the choice step. The Y6 training and re-scoring steps do not depend on it. **Answered by the operator (2026-10-22): (b)**; recorded in §7 (Y6 entry) before the choice step.

---

## 7. Deviations from the plan

Any place you departed from GOAL/ADVISORY, with the reason. An empty section is the
expected outcome. Editing code or a check to make it pass is never an acceptable
deviation — that is a `BLOCKED`.

- **operator steer 2026-10-21: data update** (session message during Y0; the operator's own words, in substance). The
  ADVISORY §1 and §5 statements on what is on disk are out of date. Checked with `ls data/physionet/` (and `./dm status`,
  read only):
  1. MIMIC-IV 3.1 `hosp` is complete, labevents included. `mimic_lab_validation_v1` is built in Y2 with the other
     templates. The ADVISORY §5 rule "labevents may be absent at Y2" no longer applies.
  2. MIMIC-IV-ED 2.2 (all 8 files) and MedNLI 1.0.0 (all 5 files) are complete. `mimic_ed_acuity_v1`,
     `mimic_ed_disposition_v1` and `mednli_v1` are built inside Y2 now, not deferred to Y2b. Their dev items enter fast dev
     v4 at Y4. The late-anchor test items of the two ED templates join MIMIC audit suite v1 under the same D35 screens.
     MedNLI test stays an additional row. These three form block B3.
  3. MIMIC-III 1.4 is still downloading. It is re-checked at Y4 and before the B2 keep decision (the operator's wording; the
     ADVISORY Y2b text says "B3's decision"). If present then, `mimic_page_category_v1` (MDACE admissions excluded) joins B3;
     otherwise B3 goes ahead without it. The loop never waits for it.
  4. `icu/` files (for example chartevents) are appearing on disk but stay out of scope (ADVISORY §6).
  5. **Re-check, 2026-10-21T22:09Z** (`./dm status`, read only). MIMIC-IV 3.1: 16 of 33 files (7.6 of 9.9 GB). The steer's
     "hosp is complete" does not match this: `hosp/pharmacy`, `hosp/omr` and `hosp/microbiologyevents` are still downloading
     (6 %, 86 % and 90 %). None of them is read by the Y1 or Y2 builders (checked: the tables read are admissions,
     diagnoses_icd, drgcodes, labevents, emar, the d_ dictionaries and the note files), and microbiology is out of scope
     (ADVISORY §6), so the built data is unaffected. MIMIC-III 1.4: 0 of 28 files, queued; the page-category template stays
     NOT MEASURED. The loop did not wait for either.
- **Y1 (2026-10-21):** the split key for every MIMIC table is `subject_split(subject_id, "mimic4")` (ADVISORY §3.3). The osler_v1
  notes reader used `"mimiciv"`; it was changed to `"mimic4"` because no MIMIC output had ever been made with the old key.
- **Y1 first run defects, fixed in the reported run** (log kept as `outputs/osler_v4/Y1/logs/y1_mimic_base_run1.log`): the
  "split mismatch" count mixed orphan notes with real mismatches, and radiology splits went through the admissions map (208,046
  notes had none). The reported run keys every split by subject and separates each excluded group with its own count.
- **Y3 mode choice (pre-registered rule silent):** ADVISORY Y3 §7 chooses the mode "with the higher kept pairs per GPU-hour
  among modes with keep rate ≥ 40 %". Neither mode reaches 40 % (reasoning off 0.2806, on 0.1735; `pilot_report.json`).
  Taken as the conservative reading: reasoning off, because it has the higher keep rate and 7.3× the kept pairs per GPU-hour
  (1,302.6 vs 178.1). Recorded as a deviation; the operator may decide otherwise (§6).
- **Y3 spot-reads replaced by structural checks:** ADVISORY Y3 §7 asks for 20 kept and 20 rejected pairs to be read. Pair text
  is MIMIC text and this agent session is hosted, so the spot-check uses sentence counts, the deleted sentence numbers and the
  rule that dropped each attempt, and no note text is printed into the session (§6, question 2).
- **Y3 pilot cue AUC must be grouped by pair:** the first pilot cue check (random 5-fold) gave 0.056 and 0.014, which is a leak
  (each pair's twin is in the other fold). The reported value groups folds by `pair_id` (0.5719 off, 0.5658 on).
- **Y4 preparation before Y3 is DONE** (2026-10-21): fast dev v4 and CG v2 were built while the Y3 build still runs. Their Y3 input is the dev pairs (250 pairs, final: the build's dev stage is complete). The GPU steps of Y4 wait for the build.
- **Training input definition** (Y4): every .jsonl under data/train/ except the dev.jsonl files. An early check reported 989 rows as training overlap for cg_v3 items; they are all in data/train/osler_v0/dev.jsonl (osler_v0's dev split, a dev input). The corrected definition is in src/meddecide/panel/training_overlap.py and is used by both Y4 builders and by the suite freeze.
- **Fast dev v4 training removals:** the 400 CodiEsp items that fast dev v3 labels as train (they occur in osler_v1's training block B3, a block osler_v1 did not keep but did train on) and 1,333 further fastdev_v2 items that match training text or record. Partitions: fastdev_v2 9,590, new_template_dev 1,980, mimic 5,562, mednli 400.
- **CG v2 composition** (Y4): cg_v3 (4,865) minus 458 panel near-duplicates, 1 duplicate and 460 MedMCQA items with training text; HeadQA dev (English; the English HEAD_EN.json restricted to dev_HEAD.json's qids; the Spanish PDFs of the archive are not used); the MedNLI dev pairs not used by fast dev v4's mednli partition (995, disjoint by design, so MedNLI is not in both); PubMedQA labelled (1,000; none matched training); a stable sample of 1,000 DDXPlus validation cases rendered from the dataset's evidence dictionary (CC BY 4.0).
- **Environment change** (Y4, 2026-10-21): `uv pip install pillow==12.3.0` into the repo's environment, for the authors' pplx_decider package (the 27B decision model imports PIL). pyproject and the lock file are unchanged.
- **Y2 deviations and interpretations** (SELF_AUDIT §7):
  - Held-out routing: heart failure and AKI items come only from test-split (late-anchor) patients and go to the held-out file; the same families in train or dev are dropped and counted.
  - Leakage rule (ADVISORY Y2): an item in dev or test whose state text occurs in an earlier split is removed (counted) before the screens. This is a data filter, not an edit to a check.
  - MS-DRG distractors: "same MDC" approximated by title-token similarity (no MDC table in the data).
  - Lab validation: blood fluid only; mEq/L accepted for sodium and potassium (same numbers as mmol/L for these ions); anemia criterion sex-specific; coded status by ICD-10-CM prefix (anemia D64 is broad).
  - mimic_rad_exam_v1 NB 0.9000 is at the D35 shortcut threshold: excluded from the audit-suite verdict; reported only.
- **Y3 AKI and heart-failure targets:** codes in N17 (AKI) and I50 (heart failure, held out) are not pair targets at all;
  the AKI reading of ADVISORY §3.4 (AKI items to held-out only) is applied conservatively to pairs (no pair is built for them).
- Operator confirmations were given in the same message: D48 = yes; D46, D47, D49, D50 = yes. They are entered in §2 as the
  operator's values. Before this message they were placeholders, and D48 was read as not confirmed.
- **Y5 teacher and pool design** (2026-10-21, fixed before any 27B score; counts from `outputs/osler_v4/Y5/anchor_pool_manifest.json`):
  1. **Pool cap.** The anchor pool has 870,161 eligible rows after exclusions (CLAIMS C061). The loop writes the first 400,000 in stable-hash
     order and scores in that order; the teacher's speed decides how far the Y5 timebox reaches. Stage A draws at most
     13,992 × 8 = 111,936 items (`configs/osler_v3/block_arm.yaml`, batch size 8), so the cap is not binding for A5 even if
     scoring stops early. A5 trains on scored rows only.
  2. **Exclusion breadth.** Every row of every file under `data/bench` (60 files, 428,211 rows, any split; the sixtieth, `data/bench/osler_v4/mimic_dev_all.jsonl`, was added at 21:40Z) is an exclusion key,
     matched by normalised text or by (source, record id). This is stricter than ADVISORY §4 Y5's list, so no listed item can
     be in the pool.
  3. **Rendering.** A noul row goes to the teacher as its native noul question with criteria Yes / No (its answer is P(true),
     which is P(yes)). A choice row uses its option labels as criteria; a score row uses its level labels in order. A row
     that cannot be sent exactly is skipped with its reason (7,564 repeated labels) and never sent altered.
  4. **Scoring path.** Chosen by parity against D39's bar (argmax agreement with the single-request reference on 40 of 40
     items): batched reference, else the packed fast path, else single-request. The fallbacks change speed, not the function.
  5. **Pool dedupe corrected before any teacher output.** The first build deduplicated by item id, which dropped 22,106
     distinct Winogrande items (CLAIMS C059). The corrected pool deduplicates by normalised text (6,136 rows removed as
     repeats of an earlier text), keeps item-id repeats whose text differs, keys the scored records by pool position, and
     the dev union fails if one id maps to two texts. The superseded manifest is kept as
     `outputs/osler_v4/Y5/anchor_pool_manifest_superseded_itemid_dedupe.json` (CLAIMS C057, C058).
- **Y6 preparation** (CPU, 2026-10-21; the runs wait for the GPU):
  1. **Which arm is which.** ADVISORY Y6's table lists "A1, A3 | KL 0.1; broad replay". The provenance of the osler_v3 runs says
     A1 = KL 0.1 on the C1 replay (`outputs/osler_v3/stage_a/A1/provenance.json`), and A3 = KL 0 on `block_A3` with the broad
     replay. The choice rule uses the arms as their provenance records them.
  2. **`distill_weight` (D48, ADVISORY Y6).** `bench.schema.SoftItem` is a training-only subclass of `Item` (`Item` itself is
     unchanged) with `teacher_probs` and `teacher`. `reorder_options` permutes the distribution with the options, the encoders
     copy it to `EncodedItem.teacher_probs` in display order, and `train.losses.teacher_reference` builds the flat reference that
     `kl_to_reference` reads (the existing loss, no new loss). `StudentConfig.distill_weight` defaults to 0.0; the trainer adds
     `distill_weight * KL(teacher || model)` over the items that have a teacher and records `distill` and `n_teacher` per step.
     Tests: `tests/test_distill_train_v4.py` (10 with the conversion check), `tests/test_soft_labels_v4.py` (26).
  3. **The A5 mix** is `block_A3` with the Y5 distributions attached to its anchor rows by normalised text, option keys equal and in
     order (`soft_labels.attach_teacher`; `scripts/osler_v4/y6_a5_mix.py`). A5's replay is regenerated for that file, since the
     driver refuses a replay whose file hash differs (the stale replay is not edited).
  4. **The choice rule and the driver.** `meddecide.train.stage_a_choice` implements ADVISORY Y6's rule as written (MR and
     knowledge lower bounds above −0.01 against A0; the highest CG v2 accuracy among the qualifying runs; a margin under 0.003 is
     a tie and A0 wins it), with 7 tests. `scripts/osler_v4/y6_stage_a.py` has three subcommands: `train` (A4, A5; the osler_v3
     recipe with `distill_weight`), `score` (A0, A1, A3 re-scored from osler_v3's checkpoints; A4, A5 new), and `choose` (the
     paired bootstrap of `blocks.keep_statistics`, lower bounds `.lo`). The zero-shot pplx CG v2 baseline that ADVISORY Y6 names
     for the point estimate is not in this driver; the choice does not depend on it (a common offset), and it is needed for the
     report only.

- **Y4 (2026-10-22): the ceiling file name.** `scripts/osler_v4/y4_mimic_suite.py` read `outputs/osler_v4/Y4/ceiling_qwen35_4b.json`,
  a name `y4_ceiling.py` never writes for `--name qwen35_4b_mimic_dev` (it writes `ceiling_qwen35_4b_mimic_dev.json`). The constant
  now names the file the aggregator writes. Only the path changed: the Naive Bayes limit (0.90), the ceiling limit (0.97) and every
  rule are as they were, and the freeze has not run yet.

- **Y4 (2026-10-22): the MedDecider runner's option cap.** The authors' prompt letters options A–J (`LETTERS = "ABCDEFGHIJ"`, ten options),
  and my runner had no cap, so the CG v2 run stopped with IndexError after 7,296 rows. The O2 runner had the cap. The runner now
  skips and counts items with more than ten options (C086: 1,000 CG v2 items, 0 in the suite, 0 in the MIMIC partition), and the
  CG v2 run is repeated in the rest chain. No item was edited or dropped silently.
- **Y4 (2026-10-22): the readout checks for the baselines.** The authors' decision models (pplx-flash and MedDecider) normalise over the
  offered options and are trained readouts, so D21 applies: label mass is a diagnostic, not a gate, and it is not read here. The gate
  for their cells is the constant-answer and below-chance checks of `meddecide.eval.health` (the thresholds are the code's). A cell that
  fails reads READOUT_FAIL and is not reported as a result (D12). Observed on the complete baselines: MedDecider-4B on MIMIC
  drg_cc_level answers A on 0.945 of items (READOUT_FAIL); pplx-flash on CG v2 nfcorpus_relevant_noul answers no on 0.908
  (READOUT_FAIL). `scripts/osler_v4/y4_baselines_report.py` applies the checks.
- **Y4 (2026-10-22): the set-level status of the baselines report.** The first version of `scripts/osler_v4/y4_baselines_report.py` gave a set's overall status PASS even when one of its template cells failed the readout gate (MedDecider-4B on the suite, `mimic_drg_cc_level_v1`). Under D12 a set-level accuracy that includes a failing cell is not a result. The report now sets such a set's status to READOUT_FAIL with the failing cells named, writes `reportable: false`, and gives the passing-cell figure as an additional, labelled row. The cell checks, the thresholds and the D21 handling are unchanged. The fix came before any claim cited a set-level figure (CLAIMS C087–C093 cite the fixed report). The report also names the templates a run did not score: the CG v2 MedDecider-4B figure has no `cg_v2_ddxplus_validate_v1` cell (49 options; the authors' prompt has ten letters), so it covers 12 of 13 templates (CLAIMS C097).
- **Y4 (2026-10-22): the readout diagnostics are not gates.** Each cell of the report also carries the median top-1 minus top-2 gap, the share of exact ties and the argmax position counts (CLAIMS C093). They explain a constant answer; they do not change any cell's status.
- **Y4 (2026-10-22): the macro in the baselines report was not the project's macro (corrected).** The report's first versions gave each set a macro accuracy computed as a per-class recall over gold keys pooled across templates, and their set-level readout checks pooled the same keys. The project's macro accuracy (G1, D24; `meddecide.eval.metrics.template_macro_accuracy`) is the unweighted mean of the per-template accuracies. The report now gives that template macro for each set, drops the per-class column, and applies the readout checks per template only (D12). Claims C087, C089, C090, C091, C098 and C100 carried the pooled figure; their values and recompute commands were corrected in place (IDs unchanged). No gate or choice used the macro: the gates use micro accuracy and the per-cell checks.
- **Y4 (2026-10-22): the trainer's option keys beyond 26 options.** osler_v2's rows key options by position: A to Z for positions 0 to 25, then two-letter base-26 keys (BA for position 26), which the 49-option DDXPlus items need. `scripts/osler_v4/y4_rederive.py` checks every scored row's key against its position (`position_key`); the keys are names only, and the macro, majority and modal figures agree under the item's own keys (CLAIMS C096, C102).
- **Y6 (2026-10-22): the readout gate in the Stage A choice (recorded before any arm is ranked).** ADVISORY Y6's rule is applied as written, with two restrictions that follow from D12 and D21: a readout cell that fails the gate (a constant answer or below chance; `meddecide.eval.readout_gate`, whose thresholds are those of `meddecide.eval.health`) is not a result. (a) An arm's MR and knowledge lower bounds are computed on the templates that pass for both that arm and A0, on the same paired items. (b) The CG v2 accuracies used for the ranking are computed on the CG v2 templates that pass for every qualifying arm and for A0, the same items for each arm. If no cell fails for any compared arm, both restrictions are the full sets and the rule applies exactly as written. The gate was checked on A0 before any other arm was ranked: no failing cell on fast dev v4 (60 templates) or CG v2 (13 templates; CLAIMS C107), and A0's re-score reproduces osler_v3's decisions (C106). The zero-shot CG v2 offset cancels in the ranking (STATE §6, question 7).
- **Y7 (2026-10-23): the own check excludes a shortcut template (recorded before the B1 keep decision).** Keep rule v2's own check (D47: the block's new templates on fast dev) is applied to the new templates that pass the Naive Bayes screen (below the 0.90 shortcut limit, D35; ADVISORY Y2). mimic_rad_exam_v1 is among B1's new templates and sits exactly at the limit (Naive Bayes dev macro 0.9000; CLAIMS C084, C095). Its gain would measure a lexical shortcut rather than the skill the own check is for, so the conservative reading is to leave it out of own. The own figure with it is reported as additional. This is decided on the screen, before any B1 keep statistic exists. The other four bounds and every margin are unchanged, and the decision is recorded here before B1's keep decision (ADVISORY Y7: take the conservative option and record a Deviation).
- **Operator decision (2026-10-22), Y6, STATE §6 question 7: the CG v2 ranking.** The operator chose (b): the qualifying Stage A arms are ranked on the CG v2 templates whose readout passes D12 for every qualifying arm, on the same items for each arm. A0 is one of the qualifying arms by definition (`choose` lists it first), so `cmd_choose` takes the intersection over A0 and the qualifying arms, and every arm (qualifying or not) is scored on those templates. The zero-shot pplx-flash set-level CG v2 figure is NOT MEASURED (its failing cell `nfcorpus_relevant_noul_v1`: CLAIMS C093, C101); its passing-cell figure is an additional row (C101). This is recorded before the choice step, which has not run.
- **Operator decision (2026-10-22), Y9, STATE §6 question 6: Gate O4's MedDecider-4B comparison on the MIMIC audit suite, pre-registered before any final result.** The operator chose (b). On the frozen suite v1 (4,253 items, 9 templates; CLAIMS C083), the MedDecider-4B comparison uses the 8 templates whose MedDecider-4B readout passes D12, with identical items for both models compared (the final Osler-4B and MedDecider-4B). The excluded template is `mimic_drg_cc_level_v1` (constant answer on 0.952 of its items; CLAIMS C088, C093). The set-level MedDecider-4B figure is NOT MEASURED, and both are reported alongside the comparison. The zero-shot pplx-flash comparison uses all 9 templates. D49's baselines are unchanged (MedDecider-4B and zero-shot pplx-decider-flash-v1.2); only the template set of the MedDecider-4B comparison is fixed here. Recorded before Y8 and Y9 run, so no final result has been seen.
- **Operator ruling (Stage B, 2026-10-23), before any final run; recorded verbatim.**

  > Operator ruling (Stage B, 2026-10-23), before any final run. Record it verbatim in STATE §7
  > and blocks.md as an operator ruling; keep rule v2's NOT KEPT verdicts stay reported unchanged.
  >
  > 1. Override for the MIMIC-family blocks only (B1, B2, B3): each enters the final mix if its
  >    own lower bound > +0.02 and its knowledge lower bound > -0.02. Their MR, general and CG v2
  >    regressions at block length are accepted and reported. Reason: the fixed 13,992-step block
  >    runs dilute the base mix (B1 cuts the C1 share by ~32%; MR is step-sensitive, osler_v3
  >    C131), and the 63,304-step final gives the C1 part about osler_v2's exposure. B4 is judged
  >    by keep rule v2 as written.
  > 2. Unchanged: Gate O4, the MR suite v1, the MIMIC audit suite v1, the final selection rule, the
  >    recipe (A0) and the step budget. Y8 records CG v2 and MIMIC at every evaluation, as planned.
  > 3. Final-mix caps: MIMIC-family share <= 35% (subsample the MIMIC rows proportionally across
  >    templates if needed and record counts), replay >= 20%, no template > 8%.
  > 4. Record fix: MIMIC-III (NOTEEVENTS, ADMISSIONS, ...) has been on disk since
  >    2026-10-22T10:02Z; the re-check before B2's decision was missed. Correct the NOT MEASURED
  >    reason for mimic_page_category_v1 to "present but not built before B2's decision (missed
  >    re-check)". Do not add a block for it.
  > 5. Continue: finish B3 and B4, then Y8.

  Applied in this loop (recorded before any final run):
  - Item 1: B1 enters the final mix (own lower bound +0.1581 > +0.02; knowledge lower bound −0.0134 > −0.02; C129). B2 enters
    (own +0.0475 > +0.02; knowledge −0.0163 > −0.02; C139). B3 enters if its own and knowledge lower bounds pass when its keep
    decision is written. The NOT KEPT verdicts of B1 and B2 stay as reported.
  - Item 4: MIMIC-III 1.4 is on disk: 28 files; NOTEEVENTS.csv.gz modified 2026-10-22 10:02:24 UTC; ADMISSIONS.csv.gz modified
    2026-10-22 05:22:30 UTC (C142). The NOT MEASURED reason for mimic_page_category_v1 is corrected in §1, §5 and blocks.md. The
    dated status records (CLAIMS C065 and the 2026-10-21 lines of §6) are history and stay as written.

- **Y7 (2026-10-23): the B3 first attempt stopped; B3 is trained again from step 0 (a deviation, recorded before any B3 verdict).** The first run (outputs/osler_v4/Y7/B3_attempt1; log logs/B3_train_attempt1.log) stopped without a traceback after step 10,450, before its evaluation at 10,494 (CLAIMS C143). Its cause is not determined: the cgroup recorded no OOM kill, and a CPU dry run of the final-mix builder (15:08–15:09Z; a partial 3.9 GB output, removed) ran at the same time, the only heavy concurrent job of this session. The trainer has no mid-run resume (optimizer state is not saved), so the same block file and replay (the trainer checks their hashes) are trained again from step 0 (retry chain PID 1154282, started 2026-10-23T15:12:08Z). The retry records the training exit code, so a repeat stop shows its signal. Attempt 1's evaluations (3,498 and 6,996) stay on record and are used in no decision. Nothing else runs on the GPU or uses much memory while B3 runs.
- **Operator ruling, item 3: the final-mix caps are implemented (2026-10-23, before the final run).** `src/meddecide/train/final_mix.py` holds the pure functions (the family limit, the replay top-up, the proportional quota) and the build, which checks every cap before it writes; `tests/test_final_mix_v4.py` covers them, including a synthetic end-to-end build. `scripts/osler_v4/y8_final_mix.py` applies item 1 (B1, B2, B3 by the own and knowledge bounds; B4 by keep rule v2) and stops unless all four keep decisions exist. Its real build runs after B3 and B4 are decided.

- **Y8 preparation (2026-10-23; before B4 runs): shared code with defaults unchanged, the B4 base rule, the Y8 driver.** `TrainConfig`'s `eval_steps` (None by default) replaces the modulus schedule only when it is given (`is_eval_step` and `evaluation_grid` in `trainer.py`; Y8's ten evaluations fall at 6,330, 12,661, …, 63,304). `run_arm` takes an optional `partition_hook` (None by default) that records CG v2 and the MIMIC partition at every evaluation for the report, never for selection (CLAIMS C145: the suite with CUDA hidden passes). `scripts/osler_v4/y8_train.py` (the X9 guard, the replay check, the partition records, CG v2 at the selected checkpoint) and `scripts/osler_v4/y8_replay.py` are written; they run once the final mix exists. The MIMIC partition is the fast dev v4 items whose template id starts with `mimic_` (5,562 items: 5,162 structured and 400 doc-support); the 400 MedNLI items carry the source label `mimic_iv` and are not in it (counts only). B4's base follows keep rule v2 as written: B3's block if keep rule v2 kept B3, else B0. The four X1 templates pass the D35 screens (CLAIMS C146, `scripts/osler_v4/y7_x1_screen.py`); the DrugProt gold label occurs in the record text for 59 of 400 dev items, recorded and not excluded (its string-presence macro is 0.2501).

- **Operator steer (Stage B, 2026-10-23, after the B3 retry began): the final run must be resumable; recorded verbatim.**

  > Operator steer (2026-10-23): before launching Y8, make the final run resumable. Save a full
  > training checkpoint (LoRA, readout, optimizer, scheduler, RNG states, data position, step) every 5,000 steps under outputs/osler_v4/Y8/,
  > and add a --resume path to y8_train.py that continues from the latest one with identical data order. Test it on the tiny model (train
  > N steps, kill, resume, compare against an uninterrupted run). Shared trainer changes must default to the earlier behaviour. Do it on
  > CPU while B4 trains; never run two GPU jobs. If it is not done and tested by the time B4's keep decision is written, launch Y8 without
  > it and record that. Record this steer in STATE §7.

  Status of the steer is recorded below as the work proceeds (CLAIMS rows for each test).
- **Incident (2026-10-23, 20:41:35Z): the disk quota ran out; the B4 training, its chain and a full CPU test run stopped together, without a traceback (CLAIMS C154).** A 20 MB write failed with "Disk quota exceeded" at 20:44:04Z. Scratch that this session made was removed from /workspace/tmp (pytest's temporary directories, 974 MB; two resume debug directories, 174 MB; the torch compile cache, 78 MB), and the writes succeeded again. The cause of the stop is not established; the quota is the shared resource at that moment. Mitigations: (1) B4 is trained again from its block file with full checkpoints every 5,000 steps (the B4 retry chain above); (2) each full CPU test run removes its temporary directory afterwards (`rm -rf /workspace/tmp/pytest-of-root`), and the full suite runs only while no GPU job writes; (3) nothing under data/ was removed, since the guardrails allow deletion only under outputs/osler_v4/ and /workspace/tmp. The commit that carries this work waits for the full suite, which runs after B4's training.
- **Operator steer (2026-10-23), status: the resumable final run is done and tested on CPU.** `src/meddecide/train/resume.py` holds the full-checkpoint format (the trainable tensors by group and position, the optimizer, the run state, the RNG states, a COMPLETE marker written last). `trainer.train` takes `full_checkpoint_dir`, `full_checkpoint_every` and `resume_from`, all off by default, and its tests (with the arm and checkpoint tests) are unchanged. `run_arm` passes them through. `scripts/osler_v4/y8_train.py --resume` writes checkpoints every 5,000 steps under outputs/osler_v4/Y8/full, cuts the logs back to the checkpoint's step, and builds the final record from the complete logs. `scripts/osler_v4/y7_block.py` takes the same options for the block runs. Tests: CLAIMS C155–C156 (9 passed on CPU: the I/O round-trips, the end-to-end resume on the tiny model, and a SIGKILL test). The steer's condition (tested before B4's keep decision) is met, so Y8 launches with resume support.

- **Y7 (2026-10-23): the B3 result and a readout defect in it (recorded before any B4 decision).** The B3 retry finished all 13,992 steps (CLAIMS C148–C149). Its keep rule v2 verdict is NOT KEPT: MR (lower −0.0188 against −0.015) and CG v2 (lower −0.0163 against −0.010) fail (C152). Under item 1 of the operator ruling it enters the final mix: its own lower bound is +0.1625 (against +0.02) and its knowledge lower bound −0.0082 (against −0.02). The readout gate flags one cell on fast dev v4, ragtruth_sentence_supported_v1: a constant answer, A on 90.5 % of its 400 items (accuracy 0.6125; C150). The base A0 answers A on 85.75 % of the same items with accuracy 0.665, and the gold answers are near balance (A 211, B 189; C151). I found no label or pipeline defect: the bias is present in the base, and B3 crosses the D12 limit of 0.90. Per D12 the cell is not a result, so the keep statistics exclude it from both comparisons (the general partition loses it); no check or limit was edited. B4 includes this template's training rows, and its readout is gated as usual.
- **Operator steer (2026-10-23), progress: the resumable final run.** Done so far, all default-preserving: `src/meddecide/train/resume.py` (the full-checkpoint format, the atomic write, the latest complete checkpoint, the log truncation; tests/test_resume_io_v4.py, 7 tests passing on CPU). The end-to-end test on the tiny model is written (tests/test_resume_trainer_v4.py: an uninterrupted six-step run against a run interrupted at step 3 and resumed); it runs after the trainer hooks are added. The trainer hooks, the `--resume` path in y8_train.py and the 5,000-step saves are added after B4's training has imported the trainer, so B4 runs the code it started with. If the test is not passing when B4's keep decision is written, Y8 launches without resume and this is recorded.

- **Deviation (2026-10-24, before the Y8 launch): the final mix is composed in memory, not written to disk.** The disk quota left under 1 GB of headroom (a 1 GB write failed at 02:19Z; CLAIMS C154 gives the quota incident). The planned ~9.5 GB mix file cannot be written, and the guardrails allow no other deletion (data/ and the other loops' scratch stay). The final mix is therefore composed in process by `meddecide.train.final_mix.compose_final_mix` from its recorded sources (osler_v2 C1 as the base; the added files of B1, B2 and B3, which enter by the operator's item 1; the broad replay pool for the top-up), with the caps checked before composition. Its identity is the SHA-256 of its item ids (`composition_sha256`, 9d25e30f…; CLAIMS C164), and the source hashes are in outputs/osler_v4/Y8/final_mix.json. The replay is computed in process from the same items (`replay_records` in scripts/osler_v4/y8_train.py) and recorded in outputs/osler_v4/Y8/replays/final_mix_replay.json. The composition reproduces on repeated runs. `scripts/osler_v4/y8_replay.py` and the file writer are superseded and not run.
- **Operator steer (2026-10-24), the quota and resumable Y8: pruning.** Full checkpoints every 5,000 steps are written under outputs/osler_v4/Y8/full; before each write only the newest complete checkpoint is kept, so two remain after it (`resume.write_full_checkpoint(keep_last=2)`; a test in tests/test_resume_io_v4.py; the default 0 keeps all, so earlier runs are unchanged). The first attempt's model folder of B3 (outputs/osler_v4/Y7/B3_attempt1/checkpoints, 27 MB) was removed to free quota; its logs are kept.

- **Y7 closure record (2026-10-24T05:58:28Z, before the Y7 commit).** (a) `blocks.md`'s B2 note said the logged step 1,650 had exactly zero loss; the logged values are 5.96e-07 (loss) and 4.6e-05 (gradient). Corrected (C172). (b) The B4 keep decision scored the selected checkpoint (step 10,494; C160); the working notes' "post-training" label for B4's 13,866 correct was wrong (C170). (c) `blocks.md`'s table rows are in block order; no value changed (C168). (d) The Y7 checklist in §3 is replaced by the Y8 checklist; the Y7 items are recorded in the §4 entry. Verification: `outputs/osler_v4/Y7/SELF_AUDIT.md`.

---

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| | | | |
