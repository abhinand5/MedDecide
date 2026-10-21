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
Last updated (UTC): `2026-10-21T21:50:18Z`
Iterations so far: `1`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| Y0 | Orientation, Nemotron serving, MIMIC guard | small | — | DONE | 2026-10-21T18:26:40Z | 2026-10-21T18:53:35Z |
| Y1 | MIMIC foundation (tables, sections, splits) | no | Y0 | DONE | 2026-10-21T18:54:18Z | 2026-10-21T19:44:00Z |
| Y2 | MIMIC structured-gold templates (+ Y2b late sources when present) | no | Y1 | DONE (mimic_page_category_v1 NOT MEASURED: MIMIC-III absent) | 2026-10-21T19:44:00Z | 2026-10-21T20:40:16Z |
| Y3 | Nemotron-crafted minimal pairs (`mimic_doc_support_v1`) | yes | Y0, Y1 | IN_PROGRESS (pilot done; build pending) | 2026-10-21T19:25:21Z | |
| Y4 | Fast dev v4, CG v2, MIMIC audit suite v1, baselines | yes | Y2, Y3 | IN_PROGRESS (CPU preparation: fast dev v4, CG v2 and the suite script are built; GPU steps wait for the Y3 build) | 2026-10-21T21:26:55Z | |
| Y5 | pplx 27B soft labels (only if D48 confirmed) | yes | Y0 | IN_PROGRESS (CPU preparation: code, tests and the anchor pool; the GPU steps follow the Y3 build, per the ADVISORY §5 order Y3 → Y5 → Y4) | 2026-10-21T21:33:00Z | |
| Y6 | Stage A: A4, A5 + re-scored A0/A1/A3 + choice | yes | Y4, Y5 | PENDING | | |
| Y7 | Stage B: blocks B1–B4 under keep rule v2 | yes | Y6 | PENDING | | |
| Y8 | Final Osler-4B (63,304 steps) | yes | Y7 | PENDING | | |
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
| minimal pairs: pilot keep rate per mode, chosen mode, cue AUC, pairs built | | `loops/osler_v4/crafted.md` | Y3 |
| fast dev v4 partitions; CG v2 size and sources; MIMIC suite v1 templates and items | | | Y4 |
| pplx 27B: revision, licence, parity, anchor items scored, overlap check (must be 0) | | | Y5 |
| Stage A: MR, knowledge, CG v2 per run; the chosen recipe | | `loops/osler_v4/stage_a.md` | Y6 |
| Stage B: kept blocks; own / MR / general / knowledge / CG v2 per block; MIMIC share | | `loops/osler_v4/blocks.md` | Y7 |
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
Task in flight: Y3 (started 2026-10-21T19:25:21Z). Y1 closed 19:41:36Z (its checklist is in the §4 entry).
Working dir:    outputs/osler_v4/Y3/ (pilot_off, pilot_on, build_off); private pairs under data/mimic_build/pairs/ (gitignored)
Server:         none running (Nemotron stopped 19:41Z so the full test suite could use the GPU). Restart before the build.

- [x] helper module (meddecide.mr.mimic_pairs) and 19 unit tests with a fake teacher
- [x] driver scripts/osler_v4/y3_pairs.py (pilot, build, cue, pilot-report, report); the six-attempt smoke test was deleted
- [x] pilot off (200 train attempts, 152 s) and pilot on (200 attempts, 687 s); pilot-report: no mode reaches a 40 % keep rate
- [x] mode decision: off (deviation, §7); loops/osler_v4/crafted.md pilot section written
- [ ] full-suite acceptance for the Y1 and Y3 code (ruff and pytest with the GPU free): running from 19:41Z
- [ ] restart Nemotron (serve_nemotron.sh, detached); `build --mode off --hours 8` (dev 250, test 250, train up to 40,000 kept)
- [ ] cue check on the build (AUC < 0.75 on dev, ADVISORY Y3 §6); crafted.md build and cue sections (report --mode off)
- [ ] SELF_AUDIT.md (outputs/osler_v4/Y3); CLAIMS rows C024 onward
- [ ] STATE updated, committed, pushed; Y3 DONE or BLOCKED
```

```
Task in flight: Y5 (CPU part started 2026-10-21T21:33:00Z; GPU part waits for the Y3 build and the server stop).
Working dir:    outputs/osler_v4/Y5/ (logs, manifests, parity, dev diagnostics, anchor scores); soft labels under data/train/osler_v4/ (private).
Pinned teacher: perplexity-ai/pplx-decider-v1.2-27b @ cce0f3f80033e9d89d80831d5281fd72124aacce (README and LICENSE: Apache-2.0).

- [x] helper module meddecide.distill.soft_labels (request and answer mapping, exclusion keys, pool order); tests/test_soft_labels_v4.py (22 passed)
- [x] driver scripts/osler_v4/y5_soft_labels.py (pool, parity, dev, anchor, join, report); ruff clean
- [x] pool (CPU): exclusions from every benchmark row, anchor templates only, request-mappable rows only, one row per normalised text (the first build deduplicated by item id; corrected, §7 item 5); manifest outputs/osler_v4/Y5/anchor_pool_manifest.json (eligible 870,161; written 400,000)
- [ ] GPU, after the Y3 build and the server stop: parity (40 of 40 argmax with the single-request reference, D39's bar), dev diagnostics, anchor scoring (≤ 5 h), join (overlap must be 0), report
- [ ] SELF_AUDIT, CLAIMS rows, STATE and commit; Y5 DONE or BLOCKED
```

Y4 CPU preparation items done since the last update (same file, §4 entry): `scripts/osler_v4/y4_mimic_dev_all.py` built `data/bench/osler_v4/mimic_dev_all.jsonl` (6,257 items, 13 templates, counts in `outputs/osler_v4/Y4/mimic_dev_all_manifest.json`); the interim CPU pytest run is in `outputs/osler_v4/Y4/logs/pytest_interim_cpu.log` (GPU-marked tests skip with the card hidden; the final run waits for the card).

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

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|
| Y2 mimic_page_category_v1 | MIMIC-III page-category template (B3 part) | NOT MEASURED — MIMIC-III 1.4 is still downloading (dm status 0/28 files at 20:40Z) | Re-check at Y4 and before the B2 keep decision; build it then if the files are present, else B3 goes ahead without it (operator steer 2026-10-21) |

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

---

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| | | | |
