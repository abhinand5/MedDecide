# osler_v3 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/osler_v3/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `STOPPED`  <!-- set to STOPPED at the hard stop (X12), or when no PENDING task can proceed without the operator -->
Run started (UTC): `2026-10-18T20:13:37Z`
Last updated (UTC): `2026-10-20T23:12:00Z`
Iterations so far: `1`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| X0 | Orientation, KL timing | small | — | DONE | 2026-10-18T20:13:37Z | 2026-10-18T20:40:13Z |
| X1 | Real sources: RAGTruth, DrugProt, MultiCaRe | no | X0 | DONE | 2026-10-18T20:26:00Z | 2026-10-18T20:40:13Z |
| X2 | Broad replay for A3 | no | X0 | DONE — 35 % replay target NOT MET (32.66 % after the 8 % cap; recorded, D-X2-2) | 2026-10-18T20:31:00Z | 2026-10-18T20:40:13Z |
| X3 | Gemma-crafted items over real text | yes | X0, X1 | IN_PROGRESS — restyle used (6,426 verified pages; C081–C085; D-X3-18; the genre list is open, Q7); claims: the second prompt fix and the family filter gave a pilot that FAILS the D44(b) check (2 flags, both tracking not-mentioned; C106, C107), so no claims are built and B2 is restyle only (D-X3-19) | 2026-10-18T20:37:00Z | |
| X4 | Charts with the D44 rule | yes | X0 | DONE — 2,000 charts built (1,715 kept; C057); chart_indicator_evidence_v1 usable (2,916 / 330 / 184 items; C059); chart_icd_support_v1 and chart_evidence_page_v1 BLOCKED under D44(b) (C061, C062); `loops/osler_v3/charts.md` | 2026-10-18T21:14:29Z | 2026-10-18T23:40:42Z |
| X5 | MIMIC (official files only; only when they exist) | no | X0 | BLOCKED — official MIMIC files absent (data/physionet/ missing; D37 forbids unofficial copies); not waited for | 2026-10-18T20:13:37Z | 2026-10-18T20:40:13Z |
| X6 | Fast dev v3, CG proxy, new baselines | yes | X1, X2, X3, X4 (X5 if built) | DONE (the extension below is recorded under X8) — CG 4,865 items; zero-shot pplx 0.7781 on 4,853 scored (C075, C076); new test pool 9,016 items (C073); six baselines additional (C077–C079); the indicator split (C080). PENDING under the operator's ruling: the restyle test pages (140) are not yet in the new test pool, and the six baselines have not been run on them (additional rows) | 2026-10-18T21:14:29Z | 2026-10-19T00:38:00Z |
| X7 | Stage A: recipe ablations A0–A3 + choice | yes | X6 | DONE — choice A0 (the control, kl_to_base 0; osler_v2's C1 mix) by the pre-registered rule (C098–C101; `loops/osler_v3/stage_a.md`); the choice first failed on a join of item ids (D-X7-1), fixed and re-run; no run's numbers changed | 2026-10-18T21:14:29Z | 2026-10-19T17:46:49Z |
| X8 | Stage B: data blocks | yes | X7 | DONE (operator, 2026-10-20: keep decisions re-run after the CG side-report crash, commit 7f47e1e; B1, B2, B3 not kept; last kept block B0 = osler_v2's C1) — B1, B2, B3 trained (13,992 steps each; B1 05:41–09:31Z, B2 09:38–13:30Z, B3 13:36–17:26Z on 2026-10-20). All three keep steps crashed in the chain (missing `b0_dev/cg_predictions.jsonl` in the labelled CG side report; the rule itself was unaffected), so the chain treated each as not kept and built B2 and B3 on B0. Fixed by the advisor (`x8_keep.py` reads B0's CG predictions from its Stage A run, the same checkpoint) and re-run: B1 NOT KEPT (MR −0.0044 [−0.0143, +0.0058]; its own new templates +0.151), B2 NOT KEPT (MR −0.0096 [−0.0209, +0.0019]), B3 NOT KEPT (MR −0.0054 [−0.0139, +0.0032]). Since B1 is not kept, B2 and B3 were built on the right base. Last kept: B0 (= osler_v2's C1). Operator decision pending on X9 (Question: the final mix equals osler_v2's) | 2026-10-19T17:46:49Z | |
| X9 | Final Osler-4B | yes | X8 | BLOCKED — skipped by the operator (2026-10-20): the kept mix equals osler_v2's C1 with osler_v2's recipe, so a final run would retrain osler_v2's model (duplicate of osler_v2's final) | | |
| X10 | Evaluation, reduced (no Gate O3 verdict; Stage A runs, B1 and the trade-off as additional rows) | yes | X8 (X9 skipped) | DONE — no verdict (operator ruling); A0–A3 and B1 scored on all 30,621 v10 rows, calibrated, against zero-shot pplx and osler_v2's Osler-4B (C119–C131; `loops/osler_v3/results.md`, `loops/osler_v3/gate_o3.md`) | 2026-10-20T21:12:00Z | 2026-10-20T23:10:00Z |
| X11 | Memorisation audit (only with MIMIC) | small | X9 | BLOCKED — not applicable: no MIMIC in a final mix (operator ruling 2026-10-20; X9 skipped) | | |
| X12 | Findings and closure — HARD STOP | no | all | DONE — `loops/osler_v3/FINDINGS.md` (Summary, §2 outcomes, the Stage A trade-off, five-claim spot-check), `loops/osler_v3/NEXT.md`; loop STOPPED | 2026-10-20T23:10:00Z | 2026-10-20T23:12:00Z |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (order exceptions in ADVISORY §5). Never run two GPU jobs
at once. A `BLOCKED` task does not block unrelated work. No row may be left `IN_PROGRESS` at closure.

---

## 2. Key values discovered during the run

| key | value | source | task |
|---|---|---|---|
| KL timing: s/step, anchor share, peak (pplx, kl 0.5) | control 0.7935 s/step; KL 0.5 1.0130 s/step (+27.7 %, mean after warm-up; median +37.8 %); anchor share 0.8775 of C1 (728,351 of 830,076; 0.8765 in the 150-step run); peak 57.08 GB in both arms; mean KL 0.2624 | `outputs/osler_v3/X0/` (kl_0, kl_0p5, anchor_share_C1.json, SELF_AUDIT.md); CLAIMS C001–C010 | X0 |
| real sources: items per template, licences, screens, leakage | train 69,464 / dev 7,878 / test 8,832 rows (RAGTruth MIT: grounded 12,026, sentence 23,294, type 400; DrugProt CC BY 4.0: 33,744); leakage 0 hits (12 protected sets); NB screens max 0.695 (none ≥ 0.90); MultiCaRe: 0 items (age and sex stated in the text) | `loops/osler_v3/sources.md` | X1 |
| broad replay: datasets, licences, A3 replay share | 162,797 rows from 10 datasets (11 template sources), all licences verified on card or README (MIT, CC BY 4.0, Apache-2.0); leakage 0 hits; C1 general replay 21.04 %; A3 33.98 % before the 8 % cap and 32.66 % after (target 35 % not met) | `loops/osler_v3/sources.md` §5; CLAIMS C033–C041 | X2 |
| Gemma-crafted: pilot keep / agreement rates; items built | claims (train pilot, after the one allowed prompt fix): kept 272 of 300 (90.7 %; C044); 'family' in 41.9 % of kept not-mentioned claims vs 0.0 % of supported or contradicted (C045; D-X3-1's 39 % does not reproduce); verifier agrees on 90 of 92 contradicted and 105 of 106 not-mentioned (C047); BLOCKED. Restyle (build): 6,426 of 6,645 kept (C048); 6,428 verifier agreements (C049); balanced 4,746 / 406 / 91 (C050); 103 phrase flags (C051); NB macro 0.7143 on balanced dev (C052); BLOCKED; no X3 rows (C055) | `outputs/osler_v3/X3/` (build.json, readthrough_*.json) | X3 |
| charts: items built; cue check | | `loops/osler_v3/charts.md` | X4 |
| fast dev v3 partitions; CG partition size and templates; zero-shot pplx on CG | | | X6 |
| Stage A: MR, CG, knowledge, general per run; mean KL; the chosen recipe | | `loops/osler_v3/stage_a.md` | X7 |
| Stage B: kept blocks and dev deltas | | `loops/osler_v3/blocks.md` | X8 |
| final Osler-4B: steps, selected step, CG trend | | | X9 |
| Gate O3 verdict (4B) | | `loops/osler_v3/gate_o3.md` | X10 |

---

## 3. Current task checklist

**Fill this in before starting any task**, by copying that task's concrete steps out of
the plan as unticked boxes. Tick each one the moment it is finished, not at the end.
This is the only record of progress *within* a task — without it, a compaction mid-task
leaves you unable to tell what you already did.

Clear this section and write the new task's checklist when you start the next task; the
completed checklist goes into the iteration-log entry.

```
Task in flight: X7 — Stage A recipe ablations (A0 to A3); X6 is DONE (see §4)
Working dir:    outputs/osler_v3/X6/ (queue log outputs/osler_v3/X6/logs/chain.log; per-step logs in the same folder)
Running jobs:   the X8 Stage B chain `scripts/osler_v3/x8_chain.sh` (detached; log outputs/osler_v3/X8/logs/chain.log; per-block logs outputs/osler_v3/X8/logs/B1.log, B2.log, B3.log). At 2026-10-19T18:14:41Z: B1 replay (CPU) after its build; then B1 training (GPU, 13,992 steps, kl 0), CG score, keep rule; then B2 and B3 the same way. The Gemma server is stopped (0 MiB at 18:00Z).

X3 closure (done 22:23Z):
- [x] x3_assemble re-run; the NB screen runs on the blocked template as a diagnostic (D-X3-12)
- [x] x3_recount.py: 31 of 31 checks agree (outputs/osler_v3/X3/recount.log)
- [x] SELF_AUDIT.md written (outputs/osler_v3/X3/SELF_AUDIT.md)
- [x] CLAIMS rows C044–C056 appended; each recompute command run
- [x] gate run: ruff clean; pytest 838 passed, 1 xfailed, 1 skipped, with one GPU-only test deselected (D-X3-15)
- [ ] commit "loop(osler_v3): X3 BLOCKED" and push loop/osler_v3

X4 closure (done 2026-10-18T23:40:42Z):
- [x] build finished: 2,000 charts (200 pilot + 1,800 build); 1,715 kept; 6,862 items (C057)
- [x] full check, original-exemption check, first-500 selection (equal to the full check; D-X4-3); checks.json and the logs
- [x] per-template screen and label tracking (C060–C063); leakage 0 (C066); screen diagnostic (C064, C065)
- [x] `loops/osler_v3/charts.md`; CLAIMS C057–C071; SELF_AUDIT in outputs/osler_v3/X4/
- [x] conversion to mix rows, blocked templates excluded and counted (C067); Gemma server stopped by PID
- [ ] commit and push the X4 closure

ORDER OF WORK after the operator ruling (2026-10-19T04:23:08Z): (1) Stage A arms A1, A2, A3 run on the UNCHANGED fast dev v3 (they read `data/bench/osler_v3/fastdev_v3.jsonl` at their start, so it must not be rebuilt while Stage A runs); (2) the choice (watcher bc1dcnrfp); (3) the claims pilot on the GPU, then the D44(b) check; (4) if it passes, the claims build (up to 20,000); (5) the fast dev v3 rebuild with the restyle dev pool (`data/bench/osler_v3/x3_dev.jsonl`, 489 pages; 400 cap) and the claims dev pool; (6) the new test pool with the restyle test pages (140; additional rows; the six baselines are re-scored on them); (7) rescore the chosen Stage A checkpoint (B0) on the new fast dev v3; (8) the X8 chain: B1 (X1 sources), B2 (restyle + claims, or restyle only), B3 (charts). The X8 chain is NOT launched as written: its B2 step is not in it yet.

X3 claims rework (written 2026-10-19T04:22:44Z, NOT run; operator ruling D-X3-17): `src/meddecide/mix/clinclaim.py` has the second prompt fix (not-mentioned claims about the same kinds of facts: finding, test, treatment, timing; never about relatives) and the filter (`family_filter`: a kept not-mentioned claim that mentions family is dropped; the count is in the pilot report). Pilot: `x3_generate.py --task claims --split train --limit 300 --out data/gen/osler_v3/claims_v2_pilot_train.jsonl` on the GPU after the Stage A choice (one GPU job at a time), then `scripts/osler_v3/x3_claims_pilot_check.py` (the D44(b) label-tracking check; PASS iff no flagged phrase). If it passes: build up to 20,000 claims (about 2 GPU hours) in the same way; if not, B2 is restyle only (D-X8-2).

X8 code (written, NOT launched; committed 65e3c45): `scripts/osler_v3/x8_build_block.py` (streamed block build: base + new source + minimal broad replay top-up to 20 %; the 8 % cap fails the build), `scripts/osler_v3/x8_keep.py` (keep rule via `keep_comparison`, the new sources removed from both sides per D-X8-1, reported beside it; CG reported beside it), `scripts/osler_v3/x8_chain.sh` (chosen Stage A arm as B0; B1 = X1 sources; B3 = X4 charts on the last kept block; `x7_block_run.py` for training at the chosen KL). Smoke test of the keep code on osler_v2 W6 reproduces C1's keep decision (output removed; not a result). X9 needs a per-evaluation CG hook (the shared `run_arm` keeps only the best checkpoint; not changed): to be written with X9.

X6 queue (written, NOT launched; `scripts/osler_v3/x6_chain.sh`): zero-shot pplx on the CG partition (scope mix_v2), then the
new test pool with osler_v2 W5's five baselines and zero-shot pplx, then `x7_stage_a.sh`. It refuses to start while the vLLM
server holds the GPU. Before launch: X4 closed; `x6_partitions.py --with-new-templates` (fast dev v3); `x6_new_test_o2.py`
rebuilt with the X4 test charts of the kept templates only; the Gemma server stopped by PID 3254457.

X9 driver (written, NOT launched; committed 1d394b0): `scripts/osler_v3/x9_final.py`. It reads the last kept block (`outputs/osler_v3/X8/last_kept.txt`) and the chosen Stage A arm's KL. `--replay-only` writes `outputs/osler_v3/X9/replay_250k.json` (the steps: the smallest multiple of eight that reaches 250,000 realised items). The run takes eight evaluations on fast dev v3 with the trainer's selection, and scores the CG partition at every evaluation inside the trainer's hook (labelled; before the selection; `cg_trajectory.json`). Temperatures are fitted and saved with the selected checkpoint. It refuses to run twice (`arm_result.json` present).
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

### X0 — DONE — 2026-10-18T20:40:13Z
- What ran: `scripts/osler_v3/x0_kl_timing.py` (kl 0.5 and a kl 0 control, 150 steps each on C1, pplx, trainer only); `scripts/osler_v3/x0_anchor_share.py`; ruff; pytest gate with the padding test deselected; Gemma serve (`envs/vllm`, one reply, stopped).
- Output: `outputs/osler_v3/X0/` (kl_0, kl_0p5, anchor_share_C1.json, logs, SELF_AUDIT.md).
- Headline: KL 0.5 costs +27.7 % step time after warm-up (1.0130 vs 0.7935 s/step; C001–C003); anchor share 0.8775 of C1 (C007); peak 57.08 GB in both arms (C008); pytest 760 passed, 1 skipped, 1 deselected (C011).
- Surprises: 87.8 % of C1 items are anchors under the §3.2 prefixes (the osler_v0 clinical general templates are anchors by design); the control arm was added (D-X0-1).
- Next: X1 and X2 (CPU); X3 and X4 need the Gemma server.

### X1 — DONE — 2026-10-18T20:40:13Z
- What ran: `scripts/osler_v3/x1_fetch.py`; `x1_real_sources.py`; `x1_checks.py`; `recount_sources.py`. New code: `meddecide/mix/ragtruth.py`, `drugprot.py`, `sentences.py`, `protected.py`; tests `tests/test_osler_v3_sources.py`.
- Output: `loops/osler_v3/sources.md`; `outputs/osler_v3/X1/` (build.json, checks.json, multicare_audit.json, fetch.json, SELF_AUDIT.md).
- Headline: X1 rows train 69,464, dev 7,878, test 8,832 (RAGTruth MIT; DrugProt CC BY 4.0); leakage 0 hits in all three (C029); no Naive Bayes screen at or above 0.90 (C031); MultiCaRe builds no direct item (C026–C028).
- Surprises: 2,182 RAGTruth responses have two or more span types or an unknown type, so the type template is small (400 / 50 / 40 items); `implicit_true` spans counted (Q1).
- Next: X3 (Gemma) reads MultiCaRe, MTSamples, CodiEsp train and PMC-Patients as substrate.

### X2 — DONE — 2026-10-18T20:40:13Z
- What ran: `scripts/osler_v3/x2_fetch_replay.py`; `x2_broad_replay.py`; `recount_sources.py`. New code: `meddecide/mix/broad_replay.py`, `meddecide/mix/protected.py`; tests `tests/test_osler_v3_replay.py`.
- Output: `data/train/osler_v3/replay_broad_train.jsonl` (gitignored); `outputs/osler_v3/X2/` (build.json, fetch.json, recount.log, SELF_AUDIT.md).
- Headline: 162,797 broad replay rows from 10 datasets (11 template sources), licences verified (C033–C034); leakage 0 hits (C038); A3 general replay share 32.66 % after the 8 % template cap, target 35 % NOT MET (C041; D-X2-2).
- Surprises: MathQA is 97.7 % AQuA-RAT's problems in tokenised form (C036); AQuA-RAT repeats problem texts (C037; Q2); SciTail, MNLI, FEVER, PAWS, OpenBookQA, SWAG, AI2 ARC copies and GLUE/SuperGLUE excluded on licence.
- Next: X3 and X4 (GPU). The A3 mix is built with the cap in X7.

### X5 — BLOCKED — 2026-10-18T20:40:13Z
- What ran: `ls data/physionet` (absent); no download, no read of any unofficial copy (D37).
- Output: none.
- Headline: MIMIC official files absent; X5 does not block X6 (ADVISORY §5); re-checked before X6.
- Surprises: an unofficial MIMIC re-upload sits in the HF cache (`thbndi/Mimic4Dataset`); it is not used (D37).
- Next: none in this loop unless the files appear.

### X3 — WIP — 2026-10-18T21:07:53Z
- What ran: `scripts/osler_v3/x3_substrate.py` (203,890 real documents: MultiCaRe 104,468 usable, PMC-Patients 98,575, MTSamples 431, CodiEsp 416 (osler_v2 training cases only); split 80/10/10 by hash; excluded: 12,317 PMC and 108 MultiCaRe documents that share a paragraph with a protected state, and the osler_v2 fast-dev CodiEsp cases); `x3_generate.py` pilots (300 documents each); `x3_assemble.py` checks.
- Output: `data/gen/osler_v3/` (gitignored); `outputs/osler_v3/X3/` (substrate.json, build.json, readthrough_*.json, logs).
- Headline: claims — kept 272 of 300 after the one allowed prompt fix, but the label-tracking check fails: 'family' / 'family members' in 39 % of not-mentioned items and 0 % of the others (D44(b)) → BLOCKED, no claim rows written. Restyle — kept 286 of 300 (95 %) with the original prompt; classes balanced at 30 or more each; no phrase flags except the page type's own section words (impression, demonstrates, there is: radiology genre; see Q4).
- Surprises: the verifier agrees with the drawn label on about 90 % of claims, but the spot read found 'contradicted' items that are only unmentioned facts (for example "tricuspid regurgitation was absent" when the passage never mentions it): label noise in the contradicted class (NEXT). The number rule reads digits only, so number words ('eight') fail it. Without a page-type title, restyle keeps 49 % and two classes nearly vanish (D-X3-2).
- Next: the restyle chain; then X4 (charts, same server) with the D44(b) check; then X3 closure (BLOCKED claims, restyle DONE with its screens).

### X3 — BLOCKED — 2026-10-18T22:23:56Z
- What ran: `scripts/osler_v3/x3_assemble.py` (closure re-run; the NB screen now also runs on the blocked template as a diagnostic); `scripts/osler_v3/x3_recount.py` (fresh-process recount of the candidates against build.json: 31 of 31 agree; `outputs/osler_v3/X3/recount.log`); `x3_generate.py` pilots and build before this closure (Gemma 4 31B, vLLM; verifier at temperature 0).
- Output: `outputs/osler_v3/X3/` (build.json, recount.log, SELF_AUDIT.md, read-through files, logs). X3 rows: 0 in train, dev and test (C055).
- Headline: claims BLOCKED: 272 of 300 candidates kept (C044); "family" in 41.9 % of kept not-mentioned claims vs 0.0 % of the others (C045; D-X3-1's 39 % does not reproduce, C046). Restyle BLOCKED: 6,426 of 6,645 candidates kept (C048); 103 label-tracking phrases beyond the class name on balanced train (C051); NB macro 0.7143 on balanced dev, below the 0.90 defect screen (C052), so the block is the phrase check, not the screen.
- Surprises: the screen had skipped the blocked template, so Q4's NB value had no artifact (now recounted); STATE said the verifier agrees on "90 % of contradicted", the count is 90 of 92 (C047); the restyle test pass wrote 145 candidates, not 91 (D-X3-10); the kept-claims read-through is empty by construction (SELF_AUDIT §5); 3 CodiEsp documents have no usable passage with no counted reason (SELF_AUDIT §5, open).
- Next: X4 (the 2,000-encounter chart build is running on the GPU server; the charts check runs when it finishes). X6 proceeds without X3 items (the X3 test pool is empty and recorded). X7 does not use X3 rows.

### X4 — DONE — 2026-10-18T23:40:42Z
- What ran: `w3b_charts.py --count 2000` (osler_v2 generator; pilot of 200 earlier; build of 1,800 from 22:13Z to 23:30Z); `x4_charts_check.py` (full: checks.json; `--original-exemptions`: checks_original_exemptions.json; `--first-charts 500`: checks_first500.json, equal to the full check, D-X4-3); `x4_screen_diagnostic.py`; `x4_to_mix.py`; the Gemma server stopped by PID (GPU 0 MiB).
- Output: `loops/osler_v3/charts.md`; `outputs/osler_v3/X4/` (checks, screen_diagnostic.json, to_mix.json, SELF_AUDIT.md, logs); mix rows in `data/train|bench/osler_v3/x4_*.jsonl` (gitignored).
- Headline: 2,000 charts built, 1,715 kept, 6,862 items (C057). The indicator template is usable: 2,916 / 330 / 184 items (C059), no label-tracking phrase flagged (C060), NB 0.6758 on dev (C063). chart_icd_support_v1 and chart_evidence_page_v1 are BLOCKED by label-tracking phrases beyond the labelled fact (C061, C062). No leakage hits (C066).
- Surprises: the choice templates' NB values fall below a uniform guess because the screen scores option texts and ignores the offered options; the ICD predictions are one class for 83.6 % of dev items (C064). The verdict does not rest on them (D-X4-5). The first-500 re-check cannot be selected from the rows (D-X4-3).
- Next: X6 (fast dev v3 and the new test pool are built; the queue runs zero-shot pplx on CG, then the five baselines and zero-shot pplx on the new test pool, then Stage A).

### X6 — DONE — 2026-10-19T00:39:50Z
- What ran: `x6_partitions.py --with-new-templates` (CG 4,865; fast dev v3 12,903); `x6_new_test_o2.py` (new test pool 9,016); `x6_chain.sh` (zero-shot pplx on CG; the new test pool with zero-shot pplx, Qwen 4B and 9B, MedDecider 4B and 9B, JEV-9B; then Stage A); `x6_baselines_summary.py`; `x6_indicator_units.py`.
- Output: `outputs/osler_v3/X6/` (manifest.json, new_test_manifest.json, zeroshot_pplx_cg, new_test per model, baselines_summary.json, indicator_units.json, logs); `loops/osler_v3/x6_baselines.md`.
- Headline: CG proxy 4,865 items; zero-shot pplx 0.7781 on its 4,853 scored items (C076). Additional rows on the 9,016-item new test pool: zero-shot pplx 0.6818, Qwen 4B 0.6226, Qwen 9B 0.6654, JEV-9B 0.6064, MedDecider 4B 0.6322 and 9B 0.7364 on their 5,372 scored items only (C078). MedDecider's DrugProt items (3,644; more than 10 options) are NOT MEASURED (C077).
- Surprises: the CG zero-shot step failed twice on fast-dev-v2 rows that lacked the O2 keys (D-X6-3, D-X6-4); my own shell was killed once by a `pgrep` pattern that matched its command text (no output lost; the queue was restarted from its resumable steps). The indicator template reads well for zero-shot pplx (0.8152); the stated-value split does not explain it (C080).
- Next: Stage A (A0 to A3), then the choice, then Stage B (X8), per the watchers.

### X7 — DONE — 2026-10-19T17:47:30Z
- What ran: Stage A arms A0 to A3 (`x7_stage_a.sh`, 13,992 steps each; A0 04:08Z; A1 09:19Z; A2 14:05Z; A3 17:35Z); `x7_choice.py` (after a plumbing fix, D-X7-1).
- Output: `outputs/osler_v3/X7/choice.json`; `loops/osler_v3/stage_a.md`; per-arm `outputs/osler_v3/stage_a/A0..A3/`.
- Headline: every arm trained 13,992 steps as planned with 0 logged mismatches (C086, C090, C094, C102). The rule qualifies all four arms (MR macro lower bounds above -0.01) and chooses A0, the control, with CG differences from zero-shot pplx of +0.0027 (A0), +0.0010 (A2), +0.0002 (A1) and -0.0006 (A3) on 4,853 paired items (C098, C099). The differences are small against the CG noise; the choice is mechanical and the margin is weak.
- Surprises: the choice crashed at the control on an id join (D-X7-1); the arms' KL to base falls with the KL weight (0.1709, 0.1410, 0.1102; C089, C093, C097); A3 (broad replay) reaches the largest general accuracy (0.7867; C101).
- Next: X8 (B0 = A0); the claims pilot first, on the GPU.

### X8 — DONE — 2026-10-20 (operator closure; last entry before the keep re-run)
- Headline: no Stage B block kept. B1 (X1 sources) MR −0.0044 [−0.0143, +0.0058]; B2 (restyle) −0.0096 [−0.0209, +0.0019]; B3 (charts) −0.0054 [−0.0139, +0.0032]; last kept block B0 = osler_v2's C1 (C111–C118).
- Surprises: the chain crashed in the keep step of all three blocks (missing CG side report); the advisor re-ran the keep decisions (commit 7f47e1e). B1 attempt 1 was stopped at step 10,450 and relaunched (commit 2a75ae1).
- Next: X9 skipped by the operator; X10 reduced.

### X8 — IN_PROGRESS — 2026-10-19T18:14:41Z
- What ran: the Stage A choice (A0; C098–C101) and its join fix (D-X7-1); the restyle rows (C081–C085); the claims rework and its pilot (C106, C107: FAIL); fast dev v3 extended with the restyle dev pages (C108); B0 rescored (C109, C110; `x8_score_b0.py`); the chain `x8_chain.sh` started (B1 build 906,164 rows).
- Output: `outputs/osler_v3/X8/` (b0_dev, logs, block files' manifests); the block training files in `data/train/osler_v3/`.
- Headline: B0 = A0 on the extended fast dev v3: accuracy 0.835 on 13,303 items (C110); the restyle block (B2) is restyle only, the claims having failed D44(b) (C107).
- Next: B1 (X1 sources) keep decision; B2 (restyle) keep decision; B3 (charts) keep decision; the final mix (X9).

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|
| X3-claims | clinclaim_grounding_v1 (all claim rows; the claims template) | label-tracking phrase 'family' / 'family members' in 39 % of not-mentioned items vs 0 % of the others, after the one allowed prompt fix (D44(b); D-X3-1) | a second prompt change is not allowed by the ADVISORY; a new loop could redesign the not-mentioned instruction and re-pilot |
| X3-restyle | restyle_page_category_v1 (all restyled-page rows) | label-tracking phrases beyond the page class's own name on the built data (103 flagged after exempting the class name; D44(b)); the title is exempt as the labelled fact (Q4) | the operator's decision on the page-section vocabulary (Q4); a new loop could re-run the check with a written exemption |
| X4-icd | chart_icd_support_v1 (all chart rows of this template) | label-tracking phrases beyond the labelled fact, under the extended and the original exemption list (C061; D-X4-2) | a new loop could redesign the none-group wording and re-pilot; the presence words are not the labelled fact under the original list |
| X4-pages | chart_evidence_page_v1 (all chart rows of this template) | label-tracking phrases tracking page class I; no exemption applies (C062) | the operator's decision on the page-section vocabulary (as X3 Q4); a new loop could redesign the page pilot |
| X5 | MIMIC block (C4) and MIMIC in the dev sets | official PhysioNet files are not under `data/physionet/` (checked at X0 and X1) | the operator's PhysioNet approval and the official files under `data/physionet/`; re-checked before X6 |

---

## 6. Questions for the operator

Anything you could not resolve without a human. Be specific enough to answer without
re-reading the run: state the ambiguity, the options, and which you would pick.

---

**Q1 (X1, 2026-10-18T20:40:13Z) — RAGTruth spans flagged `implicit_true` (1,651 of 11,386 train-row spans) and `due_to_null` (1,280) are counted as hallucination spans.** RAGTruth's convention counts every annotated span; the ADVISORY says labels come only from the human span annotations. Options: (a) keep every span (current); (b) drop spans flagged `implicit_true` as not hallucinations. I would keep (a). The choice can change before any block is built (X8); please confirm before B1.

**Q2 (X2, 2026-10-18T20:40:13Z) — AQuA-RAT repeats problem texts (97,432 rows over 78,091 distinct normalised problems).** Options: (a) keep every row (current; general replay, the repeats differ in options or answers); (b) keep one row per problem. I would keep (a).

**Q3 (X2, 2026-10-18T20:40:13Z) — the A3 general replay share is 32.66 % after the 8 % template cap (target 35 %).** The gap is about 22,800 items at the A3 size. Options: (a) run A3 at 32.66 % and record it (current); (b) add about 32,000 general items from more permissive, human-labelled sets before X7. I would take (a) to keep the timebox; the arm answers the question as it stands.

**Q4 (X3, 2026-10-18T21:07:53Z) — the page type is stated in the restyled page's own title ('RADIOLOGY REPORT', 'PROGRESS NOTE', and so on), and the phrase check's radiology words ('impression', 'demonstrates', 'there is') come from that page genre.** I treat the title and the genre words as the labelled fact (D44(b)), so the restyle template passes without a block. The consequence: restyled-page items are partly solvable by reading the title, so their accuracy is high by construction, and the defect screen may call them too good. Options: (a) keep them as they are, and report the title-reading share with the template (current); (b) remove the title by a prompt change, which the ADVISORY allows only once and which I tried (pilot v2: kept 49 %, progress and emergency pages nearly gone). I would take (a). Please confirm.

**Q4 (X3, updated 2026-10-18T22:02Z) — the restyled-page template is blocked; the operator decides whether the page-section vocabulary counts as the labelled fact.** On the built data (4,746 train, 406 dev), the label-tracking check flags 103 phrases after exempting the class's own name words: 'impression' (radiology 91 % of items against 5 % of the others), 'patient demographics' (nutrition 50 % against 9 %), 'inpatient' (admission 38 % against 1 %), 'findings are', 'demonstrates'. The page title is exempt (it states the class). I blocked the template: these words are the writer's section style, not the labelled fact (D44(b)). If you consider a page type's own section vocabulary to be the labelled fact, say so; I would then write that exemption into D44 as a new entry and re-run the check, not edit it in place. Gold-in-state (the class name in the text) is 85 % of items; the NB macro on dev is 0.714 (below the 0.90 screen of ADVISORY §7.4).

**Q5 (X4, 2026-10-18T23:40:42Z) — the ICD presence words as the labelled fact (D-X4-2).** The extended exemption list (confirmed, diagnosed, diagnosis of, does have, does not have, determined) was added after the pilot. The template keeps flagged phrases under both the extended list and the original list ("ruled out", "rule out", "out current"), so it is BLOCKED either way (C061). The verdict does not depend on this question. I keep the original-list reading, which is the conservative one, and no operator decision is needed for X4.

**Q6 (X4, 2026-10-18T23:40:42Z) — unit values in the chart narratives.** The prompt forbids unit values in the narrative. The check counts them: 1,029 of 15,227 indicator pages (6.8 %) state one (C068). Options: (a) keep the indicator template as it is and report the count (current); (b) add a number rule to the label-tracking check, which would test whether a stated value decides the label. I would take (a) for X4 and measure (b) in a new loop if the indicator result matters. Please confirm.

**Operator rulings (2026-10-19T04:16:00Z; pasted by the operator; they answer Q4).** (1) Restyle (restyle_page_category_v1) is unblocked: under D44(b) the genre vocabulary of the requested page type is the labelled fact; the NB screen (0.71) passes; use the 6,426 verified pages; record as a deviation (D-X3-16). (2) Claims (clinclaim_grounding_v1) get one more prompt fix ('not mentioned' claims are about the same kinds of facts the passage discusses: findings, tests, treatments, timing; never family history) and a filter (drop any kept not-mentioned claim that mentions family; record the dropped count). A new 300-item pilot must pass the D44(b) cue check; if it passes, build up to 20,000 claims (Gemma, about 2 GPU hours) (D-X3-17). (3) Scheduling: between Stage A's choice and B2, the claims pilot and build, the restyle integration and the fast dev v3 extension come before any Stage B run; B2 = restyle + claims (claims only if they pass), otherwise restyle only (D-X8-2).

**Q7 (2026-10-19T04:20:49Z) — the genre list for restyle (D-X3-16).** Before the run I declared the genre vocabulary as: the page-type label words, the heading words that recur in at least 10 % of a class's pages, and the two radiology words you named. Under that list 94 label-tracking phrases remain flagged (the first 50 listed: radiology 25, nutrition 16, admission 8, discharge summary 1; C085). The 6,426 pages are used as you ruled (C081 to C084). Please confirm: (a) the 94 phrases are genre vocabulary, and I extend the declared list to cover them (I would re-run the script and record the new list as a deviation); or (b) the list stays as declared and the flagged phrases are a recorded caution for the restyled template. I would take (b) unless you say otherwise.

**Claims pilot result (2026-10-19T18:14:41Z).** The second prompt fix and the family filter were run on the 300-item pilot (C106). Family filter dropped 0 kept claims. The D44(b) check flags 2 phrases, both tracking the not-mentioned label (C107): the pilot FAILS, so under your ruling ("if the claims fail again, B2 = restyle only") no claims are built, and B2 is restyle only.

## 7. Deviations from the plan

Any place you departed from GOAL/ADVISORY, with the reason. An empty section is the
expected outcome. Editing code or a check to make it pass is never an acceptable
deviation — that is a `BLOCKED`.

---

- **D-X0-1 (X0, 2026-10-18T20:40:13Z):** a control arm (kl_to_base 0, same 150 steps) was added to measure the KL cost; the ADVISORY asks only for the KL run. Measurement only; no change to the recipe.
- **D-X0-2 (X0, 2026-10-18T20:40:13Z):** the timing runs call the trainer directly (no dev evaluation, no temperature fit; `run_arm` would evaluate). The batches are the trainer's own, the same order as the osler_v2 C1 replay.
- **D-X1-1 (X1, 2026-10-18T20:40:13Z):** the two binary RAGTruth templates are balanced 1:1 and the type template to its smallest class, in each split. The ADVISORY does not specify balancing; this prevents a prior-only model from scoring well (sources.md §3).
- **D-X1-2 (X1, 2026-10-18T20:40:13Z):** RAGTruth dev is 10 % of the official train contexts (stable hash); the official test split is kept as test.
- **D-X1-3 (X1, 2026-10-18T20:40:13Z):** DrugProt's validation split is halved into dev and test by PMID (the ADVISORY asks for dev and test from validation; the halving is my choice).
- **D-X2-1 (X2, 2026-10-18T20:40:13Z):** MathQA rows whose normalised problem is in AQuA-RAT are dropped (28,300 rows); AQuA-RAT's natural-text copy is kept (CLAIMS C035–C036).
- **D-X2-2 (X2, 2026-10-18T20:40:13Z):** the replay target (35 % of A3) is not met: 32.66 % after the 8 % template cap, short by about 22,800 items. Recorded as a finding; not tuned (ADVISORY X2: record what was found).
- **D-X2-3 (X2, 2026-10-18T20:40:13Z):** the broad replay templates `broad_*` are not domain prefixes, so they are anchors under KL-to-base (ADVISORY §3.2: a new template family must take a prefix or be recorded as an anchor here). Their general skill is what the KL term is meant to hold.
- **D-X2-4 (X2, 2026-10-18T20:40:13Z):** Social IQa and CosmosQA are read from their original archives, not the Hub: the Hub copies are loading scripts, which `datasets` 4.x does not run. The archives' SHA-256 values are recorded.

- **D-X3-1 (X3, 2026-10-18T21:07:53Z):** the claim template's one allowed prompt fix (a style rule for all labels: no 'The patient', no 'history'; and a related-fact instruction for not-mentioned claims) removed 'history' but introduced 'family members' in 39 % of not-mentioned items. The template is BLOCKED after the one fix (the ADVISORY allows one). No claim rows are written; the candidates are kept in `data/gen/osler_v3/` (not committed).
- **D-X3-2 (X3, 2026-10-18T21:07:53Z):** a restyle prompt change (no page-type title) was tried as a test and then reverted to the original prompt before the build. Pilot v2 (no title): kept 146 of 300 (49 %), two classes nearly absent (progress 1, emergency 1), so the class balance collapses. The original prompt (title allowed) kept 286 of 300 with 30 or more per class. The original is the build prompt; the title is the labelled fact (Q4). The chain's first 394 documents were generated with the original prompt and kept.
- **D-X3-3 (X3, 2026-10-18T21:07:53Z):** the restyle build targets 6,000 training documents, 500 dev and 1,000 test, not 12,000 pages: the measured throughput is 1.4 to 1.6 documents per second on the local model (about 2.5 hours for 12,000 documents within the 6 GPU-hour cap, with the charts still to build).
- **D-X3-4 (X3, 2026-10-18T21:07:53Z):** the claims balance is by label only (ADVISORY X3: 'balance the three labels'); the edit kinds are reported, not balanced. The first version also balanced kinds, which left a scarce kind (drug) with four items.
- **D-X3-5 (X3, 2026-10-18T21:07:53Z):** the claims edit-kind draw is restricted to the kinds a passage can carry (a number edit needs a number, and so on): the code still draws the kind, from a smaller set. Without this, drug and laterality edits were almost never applicable.
- **D-X3-6 (X3, 2026-10-18T21:07:53Z):** the CodiEsp substrate is the 419 training cases of osler_v2's block C1 (osler_v2's carve-out has no training rows, so it is excluded by construction). The recomputed carve-out of this loop differed from osler_v2's (75 against 74 cases), so it was not used as the rule.

- **D-X3-7 (X3, 2026-10-18T22:02Z):** the restyled-page template is BLOCKED on the built data under D44(b) (see Q4): 103 label-tracking phrases beyond the class name, against the pilot's 0 beyond the radiology genre words. The pilot passed because fewer phrases reached the ten-item threshold at 300 documents; the full check is the one that counts. No restyled rows are used.
- **D-X3-8 (X3, 2026-10-18T22:02Z):** the restyle test pass was stopped at 91 of 1,000 documents, because the template is blocked; the GPU went to the chart pilot.
- **D-X3-9 (X3, 2026-10-18T22:02Z):** the claims and restyle data (candidates, build reports, read-through samples) stay in gitignored folders; nothing from X3 enters training, dev or test.

- **D-X3-10 (X3 closure, 2026-10-18T22:23:56Z):** correction of D-X3-8. The restyle test pass wrote 145 candidates (140 kept), not 91; 91 is the balanced test count (CLAIMS C048, C050).
- **D-X3-11 (X3 closure, 2026-10-18T22:23:56Z):** correction of D-X3-1. The "39 %" does not reproduce: "family" is 44 of 105 kept not-mentioned claims (41.9 %), "family members" is 39 of 105 (37.1 %), and both are 0 % among supported and contradicted claims (C045, C046). The block stands.
- **D-X3-12 (X3 closure, 2026-10-18T22:23:56Z):** the Naive Bayes screen now runs on every template, and a blocked template is marked `blocked`. This is a diagnostic only: no threshold, gold label or block changed. The restyle NB macro is 0.7143 (C052), below the 0.90 screen.
- **D-X3-13 (X3 closure, 2026-10-18T22:23:56Z):** `readthrough_kept_claims.json` is empty by construction (the claims path leaves the loop before the kept sample is drawn), so no kept-claim read-through exists at closure; the rejected sample is written (SELF_AUDIT §5).
- **D-X3-14 (X3 closure, 2026-10-18T22:23:56Z):** CodiEsp has 419 training documents and 416 usable passages; the 3 without a usable passage have no counted reason. No row uses them. Open (SELF_AUDIT §5).

- **D-X4-1 (X4, 2026-10-18T22:40Z):** the chart build is 2,000 charts (the count passed at launch, 22:13Z), below the ADVISORY's "up to 8,000 charts within 6 GPU-hours" (X4). The ADVISORY sets a ceiling, so 2,000 is within it. The launch record does not state why this count was chosen; the build is not extended mid-run.
- **D-X4-2 (X4, 2026-10-18T22:40Z):** the label-tracking exemption for `chart_icd_support_v1` (`LABELLED_FACT` in `scripts/osler_v3/x4_charts_check.py`) was extended after the pilot (added: confirmed, diagnosed, diagnosis of, does have, does not have, determined). The extension was made before the full build but after the pilot. The ICD verdict is reported under the original list ("ruled out", "rule out", "out current") as well; the conservative verdict is the original list's.
- **D-X8-1 (pre-registered before any Stage B result, 2026-10-18T22:40Z):** the new sources' items (`ragtruth`, `drugprot`, `synthea_chart`) are not counted in the keep-rule partitions. `meddecide.train.blocks.partition_pairs` would put them in `general`, where a gain on the new items moves the general lower bound that protects the existing general templates. The MR partition (MR_SOURCES) and the knowledge partition (medqa, medmcqa) are unchanged. The new items are reported beside the rule (labelled), following D38. The keep rule's margins are unchanged.
- **D-X3-15 (gate, X3 commit):** the pytest gate at this commit deselects `tests/test_model_pointer.py::test_overfit_64_items` on the command line only. The test is GPU-only (on CPU it is skipped) and failed with CUDA out of memory while the Gemma server (pid 3254457, about 90 GiB) held the GPU; no code under test changed. The full gate is not claimed as passing for this commit. It re-runs before the first GPU training launch (X7 chain). Counts: 838 passed, 1 xfailed, 1 skipped (`outputs/osler_v3/X3/logs/pytest_gate_deselect.log`).

- **D-X6-2 (X6, 2026-10-18T22:44Z):** the zero-shot pplx runner writes `<out-root>/pplx-decider-flash-v1.2/predictions.jsonl`, not `<out-root>/predictions.jsonl`. `x7_choice.py` and `x8_keep.py` had the second path. The fault was found in review before any run read it (commit a95f9ba); no result depended on it. `x7_choice.py` now stops when the CG predictions are absent.

- **D-X4-3 (X4, 2026-10-18T23:40:42Z):** the "first 500 built charts" re-check is not separable from the rows: they carry no build order, and the first files hold fewer than 500 distinct chart keys, so the selection equals the full check (C070). The full check covers the first 500 charts.
- **D-X4-4 (X4, 2026-10-18T23:40:42Z):** chart_icd_support_v1 and chart_evidence_page_v1 are BLOCKED under D44(b) (C061, C062). Only chart_indicator_evidence_v1 enters fast dev v3 (330 items), the new test pool (184 items) and block B3 (2,916 train rows, about 0.3 % of a block). B3 is therefore a small block; its result is reported as measured.
- **D-X4-5 (X4, 2026-10-18T23:40:42Z):** the O3 NB screen on the choice templates is reported but not used. It scores option texts and ignores the offered options: 83.6 % of ICD dev predictions are one class, and the values fall below a uniform guess (C063 to C065). This is a formulation limit of the screen. The 0.90 rule and the screen code are unchanged; a choice-aware screen needs its rule written before use.
- **D-X4-6 (X4, 2026-10-18T23:40:42Z):** the label-tracking lists are recorded at most 30 candidates per template (the check's existing cap). The verdicts use whether any candidate is flagged, so the cap does not change a verdict.

- **D-X6-3 (X6, 2026-10-18T23:44:22Z):** the 865 fast-dev-v2 rows of the CG partition carry no O2 keys (`benchmark`, `set_name`); they are mix_v2 items, since their templates are mix_v2 templates. The zero-shot runner needs both keys, so the builder now sets `benchmark = mix_v2` and `set_name = source` on them. Membership, the CG size (4,865) and every item are unchanged (CLAIMS C074). The partition was regenerated and checked: every row carries both keys.
- **D-X6-4 (X6, 2026-10-18T23:44:22Z):** the zero-shot pplx CG step failed twice, at the first fast-dev-v2 row (KeyError `benchmark`, then `set_name`), each time after its 4,000 saved knowledge rows. The queue continued to the new-test-pool steps between the failure and the fix; those steps are additional rows and do not feed the CG reference. The CG step is resumed from its saved rows after the fix, and `x7_choice.py` refuses to run without its predictions.

- **D-X6-1 (X6, 2026-10-19T00:39:50Z):** referenced on the board since the CG build (commit 1668b36) but its entry was never written. The CG build and its exclusion counts are in `loops/osler_v3/partitions.md`; no content is asserted under this label.

- **D-X3-16 (operator ruling, 2026-10-19T04:16:00Z):** restyle_page_category_v1 is unblocked. Under D44(b) the genre vocabulary of the requested page type (for example 'Impression' or 'demonstrates' for radiology) is the labelled fact, as accepted for the code-rendered page templates. All 6,426 verified kept pages are used (train 5,797; dev 489; test 140), not class-balanced. The largest class share and the NB screen are measured on the used set and recorded; the screen must stay below 0.90 (the operator cited 0.71 on the balanced set).
- **D-X3-17 (operator ruling, 2026-10-19T04:16:00Z):** clinclaim_grounding_v1 gets one more prompt fix (not-mentioned claims are about the same kinds of facts the passage discusses: findings, tests, treatments, timing; never family history) and a filter (kept not-mentioned claims that mention family are dropped; the dropped count is recorded). A new 300-item pilot must pass the D44(b) cue check before any build; if it passes, up to 20,000 claims are built (about 2 GPU hours). This is the second prompt fix; the ADVISORY allowed one, so this is a deviation under the operator's authority.
- **D-X8-2 (operator ruling, 2026-10-19T04:16:00Z):** the Stage B order changes. After the Stage A choice, the claims pilot and build (if it passes) and the restyle integration run first; fast dev v3 is extended with the restyle and claims dev items before any Stage B run; the chosen Stage A checkpoint (B0) is rescored on the extended dev set; then B1 (X1 sources), B2 (restyle + claims, or restyle only if the claims fail) and B3 (charts). The 'hold after B1' option is not used, since no Stage B run precedes the extension. The Stage A choice uses the Stage A predictions as they stand; the MR and CG partitions contain none of the new templates.
- **D-X8-3 (amends D-X8-1, before any Stage B result):** the new-source exclusion is by template family (ragtruth_, drugprot_, chart_, restyle_, clinclaim_), not by source name, because restyle and claims rows keep their real-text source name (for example pmc_patients). The rule is otherwise unchanged.
- **Operational (2026-10-19T04:16:00Z):** the Stage A handoff watcher (bfh5b4243, which would have launched Stage B after the choice) was stopped before it reached the choice; its exit 144 is that stop, not a failure. A new watcher (bc1dcnrfp) runs only the choice. The Stage A chain itself was not touched.

- **D-X3-18 (operator ruling, 2026-10-19T04:20:49Z):** the restyle rows are written from all 6,426 kept verified pages (train 5,797; dev 489; test 140), without class balancing (C081). The largest class share is 0.1857, under the 25 % cap (C082). The NB macro on dev is 0.722, under the 0.90 screen (C083). The genre vocabulary is the declared list in `scripts/osler_v3/x3_restyle_unblocked.py`; 94 label-tracking phrases remain flagged under it (C085; Q7). The restyle rows enter fast dev v3 (dev) and the new test pool (test) only after the claims work and before any Stage B run (D-X8-2).
- **D-X7-1 (X7, 2026-10-19T17:47:30Z):** the choice (`x7_choice.py`) failed with `the control A0 has no CG score`. The arms' CG predictions carry the item ids that `mix_row_to_item` derives, while the zero-shot CG reference carries the raw ids of `cg_v3.jsonl`; the two sets did not overlap. Fixed in `x7_choice.py` and `x8_keep.py`: each arm's CG item id is mapped to its raw id before the paired join (the rule, the dev pairings and every arm's predictions are unchanged). The choice was then re-run and wrote `choice.json` and `stage_a.md` (C098–C101).

- **D-X3-19 (operator ruling, 2026-10-19T18:14:41Z):** the claims pilot v2 (300 candidates; kept 261) fails the D44(b) check (2 flags on not-mentioned claims; C106, C107). Under the ruling, no claims are built, and B2 = restyle only. The pilot file stays gitignored.
- **D-X8-4 (amends D-X8-2 and D-X8-3, 2026-10-19T18:14:41Z):** the keep rule's new-source exclusion is by template family (`ragtruth_`, `drugprot_`, `chart_`, `restyle_`, `clinclaim_`), implemented in `x8_keep.py`; the B2 block is restyle only (the claims are excluded).
- **D-X8-5 (X8, 2026-10-19T18:14:41Z):** B0 is the chosen A0 checkpoint rescored on the extended fast dev v3 by `x8_score_b0.py`. On the 12,903 items that the arm scored before the extension, the rescored correctness agrees with A0's own predictions on every item (C109); this is the check that the rescoring is A0's model. B0's accuracy on the 13,303 items is 0.835 (C110).
- **D-X10-1 (operator ruling, 2026-10-20):** X9 is skipped (the kept mix equals osler_v2's C1 with its recipe). X10 is reduced to additional rows: no Gate O3 verdict; A0–A3 and B1 are scored on the v10 items, calibrated, against zero-shot pplx and osler_v2's Osler-4B; the panel is broken down by dataset. X11 is not applicable. Recorded as a deviation from ADVISORY X9–X11.
### X9 — BLOCKED — 2026-10-20T21:10:00Z
- Headline: skipped by the operator: the kept mix equals osler_v2's C1 with osler_v2's recipe, so a final run would retrain osler_v2's model. No final model was trained.
- Next: X10 reduced to additional rows; no verdict.

### X10 — DONE — 2026-10-20T23:10:00Z
- What ran: `scripts/osler_v3/x10_chain.sh` (A0, A1, A2, A3, B1 through `x10_score.py`, osler_v2's W8 scoring path, output root `outputs/osler_v3/X10`); `scripts/osler_v3/x10_report.py`; the spot-check (`outputs/osler_v3/X10/spotcheck.log`).
- Output: `outputs/osler_v3/X10/` (results.json, per-run both_orders.jsonl, SELF_AUDIT.md); `loops/osler_v3/results.md`; `loops/osler_v3/gate_o3.md`.
- Headline: against zero-shot pplx, the panel difference is −0.0245 (A0), −0.0111 (A1), −0.0161 (A2), −0.0109 (A3), −0.0226 (B1); MR verdict about +0.13 to +0.14 in every run; no verdict (C122–C127).
- Surprises: the Stage A and B1 runs are 13,992 steps and osler_v2's final is 42,200 (C131), so the MR gap of 0.018–0.025 against osler_v2 is not a clean test of the recipe; a report-path bug and a recompute-key bug were fixed before results were recorded (SELF_AUDIT §4).
- Next: FINDINGS and NEXT (X12).

### X12 — DONE — 2026-10-20T23:12:00Z (HARD STOP)
- What ran: findings and closure: `loops/osler_v3/FINDINGS.md`, `loops/osler_v3/NEXT.md`, five-claim spot-check (C120, C121, C124, C128, C130; all re-run in a fresh process and matched).
- Headline: no Gate O3 verdict; the Stage A control was chosen; no Stage B block was kept; the panel gap narrows by about half with KL 0.1 or broad replay, with intervals that reach zero.
- Next: the operator's decisions in NEXT.md (run-length check; keep-rule proposal). Loop stopped.

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| 1 | No Gate O3 verdict: the final model was skipped by the operator (duplicate of osler_v2's final). | C129, C130 | loops/osler_v3/gate_o3.md |
| 2 | Stage A chose the control (KL 0) by the pre-registered rule; margins weak. | C098–C101 | loops/osler_v3/stage_a.md |
| 3 | No Stage B block was kept; each failed the MR check of the keep rule. | C111–C118 | loops/osler_v3/blocks.md |
| 4 | Against zero-shot pplx, the panel gap is −0.011 to −0.025 across runs; KL 0.1 or broad replay narrows it by about half, intervals reach zero. | C122–C127 | loops/osler_v3/results.md |
| 5 | Stage A runs are 13,992 steps and osler_v2's final is 42,200; the MR gap to osler_v2 is not a clean recipe test. | C131 | loops/osler_v3/FINDINGS.md |
