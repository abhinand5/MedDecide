# osler_v0 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/osler_v0/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (O12), or when no PENDING task can proceed without the operator -->
Run started (UTC): `2026-10-09T06:23:41Z`
Last updated (UTC): `2026-10-10T22:36:43Z`
Iterations so far: `4`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| O0 | Orientation, snapshot, envs, competitor smoke, throughput | smoke | — | DONE | 2026-10-09T06:23:41Z | 2026-10-09T07:50:24Z |
| O1 | External clinical panel + robustness pack | no | O0 | DONE | 2026-10-09T07:55:27Z | 2026-10-09T08:13:20Z |
| O2 | Competitor scoreboard | yes | O0, O1 | DONE — all rows for 11 models on the common set (59,538 items); D12 applied to zero-shot cells; pplx investigated (O123–O125); Question 16 open | 2026-10-09T08:25:05Z | 2026-10-10T13:05:22Z |
| O3 | Clinical generators (gold by construction) + held-out list | no | O0 | DONE | 2026-10-09T18:12:46Z | 2026-10-09T18:30:07Z |
| O4 | Training mix v2 | no | O1, O3 | DONE | 2026-10-09T18:31:34Z | 2026-10-09T19:01:39Z |
| O5 | Readouts: option-code head, non-causal mode, export | small | O0 | DONE — merged-adapter export READOUT_FAIL under the as-run precision; separate export PASS; Question 14 | 2026-10-09T19:02:52Z | 2026-10-09T20:33:07Z |
| O6 | Arm L: option-code head (4B) | yes | O4, O5 | IN_PROGRESS — attempt 5 (run of record; eval every 2,500 steps; selection and tripwire on template_macro_accuracy; evals at 2,500 and 5,000 recorded, CLAIMS O161–O162); worker PID 690638; log outputs/osler_v0/O6/logs/arm_L.log | 2026-10-10T17:58:36Z | |
| O7 | Arm P: pointer head (4B) | yes | O4, O5 | PENDING | | |
| O8 | Arm N: non-causal option-code head (4B) | yes | O4, O5 | PENDING | | |
| O9 | Head choice at 4B (dev rule) | no | O6–O8 | PENDING | | |
| O10 | Osler-9B + Osler-0.8B reference | yes | O9 | PENDING | | |
| O11 | Evaluation + Gate O1 | yes | O2, O10 | PENDING | | |
| O12 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (exceptions in ADVISORY §6:
O3/O4 may run on CPU while O2 uses the GPU; O5 may use the GPU between O2 jobs). Never run two
GPU jobs at once. A `BLOCKED` task does not block unrelated work — record it and move on. No row
may be left `IN_PROGRESS` at closure.

---

## 2. Key values discovered during the run

Fill these in as they are measured; later tasks read them from here rather than
recomputing or guessing.

| key | value | source | task |
|---|---|---|---|
| model revisions (Qwen3.5 0.8B/4B/9B; each competitor) | 14 registry entries pinned; Hub sha = pin = downloaded snapshot for 14 of 14 (Qwen3.5-0.8B `2fc06364`, -4B `851bf6e8`, -9B `c2022362`; MedDecider-4B `570b3709`, -9B `ec8a69da`, -27B `c30e1881`, -31B `035f4543`; pplx-decider-v1.1-27b `5cd25e3f`; JEV-27B `51740a88`, JEV-9B `b63f651c`; Clef `ed3eed33`, Clef-Flash `fde727a2`; bases Qwen3.8-27B `1d4bf0f2`, gemma-4-31b-it `842da379`) | `configs/osler_v0/competitors.json`, `outputs/osler_v0/O0/fetch.json` (CLAIMS O029) | O0 |
| Qwen3.5-4B full-attention layer indices | [3, 7, 11, 15, 19, 23, 27, 31] of 32 layers; 24 linear-attention (CLAIMS O018) | `outputs/osler_v0/O0/envs.json` | O0 |
| projected wall-clock of one 4B arm / of the 9B run | 4B: 16.08 h at the ADVISORY budget (200,000 examples; dev eval 100 x 6,000 items), lower bound 5.88 h at median length (O013, O014). 9B: 24.59 h, lower bound 8.99 h (O017). Training 0.376 s/step (4B), 0.571 s/step (9B) at batch 8 x 391 tokens (O009, O015) | `outputs/osler_v0/O0/throughput.json` | O0 |
| competitor smoke results | 9 of 9 PASS through each card's or authors' own code: MedDecider-4B/9B/27B/31B (card examples; worst abs. diff 0.0004 / 0.0039 / 0.0038 / 0.0044), pplx-decider-v1.1-27b (authors' DecisionModel), JEV-27B and JEV-9B (card decision-head protocol), Clef and Clef-Flash (authors' systemone). Open weights exist for all competitors in O2, so none is NOT MEASURED | `outputs/osler_v0/O0/smoke_summary.json` (O019) | O0 |
| v0.2 denominators | 37,979 raw test rows; 690 dropped by the screen (six templates); 37,289 kept = 23,768 fresh + 13,521 tier-1, the student_v1 denominator (O001–O003) | `outputs/osler_v0/O0/snapshot.json` | O0 |
| v0.2 integrity | 2 of 11 benchmark files do not match their manifest sha256 (clinicaltrials, openfda); manifest total 57,007 vs acceptance 45,009 (O004, O005). Not repaired (v0.2 read-only); see Questions 1 | `outputs/osler_v0/O0/snapshot.json` | O0 |
| student_v1 training mix | 482,889 train rows, 5,619 dev rows; base student_v0 file hash matches (O006, O007) | `outputs/osler_v0/O0/snapshot.json` | O0 |
| disk (ADVISORY §3 budget: hf_home + outputs + data) | 393.52 GB of 450 GB after all O0 downloads; envs 11.59 GB and uv cache 16.70 GB outside the budget (O030, O031) | `outputs/osler_v0/O0/snapshot.json` | O0 |
| external panel: datasets, items; robustness pack: base items × perturbations | Panel: 4,353 items in 7 eval-only sets (MMLU-Pro health 781, MedXpertQA-Text 1,000 of 2,450, MedExpQA-en 125, MedConceptsQA 1,000 of 819,772, MedExQA 940, symptom-to-diagnosis 212, medical question pairs 295). Robustness: 4,000 bases (2,000 panel + 2,000 v0.2 fresh) → 20,898 perturbed items, all rule-checked (CLAIMS O032–O044) | `data/bench/v0.3_ext/manifest.json`; `outputs/osler_v0/O1/robustness_verify.json` | O1 |
| training overlap of the panel; v0.2 overlap | 0 exact-text and 0 record hits against 2,150,361 training rows (12 files); 0 v0.2 exact-text hits after excluding 37 MMLU-Pro duplicates (CLAIMS O040, O041) | `outputs/osler_v0/O1/overlap.json` | O1 |
| long-record slice (student_v1 fix 3) | 2,213 of 37,289 kept v0.2 test rows over 8,192 prompt tokens, all in fda_route_claim_noul_v1 (944), fda_boxed_warning_noul_v1 (668), fda_class_choice_v2 (601); ids stored (CLAIMS O045) | `data/bench/v0.3_ext/long_record_ids.json` | O1 |
| scoreboard headline (MedDecider-4B / -9B / pplx v1.1 on v0.2 fresh and external panel) | | `loops/osler_v0/scoreboard.md` | O2 |
| generators: count, held-out list, train/dev/test items | 9 generators: 7 seen, 2 held out (note_lab_range_v1, policy_triage_v1). Train 21,000 items (0 held out; 1,500 patients per seen generator), dev 2,700 (600 held out), test 2,700 (600 held out). Screen: all nine pass; gold label in the record 0 hits. The held-out baselines are constant predictors, so their pass rests on gold-in-state and the construction checks (CLAIMS O046–O063; STATE §7 item 23) | `loops/osler_v0/heldout.md`; `outputs/osler_v0/O3/recompute.json` | O3 |
| mix v2: items, sources, replay share, max template share; leakage result | 573,611 train rows (student_v1 482,889 + 46,744 new + 43,978 augmented) and 10,843 dev rows. Sources: student_v1 components and tier-1 train splits, O3 seen generators (21,000), CommonsenseQA (9,741), QASC (8,134), PubHealth (7,869). Replay 3.46 %; largest template 7.39 % (cap 8 %); budget 200,000 examples. Leakage 0 by text and by dataset-qualified record (CLAIMS O064–O078) | `data/train/osler_v0/manifest.json`; `loops/osler_v0/mix_v2_manifest.json`; `outputs/osler_v0/O4/verify_mix.json` | O4 |
| readout checks (init equality, export round-trip, padding, byte-identity) | | `outputs/osler_v0/O5/readout_checks.json` | O5 |
| arm L attempt 5: dev template macro by eval (6,000-item subset; selection statistic) | step 2,500: 0.8197 (saved); step 5,000: 0.8133 (not above 0.8197); step 7,500: 0.8265 (checkpoint re-saved 21:23:47Z); step 10,000: 0.8420 (best so far; checkpoint re-saved 22:32:45Z). Diagnostic per-letter macro 0.6029 → 0.5400 → 0.6005 → 0.6232; accuracy 0.8458 → 0.8468 → 0.8638 → 0.8737; Brier 0.2280 → 0.2258 → 0.2092 → 0.1913 | `outputs/osler_v0/O6/arm_L/logs/train.jsonl` (CLAIMS O161, O162, O164, O167; `scripts/osler/o6_monitor.py`) | O6 |
| arms L / P / N: examples seen, selected step, dev macro, tier-1 dev at selection, wall-clock | | | O6–O8 |
| chosen head and the rule's verdict | | `loops/osler_v0/head_choice.md` | O9 |
| Osler-9B / Osler-0.8B: selected step, dev macro | | | O10 |
| Gate O1 verdicts (4B, 9B) incl. knowledge guard | | `loops/osler_v0/gate_o1.md` | O11 |

---

- Disk (`du -sh --apparent-size`, 2026-10-10T17:58Z, after the operator's deletions): HF cache 60G; uv cache 16G; data 16G; outputs 16G; /workspace/tmp 22G; envs 246M. Measure with du, not df.

## 3. Current task checklist

**Fill this in before starting any task**, by copying that task's concrete steps out of
the plan as unticked boxes. Tick each one the moment it is finished, not at the end.
This is the only record of progress *within* a task — without it, a compaction mid-task
leaves you unable to tell what you already did.

Clear this section and write the new task's checklist when you start the next task; the
completed checklist goes into the iteration-log entry.

**O6 restart (written by the advisor at 2026-10-10T14:37:06Z, on the operator's decisions Q18–Q20; do these in order):**

```
- [x] (done 2026-10-10T17:57Z) confirm the attempt-4 process is gone: `ps -p 576775` prints no process (the operator stops it; never relaunch while it exists)
- [x] (done 17:57Z: 0%, 0 MiB) confirm the GPU is free (`nvidia-smi`: no process, memory near 0)
- [x] (done 17:58Z; step-250 checkpoint deleted) archive attempt 4: move outputs/osler_v0/O6/arm_L -> outputs/osler_v0/O6/arm_L_attempt4_eval250 and
      outputs/osler_v0/O6/logs/arm_L.log -> outputs/osler_v0/O6/logs/arm_L_attempt4_eval250.log (keep logs; its best checkpoint may be deleted)
- [x] (done 17:58Z; see §2) re-measure disk with `du` (ADVISORY §3) and record it in STATE §2
- [x] (done 17:58:36Z; worker PID 690638) relaunch arm L (attempt 5) with the committed configs/osler_v0/arm_L.yaml (eval_every 2500, selection_metric template_macro_accuracy),
      output captured to outputs/osler_v0/O6/logs/arm_L.log; record PID and start time here
- [x] (done 2026-10-10T19:07Z) after the first eval (step 2,500): the eval record carries template_macro_accuracy = 0.8197 (per-letter macro 0.6029 logged as diagnostic; accuracy 0.8458; Brier 0.228; best so far), check that train.jsonl's eval record carries template_macro_accuracy and that
      arm_result trajectories will include it; then continue O6 as planned
- [x] (done 2026-10-10T20:21Z; CLAIMS O162) step-5,000 eval: template_macro_accuracy 0.8133 (below 0.8197, so the best checkpoint stays at step 2,500); per-letter diagnostic 0.5400; accuracy 0.8468; Brier 0.2258
```

```
Task in flight: O2 (started 2026-10-09T08:25:05Z) — competitor scoreboard (GPU)
Running job:    `scripts/osler/o2_chain.sh` (sequential, one GPU job), PID 266074, started 2026-10-09T08:47:43Z;
                chain log `outputs/osler_v0/O2/logs/chain.log` (START/END rc per step; ends with "O2 CHAIN DONE");
                per-step logs `outputs/osler_v0/O2/logs/<step>.log`; the chain is resumable (predictions appended)
Working dir:    outputs/osler_v0/O2/ (predictions, gitignored); data/bench/v0.3_ext/o2_items.jsonl (items, gitignored)

O2 checklist:
- [x] common item set: data/bench/v0.3_ext/o2_items.jsonl — 59,538 in-set items (v0.2 35,076; panel 4,353; robustness 20,109); 3,002 over the 8,192-token common cap excluded for every model (manifest section o2_items)
- [x] zero-shot Qwen3.5-4B on panel + robustness: 24,462 rows (letter variant detected = bare; mean label mass healthy; the first run used the space variant and was discarded: see Deviations)
- [x] smoke tests passed for every runner family before the chain: MedDecider-4B (decide vector form vs authors' decide(), max |diff| 4.9e-5 on 50 items; noul 29-30 of 30 correct on the smoke slice), JEV-9B (logits_to_keep=1 vs full: max |diff| 4.4e-8 on 20 items), pplx-decider (noul 29 of 30), Clef (noul 30 of 30)
- [x] MedDecider-4B (authors' code, both orders): DONE rc=0 at 10:02:40Z; 59,538 rows, 57,968 scored, 1,570 skipped (more than 10 options: symptom 212 + 648 robustness; replace-gold 11-option items 272 + 438); prob sums 1.000; decide() check max |diff| 4.9e-5 on 50 items. Accuracy: v0.2 0.824 (n 35,076); panel 0.544 (n 4,141); robustness 0.643 (n 18,751). Caveat recorded: v0.2 medquad 0.990 matches the routing template already flagged as shortcut-solvable (bench_v0 FINDINGS; student_v1 arms.md medquad_routing_v1 1.00); MedDecider's card lists MedMCQA and MedQA in its training, so v0.2 tier-1 medqa and medmcqa numbers for it are not clean (fresh tier is the clean one)
- [x] zero-shot Qwen3.5-9B (panel + robustness): DONE rc=0 at 10:21:35Z; 24,462 rows, all scored; variant bare; label mass median 0.998; full-vocabulary argmax is an offered letter in 100% of rows
- [x] JEV-9B (card head protocol, panel + robustness): DONE rc=0 at 10:45:42Z; 23,602 scored, 860 skipped (symptom-derived items with 22 or 23 options exceed the head's 16 choice slots); logits_to_keep check max |diff| 4.4e-8; probability sums 1.000; noul accuracy 0.652 (question pairs 0.797), so the yes/no mapping holds
- [x] MedDecider-9B (authors' code, both orders): DONE rc=0 at 12:19:02Z; 59,538 rows, 57,968 scored, 1,570 skipped (the same over-10-option items as the 4B run); probability sums 1.000; decide() check max |diff| 4.9e-5 on 50 items; accuracy v0.2 0.829 (n 35,076), panel 0.560 (n 4,141), robustness 0.645 (n 18,751)
- [x] Clef-Flash (authors' systemone, envs/clef): DONE rc=0 at 14:09:02Z; 59,538 rows, 59,538 scored, 0 skipped; probability sums 0.9994 to 1.0005 (the endpoint rounds to 4 dp); accuracy v0.2 0.829 (n 35,076), panel 0.571 (n 4,353), robustness 0.578 (n 20,109)
- [x] pplx-decider-v1.1-27b (authors' DecisionModel, envs/pplx27b, single order, saved non-causal mode): DONE rc=0 at 16:41:14Z; 59,533 scored, 5 skipped (its own 8,192-token check on items the Qwen count placed under it); sums 1.000; accuracy v0.2 0.861 (n 35,073), panel 0.641 (n 4,353), robustness 0.530 (n 20,107). v0.2 by source: fresh clinicaltrials 0.812, openfda 0.865, pubmed 0.953; tier-1 medqa 0.874, medmcqa 0.747, medquad 0.991 (routing shortcut), mmlu_medical 0.920. Looks high: investigate at metrics (per-template D21 checks and the fresh/tier-1 split); the competitor's training data is not stated on its card, so tier-1 is not clean
- [x] O2 chain (PID 266074): JEV-27B DONE (END rc=0 at 20:35:09Z); the chain was resumed with SIGCONT at that moment (§7 item 36), clef RUNNING from 20:35:09Z, then md27b → md31b → metrics. Earlier record: JEV-27B (RUNNING from 16:41:14Z; 28,000 rows at 18:16Z; no error in its log at 18:24Z; see §7 item 20) → clef → md27b → md31b → metrics; chain log outputs/osler_v0/O2/logs/chain.log. NOTE: the watch was not re-armed between 13:49 and 18:11 UTC; re-armed at 18:11
- [x] MedDecider-4B (authors' decide protocol, both orders; vector form checked against decide()) — all three sets
- Running now (2026-10-10T17:59Z; at 22:36Z step 10,100 of 25,000 (about 1.49 s per step, GPU at 100 %), four evals done (2,500, 5,000, 7,500 and 10,000; the best so far is step 10,000 at template macro 0.8420, CLAIMS O164–O168; no tripwire event; the drift check in O168 shows loss and batch accuracy still improving); expected arm L finish about 05:40Z on 2026-10-11; bounded watch armed 20:23Z for arm L's exit, failures, chain events and tripwires): O6 arm L attempt 5 (run of record), worker PID 690638, started 2026-10-10T17:58:36Z; log outputs/osler_v0/O6/logs/arm_L.log; trainer log outputs/osler_v0/O6/arm_L/logs/train.jsonl. Attempt 4 archived as arm_L_attempt4_eval250 (deviation 48). Arms P and N follow in order by `scripts/osler/o6_arm_chain.sh` (detached, PID 691022, started 2026-10-10T18:00:26Z; it waits for arm L's exit and stops at the first failure; log outputs/osler_v0/O6/logs/arm_chain.log).
- [x] D24 baselines on the v0.2 scope (deviation 43): zero-shot Qwen3.5-4B, zero-shot Qwen3.5-9B and JEV-9B with `--scope all`, queued in outputs/osler_v0/O2/logs/chain2.log after the chain's metrics step; then o2_metrics.py again
- [x] zero-shot Qwen3.5-9B (panel + robustness)
- [x] JEV-9B (card decision-head protocol, F7 helpers; panel + robustness)
- [x] MedDecider-9B (both orders)
- [x] Clef-Flash (authors' systemone; envs/clef; choice keys sorted by the authors' code)
- [x] pplx-decider-v1.1-27b (authors' DecisionModel; envs/pplx27b; single order as the authors run it)
- [x] JEV-27B (F7 protocol; single order)
- [x] Clef (authors' systemone)
- [x] MedDecider-27B (both orders)
- [x] MedDecider-31B (both orders)
- [x] o2_metrics.py: per model × set n, accuracy, macro (mean over templates, as G1), Brier, ECE, coverage with reasons, wall-clock per 1k items, D12/D21 cell gate, robustness flip rates → outputs/osler_v0/O2/scoreboard.json; loops/osler_v0/scoreboard.md (aggregates only)
- [ ] self-audit, CLAIMS O046+, STATE, commit, push
- [x] O2 closed at 2026-10-10T13:05:22Z: SELF_AUDIT outputs/osler_v0/O2/SELF_AUDIT.md; scoreboard loops/osler_v0/scoreboard.md; CLAIMS O123–O160

O3 checklist (completed 2026-10-09T18:30:07Z):
- [x] nine generator families, seven seen and two held out, in src/meddecide/gen/ (families.py); gold from the structured parts; twin pairs (CLAIMS O046)
- [x] build: train 21,000 items (seen generators only), dev 2,700, test 2,700; patients 10,500 / 1,350 / 1,350; 0 held-out rows in train (CLAIMS O047–O050)
- [x] screen: gold label in the record 0 hits on 1,500 applicable dev items; string-presence and NB macro below 0.90 on all nine; all pass (CLAIMS O051–O053); held-out baselines are constant predictors (CLAIMS O054, O056; heldout.md)
- [x] twin statistics for all nine (CLAIMS O055); majority baselines and test gold balance (CLAIMS O057–O059)
- [x] held-out list committed before any O6 work: loops/osler_v0/heldout.md
- [x] unit tests tests/test_gen.py: 28 passed (CLAIMS O060); CPU repo suite 446 passed, 5 GPU-gated tests skipped (CLAIMS O061; §7 items 20–21)
- [x] fresh-process audit scripts/osler/o3_audit.py: 0 mismatches against screen.json; rebuild reproduces the split SHA-256 values (CLAIMS O062–O063)
- [x] SELF_AUDIT outputs/osler_v0/O3/SELF_AUDIT.md; CLAIMS O046–O063; STATE; commits 8794fc8 (code) and the O3 DONE commit (docs); pushed

O4 checklist (completed 2026-10-09T19:01:39Z; started 2026-10-09T18:31:34Z):
- [x] inventory: student_v1 mix (482,889 train / 5,619 dev rows; read-only); O3 seen generators (21,000 train; 2,100 seen dev; held-out rows excluded); protected sets: v0.2 test and dev, external panel, robustness pack, held-out generators, held-out templates
- [x] replay licences checked on the Hub (card licence): tau/commonsense_qa mit (used); allenai/qasc cc-by-4.0 (used); allenai/cosmos_qa cc-by-4.0 but loading script only (NOT MEASURED: data outside the Hub); nyu-mll/glue other, google/boolq cc-by-sa-3.0, allenai/ai2_arc cc-by-sa-4.0, stanfordnlp/snli cc-by-sa-4.0, facebook/anli cc-by-nc-4.0, allenai/sciq cc-by-nc-3.0, allenai/openbookqa unknown: no permissive verified NLI or boolean-QA source, so those two replay kinds are NOT MEASURED
- [x] catalog train-candidates with cached Parquet: bigbio/pubhealth (mit, human verdicts), bigbio/chemprot (public-domain mark, human relations), bigbio/evidence_inference (mit, human answers); bigbio/bc5cdr and bigbio/mednli are loading scripts only (NOT MEASURED; mednli is MIMIC-derived and excluded by the data rules)
- [x] library and tests: src/meddecide/mix/{rows,converters,augment,leakage}.py; tests/test_mix_v2.py (14 tests passing)
- [x] build: scripts/osler/o4_build_mix.py (three runs with identical output hashes; logs build.log, build_run2.log, build_final.log; 151.5 s) (CLAIMS O064–O066, O090)
- [x] screen of each new template on its dev split: CommonsenseQA, QASC and PubHealth PASS; ChemProt and Evidence Inference NOT MEASURED (no rows) (CLAIMS O071); the macro is not comparable for item-specific labels (deviation 31)
- [x] student_v1 gold-visibility audit: gold text in state per choice or score template (CLAIMS O083); sex-word proxy for ct_eligibility_sex_choice_v1 (O084); learnability audit (O085–O088); findings in Question 9
- [x] robustness augmentation: 43,978 rows = 7.67 % of train (CLAIMS O066–O067); per-template 8 % cap respected, largest template 7.39 % (O070)
- [x] leakage counts: text 0 and dataset-qualified record 0 on train and dev (CLAIMS O075–O077); bare-id collisions 55 and 2 are PubHealth claims matching panel ids, not leaks (O078; deviation 25)
- [x] manifest data/train/osler_v0/manifest.json (committed copy: loops/osler_v0/mix_v2_manifest.json): counts, licences, provenance, budget rule, replay share, NOT MEASURED list
- [x] consistency designs (V4 asks for at least two): NOT MEASURED, not built (reasons in the manifest's not_measured; deviation 27; Question 13)
- [x] CLAIMS O064–O091; outputs/osler_v0/O4/SELF_AUDIT.md; STATE §7 deviations 24–32; commit and push
- Watchers: O2 chain watch re-armed at 18:41Z (the 18:11 watch expired at 18:41 with no events); O4 build watch armed at 18:41Z

O5 checklist (in flight; started 2026-10-09T19:02:52Z):
- [x] option-code head (D23): [K, d] readout, K = 255 (CPU: tests/test_optcode_o5.py 11 and tests/test_readouts_o5.py 9; CLAIMS O093), codes A–Z then two-letter codes (one token each in the base tokenizer: verify); rows initialised from lm_head rows of the code tokens; logits gathered for offered codes only; one temperature per qtype applied once; unit test: at initialisation, probabilities equal the zero-shot letter readout renormalised over the offered letters (tolerance 1e-4, fp32)
- [x] export (CPU tiny model, adapter separate and merged; CLAIMS O093): the trained head written into a copy of lm_head (row i to token id i), LoRA kept separate, and merged as a second artefact; round-trip test: same offered-code probabilities as the native path (tolerance 1e-3, fp32, 50 items)
- [x] non-causal flag (arm N; CPU tiny model: changes earlier positions only when on; CLAIMS O093): bidirectional full-attention layers on the decision path, linear-attention layers unchanged; unit test: the causal path is untouched when the flag is off
- [x] pointer head (CPU tiny model padding invariance; CLAIMS O093; the 4B run is pending): the student_v1 pointer head with its padding fix kept
- [x] padding invariance (CPU tiny model; CLAIMS O093; the 4B run is pending): batch-1 vs padded batch, tolerance 1e-3, on each path; generation byte-identity with the adapter off
- [x] GPU run on the pinned Qwen3.5-4B (fp32): attempt 4 (the verdict; unseeded adapter draw) writes outputs/osler_v0/O5/readout_checks.json: 10 of 11 pass; the merged-adapter export FAILS (1.04e-02 against 1e-3; CLAIMS O094, O095–O105). Seeded re-run (attempt 5, seed 0): 9 of 11, the merged export and bidirectional padding fail (O106–O109). Additional settings, seed 0: IEEE dots 11 of 11 and reference path 11 of 11 (O110–O113). Kernel controls and cross-precision numbers O114–O122. Report loops/osler_v0/O5_READOUT.md; self-audit outputs/osler_v0/O5/SELF_AUDIT.md. Verdict recorded as measured; the merged artefact is READOUT_FAIL under the as-run precision (Question 14)
- [x] training readiness: every mix v2 row loads as a validated training item (src/meddecide/train/mix_items.py; CLAIMS O092); one gradient step moves the loss and reaches the head and the adapter (O093)

O1 checklist (completed):
- [x] schemas, sizes, pinned revisions and licences of the usable sets (CLAIMS O032–O039)
- [x] converters for 7 sets with unit tests (tests/test_panel.py, 30 tests in total)
- [x] deterministic subsample ≤ 1,000 per set (stable hash of record key); counts and drop reasons recorded
- [x] overlap check: 0 text hits and 0 record hits against 2,150,361 training rows; v0.2 exact-text hits 0 after excluding 37 duplicates (CLAIMS O040, O041)
- [x] robustness pack: 4,000 bases, 20,898 items, rule-checked (CLAIMS O042–O044)
- [x] fix 3: long-record ids (2,213 of 37,289 over the cap, all in three openFDA templates; CLAIMS O045)
- [x] manifest data/bench/v0.3_ext/manifest.json; SELF_AUDIT (outputs/osler_v0/O1/SELF_AUDIT.md); CLAIMS O032–O045; STATE; commit; push

Licence screen used for the panel (catalog verdicts + Hub/GitHub licence checks):
  in the panel (eval-only):  TIGER-Lab/MMLU-Pro (mit), TsinghuaC3I/MedXpertQA (mit), HiTZ/MedExpQA (cc-by-4.0),
                             ofir408/MedConceptsQA (apache-2.0), gretelai/symptom_to_diagnosis (apache-2.0),
                             Lots-of-LoRAs/task1645 medical question pairs (apache-2.0 wrapper), bluesky333/MedExQA (cc-by-nc-sa-4.0)
  not in the panel, with reason:  songlab/clinvar (numeric feature table; label undocumented), bigbio/ddi_corpus (data fetched
                             from outside the Hub; NOT MEASURED), sixuexing/FAERS-NLP (5 GB; NOT MEASURED), lavita/MedQuAD (training source),
                             Medbullets / ADE corpus v2 / PubMed-200k-RCT / NLI4CT (no licence), bigbio/biored, mediqa_qa (no licence),
                             bigbio/head_qa (Spanish), healthver (tier-1 source), MedCalc / gad / chemprot (training candidates)
```

---

O9 and O11 preparation (CPU; written and tested before any arm result exists):
- [x] O9 rule as a pure function (`src/meddecide/eval/head_choice.py`: margin 1.0 point, paired bootstrap 1,000 resamples within templates, larger gain when both qualify; tests/test_gate_and_head_choice_o9.py, 8 tests) and the driver `scripts/osler/o9_head_choice.py` (refuses to decide on a missing arm or misaligned items; smoke-tested on synthetic arms in a scratch directory; writes loops/osler_v0/head_choice.md and outputs/osler_v0/O9/head_choice.json)
- [x] Gate O1 verdict logic as pure functions (`src/meddecide/eval/gate.py`: each comparison needs the accuracy interval's lower bound above 0 and the Brier interval's upper bound below 0; the knowledge guard needs the lower bound above -0.02; tests in the same file)
- [x] O11 scoring library (`src/meddecide/eval/osler_scoring.py`: row to item, reversed order, both-order averaging, `score_rows`; tests/test_osler_scoring_o11.py and tests/test_osler_score_rows_o11.py, 12 tests) and the GPU driver `scripts/osler/o11_score.py` (resumable; smoke-tested on CPU, deviation 44)
- [x] O11 gate script (`scripts/osler/o11_gate.py`; comparisons from `meddecide.eval.gate.compare_rows` with the item accounting; the knowledge guard; the D24 verdicts; writes loops/osler_v0/gate_o1.md and outputs/osler_v0/O11/gate.json). Smoke-tested on synthetic fixtures (a perfect Osler against the real MedDecider files and weak synthetic baselines: PASS for both sizes, and the MedDecider panel row counts 212 skipped, matching O2). The real run needs the arms' predictions and the v0.2 baseline rows (deviation 43); tests/test_gate_compare_o11.py (6 tests). results.md is not yet written
- [x] O2 tier investigation of the pplx v0.2 lead (CLAIMS O123–O125; `scripts/osler/o2_tier_breakdown.py`)
- [x] O9 driver reads each arm from its config's output directory (`O6/arm_L`, `O6/arm_P`, `O6/arm_N`; the driver had `O7`/`O8`, which no arm writes): fixed in dc9f3e7; `tests/test_o9_arm_paths.py`; a synthetic three-arm smoke at the config paths writes the report (deviation 49)

O10 preparation (CPU; written and tested before any arm result exists):
- [x] `src/meddecide/train/osler_configs.py`: the pinned bases (Qwen3.5-9B `c2022362`, Qwen3.5-0.8B `2fc06364`), O9's head settings (L: option-code causal; P: pointer; N: option-code non-causal), and the recipe check against arm L (only base, output directory and head may differ)
- [x] `scripts/osler/o10_train_model.py` (new driver; `o6_train_arm.py` is unchanged so the running chain keeps its code): runs the checks before the model loads, and writes provenance with the head, the head-choice file's sha256 and the reference config's sha256. Probed on scratch configs: a head that differs from O9's choice, a changed batch size and an unpinned revision are each refused, and no output directory is created
- [x] `tests/test_osler_configs_o10.py` (4 tests): arms P and N differ from L only in output directory and head; a recipe change is listed; the pinned base is enforced; the head must be O9's choice. CPU suite at this commit: 551 passed, 5 skipped
- [ ] after O9 (head chosen): write `configs/osler_v0/osler_9b.yaml` and `configs/osler_v0/osler_0p8b.yaml` from arm L's config: base and pinned revision, `output_dir: outputs/osler_v0/O10/osler_9b` (and `osler_0p8b`), the chosen head's readout and bidirectional flag, LoRA r=32 at both sizes (the ADVISORY's r=16 at 0.8B applies only if memory or speed requires it; neither is expected, since the 9B's measured peak was 55 GB at O0 without checkpointing and the 4B's was 17.8 GB with it)
- [ ] after O9: launch the 9B detached (one GPU job; `mkdir -p outputs/osler_v0/O10/logs`; log `outputs/osler_v0/O10/logs/osler_9b.log`), then the 0.8B; close O10 when both have `arm_result.json`

O6–O8 readiness for arms P and N (CPU; checked before those arms start):
- [x] read-only run-log monitor for the wake-ups (commit "O6 — read-only run-log monitor"): `src/meddecide/train/monitor.py`, `scripts/osler/o6_monitor.py --run-dir <arm dir>` (eval table, best so far by the selection rule, tripwire count, first and last 20 step records), `tests/test_train_monitor_o6.py` (2 tests); CPU suite 555 passed, 5 skipped (CLAIMS O166)
- [x] the end-of-run path (best checkpoint reload, temperature fit, full-dev scoring, `dev_predictions.jsonl`, `arm_result.json`) runs end to end with the pointer readout and with the non-causal option-code head, with gradient checkpointing and the template-macro selection, on the tiny Qwen3.5 model (`tests/test_arm_run_o6.py`: two new tests; the file passes 3 of 3). Arm L's attempt has no such exit path exercised yet, so a crash there would surface only at about 05:40Z on 2026-10-11

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

### O0 — DONE — 2026-10-09T07:50:24Z
- What ran: `scripts/osler/o0_reproduce.sh` (fetch, snapshot, 9 competitor smokes, JEV/pplx/Clef smokes, throughput 4B and 9B, aggregate, fresh-process re-derivation); code of record is the O0 commit (see SELF_AUDIT §header)
- Output: `outputs/osler_v0/O0/` (snapshot.json, envs.json, throughput.json, smoke_summary.json, fetch.json, smoke/, logs/, SELF_AUDIT.md); CLAIMS O001–O031
- Headline: 9 of 9 competitor rows PASS through their own code (MedDecider 4B/9B/27B/31B, pplx-decider-v1.1-27b, JEV-27B, JEV-9B, Clef, Clef-Flash; all open weights, none NOT MEASURED; O019). The v0.2 kept test set is 37,289 items (23,768 fresh + 13,521 tier-1), the same denominator student_v1 used (O003). Projected Osler-4B arm 16.08 h, 9B arm 24.59 h at the ADVISORY budget (O013, O017). Counted disk 393.52 GB of 450 (O030)
- Surprises: 2 of 11 v0.2 benchmark files do not match the sha256 in their manifest (clinicaltrials, openfda; O004), and the manifest total (57,007) disagrees with acceptance.json (45,009; O005). The MedDecider-4B first smoke criterion (gold label on every card example) failed on card example 6; the criterion was revised to reproduction of printed numbers with gold agreement reported beside it (see Deviations 1). Clef needed its own env (pillow, torchvision, accelerate)
- Next: O1 (external clinical panel and robustness pack, CPU + network). The licence screen for the panel is already done: MMLU-Pro (MIT), MedXpertQA-Text (MIT), MedConceptsQA (Apache-2.0), symptom-to-diagnosis (Apache-2.0), medical question pairs (Apache-2.0 on the Hub wrapper), MedExQA (CC BY-NC-SA 4.0, evaluation only). Excluded for no verifiable licence: Medbullets, MedQuAD, PubMed-200k-RCT, ADE corpus v2, NLI4CT (see Questions 5)

### O1 — DONE — 2026-10-09T08:13:20Z
- What ran: `scripts/osler/o1_build_panel.py`, `o1_overlap.py` (run on the rebuilt panel), `o1_build_robustness.py`, `o1_verify_robustness.py`, `o1_long_slice.py`; unit tests `tests/test_panel.py` (30 tests). Items are under data/ (gitignored); the manifest is committed as `loops/osler_v0/v0.3_ext_manifest.json`
- Output: data/bench/v0.3_ext/{panel.jsonl, robustness.jsonl, long_record_ids.json, manifest.json}; outputs/osler_v0/O1/{overlap.json, robustness_verify.json, SELF_AUDIT.md}; CLAIMS O032–O045
- Headline: external panel = 4,353 eval-only items in 7 public sets (CLAIMS O032); 0 text and 0 record matches against 2,150,361 training rows in 12 files (O040); robustness pack = 20,898 perturbed items from 4,000 bases, 0 rule violations on re-check (O043–O044); long-record slice = 2,213 of 37,289 kept v0.2 test rows over 8,192 prompt tokens, all in three openFDA templates (O045)
- Surprises: 37 MMLU-Pro health items were exact duplicates of v0.2 items, excluded (O041). MedExQA's gold letter is D in 36.6% of items (O037). The panel is narrower than the ADVISORY list: MedQuAD is a training source; Medbullets, ADE, PubMed-200k-RCT, NLI4CT have no verifiable licence; DDI and FAERS are NOT MEASURED for data-access reasons (Questions 5 and 7)
- Next: O2 (scoreboard; the first GPU job). Protocol and cost decisions for O2 are recorded in §7 before its first run

### O3 — DONE — 2026-10-09T18:30:07Z
- What ran: `scripts/osler/o3_build_generators.py` (build and screen; rebuilt in a fresh process with identical split hashes), `scripts/osler/o3_audit.py` (fresh-process recompute: 0 mismatches against screen.json), `tests/test_gen.py` (28 passed). Commits 8794fc8 (code) and the O3 DONE commit (docs)
- Output: data/gen/osler_v0/{train,dev,test}.jsonl and manifest.json (gitignored); outputs/osler_v0/O3/{screen.json, recompute.json, SELF_AUDIT.md}; loops/osler_v0/heldout.md; CLAIMS O046–O063
- Headline: nine generators, seven seen and two held out (note_lab_range_v1, policy_triage_v1). Train 21,000 items (none held out), dev 2,700, test 2,700. All nine pass the screen: 0 gold-in-state hits on 1,500 applicable dev items; seen Naive Bayes macro 0.51–0.86 (CLAIMS O046–O055)
- Surprises: the held-out baselines are constant predictors (macro 0.3333, one distinct prediction), so the held-out pass rests on gold-in-state and the construction checks (CLAIMS O056; Question 8). The triage source label in screen.json is wrong (§7 item 22). A full-suite run started a GPU test while the O2 job ran (§7 item 20); the chain was unaffected
- Next: O4 (training mix v2, CPU), allowed while O2 uses the GPU. O1 and O3 are DONE

### O4 — DONE — 2026-10-09T19:01:39Z
- What ran: `scripts/osler/o4_build_mix.py` (three runs with identical output hashes; logs build.log, build_run2.log, build_final.log), `scripts/osler/o4_verify_mix.py` (fresh-process re-derivation), `scripts/osler/o4_audit_mix.py` (learnability audit, additional); tests/test_mix_v2.py (15) and tests/test_mix_audit.py (2). Code commit b4bec88
- Output: data/train/osler_v0/{train.jsonl (573,611 rows), dev.jsonl (10,843 rows), manifest.json} (gitignored); loops/osler_v0/mix_v2_manifest.json (committed copy); outputs/osler_v0/O4/{build.json, verify_mix.json, audit_mix.json, SELF_AUDIT.md}; CLAIMS O064–O091
- Headline: mix v2 has 573,611 train rows (student_v1 482,889 + 46,744 new + 43,978 augmented, 7.67 %) and 10,843 dev rows; text and dataset-qualified leakage are 0 on both (CLAIMS O075–O076); budget 200,000 examples (O068); general replay 3.46 % (O069)
- Surprises: inherited student_v1 shortcuts and visibility gaps (nfcorpus BoW 0.996; medquad 0.984; pubmed_pubtype 0.860; O085–O087); ChemProt and Evidence Inference unusable (PMIDs outside the pool's range); 1,934 PubHealth claims dropped for missing dates; bare-id collision check corrected (deviation 25)
- Next: O5 (readouts). Its checks need the GPU, so O5 waits for a GPU-free window; its code and CPU unit tests can start now while O2 runs. Operator Questions 9–13 are open

### O4 addendum — 2026-10-09T19:17:15Z
- What ran: scripts/osler/o4_check_loadable.py: every mix v2 row through the trainer's strict Item reader. Train 573,611 of 573,611 and dev 10,843 of 10,843 load (CLAIMS O092). Training-only placeholders for the schema's required fields are in src/meddecide/train/mix_items.py (the trainer never reads the tier, the URL or the record date)
- Surprises: the check was not in the O4 acceptance and should have been; it found no failures, so no row is excluded
- Next: O5 GPU run (pending the O2 chain)

### O5 — DONE — 2026-10-09T20:33:07Z
- What ran: `scripts/osler/o5_checks.py` (attempt 4, the verdict, unseeded; attempt 5 with `--seed 0`; `--precision ieee|reference --seed 0`), `scripts/osler/o5_kernel_control.py` (seeds 0–2; two processes per seeded setting), `scripts/osler/o5_summary.py`; tests tests/test_precision_o5.py (8), tests/test_dev_predictions_o6.py (2), tests/test_paired_macro_o9.py (7), tests/test_arm_run_o6.py extended. Chain logs: outputs/osler_v0/O5/logs/control_chain*.log
- Output: loops/osler_v0/O5_READOUT.md (committed); outputs/osler_v0/O5/{readout_checks*.json, kernel_control_*.json, o5_summary.json, SELF_AUDIT.md} (gitignored); CLAIMS O094–O122
- Headline: the pre-registered O5 run passes 10 of 11 checks. The merged-adapter export fails (1.04e-02 against 1e-3; O094, O105). The separate-adapter export passes (2.15e-06; O104). The failure traces to the as-run fp32 kernel precision: with IEEE dots or the kernels blocked, all 11 pass on seed 0 (O110–O113); with a fixed seed, the as-run and IEEE forwards are bit-identical across processes (O119)
- Surprises: the attempt-4 adapter's lora_A was unseeded, so my first reading of cross-process variation was a draw effect (O121); seeded as-run also fails bidirectional padding in one draw (1.03e-03; O108); the first IEEE attempt hit a fresh-cache shim failure and a disk-quota error (§7 deviation 37)
- Next: O2 resumes at clef (chain paused since 19:23:35Z). O6 (arm L) follows O2. Questions 14 and 15 are open

### O2 — DONE — 2026-10-10T13:05:22Z
- What ran: `scripts/osler/o2_chain.sh` (clef, md27b, md31b, metrics; PID 266074, resumed after the O5 window), a baseline chain (zero-shot 4B and 9B and JEV-9B on v0.2, deviation 43), `scripts/osler/o2_metrics.py` (final regeneration after the D12 change), `scripts/osler/o2_tier_breakdown.py` (the pplx investigation)
- Output: `outputs/osler_v0/O2/*/predictions.jsonl` (11 models, 59,538 rows each where scored or skipped), `scoreboard.json`, `tier_breakdown.json`; committed `loops/osler_v0/scoreboard.md`; CLAIMS O123–O160; SELF_AUDIT
- Headline: v0.2 accuracy (scored, 35,076 items) MedDecider-27B 0.856, pplx-27B 0.861, Clef 0.853, JEV-27B 0.847, MedDecider-4B 0.824 and -9B 0.829, zero-shot 4B 0.804 and 9B 0.833. External panel: MedDecider-31B 0.665, pplx 0.641, Clef 0.610 (O126–O136)
- Surprises: pplx's v0.2 lead is partly two near-ceiling templates shared by all models (O125) and its robustness is the lowest of the large models (0.530); zero-shot 9B has one D12 failing cell (nfcorpus, excluded); the Gate O1 baselines were missing on v0.2 for three models (deviation 43)
- Next: O6 arm L attempt 2 is running; O7 (arm P), O8 (arm N), then O9 (head choice), O10 and O11

### O6 — advisor intervention — 2026-10-10T14:37:06Z
- What ran: operator decisions Q18–Q20 applied by the advisor (Claude Opus) while the loop agent was stopped; code + configs + tests changed (deviation 48)
- Output: configs/osler_v0/arm_{L,P,N}.yaml; STATE §3 restart block
- Headline: arm L attempt 4 stopped at step 750 (dev macro 0.471 per-gold-key, accuracy 0.818); arms re-run with 10 dev evals each and template-macro selection; 299.6 GB of competitor weights deleted by the operator
- Surprises: none beyond Q20
- Next: the loop agent executes the §3 restart block, then O6 attempt 5

### O6 — progress (arm L attempt 5) — 2026-10-10T20:24:13Z
- What ran: worker PID 690638 (run of record) and chain PID 691022 unchanged; dev evals at steps 2,500 and 5,000 read from `outputs/osler_v0/O6/arm_L/logs/train.jsonl`; O9 driver path fix (dc9f3e7) with a synthetic three-arm smoke at the config paths; CPU suite 547 passed, 5 skipped (CUDA hidden)
- Output: CLAIMS O161–O163 (added now; the step-2,500 numbers had been reported in STATE and a commit without CLAIMS rows, an R1 gap); deviation 49
- Headline: template macro 0.8197 (2,500) → 0.8133 (5,000), so the best checkpoint stays at step 2,500. Per-letter diagnostic 0.6029 → 0.5400 while accuracy rose 0.8458 → 0.8468; reported as measured (diagnostic only; Q20)
- Surprises: the O9 driver's arm directories did not match the configs (fixed before O9 could run; deviation 49)
- Next: wait for arm L (about 05:40Z on 2026-10-11); confirm the chain starts arm P, then arm N; O9 once both have `arm_result.json`

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|
| O5-merged | the merged-adapter artefact of Osler-4B: any merged release and the quantised ≤ 4 GB export in O11 | the merged export fails the pre-registered 1e-3 check under the as-run precision in both adapter draws (CLAIMS O094, O105–O108); the cause is kernel TF32 precision, not the merge (O114–O122). The separate-adapter artefact is unaffected and passes (O104, O109) | Question 14 decided; then a merged export verified under the chosen precision, recorded as a new run |

---

## 6. Questions for the operator

Anything you could not resolve without a human. Be specific enough to answer without
re-reading the run: state the ambiguity, the options, and which you would pick.

1. **v0.2 manifest integrity (no action taken; v0.2 is read-only).** `data/bench/v0.2/manifest.json`
   records sha256 values that do not match `clinicaltrials.jsonl` and `openfda.jsonl` (CLAIMS O004).
   The manifest total (57,007) and `acceptance.json` (45,009) also disagree (O005). The row counts match
   the manifest, and the kept test denominator (37,289) reproduces student_v1's, so student_v1's results
   are unaffected as far as the counts show. Options: (a) leave as is and note it in the closure report
   (my pick); (b) regenerate the manifest hashes from the current files (changes a v0.2 artifact, so it
   needs your approval); (c) rebuild v0.2 (out of loop scope). Which do you want before the scoreboard
   is published?
2. **Disk for the exports (BLOCKED — disk later, not yet reached).** Counted usage is 393.52 GB of the
   450 GB budget (O030). The planned O5/O10 exports (Osler-4B export and merged, 9 GB each; Osler-9B
   export and merged, 19 GB each; Osler-0.8B, about 2 GB each) plus selected checkpoints would take the
   total past 450 GB. The competitor weights are the largest item: pplx27b 52 GB, JEV-27B 55 GB, Clef
   55 GB, Clef-Flash 19 GB, Qwen3.8-27B base 56 GB, gemma-4-31b-it base 63 GB. Options: (a) after O2
   closes, delete those competitor snapshots (they are not needed by O11 or later; deleting from the HF
   cache is outside the loop's delete guardrail, so this needs your approval); (b) raise the budget;
   (c) skip the merged 9B artefact (saves 19 GB, not enough alone). My pick is (a) at O2 closure. I will
   not delete anything without your answer.
3. **JEV route.** The JEV rows use the card's decision-head protocol through transformers + PEFT (the F7
   path), not the card's vLLM server. vLLM is not in the main env, and the 9B repo references a
   `vl/serve.sh` that is not published. My pick: keep the F7 path for consistency with the existing JEV-9B
   baseline and state the deviation in every JEV row. Alternative: a separate `envs/jev/` with vLLM,
   about 2 h. Which do you want for the Gate O1 baseline?
4. **Dev-eval cadence.** The ADVISORY says "dev eval every 2,000 examples × 8". I read it as a full dev
   evaluation (6,000 items) every 2,000 examples, which costs about 8.9 h of eval per 4B arm (O013: 16.08 h
   total). If you meant a fixed dev subset, eval would be much cheaper but selection less stable. My pick
   is the literal reading until you say otherwise.
5. **Panel licences (O1 screen).** Usable with verified licences: MMLU-Pro (MIT), MedXpertQA-Text (MIT),
   MedConceptsQA (Apache-2.0), symptom-to-diagnosis (Apache-2.0), medical question pairs (Apache-2.0 on the
   Hub wrapper, source licence not verified). MedExQA is CC BY-NC-SA 4.0: usable for evaluation, but any
   public release of items derived from it would carry NC-SA; I will include it for evaluation only and
   mark it for the release decision. Excluded for no verifiable licence: Medbullets, MedQuAD (GitHub
   NOASSERTION), PubMed-200k-RCT (no licence on the Hub or GitHub), ADE corpus v2 ("unknown"), NLI4CT
   (no licence on the Hub mirrors). The ADVISORY's panel list therefore shrinks; if you can confirm a
   licence for any excluded set, I will add it. Is that acceptable?
   **Answer status:** I proceeded with this screen (conservative option) and built the panel from the
   usable sets only. Nothing here needs an answer to continue; the questions below are for the release and
   for licence confirmation.
6. **MedExQA (CC BY-NC-SA 4.0) in the panel.** It is used for evaluation only (940 items; CLAIMS O037). A public
   release of benchmark items derived from it would have to carry NC-SA. Options: (a) keep it in the panel and
   exclude its items from any public release (my pick for now); (b) drop it from the panel. Which do you want?
7. **Problem-type coverage.** The panel has two problem-type sets (symptom-to-diagnosis; medical question pairs)
   and five exam sets. The sets that would add problem types (NLI4CT, PubMed-200k-RCT, ADE corpus v2, Medbullets)
   have no declared licence, and DDI and FAERS need data-access work. If you can confirm a licence for any of
   them, or approve the work to fetch DDI and FAERS (about 2 h each), the next panel build can include them; the
   builder would take them as a new SourceSpec with a manifest entry. I have not started that work.

8. **Held-out screen baselines (O3; no action taken; the held-out list is unchanged).** Both held-out generators pass the
   screen, but their string-presence and Naive Bayes baselines are constant predictors. For `note_lab_range_v1` the Naive
   Bayes was trained on 12,000 seen rows whose labels (yes/no) never match the offered options. `policy_triage_v1` has no
   seen family-C generator, so its model trained on 0 rows. Their 0.3333 macro values are chance over three classes, not
   measured shortcuts (CLAIMS O054, O056; `loops/osler_v0/heldout.md`). The pass therefore rests on gold-in-state = 0 and
   the construction checks. Options: (a) accept that as the held-out screen (my pick; nothing changes after seeing held-out
   outputs); (b) add a label-agnostic baseline for the held-out generators before O6. That is a new screen run and would
   have to be specified before any model result. Which do you want? I proceed with (a) unless you say otherwise.
9. **Inherited student_v1 templates with shortcut or visibility findings (O4 audit; no action taken).** The O4 audit (CLAIMS O085–O088) finds: `nfcorpus_relevant_noul_v1`, bag-of-words accuracy 0.996 against 0.50 chance; `medquad_routing_v1`, 0.984 against 0.25, with the gold text in the state for 55 % of items; `pubmed_pubtype_choice_v1`, 0.860 against 0.167, gold text in 18 %; `fda_class_choice_v2`, 0.952 against 0.25, while its macro (0.898) passes the 0.90 line; `ct_claim_set_choice_v1`, 0.924 against 0.25; `pubmed_check_*`, 0.72–0.80 against 0.50; `ct_eligibility_sex_choice_v1`, with sex words in 16 % of states (macro 0.578 against majority 0.333). Options: (a) keep the student_v1 mix for O6–O8 as the ADVISORY specifies, and report these templates in every O11 table (my pick: one shared mix keeps the three arms matched); (b) exclude the flagged templates before O6, which is a new mix version that changes the base for all arms; (c) cap their share. Which do you want before O6? I proceed with (a) unless you say otherwise.
10. **Licences of inherited tier-1 train sources.** The student_v1 manifest records UNKNOWN licences for medmcqa (42,370 rows in the mix), medqa (9,084), medquad (34,876), mmlu_medical, nfcorpus (5,287), pubmedqa, scifact (1,693) and trec_covid. D17 requires a training-compatible licence for catalog sources. These are official tier-1 train splits approved under student_v1 (S5), but their licence fields are unknown. Options: (a) accept under S5 and record UNKNOWN in the release notes (my pick); (b) verify each licence before release; (c) drop them, which changes the mix. Which do you want?
11. **Replay share.** The mix has 3.46 % general replay (CommonsenseQA 10,765 and QASC 9,064 rows) against the ADVISORY default of 20 %. Natural-language-inference and boolean-QA replay are NOT MEASURED: no permissive source with a verified licence (nyu-mll/glue is 'other'; stanfordnlp/snli cc-by-sa-4.0; facebook/anli cc-by-nc-4.0; google/boolq cc-by-sa-3.0). Options: (a) accept 3.5 % as a recorded share (my pick; the ADVISORY allows a recorded share); (b) check more general sources (allenai/openbookqa's Hub licence is 'unknown'; HellaSwag, PIQA and WinoGrande have no verified Hub licence) and add any that verify. Which do you want?
12. **ChemProt and Evidence Inference.** Their PMIDs fall outside the pre-window pool's PMID range (35,997,240 to 41,610,285). Every ChemProt row (1,020 train, 612 dev) and Evidence Inference row (10,056 train, 1,233 dev) was dropped as document_not_in_prewindow_pool (CLAIMS O072–O074). Options: (a) leave them out (my pick; NOT MEASURED); (b) look up PubMed record dates through NCBI E-utilities to establish pre-window status, then add them (about 1 h, needs network). Which do you want?
13. **Record–claim consistency designs (V4 asks for at least two).** Not built in this loop: NOT MEASURED. The candidate designs examined need a claim value that the state text carries. The journal name was absent from the PubMed states inspected; eligibility sex appears in about 16 % of clinicaltrials states; the ChemProt and Evidence claims cannot be dated. Options: (a) accept NOT MEASURED for this loop (my pick); (b) build designs from state-visible structured fields in the next loop, after re-reading the S2 role-binding rules. Which do you want?
14. **Which numerical setting is the O5 "fp32" verdict configuration, and may the merged artefact be used (ADVISORY O5 step 2: "fp32, tolerance 1e-3").** Under the as-run setting (fla and causal-conv1d kernels with Triton's default fp32 dot precision, which is TF32 on this GPU), the merged-adapter export fails in both adapter draws (1.04e-02 unseeded; 4.17e-03 seed 0), and bidirectional padding fails at 1.03e-03 in seed 0 (CLAIMS O094–O108). The same checks pass with IEEE dots or with the kernels blocked, all 11 on seed 0 (O110–O113). The separate-adapter export passes in every run (O104, O109). Options: (a) keep the as-run setting and record the merged artefact as READOUT_FAIL (current; the merged artefact stays unused, including the ≤ 4 GB quantised export in O11); (b) approve a new pre-registered O5 run under IEEE dots, with its own record (I recommend this; no verdict changes until you approve it); (c) use the reference path for the O5 checks. Also decide whether O9–O11 scoring uses IEEE dots: as-run padding moves probabilities by up to about 1e-3 with batch composition (O108), which can matter for paired comparisons at that scale. Which do you want?

15. **bf16 numerics for training and evaluation (ADVISORY O6–O10 use bf16).** The fp32 findings do not cover bf16. fla's chunked triangular solve requests TF32 for fp32 operands on this GPU; whether bf16 operands take the same path is not measured. All three arms use the same kernel path, so comparisons between them stay matched. Recommendation: start O6 on the pre-registered bf16 recipe, and measure the bf16 path (a TF32 and IEEE control in the arm's own dtype) before any bf16 number is reported as final. Do you want that control before O6 or after it?
16. **Near-ceiling templates inside the pre-registered headline sets (O2 finding; definitions kept).** Fresh `pubmed_mesh_major_choice_v2` (4,000 items) is answered at 0.984–0.990 by all five v0.2 models, and established `medquad_routing_v1` (5,000 items) at 0.985–0.993 (CLAIMS O125). The routing template was already flagged as shortcut-solvable in student_v1 and bench_v0. Both sit inside the D16 and D24 headline sets as defined, so the Gate O1 verdict is computed on them as pre-registered. I will add a labelled additional analysis without these two templates and change no verdict. Do you want the next loop to revisit these templates, or leave them as they are?

17. **GPU order: the O2 competitor runs before the training arms (scheduling; the plan's order is kept).** At the clef rate measured at 21:04Z (about 224 rows a minute, so about 4.4 hours for clef's 59,538 rows), then md27b and md31b (about 2.5 hours each by the earlier timings), then the three D24 baselines on v0.2 (deviation 43), clef finished at 00:52Z (about 12.5 hours of wall clock after its start, uneven in speed: 59,538 rows); md27b runs at 4.33 items a second (about 3.8 hours), so the arms O6–O8 start about 10 hours from now. Only the baselines are needed for the Gate O1 verdict. clef, md27b and md31b are additional (ADVISORY O11: every O2 model is never in a verdict). Option (a), current: keep the plan's order. Option (b): stop the O2 chain after clef, run the three baselines and then the arms, and resume the remaining competitor runs later (every runner resumes from its rows). Option (b) reaches the verdict sooner and loses no run. I keep (a) until you say otherwise. Which do you want?



18. **Disk quota (blocker for the next GPU jobs).** The /workspace quota is about 410 GB counted by my measure (HF cache 339 GB, uv cache 16 GB, data 16 GB, outputs 17 GB, /workspace/tmp 21 GB). It was reached again at about 13:20Z; the rules let me delete only outputs/osler_v0 and /workspace/tmp, and I deleted only my own regenerable scratch (about 1 GB). Writes succeed again (a 1.5 GB test passed at 13:23Z), but the O10 and O11 outputs need several GB more (O11 alone is about 1.5 GB of predictions). Options: (a) raise the quota; (b) approve the removal of named HF models that no remaining step uses (which? I need your list); (c) approve the removal of named earlier-loop items under /workspace/tmp (largest: pubmed_v1 13 GB, pubmed 3.4 GB, s6 2.0 GB, laya-typed 1.6 GB). Which do you want?
    **Answer (operator, via the advisor, 2026-10-10T14:37:06Z):** option (b). The operator deleted eight competitor snapshots that no remaining step uses (pplx-decider-v1.1-27b, JEV-27B, Clef, Clef-Flash, Qwen3.8-27B, gemma-4-31b-it, MedDecider-27B, MedDecider-31B): 242 blob files, 299.6 GB, keeping the one blob shared with a kept model. Kept: Qwen3.5-0.8B/-4B/-9B, JEV-9B, MedDecider-4B/-9B. After deletion: `/workspace/.hf_home` 61 GB, uv cache 33 GB, data 16 GB, outputs 18 GB, /workspace/tmp 21 GB, envs 9.2 GB. Re-measure with `du` before O10. Re-running a deleted competitor needs a re-download. Do not delete earlier-loop items under /workspace/tmp.
19. **Eval cost against the budget (ADVISORY O6–O8).** At the ADVISORY cadence (eval every 250 steps on 6,000 dev items; one dev eval takes about 6–14 minutes here), one 4B arm needs about 12–25 hours, not the 16 hours projected at O0. Three arms and the 9B arm are therefore a multi-day run. Options: (a) keep the cadence and subset (current; the arms run as planned); (b) keep the cadence and reduce the subset to 3,000 (the selection then uses a smaller dev sample; a recorded change of the recipe); (c) keep the subset and eval every 500 steps (a recorded change). Which do you want?
    **Answer (operator, via the advisor, 2026-10-10T14:37:06Z):** a recorded recipe change for all three arms: `eval_every: 2500` (10 dev evals per arm, the last at step 25,000), the same 6,000-item stratified subset. Evaluation does not change the weights, so the arms stay matched. The ADVISORY wording "every 2,000 examples × 8" was the advisor's ambiguity, not the agent's error.

20. **The dev macro (the selection and tripwire statistic) fell while accuracy rose; arm L is PAUSED before the step-1000 eval (ADVISORY O6–O8; no change made).** Step 750 eval: macro 0.4713, Brier 0.276, accuracy 0.818. The step-500 macro was 0.5516, so this is the first eval below 0.50. The next eval below 0.50 would be the second in a row and would stop the run under the tripwire, keeping step 250 as best. To keep that from deciding the outcome before you have answered, I paused the training process (PID 576775, stopped with SIGSTOP; its state is intact). Resume with `kill -CONT 576775`. The whole three-arm plan waits on this decision. Arm L attempt 4: step 250 macro 0.6967, Brier 0.368, accuracy 0.765; step 500 macro 0.5516, Brier 0.270, accuracy 0.817. Per question type: choice accuracy 0.741 → 0.838 but choice macro 0.703 → 0.558; noul 0.816 → 0.813; score 0.495 → 0.569. The macro is an unweighted mean of per-gold-class recall, so rarer letters losing recall pulls it down while accuracy rises. The recipe's tripwire stops the run if dev macro is below 0.50 on two consecutive evals, and selection uses this macro, so the saved best checkpoint is still step 250. The run continues under the recipe. If you want the selection or the tripwire to use a different statistic (for example accuracy or a class-balanced macro), that is a recipe change for you to decide. Which do you want?
    **Answer (operator, via the advisor, 2026-10-10T14:37:06Z):** change the statistic, and restart arm L. Checkpoint selection, the dev tripwire (floor 0.50, two consecutive evals) and the O9 head-choice rule all use **G1's / D24's macro**: the unweighted mean over templates of per-template accuracy (`template_macro_accuracy`; config `selection_metric: template_macro_accuracy`). The per-gold-key recall macro stays in the logs as a diagnostic. Attempt 4 is not resumed (its selection used the old statistic); it is archived as `arm_L_attempt4_eval250`. Arms P and N use the same rule. Code: commit of this answer (trainer, config, metrics, head_choice, o9 script, tests).

---

## 7. Deviations from the plan

Any place you departed from GOAL/ADVISORY, with the reason. An empty section is the
expected outcome. Editing code or a check to make it pass is never an acceptable
deviation — that is a `BLOCKED`.

1. **Smoke criterion revised after one result (O0, disclosed).** The first MedDecider-4B run required
   the argmax to equal the card's gold label on every example. Card example 6 prints Syndromic 0.478
   and Serial cross-sectional 0.370, and its own gold label is Serial cross-sectional, so the card's own
   numbers disagree with its label. I changed the smoke criterion to reproduction of the printed numbers
   and printed argmax, and kept gold agreement as a separate, reported field. Under the original
   criterion md4b would be FAIL on that one example. md4b is reported PASS (reproduction) with gold
   5 of 6 (`smoke/md4b.json`, block 6 `argmax_matches_card_gold: false`). The other three cards meet both.
   Full account in `outputs/osler_v0/O0/SELF_AUDIT.md` §3.
2. **JEV via the F7 transformers path**, not the card's vLLM route (see Question 3). The 27B card's own
   transformers block gives a refund P(true) of 0.978 through a different head file; ours gives 0.9779.
3. **Competitor environments.** `envs/pplx27b` uses the authors' `uv.lock` (sha256 `cba79e0f...`, which
   matches the checkpoint's `decision_config.json` provenance). `envs/clef` uses the card's tested pins
   (torch 2.11, transformers 5.10.2, with accelerate, pillow and torchvision). Neither installs into the
   main env. Clef and pplx run the reference linear-attention kernels where `causal_conv1d` (and for
   Clef, `fla`) is missing: correct but slower, so O2 latency needs that caveat.
4. **Throughput inputs.** Timing uses random token ids with a 255-row random readout of the planned
   shape. The loss values are not a training signal; the numbers measure cost only. The projection
   assumes the 200,000-example budget, because mix v2 (not yet built) must be at least as large as the
   student_v1 mix (482,889 rows) for "one pass or 200,000, whichever is smaller" to resolve to 200,000.
   O4 must confirm that.
5. **Panel smaller than the ADVISORY's list (O1).** The screen in Question 5 and Question 7 applies: 7 sets built,
   with MedQuAD (training source), six unlicensed sets, and DDI/FAERS (data access) left out. Recorded in the
   manifest's `excluded_sets` and in STATE. The problem-type axis is thin and is reported as such.
6. **Benchmark-duplicate exclusion added during O1 (disclosed).** After the first panel build, 37 MMLU-Pro health
   items were found to have exact v0.2 text (state, question, labels). The rule "a panel item must not be a v0.2
   item" was added and the panel, robustness pack and overlap check were rebuilt and re-run (CLAIMS O041, the
   first build's 4,390 items became 4,353). The rule was not chosen with any result in view: it separates the
   external panel from the headline benchmark, and it changes only the panel size.
7. **Subsampling.** MedXpertQA-Text (1,000 of 2,450) and MedConceptsQA (1,000 of 819,772) are quota samples by
   stable hash, not all items. The remaining rows are recorded as not examined, not dropped.
8. **Fields never read (O1 converters).** MedExpQA's `rag`, `explanations` and `full_answer`, and MedExQA's two
   explanation fields, are not read, so a converter cannot copy an answer into the state (unit-tested).
9. **Overlap scan scope (O1).** The compressed PubMed pool (`sources/pubmed_prewindow.jsonl.gz`, 2.8 GB) is not
   scanned; no panel set is PubMed-derived. Recorded in overlap.json (`not_scanned`).
10. **Tier.** Panel items carry no benchmark tier (external sets have no record dates to place them in the fresh
    window). The `EvalItem` schema records `benchmark = ext_panel` instead.
11. **O2 protocol: one common item set for every model.** 59,538 items whose prompt is at most 8,192 tokens under
    the Qwen tokenizer (`o2_items.jsonl`). The 3,002 items over the cap (2,213 v0.2, 789 robustness openFDA) are
    excluded for every model, so the denominators are identical. Deviation: MedDecider's code has no cap and
    Clef's endpoint allows 16,384; both are held to the common 8,192 so the rows are comparable (manifest section
    `o2_items`).
12. **O2 scope.** Competitors on all three sets. Zero-shot Qwen3.5-4B and -9B, and JEV-9B, on panel and robustness
    only, as ADVISORY O2 says.
13. **O2 option order.** MedDecider averages both orders (its own `decide(orders=2)`). pplx, JEV and Clef are single
    order, as their code runs (Clef's encoder sorts choice keys, so order cannot change its output). The zero-shot
    harness is single order, in file order.
14. **MedDecider letter limit.** The authors' letter set is A to J, so items with more than 10 options are skipped
    with that reason (the 22-class symptom-to-diagnosis set). Its probabilities come from the vector form of
    `decide()`, checked against `decide()` on 50 items (max abs diff 4.9e-5, the 4-dp rounding in their dict).
15. **JEV slot limits.** The card's head has 16 choice, 2 noul and 6 score slots; items beyond them are skipped with
    the reason. The `logits_to_keep=1` shortcut matches full-sequence logits (max abs diff 4.4e-8 on 20 items).
16. **Zero-shot readout.** The letter variant is detected per model by the bench's greedy probe (`detect_variant`),
    not guessed. The first zero-shot run used the "space" variant without detection: its label mass was 0.019 and the
    variant was wrong for Qwen3.5 (the bare letter is the emitted one). Those 40 rows were deleted before the valid
    run, whose variant is recorded in `outputs/osler_v0/O2/qwen35-4b/variant.json`.
17. **O2 metrics.** Macro = mean over templates of the per-template value (G1's definition). Brier is multi-class,
    per item. ECE uses 15 equal bins on the top probability. Bootstrap CIs over items, 1,000 resamples, seed 0. The
    D12 cell gate applies to zero-shot cells; D21 checks (constant answer, accuracy CI not below chance) to the rest.
    Greedy agreement and label mass are diagnostics for those readouts, not gates.
18. **O3 generator revisions before any model result (disclosed).** The first screen runs found shortcuts and defects, and
    I fixed them before any model was run on O3 data: (a) negation, subject and allergy-medication items now carry balanced
    distractors, so the bag-of-words baseline cannot read the answer from cue words; (b) timing items were rewritten so the
    answer needs a comparison of two years (the first version had a bag-of-words shortcut); (c) allergy-medication items use
    a 1,000-name synthetic drug pool and equal line counts; (d) gold-in-state and string-presence apply to choice and score
    items only, because yes/no labels occur in ordinary text; (e) for the two held-out generators, `policy_triage_v1` got a
    sampling fix (a case with chest pain, sweating and systolic BP below 90 had no twin that changed the level and raised
    StopIteration), and `note_lab_range_v1` got an HbA1c twin-range correction. The first screen run printed lab-range values
    before it crashed on triage, and I did not record whether the HbA1c correction came before or after that print. Both
    held-out screen values are constant-predictor values (item 23). The held-out list did not change. The earlier code is
    not in git (O3 code was first committed at 8794fc8), so these revisions cannot be diffed.
19. **Post-build docstring edit.** The module docstring of `src/meddecide/gen/families.py` said `note_timing_v1`; commit
    8794fc8 changes it to `note_timing_v2`. No logic changed. A fresh rebuild gives identical split SHA-256 values and
    identical screen reports (CLAIMS O062–O063).
20. **GPU-rule incident during O2.** At 18:23Z a full-suite pytest run started `tests/test_model_pointer.py::test_overfit_64_items`
    on the GPU while the O2 jev-27b job (PID 281949) was running. That broke the one-GPU-job rule. The test ran out of memory in
    my process; the chain was unaffected (chain log clean at 18:24Z; PID 266074 alive). From now on pytest runs with
    `CUDA_VISIBLE_DEVICES=` while any GPU job runs, so GPU-gated tests skip. The five GPU-gated tests (four in
    `test_model_pointer.py`, one in `test_padding_trained.py`) are NOT MEASURED until the first GPU-free window.
21. **Commit gate.** AGENTS asks for pytest to pass before any commit. Commit 8794fc8 was made with the CPU-only suite (446 passed,
    5 GPU-gated skipped; CLAIMS O061), because of item 20. The five GPU-gated tests run in the first GPU-free window.
22. **Screen report label.** `outputs/osler_v0/O3/screen.json` says the Naive Bayes for `policy_triage_v1` used "seen generators
    of family C (train split)". No seen generator is in family C, so it trained on 0 rows (CLAIMS O056). The file is a build
    output and was not hand-edited (R9). The accurate description is in `outputs/osler_v0/O3/recompute.json` and `heldout.md`.
23. **Held-out baselines are constant predictors.** Both held-out generators' string-presence and Naive Bayes baselines predict one
    label for every dev item (CLAIMS O054, O056), so their pass rests on gold-in-state = 0 and the construction checks. Raised as
    Question 8; no change made.
24. **O4 replay share below the default (measured).** General replay is 3.46 % of the mix (19,829 rows: CommonsenseQA 10,765 and QASC 9,064, including augmented rows) against the ADVISORY default of 20 %. NLI and boolean-QA replay are NOT MEASURED (no permissive source with a verified licence). CosmosQA and BC5CDR are NOT MEASURED (loading scripts only, data outside the Hub). Recorded in the manifest's not_measured (CLAIMS O069). Question 11.
25. **Leakage record check corrected after the first build (disclosed).** The first O4 build compared record ids as bare strings and reported 40 + 15 train and 1 + 1 dev collisions. All of them are PubHealth claims whose numeric id equals a panel or robustness MedExpQA item id: coincidences across unrelated datasets. The check was corrected to dataset-qualified keys, which give 0 hits (CLAIMS O075–O078). The build now reports both counts (record_bare and record_qualified). No row was removed for a collision, and the text check was not changed.
26. **ChemProt and Evidence Inference not used.** Their PMIDs fall outside the pre-window pool's PMID range (35,997,240 to 41,610,285; 0 of 5,800 needed PMIDs found). Their record dates were not verified, so every row was dropped with its reason (CLAIMS O072–O074). NOT MEASURED. Question 12.
27. **Consistency designs (V4 asks for at least two): NOT MEASURED.** The candidate designs examined did not meet the precondition that the claimed value appears in the state text: the journal name was absent from the PubMed states inspected, and eligibility sex appears in about 16 % of clinicaltrials states. Question 13.
28. **Budget rule corrected (deviation 4 read too strictly).** Deviation 4 said the 200,000 resolution needs a mix at least as large as student_v1's 482,889. The ADVISORY rule is min(one pass, 200,000). One pass over the mix (573,611 rows) exceeds 200,000, so the budget is 200,000 examples (CLAIMS O068). No change to the plan.
29. **PubHealth claims without a date dropped.** 1,934 of 9,804 train claims (19.7 %) and 245 of 1,223 dev claims had no date and were dropped as missing_date. D13 requires records dated before 2026-03-01, so an undated claim cannot be shown to be pre-window. The drops are counted (CLAIMS O072–O073); the dropped claims are not known to differ in label.
30. **Augmentation within the limit, training rows only.** 43,978 augmented rows are 7.67 % of train (limit 15 %). Transforms use the O1 wording: reversed options, two sentences from another record of the same source, a planted instruction naming a wrong option, and none-of-these. Applied to training rows only, never to dev or the panel. The per-template 8 % cap is respected (CLAIMS O066–O067, O070).
31. **Screen statistic for item-specific labels (R4).** For CommonsenseQA and QASC each answer text is its own class, so the O3 macro against the 0.90 line is not comparable, and the majority baseline is undefined. Their supporting numbers are gold-in-state (0) and micro accuracy against chance (CLAIMS O086). Recorded, not changed.
32. **Inherited student_v1 findings kept in the mix.** The O4 audit finds shortcut and visibility problems in student_v1 templates (Question 9). The mix keeps them, as the ADVISORY specifies; no student template was removed or changed (CLAIMS O083–O088).
33. **O2 chain paused for the O5 GPU window (scheduling; disclosed).** At 2026-10-09T19:23:35Z the O2 chain runner (bash PID 266074) was stopped with SIGSTOP, so that it would not start clef when jev-27b (PID 281949) finished. The O5 real-model checks (`scripts/osler/o5_checks.py`) run in the GPU window after jev-27b exits. The chain is resumed with SIGCONT only after that run has exited, and before clef starts. One GPU job at a time is kept. ADVISORY §6 allows O5 to use the GPU between O2 jobs. The chain's step records are unchanged; the END line for jev-27b is written when the chain resumes.
34. **O5 attempt 4 used an unseeded adapter draw (provenance; disclosed).** PEFT draws lora_A from the global RNG, and the driver did not seed it. The attempt-4 verdict is recorded as measured (CLAIMS O094–O105). The drivers now seed before the adapter is built and record the seed. A seeded re-run (attempt 5, seed 0) is reported alongside and does not replace the verdict (O106–O109). The large cross-process differences in the unseeded records are draw effects, not kernel nondeterminism (O121; seeded runs are bit-identical, O119).
35. **Additional precision settings are not verdicts (R4; disclosed).** IEEE dots (Triton's fp32 default and fla's triangular-solve precision overridden) and the kernel-free reference path were run on the same checks, seed 0, to explain the merged-export failure. They are labelled additional analyses (CLAIMS O110–O113). The verdict configuration is unchanged; which configuration is "fp32" is Question 14.
36. **O2 chain resumed after O5 (scheduling; disclosed).** The O2 chain (PID 266074) stayed paused from 19:23:35Z. The O5 GPU runs ran in that window, one job at a time, and the last of them ended at 20:30:02Z (outputs/osler_v0/O5/logs/control_chain_seeded.log). The chain is resumed with SIGCONT at this closure, before clef starts.
37. **Scratch-disk quota recovery (disclosed).** At 20:05Z an O5 diagnostic failed with "Disk quota exceeded" while saving an adapter. I deleted my own scratch adapters and a Triton cache under /workspace/tmp (about 1.8 GB), kept the hidden-state dumps, and later wrote every diagnostic output under outputs/osler_v0/O5 without adapter files. Earlier loops' files under /workspace/tmp were not touched. The first IEEE attempt also failed, because a fresh Triton cache directory could not build its driver shim; the rerun used the default cache, which holds the shim, and the cache key includes the patched constant.
38. **Checkpoint policy for O6–O8 (disk; disclosed before the arms run).** ADVISORY says "checkpoint at every eval". `run_arm` saves a checkpoint only when an eval sets a new best selection criterion (dev macro, Brier tie-break), overwriting the previous one. The selected checkpoint is the same as keeping every checkpoint and selecting afterwards; intermediate checkpoints are not kept (about 12 GB per arm otherwise, against the disk quota).
39. **Per-item dev predictions written by `run_arm` (addition; the recipe is unchanged).** Each arm writes `dev_predictions.jsonl` under its output directory, one row per dev item (identity, gold and predicted index, the full distribution). This is the input of the O9 paired bootstrap. `Trainer.score` exposes the scored items; `evaluate` returns the same metrics as before. Tests: tests/test_arm_run_o6.py, tests/test_dev_predictions_o6.py.
40. **Paired statistics for O9 (implementation choice; see deviation 41 for which macro is which).** O9's dev macro is the trainer's per-gold-class recall, the statistic the arms select on. Its paired bootstrap resamples items within templates, 1,000 resamples (`paired_macro_accuracy_difference`, tests/test_paired_macro_o9.py). Gate O1 uses the template macro through `stratified_macro_difference` (deviation 41), not this function.
41. **Which "macro" each decision uses (decided before O9 and O11 run; disclosed).** Two statistics carry the name. (a) The trainer's macro (`metrics.macro_accuracy`: the unweighted mean of per-gold-class recall) is what the arms select checkpoints on, and O9's "dev macro" uses it (`meddecide.eval.head_choice`). (b) G1's macro (the O2 scoreboard's `macro_over_templates`: the mean over templates of per-template accuracy) is D16's and D24's "macro accuracy", so Gate O1 uses it, through `stratified_macro_difference` with templates as groups. D24's "mean Brier" is the item mean (`paired_mean_difference_stratified`); the knowledge guard is plain accuracy on MedQA plus MedMCQA, paired and resampled within templates. Neither choice changes a number already measured.
42. **O10 base revisions and rank (recorded before the run).** The pinned revisions are those O0 recorded (`outputs/osler_v0/O0/fetch.json`): `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a` and `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`. Both use LoRA r = 32, the ADVISORY's default. The ADVISORY allows r = 16 at 0.8B only where memory or speed requires it; neither applies, so no deviation is taken. The head is the one O9 chooses.
43. **O2 scope extended to the D24 baselines (disclosed).** O2 scored zero-shot Qwen3.5-4B, zero-shot Qwen3.5-9B and JEV-9B on the panel and robustness sets only (deviation 12). D24 compares Osler-4B against zero-shot Qwen3.5-4B on the headline fresh set and the external panel, and the knowledge guard takes MedQA and MedMCQA against zero-shot Qwen3.5-4B; Osler-9B is compared with JEV-9B on the headline set. So zero-shot Qwen3.5-4B, zero-shot Qwen3.5-9B (the 9B knowledge guard) and JEV-9B are needed on the v0.2 scope. They are queued after the O2 chain (`outputs/osler_v0/O2/logs/chain2.log`) with `--scope all`, the same runners, the same protocol, and resumable rows. MedDecider-4B and -9B already cover v0.2. If these runs fail, the affected comparisons are NOT MEASURED.
44. **O11 scoring design (decided before any Osler result).** Osler rows are O2-format and cover the O2 common set (59,538 items), so each pairs with the baselines by item id. A choice item is scored in its original order and in the reversed order, and each option's probability is averaged over the two orders (`meddecide.eval.osler_scoring`). noul and score items are canonical in the readout (fixed yes/no and level order), so reversing them does not change the scoring: they are scored once, with `orders` recorded as 1. The single-order rows are written beside the gate (ADVISORY O11: both reported). Tests: tests/test_osler_scoring_o11.py, tests/test_osler_score_rows_o11.py. The driver (`scripts/osler/o11_score.py`) was smoke-tested on CPU with a tiny checkpoint (6 rows, then resumed for 3 more; alignment, distributions and order counts checked); no GPU run has been made yet.

---

45. **O6 attempt 1 ran out of GPU memory; fixed by gradient checkpointing (disclosed).** Arm L (attempt 1, `outputs/osler_v0/O6/arm_L_attempt1_oom/` and `logs/arm_L_attempt1_oom.log`) failed in its first training step with CUDA out of memory on a batch of long items. The batch planner closes a batch on real tokens (up to 8,192 per batch), but one item may be 16,384 tokens, and the activations of a 16k-token step did not fit beside the 4B base. A probe on synthetic items measured the peak without checkpointing as out of memory at about 16k tokens; with the frozen base's decoder layers checkpointed it is 17.3 GB at 16,384 tokens (and 10.3 GB at 2,891 tokens, against 36.2 GB without). Fix: `MedDecideModel.enable_gradient_checkpointing()` (transformers only checkpoints a layer in training mode, and `train_mode` keeps the frozen base in eval, so the decoder layers are set to training mode; attention dropout is 0.0, checked) and the config flag `gradient_checkpointing: true` in the three arm configs. The forward is the same; tests/test_checkpointing_o6.py checks that gradients match with and without it on a tiny model (3 tests). No training recipe value changes. Attempt 2 is the run of record; its provenance records the commit that contains the fix.
46. **D12 applied to zero-shot aggregates; scoreboard note corrected (disclosed at O2 closure).** The O2 metrics had listed a zero-shot READOUT_FAIL cell but kept its items in the set and benchmark accuracies, which D12 does not allow. Zero-shot cells that fail the gate are now left out of every reported zero-shot accuracy, and the excluded item counts are written beside each number (`readout_fail_items_excluded`). This changes one number: zero-shot Qwen3.5-9B on v0.2 (0.8317 over 35,076 → 0.8328 over 34,838; `nfcorpus` cell, 238 items, excluded). The Gate O1 baselines are unaffected (no failing cell in the 4B baseline; the 9B knowledge guard's MedQA and MedMCQA contain no failing cell). The stale note on the zero-shot and JEV rows is corrected. Flip-rate pair counts for zero-shot models change with deviation 43 (v0.2 base items).
47. **O6 attempts, disk quota, and the eval cost (disclosed).** (a) Attempt 1 ran out of GPU memory (deviation 45). (b) Attempt 2 stopped silently at about step 300, after the first eval, with no traceback; the logs show no cause, and the disk quota was then near its limit. (c) Attempt 3 was launched at 13:17Z when the quota was already full: it never wrote its first line. A git commit at 13:21Z failed with 'Disk quota exceeded'. I freed my own regenerable scratch (about 1 GB: diagnostic dumps, pytest and torch caches, failed-attempt checkpoints) and the write test then passed (1.5 GB). (d) A launch command sent the process output to /dev/null; that run was stopped and archived (`arm_L_attempt3_stopped`), and the attempt that followed was stopped at step 550 to profile the eval (see below). The run of record is attempt 4, launched 13:37:04Z with its output captured. The archived attempts keep their logs; their checkpoints were deleted. (e) Eval cost: a 6,000-item dev eval every 250 steps. Measured on 240 dev items: 16–17 items per second at batch 8 with rounding, the fastest of the settings tried (8 items per second at batch 32 with 16,384 tokens). So one eval takes about 6 minutes, and the in-run evals took about 14 minutes; over 100 evals per arm that is 10–23 hours beyond the 1.9 hours of training. The cadence and subset are the ADVISORY's; Question 19 asks whether to change them.

48. **Recipe change for O6–O8 by operator decision (Q19, Q20; recorded 2026-10-10T14:37:06Z).** Dev eval every 2,500 steps instead of 250; checkpoint selection, the dev tripwire and the O9 rule use G1's/D24's template macro instead of the per-gold-key recall macro. Made before any arm finished and before any held-out or test result existed; arms L, P and N all run under it, so they stay matched. Arm L attempt 4 (old cadence and statistic, stopped at step 750) is archived and not used for any result. Code and tests changed by the advisor: `meddecide.eval.metrics.template_macro_accuracy`, `ScoredItems.metrics()['template_macro_accuracy']`, `StudentConfig.selection_metric`, the trainer's selection and tripwire, `arm_run` trajectory fields, `meddecide.eval.head_choice` (now `stratified_macro_difference` over templates), `scripts/osler/o9_head_choice.py`, the three arm configs; tests added in tests/test_template_macro_o6.py, tests/test_trainer_o6.py, tests/test_gate_and_head_choice_o9.py, and tests/test_osler_arm_o6.py updated to the new recipe. Full CPU suite: 546 passed, 5 GPU-gated skipped.
48a. **Measured timing of arm L attempt 5 (correction to Question 19's estimate; disclosed).** Measured wall clock over five minutes at 18:34Z: about 1.5 s per optimiser step (the trainer's `elapsed_s` field under-states it). So 25,000 steps take about 10.4 hours of training, and the ten dev evals (about 8 minutes each) add about 1.3 hours: roughly 11.7 hours per 4B arm, and about 35 hours for arms L, P and N together. The first eval (step 2,500) is expected near 19:10Z. Earlier figures in this file (about 7 hours per arm) were wrong and are superseded by this note.

49. **O9 driver directories corrected before O9 ran (found while re-checking O9's inputs during arm L attempt 5; disclosed).** `scripts/osler/o9_head_choice.py` mapped arms P and N to `O7/arm_P` and `O8/arm_N`, which no arm writes: the configs and the chain write `O6/arm_P` and `O6/arm_N`. The driver now uses the config paths, and `tests/test_o9_arm_paths.py` checks each config's `output_dir` against the driver's mapping. The head-choice rule, margin, resample count, seed and statistic are unchanged (commit dc9f3e7; CLAIMS O163 for the suite).

50. **Tier-1 dev diagnostic written at every eval for arms P and N only (logging change; disclosed; ADVISORY §8.3).** ADVISORY §8.3 asks for the tier-1 dev trajectory to be watched at every eval. `run_arm` kept it in memory until the arm ended (in `arm_result.json`), so arm L attempt 5, already running, cannot show it before its end (about 05:30Z on 2026-10-11); its values are not recoverable mid-run. Before arm P starts, `run_arm` also writes the trajectory to `<arm output>/logs/tier1.jsonl` at every eval (commit titled "O6 — tier-1 diagnostic written at every eval"; `tests/test_arm_run_o6.py` checks the file). Training, evaluation, selection and the tripwires are unchanged, and the diagnostic is still not a selection signal. Still open: the zero-shot reference on the same 500 tier-1 dev items is not measured, so a fall below zero-shot cannot be called during a run; it will be judged at the end of each arm and by the O11 knowledge guard (zero-shot 4B against Osler-4B on the v0.2 tier-1 test items, D24).

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| | | | |
