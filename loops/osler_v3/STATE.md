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

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (X12), or when no PENDING task can proceed without the operator -->
Run started (UTC): `2026-10-18T20:13:37Z`
Last updated (UTC): `2026-10-18T20:40:13Z`
Iterations so far: `1`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| X0 | Orientation, KL timing | small | — | DONE | 2026-10-18T20:13:37Z | 2026-10-18T20:40:13Z |
| X1 | Real sources: RAGTruth, DrugProt, MultiCaRe | no | X0 | DONE | 2026-10-18T20:26:00Z | 2026-10-18T20:40:13Z |
| X2 | Broad replay for A3 | no | X0 | DONE — 35 % replay target NOT MET (32.66 % after the 8 % cap; recorded, D-X2-2) | 2026-10-18T20:31:00Z | 2026-10-18T20:40:13Z |
| X3 | Gemma-crafted items over real text | yes | X0, X1 | IN_PROGRESS — claims template BLOCKED (D-X3-1, D44(b)); restyle build running (D-X3-2) | 2026-10-18T20:37:00Z (first X3 file written) | |
| X4 | Charts with the D44 rule | yes | X0 | PENDING | | |
| X5 | MIMIC (official files only; only when they exist) | no | X0 | BLOCKED — official MIMIC files absent (data/physionet/ missing; D37 forbids unofficial copies); not waited for | 2026-10-18T20:13:37Z | 2026-10-18T20:40:13Z |
| X6 | Fast dev v3, CG proxy, new baselines | yes | X1, X2, X3, X4 (X5 if built) | PENDING | | |
| X7 | Stage A: recipe ablations A0–A3 + choice | yes | X6 | PENDING | | |
| X8 | Stage B: data blocks | yes | X7 | PENDING | | |
| X9 | Final Osler-4B | yes | X8 | PENDING | | |
| X10 | Evaluation + Gate O3 (Osler-4B; 9B deferred) | yes | X9 | PENDING | | |
| X11 | Memorisation audit (only with MIMIC) | small | X9 | PENDING | | |
| X12 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (order exceptions in ADVISORY §5). Never run two GPU jobs
at once. A `BLOCKED` task does not block unrelated work. No row may be left `IN_PROGRESS` at closure.

---

## 2. Key values discovered during the run

| key | value | source | task |
|---|---|---|---|
| KL timing: s/step, anchor share, peak (pplx, kl 0.5) | control 0.7935 s/step; KL 0.5 1.0130 s/step (+27.7 %, mean after warm-up; median +37.8 %); anchor share 0.8775 of C1 (728,351 of 830,076; 0.8765 in the 150-step run); peak 57.08 GB in both arms; mean KL 0.2624 | `outputs/osler_v3/X0/` (kl_0, kl_0p5, anchor_share_C1.json, SELF_AUDIT.md); CLAIMS C001–C010 | X0 |
| real sources: items per template, licences, screens, leakage | train 69,464 / dev 7,878 / test 8,832 rows (RAGTruth MIT: grounded 12,026, sentence 23,294, type 400; DrugProt CC BY 4.0: 33,744); leakage 0 hits (12 protected sets); NB screens max 0.695 (none ≥ 0.90); MultiCaRe: 0 items (age and sex stated in the text) | `loops/osler_v3/sources.md` | X1 |
| broad replay: datasets, licences, A3 replay share | 162,797 rows from 10 datasets (11 template sources), all licences verified on card or README (MIT, CC BY 4.0, Apache-2.0); leakage 0 hits; C1 general replay 21.04 %; A3 33.98 % before the 8 % cap and 32.66 % after (target 35 % not met) | `loops/osler_v3/sources.md` §5; CLAIMS C033–C041 | X2 |
| Gemma-crafted: pilot keep / agreement rates; items built | claims (after the one prompt fix): kept 272 of 300 (90.7 %), verifier agrees on 90 % of contradicted, 105 of 106 not-mentioned; claims BLOCKED on the phrase check (D-X3-1). Restyle (original prompt): kept 286 of 300 (95.3 %); build pending (chain) | `outputs/osler_v3/X3/` (build.json, readthrough_*.json) | X3 |
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
Task in flight: X3 — Gemma-crafted items over real text (restyle build; claims BLOCKED)
Working dir:    outputs/osler_v3/X3/
Running jobs:   restyle build chain pid 3259323 (train 6000 docs, then dev 500, then test 1000), log outputs/osler_v3/X3/logs/restyle_chain.log (started 2026-10-18T21:07:16Z); vLLM Gemma server pid 3254457 on 127.0.0.1:8000 (started 2026-10-18T20:49:17Z). Restarting after a crash: the generator is resumable (same command, the done keys are skipped).

- [x] ruff and pytest gate (padding test deselected on the command line only): ruff clean; 760 passed, 1 skipped, 1 deselected, exit 0 (outputs/osler_v3/X0/logs/pytest_gate.log)
- [x] MIMIC official files absent (data/physionet/ missing, expected)
- [x] envs/vllm serves Gemma 4 31B (osler_v2 W3b command); one chat reply checked (finish stop); server stopped, GPU 0 MiB
- [ ] KL timing: 150 steps on the C1 rows, pplx, kl_to_base 0.5, the §3.2 prefixes; s/step, anchor share, peak GB (control kl 0 also run, same steps)
- [ ] acceptance check run and passed
- [ ] self-audit (R6) written to SELF_AUDIT.md
- [ ] CLAIMS.md rows appended
- [ ] STATE updated, committed, pushed
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

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|
| X3-claims | clinclaim_grounding_v1 (all claim rows; the claims template) | label-tracking phrase 'family' / 'family members' in 39 % of not-mentioned items vs 0 % of the others, after the one allowed prompt fix (D44(b); D-X3-1) | a second prompt change is not allowed by the ADVISORY; a new loop could redesign the not-mentioned instruction and re-pilot |
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

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| | | | |
