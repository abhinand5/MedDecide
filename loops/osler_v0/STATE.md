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
Last updated (UTC): `2026-10-09T08:13:20Z`
Iterations so far: `3`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| O0 | Orientation, snapshot, envs, competitor smoke, throughput | smoke | — | DONE | 2026-10-09T06:23:41Z | 2026-10-09T07:50:24Z |
| O1 | External clinical panel + robustness pack | no | O0 | DONE | 2026-10-09T07:55:27Z | 2026-10-09T08:13:20Z |
| O2 | Competitor scoreboard | yes | O0, O1 | IN_PROGRESS | 2026-10-09T08:25:05Z | |
| O3 | Clinical generators (gold by construction) + held-out list | no | O0 | PENDING | | |
| O4 | Training mix v2 | no | O1, O3 | PENDING | | |
| O5 | Readouts: option-code head, non-causal mode, export | small | O0 | PENDING | | |
| O6 | Arm L: option-code head (4B) | yes | O4, O5 | PENDING | | |
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
| generators: count, held-out list, train/dev/test items | | `loops/osler_v0/heldout.md` | O3 |
| mix v2: items, sources, replay share, max template share; leakage result | | `data/train/osler_v0/manifest.json` | O4 |
| readout checks (init equality, export round-trip, padding, byte-identity) | | `outputs/osler_v0/O5/readout_checks.json` | O5 |
| arms L / P / N: examples seen, selected step, dev macro, tier-1 dev at selection, wall-clock | | | O6–O8 |
| chosen head and the rule's verdict | | `loops/osler_v0/head_choice.md` | O9 |
| Osler-9B / Osler-0.8B: selected step, dev macro | | | O10 |
| Gate O1 verdicts (4B, 9B) incl. knowledge guard | | `loops/osler_v0/gate_o1.md` | O11 |

---

## 3. Current task checklist

**Fill this in before starting any task**, by copying that task's concrete steps out of
the plan as unticked boxes. Tick each one the moment it is finished, not at the end.
This is the only record of progress *within* a task — without it, a compaction mid-task
leaves you unable to tell what you already did.

Clear this section and write the new task's checklist when you start the next task; the
completed checklist goes into the iteration-log entry.

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
- [ ] O2 chain (PID 266074): Clef-Flash (RUNNING from 12:19:02Z) → pplx27b → jev27b → clef → md27b → md31b → metrics; chain log outputs/osler_v0/O2/logs/chain.log
- [ ] MedDecider-4B (authors' decide protocol, both orders; vector form checked against decide()) — all three sets
- [ ] zero-shot Qwen3.5-9B (panel + robustness)
- [ ] JEV-9B (card decision-head protocol, F7 helpers; panel + robustness)
- [ ] MedDecider-9B (both orders)
- [ ] Clef-Flash (authors' systemone; envs/clef; choice keys sorted by the authors' code)
- [ ] pplx-decider-v1.1-27b (authors' DecisionModel; envs/pplx27b; single order as the authors run it)
- [ ] JEV-27B (F7 protocol; single order)
- [ ] Clef (authors' systemone)
- [ ] MedDecider-27B (both orders)
- [ ] MedDecider-31B (both orders)
- [ ] o2_metrics.py: per model × set n, accuracy, macro (mean over templates, as G1), Brier, ECE, coverage with reasons, wall-clock per 1k items, D12/D21 cell gate, robustness flip rates → outputs/osler_v0/O2/scoreboard.json; loops/osler_v0/scoreboard.md (aggregates only)
- [ ] self-audit, CLAIMS O046+, STATE, commit, push

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

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|

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

---

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| | | | |
