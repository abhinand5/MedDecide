# osler_v4 — Stage B blocks (Y7)

## Operator ruling (Stage B, 2026-10-23)

Recorded verbatim, as the operator asked (STATE §7 carries the same text):

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

Applied in this loop, before any final run:

- **Item 1.** B1 enters the final mix (own lower bound +0.1581 > +0.02; knowledge lower bound −0.0134 > −0.02; C129). B2 enters
  (own +0.0475 > +0.02; knowledge −0.0163 > −0.02; C139). B3 enters if its own and knowledge lower bounds pass when its keep
  decision is written. The NOT KEPT verdicts of B1 and B2 stay as reported.
- **Item 3 (the final-mix plan).** The final mix is C1 (830,076 rows; replay 174,609) plus the entering MIMIC-family sources (B1's
  eleven MIMIC files; B2's pairs; B3's ED and MedNLI rows if B3 enters), plus B4's X1 rows if B4 is kept by keep rule v2 (judged on
  B0). The family cap is applied to the whole family, MedNLI included (the stricter reading): if the family exceeds 35 %, its rows
  are subsampled proportionally across templates by stable-hash rank, and the counts are recorded. The replay is topped up from
  osler_v3's broad pool to at least 20 % (base ids excluded), and no template may exceed 8 %. The numbers are in the Y8 claims.
- **Item 4.** MIMIC-III 1.4 is on disk: 28 files; NOTEEVENTS.csv.gz modified 2026-10-22 10:02:24 UTC; ADMISSIONS.csv.gz modified
  2026-10-22 05:22:30 UTC (C142). The NOT MEASURED reason for mimic_page_category_v1 is corrected here and in STATE §1 and §5. No
  block is added for it. The dated status records (CLAIMS C065 and the 2026-10-21 lines of STATE §6) stay as written.

Keep rule v2 (D47; `src/meddecide/train/keep_v2.py`). Each block adds one source to the last kept block (D34). The recipe is
the chosen Stage A recipe A0 (KL to base 0, 13,992 steps, pplx-decider-flash-v1.2 attention LoRA; `loops/osler_v4/stage_a.md`).

Each check is a paired percentile bootstrap within templates (1,000 resamples, seed 0, 95 % interval; `src/meddecide/eval/paired.py`),
computed as the candidate minus the base. The base is B0, the A0 predictions in `outputs/osler_v4/Y6/A0`. The verdict uses each
lower bound against its limit:

| check | partition | limit (lower bound must be) |
|---|---|---|
| own | the block's new templates, fast dev v4 (shortcut templates removed, see B1) | above +0.02 |
| MR | osler_v3's MR partition (new templates removed) | above −0.015 |
| general | every other item (new templates removed) | above −0.01 |
| knowledge | MedQA and MedMCQA (new templates removed) | above −0.02 |
| CG v2 | the CG v2 set (templates passing the readout gate for both models) | above −0.01 |

The MIMIC share is the fraction of a block's rows whose template id starts with `mimic_` (labelled; the ADVISORY Y7 cap is 35 %).
Every block's largest template share is capped at 8 %, and the replay share is at least 20 %.

## Block table

| block | adds | base | rows | replay share | MIMIC share | largest template | own | MR | general | knowledge | CG v2 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | A0 (no new run) | — | 830,076 (osler_v2 C1) | 0.2104 | 0 | not recorded | — | — | — | — | — | base |
| B1 | the eleven non-ED MIMIC structured-gold training files (427,374 rows) and a replay top-up (96,102 rows) | B0 | 1,353,552 | 0.2000 | 0.3157 | snli_nli_choice_v1, 0.0456 | +0.1737 [+0.1581; +0.1900] | −0.0346 [−0.0458; −0.0228] | −0.0119 [−0.0261; +0.0011] | −0.0050 [−0.0134; +0.0037] | −0.0214 [−0.0276; −0.0147] | **NOT KEPT** (MR, general and CG v2 fail) |
| B2 | the Y3 `mimic_doc_support_v1` pairs (20,250 rows; no top-up needed) | B0 | 850,326 | 0.2053 | 0.0238 | snli_nli_choice_v1, 0.0725 | +0.0775 [+0.0475; +0.1075] | −0.0116 [−0.0225; −0.0006] | +0.0041 [−0.0044; +0.0124] | −0.0079 [−0.0163; +0.0010] | −0.0138 [−0.0203; −0.0076] | **NOT KEPT** (MR and CG v2 fail) |

Each cell is point [lower; upper]. Item counts: B1 own 3,962, MR 3,197, general 9,973, knowledge 4,040, CG v2 8,307; B2 own 400, MR 3,197, general 13,935, knowledge 4,040, CG v2 8,307.

Claims: block mix C123; B1 training C124–C126; B1 scoring and readout C127–C128; B1 keep C129; the shortcut-included own figure C130. B2 pre-check C131; B2 mix and replay C132–C133; B2 training C134–C136; B2 scoring, readout and keep C137–C139.

### B1 notes

- **Verdict: not kept.** Three of five checks fail. The own check passes by a wide margin, so the MIMIC templates are learned on
  their own items; the regressions are on the MR, general and CG v2 partitions.
- **Shortcut check.** `mimic_rad_exam_v1` has a Naive Bayes dev screen of 0.9000 (at the D35 limit), so it is removed from the
  own check (STATE §7, a deviation recorded before the decision). With it included, own is +0.1609 [+0.1464; +0.1762] (C130,
  additional, not in the verdict).
- **Readout.** No readout cell fails on fast dev v4 (60 templates) or CG v2 (13 templates) for either model (C128), so no
  template is excluded for readout.
- **Consequence.** B1 is not part of the last kept block, so B2 builds on B0 (the A0 mix, osler_v2 C1).

### B2 notes

- **Verdict: not kept.** Two of five checks fail: MR (lower −0.0225 against −0.015) and CG v2 (lower −0.0203 against −0.010). The
  own check passes on 400 items (lower +0.0475), so the Y3 pairs are learned, but the MR and CG v2 partitions do not keep the gain.
- **Pre-check.** The naive-Bayes dev screen of the Y3 template is 0.538, and no gold label appears in the record text (C131), so
  no template is excluded from own.
- **Readout.** No readout cell fails on either set (C138).
- **Consequence.** B2 is not part of the last kept block, so B3 builds on B0 as well.
- **Training record.** The run completed 13,992 steps (C134). Its fast dev v4 evaluations were accuracy 0.7359, 0.7664, 0.7716 and
  0.7760 at steps 3,498, 6,996, 10,494 and 13,992 (C135). One logged step (1,650) has loss, CE, Brier and gradient exactly zero on a
  single correct item (a valid one-item batch; minimum batch size 1). The non-finite check is clean. Hypothesis, not separately
  tested: a saturated probability in float32.

### B3 and B4

- **B3** (ED acuity and ED disposition, 40,000 train rows each, and MedNLI, 11,232 train rows) on B0, with a replay top-up. Running:
  chain PID 1132279 (`outputs/osler_v4/Y7/logs/y7_b3_chain.sh`). MIMIC-III page category (`mimic_page_category_v1`) is NOT MEASURED:
  present but not built before B2's decision (missed re-check). The MedNLI rows carry the source label `mimic_iv` in the Y2 build; no keep partition uses it.
- **B4** (osler_v3's X1 real sources, RAGTruth and DrugProt, 69,464 train rows) on the last kept block. Pending.
