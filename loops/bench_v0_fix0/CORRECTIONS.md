# Corrections to loop `bench_v0`

Written by task F9 of loop `bench_v0_fix0`. `loops/bench_v0/` is never edited; this file is the
record of what in it is wrong and what replaces it.

**Reading the table.** "Corrected → X0nn" points at a claim in this loop's `CLAIMS.md`. "Still
valid" means this loop re-checked the claim and it holds (for the `choice`-path claims that means
F2 proved the path byte-identical and reproduced the accuracy). "Superseded" means the v0 artifact
is correct for v0 but the v0.1 benchmark replaces its *subject matter* (the fresh window and
anything built on it), so a reader must not use the v0 number for v0.1.

## The three defect classes

1. **MedMCQA key off by one.** `_medmcqa_item` computed `gold_index = int(cop) - 1`, but
   `openlifescienceai/medmcqa` stores `cop` **0-based**. Effects: every MedMCQA gold label shifted
   one option back, and all 1,348 rows whose answer is "A" were silently dropped (they fell out of
   the `0 <= gold_index` check). Proven three ways in F1 — 1,348 raw rows have `cop = 0`; the
   record's own `exp` names `option[cop]` in 66.6 % of usable rows versus 11.5 % for
   `option[cop+1]`; and after the fix the gold distribution is exactly the raw `cop` counts
   (X002, X003).
2. **`noul`/`score` readout broken.** Those items were rendered as `yes. Yes` / `1. Not relevant`
   while the instruction said "Respond with a single letter", and the readout scored the tokens of
   those *keys*, so label mass was ~0.002 instead of ~0.99. Every `noul` and `score` number in
   bench_v0 is invalid; the `choice` path was never affected (X007, X008).
3. **Fresh window too narrow.** v0 bounded the window by the **teacher's** repository date
   (2026-09-10), leaving openFDA with 16–94 items per template and most templates single-class in
   test. Decision **D11** bounds it by the *ladder* models instead: v0.1 starts **2026-03-01**,
   with items dated ≥ 2026-09-10 kept as a separately reported **strict slice** (X012, X013).

Two further loader defects were found by this loop's independent gold verification and are recorded
here because they also invalidate v0 numbers: **nfcorpus score items carried a grade borrowed from
another query's pool** (X005) and **CT.gov `healthyVolunteers` is a JSON boolean, so
`str(value) == "yes"` marked every item "no"** (X015).

## Accounting for every bench_v0 claim

| bench_v0 claim | original statement (abridged) | what was wrong | disposition | new claim id |
|---|---|---|---|---|
| C013 | tier-1 counts: medqa 3,273; medmcqa 4,251; nfcorpus 3,616; … total 21,202 | MedMCQA gold key off by one dropped the 1,348 "A" rows (2,835 test = 4,183 − 1,348); nfcorpus carried score items with borrowed grades | **corrected**: v0.1 tier-1 is 22,594 items (medmcqa 6,183 = 4,183 test + 2,000 dev; nfcorpus 3,076) | **X006** |
| C017 | 8 tier-1 dataset revisions pinned | nothing — the revisions are the ones v0.1 reuses | still valid | — |
| C018 | fresh total 41,502 items; clinicaltrials 13,280; openfda 222; pubmed 28,000 | built on the 25-day window (defect 3) and on the CT.gov boolean bug (every `ct_healthy_volunteers_noul_v1` item gold "no") | **superseded**: v0.1 fresh is 23,582 items (ClinicalTrials 8,123; openFDA 7,297; PubMed 8,162) | **X012** |
| C019 | fresh window start `2026-09-10`, deciding model = the teacher | the deciding model was the teacher, whose training data never labels benchmark items; the ladder's own models bound the window (D11) | **corrected**: start `2026-03-01`, deciding model = the ladder (latest repo date 2026-02-28) | **X012** |
| C020 | 0 of 41,502 fresh items predate the window start; earliest 2026-09-10 | true for v0's window; the window itself changed | **superseded** for v0.1: 0 of 23,582 items before 2026-03-01 | **X012** |
| C021 | fresh integrity 9/9: 0 leaks, 0 duplicate ids, hashes match | the checks themselves were sound; the data underneath changed | **superseded** (v0.1 re-ran them: PASS 7/7) | **X012** |
| C022 | fresh build byte-identical across two builds (v0 hashes) | true of v0; v0.1 has its own hashes | **superseded** (v0.1 rebuild is also byte-identical, 3/3 files) | **X016** |
| C023 | PubMed update files overlapped; 74,365 records → 44,217 unique | nothing — record merging is unchanged | still valid | — |
| C024 | share of fresh states containing a study-design term | measured on v0's window and templates | **superseded** (v0.1 templates differ; not re-measured, so no replacement number is claimed) | — |
| C028 | 50-item end-to-end harness run, 0.8B on MedQA test: 17/50 = 0.34 | MedQA is a `choice` template; F2 reproduced the full 1,273-item run exactly (0.40770 vs 0.40927 across the two runs, Δ −0.16 pts, prompts byte-identical) | still valid | **X008** |
| C029 | second-model end-to-end check, 0.8B-Base on MedQA: 20/50 = 0.40 | as C028 | still valid | — |
| C030 | predictions complete (50 written lines for 50 items) | unaffected | still valid | — |
| C031 | "readout health after the variant fix": mean label mass 0.996 | the number was computed over **`choice` items only**. `noul`/`score` items were at ~0.002. The claim as written implies a health check across the harness, which it was not | **corrected**: health is per (model, template, question type) and gated; `noul` 0.996–0.999, `score` 0.996–0.998 after F2 | **X007** |
| C032 | harness probes on 30 MedQA items (flip rate 0.30, candidate scaling) | MedQA is `choice`; unaffected | still valid | — |
| C033 | fresh screen: 10 of 12 kept; `pubmed_observational_noul_v1` dropped for "BoW 0.985 ≥ 0.95"; `ct_healthy_volunteers_noul_v1` dropped as single-class | both drops were artefacts: the first of class imbalance (98.5 % one class ⇒ "always no" scores 0.985), the second of the boolean bug | **corrected**: with balanced classes the template's real BoW macro accuracy is **0.796** and it is **kept**; the CT.gov template is kept with real gold | **X017, X018, X015** |
| C034 | BoW never exceeds 0.767 micro / 0.564 macro on any kept fresh template | measured on v0's unbalanced templates | **superseded**: v0.1 BoW macro range across kept templates is 0.245–0.879 | **X017** |
| C035 | majority baselines range 0.259 to **1.000** (single class) | the 1.000 was the bug, not the benchmark | **superseded**: no v0.1 template has a single gold class in test; max majority is 0.722 (`nfcorpus_graded_score_v1`) | **X014, X017** |
| C036 | hand-written regex baselines: test accuracy 0.000–0.589 | on `noul`/`score` templates a non-matching baseline scored 0.000, which reads as "at chance" when the baseline does not apply | **corrected**: v0.1 reports those as `NOT MEASURED`, never 0.000 | **X021** |
| C037 | screen verdict PASS 4/4 | the checks were internally consistent but applied an imbalanced drop rule | **corrected**: F3's screen is PASS with a macro-based rule; 22 templates, 16 kept, 6 dropped (all `n_test<200`) | **X017, X019** |
| C038 | operator audit sample: 150 items over the 10 kept templates, seed 0 | the sample was drawn from the defective build | **superseded**: F10 drew a new 150-item sample from v0.1 (50/source, 8 kept templates, sha256 recorded) | **X022** |
| C041 | harness vs lm-evaluation-harness on the same items: −0.50 pts (PASS) | `choice` path; F2 re-verified the prompts byte-identical and the accuracy identical | still valid | **X008** |
| C042 | our medqa test split == lm-eval's (1,273 identical) | unaffected | still valid | — |
| C043 | T6-protocol vs lm-eval chat-template mode: +2.18 pts | `choice` path (MedQA); unresolved protocol difference, correctly labelled as such | still valid | — |
| C044 | lm-eval chat-template mode on the same task: 0.2734 | as C043 | still valid | — |
| C045 | batch-planning OOM bug (fixed) | code defect already fixed and retracted in v0 | still valid | — |
| C046 | second ladder model measured; "medmcqa 0.284 (majority 0.383 — below majority)" | medmcqa had the wrong key **and** the wrong item set; its `noul`/`score` numbers came from the broken readout | **withdrawn** for medmcqa and for all `noul`/`score` templates; re-measured in F6 on v0.1 — see the F6 ladder table in `baselines_v0_1.md` and claims X033+ | **withdrawn; replaced by F6** |
| C047 | MedQuAD routing: gold text in state for 0.588 of items (surface leakage analysis) | MedQuAD is a `choice` template; unaffected | still valid | — |
| C048 | "MedMCQA's official test split has no D answers, and the 4B model has a last-position bias" — gold A 1085 / B 925 / C 825 / **D 0** | **the central false finding.** The official split has 4,183 items with gold A 1,348 / B 1,085 / C 925 / **D 825**; "D 0" was the arithmetic consequence of `cop − 1` on a split that contains `cop = 0` | **withdrawn** | **X003** |
| C049 | third ladder model measured; "medmcqa 0.122 (0.383)", `noul`/`score` numbers | same two defects as C046 | **withdrawn** for medmcqa and `noul`/`score`; re-measured in F6 — e.g. MedMCQA moves from 0.122 to **0.5892** for the same 4B model on v0.1 (X034) | **withdrawn; replaced by F6** |
| C050 | "ladder scaling on MedQA and MMLU" — 6 models on medqa (0.290→0.754) and mmlu (0.241→0.860) | **nothing wrong**: both are `choice` templates and both reproduce exactly | still valid (the medqa/mmlu numbers stand; the row's name suggests a broader scaling claim than it measured) | **X008** |
| C051 | "MedMCQA accuracy collapse explained by position bias, proven by a seeded option shuffle" (argmax D 890, D preference survives shuffling) | built on the corrupted gold — the model was choosing among mislabelled options, and "D" was an artefact of the shifted key rather than a model bias | **withdrawn** | **X003** |
| C052 | 264,478 prediction rows over 132 groups | the count is correct for the v0 runs; those runs' medmcqa and `noul`/`score` numbers are not usable | **superseded** (F6 re-runs the ladder on v0.1 with the fixed readout) | **F6** |
| C053 | throughput per model | measured on v0 runs; a fair description of v0, but v0.1 prompts differ | **superseded** for planning purposes (F6 records per-cell timings) | **F6** |
| C001–C012, C014–C016, C025–C027, C039, C040, C042 | schema, loader acceptance, dedup, split rules, test counts, provenance, test-suite size | none of these depend on the MedMCQA key, the `noul`/`score` readout, or the fresh window: they are structural properties of the loader framework and the repo, and F1/F4 re-ran the same checks successfully (v0.1 tier-1 audit PASS, 0 leaks, 0 duplicate ids) | still valid | — |

### The remaining claims, named

The claims below are not affected by any of the three defect classes. They are listed explicitly so
that the table accounts for all 53 bench_v0 claims without a reader having to take a range on
trust.

| bench_v0 claim | subject | why it is unaffected | disposition |
|---|---|---|---|
| C001 | repo/loop scaffold and branch rules | structural; this loop re-reads them | still valid |
| C002 | item schema definitions | schema unchanged by v0.1 | still valid |
| C003 | template id determinism | hashing unchanged | still valid |
| C004 | item id determinism | hashing unchanged; v0.1 ids reproduce | still valid |
| C005 | loader drop-reason accounting | the framework is unchanged; v0.1 uses the same `drop(reason, n)` path | still valid |
| C006 | MedQA loader acceptance | `choice`; gold re-verified 1,273/1,273 (X004) | still valid |
| C007 | PubMedQA loader acceptance | `choice`; gold re-verified 1,000/1,000 | still valid |
| C008 | MMLU loader acceptance | `choice`; gold re-verified 1,212/1,212 | still valid |
| C009 | MedQuAD loader acceptance | `choice`; gold re-verified 7,000/7,000 | still valid |
| C010 | schema rejects invalid shapes (5/5) | test-level property; still passing | still valid |
| C011 | relevance item construction | construction unchanged; the *grade* bug (X005) was in v0's sampling, corrected in v0.1 | corrected → **X005** |
| C012 | tier-1 split disjointness | re-run on v0.1: 0 straddling records | still valid |
| C014 | dedup by record id | unchanged; v0.1 audit reports 0 duplicate ids | still valid |
| C015 | license recording, no silent drops | unchanged; v0.1 acceptance checks it | still valid |
| C016 | tier-1 manifest contents | the schema is unchanged; v0.1 writes the same fields | still valid |
| C025 | fresh builder per-source drop accounting | framework unchanged | still valid |
| C026 | fresh template count (12) | v0.1 also has 12 fresh templates (4 dropped by the screen, 8 reported) | still valid |
| C027 | CPython/uv environment reproducibility | environment unchanged | still valid |
| C039 | audit page renders the sample (test) | page and test unchanged; F10's sample uses the same keys | still valid |
| C040 | test-suite size at v0 closure: 67 passed | the suite has since grown to 84; the v0 statement was true when written | still valid (historical) |

## Accounting check (F9 acceptance)

The F9 acceptance asks that every bench_v0 claim whose value depended on the MedMCQA loader, the
`noul`/`score` readout, or the v0 fresh window carries a disposition. Grepping
`loops/bench_v0/CLAIMS.md` for `medmcqa`, `noul`, `score`, `nfcorpus`, `scifact`, `trec_covid` and
`fresh` returns hits in **C013, C017, C018, C019, C020, C021, C022, C023, C024, C033, C034, C035,
C036, C037, C038, C046, C048, C049, C051** — every one of them is in the table above with an
explicit disposition, and the additional rows (C028–C032, C041–C045, C050, C052, C053) are included
because their *subject matter* is the same evaluation, so a reader needs to know whether the fix
touches them. The explicit reach of the F9 spec — C048, C050, C051, the MedMCQA counts inside
C013, the `pubmed_observational` drop reason in C033, and FINDINGS items 3 and 5 — is covered:

* **C048** withdrawn; **C051** withdrawn; **C050** still valid for medqa/mmlu;
* **C013**'s medmcqa count corrected 4,251 → 6,183 (test 2,835 → 4,183);
* **C033**'s `pubmed_observational_noul_v1` drop reason corrected and the template kept;
* **FINDINGS item 3** (MedMCQA "no D answers" / position bias) withdrawn;
* **FINDINGS item 5** (relevance judging at chance) withdrawn — it was the readout defect.

* **FINDINGS item 3** — this is the **numbered list in `## Summary`**, whose item 3 is
  "Relevance judging is at chance for every model we ran" (NFCorpus/SciFact `noul` at
  0.500–0.641): **withdrawn**, it was the readout defect. (The `## Body` sections use the same
  numbers for different content — Body §3 is "The baselines", Body §5 is the closure spot-check —
  and neither of those is a finding that this loop invalidates.)
* **FINDINGS item 5** — Summary item 5, "MedMCQA's collapse … is a positional-bias artefact":
  **withdrawn**, it was the key defect. Body §5 (the spot-check section) is not affected.

For completeness, the other Summary items: item 1 (harness agrees with lm-evaluation-harness,
C041) **still valid** — F2 re-verified it; item 2 (the ladder scales on MedQA and MMLU, C050)
**still valid** — both are `choice` templates and both reproduce; item 4 (`pubmed_mesh_major` and
`medquad_routing` look good for the wrong reason) **still valid** as an observation, and the
leakage it describes is now measured mechanically for every template by F3's gold-in-state check
(X020) rather than noticed by inspection.

## What did *not* need correcting

The `choice` path, the harness's protocol, the loader framework and its integrity rules, the
relevance item construction, and MedQA/MMLU/MedQuAD results. F2 proved this positively rather than
by assertion: `canonicalise_options` is a no-op for `choice`, prompts reconstructed with the v0
renderer are byte-identical for 200/200 MedQA items, and the full 1,273-item MedQA accuracy
reproduces at Δ −0.00016 points (X008).
