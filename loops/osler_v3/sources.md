# osler_v3 — sources (X1 real sources; X2 broad replay)

Aggregates only (R9, the public repo): counts, licences, revisions, leakage and screen numbers. No item, sentence, abstract
or case text is in this file. Every number has a CLAIMS row (`loops/osler_v3/CLAIMS.md`, ids C013–C046). Built by
`scripts/osler_v3/x1_fetch.py`, `x1_real_sources.py`, `x1_checks.py`, `x2_fetch_replay.py`, `x2_broad_replay.py`; re-derived in a
fresh process by `scripts/osler_v3/recount_sources.py` (41 checks, all agree).

## Summary

- **X1 (real sources): built and clean.** RAGTruth (MIT; human span annotations) gives three templates; DrugProt (CC BY 4.0;
  human chemical–gene relations) gives one. Train 69,464 rows, dev 7,878, test 8,832. Zero leakage hits against all twelve protected
  sets, zero overlap between train and dev or test, and no Naive Bayes screen at or above 0.90.
- **MultiCaRe builds no direct item.** Its structured fields are age and sex. The case text states age in 92 % of age-known cases and
  a gender term in 99.5 % of gendered cases, so no field qualifies (ADVISORY X1 step 3). It stays the real-text substrate of X3.
- **X2 (broad replay): built, clean, target not met.** Eleven sources (ten datasets; MathQA contributes only its non-duplicate
  problems) give 162,797 train rows, all with a licence verified on the Hub card or the README and zero leakage hits. With it, the
  A3 mix (C1 + broad replay) reaches **32.7 %** general replay after the 8 % template cap, not 35 % (see §5).

## 1. X1 sources and licences

| source | licence (how verified) | revision / file | split used |
|---|---|---|---|
| RAGTruth | MIT (LICENSE file of the release) | github.com/ParticleMedia/RAGTruth @ c103204b (response.jsonl, source_info.jsonl) | official train → train and dev; official test → test |
| DrugProt | CC BY 4.0 (dataset card) | OpenMed/drugprot-parquet @ baba427c (train, validation) | train → train; validation → dev and test by PMID (one half each) |
| MultiCaRe | CC BY 4.0 (dataset card) | OpenMed/multicare-cases @ e5b0a541 (cases.parquet) | audit only; no items |

SHA-256 of each file is in `outputs/osler_v3/X1/fetch.json`. The processed RAGTruth mirror (`wandb/RAGTruth-processed`, no licence on
its card) was not used.

## 2. X1 items per template and split (after balancing)

| template | train | dev | test | gold |
|---|---|---|---|---|
| `ragtruth_response_grounded_v1` (noul) | 12,026 | 1,370 | 1,886 | 1:1 yes/no in each split |
| `ragtruth_sentence_supported_v1` (noul) | 23,294 | 2,700 | 3,262 | 1:1 yes/no in each split |
| `ragtruth_halluc_type_v1` (choice, 5 options) | 400 | 50 | 40 | 80 / 10 / 8 per class (equal) |
| `drugprot_relation_v1` (choice, 14 options) | 33,744 | 3,758 | 3,644 | positives and negatives 1:1 |
| **total** | **69,464** | **7,878** | **8,832** | |

## 3. How the rows were made (every drop and every balancing step is counted in `outputs/osler_v3/X1/build.json`)

- **RAGTruth responses:** 17,790 read. 29 `truncated` and 144 `incorrect_refusal` responses dropped (quality is not `good`).
  Remaining 17,617 split by the official split, then 10 % of the official train contexts to dev by stable hash (1,442 dev
  responses, 13,500 train, 2,675 test). A context never crosses splits.
- **Labels:** a response is hallucinated when it has any annotated span (RAGTruth's convention). A sentence is hallucinated when a
  span overlaps it. The annotator `meta` field is never read. The `implicit_true` flag is on 1,651 of the 11,386 train-row spans and
  `due_to_null` on 1,280 (counted in `meta`, not used). Open question Q1 in STATE.
- **Sentences:** a deterministic splitter (`meddecide/mix/sentences.py`); at most 20 sentences per response (2,355 of 142,566
  sentences dropped by the cap).
- **Balancing (my decision, not in the ADVISORY):** each binary template is balanced 1:1 and the type template to its smallest
  class, by stable-hash order, in each split. This keeps a prior-only model at chance; the screens below test the rest.
- **Type template:** responses whose spans have two or more types (1,778 train, 210 dev, 194 test) are excluded, and counted as
  such. The Subtle Conflict class is rare (80 train responses, 8 test responses), so this template is small. It is additional
  evidence only.
- **DrugProt:** 4,250 abstracts (3,500 train; 750 validation split into 381 dev and 369 test by PMID). Positive pairs: one relation
  type per (chemical, gene) pair; 231 pairs with two or more types dropped. Every relation was chemical–gene (0 others). Negatives:
  same-sentence pairs with no annotated relation, sampled 1:1 with the positives of their split (train 16,872 of 47,277 candidates;
  dev 1,879 of 4,942; test 1,822 of 4,678). The 13 DrugProt relation types plus "no relation" are offered in every item.

## 4. MultiCaRe audit (no items built)

85,653 articles; 110,182 cases. Age known for 100,530 cases (91.2 %); the case text states an "N-year-old" or "aged N" phrase in 92.2 %
of those, and the first number equals the field in 90.8 % of age-known cases. Gender is Male or Female for 103,049 cases; a gender term
appears in the text of 99.5 % of them. Decision: no structured field is non-trivial to infer, so no direct item is built (ADVISORY
X1 step 3). MultiCaRe is the real-text substrate of X3 (claims and restyled pages), with its own checks there.

## 5. X2 broad replay (general human-labelled sets; D36)

| source | licence (how verified) | train rows kept (of read) | duplicates dropped |
|---|---|---|---|
| StrategyQA (`ChilleD/StrategyQA`) | MIT (card) | 1,603 (1,603) | 0 |
| QuaRTz (`allenai/quartz`) | CC BY 4.0 (card) | 2,684 (2,696) | 12 |
| Social IQa (`allenai/social_i_qa`) | CC BY 4.0 (README; loader archive at the AI2 URL, SHA-256 recorded) | 33,381 (33,410) | 29 |
| CosmosQA (`allenai/cosmos_qa`) | CC BY 4.0 (card; CSV at a pinned GitHub commit) | 25,262 (25,262) | 0 |
| AQuA-RAT (`deepmind/aqua_rat`) | Apache-2.0 (card) | 97,432 (97,467) | 35 |
| MathQA (`allenai/math_qa`, archive) | Apache-2.0 (card) | 695 (29,837) | 28,300 problems also in AQuA-RAT (normalised text); 842 options unparsed |
| BIG-bench subtasks: snarks, ruin_names, sports_understanding, causal_judgment, date_understanding (`tasksource/bigbench`) | Apache-2.0 (card) | 145, 359, 789, 151, 296 | 1 |
| **total** | | **162,797** (11 sources; 10 datasets) | |

Rules and checks:
- Labels come from the source's answer key or from the one option scored 1. Rationale, annotation, formula and explanation fields are
  never read (unit tests check this with a planted secret string).
- MathQA's problems are AQuA-RAT's train problems in tokenised form: 97.7 % of MathQA problems match an AQuA-RAT problem on normalised
  text. AQuA-RAT's copy is kept and a MathQA row is dropped when its normalised problem is in AQuA-RAT. AQuA-RAT itself repeats some
  problem texts (97,432 rows over 78,091 distinct normalised problems).
- The leakage check against every protected set (`meddecide.mix.protected`) gives **0 hits** for the broad replay (`build.json`).
- **Replay share (the measured target; ADVISORY X2):**

| mix | items | general replay | share |
|---|---|---|---|
| C1 (osler_v2 kept mix; replay = SNLI, WinoGrande, HellaSwag, CommonsenseQA, BoolQ, QASC, ARC-Easy, ARC-Challenge) | 830,076 | 174,609 | 21.04 % |
| A3 before the template cap (C1 + broad replay) | 992,873 | 337,406 | 33.98 % |
| A3 with the 8 % template cap (AQuA-RAT limited to 77,864 rows; A3 = 973,305) | 973,305 | 317,838 | **32.66 %** |

  The 8 % per-template cap binds only on AQuA-RAT (97,432 rows, 9.8 % of A3). The largest C1 template is 7.43 % (`snli_nli_choice_v1`),
  so C1 itself satisfies the cap. **The 35 % target is not met: 32.66 %, short by about 2.3 points (about 22,000 items).** The timebox
  ended with ten usable general sources; the remaining gap needs more permissive human-labelled sets (NEXT proposal).

## 6. Leakage and screens

Leakage (`outputs/osler_v3/X1/checks.json`; `outputs/osler_v3/X2/build.json`): train, dev and test X1 rows and the broad replay each have
0 hits on the normalised text keys, the bare record ids, the dataset-qualified record ids, and the PubMed ids, against: osler_v0's v0.2
dev and test, the v0.2 held-out templates, the external panel, the robustness pack, the held-out generators, the MR suite v1 test and
held-out files, the MR eval file, the knowledge dev v2, the fast dev v2 and the EHR-Rel test. Train vs dev and train vs test: 0 shared
text keys, 0 PMIDs, 0 RAGTruth contexts.

O3 Naive Bayes screen on dev (fitted on the template's train rows; `checks.json`):

| template | NB macro accuracy | majority (dev) | chance | flag (≥ 0.90) |
|---|---|---|---|---|
| `drugprot_relation_v1` | 0.3455 | 0.5000 | 0.0714 | no |
| `ragtruth_halluc_type_v1` | 0.3400 | 0.2000 | 0.2000 | no |
| `ragtruth_response_grounded_v1` | 0.6949 | 0.5000 | 0.5000 | no |
| `ragtruth_sentence_supported_v1` | 0.5826 | 0.5000 | 0.5000 | no |

The grounded template is the most learnable by vocabulary (0.69); it is below the 0.90 screen but above chance, and it is recorded.
The broad replay sources are not screened (they are general replay, not evaluation templates).

## 7. Choices and deviations (recorded in STATE §7)

1. Balancing of the binary and type templates (above): my choice; the ADVISORY does not specify it.
2. RAGTruth dev is 10 % of the official train contexts; the official test is kept as test (ADVISORY: split by context).
3. DrugProt's validation split is halved into dev and test by PMID (ADVISORY: "use the validation split as dev and test").
4. MathQA's duplicated problems are dropped in favour of AQuA-RAT's natural text (see §5).
5. The X2 replay target (35 %) is not met: 32.66 % after the cap (see §5). Recorded as a finding, not tuned.
