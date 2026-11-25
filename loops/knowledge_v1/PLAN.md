# knowledge_v1 — synthetic medical and clinical general-knowledge data at scale, for the 4B and 9B students

Status: PLAN for DeepSeek to execute as the main loop. Decisions D61 (local teacher d3-ultra), D28 (MIMIC local-only) apply.
Teacher: vllm-sr/d3-ultra, local (revision ccdcfa611649a59e095e0b5e1732a15cedafadd6).
Students: vllm-sr/d3-mini (4B) and vllm-sr/d3-flash (9B), per the d3_distill plan.

## 1. Goal and the honest odds
Goal: general medical and clinical knowledge on public benchmarks (knowledge test, the 7-set panel, MedNLI) above zero-shot
pplx-flash, and on the MR and MIMIC audit suites within the gate's margins.
What the evidence says (aggregates only): knowledge items distilled from the teacher already move the 4B and 9B knowledge
scores on dev; the panel is where we lose. The gap is breadth of clinical reasoning, not volume of multiple-choice items.
Odds (judgement, R10): a student that beats zero-shot pplx-flash on the panel lower bound: about 1 in 2 with this plan; a
student above the teacher on knowledge: unlikely. "Astronomical" means breadth and volume with a verified gold label, not an
unchecked number of items.

## 2. Sources (licence-checked before any use; only what is public-domain, openly licensed, or derived by code from such sources)
| source | use | licence check |
|---|---|---|
| PubMed citations and abstracts | study design, population, outcome and evidence-type questions; vignette facts | NLM terms: free use of citations; check abstract reuse terms before training at scale |
| MeSH descriptors and tree | disease-category, mechanism-category and drug-class questions; distractors from sibling branches | public domain (US government) |
| ICD-10-CM and ICD-10-PCS titles, MS-DRG titles | coding hierarchy and procedure questions | public domain (CMS, CDC) |
| RxNorm | ingredient, class and dose-form questions | NLM terms; check before scale |
| openFDA drug labels | indication and contraindication questions from label sections | public domain (openFDA terms) |
| MedMCQA and MedQA training splits | teacher soft labels, gold kept | official training splits only; never test or dev |
| synthetic families with no MIMIC content | lab interpretation, medication indication, triage (already built) | code-built |
| MIMIC-derived structured items | local teacher only (D28) | local only; never leave the machine |
Excluded: SNOMED CT and UMLS strings (licence restricts to self-hosted models; keep out of any hosted use, and avoid for training data
unless the licence is checked), CPT descriptors, 3M APR-DRG descriptions, InterQual or MCG content, textbooks and guidelines (copyright),
LiveQA and HealthSearchQA (not openly licensed).

## 3. Families (gold by construction: code computes the label from structured fields; the teacher writes text only)
F1 Evidence design from PubMed metadata: given a study's facts (publication type, design terms, population size class, comparator),
ask the design, the evidence level, or whether a claim in the abstract is supported. Gold from metadata. Facts from the abstract.
F2 Mechanism and category from MeSH: a case built from a descriptor's tree position; question asks the parent category; distractors
are siblings in the same branch (hard negatives, drawn by code).
F3 Drug indication and contraindication from openFDA labels: a patient vignette built from facts; question asks the indicated
class or the contraindicated condition; gold from the label section; distractors from the same drug class.
F4 Coding hierarchy from ICD-10-CM and MS-DRG titles: which code family, which MS-DRG severity split, which parent category.
F5 Multi-step clinical reasoning composed from F1–F4 facts (two facts combined: a finding and a rule), gold from the composition code.
F6 Existing families: lab interpretation, medication indication, triage (synth_v1), and the MedMCQA/MedQA training items (teacher soft labels).

## 4. Generation pipeline (per item)
1. Code selects the source record and computes the facts and the gold label. Seeded; stable hash id.
2. Code selects distractors: hard negatives from the same branch or class, deduplicated by code and title; the count is 4 or 5 options.
3. Render the teacher prompt with the facts only (never the label). The teacher writes the case text, a length target of 80–200 words.
4. Verify in code: every fact re-extracts from the text (rules as in src/meddecide/synth/verify.py); the label string and its
   sentence forms do not appear; no meta-token leak; text length within the range.
5. Teacher probability vector over the offered options is recorded as the soft label; gold is kept alongside.
6. Leak probe: a second local call asks which option the text supports, without the label; recovery above chance on a family is
   reported and the family's facts are reviewed (not the label).
7. Dedupe by text hash and by case id; overlap checked against every frozen test set by text hash (must be 0).

## 5. Scale and throughput
- Target tiers: 1 M items first; 5 M only if the 1 M tier beats the gate on dev. The teacher runs at the batched rate from the
  d3_distill step 0 gate (target ≥ 25 items/s). At 25/s, 1 M items is about 11 hours of teacher time; 5 M is about 56 hours.
- Each family is capped at 25% of the synthetic block; no family above 30% of the mix.
- Keep rate and leak rate are recorded per family and per day. A family with keep rate below 50% is reworked before more items.

## 6. Mix and training
- Mix: the d3_distill mix (distillation labels on MedMCQA and MIMIC-grounded items) plus the knowledge_v1 block, capped at 40% of
  rows for the synthetic blocks together; replay of general public items at least 20%.
- Two students (d3-mini, d3-flash), one arm each at 13,992 steps, then the final at 63,304 steps for the better arm, using the
  distill_v0 recipe with the D57 dosing fix and per-item weighting (D51). Calibration as before.
- Selection on dev only: the knowledge dev (MedQA and MedMCQA dev splits, not test), the CG v2 dev proxy, the MR and MIMIC dev
  partitions. The frozen test sets are never used for choices.

## 7. Evaluation
- Gate O7 as in distill_v0, unchanged: panel lower bound > 0 against zero-shot pplx-flash; knowledge lower > −0.02; MR and MIMIC
  within −0.02 of the osler_v4 final.
- Additional rows (labelled): MedNLI test, PubMedQA labelled test, the teacher's own numbers on the same sets, and the panel per dataset.
- A student that beats the gate on the panel but not on knowledge, or the reverse, is reported as such (R4).

## 8. Checks that can stop the plan
- Leak probe above chance on a family after review of its facts: that family is dropped.
- Any test-set overlap above zero: stop and remove the overlapping items; report.
- Teacher probability check failures above 2% in a day: stop that family.
- Any MIMIC content outside the local GPU: stop.
- The 1 M tier below the dev gate after the 9B arm: no 5 M tier; report the result.

## 9. Timeline (planning estimates)
- Week 1: sources and licence checks; families F1–F5 built; 1 M-item label set (about 2 days of teacher time with checks).
- Week 2: students trained (4B, then 9B); arms at 13,992 steps; the final for the better arm; gate and report.
- Scale to 5 M only if the 1 M tier passes its dev gate.

## 10. Decisions needed from the operator before week 1
1. Confirm the PubMed and RxNorm abstract and citation terms (a check, not a decision: read the terms and record them).
2. Confirm that SNOMED and UMLS strings stay out of the training data.
3. Confirm the 1 M tier and the 5 M contingency.
