# MedDecide — Program Plan

**Status:** authored 2026-10-06 from the planning session between the operator
(Abhinand) and the advisor (Claude). This is the long-term plan. Each stage below runs
as one autonomous loop (`loops/<name>/`) and ends at a **hard stop** for operator +
advisor review. The next loop is planned only after that review.

---

## 1. Goal

Build the open medical decision model the field cites: **MedDecide**, a calibrated,
typed "System One" model for medicine, plus **MedDecide-Bench**, a public,
regenerable, contamination-resistant benchmark. Ship weights, benchmark, and code;
write it up as a solo paper (venue decided later — research quality first).

**What "SoTA" means here** (decided, do not reopen inside a loop):

- Best accuracy **and** calibration (Brier, ECE) on MedDecide-Bench **tier 2 (fresh)**
  at every size tier of the ladder, against every open decision model and zero-shot
  base model we can run.
- Headline target: the ≤4B MedDecide beats 27B-class general decision models
  (JEV-27B, pplx-decider-v1-27b, GEV-26B-Decide) on the fresh tier — the pattern
  MedJev showed on a narrow task, generalised to a broad medical skill mix.
- Tier 1 (established test sets) is reported for comparability, with a contamination
  probe beside every number.

**Working name:** MedDecide (the operator picks the final name after results; shortlist
from planning: Iaso, Paeon, Aegle, Epione, Telesphoros). Avoid "MedJev" — an existing
model (`junma/MedJev-Qwen3.5-0.8B`) and "Jev" is a trademark. **Update 2026-10-09:** the
operator calls the model series **Osler** (Osler-0.8B / -4B / -9B); MedDecide remains the
repo and benchmark name.

---

## 2. Background the agent needs (condensed)

A decision model renders a prompt with the state, the question, and the options, then
reads the probability of each option from **one forward pass** (no decoding). What is
genuinely new versus zero-shot classification: calibration as a trained property
(Brier / log-loss / distillation from a calibrated teacher / temperature fitting),
schema-agnostic options defined at request time, many questions per state, and serving
stacks built for prefill-only readout.

Readouts: **verbalizer** (one-token logprobs over option letters — JEV), **pointer head**
(a small head scores option indices directly — MedJev), **joint schema head** (options
cross-attend to each other and to the state — Clef), **per-option `[MASK]`** (encoders —
Laya, Julia-1, GLiNER2.5-Decide).

Known failure modes to measure, not assume away: confidently wrong (high label mass,
wrong argmax), option-order sensitivity, collapse at high option counts, poor rejection
when the right answer is missing, thresholds that do not transfer, distillation
inheriting the teacher's errors, domain fine-tuning costing generality.

Numbers in the operator's survey (`STATE_OF_DECISION_MODELS_05102026.md`, kept outside
this repo) are vendor or third-party claims. **Unverified** — every baseline we cite is
re-measured on MedDecide-Bench.

---

## 3. Settled decisions (decision log)

| # | decision | date | rationale |
|---|---|---|---|
| D1 | MedDecide first; retrieval (MedColBERT/MedEmbed) is a parallel track that later uses MedDecide as its relevance judge | 2026-10-06 | decision models are the timely contribution; one judge serves both |
| D2 | Ladder: LFM2.5-350M → Qwen3.5-0.8B (+Base) → 4B (MedGemma-1.5-4b-it vs Qwen3.5-4B bake-off) → Qwen3.5-9B | 2026-10-06 | operator choice; all verified on HF; MedGemma licence permits derivatives (operator-checked) |
| D3 | Architecture: frozen base + decision-path LoRA + pointer head ("Blocks of Experts" — base generation untouched) | 2026-10-06 | proven medical recipe (MedJev); clean paper point: generation byte-identical |
| D4 | Teacher: DeepSeek-V4.1-Flash, self-hosted on 4×RTX PRO 6000 (PLE offloaded to RAM) | 2026-10-06 | operator choice; self-hosting gives full option logprobs and allows credentialed data |
| D5 | Agent: DeepSeek via official API / OpenRouter, running on the 1×PRO 6000 pod | 2026-10-06 | operator choice; agent never reads credentialed text |
| D6 | Benchmark: tier 1 established + tier 2 fresh with structured-field gold; fresh v0 sources = ClinicalTrials.gov, openFDA labels, PubMed | 2026-10-06 | contamination-proof, regenerable ("living") benchmark; minimal work |
| D7 | Release: weights + benchmark (public sources) + code open; training data private | 2026-10-06 | allows UMLS/MIMIC-derived training data |
| D8 | MIMIC: main public model is MIMIC-free in training but evaluated on MIMIC-derived sets; a separate **MedDecide-clinical** trains on MIMIC, released via credentialed access | 2026-10-06 | keeps the headline release unblocked |
| D9 | Control: no cash cap; every gate is a hard stop for operator + advisor review | 2026-10-06 | operator steers between loops |
| D10 | Human audit: the operator audits ~150 fresh-tier items per benchmark build via a keyboard-driven local HTML page | 2026-10-06 | validates templates, not every label |
| D11 | Fresh window start is bounded by the **ladder** models (our bases), not the teacher: v0.1 starts 2026-03-01 (latest ladder date 2026-02-28). Items dated ≥ 2026-09-10 form a **strict slice** reported separately | 2026-10-06 | the teacher never labels benchmark items; later baselines seeing records can only advantage them (conservative for us); v0's 25-day window left openFDA templates with 16–94 items |
| D12 | **Readout-health gate** on every reported (model, template) cell: median label mass ≥ 0.5, greedy agreement ≥ 0.9, accuracy CI not below chance; failing cells are `READOUT_FAIL`, never accuracies. Every question type needs its own reference validation | 2026-10-06 | bench_v0 reported a broken `noul` readout and an off-by-one MedMCQA key as findings |
| D13 | Gold-only training data for loop 1: official tier-1 **train** splits + **pre-window** structured-gold items (same builders, records dated before 2026-03-01). No LLM labels, no teacher, no UMLS/MIMIC, no HF datasets outside the vetted list | 2026-10-06 | exam train splits alone teach exam MCQ, not the fresh tier's decision types; pre-window records give unlimited gold with a clean time split |
| D14 | **Held-out templates** are excluded from training entirely and reported separately; loop 1 holds out `ct_phase_choice_v1`, `fda_boxed_warning_noul_v1`, `pubmed_humans_noul_v1` and one record–claim consistency template | 2026-10-06 | a general decision model must be measured on decision types it never trained on |
| D15 | **Record–claim consistency** is a first-class template family (document-grounded verification): a stated attribute is supported or a near-miss swap; role-binding designs so a string-presence rule cannot solve it | 2026-10-06 | verification against a document is the central real-world decision; gold is known by construction |
| D16 | **Gate G1:** on the headline fresh set (v0.2 kept templates excluding the superseded `pubmed_mesh_major_choice_v1` / `fda_class_choice_v1`), for each baseline B ∈ {zero-shot Qwen3.5-0.8B, JEV-9B}: item-paired bootstrap 95 % CI (1,000 resamples, items within templates) of MedDecide − B in macro accuracy and in mean Brier. PASS iff accuracy CI lower bound > 0 **and** Brier CI upper bound < 0 against **both** baselines on **seen** templates; held-out templates reported with the same rule, separately | 2026-10-06 | fixed before results so the claim cannot drift |
| D17 | Training sources extended: official train splits, pre-window structured-gold items, and catalog (`docs/benchmark/dataset_catalog.md`) datasets with **human or structured** labels and a training-compatible licence. LLM-labelled datasets (e.g. `LocalLLaMA/typed-decisions`) are never training data; they may validate tooling | 2026-10-08 | student_v0 did not transfer to held-out decision types; diversity is the first lever |
| D18 | Loop 1b compares three decision heads under matched conditions: pointer head, letter-readout LoRA, and a Clef-style joint head via Unsloth (`FastDecisionModel` / `DecisionTrainer`, its own environment). The joint head is no longer gated on pointer-head failure | 2026-10-08 | operator decision; the paper's architecture claim needs the plain-LoRA control and the published joint-head recipe |
| D19 | G1 baselines are exactly zero-shot `Qwen/Qwen3.5-0.8B` and JEV-9B (D16). Ablations and sibling arms are reported as additional, never in a verdict; model selection and temperatures never use held-out templates | 2026-10-08 | student_v0's `g1.md` put the Base ablation in the verdict |
| D20 | Training sources extended again: (a) **gold-by-construction generated decisions** — items whose text is assembled by code from structured parts and whose label is computed by the same code (clinical notes with minimal pairs, eligibility with an "insufficient information" option, policy-defined triage, code assignment with "none of these"); no LLM writes any text or label; (b) **general human-labelled decision data** as replay (permissive licence, recorded per source). Some generators are held out entirely and used only for evaluation | 2026-10-09 | student_v1: no arm transferred to held-out types and decision-path knowledge fell to the majority-class rate; the leading open medical decision models train on generated clinical notes plus general replay |
| D21 | **D12 for trained readouts.** Zero-shot cells keep D12 unchanged. For a trained model whose readout normalises over the offered options only (a separate readout, pointer or joint head), the gate is: constant-answer check, greedy agreement ≥ 0.9 where a greedy path exists, accuracy CI not below chance. Full-vocabulary label mass is reported as a diagnostic, never as a gate | 2026-10-09 | student_v1 arm B failed D12 on 8 of 11 fresh templates on label mass alone while its top token was an offered letter in 152/152 probe items (V100, V101) |
| D22 | **Scope after loop 1b:** the targets are Osler-4B (`Qwen/Qwen3.5-4B`) and Osler-9B (`Qwen/Qwen3.5-9B`); Osler-0.8B (`Qwen/Qwen3.5-0.8B`) is trained with the same recipe as a reference point in the release. The MedGemma bake-off and LFM2.5-350M are dropped from the training ladder (zero-shot rows stay as they were). Supersedes D2 for training | 2026-10-09 | operator decision; Qwen3.5-4B is far ahead of the other ≤4B bases zero-shot on v0.1/v0.2, and the closest open medical competitors share these bases |
| D23 | **Default readout: option-code readout with a separate head** (pplx-style): a `[K, d]` readout matrix (K ≤ 255 option codes), rows initialised from the base `lm_head` rows of the code tokens, trained with the decision-path LoRA, softmax over the valid codes of the current question, temperature per question type applied once. It exports to a plain `lm_head` (row i → token id i) so standard serving stacks reproduce it. The base stays frozen; the LoRA is applied on the decision path only. Pointer head and non-causal full attention are matched ablations at 4B | 2026-10-09 | operator + advisor review: student_v1's heads were within 0.6 pts at 0.8B; the letter readout starts from a readout the base already has and ships through standard stacks; the separate head removes the off-option mass that failed D12 |
| D24 | **Gate O1** (osler_v0): for Osler-4B vs {MedDecider-4B, zero-shot Qwen3.5-4B} and Osler-9B vs {MedDecider-9B, JEV-9B}: item-paired bootstrap 95 % CI (1,000 resamples, items within datasets/templates) of Osler − B in macro accuracy and in mean Brier. PASS iff accuracy CI lower bound > 0 **and** Brier CI upper bound < 0 against **both** baselines on **(i)** the headline fresh set, seen templates (as D16) **and (ii)** the external clinical panel; plus a **knowledge guard**: on MedQA + MedMCQA, the accuracy CI lower bound of Osler − zero-shot same-size base > −0.02. Held-out templates, held-out generators and the robustness pack are reported with the same rule, separately. Other models (27B/31B decision models, sibling arms) are reported as additional, never in the verdict | 2026-10-09 | fixed before results; extends D16 to the sizes and competitors the program now targets |
| D25 | From `osler_v0` the loop agent is Claude Haiku 5.5 (`claude-haiku-5-5`) running in Claude Code on the 1×PRO 6000 pod; the advisor (plans, reviews at hard stops) is Claude Opus. Supersedes D5 for the agent | 2026-10-09 | operator decision |
| D26 | **Data first; architecture frozen.** After osler_v0 the readout stays the option-code head (D23; the dev rule kept L) and no further head or attention ablations run until the data work is done. Loops compare **data blocks** with short, matched 4B runs and keep a block only by a pre-registered dev rule; the 9B is trained once per loop, at the end | 2026-10-13 | osler_v0: three heads within 0.5 pt; every gate failure (external panel, held-out templates, knowledge) traced to data; the arms cost 35 GPU-hours |
| D27 | **Training text may be written by the teacher; labels never are.** Amends D13/D20: a training item's label is computed by code from structured facts or taken from a structured source field, as before; its text may come from (a) real records (MIMIC, D28), (b) code rendering of real or simulated values, or (c) teacher prose written from the code's facts and **kept only if a second teacher call extracts the same facts back** (`meddecide.mr.teacher_prose`). Teacher-written labels, teacher option distributions and teacher judgements of correctness remain excluded | 2026-10-13 | operator decision; osler_v0's code-templated text did not transfer (external panel loss, held-out templates answered with one option) |
| D28 | **MIMIC enters training.** MIMIC-III, MIMIC-IV, MIMIC-IV-Note and MIMIC-IV-ED (operator credentialed) are training and evaluation sources for the public model. Processing is local only: MIMIC text or values never go to a hosted service (the hosted teacher receives code-generated facts only; `teacher_prose` refuses credentialed sources unless the teacher is local). Splits are by patient; MIMIC-IV evaluation uses patients with `anchor_year_group` from 2014 (no overlap with MIMIC-III's years). MDACE is evaluation-only. A memorisation / PHI-leak audit precedes any release. Supersedes "MIMIC not before the clinical loop" | 2026-10-13 | operator decision; precedent: BioClinical-ModernBERT (MIMIC-III + MIMIC-IV-Note, MIT, ungated); PhysioNet's responsible-use note concerns hosted APIs, not weights |
| D29 | **Target domain: medical-record audit and payment-integrity decisions**, alongside general clinical decisions. First task families: record page category (eleven classes + other; vitals pages are flowsheets), ICD-10-CM code support (which code the documentation supports, by the Official Guidelines), clinical-indicator evidence (published thresholds for sepsis, acute respiratory failure, AKI, hyponatraemia). Claim edits (NCCI/MUE), DRG validation and coverage criteria follow in a later loop. CPT descriptors (AMA) and InterQual/MCG criteria are never used | 2026-10-13 | operator decision (the open-source audience of payment-integrity vendors) |
| D30 | **Teacher service.** The teacher (DeepSeek-V4.1-Flash) may be a hosted endpoint (OpenCode Go or the DeepSeek API, zero data retention) for prose from code-generated facts; anything that reads MIMIC text uses a self-hosted teacher. Credentials only from environment variables (`TEACHER_BASE_URL`, `TEACHER_API_KEY`, `TEACHER_MODEL`, `TEACHER_LOCAL`, `TEACHER_EXTRA_HEADERS`) | 2026-10-13 | operator decision |
| D31 | **Gate O2** (osler_v1), fixed before results: Osler-4B vs {MedDecider-4B, zero-shot Qwen3.5-4B}, Osler-9B vs {MedDecider-9B, JEV-9B}; item-paired bootstrap 95 % CI (1,000 resamples, within templates) of Osler − B, accuracy = template macro (G1), Brier = item mean. PASS iff **(i) MR audit suite** (the osler_v1 MR test sets, generated and MIMIC parts pooled by template): accuracy lower bound > 0 and Brier upper bound < 0 against both baselines; **(ii) external clinical panel** (osler_v0's): accuracy lower bound > −0.02 against both baselines (non-inferiority; superiority reported separately); **(iii) knowledge guard** as D24. Additional rows (d1-3B, siblings, larger models) never enter a verdict | 2026-10-13 | the loop's question is record-audit skill without losing general clinical skill; the panel moves from superiority to non-inferiority because this loop adds no broad-clinical data (operator to confirm before V9 runs) |
| D32 | **Synthetic-first; MIMIC when approved.** PhysioNet approval takes about two weeks, so osler_v1 does not wait for it. Pre-MIMIC sources: **Synthea** (MIT; `richardyoung/synthea-575k-patients`, streamed, never downloaded whole) for encounter values, medications and charges rendered as record pages and as facts for teacher narrative pages; **MTSamples** (licence uncertain; operator accepts it) for real note-type text; **CodiEsp** (CC BY 4.0) for ICD-10 coding of real clinical cases; **PMC-Patients** (CC BY-NC-SA 4.0 overall; operator chose all of it, the `comm`/`noncomm` subset is recorded per row) for citation-linked patient similarity; NHS synthetic notes (MIT) and the CC BY clinical-case ICD-10 set as evaluation only. Unofficial re-uploads of MIMIC on the Hub are never used (DUA). MIMIC (D28) joins as a block when the files exist, in this loop or the next | 2026-10-13 | operator decision after the Hub survey; PhysioNet review time |
| D33 | **Attention-only LoRA is the default adapter** from osler_v2: LoRA on the attention projections only (osler_v1 B0b's targets), the readout unchanged (D23). Block runs and final runs use it | 2026-10-17 | osler_v1: knowledge dev −0.008 [−0.042, +0.026] (B0b) vs −0.060 [−0.102, −0.020] (B0) at equal dev template macro; the final Osler-4B (attention-only) passed the knowledge guard at +0.044 [+0.032, +0.055] on 5,456 test items, where osler_v0's all-layer 4B failed it |
| D34 | **Block design v2.** Each candidate block adds one source to the **last kept** block (not to every earlier candidate), so a rejected source never rides along. The fixed dev set is assembled once, before any block runs, and holds the dev items of **every** candidate source. The knowledge dev set uses all MedQA and MedMCQA dev items available in the mix's dev splits, at least 4,000 items, so the −0.02 margin can be met by a model equal to its base. The keep rule (three lower bounds: MR > 0, general > −0.01, knowledge > −0.02) is unchanged | 2026-10-17 | osler_v1: B3 contained B2's rejected prose; B2 was judged on a dev set without its own templates; the 500-item knowledge check failed a model 0.8 points from its base |
| D35 | **MR suite v1 and Gate O3** (osler_v2), fixed before any osler_v2 result. The MR suite v1 is the osler_v1 MR test suite plus the new sources' test templates, **minus** (a) any template on which zero-shot Qwen3.5-4B scores ≥ 0.97 on its dev items (ceiling), and (b) any template whose O3 Naive Bayes screen on dev is ≥ 0.90 (vocabulary shortcut); both are reported, never in the verdict. Gate O3 = D31's three checks on the MR suite v1, the external panel (non-inferiority, lower bound > −0.02) and the knowledge guard, for Osler-4B vs {MedDecider-4B, zero-shot Qwen3.5-4B} and Osler-9B vs {MedDecider-9B, JEV-9B} | 2026-10-17 | osler_v1: three templates at 1.000 and a vocabulary-solvable template (42 % of the MR test items; word overlap AUC 0.97) dominated the MR suite |
| D36 | **Replay at 20 % or more**, with share-alike sources (CC BY-SA: ARC, BoolQ, SNLI) allowed for training: the training data is never distributed (release policy), so the share-alike clause does not reach the weights. Licences are recorded per source | 2026-10-17 | osler_v1 replay stayed at 12.6–15.3 % and knowledge fell as the replay share fell (B2 −0.022, B3 −0.040 against B1) |
| D37 | **No unofficial copies of credentialed data, ever** — including Hub datasets that reproduce MIMIC tables or notes (e.g. `lucky9-cyou/MIMIC-IV`, `ricardosantoss/ehrcomplete_icdfiltered`), before or after the operator's PhysioNet approval. MIMIC enters only from the official PhysioNet files (D28) | 2026-10-17 | the operator's PhysioNet approval is pending; these copies would breach the data-use agreement and taint the release |
| D38 | **Local teacher and whole synthetic charts.** Gemma 4 31B (`google/gemma-4-31B-it-qat-w4a16-ct`, Apache-2.0) served locally by vLLM, with the operator's serving setup and the `gemma-4-31B-it-assistant` draft model, is a teacher for prose from code-built facts (local, so no budget cap and MIMIC-eligible once official). Its first use is whole multi-page charts for chart-level decisions (`meddecide.mr.charts`): code pages carry every measured fact, narrative pages carry none (checked by re-extraction), and labels are computed by code. A chart block is kept only by the keep rule; its effect on real-text dev templates is reported beside it | 2026-10-17 | operator decision; osler_v1: synthetic single pages did not transfer to real text, so charts are judged on real-text dev |
| D39 | **pplx-decider-flash-v1.2 is a candidate base for Osler-4B**, chosen on dev. `perplexity-ai/pplx-decider-flash-v1.2` (Apache-2.0) is Qwen3.5-4B (our revision) distilled from pplx's 27B decision models, with noncausal full attention and a 255-row readout; ported as `PplxDecisionModel` (token-identical prompts and 40/40 argmax parity with its reference code). Rule, fixed before any osler_v2 result: the first block C0 is trained twice, on Qwen3.5-4B (C0q) and on pplx-flash (C0p), with the same data, adapter (D33) and steps; **pplx is chosen iff** on fast dev v2 the paired lower bounds of C0p − C0q are > −0.01 on the MR partition, > −0.01 on general and > −0.02 on knowledge, **and** the MR or general point difference is > 0; otherwise Qwen3.5-4B. If pplx is chosen, every Gate O3 (D35) comparison that names zero-shot Qwen3.5-4B — the MR suite, the external panel and the knowledge guard — uses zero-shot pplx-decider-flash-v1.2 instead, scored through its authors' code; MedDecider-4B stays. **Osler-9B is deferred** (operator, 2026-10-17); Gate O3's 9B verdict is NOT MEASURED in osler_v2 | 2026-10-17 | operator decision; pplx-decider-flash-v1.2 (Decision Index 0.3: 60.05) is the strongest open ≤ 10B decision model, and pplx's 27B led our external panel (0.641) |
| D40 | **Training speed recipe:** attention-only LoRA (D33) with gradient checkpointing only for batches above 8,192 padded tokens (`checkpoint_above_tokens: 8192`). Measured by the advisor on 300 real B1 batches: 6,838 real tokens/s (Qwen) and 6,406 (pplx) against 3,744 for osler_v1's all-layer LoRA with checkpointing on every batch; peak 58 GB; same numbers either way (tested). The all-layer adapter's MLP LoRA was the main cost (+55 % step time) | 2026-10-17 | osler_v1 trained at ~3,700 tokens/s on a 96 GB GPU; profiled: the kernels run at 7,600–11,500 tokens/s |

**Recorded risk (overruled objection):** the teacher was fixed without a comparison
against Gemma-4-31B / Qwen3.8-27B. If the teacher gate shows ECE > 0.05 after
temperature fitting, or a distilled student plateaus below the gold-only student, the
teacher is the first suspect.

---

## 4. Stages

Each stage = one loop. **Gate** = the hard-stop condition reviewed with the operator.
Estimates are planning guesses, not measurements; loop 0 measures real throughput.

### Loop 0 — `bench_v0`: benchmark, harness, baselines, teacher gate  *(done — see review)*

Build MedDecide-Bench v0 (tier 1 + fresh tier from ClinicalTrials.gov, openFDA,
PubMed), the eval harness (accuracy, Brier, ECE, latency, option-shuffle, candidate
count, abstention, label-mass), validate the harness against a published number, run
zero-shot baselines over the ladder and single-GPU decision models, run the DeepSeek
teacher pipeline gate, build the audit page.
- **Gate G0:** harness reproduces a published number (±2 pts); benchmark manifest
  complete; fresh templates screened (regex ≥95% dropped); teacher ECE ≤ 0.05 after
  temperature fitting on ≥2k gold dev items (or BLOCKED); operator audit done.
- **Compute:** ~30–50 h on 1×PRO 6000; a few hours on the 4×PRO 6000 teacher.
- Spec: `loops/bench_v0/ADVISORY.md`.

### Loop 0b — `bench_v0_fix0`: repair and re-measure  *(done — see `loops/bench_v0_fix0/FINDINGS.md`)*

Inserted after the bench_v0 review. Fix the MedMCQA key and verify every tier-1 gold
against its raw record; fix the `noul`/`score` readout and validate it against
lm-evaluation-harness; add the readout-health gate (D12); template screen v2
(gold-in-state, regex, macro-based drop rule); rebuild the fresh tier as v0.1 (D11,
class-balanced, strict slice); contamination probe; re-run ladder baselines; first
decision-model baselines; corrections record.
- **Gate G0 (re-applied):** every reported cell passes the health gate; `choice` and
  `noul` readouts agree with a reference within ±2 pts; v0.1 manifest, screen, and
  corrections complete; operator audit of the v0.1 sample.
- Spec: `loops/bench_v0_fix0/ADVISORY.md`.

### Loop 1 — `student_v0`: benchmark v0.2 + the first trained model  *(done — read the advisor note at the top of `loops/student_v0/FINDINGS.md`)*

- **Benchmark v0.2:** fix the graded-score option set; near-miss distractors for the MeSH
  and FDA-class templates (MeSH tree siblings; classes sharing a mechanism); a
  constant-answer check in the D12 gate; a **record–claim consistency** template family
  (does the record support this stated fact? — role-binding designs a string match cannot
  solve, gold by construction); a **long-record slice** (> 8,192 tokens); a shared-prefix
  (many questions per record) throughput measurement; HLE Biology/Medicine multiple-choice
  from `cais/hle` as a supplementary test.
- **Training data (D13):** tier-1 official train splits + **pre-window structured-gold**
  items (same builders, records dated before 2026-03-01), with **held-out templates (D14)**
  excluded entirely; a mechanical leakage check against every test and dev split.
- **Model:** MedDecide-0.8B = frozen Qwen3.5-0.8B + decision-path LoRA + pointer head,
  CE + Brier, option-order augmentation, per-`qtype` temperature on dev; ablation from
  Qwen3.5-0.8B-Base; byte-identical generation with the adapter off.
- **Gate G1 (D16):** item-paired bootstrap differences against zero-shot Qwen3.5-0.8B and
  JEV-9B on the headline fresh set, seen and held-out templates separately (full rule in
  D16). PASS on seen → scale and add teacher/ontology data; FAIL on held-out only → data
  diversity first; FAIL everywhere → diagnose the recipe at 0.8B.
- Also: a catalog of candidate HF datasets (documentation only), and a read-back of the
  operator's 150-item audit if it is available.
- Spec: `loops/student_v0/ADVISORY.md`.

### Loop 1b — `student_v1`: correct G1, fix padding, diversify data, compare three heads  *(done — see `loops/student_v1/FINDINGS.md`)*

student_v0, read per D16: the trained 0.8B beats zero-shot by a wide margin on **seen** decision
types (macro +0.163, Brier −0.230) but not on **held-out** types (macro −0.018); G1 vs JEV-9B was
never computed; a padding bug makes trained scores batch-dependent; the selected checkpoint is
step 1,000 of 44,152.
- Recompute G1 per D16 (JEV-9B on v0.2); fix and measure the padding bug; score held-out dev
  across 22 saved checkpoints (does transfer rise or fall with training?).
- **Data diversity (D17):** PubMed pre-window, ≥ 8 new train-only decision types, ≥ 2 new
  record–claim designs, `score` items, vetted catalog sources; no template > 8 % of the mix.
- **Three matched heads (D18):** A pointer head, B letter-readout LoRA, C Clef-style joint head
  via Unsloth — same data, order, step budget (15,000), base, prompt cap and evaluation.
- **Outcome logic:** an arm that passes G1 on held-out → the recipe to scale; all arms fail on
  held-out → data diversity / teacher next; arms indistinguishable → keep the simplest.
- Spec: `loops/student_v1/ADVISORY.md`.

### Loop 1c — `osler_v0`: competitor scoreboard, clinical generators, Osler-4B / -9B  *(done — see `loops/osler_v0/FINDINGS.md`)*

Inserted after the student_v1 review (D20–D24). Loops 2–4 below are re-planned after its
review; loop 3's 27B-class baseline runs move here because they fit on one PRO 6000.
- **Scoreboard first:** every open competitor through its authors' inference code on v0.2,
  the external clinical panel and the robustness pack (MedDecider 4B/9B/27B/31B,
  pplx-decider v1.1, JEV-27B, Clef, Clef-Flash; zero-shot Qwen3.5-4B/9B).
- **Data (D20):** gold-by-construction clinical generators (some held out), general
  human-labelled replay, the record–claim designs and catalog sources deferred by student_v1.
- **Models (D22, D23):** three matched 4B arms (option-code readout; pointer; non-causal
  option-code readout), a pre-registered head choice on dev, then Osler-9B and the
  Osler-0.8B reference with the chosen head.
- **Gate O1 (D24).**
- Spec: `loops/osler_v0/ADVISORY.md`.

### Loop 1d — `osler_v1`: medical-record audit data, MIMIC, teacher prose  *(done — see `loops/osler_v1/FINDINGS.md`)*

Inserted after the osler_v0 review (D26–D32). Data first: no architecture work; synthetic-first while MIMIC approval is pending (D32).
- **New task families (D29):** page category, ICD-10-CM code support, clinical-indicator evidence — generated by
  code (labels from rules), from real MIMIC notes with structured labels (D28), and as teacher prose verified by
  re-extraction (D27, D30).
- **Knowledge fix:** general replay raised from 3.5 % to the 20 % the osler_v0 ADVISORY set.
- **Training speed:** larger token batches without gradient checkpointing where memory allows; calibration saved with
  the checkpoint.
- **Data blocks** compared by short matched 4B runs and a pre-registered dev keep-rule; then one Osler-4B and one
  Osler-9B on the kept mix; **Gate O2 (D31)**.
- Spec: `loops/osler_v1/ADVISORY.md`.

### Loop 1e — `osler_v2`: real text, pplx-flash as a candidate base, replay 20 %, Osler-4B  *(current)*

Inserted after the osler_v1 review (D33–D37). osler_v1's Osler-4B passed the knowledge guard (+4.4 points) but lost real-text
ICD coding and the external panel to MedDecider-4B by small margins; its kept mix had no real clinical text.
- **Adapter:** attention-only LoRA (D33).
- **Blocks v2 (D34):** replay ≥ 20 % (D36) → real text (CodiEsp, MTSamples, EHR-Rel) → teacher prose → whole synthetic
  charts from a local Gemma 4 31B (D38) → MIMIC when the official files exist; each added to the last kept block; a dev set with every source's templates and a powered
  knowledge check.
- **Base:** Qwen3.5-4B or pplx-decider-flash-v1.2, chosen on dev at the first block (D39); training ~1.8× faster (D40).
- **MR suite v1 and Gate O3 (D35)** for Osler-4B; Osler-9B deferred (D39).
- Spec: `loops/osler_v2/ADVISORY.md`.

### Loop 2 — `teacher_data`: the data moat

- Bulk teacher labelling: DeepSeek-V4.1-Flash **non-thinking** option distributions
  (prefill-only, cheap); **thinking** only for low-confidence items (GEV-style
  cascade), reasoning folded into final probabilities.
- Unlabelled state pools: PubMed/PMC abstracts and case reports, ClinicalTrials.gov,
  drug labels, consumer health text (MedlinePlus), medical QA pools — all strictly
  dated **before** the fresh-tier window.
- **UMLS ontology-controlled synthetic decisions** (private): abbreviation
  disambiguation, concept relations, drug ↔ ingredient ↔ class, hierarchy,
  vocabulary-shift relevance (layperson ↔ clinical). Generated and labelled only on
  self-hosted models.
- Retrain 0.8B on gold + teacher (KL to full distributions) + UMLS synthetic.
- **Gate G2 (distillation gate):** adding teacher data improves the fresh tier over
  gold-only; each data source's marginal value measured (ablate one at a time).
  Fail → drop distillation, spend budget on gold + UMLS.

### Loop 3 — `ladder`: scale across the ladder

- Train LFM2.5-350M, 4B bake-off (MedGemma-1.5-4b-it vs Qwen3.5-4B on the same data,
  winner proceeds), Qwen3.5-9B.
- Run 27B-class baselines on the 4×PRO 6000 window: JEV-27B, pplx-decider-v1-27b,
  GEV-26B-Decide (and hosted Jev if the operator provides access).
- **Gate G3 (SoTA check):** per size tier, MedDecide vs every baseline on the fresh
  tier with CIs; the ≤4B vs 27B-class comparison is the headline.

### Loop 4 — `robustness`: make the claim survive review

- Ablations: verbalizer vs pointer readout; adaptive thinking on the 4B; data-source
  ablations; LoRA rank.
- Calibration: ECE/Brier before/after temperature per question type; reliability
  diagrams.
- Robustness: option-order shuffle, candidate-count scaling (2 → 16 → 64+),
  none-of-the-above rejection (detection vs false rejection), cache/batch drift on
  hybrid-attention bases, near-duplicate options.
- Generalisation: MIMIC-derived decision sets (needs DUA — see §6), out-of-domain
  general decision benchmarks (does medical tuning cost generality?).
- Joint schema head **only** if the pointer head fails at high cardinality.
- **Gate G4 (claim-ready):** every paper claim has a CLAIMS row, a CI, and a passing
  robustness check, or is dropped.

### Loop 5 — `clinical` (optional): MedDecide-clinical

Train on MIMIC-derived decisions (local models only, nothing committed). Memorisation
audit. Release through credentialed access. Separate from the public headline model.

### Loop 6 — `release`: benchmark + model + paper

Benchmark datasheet, regeneration scripts with a newer date window, model cards
(provenance, limits, intended use, "not for unsupervised clinical decisions"),
HF release, arXiv draft. Final name chosen here.

### Cross-track handoff — MedDecide as the retrieval judge

After G3, MedDecide's medical relevance skill is compared with gold relevance
judgments (BioASQ / TREC-COVID / NFCorpus train qrels) against an off-the-shelf
judge. If it agrees better, it becomes the relevance judge for the MedColBERT/MedEmbed
data moat (see `MedColBERT/docs/plans/v2-retrieval-plan.md`).

---

## 5. Benchmark design (MedDecide-Bench)

### Tier 1 — established (comparability)

| skill | sources (examples) | types |
|---|---|---|
| medical knowledge | MedQA (USMLE 4-option), MedMCQA (validation as test — test labels unreleased), PubMedQA (labelled), MMLU medical subsets | choice, noul |
| evidence / claim verification | SciFact, HealthVer, PubHealth | choice |
| relevance judging | TREC-COVID, NFCorpus (BEIR qrels), BioASQ if access terms allow | noul, score |
| routing / question type | MedQuAD question types | choice |

Every tier-1 number is shown beside a contamination probe score.

### Tier 2 — fresh (headline)

Items from records **first posted / first effective / first published after** the
latest training cutoff of any ladder model and the teacher. Gold = deterministic
lookups of structured fields. v0 sources: ClinicalTrials.gov API v2, openFDA
`drug/label`, PubMed update files (MeSH headings, publication types). Rebuildable for a
later date window with one argument — the "living benchmark" property.

Deferred to v2: PMC case reports (final diagnosis), OpenAlex citation-based relevance.

### Metrics

Accuracy (micro + macro), majority-class baseline beside every accuracy, Brier, ECE
(15 bins), label mass, p50/p95 latency, option-shuffle flip rate, candidate-count
degradation, abstention (detection rate + false-rejection rate). Bootstrap 95% CIs for
every headline comparison.

---

## 6. Operator prerequisites by stage

| stage | prerequisite |
|---|---|
| loop 0 | `HF_TOKEN` on the pod with access to `google/medgemma-1.5-4b-it` accepted; GitHub push access from the pod to this repo; agent harness with DeepSeek API key; teacher endpoint for T10 |
| loop 0 | ~2 h for the 150-item audit |
| loop 1 | `cais/hle` terms accepted on the HF account behind `HF_TOKEN` (for the HLE supplementary test); the 150-item audit file at `outputs/bench_v0_fix0/F10/audit_v0.jsonl` |
| loop 2 | 4×PRO 6000 teacher window (days); UMLS files available on the pod (`data/private/umls/`) |
| loop 3 | 4×PRO 6000 window for 27B baselines |
| loop 1d | MIMIC-III, MIMIC-IV, MIMIC-IV-Note, MIMIC-IV-ED under `data/physionet/` (CITI training approved, DUAs signed; operator download); MDACE annotations; teacher endpoint variables in `/workspace/.secrets.env` (D30) |
| loop 6 | final name; licence check of every benchmark source for redistribution |

---

## 7. Open decisions (operator, later)

- Final model name.
- Venue.
- Whether the joint schema head or RLCD enters scope (only on evidence from loop 4).
- Whether MedDecide-clinical is built.
