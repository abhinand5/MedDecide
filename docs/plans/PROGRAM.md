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
| D23 | **Default readout: option-code readout with a separate head** (pplx-style): a `[K, d]` readout matrix (K ≤ 255 option codes), rows initialised from the base `lm_head` rows of the code tokens, trained with the decision-path LoRA, softmax over the valid codes of the current question, temperature per question type applied once. It exports to a plain `lm_head` (row i → token id i) so standard serving stacks reproduce it. The base stays frozen; the LoRA is applied on the decision path only. Pointer head, non-causal full attention and head-only (no LoRA) are matched ablations at 4B | 2026-10-09 | operator + advisor review: student_v1's heads were within 0.6 pts at 0.8B; the letter readout starts from a readout the base already has and ships through standard stacks; the separate head removes the off-option mass that failed D12 |
| D24 | **Gate O1** (osler_v0): for Osler-4B vs {MedDecider-4B, zero-shot Qwen3.5-4B} and Osler-9B vs {MedDecider-9B, JEV-9B}: item-paired bootstrap 95 % CI (1,000 resamples, items within datasets/templates) of Osler − B in macro accuracy and in mean Brier. PASS iff accuracy CI lower bound > 0 **and** Brier CI upper bound < 0 against **both** baselines on **(i)** the headline fresh set, seen templates (as D16) **and (ii)** the external clinical panel; plus a **knowledge guard**: on MedQA + MedMCQA, the accuracy CI lower bound of Osler − zero-shot same-size base > −0.02. Held-out templates, held-out generators and the robustness pack are reported with the same rule, separately. Other models (27B/31B decision models, sibling arms) are reported as additional, never in the verdict | 2026-10-09 | fixed before results; extends D16 to the sizes and competitors the program now targets |

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

### Loop 1c — `osler_v0`: competitor scoreboard, clinical generators, Osler-4B / -9B  *(current)*

Inserted after the student_v1 review (D20–D24). Loops 2–4 below are re-planned after its
review; loop 3's 27B-class baseline runs move here because they fit on one PRO 6000.
- **Scoreboard first:** every open competitor through its authors' inference code on v0.2,
  the external clinical panel and the robustness pack (MedDecider 4B/9B/27B/31B,
  pplx-decider v1.1, JEV-27B, Clef, Clef-Flash; zero-shot Qwen3.5-4B/9B).
- **Data (D20):** gold-by-construction clinical generators (some held out), general
  human-labelled replay, the record–claim designs and catalog sources deferred by student_v1.
- **Models (D22, D23):** four matched 4B arms (option-code readout; pointer; non-causal
  option-code readout; head-only), a pre-registered head choice on dev, then Osler-9B and the
  Osler-0.8B reference with the chosen head.
- **Gate O1 (D24).**
- Spec: `loops/osler_v0/ADVISORY.md`.

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
| loop 4 | MIMIC-IV access: CITI "Data or Specimens Only Research" training uploaded at physionet.org/settings/training/, DUA signed per dataset |
| loop 6 | final name; licence check of every benchmark source for redistribution |

---

## 7. Open decisions (operator, later)

- Final model name.
- Venue.
- Whether the joint schema head or RLCD enters scope (only on evidence from loop 4).
- Whether MedDecide-clinical is built.
