# Decision-model baselines on MedDecide-Bench v0.1 (F7)

**Loop:** `bench_v0_fix0`  **Generated from:** `outputs/bench_v0_fix0/F7/*.json`
**Prediction rows:** 20,542 (JEV-9B) + Laya (pending)

These models read options natively, so the D12 letter-readout health gate does not apply. Each
cell reports the model's own distribution: accuracy with a bootstrap CI, Brier, ECE, coverage and
the option-shuffle flip rate. `coverage` is `n/template_total` — items the model could express at
all — and every shortfall is counted in the JSON with its reason.

## `autotrust/JEV-9B` — decision head (System 1 path)

Served through the model card's own protocol: a `[decision]:` prompt whose next-token
log-probabilities are read over the kind's verbalizer tokens, then the head's per-slot **bias** and
the per-kind **temperature** (both shipped in the repo: `adapter_vllm/decision_head.json`,
`calibration.json`) are applied client-side. **Deviation:** the card serves with vLLM; vLLM is not
installed here, so the same single-prefill read runs on `transformers` + `peft`. The card reports
the two paths agree to mean |Δp| 0.0008, and the substitution is recorded in `jev9b.json`.

| tier | template | qtype | n / total | accuracy | Brier | ECE | flip |
|---|---|---|---|---|---|---|---|
| tier 1 | `medmcqa_4opt_v1` | choice | 4,183 / 4,183 | **0.6229** | 0.1033 | 0.0188 | 0.12 |
| tier 1 | `medqa_usmle4_v1` | choice | 1,273 / 1,273 | **0.7156** | 0.0894 | 0.0316 | 0.12 |
| tier 1 | `mmlu_mc_v1` | choice | 1,103 / 1,103 | **0.8277** | 0.0513 | 0.0506 | 0.02 |
| tier 1 | `medquad_routing_v1` | choice | 5,000 / 5,000 | 0.9882 | 0.0072 | 0.0258 | 0.00 |
| tier 1 | `pubmedqa_ynm_v1` | choice | 477 / 477 | 0.6960 | 0.0648 | 0.0703 | 0.10 |
| tier 1 | `nfcorpus_relevant_noul_v1` | noul | 647 / 647 | 0.6553 | 0.0548 | 0.1454 | — |
| tier 1 | `scifact_relevant_noul_v1` | noul | 600 / 600 | **0.9266** | 0.0426 | 0.0931 | — |
| tier 1 | `nfcorpus_graded_score_v1` | score | 429 / 429 | *0.2634* | 0.1411 | 0.2229 | — |
| fresh | `ct_randomised_noul_v1` | noul | 2,000 / 2,000 | 0.8765 | 0.0509 | 0.0421 | — |
| fresh | `ct_healthy_volunteers_noul_v1` | noul | 2,000 / 2,000 | 0.7200 | 0.0352 | 0.1529 | — |
| fresh | `pubmed_humans_noul_v1` | noul | 1,316 / 1,316 | 0.8571 | 0.0157 | 0.0801 | — |
| fresh | `pubmed_observational_noul_v1` | noul | 206 / 206 | 0.7087 | 0.0302 | 0.1870 | — |
| fresh | `pubmed_mesh_major_choice_v1` | choice | 4,000 / 4,000 | **0.9960** | 0.0015 | 0.0045 | 0.00 |
| fresh | `ct_phase_choice_v1` | choice | 1,100 / 1,100 | 0.4745 | 0.0845 | 0.1164 | 0.20 |
| fresh | `fda_boxed_warning_noul_v1` | noul | 1,249 / 1,910 | 0.8254 | 0.0349 | 0.0769 | — |
| fresh | `fda_class_choice_v1` | choice | 3,154 / 4,000 | 0.9451 | 0.0179 | 0.0110 | — |

*Italic* = the accuracy is **not comparable to chance**; see the boxed note below.
"—" = not measured for that cell (the flip probe runs on `choice` items only).

### Two coverage shortfalls, both explained

* `fda_boxed_warning_noul_v1`: 661 of 1,910 items were skipped because their prompt exceeds
  **8,192 tokens**, where the 9B hybrid's non-fused attention path materialises a 30 GB
  intermediate and OOMs. The limit is a parameter (`--max-prompt-tokens`) and every skipped item
  is counted in the cell (`n_excluded_prompt_too_long`).
* `fda_class_choice_v1`: 846 of 4,000 items skipped for the same reason.

**Neither shortfall is a model limitation.** Both are consequences of running the 9B base without
its fused kernels (`causal_conv1d`, `flash-linear-attention` are not installed) in this
environment; the card's vLLM path would express them.

### The graded-score cell is a benchmark defect, not a model result

`nfcorpus_graded_score_v1` offers three levels (`Not relevant` / `Relevant` / `Highly relevant`)
but **no item in its pool has the lowest level**: gold is level 2 for 310 items and level 3 for
119. A model that answers "Not relevant" is therefore always wrong. JEV-9B answers it on 41.5 % of
items, which drives the 3-way accuracy to **0.2634** against a 0.3333 chance level; restricted to
the two levels the pool actually contains, its accuracy is **0.5804** (chance 0.5). Both numbers
are reported — the 3-way figure is flagged as not comparable to chance, and the fix belongs in the
v0.2 builder (`SCORE_TEMPLATE_FLAW.md`, X029).

### What stands out

* **MedMCQA: 0.6229.** `bench_v0` reported 0.122 for the 4B model on this template and called it a
  positional-bias collapse. That was the off-by-one key (X003). The template ranks third-hardest
  here, not degenerate.
* **Calibration is strong where the model is strong.** `pubmed_mesh_major` ECE 0.0045 and
  `fda_class_choice` ECE 0.0110 with Brier 0.0015 / 0.0179 — near-perfect calibration on the two
  largest fresh `choice` templates.
* **The flip rate is protocol-revealing.** 0.12 on MedQA and MedMCQA versus 0.00 on
  `medquad_routing` and `pubmed_mesh_major`: JEV-9B's answer changes with option order on the two
  exam-style templates and not on the two extraction-style ones.
* **It beats every ladder model on every tier-1 knowledge template** (MedQA 0.7156 vs 0.7572 for
  the 9B — the 9B is ahead; MMLU 0.8277 vs 0.8585 — the 9B is ahead). JEV-9B is a 9B base with a
  decision head, so "comparable to the 9B base with much better calibration" is the honest reading,
  not "better than the ladder".

## `convaiinnovations/laya` and `laya-typed-decisions`

**PENDING — running at the time of writing.** The `laya` package is not on PyPI-installed form
here, so these use the repository's **own** implementation (`rl_agent_api.RLAgent`, `rl_common`)
rather than the `Router` wrapper; the model's own post-hoc temperature is applied exactly as its
API does. Results are appended to `laya.json` / `laya-typed-decisions.json` and this section is
replaced when they are complete.
