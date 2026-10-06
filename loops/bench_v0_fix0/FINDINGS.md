# FINDINGS — MedDecide loop `bench_v0_fix0` (repair loop for `bench_v0`)

## Summary

**The question.** `bench_v0` built MedDecide-Bench v0 and an eval harness, and an advisor review
then found defects in it. This loop asks: were those defects real, what do they invalidate, and
what do the numbers look like once they are fixed?

**What we did.** Repaired the three defect classes the review named — the MedMCQA answer key, the
`noul`/`score` readout, and the fresh window — and then, because each repair exposed more, verified
every tier-1 gold label against its raw source record, rebuilt the benchmark as **v0.1**, screened
it with a balanced rule, re-ran all six ladder models and three decision models through a
readout-health gate, ran a contamination probe, and wrote a corrections record covering **all 53
`bench_v0` claims**.

**What we found.**

1. **Both headline `bench_v0` findings were wrong, and neither was a model result.**
   `C048` claimed MedMCQA's official test split "has no D answers" and that the 4B model has a
   last-position bias; `C051` claimed an option shuffle "proved" it. The loader read MedMCQA's
   `cop` field as 1-based when it is **0-based**, which shifted every gold label one option back and
   silently dropped all **1,348** rows whose answer is "A". The official split has **4,183** items
   with gold A 1,348 / B 1,085 / C 925 / **D 825**, identical to the raw `cop` counts, and the "D
   preference" was an artefact of the shifted key. Both claims are **withdrawn** (X002, X003, X026).
2. **The `noul`/`score` readout was reading tokens the model had no reason to produce.** Those items
   rendered as `yes. Yes` while the instruction said "Respond with a single letter", and the readout
   scored the `yes`/`no` key tokens: label mass ~**0.002**. Every `noul`/`score` number in
   `bench_v0` is invalid. After the fix the same items read at label mass **0.996–0.999** with
   greedy agreement **0.96–1.00** — and across all 8,679 `noul` items in the final 0.8B run the
   median is **0.9926** — while the previously validated `choice` path is provably unchanged
   (X007, X008).
3. **The fresh window was bounded by the wrong model.** `bench_v0` started it at the *teacher's*
   repository date, which is not a contamination boundary — the teacher never labels benchmark
   items. Decision **D11** bounds it by the ladder instead: v0.1 starts **2026-03-01**, with records
   dated ≥ 2026-09-10 kept as a separately reported **strict slice** of 2,896 items (X012, X013).
4. **The template screen was ranking templates by class imbalance and calling it shortcut
   learnability.** `bench_v0` dropped `pubmed_observational_noul_v1` for a 0.985 BoW score that the
   *majority class alone* produced, while keeping a template at 0.926. With v0.1's balanced 103/103
   split the same template's true BoW macro accuracy is **0.796** and it is **kept** — the cleanest
   single piece of evidence that the old drop was an artefact (X017, X018).
5. **Four further defects surfaced from the repairs, three of them real.**
   (a) nfcorpus score items carried a grade borrowed from another query's pool (X005);
   (b) CT.gov's `healthyVolunteers` is a JSON boolean, so `str(value) == "yes"` marked **every**
   item "no" (X015);
   (c) `nfcorpus_graded_score_v1` offers a lowest level that **no item in its pool has**, so a model
   answering it is guaranteed wrong (X029);
   (d) the fourth was mine: two of my own validation designs were broken before the real bug
   surfaced, and both are recorded rather than deleted (see "What did not work").
6. **The readout-health gate withheld 5 of 96 ladder cells instead of printing them.**
   Five ladder cells and (by the same rule) the graded-score cell for both decision models fail:
   MedGemma-1.5-4b on MedQA and Qwen3.5-0.8B-Base on MedMCQA / `fda_boxed_warning` miss the greedy
   agreement threshold (0.86–0.88 < 0.90), and the 4B and 9B on `nfcorpus_graded_score_v1` sit
   with an accuracy CI entirely below chance. Each is reported as `READOUT_FAIL — <check>` with no
   accuracy (X031, X035). This is the gate doing exactly what D12 created it for.
7. **The ladder scales on biomedical knowledge once the key is right, and only there.**
   MedQA (majority 0.277): **0.2914 → 0.3715 → 0.4139 → [MedGemma: `READOUT_FAIL`] → 0.7030 →
   0.7572** for 350M, 0.8B-Base, 0.8B, MedGemma-4b, 4B, 9B. MMLU (0.325): 0.2411 → 0.4569 →
   0.4605 → 0.4768 → 0.8168 → **0.8585**. MedMCQA (0.322): 0.3246 → [`READOUT_FAIL`] → 0.3791 →
   0.4209 → 0.5892 → **0.6562** (X039). On the fresh tier the same models are at ~0.50 on every
   `noul` template and near chance on `choice` — the benchmarks that matter are not saturated by
   scale alone.
8. **Contamination is measured, and the probe cannot settle it.** Min-K% Prob across 6 models × 8
   tier-1 sources gives mean gaps from **−1.52** (350M) to **+0.82** (0.8B-Base). The gap **does not
   grow with model size** (4B +0.796 vs 9B +0.764), and it cannot separate memorisation from "the
   original order is more predictable prose". No tier-1 accuracy is adjusted by it (X027, X028).
9. **Decision models separate sharply.** `autotrust/JEV-9B` (9B + decision head) scores MedQA
   **0.7156**, MMLU **0.8277**, MedMCQA **0.6229** with ECE 0.005–0.05 — just under the plain
   Qwen3.5-9B on knowledge, far better calibrated. Both `convaiinnovations/laya` checkpoints
   (~400M) are **at or below the majority baseline** on MedQA (0.2655 / 0.2608 vs 0.277) and MMLU
   (0.3436 / 0.3490 vs 0.325) — below even the 350M ladder model (X036, X037).

**What it means.** The repairs hold and are checkable: tier-1 gold verifies **0 mismatches of
15,915 checked items** across all eight sources (X004); v0.1 builds deterministically and passes
7/7 acceptance checks (X012, X016); 96 ladder cells and 48 decision-model cells are measured with
every coverage shortfall counted; the readout-health gate withholds rather than reports. And the
corrections record leaves no `bench_v0` claim standing that depended on the broken parts (X025).
`bench_v0`'s `choice`-path results — MedQA, MMLU — stand unchanged (X008).

**What did not work, or is not measured.**

* **The teacher gate is `BLOCKED`** — `TEACHER_BASE_URL` and `TEACHER_API_KEY` are unset on the pod,
  for the second loop running. No teacher number appears or is estimated (X024).
* **The human label audit is `BLOCKED — awaiting operator`**: 150 sampled items are staged, but
  whether their gold labels are correct is a human judgement (X022, X023).
* **The contamination probe cannot decide the question it was built for** (see item 8).
* **Some cells carry recorded coverage shortfalls, none silent:** JEV-9B skips 661/1,910 and
  846/4,000 items on the two openFDA templates (prompts over 8,192 tokens, where the 9B hybrid's
  unfused attention path allocates 30 GB), and `laya` skips 21/4,183 MedMCQA items whose options
  collapse onto one key after its own 48-token truncation.
* **Two of my own validation attempts were broken** before they caught anything: a bare prompt
  collapsed label mass to 1.2e-04, and a comparison used canonical letters where the reference
  produced `yes`/`no`. Both are in `readout_validation.md`.

## Body

### 1. The MedMCQA key, and the two findings that rested on it

`_medmcqa_item` computed `gold_index = int(cop) - 1`. `cop` is 0-based. Three independent proofs
(X002): the raw validation split has `cop = 0` **1,348** times; the record's own `exp` explanation
names `option[cop]` in **1,462 of 2,194** usable rows (66.6 %) versus **253** (11.5 %) for
`option[cop+1]`, and writes "Ans-a" for `cop = 0`; and after the fix the built gold distribution is
exactly the raw `cop` counts. The off-by-one also *dropped* every "A" row, which is why
`bench_v0` saw 2,835 test items and "no D answers". `C048` and `C051` are withdrawn; `C013`'s
MedMCQA count is corrected 4,251 → 6,183 (test 2,835 → 4,183).

The verifier that found this (`scripts/bench/verify_gold.py`) checks every tier-1 item's gold
against an independently written mapping for its source: **0 mismatches of 15,915 checked items**
across medqa 1,273; medmcqa 4,183; pubmedqa 1,000; mmlu 1,212; medquad 7,000; trec_covid 195;
nfcorpus 752; scifact 300 (X004). The remaining 2,679 pairs are unjudged relevance pairs and are
counted as *unverifiable*, never as passes.

### 2. The readout

`canonicalise_options` re-keys every item to `A`, `B`, `C`, … — a no-op for `choice`, a conversion
for `noul` (`A. Yes` / `B. No`) and `score` (lowest first) — and gold follows its content. Measured
on 200 `noul` + 200 `score` items per model (X007):

| model | `noul` mass | `score` mass | `noul` greedy | `score` greedy |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0.9960 (was ~0.002) | 0.9977 | 1.00 | 1.00 |
| Qwen3.5-4B | 0.9994 | 0.9963 | 1.00 | 0.96 |

The `choice` path is unchanged, and that is proven rather than asserted: canonicalisation is a
no-op for all 1,273 MedQA items, prompts reconstructed with the `bench_v0` renderer are
byte-identical for 200/200 of them, and the full run reproduces 0.40770 vs 0.40770 (Δ −0.00016 pts)
(X008). Against lm-evaluation-harness on a yes/no task under an **identical** protocol our
implementation scores **0.5828 vs 0.5828 (Δ 0.00 pts)**; the letter readout scores 0.6184 on the
same items, a **protocol** difference reported as such (X009, X010).

### 3. The benchmark: v0.1

| quantity | v0 | v0.1 |
|---|---|---|
| tier-1 items | 21,202 (medmcqa 2,835 test) | **22,594** (medmcqa 4,183 test + 2,000 dev; nfcorpus 3,076) |
| fresh items | 41,502 on a 25-day window | **23,582** on 2026-03-01 → build date (X012) |
| fresh window decided by | the teacher's repo date | the ladder (D11); strict slice ≥ 2026-09-10 = **2,896** items (X013) |
| class balance | majority up to 1.000 | **0** single-class groups; K recorded per template (X014) |
| build | — | acceptance **PASS 7/7**, rebuild **byte-identical** (X016) |

Fewer fresh items is not an improvement in size — it is what balancing costs, and the tier is now
interpretable: no template has a single gold class in test, and the maximum majority share is 0.722.

The screen (X017–X021) keeps 16 of 22 templates; every drop is `n_test<200`, none is a shortcut
drop. `noul`/`score` regex baselines are reported `NOT MEASURED` rather than a misleading 0.000.
F3's gold-in-state check flags nothing: for `pubmed_mesh_major_choice_v1` the gold text is in the
state for 0.462 of items against a 0.500 majority, and its BoW macro accuracy is 0.246 — the high
scores that template produces are earned by reading the abstract, not by a leak (X032).

### 4. The ladder baselines (F6)

96 cells (6 models × 8 tier-1 + 8 fresh kept templates), **91 PASS, 5 READOUT_FAIL**, 181,464
prediction rows, coverage equal to the template's item count in every scored cell (X039).

| model | MedQA (maj 0.277) | MMLU (0.325) | MedMCQA (0.322) |
|---|---|---|---|
| LFM2.5-350M | 0.2914 | 0.2411 | 0.3246 |
| Qwen3.5-0.8B-Base | 0.3715 | 0.4569 | *READOUT_FAIL* |
| Qwen3.5-0.8B | 0.4139 | 0.4605 | 0.3791 |
| MedGemma-1.5-4b-it | *READOUT_FAIL* | 0.4768 | 0.4209 |
| Qwen3.5-4B | 0.7030 | 0.8168 | 0.5892 |
| Qwen3.5-9B | **0.7572** | **0.8585** | **0.6562** |

MedMCQA's 4B number was **0.122** in `bench_v0` and is **0.5892** here — the difference is the key,
not the model (X034). On the fresh tier every model is near 0.50 on the `noul` templates
(350M: 0.4995 `ct_healthy_volunteers`, 0.5030 `ct_randomised`, 0.4969 `pubmed_humans`, 0.5000
`pubmed_observational`), which is the honest picture of zero-shot capability on genuinely unseen
records.

### 5. Decision-model baselines (F7)

JEV-9B through its own decision-head protocol (vLLM not installed; `transformers`+`peft`, with the
card reporting |Δp| 0.0008 between the paths): MedQA **0.7156**, MMLU **0.8277**, MedMCQA
**0.6229**, `pubmed_mesh_major` **0.9960** (ECE 0.0045), `fda_class_choice` **0.9451** (ECE 0.0110).
Its option-shuffle flip rate is 0.12 on the exam templates and 0.00 on the two extraction templates.
Both Laya checkpoints are at or below the majority baseline on biomedical knowledge (X036, X037) and
flip more (0.18–0.33; X038) — expected for models trained on business workflows, and a useful
construct-validity datum: a calibrated decision architecture does not by itself produce medical
knowledge. Full tables: `decision_models_v0_1.md`.

### 6. Contamination (F5)

48 cells, Min-K% Prob with a deterministic reorder control. See Summary item 8; the full table is in
`contamination.md`. The largest single cell is MedQuAD/9B at **+3.52**, the one place where the
memorisation reading is materially more plausible than the prose reading — flagged as a flag, not a
finding, because the probe cannot prove it.

### 7. The five-claim spot-check (re-run in fresh processes at closure)

| # | claim | re-derived value | artifact |
|---|---|---|---|
| 1 | MedMCQA test gold distribution (X003) | **4,183 items; A 1,348 / B 1,085 / C 925 / D 825** — exactly the raw `cop` counts | `data/bench/v0.1/tier1/medmcqa.jsonl` |
| 2 | `noul` label mass after the readout fix (X007) | median **0.9926** over **8,679** `noul` prediction rows (0.8B) | `outputs/bench_v0_fix0/F6/preds_qwen3p5-0p8b.jsonl` |
| 3 | v0.1 fresh totals (X012) | **23,582** = 8,123 + 7,297 + 8,162, counted from the files | `data/bench/v0.1/fresh/*.jsonl` |
| 4 | JEV-9B MedQA accuracy (X030) | **0.7156** (3,644 / 5,092 rows — the row count exceeds the item count because the file accumulates across runs; the unique-item figure is 1,273 at 0.7156) | `outputs/bench_v0_fix0/F7/preds_jev9b.jsonl` |
| 5 | 350M MedQA accuracy (X039) | **0.2914** (371 / 1,273), gate PASS | `outputs/bench_v0_fix0/F6/model_lfm2p5-350m.json` |

All five reproduce. Spot-check 4 also surfaced the one artifact hygiene issue worth recording: the
prediction files are **append-only across re-runs**, so a row count can exceed the item count and
any consumer must deduplicate by `item_id`. The F6/F7 reports read the per-cell summaries, which are
deduplicated by construction; this is noted so a future loop does not mistake a row count for an
item count.

### 8. The corrections, in one place

`CORRECTIONS.md` dispositions **all 53** `bench_v0` claims: **4 withdrawn** (C046, C048, C049,
C051 — the MedMCQA and `noul`/`score` results), **8 corrected** (C011, C013, C019, C031, C033,
C035, C036, C037), **9 superseded** (the fresh-tier claims, whose v0 artifacts are correct for v0
but describe a window v0.1 replaces), and **32 still valid** (schema, loader framework,
`choice`-path results). The accounting is checked by script, not by eye: 53 of 53 named, 0
unaccounted (X025, X026).

## Pointers

* Corrections: `CORRECTIONS.md`  ·  Claims: `CLAIMS.md` (X001–X039)
* Readout evidence: `readout_validation.md`  ·  Screen: `template_screen_v0_1.md`
* Ladder table: `baselines_v0_1.md`  ·  Decision models: `decision_models_v0_1.md`
* Contamination: `contamination.md`  ·  Next loop: `NEXT.md`
* Raw artifacts (gitignored): `outputs/bench_v0_fix0/{F0..F10}/`
* The benchmark: `data/bench/v0.1/{tier1,fresh}/` (gitignored) with committed manifests alongside
