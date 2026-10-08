# FINDINGS — student_v0: benchmark v0.2, a structured-gold training set, and the first trained MedDecide (0.8B)

> **Advisor note (2026-10-08, added after closure).** Three corrections; the rest of this file is
> left as the loop wrote it.
>
> 1. **G1 was computed against the wrong baseline set.** D16 fixes the G1 baselines as
>    **zero-shot `Qwen/Qwen3.5-0.8B` and JEV-9B**. `g1.md` used the **Base ablation** (S10) in
>    place of JEV-9B. The Base ablation is itself a trained MedDecide (same recipe and data), so
>    "seen templates FAIL: base accuracy CI lower bound −0.0176" compares two MedDecide variants;
>    it is not a G1 result. Read per D16: **seen templates vs zero-shot PASS** (macro +0.1633
>    [+0.1530, +0.1741], Brier −0.2298 [−0.2345, −0.2253]); **seen vs JEV-9B NOT MEASURED**;
>    **held-out vs zero-shot FAIL on accuracy** (macro −0.0176 [−0.0301, −0.0048]). So G1 is
>    **incomplete on seen templates and FAIL on held-out templates**, which is ADVISORY §2's
>    **second** branch ("learned templates, not decisions; data diversity first"), not the third.
>    `student_v1` V1 re-runs G1 with the D16 baselines.
> 2. **The evaluated MedDecide is the step-1,000 checkpoint** (≈2 % of the pass, ~8,000 items
>    seen); the converged checkpoint scores lower on test (tier 1 0.6141 vs 0.6505; fresh 0.7767
>    vs 0.8227). Most of the gain is acquired in the first ~1,000 steps.
> 3. **Closure gaps:** S9 and S11 are left `IN_PROGRESS` in STATE.md, and §7's five-claim
>    spot-check was never appended. Both are carried into `student_v1`'s closure requirements.


**Loop:** `student_v0` (loop 1 — the first trained MedDecide) · **Branch:** `loop/student_v0`
**Run window:** 2026-10-06T18:06:22Z → 2026-10-08 (S14 hard stop) · **Status:** `STOPPED`
**Committed here:** this file, `g1.md`, `CLAIMS.md`, `STATE.md`, `NEXT.md`, the loop's markdown
reports and code. **Not committed:** items, training data, predictions, checkpoints — all raw
outputs stay in the gitignored `outputs/`.

Every number below names an artifact and a CLAIMS id (`S###`) whose `recompute` command
reproduces it (AGENTS.md R1). `g1.md` is the aggregate verdict; this file is the readable
record. Where a value is unavailable it says `NOT MEASURED — <reason>` (R3) rather than
estimating.

---

## Summary

**The question.** Does the MedDecide recipe — a frozen base, a decision-path LoRA and a
pointer head, trained with cross-entropy + Brier on gold labels — add real skill over the
zero-shot base model and over an existing 9B decision model, including on decision types it
never saw in training? (ADVISORY §2.)

**What we did.** Built **v0.2 of the benchmark** — 45,009 items = 35,263 carried identical
from v0.1 + 9,746 new `_v2` items (S014) — and a **gold-only training mix of 212,481 items**
(S053) whose four-part leakage check is 0 on all four kinds (S055). Trained two 0.8B models,
same recipe and data, one full pass each: **MedDecide-0.8B on the instruct base** (44,152
steps / 13,867.5 s; S084) and a **Base ablation** (44,152 steps / 13,760.4 s; S073). Added the
missing baseline — a **full-v0.2 zero-shot `Qwen/Qwen3.5-0.8B`** run (S080) — and applied the
pre-registered **D16** paired-CI rule (S092) on **identical item sets** (37,289 shared ids,
0 differences; S097).

**What we found.**

1. **G1 = FAIL, on all four rules** (S092). The first trained MedDecide **does not beat both
   baselines on the fresh sets** by the pre-registered paired-CI rule. The verbatim failures:
   seen templates *"base: accuracy CI lower bound -0.0176 is not > 0"*; held-out templates
   *"base: no items scored by both models in this set; zeroshot: accuracy CI lower bound
   -0.0301 is not > 0"*; strict seen *"base: accuracy CI lower bound -0.0589 is not > 0"*;
   strict held-out *"base: no items scored by both models in this set; zeroshot: accuracy CI
   lower bound -0.0181 is not > 0"*. This is a **failure, not a near-miss**: against the Base
   ablation the accuracy difference on the fresh seen set is *negative*, not merely
   insignificant (−0.0115, 95 % CI [−0.0176, −0.0060]; S093).
2. **The model is far above the letter-readout zero-shot baseline on same-template fresh
   cells** — macro accuracy 0.8894 vs 0.7262 on the 17,442 fresh seen items (difference
   +0.1633 [0.1530, 0.1741]; S093) — but **below it on the held-out aggregate** (0.5544 vs
   0.5720 on the 6,326 fresh held-out items, difference −0.0176 [−0.0301, −0.0048]; S093/
   S094). Two of the four held-out cells are individually *above* zero-shot
   (`fda_boxed_warning` +0.0068, `pubmed_humans` +0.0912), two are clearly below
   (`ct_arm_role` −0.0785, `ct_phase` −0.0900; S094). The pattern is a **readout effect plus
   same-template training**, not leakage: the four D14 templates have **0** training items
   (S100) and the leakage check removed 2,188 duplicate rows with 0 collisions on re-check
   (S044, S055).
3. **The instruct base materially improves pointer-head readout health**: on identical item
   sets, the instruct model passes **18 of 19** D12 cells (7/8 tier-1 + 11/11 fresh) and the
   Base ablation **11 of 19** (5/8 + 6/11) (S096; S082, S075).
4. **Two of the three S9 training attempts are not results.** Run 1 failed by **pipeline**
   (an accidental length curriculum; S087) and run 2 **diverged** (S088); both were stopped by
   operator ruling. Only run 3 is a result, and it completed the full planned pass (S084).
5. **Two recipe-level defects remain measured but unfixed**: the trained head's logits depend
   on batch composition (left padding) in fp32, at 17.9× the no-padding control (S091), and
   spiking batches are **length-driven** (S090). Both are plausible contributors to run 2's
   divergence and both are candidates for the next loop's first fix.
6. **The macro-first selection rule behaved differently by model** — on the instruct run it
   picked the checkpoint with better readout health (18/19 vs the converged 12/19), on the
   Base ablation it picked the worse one (11/19 vs the converged 16/19) (S096). It is
   **validated on one model, not both**.
7. **The dev sample inverted the model ranking**: the Base ablation's selected checkpoint had
   *higher* full-dev macro accuracy than the instruct model's (0.7180 vs 0.6653; S079) yet
   scored **~7 points worse on tier-1 test** (0.5782 vs 0.6505; S075, S082).
8. **`score` is not a working qtype**: the training mix contains **0** score items (S053), the
   dev split has 33 (< the 50-item fit floor) so its temperature is `NOT FITTED` (S085,
   S046), and its one tier-1 cell reads `READOUT_FAIL — constant_answer` (S092).

**What it means.** ADVISORY §2's **third branch applies: G1 FAIL everywhere → diagnose the
recipe (readout, capacity, loss, data) at 0.8B before spending anything larger.** The failure
is *not* this loop's earlier pipeline artefact — run 3 completed its pass with a stable dev
curve, the leakage controls held, and the D12 gate excluded failing cells rather than
flattering them — but the loop also does not have a clean recipe claim: the padding/batch-
composition dependence, the length-driven gradient tail, the selection-rule inconsistency and
the `score` hole are all measured and all unresolved.

**What did not work.** (a) The first batching implementation (run 1). (b) The un-fixed recipe
(run 2, diverged). (c) Choosing the checkpoint by dev macro accuracy as a *general* rule —
it helped the instruct run and hurt the Base ablation. (d) The `score` question type, from
data construction to calibration. (e) The JEV-9B comparison on v0.2 was never commissioned
(`NOT MEASURED`, S095). (f) The operator's human audit was not completed, so the label-quality
read-back is `NOT MEASURED` (S099).

---

## 1. The verdict: G1 = FAIL on all four rules

`loops/student_v0/g1.md` + `outputs/student_v0/S11/g1.json`, generated by `scripts/bench/g1.py`
at 2026-10-08T02:57:24Z, git `2596affb`, seed 0, 1,000 resamples, alpha 0.05 (S092). The D16
rule: **PASS iff, against both baselines, the accuracy-difference CI lower bound is > 0 and the
Brier-difference CI upper bound is < 0.**

| rule | result | detail (verbatim from `g1.md`) |
|---|---|---|
| **G1 on seen templates** (decides the §2 branch) | **FAIL** | base: accuracy CI lower bound -0.0176 is not > 0 |
| G1 on held-out templates (D14, reported separately) | **FAIL** | base: no items scored by both models in this set; zeroshot: accuracy CI lower bound -0.0301 is not > 0 |
| strict slice, seen templates (additional, same rule) | **FAIL** | base: accuracy CI lower bound -0.0589 is not > 0 |
| strict slice, held-out templates (additional, same rule) | **FAIL** | base: no items scored by both models in this set; zeroshot: accuracy CI lower bound -0.0181 is not > 0 |

**The paired numbers behind the two headline rules** (MedDecide − baseline; S093):

| set | baseline | n items | macro-accuracy difference | paired 95 % CI | mean-Brier difference | paired 95 % CI |
|---|---|---|---|---|---|---|
| `fresh_seen` | `base` | 17,236 | **−0.0115** | [−0.0176, −0.0060] | −0.0243 | [−0.0298, −0.0186] |
| `fresh_seen` | `zeroshot` | 17,442 | **+0.1633** | [+0.1530, +0.1741] | −0.2298 | [−0.2345, −0.2253] |
| `fresh_heldout` | `base` | 0 | `NOT MEASURED` — no items scored by both models | — | — | — |
| `fresh_heldout` | `zeroshot` | 6,326 | **−0.0176** | [−0.0301, −0.0048] | −0.0113 | [−0.0198, −0.0022] |
| `fresh_strict_seen` | `base` | 1,898 | **−0.0279** | [−0.0589, −0.0043] | −0.0900 | [−0.1081, −0.0730] |
| `fresh_strict_heldout` | `zeroshot` | 699 | **+0.0323** | [−0.0181, +0.1030] | −0.0599 | [−0.0886, −0.0330] |
| `tier1` | `base` | 12,159 | +0.0700 | [+0.0544, +0.0832] | −0.0386 | [−0.0425, −0.0345] |
| `tier1` | `zeroshot` | 13,283 | +0.0244 | [+0.0119, +0.0360] | −0.0552 | [−0.0607, −0.0498] |

The Base ablation wins the headline seen comparison outright (*negative* difference, CI entirely
below zero), and the held-out comparison against zero-shot is negative. That is why the failure
is stated as a failure.

**Inputs** (all 37,289 rows, paired by path; S097): `meddecide` = `outputs/student_v0/S9_run3/
preds_meddecide-0p8b-lora-pointer__test.jsonl` (run `…20:58:27Z`), `base` = `outputs/student_v0/
S10/preds_meddecide-0p8b-lora-pointer__test.jsonl` (run `…01:35:06Z`,
`checkpoint_meta.base_model_id=Qwen/Qwen3.5-0.8B-Base`), `zeroshot` = `outputs/student_v0/S11/
preds_qwen3p5-0p8b.jsonl` (S080). **Pair by path**: S10's report files carry the instruct
`model_id` because the auto-launched evaluation omitted `--model-id` (S078); raw outputs were
left unedited (R9).

**Item sets** (S092): headline fresh 23,768 items / 11 templates (seen 17,442 / 7; held-out
6,326 / 4); strict slice (`record_date >= 2026-09-10`) 2,653 items / 11 templates (seen 1,954;
held-out 699); tier 1 13,716 items / 10 templates (12,159 in the base comparison after D12
exclusions). A D12-failing `(model, template)` cell drops that model's items of that template
from every comparison; the excluded cells are listed by name in `g1.md` and S092 (9 cells:
`meddecide` × 1, `base` × 8).

---

## 2. The models, all measured on the same item sets

| model | pass | wall / throughput | rule-selected checkpoint | tier-1 (13,521 items) | fresh (23,768 items) |
|---|---|---|---|---|---|
| **MedDecide-0.8B** (instruct `Qwen/Qwen3.5-0.8B`) | full pass, 44,152 steps / 212,481 items / 201,539,105 tokens; 13,867.5 s of a 14,174.0 s budget; 22.64 items/s; GPU peak 38.39 GB; **88 dev evals, a checkpoint at every eval** (S084) | `best/` = **step 1,000** by "highest dev macro accuracy" — 0.6798 macro, the maximum of all 88 evals (S084, S086) | micro **0.6505**, 7/8 D12 gates (S082) | micro **0.8227**, 11/11 gates (S082) |
| **Base ablation** (`Qwen/Qwen3.5-0.8B-Base`, same recipe/data/steps) | full pass, 44,152 steps; 13,760.4 s; 22.80 items/s; GPU peak 38.39 GB (S073) | `best/` = **step 17,000** by the same rule (margin 0.000691 over step 2,000; S074) | micro **0.5782**, 5/8 gates (S075) | micro **0.7577**, 6/11 gates (S075) |
| **Zero-shot `Qwen/Qwen3.5-0.8B`** (letter readout, D12 applied) | one pass over v0.2 test, 1,376.2 s (S080) | n/a | micro **0.6260**, 8/8 gates (S080) | micro **0.6771**, 11/11 gates (S080) |

Additional, explicitly labelled additional analyses (never substituted for the official model):

* **Instruct, converged step 44,000**: tier-1 **0.6141** (5/8) and fresh **0.7767** (7/11) —
  *lower* than the rule's pick on both tiers and with 6 fewer gates (S082).
* **Base ablation, converged step 44,000**: tier-1 **0.6090** (7/8) and fresh **0.7894** (9/11) —
  *higher* than its rule's pick (S076). The opposite of the instruct run.

**Dev numbers** (16,454-item S6 dev split, calibrated; S079): instruct step 1,000 — accuracy
0.7497 / macro 0.6653 / Brier 0.3215; Base ablation step 17,000 — 0.7373 / **0.7180** / 0.3131.
The dev sample ranks the two models the *opposite* way to the test tiers (item 7 of the
Summary; S079).

**Temperatures** (fit on dev, S085): instruct `choice` **1.2190** (n=10,967, FITTED) and `noul`
**2.8959** (n=5,454, FITTED), `score` **`NOT FITTED — too few dev items (n=33 < 50)`** (with
1.1131 recorded beside it as a diagnostic, not a fitted value). The Base ablation's fits are
4.1382 / 3.6320 / `NOT FITTED` (S073 artifact; `s10_temperature`). `score` has **no training
items at all** (S053), so its temperature is not identifiable from this mix.

---

## 3. Readout or capability? The held-out templates

The design's leakage control is the four D14 held-out templates, which have **0** training items
(S100: `ct_phase_choice_v1` 0, `fda_boxed_warning_noul_v1` 0, `pubmed_humans_noul_v1` 0,
`ct_arm_role_noul_v1` 0 of 212,481). Per-template accuracy on the fresh held-out set (S094):

| D14 held-out template | n | MedDecide | zero-shot | difference | Base ablation |
|---|---|---|---|---|---|
| `ct_arm_role_noul_v1` | 2,000 | 0.6165 | 0.6950 | **−0.0785** | `NOT MEASURED` — cell excluded by D12 |
| `ct_phase_choice_v1` | 1,100 | 0.2555 | 0.3455 | **−0.0900** | `NOT MEASURED` — excluded |
| `fda_boxed_warning_noul_v1` | 1,910 | 0.6215 | 0.6147 | +0.0068 | `NOT MEASURED` — excluded |
| `pubmed_humans_noul_v1` | 1,316 | 0.7242 | 0.6330 | +0.0912 | `NOT MEASURED` — excluded |
| macro over the four | 6,326 | **0.5544** | **0.5720** | **−0.0176** | — |

The four D14 cells are **not uniformly improved**, and the aggregate is below the zero-shot
letter baseline. By contrast, on the seen templates that *are* in the mix the trained model is
far ahead of zero-shot (macro 0.8894 vs 0.7262; S093), with the largest single gain on
`ct_claim_set_choice_v1` — a record–claim template with **19,997** training items (S100) — where
it reads 0.9417 against zero-shot's 0.2517 (S082, S093). That is the readout effect plus
same-template training; the held-out cells are the control that keeps it from being read as
decision skill. Leakage itself was checked mechanically: 2,188 rows removed by normalised
`(state+question)` hash with **0 collisions** on an independent re-check (S044), and the
training mix's own four-part check is 0 on all four kinds with 7,711 removals counted (S055).

Two caveats the next loop should carry:

* **The held-out aggregate is a small, heterogeneous set** (4 templates; 6,326 items) and two of
  its cells move the other way. "Below zero-shot on held-out" is the aggregate statement
  (S093), not a per-cell one.
* The Base ablation cannot be compared on the held-out set at all: **all four of its held-out
  cells fail D12** (3 constant-answer, 1 below-chance), so that comparison is
  `NOT MEASURED — no items scored by both models` (S092, S094). This is itself a finding about
  the Base ablation's readout health, not about its accuracy.

---

## 4. The three runs that are not results

### 4.1 Run 1 — failed by **pipeline**, not by recipe (S087)

Launched 2026-10-07T02:34:31Z (git `c4845d83`); stopped by operator instruction at 04:47Z and
preserved under `outputs/student_v0/S9_run1_sorted/`. Root cause: `iter_batches` sorted the
whole epoch by `char_proxy` **after** shuffling — an accidental length curriculum. The dev curve
shows the signature: best accuracy at **step 17,500** (0.5963 / macro 0.6248) then a slide to
0.4459 at step 36,500 over 73 evals; `best.json` (Brier-first, the superseded rule) picked step
18,000. No `planned_epoch.json` exists for run 1 — the plan dump was added with the fix — and
the guard is the regression test `tests/test_train_student.py::test_iter_batches_has_no_length_
curriculum` (the buggy ordering scores |rho| > 0.9; the fixed plan measures `rho_tokens`
**−0.000227** on both run 3 and S10; S087). Recorded here as failed-by-pipeline; it is **not**
evidence about the recipe.

### 4.2 Run 2 — **diverged** (S088)

Launched 05:22:33Z, stopped at **step 21,574 = 48.5 %** of the planned pass (104,134 items,
98.5 M tokens; rho = −0.000227, so not the curriculum). Artifact-derived gradient picture
(pre-clip `grad_norm` from `logs/train_steps.jsonl`):

| step window | n steps | p50 | p95 | max | >100 | >1000 |
|---|---|---|---|---|---|---|
| 1–4,000 | 4,000 | 5.61 | 100.8 | 7,632.1 | 202 | 18 |
| 4,001–8,000 | 4,000 | 3.80 | 143.8 | 100,257.3 | 269 | 16 |
| 8,001–12,000 | 4,000 | 2.79 | 98.3 | 23,545.9 | 199 | 16 |
| 12,001–16,000 | 4,000 | 1.59 | 60.0 | 4,808.0 | 131 | 11 |
| 16,001–18,000 | 2,000 | 1.47 | 35.3 | 3,407.5 | 50 | 5 |
| **18,001–20,000** | 2,000 | **10.49** | **3,908.7** | **43,186,144** | 482 | 199 |
| **20,001–21,574** | 1,574 | **47.62** | 3,566.0 | 240,868.7 | 576 | 179 |

Loss exceeded 2 on 27.5 % of steps after 18k (14.7 % overall), and dev collapsed: macro
**0.1149** / accuracy 0.3358 at step 21,500 (from macro 0.2785 at 18,000). 444 steps had a
pre-clip norm above 1,000. **Correction recorded rather than smoothed:** the prose summary
carried in STATE/the brief ("pre-clip grad norm averages 20–95, spikes to ~25,000") is *not*
reproduced by a direct read of the per-step log, which shows a much larger tail (p50 up to 47.6,
p95 ≈ 3.6k, and one 4.3e7 outlier). The window table above is the artifact-derived version and
is what S088 claims. Run 2 is recorded as diverged — not a result about the recipe.

### 4.3 S9-diag — one change at a time, and what it found (S089–S091)

Five arms, 2,000 steps each, on a fixed 20k-item subset (`limit=20000, stride=10`) and the same
2,019-item dev sample (S089):

| arm | change | loss first→last 500 | slope/1k (95 % CI) | grad p50 / p95 / max | #>100 | dev macro @500/1000/1500/2000 |
|---|---|---|---|---|---|---|
| `baseline` | run-2 recipe | 1.053 → 0.797 | −0.1915 ± 0.0609 | 5.79 / 31.07 / **971.9** | 14 | .599/.600/.613/.570 |
| `a_low_lr` | head 1e-4, LoRA 5e-5 | 1.006 → 0.695 | −0.2286 ± 0.0547 | 10.00 / 33.35 / 615.2 | 10 | .556/.627/.592/.582 |
| `b_cap2048` | prompt cap 2048 | 1.252 → 0.939 | −0.2168 ± 0.0526 | 8.65 / 31.90 / **325.4** | 10 | .519/.585/.622/.650 |
| **`d_rank8` ← picked** | **LoRA r=8, alpha 16** | 1.080 → 0.735 | −0.2498 ± 0.0605 | 6.10 / **29.47** / 457.8 | 10 | **.649/.655/.681/.666** |
| `e_rank8_lowlr` | r=8 + low LRs | 1.027 → 0.687 | −0.2465 ± 0.0536 | 9.65 / 32.86 / **310.0** | **7** | .569/.594/.605/.569 |

`d_rank8` was picked for the best dev macro at every eval among cap-comparable arms, the lowest
p95 and a worst-case norm less than half the baseline's. Run 3 then used it and completed the
full pass (S084). **Attribution caveat:** the arms were run one at a time on a subset that
averages 653 tokens/item against the full mix's 948, so *which* change (rank vs warmup+cosine)
was decisive is **not measured**; the pick bundles them.

**The spiking tail is length-driven (S090).** Baseline spike steps (pre-clip grad > 100) carry a
mean batch max length of **5,126** tokens against **2,217** overall (2.31×);
`fda_class_choice_v1` is 6.27 % of batches but **22.2 %** of spikes, `fda_class_choice_v2`
4.30 % → 11.1 %, while the short `medmcqa_4opt_v1` (29.79 % of batches) is only 8.3 % of spikes.
Capping prompts at 2,048 (`b_cap2048`) removes the worst tail (max 971.9 → 325.4) without
moving the bulk (p95 31.07 → 31.90).

**The trained head's logits depend on batch composition (S091).** Same item scored alone vs
inside a left-padded batch, tolerance 1e-3, 8 items: fresh/untrained head bf16 **8.00e-3 FAIL**
but fp32 **3.97e-4 PASS** (position handling is sound; bf16 accumulation is not); **trained head
(run 2's `S9/best`) bf16 9.45e-2 FAIL, fp32 2.51e-2 FAIL**, and the uniform-batch control is
1.40e-3 in fp32 — i.e. **17.9×** the control, and 63× the fresh-head fp32 figure. Training
amplified the discrepancy by ~60×. Mechanism not established (candidates: `key_end` marker
positions under left padding, or the head exploiting padded positions); magnitude is small next
to a trained head's logit spread, so it is recorded as a **contributing noise source and a
correctness bug in the padding path**, not on its own the divergence mechanism.

---

## 5. Findings the next loop needs

1. **The trained model has a readout advantage, not (yet) a capability advantage.** It beats
   the letter-readout zero-shot baseline on same-template fresh cells (macro +0.1633,
   S093) and loses to it on the D14 held-out aggregate (−0.0176, S093/S094), with 0 held-out
   training items (S100) and 2,188 duplicate rows removed at build time with 0 collisions
   (S044/S055). Hypothesis for the next loop (prose, no numbers): the pointer head reads option
   states directly, so any template whose construction cues it saw in training becomes easier,
   and the effect does not transfer. That needs a **construction-cue experiment** — train with
   one cue family held out and measure the same items with and without the pointer head.
2. **The instruct base is doing real work for readout health** (S096: 18/19 vs 11/19 gates on
   identical item sets). Any future Base-vs-instruct comparison must report gate counts beside
   accuracies, because D12 suppresses the accuracy of a failing cell (S092).
3. **The macro-first selection rule is validated on one model, not both** (S096, S082, S076).
   On the instruct run it picked the better checkpoint (18/19 vs 12/19); on the Base ablation it
   picked the worse one (11/19 vs 16/19). With 88 checkpoints on disk per run (S084), the next
   loop can re-examine selection on dev without retraining.
4. **The dev sample inverted the model ranking** (S079: Base 0.7180 vs instruct 0.6653 dev
   macro, yet 0.5782 vs 0.6505 tier-1 test; S075/S082). The 2,019-item template-stratified dev
   sample is not a proxy for the reported battery, and the selection rule keys on it.
5. **The padding/batch-composition dependence and the length-driven gradient tail need a real
   fix** (S091, S090) before another long run: either make the head padding-invariant (mask
   padded positions out of the marker/state gather), or score items without padding, and keep
   the 2,048-token cap as the documented fallback for the long-record tail (it bounds the tail;
   it also removes the long-record capability S3/S11 measure).
6. **`score` needs training items (and ≥ 50 dev items) before it can be reported at all**
   (S053, S046, S085). Today its tier-1 cell is a constant answer (`READOUT_FAIL`, S092).
7. **JEV-9B on v0.2 is `NOT MEASURED`** (S095): F7 measured it on v0.1 only. G1 legitimately ran
   without it, but the report must say the comparison does not exist rather than imply it.
8. **Two zero-shot numbers changed during the loop and the newer ones are authoritative**:
   `fda_route_claim_noul_v1` zero-shot is **0.9525** (the 0.8875 predates the truncation fix;
   S083), and S2's pre-fix 0.8875 must not be quoted again.
9. **The MeSH template is still saturated** (9B zero-shot 0.9830 against the ≤ 0.90 criterion;
   S021) even after near-miss distractors (S017), and its v0.2 zero-shot cell is 0.9580 with the
   trained model at 0.9685 (S093). It is not discriminating between models.
10. **PubMed pre-window training data is `NOT MEASURED`** (S058: the annual baseline is 1,334
    files / 31.4 GB / ~1.7 h, over the 60-minute box; a subset read would be a silent
    subsample). The training mix therefore has no PubMed records.
11. **S10's report files carry the instruct `model_id`** (S078). Pair inputs by path; the raw
    artifacts were left unedited (R9).

---

## 6. What is `NOT MEASURED`

| item | reason | source |
|---|---|---|
| JEV-9B on v0.2 (any tier) | F7 measured JEV-9B on v0.1; no v0.2 prediction file exists | S095 |
| held-out comparison vs the Base ablation | all four of its held-out cells fail D12, so the intersection is empty | S092/S094 |
| `score` temperature | 33 dev items < the 50-item floor, and no score training items exist | S085/S053 |
| PubMed pre-window training data | annual baseline over the time box; a subset would be a silent subsample | S058 |
| the operator's human audit (accept/reject per template) | `outputs/bench_v0_fix0/F10/audit_v0.jsonl` does not exist; only the staged sample does | S099 |
| which single diag change made run 3 stable | arms were run one at a time on a lighter subset | S089 |
| `nfcorpus_graded_score_v2` (both models, and every Base cell that failed) | D12 `READOUT_FAIL`; no accuracy is presented for a failing cell | S092 |

---

## 7. Spot-check — five claims re-run in fresh processes

(To be appended in the S14 self-audit; the five commands and their outputs are recorded below
once re-run.)

---

## 8. Method, provenance and what the next loop inherits

* **Benchmark v0.2**: 45,009 items, manifest sha256 `a81f2a03…9ed0db60b`; 11/11 independent
  verification checks PASS (S014, S015). The three repairs (score level set, MeSH near-miss
  distractors, mechanism-sharing FDA class distractors) are described in
  `loops/student_v0/bench_v0_2.md` and `template_screen_v0_2.md`.
* **Training mix**: 212,481 train items (tier-1 official train splits + pre-window
  structured-gold), 16,454 dev, 0 score items, four D14 templates excluded with counts (S053,
  S054, S056, S100).
* **Recipe**: frozen base + LoRA r=8/alpha 16 on 12 projections + per-option pointer head
  (`option_state=key_end`, 793,089 trainable params — S061/S067 plus the rank change in S089),
  CE + 1.0×Brier, linear warmup 3 % + cosine decay to zero on both groups, batch 8 / 8,192-token
  prompt cap, one epoch (S084).
* **Hardware / cost**: one RTX PRO 6000 96 GB, one GPU job at a time; the two full passes cost
  13,867.5 s + 13,760.4 s of training (plus 88 dev evals each) (S084, S073).
* **Process lessons this loop paid for** (recorded so they are not re-learned): a batching bug
  can masquerade as a recipe result (run 1, S087); a diverging run must be stopped and recorded,
  not re-labelled (run 2, S088); a selection rule can be "not the problem" and still not
  generalise (S096); and a CPU-only re-run of a GPU check needs its kernel path recorded
  (S098 records that S12 ran the reference PyTorch kernels because the fused kernels are
  CUDA-only).
* **What the next loop inherits**: `outputs/student_v0/S9_run3/checkpoints/step_<n>/` (88
  checkpoints) and `S10/checkpoints/step_<n>/` (88) with their `run.json` trajectories, so
  selection, calibration and readout experiments can be re-run without retraining; the v0.2
  benchmark and all prediction logs are on disk and gitignored.
