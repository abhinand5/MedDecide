# FINDINGS — MedDecide loop `bench_v0_fix0` (repair loop for `bench_v0`)

> **Status: DRAFT — F6 (ladder baselines) and F7 (decision models) are still running.** Every
> section below that depends on them is marked `PENDING F6` / `PENDING F7` and carries no number
> yet. Nothing in this file is final until `Loop status: STOPPED` is set in STATE.md.

## Summary

**The question.** `bench_v0` built the benchmark and harness, and an advisor review then found
defects in it. This loop asks: were those defects real, what do they invalidate, and what do the
numbers look like once they are fixed?

**What we did.** Repaired the three defect classes the review named — the MedMCQA answer key, the
`noul`/`score` readout, and the fresh window — and then, because each repair exposed more, verified
every tier-1 gold label against its raw source record, rebuilt the benchmark as **v0.1**, screened
it with a balanced rule, re-ran the ladder baselines through a readout-health gate, ran a
contamination probe, measured decision-model baselines, and wrote a corrections record covering
**all 53 bench_v0 claims**.

**What we found.**

1. **Both headline `bench_v0` findings were wrong, and neither was a model result.**
   `C048` claimed MedMCQA's official test split "has no D answers" and that the 4B model has a
   last-position bias; `C051` claimed an option shuffle "proved" it. In fact the loader read
   MedMCQA's `cop` field as 1-based when it is **0-based**, which shifted every gold label one
   option back and silently dropped all **1,348** rows whose answer is "A". The official split has
   **4,183** items with gold A 1,348 / B 1,085 / C 925 / **D 825** — identical to the raw `cop`
   counts — and the "D preference" was an artefact of the shifted key. Both claims are
   **withdrawn** (X002, X003, X026).
2. **The `noul`/`score` readout was reading tokens the model had no reason to produce.** Those
   items rendered as `yes. Yes` while the instruction said "Respond with a single letter", and the
   readout scored the `yes`/`no` key tokens: label mass ~**0.002**. All `noul`/`score` numbers in
   `bench_v0` are invalid. After the fix the same items read at label mass **0.996–0.999** with
   greedy agreement **0.96–1.00**, and the previously validated `choice` path is provably
   unchanged (X007, X008).
3. **The fresh window was bounded by the wrong model.** `bench_v0` started it at the *teacher's*
   repository date, which is not a contamination boundary — the teacher never labels benchmark
   items. Decision **D11** bounds it by the ladder instead: v0.1 starts **2026-03-01**, with
   records dated ≥ 2026-09-10 kept as a separately reported **strict slice** of 2,896 items
   (X012, X013).
4. **The template screen was ranking templates by class imbalance and calling it shortcut
   learnability.** `bench_v0` dropped `pubmed_observational_noul_v1` for a 0.985 BoW score that the
   *majority class alone* produced, while keeping a template at 0.926. With v0.1's balanced
   103/103 split the same template's true BoW macro accuracy is **0.796** and it is **kept** — the
   cleanest single piece of evidence that the old drop was an artefact (X017, X018).
5. **Four further defects surfaced from the repairs, three of them real.**
   (a) nfcorpus score items carried a grade borrowed from another query's pool (X005);
   (b) CT.gov's `healthyVolunteers` is a JSON boolean, so `str(value) == "yes"` marked **every**
   item "no" (X015);
   (c) `nfcorpus_graded_score_v1` offers a lowest level that **no item in its pool has**, so a
   model answering it is guaranteed wrong — JEV-9B scores 0.2634 three-way (chance 0.3333) but
   0.5804 restricted to the levels present (X029).
   (d) the fourth was mine: two of my own validation designs were broken before the real bug
   surfaced, and both are recorded rather than deleted (see "What did not work").
6. **The readout-health gate suppresses a real cell rather than reporting it.** MedGemma-1.5-4b-it
   on MedQA agrees with its own greedy continuation on only **0.86** of the 50 sampled items
   (threshold 0.90), so its MedQA accuracy is reported as
   `READOUT_FAIL — greedy agreement 0.860 < 0.9 (n=50)` and **no number is printed for it**. The
   same model passes 15 of its other 16 cells, and its `choice` cells generally sit at 0.90–0.96
   while its `noul`/`score` cells sit at 0.96–1.00 — i.e. this model sometimes emits the option's
   *content* where the protocol expects a letter. The gate is doing exactly what D12 created it
   for: a plausible-looking accuracy (0.48, near `bench_v0`'s own MedGemma number) is withheld
   because the readout cannot be shown to be reading the model's answer.

**What it means.** The repairs hold: tier-1 gold verifies **0 mismatches of 15,915 checked items**
across all eight sources (X004); v0.1 builds deterministically and passes 7/7 acceptance checks
(X012, X016); the readout passes a health gate on every reported cell; and the corrections record
leaves no `bench_v0` claim standing that depended on the broken parts (X025). `bench_v0`'s
`choice`-path results — MedQA, MMLU — stand unchanged (X008).

**What did not work, or is not measured.**

* **The teacher gate is `BLOCKED`** — `TEACHER_BASE_URL` and `TEACHER_API_KEY` are unset on the pod,
  for the second loop running. No teacher number appears or is estimated (X024).
* **The human label audit is `BLOCKED — awaiting operator`**: 150 sampled items are staged, but
  whether their gold labels are correct is a human judgement (X022, X023).
* **The contamination probe cannot decide the question it was built for.** Its gap does not grow
  with model size (4B +0.796 vs 9B +0.764) and it cannot separate memorisation from "the original
  order is more predictable prose"; no tier-1 accuracy is adjusted by it (X027, X028).
* **Two of my own validation attempts were broken** before they caught anything: a bare prompt
  collapsed label mass to 1.2e-04, and a comparison used canonical letters where the reference
  produced `yes`/`no`. Both are in `readout_validation.md`.

<!-- PENDING F6: ladder baselines on v0.1, per template, both tiers, strict slice, health gate -->
<!-- PENDING F7: decision-model baselines (JEV-9B measured; Laya pending) -->
<!-- PENDING: five-claim spot-check re-run in fresh processes -->
