# NEXT — proposals for the next loop (not decisions)

The `student_v0` loop stopped at its hard stop with **G1 FAIL on all four rules** (see
`FINDINGS.md` §1 and `g1.md`). ADVISORY §2's third branch applies: *diagnose the recipe
(readout, capacity, loss, data) at 0.8B before spending anything larger.* Everything below is
a proposal for the operator to accept, change or reject; none of it is a result, and none of it
was started. Ordered by my estimate of value per GPU-hour.

## 1. Fix the batch-composition (padding) dependence before any long run

*What is measured:* on the trained head, the same item scored alone vs inside a left-padded
batch differs by 2.51e-2 in fp32 (17.9× the uniform-batch control), while the fresh head passes
in fp32 (3.97e-4) — S091. The head reads `key_end` states by position, so a shift in the
marker/pad layout is a plausible cause; it is not established.

*Proposal:* make the head padding-invariant — gather the option states at their own marker
positions with the padded rows excluded, or construct the batch so the answer position is
identical for every row — then re-run the diag padding probe as a **unit test** (fp32, 8 items,
tolerance 1e-3). Cheap, CPU-only, and it removes a known correctness bug from every future
number.

## 2. Re-audit the length-driven gradient tail

*What is measured:* spiking steps carry 2.31× the mean batch max length, and long openFDA /
ClinicalTrials templates are over-represented among them (S090); run 2 diverged with p50 rising
to 47.6 and one 4.3e7 pre-clip norm (S088); a 2,048-token cap bounds the tail (max 971.9 →
325.4) but removes the long-record capability S3/S11 measure (S089).

*Proposal:* keep the 8,192 cap for the run that measures long records, but (a) normalize the
loss per real token rather than per padded batch, (b) log `grad_norm` against batch max length
for every step so the tail is attributable without a diag run, and (c) test a length-stratified
gradient clip. Report the tail distribution with every training run, not only in a diag.

## 3. Give `score` training items — or drop it from the reported battery

*What is measured:* 0 score items in training (S053), 33 in dev (< the 50-item fit floor) so its
temperature is `NOT FITTED` (S085), and its one tier-1 cell is `READOUT_FAIL — constant_answer`
(S092). The official NFCorpus train qrels are binary, so the v0.2 rule cannot build a score
template from them (S046).

*Proposal:* build `score` items from a source with graded human labels (the S13 catalog lists
candidates with human/structured provenance), or report `score` as `NOT MEASURED` everywhere
until such data exists. Do not report a score accuracy from a model that never saw a score item.

## 4. Re-think the dev sample and the selection rule — as a *measured* question

*What is measured:* the 2,019-item template-stratified dev sample inverted the S9-vs-S10 test
ranking (dev macro 0.7180 vs 0.6653, test tier-1 0.5782 vs 0.6505; S079), and the macro-first
rule helped the instruct run (18/19 vs 12/19 gates) but hurt the Base ablation (11/19 vs 16/19;
S096). 88 checkpoints per run are already on disk (S084, S073).

*Proposal:* without retraining, evaluate a handful of selection rules (dev macro, dev Brier,
last-step, per-template balanced variants) on the saved checkpoints and report which rule would
have picked which test-optimal checkpoint. Then freeze the rule that generalises across both
models, or select per-model with a dev-based guard and report the choice explicitly.

## 5. Separate readout from capability with a construction-cue experiment

*What is measured:* a large same-template advantage over the letter-readout zero-shot baseline
(+0.1633 macro on seen templates) and a small held-out deficit (−0.0176; S093/S094), with 0
held-out training items (S100). The hypothesis (prose, unproven) is that the pointer head
rewards construction cues learned per template.

*Proposal:* build a paired experiment on the same items — (a) pointer head, (b) adapter-off
letter readout, (c) adapter-on letter readout — on templates whose cue families are held out by
construction, so the readout contribution is measured rather than inferred. If the pointer-head
advantage survives on templates with unseen cue families, it is capability; if it does not, the
next loop should invest in cue diversity, not scale.

## 6. Keep the readout-health framing for instruct-vs-Base comparisons

*What is measured:* 18/19 vs 11/19 D12 gates on identical item sets (S096), with four Base
held-out cells excluded entirely (S092).

*Proposal:* every future Base/ablation comparison reports gate counts beside accuracies and
names the excluded cells. If a Base-trained head is compared against an instruct letter
baseline, add a control that isolates the instruct tuning of the letter path itself.

## 7. Close the two missing measurements

* **Byte-identity on the shipped kernel path.** S12 ran CPU-only on transformers' reference
  PyTorch kernels (S098); a GPU re-run with `causal_conv1d` / `fla` bound would confirm the
  adapter-off identity on the path the model actually ships with. ~10 minutes.
* **JEV-9B on v0.2.** `NOT MEASURED` (S095); one 30-40 min GPU job per the S11 plan would give
  the next G1 its second baseline. (F7 measured v0.1 only.)
* **The operator's human audit** (`outputs/bench_v0_fix0/F10/audit_v0.jsonl`) is still pending
  (S099); without it the label-quality read-back cannot be reported.

## 8. Housekeeping for the next loop's CLAIMS

* The S014 manifest hash is not reproducible from the current file (S101): the manifest was
  updated by the S2/S3 builders after the S1 build. Prefer an explicit `manifest_sha256` field
  written by the builder, or hash a dedicated immutable file, so a hash claim stays checkable.
* Replace prose summaries of run statistics with artifact-derived tables where the two disagree
  (the run-2 gradient summary is the example: S088).
* Keep one checkpoint per dev eval in every future training run — it made the selection
  question answerable after the fact without retraining, at ~28 MB per checkpoint.
