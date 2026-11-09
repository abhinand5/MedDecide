# d3_distill — STATE

Branch `distill/d3_distill`. Loop status: `RUNNING`. Last updated `2026-11-09T18:25Z`.
Operator: goal round 1 — do PLAN.md steps 0–3 in order, commit to `distill/d3_distill` only, no push, no branch deletion,
no MIMIC data.

## Environment (facts)

- Model env: `/workspace/venvs/d3` (torch 2.14.0+cu130, transformers 5.17.0, fla 0.6.0). Installed from
  `envs/d3/requirements.txt`; `pytest`/`ruff` added as dev tools only. `uv sync` has not been run.
- Snapshots in `$HF_HOME/hub` (`HF_HOME=/workspace/.hf_home`, `source scripts/pod_env.sh`):
  - teacher `vllm-sr/d3-ultra@ccdcfa611649a59e095e0b5e1732a15cedafadd6`, 49 GB, 11 shards + `readout.safetensors`;
  - `vllm-sr/d3-mini@df99bf4ca4418bc6908ee68e9b7479af1f3b8800`, 9.1 GB, 2 shards + readout (4.54B, hidden 2560, 32 layers);
  - `vllm-sr/d3-flash@67e03e1bc157961c582ee7c6a35d9b15fada3dec`, 16.8 GB, 4 shards + readout (8.39B, hidden 4096, 32 layers).
- GPU: NVIDIA RTX PRO 6000 Blackwell, 96 GB. One GPU job at a time.
- Validation of the teacher's public path passed earlier: `outputs/d3_bench/validate/validate.json` (7/7, 2026-11-09T15:32Z).

## Steps (PLAN.md)

| step | status | note |
|---|---|---|
| 0 — teacher throughput gate (≥ 25 items/s on 1,000 items; probability check every item) | **DONE 2026-11-09T16:05Z — PASS** | batched path 29.056 items/s end to end (34.417 s for 1,000), 0 invalid, 0 constant; `outputs/d3_distill/step0/throughput_batched.json`; SELF_AUDIT PASS |
| 1 — label set (CPU build, GPU label) | **DONE 2026-11-09T18:18Z** | 200,000 labels, **0 invalid**, 0 constant, sums within 2e-7, 26.5 rows/s (7,548 s); `labels.jsonl` sha256 `76a67a50…`; audit PASS (C018–C020) |
| 2 — students (port parity, arms, finals) | **IN_PROGRESS** | port parity PASS for both students (C023, C024); pplx baseline done (C021); training chain launching |
| 3 — Gate O7 | PENDING — frozen sets now present (operator update) | panel/knowledge/mr_verdict against a locally scored pplx-flash baseline; MIMIC BLOCKED (operator); MR vs osler_v4 final NOT MEASURED (no weights) |

## Step 0 — result

- Path: `D3.prepare` renders/tokenizes one one-question request per item (the card's shape); `D3.probabilities` runs the
  model's own `prefix_bidir` mask, forward, FP32 readout and softmax over length-sorted, left-padded batches of 8. It is
  not the one-request-per-call `system_one` API.
- Gate: 29.056 items/s end to end (prepare 0.390 s + forward 34.021 s), 0 invalid distributions, 0 constant, dev accuracy
  0.945 (identical to the single-request run). Gate basis: steady state after a 24-row warm-up that pays Triton autotune
  once; `TRITON_CACHE_AUTOTUNING=1`.
- The prior single-request figure 6.734 items/s (`outputs/d3_distill/step0/throughput.json`) is kept as a fact, not a
  verdict. Batched/single parity on 40 items: argmax 40/40, max abs difference 0.0354, mean 0.0028. **Operator decision
  2026-11-09: the batched path stands as the teacher; the 1e-4 bar applies to the students' port only.** Recorded in
  CLAIMS C003 and in the step 0 self-audit.
- Why batch 8: probe sweep on 256 items gave 26.9 (rows 4), 28.4 (8), 27.0 (16), 10.1 (32), 13.3 (64) items/s.
- Implication for step 1: 200,000 labels ≈ 1.9 h of GPU plus loader overhead. Gate margin is 16 %, so the 60,000 fallback
  is not needed.

## Step 1 — label run result (posted to STATE 2026-11-09T18:25Z, as requested)

- **200,000 of 200,000 pool rows labelled; 0 invalid; 0 constant.** 50 windows of 4,000, length-sorted batches of 8,
  teacher temperature 1.046; 26.5 rows/s; 7,547.8 s (2 h 06 m); peak 54.73 GB; teacher revision `ccdcfa61…`.
- Every answer is a distribution over exactly the row's options: sums in [0.9999997983, 1.0000002007], 0 non-finite or
  negative. Pools option counts: 2 → 9,546; 3 → 2,168; 4 → 182,929; 5 → 4,130; 9 → 128; 10 → 178; 12 → 921.
- `labels.jsonl` sha256 `76a67a505ba9f406b2015baaaa4631fb70b10524c02c82b494e21e340aa0e361`; pool sha256
  `2b41be76ea414ea346f57aefe7b8667666362bef1434c4587ee482e7f3ff895b` (C018).
- Per source: medmcqa 182,777; gen:code_assignment 1,842; gen:code_family 1,842; gen:criterion_eligibility 1,840;
  gen:note_allergy_med 1,844; gen:note_negation 1,844; gen:note_subject 1,844; gen:note_timing 1,844;
  mr:mr_icd_support_template_v2 1,672; mr:mr_indicator_evidence 1,730; mr:mr_page_category 921. Splits: train
  189,995 / dev 10,005 (never trained on).
- Teacher top-1 gold accuracy (diagnostic, in-distribution, not a benchmark): 0.9330 overall; MedMCQA 0.9354 (170,965 /
  182,777); synthetic families 0.742–1.000 (code_family 0.7416 and code_assignment 0.8040 are the hard ones) (C019).
- Independent audit (`scripts/d3_distill/step1_audit.py`, fresh process): counts close on both sides (0 duplicate ids,
  0 labels without a pool row, 0 pool rows without a label); 0 text matches with the three frozen Y9 sets (C016, C020).
- No MIMIC anywhere in the pool or the labels.

## Step 1 — pool (built 2026-11-09T16:11Z)

- `data/train/d3_distill/pool_200k.jsonl`: 200,000 rows, sha256 in `outputs/d3_distill/step1/pool_manifest.json`.
  182,777 MedMCQA official train + 17,223 code-generated: 12,900 from the 7 non-held-out `meddecide.gen` families and
  4,323 from the 3 `meddecide.mr` audit generators; 189,995 train / 10,005 train-side dev (5 % by row hash);
  190,894 choice + 9,106 noul.
- Drops, all counted: 45 duplicate MedMCQA texts, 4 cross-source duplicate texts, 296 held-out generator items,
  2,440 patient ids outside the train split. No MedMCQA train question appears in the validation (test) split (0).
- Enforced exclusions: MedMCQA validation texts (the knowledge test side), within-pool duplicates, held-out families
  (`note_lab_range_v1`, `policy_triage_v1`, AKI, heart-failure) and non-train generator patients.
- Enforced overlap: with the three frozen Y9 sets now on the machine, the pool was checked against all 22,912 of their
  items by exact normalised text key: **0 matches** (C016). Still NOT MEASURED (files absent): the MIMIC audit suite,
  MIMIC held-out, MedNLI test, the CG v2 dev proxy (needs the bench v0.2 manifest and osler_v1 B0-B3). The d3_bench run
  rows hold predictions without item text and cannot be hash-checked.

## Constraints and deviations (recorded before step 1 runs)

1. **MIMIC is out of scope on this machine (operator instruction: "don't … touch MIMIC data").** The plan's MIMIC-grounded
   synth block (16,582) and the MIMIC audit-suite part of Gate O7 are therefore NOT MEASURED, not silently dropped. Only
   MedMCQA train and the code-generated (no MIMIC, no teacher) synthetic families are used for the label pool. The
   `data/physionet` and MIMIC trees are not present in this checkout in any case.
2. **Frozen benchmark files are absent from this checkout.** `data/` does not exist; `data/bench/osler_v4/y9/*.jsonl`, the
   MR-suite files, the MIMIC audit suite, MedNLI test and `data/bench/osler_v4/cg_v2.jsonl` are not on this machine. The
   d3_bench run files hold predictions only (no item text), so they cannot be hash-checked either. Network is reachable
   (checked 2026-11-09T16:18Z), so the public test splits that *are* regenerable were rebuilt for diagnostics:
   `data/bench/d3_distill/medmcqa_val.jsonl` (4,183; sha `f1b85296…`) and `medqa_test.jsonl` (1,273; sha `0190a547…`),
   manifest in `outputs/d3_distill/step3/eval_sets_manifest.json`. They are not the frozen Y9 knowledge set and cannot
   stand in for Gate O7 (the d3_bench results.md says the same); they are additional diagnostics only.
   The absent files are recorded as NOT MEASURED with the reason, and no row drops silently.
3. **Frozen sets present (operator update, 2026-11-09).** `data/bench/osler_v4/y9/{knowledge,panel,mr_verdict}.jsonl`
   are now on this machine and their SHA-256 matches the originals (CLAIMS C014). They are evaluation inputs only, never
   committed, and never trained on. The MIMIC sets and every MIMIC-derived output stay off this machine (operator
   instruction), so the MIMIC part of Gate O7 stays BLOCKED. Gate O7 will be computed on the full frozen sets against a
   zero-shot pplx-decider-flash-v1.2 baseline scored here with the authors' code (snapshot present, CLAIMS C015); the MR
   check against the osler_v4 final stays NOT MEASURED (its predictions and weights are not on this machine).
4. **The CG v2 dev-proxy exclusion cannot be exact** without `cg_v2.jsonl` and osler_v1 B0–B3. The MedMCQA share of CG v2 is
   the knowledge dev v2 carve (3,000 of 182,822 train rows); any such overlap can only bias dev-side arm selection, never a
   reported gate, and Gate O7 does not use CG v2. Recorded as NOT MEASURED with that mitigation.
5. **No distill_v0 artifacts exist on this branch or in this repo's history.** `loops/` has no `distill_v0`; the "same
   recipe as distill_v0" is therefore reconstructed from the osler_v6 pre-registration (D57/D51) and the d3_distill PLAN
   bullets, and written into a `loops/d3_distill/ADVISORY.md` before any arm starts. Where the plan is silent (arm
   definition, dev-selection rule, LoRA target), the conservative reading is pre-registered and flagged for the operator.
6. PROVENANCE GAP carried from the hand-over: the d3_bench `results.md` says d3-ultra was not scored (disk quota), but
   `outputs/d3_bench/run/` now holds 1000 knowledge + 1000 panel + 500 mr_verdict rows. The PLAN's "3.5 items/s" and its
   benchmark comparison still have no committed CLAIMS row on this branch. Step 0 does not rely on them (it measures the
   teacher directly); steps 1–3 will cite the d3_bench aggregate only after it is committed as aggregates.

## Questions for the operator

1. MIMIC is excluded per your instruction, so step 3's "MR and MIMIC within −0.02 of the osler_v4 final" will have the
   MIMIC half NOT MEASURED. Confirm that reading (the alternative is that "don't touch MIMIC" meant "don't modify", which
   would allow local read-only scoring later).
2. Step 2's arm definition and dev-selection rule are not in the PLAN (no distill_v0 artifacts). Proposed, to be frozen in
   ADVISORY.md before any run: per student, arm A = D57 dosing + KL + gold CE, arm B = the same without KL (control); select
   on a train-side dev carve of the label pool (never a test set) by knowledge-dev accuracy + teacher KL; then one final at
   63,304 steps on the selected arm. Say if you want a different rule.

## Step 2 — port parity result (2026-11-09T18:45Z)

- **d3-mini PASS**: harness vs the student's own `D3.probabilities`, 40 items, one row at a time: max abs difference
  **0.0** (exact), argmax 40/40; `system_one` diagnostic 9.7e-08. Batch-8 composition diagnostic max 0.0113 (bf16
  padding numerics, the same phenomenon accepted for the teacher; not the gate). Peak 9.31 GB.
- **d3-flash PASS**: same check: max abs difference **0.0**, argmax 40/40; `system_one` 1.07e-07; batch-8 max 0.0149;
  peak 17.03 GB.
- The first parity run failed at 0.019 because the harness compared a 40-row padded pass against one-row references;
  per-row comparison is the prescribed check and is exact. The failure was in the diagnostic's batching/order, not in the
  port (the noul rendered-order bug in the diagnostic was fixed; the model always read No,Yes).
- Smoke runs of the arm driver: probe path 3 steps, band 0.429; train path 6 steps with eval/checkpoint writing.

## Step 2 — run in flight (2026-11-09T18:51Z)

- Chain PID **58940**, log `outputs/d3_distill/step2/logs/chain.log`, started 18:45:28Z. Stages: mini probe → mini arms
  (s0, s1) → selection → mini final → mini scores → flash probe → flash arms → selection → flash final → flash scores →
  Gate O7 report. One GPU job; every stage skips when its result exists.
- **λ calibration (mini)**: probe s0 = 0.431749 over 200 steps at λ=1 → **λ = 0.877441**, frozen. The arm's first 200
  steps must show a pre-clip share in 0.30–0.50 or the arm stops (D57).
- **d3-mini zero-shot dev accuracy = 0.8141 (8,145/10,005)** on the train-side dev carve; this is the arm's reference.
- Sampler: 23,744 steps/epoch, 189,952 of 189,995 train rows used per epoch (43 tail rows unused, logged), so the
  13,992-step arm sees 0.59 epochs and the 63,304-step final 2.67 epochs.
- Timing measured before launch: 60 steady-state mini steps at 0.11 s/step (peak 33.6 GB on a 4,096-row corpus); the
  probe's three backward passes peak at 46.8 GB. Expected chain wall clock ~12–15 h for both students including
  evaluations and scoring, subject to the long synthetic pages.

## Step 2 — execution plan (frozen 2026-11-09T16:40Z)

One detached chain, `scripts/d3_distill/step2_chain.sh` (log `outputs/d3_distill/step2/logs/chain.log`), runs:
1. the pplx-decider-flash-v1.2 baseline on the three frozen Y9 sets (authors' code, snapshot pinned; the Gate O7 baseline);
2. per student (mini first, then flash): port parity (1e-4, 40 items) → λ probe (200 steps at λ=1) → arms seeds 0/1
   (13,992 steps) → selection on the dev carve → final (63,304 steps) → scores on the Y9 sets and the public splits;
3. the Gate O7 report (`loops/d3_distill/GATE_O7.md`, aggregates only).

Stamp per run: config.json (data hashes, sampler, band), steps.jsonl, evals/, checkpoints/, arm_result.json, and a
`step2_audit.py` SELF_AUDIT after the chain returns. One GPU job at a time; every stage skips when its result exists.

## Step 2 — pre-registration

- `loops/d3_distill/ADVISORY.md`, written 2026-11-09T16:15Z before any student forward: port parity (1e-4 on 40 items),
  recipe (attention-only LoRA r=32/alpha=32 on q/k/v/o + trained readout; CE + λ·KL; batch 8; AdamW 1e-4/1e-3; cosine
  3 % warmup), D57 dosing and λ calibration, arms = seeds 0/1 at 13,992 steps, selection on the train-side dev carve,
  final at 63,304 steps, and the Gate O7 table with the checks that are NOT MEASURED here.
- Harness: `src/meddecide/distill/d3_train.py` reuses the student's own `D3.prepare` / `D3.encoder` / readout; the
  adapters start at zero so step 0 is exactly the frozen model. Unit tests in `tests/test_d3_batch.py` (12 pass).
- Drivers: `scripts/d3_distill/step2_parity.py`, `step2_arm.py` (probe/arm/final, resume, evals, checkpoints),
  `step2_select.py`. Scoring: `scripts/d3_distill/score_students.py`, eval sets: `build_eval_sets.py`.

## Iteration log

- 2026-11-09T15:44Z — session start; PLAN/STATE read. `data/` absent, snapshots for all three models present.
- 2026-11-09T16:05Z — step 0 gate PASS (29.056 items/s batched, 0 invalid); SELF_AUDIT written; CLAIMS C001–C005.
- 2026-11-09T16:11Z — step 1 pool built: 200,000 rows (C007–C010).
- 2026-11-09T16:12Z — step 1 labelling launched detached (PID 23702, log `outputs/d3_distill/step1/logs/label.log`);
  26.9 rows/s on the first 4,000-row window, 0 invalid; ETA ~18:20Z.
- 2026-11-09T16:19Z — public test splits rebuilt for step-3 diagnostics (MedMCQA validation, MedQA test); step-3 gate
  recorded BLOCKED on the missing frozen sets.
- 2026-11-09T16:25Z — step 2 pre-registration and harness written and unit-tested; parity/arms wait for the GPU.
- 2026-11-09T16:33Z — operator update: the three frozen Y9 sets were placed on the machine (hashes match); the teacher
  batched path is accepted; MIMIC stays off the machine; step 3 is to run the panel/knowledge/mr_verdict checks against
  zero-shot pplx-flash. The pplx-decider-flash-v1.2 snapshot (pinned revision) was downloaded for that baseline.
- 2026-11-09T16:29Z — the exact 1,000-item d3_bench knowledge subsample was rebuilt from public test splits (all ids
  matched; C013), so step 3 can at least compare the students to the d3_bench rows on the same items. Panel/MR remain
  unreconstructible (v0.2 exclusion manifest and frozen MR pids absent); the gate stays NOT MEASURED.
