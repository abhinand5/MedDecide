# d3_distill — STATE

Branch `distill/d3_distill`. Loop status: `RUNNING`. Last updated `2026-11-09T16:20Z`.
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
| 1 — label set (CPU build, GPU label) | PENDING | pool build next; 200,000 target |
| 2 — students (port parity, arms, finals) | PENDING | depends on step 1 labels |
| 3 — Gate O7 | PENDING | depends on step 2 |

## Step 0 — result

- Path: `D3.prepare` renders/tokenizes one one-question request per item (the card's shape); `D3.probabilities` runs the
  model's own `prefix_bidir` mask, forward, FP32 readout and softmax over length-sorted, left-padded batches of 8. It is
  not the one-request-per-call `system_one` API.
- Gate: 29.056 items/s end to end (prepare 0.390 s + forward 34.021 s), 0 invalid distributions, 0 constant, dev accuracy
  0.945 (identical to the single-request run). Gate basis: steady state after a 24-row warm-up that pays Triton autotune
  once; `TRITON_CACHE_AUTOTUNING=1`.
- The prior single-request figure 6.734 items/s (`outputs/d3_distill/step0/throughput.json`) is kept as a fact, not a
  verdict. Batched/single parity on 40 items: argmax 40/40, max abs difference 0.0354 (numerics, not 1e-4).
- Why batch 8: probe sweep on 256 items gave 26.9 (rows 4), 28.4 (8), 27.0 (16), 10.1 (32), 13.3 (64) items/s.
- Implication for step 1: 200,000 labels ≈ 1.9 h of GPU plus loader overhead. Gate margin is 16 %, so the 60,000 fallback
  is not needed.

## Constraints and deviations (recorded before step 1 runs)

1. **MIMIC is out of scope on this machine (operator instruction: "don't … touch MIMIC data").** The plan's MIMIC-grounded
   synth block (16,582) and the MIMIC audit-suite part of Gate O7 are therefore NOT MEASURED, not silently dropped. Only
   MedMCQA train and the code-generated (no MIMIC, no teacher) synthetic families are used for the label pool. The
   `data/physionet` and MIMIC trees are not present in this checkout in any case.
2. **Frozen benchmark files are absent from this checkout.** `data/` does not exist; `data/bench/osler_v4/y9/*.jsonl`, the
   MR-suite files, the MIMIC audit suite, MedNLI test and `data/bench/osler_v4/cg_v2.jsonl` are not on this machine. The
   only frozen benchmark text here is the d3_bench subsample extract (`outputs/d3_bench/run/{knowledge,panel,mr_verdict}.jsonl`).
   Overlap checks will be run against every available frozen text and against the MedMCQA validation split (the knowledge
   test side); the absent files are recorded as NOT MEASURED with the reason, and no row drops silently.
3. **The CG v2 dev-proxy exclusion cannot be exact** without `cg_v2.jsonl` and osler_v1 B0–B3. The MedMCQA share of CG v2 is
   the knowledge dev v2 carve (3,000 of 182,822 train rows); any such overlap can only bias dev-side arm selection, never a
   reported gate, and Gate O7 does not use CG v2. Recorded as NOT MEASURED with that mitigation.
4. **No distill_v0 artifacts exist on this branch or in this repo's history.** `loops/` has no `distill_v0`; the "same
   recipe as distill_v0" is therefore reconstructed from the osler_v6 pre-registration (D57/D51) and the d3_distill PLAN
   bullets, and written into a `loops/d3_distill/ADVISORY.md` before any arm starts. Where the plan is silent (arm
   definition, dev-selection rule, LoRA target), the conservative reading is pre-registered and flagged for the operator.
5. PROVENANCE GAP carried from the hand-over: the d3_bench `results.md` says d3-ultra was not scored (disk quota), but
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

## Iteration log

- 2026-11-09T15:44Z — session start; PLAN/STATE read. `data/` absent, snapshots for all three models present.
- 2026-11-09T16:05Z — step 0 gate PASS (29.056 items/s batched, 0 invalid); SELF_AUDIT written; CLAIMS C001–C005.
