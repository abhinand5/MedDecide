# d3_distill — distil vllm-sr/d3-ultra into vllm-sr/d3-mini (4B) and vllm-sr/d3-flash (9B)

Status: PLAN, awaiting the operator's go (rule decision D61 below). Teacher pinned: vllm-sr/d3-ultra, revision
ccdcfa611649a59e095e0b5e1732a15cedafadd6, Apache-2.0, 26.09B parameters, run locally on the GPU.

## Why this teacher, and why it is allowed
- The subsample benchmark (loops/d3_bench, aggregates) put d3-ultra above Clef-Flash and pplx-flash on knowledge, panel and MR
  verdict, close to the d1-ultra teacher. Subsamples are small; this is a direction, not a result.
- Runs locally: no hosted API, no provider terms, no data leaves the machine. MIMIC-derived items may go to this teacher, because it
  is a local model (D28 allows local processing).
- Apache-2.0 open weights: training on its outputs needs an operator decision only (D61), not a licence check.

## D61 (to be recorded before any labelling)
Soft labels from d3-ultra on general public training items and MIMIC-derived training items, produced locally, with gold kept
alongside (amends D13/D17/D48 for this local teacher only). Frozen test-side sets, the external panel, the knowledge test, the
MIMIC audit suite and held-out items are never sent to the teacher for labels.

## Step 0 — throughput (gate: go/no-go before any labelling)
The subsample run measured 3.5 items per second, sequential. Needed: about 200,000 items, so batching is required.
- Batch the forward passes (questions from the same batch share the padding; keep one question per request as the card requires).
- Gate: at least 25 items per second on 1,000 items, with the probability check (sum to 1, one entry per option, non-constant)
  passing for every item. If the gate fails, the target drops to 60,000 items and the plan is re-timed.

## Step 1 — label set (CPU to build, GPU to label)
- Sources: MedMCQA training (182,822), the synthetic families with no hosted-only content, the MIMIC-grounded synth block (16,582
  items; excluded: icd_specificity, held-out heart-failure and AKI), and any further official training split we verify.
- Excluded everywhere: the frozen knowledge test, the MedNLI test, the external panel, the MR suites, the MIMIC audit suite, the
  held-out files and the CG v2 dev proxy. Overlap checked by text hash; counts recorded.
- Target 200,000 items; labels written to gitignored outputs/d3_distill/, with per-source counts and the file hash.

## Step 2 — students (same recipe as distill_v0; both students)
- Students (operator swap, 2026-11-07): vllm-sr/d3-mini (4.54B, Apache-2.0) and vllm-sr/d3-flash (8.39B, Apache-2.0). Both are the
  same decision architecture as the teacher, so their decision heads are already trained; the readout-from-lm_head confound of the
  Clef-based plan does not apply. The 4B swap replaces pplx-flash as the student; zero-shot pplx-flash stays the gate baseline
  (pre-registered, unchanged).
- Port step (new, required before labelling starts): a training harness for the d3 architecture (text-only path; the vision
  encoder is frozen and unused). Gate: the loaded student reproduces its own decision probabilities on 40 items within 1e-4, the
  same parity check as the pplx and Clef ports. Timebox 3 h; if it fails, both students are BLOCKED and the teacher labels still
  go ahead as a dataset.
- Dosing (D57): at least 3 labelled items per batch; per-item weighting (D51); λ calibrated on the first 200 steps (30–50 % gradient share).
- Two arms per student at 13,992 steps, then one final at 63,304 steps for the best arm (the distill_v0 recipe unchanged).

## Step 3 — gate (Gate O7, unchanged from distill_v0)
- Panel lower bound > 0 against zero-shot pplx-flash; knowledge lower > −0.02; MR and MIMIC within −0.02 of the osler_v4 final.
- The teacher is reported as an additional row. The students are judged on the gate only.

## Timing (planning estimates; to be replaced by measurements)
- Port step: up to 3 h. Step 0: 1–2 h. Labelling 200k items at the gated rate: 3–5 h (16 h if the gate fails and throughput stays at 3.5/s).
- 4B arm + final: about 20 h. 9B arm + final: about 35 h. Gate and findings: 3 h.
- Total: about 2.5 days, with the 9B running after the 4B. One GPU job at a time.

## Stop conditions
- Step 0 gate fails and the 60k target also cannot finish in 12 h: stop and report.
- Any teacher output that fails the probability check is dropped and counted; a rate above 2 % stops the labelling.
- Any MIMIC content outside the local GPU stops the run immediately.
