# distill_v0 — distilling the d1-ultra decision distribution into Osler-4B and Osler-9B

Loop `distill_v0`, branch `loop/distill_v0`, 2026-11-03 → 2026-11-05. Teacher:
`liquid/d1-ultra-20261015` through the OpenRouter decisions API (30,259,634 prompt tokens, **$7.90** of the
$15 cap). Students: Osler-4B on `perplexity-ai/pplx-decider-flash-v1.2` and Osler-9B on
`Cloudflare/clef-flash` (text tower loaded in memory), both with a frozen base, LoRA r=32/α=32 on
q/k/v/o and an option-code readout initialised from `lm_head`.

**Outcome: Gate O7 FAIL for both students.** The distillation transferred the trained decision
families strongly — both students close ~14 points of MR-verdict accuracy over the zero-shot base and
~13–14 points on the MIMIC audit suite — but neither clears the two external checks: the panel lower
bound against zero-shot pplx-flash is ≤ 0 (4B −2.18 points [−3.43, −0.81]; 9B −0.27 points
[−1.98, +1.33]), and the MIMIC-suite lower bound against the osler_v4 final is below −2 points
(4B −3.42 [−4.49, −2.45]; 9B −2.54 [−3.55, −1.47]). The 9B passes the knowledge guard and ties
zero-shot pplx on the panel and beats the osler_v4 final there by 3.4 points; the 4B misses the
knowledge bound by 0.15 points. Both students are on the measured ≤10B medical frontier. Per the plan
this is the hard stop: no further training.

## Gate O7 — the frozen verdict (task D6)

Four pre-registered checks on the frozen test-side sets; item-paired stratified-bootstrap lower bounds
(1,000 resamples, seed 0; `meddecide.eval.gate_o4`), exactly the estimator osler_v4's Gate O4 used
(the D6 wiring validation reproduces osler_v4's published Gate O4 numbers: panel −0.0421
[−0.0584, −0.0269], knowledge −0.0067, MR +0.1387, MIMIC +0.1654).

| check | set | baseline | Osler-4B Δ acc [95 %] | verdict | Osler-9B Δ acc [95 %] | verdict | limit |
|---|---|---|---|---|---|---|---|
| i | panel | zero-shot pplx-flash | **−0.0218** [−0.0343, −0.0081] | FAIL | **−0.0027** [−0.0198, +0.0133] | FAIL | lower > 0 |
| ii | knowledge | zero-shot pplx-flash | **−0.0117** [−0.0215, −0.0013] | FAIL | **−0.0009** [−0.0150, +0.0150] | PASS | lower > −0.02 |
| iii | mr_verdict | osler_v4 final | **+0.0203** [+0.0107, +0.0317] | PASS | **+0.0163** [+0.0054, +0.0271] | PASS | lower > −0.02 |
| iv | mimic_suite | osler_v4 final | **−0.0342** [−0.0449, −0.0245] | FAIL | **−0.0254** [−0.0355, −0.0146] | FAIL | lower > −0.02 |

- Paired items per check: 4,353 (panel), 5,435 (knowledge; 21 items not scored by the baseline), 13,103
  (mr_verdict), 4,253 (mimic_suite). Brier deltas are reported alongside in `gate_o7.json` (same signs
  as accuracy for iii/iv; for i/ii the students' Brier is also worse than pplx-flash).
- Verdict: `Gate O7: FAIL` (`outputs/distill_v0/D6/gate_o7.json`; independent re-derivation 12/12
  pass, `audit.json`).
- Readout health: no student or baseline cell failed its readout check (`student_readout_failures` and
  `baseline_readout_failures` are empty for all four checks).

## Where the students stand (scoreboard, micro accuracy on the nine frozen sets)

| model | params | knowledge | panel | mr_verdict | mednli | mr_reported | mr_heldout | mimic_suite | pairs | mimic_heldout |
|---|---|---|---|---|---|---|---|---|---|---|
| **Osler-4B** | 4B | 0.6752 | 0.5392 | **0.8893** | 0.8840 | 1.0000 | 0.3510 | 0.7585 | **0.9680** | 0.6727 |
| **Osler-9B** | 9B | 0.6837 | **0.5596** | **0.8960** | 0.8755 | 1.0000 | 0.3305 | 0.7682 | **0.9680** | 0.6909 |
| pplx-flash (base, zero-shot) | 4B | 0.6835 | 0.5615 | 0.7466 | 0.8720 | 0.9984 | 0.6130 | 0.6240 | 0.8740 | 0.7000 |
| osler_v4 final | 4B | 0.6765 | 0.5254 | 0.8862 | 0.8762 | 1.0000 | 0.4572 | **0.7947** | 0.9660 | 0.6364 |
| MedDecider-4B | 4B | 0.6703 | 0.5436 | 0.6681 | 0.8488 | NOT MEASURED | 0.5205 | 0.6019 | 0.6920 | 0.6636 |
| MedDecider-9B | 9B | 0.6997 | 0.5603 | 0.6653 | 0.8425 | NOT MEASURED | 0.4024 | 0.6189 | 0.7880 | 0.6455 |
| JEV-9B | 9B | 0.6448 | 0.5127 | 0.6303 | 0.8221 | 0.9950 | 0.4041 | 0.5885 | 0.8920 | 0.7000 |
| Qwen3.5-4B | 4B | 0.6162 | 0.4831 | 0.6176 | 0.7862 | 0.9939 | NOT MEASURED | 0.5603 | NOT MEASURED | 0.6364 |
| Qwen3.5-9B | 9B | 0.6774 | 0.5419 | 0.6315 | 0.8361 | 0.9945 | NOT MEASURED | 0.6029 | NOT MEASURED | 0.6727 |
| Clef-Flash (zero-shot) | 9B | 0.6864 | 0.5711 | 0.6830 | 0.8636 | 0.9953 | NOT MEASURED | 0.6306 | NOT MEASURED | 0.7182 |
| pplx-27B (out of tier) | 27B | 0.7823 | 0.6510 | 0.8093 | 0.8748 | 1.0000 | 0.8990 | 0.6729 | 0.9400 | 0.7818 |
| osler_v2 | 4B | 0.6752 | 0.5343 | 0.8963 | 0.8692 | 1.0000 | 0.3527 | 0.5551 | 0.9140 | 0.5091 |

Macro (template) numbers and per-set item counts are in `gate_o7.json` (`scoreboard`).

**Frontier.** On the measured model set restricted to ≤ 10B, both students are on the (weak) Pareto
frontier: no measured model dominates them on the nine sets — they lose the panel/knowledge to
pplx-flash and Clef-Flash but win MR verdict, MIMIC suite and pairs by wide margins. pplx-27B (27B)
dominates them on panel, knowledge, held-out MR and held-out MIMIC, but is outside the ≤10B tier.
`frontier`: `student4b on_frontier true`, `student9b on_frontier true`, no dominators.

**What the distillation bought, set by set.** Both students beat their own zero-shot base on the
trained decision families: mr_verdict +14.3/+14.9 points, mimic_suite +13.5/+14.4, pairs +9.4, while
losing held-out MR (−26.2/−28.3 vs the base) and matching or nearly matching the base on knowledge
(−0.8/+0.0) and panel (−2.2/−0.3). Against the osler_v4 final the students are better on mr_verdict
(+0.3/+1.0), panel (+1.4/+3.4), knowledge (−0.1/+0.7) and pairs (+0.2), and worse on mimic_suite
(−3.6/−2.7) and mr_heldout (−10.6/−12.7), better on mimic_heldout (+3.6/+5.5).

## The teacher ceiling (task D1)

Zero-shot `d1-ultra` on the same nine sets, noul-only prompting (the API returns option labels; the
open-text arms use the same label vocabulary to stay comparable — parity 37/40 argmax, max |Δp| 0.208):

| knowledge | panel | mr_verdict | mednli | mr_reported | mr_heldout | mimic_suite | pairs | mimic_heldout |
|---|---|---|---|---|---|---|---|---|
| 0.8167 | 0.6825 | 0.7387 | 0.8812 | 1.0000 | 0.8322 | 0.6901 | 0.9360 | 0.7364 |

The teacher is far ahead on knowledge (+14 points over both students) and the panel (+12/+15 over the
students), weaker than both students on mr_verdict (0.739 vs 0.889/0.896) and the MIMIC suite (0.690
vs 0.759/0.768). The students' MR/MIMIC strength therefore comes from the gold CE on the distilled
families, not from copying the teacher; the teacher's knowledge and panel advantage did not
transfer through 39 % KL items in 63,304 steps (hypothesis below).

## Labels and dose (tasks D2–D5)

- **Basket** 494,803 pool rows / 494,748 unique text keys (the GOAL's 508,516 was a per-template
  estimate; the realisable pool is 494,803 and every available row is used — deviation 2). Teacher
  items 39.0 % of pool slots, MIMIC 39.3 % (band 38–45 % / 37–42 %).
- **Labels**: 494,748 frozen teacher distributions, 0 failures/skips, 0 test items, no rule-lookup
  row; $6.63 label cost; loop spend $7.90 of $15 (teacher ceiling $1.27).
- **Training**: 63,304 steps each; λ calibrated at the first 200 steps to 3.2067 (4B) and 2.2077
  (9B) for a predicted 40 % distillation gradient share, then frozen; realised dose teacher 38.94 %
  of item slots in both runs, MIMIC 39.35 %, mean 1.46 teacher items per batch (3.12 per 8 slots).
  Four full checkpoints and ten per-eval checkpoints exist per run; final dev S 0.8486 (4B) /
  0.8357 (9B).
- **Costs**: $7.90 total (teacher ceiling $1.27 + basket labels $6.63); 30.26 M prompt tokens; 16.7 h
  (4B) and 27.8 h (9B) of a single RTX PRO 6000 Blackwell.

## Deviations from the plan (full list in STATE.md §6)

1. **Basket size** 494,803 vs the GOAL's 508,516 (nine per-template counts stale in the GOAL; the
   realisable pool is what exists). Every available row is used; shares stay in band. Not a silent
   drop — `composition.json` lists available/selected per template.
2. **Text keys** — D2 rows are keyed by text SHA-256, not `item_id` (Winogrande twins share item
   ids: 5,232 ids, 27,340 rows); pool indices (`basket_indices`) map deterministically.
3. **D2 restarts** — three launch attempts (rate limit, disk quota); the 1,728 labels lost to a
   resume truncation at 90,528 are booked in the ledger ($0.0079) and documented as a gap; the final
   label file is complete (494,748 rows) and hashed.
4. **λ calibration window** — the first attempt measured 200 *teacher-bearing* steps; fixed to the
   first 200 steps of the run before the reported runs (archived under `logs/run1_stopped/`).
5. **Dose normalisation** — the mean teacher items per actual batch is 1.46, not 3.2, because the
   token-capped length-bucketed sampler averages 3.74 items per batch; per 8 item slots the mean is
   3.12, inside the planned band. The share tripwire is the operative check and passes.
6. **9B memory fallback** — the first 9B attempt OOM'd at ~90 GB because the F2 speed setting
   disabled checkpointing at the 8,192-token cap; the 9B checkpoints every batch (token cap, batch
   size and steps unchanged). Peak memory ~23 GB.
7. **D6 loader** — the 4B checkpoint has no `decision_config.json` (it is a saved student, not a
   hub checkpoint); the loader now dispatches on the `backbone` field.
8. **MIMIC authorization** — the operator's 2026-11-03 email confirmations cover the MIMIC-derived
   baskets sent to the hosted teacher (recorded in the D2 manifest and STATE).

## Hypotheses (unmeasured; prose only)

1. *Why MR and MIMIC moved but the panel and knowledge did not.* The MR/MIMIC families are
   template-homogeneous and their labels are deterministic (gold from structured fields or
   code-computed verdicts); the teacher's option distributions on those families are close to the
   gold and the KL sharpens the student toward the decision path. The panel and knowledge sets ask
   general clinical questions whose accuracy is bounded by the base model's stored knowledge and its
   option-code readout, which 63k steps of LoRA on 39 % teacher items does not move. The teacher's
   knowledge (0.817) shows the information was available to distil; it did not transfer.
2. *Why the MIMIC-suite gap to the osler_v4 final persists.* The final was trained with roughly 60 %
   of the gradient on MIMIC items (osler_v4's advisor note); these students give MIMIC 39 % of slots,
   a fifth of them gold-only rows without teacher probabilities, so the MIMIC-specific decision
   patterns are diluted relative to the final's recipe.
3. *Why held-out MR collapses.* Both students' mr_verdict gain is concentrated in the seven trained
   templates; the held-out templates change the question form and the students fall to 0.33–0.35,
   below the final and far below the zero-shot base — consistent with the prior loops' finding that
   held-out skill falls with training, and with template-family overfitting.
4. *Why the 9B ≈ or > the 4B off-panel but not on held-out MR.* The 9B's extra capacity improves
   panel (+2.0) and MIMIC (+1.0) and knowledge (+0.85) but not the held-out MR templates, which is
   consistent with the held-out failure being a decision-path property rather than a capacity limit.

## Artifacts and recompute commands

| what | artifact | recompute |
|---|---|---|
| teacher ceiling | `outputs/distill_v0/D1/ceiling.json` | `uv run --frozen python scripts/distill_v0/d1_rederive.py --out outputs/distill_v0/D1` |
| basket + labels | `outputs/distill_v0/D2/{composition,label_manifest,labels}.jsonl` | `uv run --frozen python scripts/distill_v0/d2_rederive.py --out outputs/distill_v0/D2` |
| 4B run | `outputs/distill_v0/D4/{arm_result,dose_summary,audit}.json` | `uv run --frozen python scripts/distill_v0/d4_rederive.py --out outputs/distill_v0/D4 --steps 63304` |
| 9B run | `outputs/distill_v0/D5/{arm_result,dose_summary,audit}.json` | `uv run --frozen python scripts/distill_v0/d4_rederive.py --out outputs/distill_v0/D5 --steps 63304` |
| Gate O7 + scoreboard + frontier | `outputs/distill_v0/D6/gate_o7.json` | `uv run --frozen python scripts/distill_v0/d6_rederive.py --out outputs/distill_v0/D6` |
| predictions | `outputs/distill_v0/D6/student{4b,9b}_<set>/predictions.jsonl` | regenerate with `d6_gate.py` (GPU) |

All numbers in this file have a CLAIMS row (`loops/distill_v0/CLAIMS.md`, C-D1…C-D6).

## Verdict

**Gate O7 FAIL** for both students; the ≤10B frontier placement is achieved and measured. The recipe
transfers decision-family behaviour (MR, MIMIC, pairs) but not general knowledge or the external
panel, and it underperforms the osler_v4 final on the MIMIC suite — the two failed checks. Per the
plan the loop stops here for operator review; the next loop should decide whether the panel/knowledge
gap is a data problem (distil more knowledge/panel items, different weighting) or an architecture
problem (readout, capacity), and whether the MIMIC mix should be rebalanced toward the final's recipe.
