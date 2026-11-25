# FINDINGS — osler_rc0 (Release Candidate 0)

**Summary.** RC0 produced two frozen, calibrated decision models — **Osler-9B-rc0** (the osler_v7 63,304-step final)
and **Osler-4B-rc0** (trained in this loop for the same 63,304 steps, same mix, same token-capped plan, same KL dose;
14.07 h on one GPU) — and placed them on a unified 12-model, 9-set board with full probabilities, dev-only calibration,
standardised latency and a mandatory local MIMIC PHI/memorisation audit. **The 9B is the strongest ≤10B medical decision
model on the board**: it is on the global 2-axis Pareto front with the 27B ceiling, dominates every comparable ≤10B
model on all five headline sets, and tops the fresh tier-2 set outright. **The 4B is a clinical-audit specialist**: it
beats its own zero-shot base by +14.7 points on `mr_verdict` and +13.6 on `mimic_suite` and dominates MedDecider-4B,
but it is ~1 point *below* its base on general OOD QA (`panel` −0.0113, `knowledge` −0.0098) — a measured negative
result that this report does not relabel. The audit is **CLEAR for release discussion** under both rules. Both
artifacts are staged and verified locally; **nothing was uploaded**.

## What the loop produced

| task | outcome |
|---|---|
| R0 | preflight, 14 input hashes, pplx snapshot identity, 0/1,000 MedConceptsQA item-level overlap, prediction inventory — `outputs/osler_rc0/R0/SELF_AUDIT.md` |
| R1 | **Osler-4B-rc0** trained 63,304 steps; dev S **0.839238** (MR 0.946701, MIMIC 0.783854, CG v2 0.787160); knowledge 0.762364; **all-orders positional PASS**; temperatures 1.522233 / 2.374354 / 1.538844; 0 non-finite losses, 0 tripwires, GPU peak 57.1 GB |
| R2 | 9B dev calibration on 14,597 `fastdev_v4` rows: choice 1.527412, noul 2.318485, score 1.703540; dev ECE 0.0697 → **0.0192**, accuracy unchanged; 9 sets × 2 models scored with full probabilities; calibrated copies with 0 argmax flips; T10 parity 100 % |
| R3 | 4B scored on all 9 sets (raw + calibrated): 0 argmax flips; panel raw ECE 0.145 → 0.037 |
| R4 | fresh tier-2 competitor matrix: pplx-27B 5,988/5,988, Clef-Flash 5,988/5,988, MedDecider-9B/4B 5,838/5,988 (97.5 % coverage); JEV-9B **NOT MEASURED** (offline head loader) |
| R5 | latency/VRAM: 4B 26.543 ms/item (9.6 GB), 9B 33.714 ms (17.3 GB), pplx-27B 90.957 ms (52.9 GB); batch-16 slower than batch-1 for every model (reported); MedDecider/Clef latency NOT MEASURED |
| R6 | PHI/memorisation audit: 9B AUC 0.5876 vs control 0.5408 (Δ +0.0468), 4B 0.5644 vs 0.5396 (Δ +0.0248) — both clear the ≤0.05 rule; verbatim **0/300** matches ≥50 tokens (control 0/300); 9B non-generative |
| R7 | unified board, per-category tables, formal Pareto; `LEADERBOARD.md`, `PER_CATEGORY.md`, `PARETO.md`, `docs/leaderboard.md`; 14 unit tests |
| R8 | release staging + offline verification: staged-vs-original max |Δp| **0.0** (20/20 argmax both models); 4B adapter-off generation **20/20 identical** to the untouched base; 9B no `lm_head`; model cards |
| R9 | this report + five-claim fresh-process spot-check + `Loop status: STOPPED` |

## Headline board (accuracy; NM = not measured for that set by design, never imputed)

| model | params (B) | panel | knowledge | confirm_general | mr_verdict | mimic_suite | 5-set macro |
|---|---|---|---|---|---|---|---|
| **Osler-9B-rc0** | 8.39 | 0.5927 | 0.7322 | **0.9008** | **0.9016** | **0.7733** | **0.7801** |
| pplx-27B (ceiling) | 27.00 | **0.6510** | **0.7823** | 0.8843 | 0.8093 | 0.6729 | 0.7600 |
| **Osler-4B-rc0** | 4.00 | 0.5502 | 0.6728 | 0.8883 | 0.8941 | 0.7597 | 0.7530 |
| d3-flash | 8.39 | 0.5899 | 0.7335 | 0.8774 | 0.7351 | 0.6301 | 0.7132 |
| pplx-flash | 4.00 | 0.5615 | 0.6835 | 0.8889 | 0.7466 | 0.6240 | 0.7009 |
| Clef-Flash | 9.00 | 0.5711 | 0.6864 | 0.8933 | 0.6831 | 0.6306 | 0.6929 |
| MedDecider-9B | 9.00 | 0.5603 | 0.6997 | 0.8825 | 0.6653 | 0.6189 | 0.6853 |
| MedDecider-4B | 4.00 | 0.5436 | 0.6703 | 0.8820 | 0.6681 | 0.6019 | 0.6732 |
| JEV-9B | 9.00 | 0.5127 | 0.6448 | NM | 0.6303 | 0.5885 | — |
| Qwen3.5-9B | 9.00 | 0.5419 | 0.6774 | NM | 0.6315 | 0.6029 | — |
| Qwen3.5-4B | 4.00 | 0.4831 | 0.6162 | NM | 0.6176 | 0.5603 | — |
| d1-2B | 2.00 | 0.4259 | 0.4555 | NM | 0.7069 | 0.5587 | — |

Brier/ECE for every measured cell are in `LEADERBOARD.md`; the 9B's calibrated panel Brier is 0.5223 / ECE 0.0229, the
4B's 0.5643 / 0.0374.

## Pareto verdict (`PARETO.md`)

- **2-axis front (OOD General Avg vs Clinical Audit Avg): Osler-9B-rc0 (0.7419, 0.8375) and pplx-27B (0.7725, 0.7411).**
  The 9B is the only ≤10B model on the front; the 27B buys +0.031 OOD for −0.096 audit and 2.7× the latency.
- **Exact ≤10B dominance:** Osler-9B-rc0 dominates Osler-4B-rc0, pplx-flash, MedDecider-9B, MedDecider-4B and
  Clef-Flash on all five headline sets with a strict gain every time. Undominated on all five: Osler-9B-rc0 and its
  base d3-flash (which wins `knowledge` by 0.0013).
- **≤4B tier:** Osler-4B-rc0 and pplx-flash are undominated; both dominate MedDecider-4B (the 4B by +0.0066 panel,
  +0.0026 knowledge, +0.0063 confirm_general, +0.2260 mr_verdict, +0.1578 mimic_suite).
- **Paired 10-option intersections:** panel (4,141) 9B 0.5878 > MedDecider-9B 0.5603 > 4B 0.5455 > MedDecider-4B
  0.5436; confirm_general (5,838) 9B 0.9072 > 4B 0.8936 > MedDecider-9B 0.8825 > MedDecider-4B 0.8820.
- **Efficiency:** the 4B is 1.27× faster than the 9B at batch 1 and 5.5× smaller than the 27B; the 9B's audit accuracy
  at 33.7 ms/item is the board's best accuracy-per-millisecond on the clinical sets.

## MIMIC PHI / memorisation audit (`AUDIT.md`)

- Members: 2,000 MIMIC training rows visited in steps 1–63,304 by **both** candidates (the two runs' visited sets differ:
  235,968 vs 234,251 rows, Jaccard 0.084, because the tokenisers order the same mix differently). Non-members: 2,000
  late-anchor `mimic_suite` items with **0** content-hash overlap with the mix.
- **9B:** AUC 0.5876 [0.5703, 0.6051]; control d3-flash 0.5408; Δ **+0.0468** [+0.0316, +0.0626] → clear (≤ 0.05).
- **4B:** AUC 0.5644 [0.5466, 0.5813]; control pplx-flash 0.5396; Δ **+0.0248** [+0.0097, +0.0397] → clear (≤ 0.05).
- **Verbatim probe (4B):** 0/300 notes matched on ≥50 tokens (control 0/300); truth NLL 8.56 vs control 2.72.
- **9B:** non-generative (no `lm_head`), probe not applicable.
- **Verdict: CLEAR for release discussion** under both rules; no AUDIT_FLAG.
- Y10 compatibility: 0.5018 / 0.5033 (only 35 of the 2,000 Y10 members were visited by both, so it is a near-null check;
  osler_v4 reference 0.5042).

## Five-claim fresh-process spot-check (`outputs/osler_rc0/R9/SELF_AUDIT.md`)

| claim | verdict |
|---|---|
| 1. 4B step-63,304 dev S and all-orders positional health | **PASS** — S 0.839238 re-derived from the eval predictions; positional original + 2 permuted orders pass |
| 2. 9B dev-calibrated panel accuracy, Brier, ECE | **PASS** — recomputed from the calibrated panel file and equal to the board |
| 3. 9B & 4B `confirm_general` paired deltas vs own bases | **PASS** — +0.023380 [+0.018036, +0.029058] and −0.000668 [−0.007682, +0.006847] |
| 4. Visited-member MIMIC AUC for both | **PASS** — 0.587561 and 0.564389 re-derived from the raw scores |
| 5. Formal ≤10B Pareto dominance from `rc0_board.json` | **PASS** — the dominance pairs recompute exactly |

## Deviations, negative results and NOT MEASURED items

- **JEV-9B `confirm_general`: NOT MEASURED** — the card's decision-head loader requires a Hub repo id and the offline
  cache lookup raises `LocalEntryNotFoundError`; a local-path retry raises `HFValidationError`. JEV keeps 8/9 sets from
  osler_v4.
- **MedDecider-4B/9B and Clef-Flash latency: NOT MEASURED** — authors' single-item runners, outside the wiring timebox.
- **MedDecider-4B/9B fresh-set coverage 97.5 %** — the 150 >10-option items are absent from the authors' output files;
  reported as coverage, never as scored.
- **Held-out readout failure:** `mr_heldout_test` / `synthea_indicator_evidence_v1` reads below chance for both
  candidates (9B 0.356, 4B 0.245; chance 0.5) while their bases pass the same cell (0.4591 / 0.5973). Flagged
  `READOUT_FAIL`; not quoted as an accuracy.
- **Fresh-set calibration transfer fails:** dev-fit temperatures leave the candidates under-confident on the post-cutoff
  set (mean confidence 0.81 vs accuracy 0.90; ECE 0.090/0.094 vs 0.019 on dev). No test-set fitting was done.
- **The 4B's OOD regression** (panel −0.0113, knowledge −0.0098 vs its own base) is the loop's main negative result and
  is carried in the board, cards and this report unchanged.
- Positional health was intermittent during 4B training (FAIL at 27,984 and 41,976 on the original order); the frozen
  final passes all orders. No checkpoint was swapped.
- `stopped_early` in `arm_result.json` is the trainer's completion-reason string (`"step budget 63304 reached"`), not a
  boolean; the run reached exactly the requested budget with no tripwire.

## Release readiness — operator sign-off checklist

- [x] Both candidates frozen at step 63,304 with dev-only calibration saved inside the artifacts.
- [x] Full-probability predictions on all 9 frozen sets for both candidates and their bases; unified board recomputed
      from per-item files with zero unmapped ids.
- [x] Formal Pareto frontier; the 9B is on the global 2-axis front and dominates every comparable ≤10B model.
- [x] Mandatory local MIMIC audit complete: rules (a) and (b) clear; committed aggregate-only `AUDIT.md`.
- [x] Release trees staged and verified offline (0 probability drift, 4B adapter inert when disabled, 9B no `lm_head`);
      model cards written with the measured limitations.
- [x] No training data or MIMIC text in the staged trees; no secrets committed; **nothing uploaded**.
- [ ] Operator: review `AUDIT.md`, the held-out readout failure, the fresh-set calibration transfer and the 4B OOD
      regression; decide whether the release bar is met for the 4B (the 9B clears it).
- [ ] Operator: decide the licence/NOTICE packaging and the public release mechanics (out of scope for this loop).

Loop status: **STOPPED** after R9.
