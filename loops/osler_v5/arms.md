# osler_v5 — Z2 arms and the choice

Status: **Z2 DONE** (2026-10-28). Choice: **none**. No arm qualifies under the pre-registered D56c rule (ADVISORY §8.4), so outcome (4)
applies (ADVISORY §4 Z2: choose nothing and report). Z3 is not started. Claims: C041–C077 in `CLAIMS.md`. The audit is
`outputs/osler_v5/Z2/SELF_AUDIT.md` (raw outputs are gitignored). Run provenance: `outputs/osler_v5/Z2/<arm>/run_provenance.json` and
`arm_result.json` (code at run time: F0 34c96c8, F2 f472a9a, F2s1 0c13fb9; C077).

The corrected mix (ADVISORY §8.2, D56a) drops the 155 `fda_route_choice_v1` rows. Its composition hash is
`78d7e5478060fc828784cb3365569e177c6961270d38a76a9cff144684e9f3be` (C059; STATE §2). All corrected arms train on it.

## 1. The rule applied (D56c; `meddecide.eval.d56c_choice`, `scripts/osler_v5/z2_choose_d56c.py`)

An arm **qualifies** if it passes positional health in all three orders at 13,992 steps, its knowledge-dev lower bound against F0 is
above −0.01, and its MR and MIMIC lower bounds against F0 are above −0.03. Paired bootstrap, 1,000 resamples, seed 0, 95 % intervals.
The control's own bounds are zero. The choice is the qualifier with the highest CG v2 micro. A qualifier within 0.010 of that value that
is the fuller fix wins the tie. Two arms of the same fix level in one tie are refused by the code, because the rule does not order them.

## 2. The corrected arms (the candidates)

| arm | recipe | step | S | MR macro | MIMIC macro | CG v2 micro | knowledge dev | positional, all three orders | claim |
|---|---|---|---|---|---|---|---|---|---|
| F0 | osler_v4 recipe (per-step weighting, no augmentation), seed 0 | 6,996 | 0.8082 | 0.9239 | 0.7361 | 0.7647 | 0.7480 | pass (original 0.4269 against 0.7405) | C064 |
| F0 | | 13,992 | 0.8296 | 0.9325 | 0.7822 | 0.7740 | 0.76575 (3,063 of 4,000) | **fails** in the original order (0.3647 against 0.7924; limit 0.3962); both permutations pass | C065 |
| F2 | per-item weighting and option augmentation, seed 0 | 6,996 | 0.8064 | 0.9233 | 0.7242 | 0.7718 | 0.7500 | pass (original 0.5311 against 0.7804) | C060 |
| F2 | | 13,992 | 0.8259 | 0.9375 | 0.7558 | 0.7844 | 0.7595 (3,038 of 4,000) | pass (original 0.5391 against 0.8603) | C062 |
| F2s1 | F2 at seed 1 | 6,996 | 0.8021 | 0.9226 | 0.7157 | 0.7680 | 0.7392 | pass (original 0.6453 against 0.7525) | C067 |
| F2s1 | | 13,992 | 0.8243 | 0.9442 | 0.7496 | 0.7791 | 0.76325 (3,053 of 4,000) | pass (original 0.5271 against 0.7585) | C068 |

Paired against F0 at 13,992 (C066, C069):

| arm | knowledge micro [interval] | MR macro [interval] | MIMIC macro [interval] |
|---|---|---|---|
| F2 | −0.0063 [−0.0150, +0.0030] | +0.0050 [−0.0029, +0.0134] | −0.0264 [−0.0343, −0.0186] |
| F2s1 | −0.0025 [−0.0115, +0.0058] | +0.0117 [+0.0019, +0.0215] | −0.0327 [−0.0420, −0.0239] |

Qualification (C070, from `outputs/osler_v5/Z2/choice_d56c.json`):

| arm | positional, all orders | knowledge lower bound (> −0.01) | MR lower bound (> −0.03) | MIMIC lower bound (> −0.03) | qualifies |
|---|---|---|---|---|---|
| F0 | fails (original order) | control | control | control | no |
| F2 | passes | −0.0150 **fails** | −0.0029 passes | −0.0343 **fails** | no |
| F2s1 | passes | −0.0115 **fails** | +0.0019 passes | −0.0420 **fails** | no |

F2 has the highest CG v2 micro (0.7844), but it does not qualify, so the CG-first rule selects nothing. **Choice: none (outcome (4)).**

## 3. Additional rows (labelled; none is a candidate and none is used in a choice)

**Zero-shot knowledge reference (C075).** The Z1 blend at α = 0 (the frozen base with its own readout) scores 0.7725 (3,090 of 4,000)
on the same items. Against it, at 13,992: F0 −0.00675 [−0.01425, +0.00050]; F2 −0.0130 [−0.02125, −0.00525]; F2s1 −0.00925
[−0.01825, −0.00050]. The osler_v4 final's α = 1 drop was −0.0140 (C036). This is the "knowledge below zero-shot" trigger; it is
confirmed as measured (SELF_AUDIT §3).

**Training and answer checks (C073, C074).** No arm has non-finite losses or gradients or exact-zero losses in 279 logged steps.
Median gradient norms are 13.03 (F0), 11.67 (F2) and 9.64 (F2s1). One spike, F2s1 at step 2,900 (pre-clip norm 439.0, loss 2.52), was
back to loss 0.0019 by step 3,000. At 13,992 no arm puts one key on a majority of DDXPlus items (modal shares 0.068 to 0.130). The
two-letter-gold items predicted as BM are 69 (F0), 64 (F2) and 104 (F2s1) of 499.

**The old mix (ADVISORY §8.1; defective, not candidates).**

| arm | step | S | CG v2 micro | knowledge | positional | claim |
|---|---|---|---|---|---|---|
| R0 (osler_v4 recipe, seed 0) | 6,996 | 0.8095 | 0.7648 | 0.7592 | fails (original 0.2946 against 0.7565) | C041 |
| R0 | 13,992 | 0.8288 | 0.7762 | 0.7558 | passes in all orders | C043, C044 |
| R0 prime (seed 1) | 6,996 | 0.7896 | 0.7127 | 0.7600 | fails in all orders (two-letter 0.0000, 0.0163, 0.0106) | C050, C072 |
| R0 prime | 13,992 | 0.8159 | 0.7346 | 0.7680 | fails in all orders (two-letter 0.0000, 0.0041, 0.0021) | C052, C053 |
| R2 (per-item and augmentation, seed 0) | 6,996 | 0.8053 | 0.7737 | 0.7502 | passes in all orders | C054, C055 |
| R2 | 13,992 | 0.8311 | 0.7986 | 0.7692 | passes in all orders | C056 |
| R3 | stopped at step 5,850 of 13,992 | — | — | — | no evaluation | C071 |

R2 minus R0 at 13,992: knowledge +0.0135 [+0.0045, +0.0225], MR −0.0008, MIMIC −0.0146 [−0.0234, −0.0063] (C057). R0 prime minus R0
is knowledge +0.0123 (C058). The seed repeat of R0 fails positional health at 13,992 (C052).

**Readout scale (D56d, additional; C061, C063).** Relative to pplx's base readout, F2's readout rows 26 to 254 have mean distance 0.330 at
13,992. The osler_v4 final's distance is 0.0057. The paired difference is +0.3244 (38 of 229 rows larger). No choice uses this.

## 4. What the arms show

- Nothing qualifies. Both per-item arms fail the knowledge and MIMIC bounds against F0 (C066, C069). F0 fails positional health in the
  original order at 13,992 and passes it at 6,996 (C064, C065).
- The corrected mix does not clearly fix the positional check at seed 0. The old-mix R0 passes at 13,992 (two-letter 0.4469 against
  0.3642; C044), and the corrected F0 fails (0.3647 against 0.3962; C065). The seed-1 old-mix repeat collapses to 0.0000 (C052). The
  seed spread at this step is large, so no mix effect on the positional check is isolated. ADVISORY §8.8 asks for exactly this question,
  and the answer is that it is not settled here.
- The per-item arms pass the positional check in every order at 13,992, which the corrected F0 does not.
- Hypothesis, not tested: the MIMIC macro falls under per-item weighting because that weighting lowers MIMIC's share of the applied
  update (C010 to C011). The bounds fail for that reason, if it is the reason.
- Hypothesis, not tested: the augmentation moves the two-letter readout rows (C063). No choice uses that.

## 5. Not started or blocked

- **Z3** (final Osler-4B, 63,304 steps on the chosen recipe): BLOCKED. No arm qualifies under D56c (C070), so there is no recipe to train.
  Start time: none.
- **Z4** (blend α on dev and export), **Z5** (Gate O5 and scoreboard), **Z6** (membership audit): BLOCKED. Each needs the Z3 final.
- The reading of ADVISORY §4 Z2's outcome (4) is recorded in STATE §7.12.
