# osler_v3 — FINDINGS

Loop: osler_v3 (keep the base's general skill with KL-to-base; real sources and Gemma-crafted real-text items). Branch
`loop/osler_v3`. Run closed 2026-10-20 by the operator's reduced scope (X9 skipped; X10 reduced to additional rows; no Gate O3
verdict). Every number below has a CLAIMS row in `loops/osler_v3/CLAIMS.md`.

## Summary

1. **No Gate O3 verdict was computed.** X9 (the final Osler-4B) was skipped by the operator: the kept mix equals osler_v2's C1 with
   osler_v2's recipe. osler_v2's FAIL (external panel vs zero-shot pplx, −0.0198 [−0.0344, −0.0060]) stands. (C129, C130;
   `loops/osler_v3/gate_o3.md`.)
2. **Stage A chose the control (A0; KL-to-base 0)** by the pre-registered rule on the clinical-general proxy. The margins are weak:
   CG accuracy 0.7805 for A0 against 0.7772–0.7788 for the three variants; A0 is +0.0027 over zero-shot pplx on the CG items. No
   variant beat the control on the dev proxy (C098–C101; `loops/osler_v3/stage_a.md`).
3. **Stage B: no block was kept.** RAGTruth and DrugProt (B1), restyle pages (B2) and indicator charts (B3) each failed the MR check
   of the keep rule (MR differences from B0: −0.0044, −0.0096, −0.0054; lower bounds −0.0143, −0.0209, −0.0139). B3 also failed
   the general check. The last kept block is B0, osler_v2's C1 mix (C111–C118; `loops/osler_v3/blocks.md`).
4. **Trade-off on the external panel (additional rows, calibrated).** Against zero-shot pplx the panel difference is −0.0245 for A0,
   −0.0111 for A1 (KL 0.1), −0.0161 for A2 (KL 0.5), −0.0109 for A3 (broad replay), and −0.0226 for B1. The MR verdict difference
   against pplx is about +0.134 to +0.140 in every run (C122–C126). The KL-to-base term and the broader replay narrow the panel gap
   by roughly half, but their intervals include zero, and KL 0.5 does not narrow it further (C122–C127).
5. **The Stage A runs are about one third the length of osler_v2's final.** osler_v2's final ran 42,200 steps; every Stage A run and B1
   ran 13,992 steps (C131). Against osler_v2's Osler-4B, every run is 0.018–0.025 lower on the MR verdict, the control included
   (C122–C126). The control has the same mix and recipe as osler_v2's final, so the gap is not explained by the KL term. Run length
   is the likely cause, but this loop did not test it (hypothesis; R10).

## Which ADVISORY §2 outcomes apply

- **(1) Gate O3 passes:** not applicable (no final model; no verdict).
- **(2) KL closes the panel gap on test but costs MR:** only partly. The panel gap narrows with KL 0.1 and with broad replay, to
  about −0.011 against pplx (intervals reach zero), but none of the KL arms costs MR against the control: the MR difference from
  A0 is −0.0008 for A1 and +0.0008 for A3 (C122–C125). The trade-off curve is in §3 below.
- **(3) No Stage A variant beats the control on the dev proxy:** applies. The ADVISORY asks for the panel per dataset; that is in §4.
- **(4) A Stage B source is not kept:** applies to all three blocks (B1, B2, B3). They are dropped, not tuned (C111–C116).

## 1. Stage A and the choice

Rule and results: `loops/osler_v3/stage_a.md` (C098–C101). All four arms qualified; A0 has the highest CG difference from zero-shot
pplx (+0.0027). The choice is mechanical and its margin is weak (C099).

## 2. Stage B: the keep rule and why no block passed

The keep rule (margin 0 on the MR partition, −0.01 on general, −0.02 on knowledge; unchanged) decided all three blocks against B0
(C111–C116). Two findings:
- The new-source gains are real on their own templates and outside the verdict: beside the rule (labelled, not in the verdict), the
  1,980 new-source items gain +0.151 for B1, +0.015 for B2 and +0.026 for B3 against B0 (C113, C115, C117).
- The rule has no path for a block whose gain lies outside the MR partition. The MR check requires its lower bound to exceed 0 on
  3,597 items; a block with no MR effect has a point near zero and an interval about 0.02 wide, so it fails whatever its own gain.
  All three blocks showed this: MR points −0.004 to −0.010. This is an observation about the rule, not a change to it.

## 3. The Stage A trade-off (additional rows)

| run | KL to base (CG) | CG accuracy | panel vs pplx | MR vs pplx | MR vs osler_v2 |
|---|---|---|---|---|---|
| A0 (control) | 0.1709 | 0.7805 | −0.0245 [−0.0379, −0.0114] | +0.1353 | −0.0224 |
| A1 (KL 0.1) | 0.1410 | 0.7780 | −0.0111 [−0.0246, +0.0008] | +0.1345 | −0.0232 |
| A2 (KL 0.5) | 0.1102 | 0.7788 | −0.0161 [−0.0274, −0.0043] | +0.1402 | −0.0175 |
| A3 (broad replay) | 0.1709 | 0.7772 | −0.0109 [−0.0235, +0.0025] | +0.1361 | −0.0216 |
| B1 (X1 real sources) | 0.1586 | 0.7834 | −0.0226 [−0.0360, −0.0090] | +0.1329 | −0.0248 |

KL-to-base lowers the drift on CG (0.1709 → 0.1410 → 0.1102; C128) and moves the panel gap by about 0.01 in its favour, but the
panel is not monotone in KL (A2 is worse than A1). These are single runs at 13,992 steps; the panel intervals are about ±0.013 wide.

## 4. External panel by dataset (C127; `loops/osler_v3/results.md` §3)

Against zero-shot pplx, the largest losses are on `medical_question_pairs` (295 items: −0.05 to −0.11 for every run) and, for three
of the five runs, on `symptom_to_diagnosis` (212 items: A0 −0.052, A2 −0.061, B1 −0.080; A3 +0.033). `medxpertqa_text` (1,000) is
negative for all five runs (−0.011 to −0.027). `medexqa_test` (940) is near zero to slightly positive. These per-dataset sets are small,
and the per-dataset intervals are wide; the per-dataset rows are additional and carry no verdict.

## 5. Known limits

- A single seed per run; no repeat runs (each run is one training with seed as configured in its arm_result; no seed sweep here).
- The Stage A runs are shorter than osler_v2's final (C131), so their results do not measure the final recipe's drift.
- Brier and per-dataset Brier are in `outputs/osler_v3/X10/results.json`, not summarised here.
- The 21 knowledge items that zero-shot pplx cannot score are excluded from the pplx paired knowledge rows (C120).
- MIMIC was not available (X5 BLOCKED); no memorisation audit (X11 not applicable).
- Blocked: the claims template (clinclaim_grounding_v1) failed the D44(b) check after the second prompt fix (C107); the restyle
  template was unblocked by the operator (D-X3-16) and is in B2.

## 6. Five-claim spot-check (re-run in a fresh process)

Output: `outputs/osler_v3/X10/spotcheck.log` (gitignored; the outputs are reproduced here).

- C120 (pplx micro accuracy): `{'mr_verdict': (0.755, 13103, 13103), 'external_panel': (0.5608, 4353, 4353), 'knowledge': (0.6837, 5435, 5456)}`. Match.
- C121 (osler_v2 micro accuracy): `{'mr_verdict': (0.8981, 13103, 13103), 'external_panel': (0.5419, 4353, 4353), 'knowledge': (0.6849, 5456, 5456)}`. Match.
- C124 (A2 against both baselines): pplx MR +0.1402 [+0.1273, +0.1532], panel −0.0161 [−0.0274, −0.0043], knowledge +0.0029; osler_v2 MR −0.0175, panel +0.0037, knowledge +0.0007. Match.
- C128 (KL to base): A0 0.1709, A1 0.141, A2 0.1102, A3 0.1709; B1 0.1586. Match.
- C130 (cross-check with osler_v2's W8 gate): −0.0198 [−0.0344, −0.006], n 4353. Matches C129, the value in this report. Match.

Denominators and rows: 30,621 rows per run (C119); all five records report a calibration match.

## 7. Where the numbers come from

- Scoring: `scripts/osler_v3/x10_score.py` (osler_v2's W8 path, output root `outputs/osler_v3/X10`), run by `scripts/osler_v3/x10_chain.sh`.
- Report: `scripts/osler_v3/x10_report.py` → `outputs/osler_v3/X10/results.json`, `loops/osler_v3/results.md`.
- Keep decisions: `scripts/osler_v3/x8_keep.py` (outputs `outputs/osler_v3/X8/keep_B*.json`).
