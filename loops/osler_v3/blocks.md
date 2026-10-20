# osler_v3 — Stage B: data blocks (ADVISORY X8; aggregates only)

Built by `scripts/osler_v3/x8_build_block.py`; kept or not by `scripts/osler_v3/x8_keep.py` (the rule is `meddecide.train.blocks`,
unchanged). B0 is the chosen Stage A checkpoint (A0, osler_v2's C1 mix). Every block is B0's mix plus one new source, with the
replay top-up rule (at least 20 % replay; no template above 8 %). Operator ruling 2026-10-20: X8 is DONE; the last kept block is B0.

| block | added to osler_v2's C1 | rows | replay share | largest template share | keep rule (MR / general / knowledge) | decision |
|---|---|---|---|---|---|---|
| B0 | nothing (A0, the control) | 830,076 (C1) | 21.0 % (C1) | — | — (the current block) | last kept |
| B1 | X1 real sources (RAGTruth, DrugProt; 69,464 rows) + top-up | 906,164 | 20.0 % (181,233) | 6.81 % | −0.0044 [−0.0143, +0.0058] / −0.0016 [−0.0094, +0.0071] / +0.0056 [−0.0018, +0.0131] | not kept (MR lower bound below 0) |
| B2 | X3 restyle train pages (5,797 rows; claims excluded, D-X8-2) | 835,873 | 20.89 % | 7.38 % | −0.0096 [−0.0209, +0.0019] / −0.0003 [−0.0088, +0.0075] / −0.0060 [−0.0142, +0.0018] | not kept (MR lower bound below 0) |
| B3 | X4 indicator charts (2,916 train rows) | 832,992 | 20.96 % | 7.40 % | −0.0054 [−0.0139, +0.0032] / −0.0049 [−0.0139, +0.0043] / −0.0049 [−0.0136, +0.0038] | not kept (general lower bound below −0.01; MR lower bound below 0) |

Claims: C111–C118 (`loops/osler_v3/CLAIMS.md`). Each row's numbers are the keep-rule statistics on 3,597 MR items, 7,726 general
items and 4,500 knowledge items against B0, bootstrapped (1,000 resamples, seed 0).

**Beside the rule (labelled, D-X8-1 and D-X8-4; not in the verdict):** on the 1,980 new-source dev items (six templates:
RAGTruth, DrugProt, chart indicators, restyle), B1 +0.1510 [+0.1229, +0.1779], B2 +0.0152 [+0.0019, +0.0302], B3 +0.0256
[+0.0103, +0.0420]. On the 4,853 clinical-general items, zero-shot pplx scores 0.7781; B0 0.7808 (+0.0027), B1 0.7836 (+0.0056),
B2 0.7733 (−0.0047), B3 0.7748 (−0.0033).

**What this means for the keep rule.** The rule requires the MR lower bound to exceed a margin of 0. The new-source gains land on
their own templates, outside the MR partition, so the MR comparison sees only noise around zero, and a block whose gain lies outside
the MR partition cannot pass that check unless its MR interval excludes zero. All three blocks failed the MR check with points between
−0.004 and −0.010. This is recorded as a finding (FINDINGS.md), not a change to the rule (D24/D31/D35 and the margins are unchanged).

**Process record.** The chain crashed in the keep step of all three blocks (missing B0 CG predictions in the labelled CG side report),
so the chain treated each block as not kept. The advisor re-ran the keep decisions after the fix (commit 7f47e1e); the decisions above
are from those re-runs. B1 attempt 1 was stopped at step 10,450 and relaunched from B1 (commit 2a75ae1); the reported B1 is the
complete run (13,992 steps).
