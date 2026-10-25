# osler_v4 — the memorisation and PHI audit (Y10; counts and figures only)

Generated 2026-10-25T12:26:03Z by `scripts/osler_v4/y10_audit.py` from `outputs/osler_v4/Y10/` (counts only; no item or note text). This is a report for the release decision, not a gate (ADVISORY Y10).

## 1. Membership inference

Members: 2000 training rows of the final mix; non-members: 2000 items of the frozen MIMIC audit suite v1 (late-anchor test items; no training overlap). Both sides are allocated per template in proportion to the suite's counts. Score: the negative per-item loss; 0.5 means no measurable membership signal.

| model | AUC | interval (95 %, bootstrap, 1,000 resamples, seed 0) | scored members | scored non-members |
|---|---|---|---|---|
| Osler-4B (final) | 0.5042 | [0.4853; 0.5221] | 2000 | 2000 |
| zero-shot pplx-decider-flash-v1.2 (control) | 0.5090 | [0.4926; 0.5274] | 2000 | 2000 |

The test items are later admissions than most training items, so the control shows how much of any gap is train-to-test shift alone. The AUC is a measurement, not a verdict.

## 2. Verbatim probe

300 train-split discharge notes; a 128-token prefix, the next 64 tokens as the truth, greedy generation of 64 tokens; a match is at least 50 consecutive aligned tokens.

| model | matches | runs of at least 50 | at least 25 | at least 10 | longest run (median) |
|---|---|---|---|---|---|
| Osler-4B (final, exported, adapter merged) | 0 | 0 | 0 | 0 | 0 |
| zero-shot base (control) | 0 | 0 | 0 | 0 | 1 |

Additional, not a gate: the causal export of the final model against its decision path, at the answer position, on 20 membership items (60 offered options). The argmax agrees on 16 items; the largest absolute probability difference is 0.7681. The decision path uses non-causal full attention, so the probe measures the causal LM that ADVISORY Y10 names, not the decision path.

Additional, labelled sanity figure: the mean negative log-likelihood of the true continuation, in nats per token (a random model is near the natural log of the vocabulary size): final 7.673 (median of the notes 7.641); zero-shot base 2.725 (median 2.721).

