# osler_v3 — Stage A: the recipe choice (ADVISORY X7; aggregates only)

Written 2026-10-19T17:46:43Z by `scripts/osler_v3/x7_choice.py`; the rule is in its docstring and was fixed before any Stage A run.

**Chosen recipe: A0** (qualifying: A0, A1, A2, A3; tie rule applied: False).

| arm | KL to base (CG) | MR macro vs A0 (point, 95 % CI) | qualifies | CG acc | CG - zero-shot pplx | knowledge acc | general acc |
|---|---|---|---|---|---|---|---|
| A0 | 0.1709 | control | control | 0.7808 | 0.0027 | 0.7744 | 0.7813 |
| A1 | 0.1410 | +0.0008 [-0.0073, +0.0085] | True | 0.7783 | 0.0002 | 0.7713 | 0.7797 |
| A2 | 0.1102 | +0.0000 [-0.0076, +0.0082] | True | 0.7791 | 0.001 | 0.7724 | 0.7825 |
| A3 | 0.1709 | -0.0005 [-0.0094, +0.0082] | True | 0.7775 | -0.0006 | 0.7713 | 0.7867 |
