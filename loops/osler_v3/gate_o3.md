# osler_v3 — Gate O3: NOT APPLIED (operator ruling, 2026-10-20)

No Gate O3 verdict was computed for this loop, and none is written here.

Why: X9 (the final Osler-4B) was skipped by the operator, because the kept mix equals osler_v2's C1 with osler_v2's recipe, so a
final run would retrain osler_v2's model. With no new final model, D35, D39 and D44 have nothing to apply to. osler_v2's Gate O3
verdict (FAIL on the external panel against zero-shot pplx, −0.0198 [−0.0344, −0.0060]) stands unchanged in
`loops/osler_v2/`.

What was measured instead (additional rows, calibrated; not a verdict): `loops/osler_v3/results.md` (from
`outputs/osler_v3/X10/results.json`): the four Stage A runs and block B1 against zero-shot pplx-decider-flash-v1.2 and against
osler_v2's Osler-4B, on the MR suite v1 verdict templates, the external panel (overall and per dataset) and the knowledge items.
CLAIMS C119–C131.

The 9B stays NOT MEASURED (deferred, D39).
