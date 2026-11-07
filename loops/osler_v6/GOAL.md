# GOAL — osler_v6 (distillation with a dosing fix)

Follow `loops/osler_v6/ADVISORY.md` (the pre-registered design, §2) exactly. The loop distils the pplx-decider-v1.2-27b teacher
into the Osler-4B on general anchor items, with the D57 dosing rule, and checks it against the pre-registered choice rule and
Gate O6. Nothing in ADVISORY §2 may change after a run starts.

## Read first, in this order
1. `AGENTS.md` (Current loop section, R1–R10, guardrails).
2. `docs/plans/PROGRAM.md`, D48, D57, D58 (and D59–D61 for context).
3. `loops/osler_v6/ADVISORY.md` (the plan; §6 is open and must not be used for any run).
4. `loops/osler_v6/STATE.md` (what is done; the task board).
5. `loops/osler_v5/FINDINGS.md` (closure addendum) and `loops/osler_v4/FINDINGS.md`.

## Each iteration
1. Read STATE.md. Take the first PENDING task whose dependencies are DONE.
2. Mark it IN_PROGRESS with `date -u +%FT%TZ`; copy its steps into the STATE checklist.
3. Execute, run its acceptance check, self-audit (R6: SELF_AUDIT.md in `outputs/osler_v6/<task>/`).
4. Append CLAIMS.md rows (every number has one; NOT MEASURED and BLOCKED are valid).
5. Mark DONE or BLOCKED — reason; commit `loop(osler_v6): Z<n> <DONE|BLOCKED> — <headline>`; push `loop/osler_v6`.

GPU jobs: one at a time, detached, logs under `outputs/osler_v6/<task>/logs/`, PID in STATE, polled every 20 minutes or more.
Test disk headroom (2 GB write) before each job.

## Do not
- Write to any earlier `outputs/osler_v4/`, `outputs/osler_v5/` or `loops/*/` (except this loop's own files). Reuse osler_v4's
  Y5 results by copying their ids and hashes, not by appending to them.
- Send MIMIC text to any non-local service, or commit it, or commit items, predictions, checkpoints or keys.
- Train on DDXPlus, any test split, the panel or its copies, MDACE, knowledge dev v2 items or held-out items.
- Use the 27B's soft labels on any dev or test set.
- Change D57, D58, the frozen sets, the choice rule or Gate O6 after a run, or to make something pass (that is BLOCKED).
- Change the head, the attention, the base model or the adapter (D26, D33).
- Edit AGENTS.md beyond the Current loop section at Z0.
- Start the next loop, or anything in ADVISORY §6.

## Done when
Every task is DONE or BLOCKED with a reason; no task is IN_PROGRESS; `FINDINGS.md` exists with a Summary first and a five-claim
spot-check; `Loop status: STOPPED` at the top of STATE.md; committed and pushed. Then stop and report: tasks done, tasks blocked and
why, the Gate O6 verdict, and what comes next.
