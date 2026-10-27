# GOAL — osler_v5 (fix and max the Osler-4B)

osler_v4's Osler-4B leads every open model we measured on real-record audit (MIMIC suite 0.800 vs pplx-27B 0.682, zero-shot
pplx-flash 0.634) and fails Gate O4 on the external panel against its own base (−0.042, limit −0.02). The advisor traced the
general-skill loss to three causes visible on dev: the readout collapses on options past F (DDXPlus two-letter gold 2 of 499),
per-step loss averaging gave MIMIC about 60 % of the gradient for 36 % of the items, and nothing pulled the weights back toward
the base.

This loop fixes those and nothing else:
- **Per-item loss weighting** (D51) and **option-count augmentation with a positional health check** (D52);
- **four short recipe arms** on osler_v4's exact mix, chosen by a **dev composite** of MR, MIMIC and CG (D54);
- a **63,304-step final** (the same length as osler_v4's, for a one-to-one comparison);
- a **weight-space blend toward the base** chosen on dev (D53);
- **Gate O5** (D55: Gate O4 unchanged; the MedDecider-4B panel comparison stays NOT MEASURED), a membership audit, and a hard stop.

If the gate passes, this recipe goes to Osler-9B next loop unchanged.

## Read first, in this order

1. `AGENTS.md` (repo root): project rules, R1–R10, loop mechanics, guardrails.
2. `docs/plans/PROGRAM.md`, D49–D55.
3. `loops/osler_v4/FINDINGS.md` and `loops/osler_v4/NEXT.md`.
4. `loops/osler_v5/ADVISORY.md`: this loop's plan. Read it in full at the start of every iteration.
5. `loops/osler_v5/STATE.md`: what has been done. If it still contains `<fill in bootstrap>`, start with Z0.

Do not improvise the plan. Where STATE.md disagrees with your recollection, the file wins.

## Each iteration

1. Read STATE.md.
2. Take the **first** `PENDING` task whose dependencies are `DONE` (GPU order in ADVISORY §5).
3. Mark it `IN_PROGRESS` with a timestamp from `date -u +%FT%TZ` and copy its steps into the STATE checklist.
4. Execute it, run its acceptance check, and self-audit (R6).
5. Append to `loops/osler_v5/CLAIMS.md`.
6. Mark it `DONE` or `BLOCKED — <reason>` and log it.
7. Commit (`loop(osler_v5): Z<n> <DONE|BLOCKED> — <headline>`) and push `loop/osler_v5`. Run the suite only when the commit
   changes code (ADVISORY §3.6).

**Long GPU jobs:** detached, logs under `outputs/osler_v5/<task>/logs/`, PID in STATE, polled at 20 minutes or more, one at a
time, outputs checked for completeness and not just exit codes. Check disk headroom (a 2 GB test write) before each.

## The contract

Every number carries its artifact and a recompute command (CLAIMS.md). `NOT MEASURED — <reason>` and `BLOCKED — <reason>` are
valid results; a fabricated number is a critical failure.

Gate O5's baselines are exactly MedDecider-4B and zero-shot pplx-decider-flash-v1.2. The osler_v4 final, osler_v2, every arm,
the unblended checkpoint and every other model are additional rows.

Held-out items, test sets, the MIMIC audit suite and the external panel never influence any choice. If a result is unwelcome,
report it and name the ADVISORY §2 outcome it implies.

## Do not

- Send MIMIC text, values or identifiers to any non-local service, or commit them (or items, predictions, checkpoints, keys).
- Use unofficial copies of credentialed data (D37), CPT descriptors or APR-DRG descriptions.
- Train on DDXPlus, any test split, the panel or its copies (e.g. `MaziyarPanahi/MedExQA-clean`), MDACE, knowledge dev v2, CG
  items or held-out items.
- Add a data source or change the mix, except the D56 correction (drop `fda_route_choice_v1`; the corrected hash is recorded in STATE §2).
- Change D51–D55, the frozen sets, the choice rules, the blend rule or the closed list after any run, or to make something
  pass (that is `BLOCKED`).
- Change the head, the attention, the base model or the adapter (D26, D33).
- Re-score a baseline that osler_v4 already scored on the same frozen set.
- Run two GPU jobs at once, or spawn sub-agents for GPU work.
- Edit earlier `loops/*/`, earlier `outputs/*/`, `data/bench/*` frozen sets, `AGENTS.md` (except Z0's "Started" field) or
  `docs/plans/`.
- Delete outside `outputs/osler_v5/` and `/workspace/tmp`; touch `/workspace/download-manager/`; stop or reconfigure pods,
  volumes, HF repos or remote services; upload anything.
- Start the next loop, or anything in ADVISORY §6.

## Done when

- Every task is `DONE` or `BLOCKED — <reason>`; none is `IN_PROGRESS`.
- `probes.md`, `arms.md`, `blend.md`, `gate_o5.md`, `results.md` and `audit.md` exist, as far as their tasks ran.
- `FINDINGS.md` exists: Summary first, the five-claim spot-check appended with its re-run outputs.
- Every reported number has a CLAIMS row; `NEXT.md` lists proposals; everything committable is pushed.

Then set `Loop status: STOPPED` at the top of STATE.md, commit, push, **stop**, and report: tasks completed, tasks blocked and
why, the Gate O5 verdict, and what you would do next.


## Corrections of 2026-10-27

ADVISORY §8 (D56) supersedes Z2–Z3 where they differ: stop R3; drop the `fda_route_choice_v1` rows from the mix (recorded hash); re-run F0, F2 and F2s1 on the corrected mix; choose by the pre-registered rule; then the final on the corrected mix.
