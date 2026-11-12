# GOAL — osler_v7 (data-engineered Pareto recipe)

Follow `loops/osler_v7/ADVISORY.md` exactly. The loop measures three levers on non-panel dev data before training anything,
then trains one arm per base, selects by the pre-registered dev rule, and gates each model against its own zero-shot base.
Nothing in ADVISORY may change after a run starts. Any change needs an operator instruction recorded in STATE §2.

## Targets
- **Osler-9B** on `vllm-sr/d3-flash` (revision 67e03e1bc157961c582ee7c6a35d9b15fada3dec). Lead target.
- **Osler-4B** on `perplexity-ai/pplx-decider-flash-v1.2` (the base pinned for osler_v2 onwards). Parity target.

## Read first, in this order
1. `AGENTS.md` (Current loop section, R1–R10, guardrails).
2. `docs/plans/PROGRAM.md`: D22, D23, D26, D28, D30, D33, D39, D45, D60, D61.
3. `loops/osler_v7/ADVISORY.md` (the plan; §0 open decisions must be settled before T6).
4. `loops/osler_v7/STATE.md` (board, operator approvals).
5. `loops/osler_v6/FINDINGS.md` and `loops/d3_distill/FINDINGS.md` (what was measured; the d3 gate report `loops/d3_distill/GATE_O7.md`).

## Each iteration
1. Read STATE.md. Take the first PENDING task whose dependencies are DONE and whose approval is recorded (GPU tasks need the
   operator's approval line in STATE §2).
2. Mark it IN_PROGRESS with `date -u +%FT%TZ`; copy its steps into the STATE checklist.
3. Execute, run its acceptance check, self-audit (R6: SELF_AUDIT.md in `outputs/osler_v7/<task>/`).
4. Append CLAIMS.md rows (every number has one; NOT MEASURED and BLOCKED are valid results).
5. Mark DONE or BLOCKED — reason; commit `loop(osler_v7): T<n> <DONE|BLOCKED> — <headline>`; push `loop/osler_v7`.

GPU jobs: one at a time, detached, logs under `outputs/osler_v7/<task>/logs/`. Poll every 20 minutes or more. Check disk
headroom (2 GB write) before each job.

## Do not
- Use the external panel, its datasets or copies, the knowledge and MR frozen sets, or the MIMIC audit-suite and held-out items
  for training, design, selection or calibration. After v7 starts, the panel is reported only as an additional row.
- Send MIMIC text to any hosted service (D28), or commit MIMIC-derived text, items, predictions, checkpoints or keys.
- Use d1-ultra (hosted) for any MIMIC-derived item (D60 blocks it until the operator confirms a BAA).
- Commit raw items, predictions, labels, checkpoints or secrets. Commit code, configs, manifests (IDs, hashes, counts) and
  aggregates only.
- Change the head, the attention mode or the adapter scope (D26, D33), or change the base beyond the two targets above.
- Edit `main`, `dev`, or any other loop's branch, or `docs/plans/`. Edit AGENTS.md only at Z-setup (already done).
- Make anything pass by changing a threshold, a gold label or a check (R8). That is BLOCKED.
- Start the next loop, or anything in ADVISORY §0 that is still open.

## Done when
Every task is DONE or BLOCKED with a reason; no task is IN_PROGRESS; `FINDINGS.md` has a Summary first and a five-claim
spot-check; `Loop status: STOPPED` is at the top of STATE.md; committed and pushed. Then stop and report: tasks done, tasks
blocked and why, the gate verdicts for each model, and what comes next.
