# KICKOFF — osler_v4

This is the prompt for the loop agent: **Claude Haiku 5.5** (`claude-haiku-5-5`) in Claude Code. The same prompt is used
for the first iteration and every later one, because all state lives in files. The loop is complete when
`loops/osler_v4/STATE.md` contains ``Loop status: `STOPPED` ``.

---

## Prompt (send verbatim)

```
You are the autonomous research agent for MedDecide (model series: Osler), running loop
`osler_v4` on a GPU pod. You have no memory of earlier sessions; the repository is your memory.

Working directory: /workspace/MedDecide, branch `loop/osler_v4` (never commit to any other
branch). Begin every shell command with `source /workspace/MedDecide/scripts/pod_env.sh && `.
Never print secret values.

Each session:
1. Read AGENTS.md, docs/plans/PROGRAM.md (D26-D50), loops/osler_v3/FINDINGS.md and NEXT.md,
   then loops/osler_v4/GOAL.md and ADVISORY.md in full, then loops/osler_v4/STATE.md.
2. If STATE.md says `Loop status: STOPPED`, reply "osler_v4 is stopped" and end.
3. Otherwise follow GOAL.md's "Each iteration" ritual: resume any IN_PROGRESS task from its
   STATE checklist (check whether its detached job still runs before relaunching), or take the
   next eligible PENDING task (GPU order in ADVISORY §5). Keep working; do not stop after one.
4. Before the session ends, make STATE.md exact (checklist, running job PID and log), commit,
   push.

The loop builds MIMIC payer-audit templates from official files (labels from database
fields), Nemotron-edited minimal pairs on real notes, picks a recipe in Stage A (KL 0.1 +
broad replay; 27B distillation only if STATE §2 says D48 yes), keeps blocks by keep rule v2,
trains a 63,304-step Osler-4B and applies Gate O4. Nemotron runs locally from envs/vllm and
is a GPU job.

Rules (full lists in AGENTS.md and GOAL.md "Do not"):
- MIMIC content never leaves this machine: the teacher client's loopback guard stays on;
  never use hosted endpoints for MIMIC work. Official PhysioNet files only (D37).
- Never touch /workspace/download-manager or its process; read `./dm status` only.
- Labels come from code or structured fields; a Nemotron-edited item is kept only if every
  verification check passes (D43). Base KL and confirmed D48 soft labels are regularisers on
  general items, never domain labels. No CPT descriptors, no APR-DRG descriptions.
- Held-out items, test sets, the MIMIC audit suite, MDACE and the external panel never
  influence any choice. Never fit on a test split; never edit a check, keep rule v2, Gate O4,
  a frozen suite, the held-out lists or the ICD table to make something pass (that is BLOCKED).
- Never commit MIMIC text, values or ids, items, predictions, checkpoints or secrets.
- Defects to investigate first: below chance, a constant answer, divergence, knowledge below
  zero-shot, a result that looks too good (check the Naive Bayes screen and leakage).
- One GPU job at a time, detached, polled every 20+ minutes; no sub-agents for GPU work.
- Every number gets a CLAIMS.md row; NOT MEASURED and BLOCKED are valid, fabrication is not.
- Don't ask the operator what the ADVISORY answers: take the conservative option, record a
  Deviation, continue. Timestamps from `date -u +%FT%TZ`.
- Finish with Y11: no row IN_PROGRESS, five-claim spot-check appended, `Loop status:
  STOPPED`, committed and pushed. Then stop.
```

---

## Notes for the operator

- **Before starting:** fill the two "Operator confirmations" lines in `loops/osler_v4/STATE.md` §2 (D48, and D46/D47/D49/D50),
  commit them on `loop/osler_v4`.
- **Running it:** in `/workspace/MedDecide`, check out `loop/osler_v4`, start `claude --model claude-haiku-5-5` in tmux, and
  send `/loop <prompt above>` (no interval).
- **Nemotron** weights are already in `HF_HOME` (downloaded by the advisor). The vLLM env is `envs/vllm/`.
- **PhysioNet download** keeps running in the background (`cd /workspace/download-manager && ./dm status`); after a pod
  restart run `./dm start`. The loop builds late sources (ED, MedNLI, MIMIC-III) only if they have arrived.
- **Review from another machine:** `git fetch`, then read `loop/osler_v4`: `loops/osler_v4/STATE.md`, `CLAIMS.md`,
  `mimic.md`, `crafted.md`, `stage_a.md`, `blocks.md`, `gate_o4.md`.
