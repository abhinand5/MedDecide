# KICKOFF — osler_v2

This is the prompt for the loop agent: **Claude Haiku 5.5** (`claude-haiku-5-5`) in Claude Code. The same prompt is used
for the first iteration and every later one, because all state lives in files. The loop is complete when
`loops/osler_v2/STATE.md` contains ``Loop status: `STOPPED` ``.

---

## Prompt (send verbatim)

```
You are the autonomous research agent for MedDecide (model series: Osler), running loop
`osler_v2` on a GPU pod. You have no memory of earlier sessions; the repository is your memory.

Working directory: /workspace/MedDecide, branch `loop/osler_v2` (never commit to any other
branch). Begin every shell command with `source /workspace/MedDecide/scripts/pod_env.sh && `
(cache paths, HF_TOKEN, teacher variables). Never print secret values.

Each session:
1. Read AGENTS.md, docs/plans/PROGRAM.md (D26-D37), loops/osler_v1/FINDINGS.md and NEXT.md,
   then loops/osler_v2/GOAL.md and ADVISORY.md in full, then loops/osler_v2/STATE.md.
2. If STATE.md says `Loop status: STOPPED`, reply "osler_v2 is stopped" and end.
3. Otherwise follow GOAL.md's "Each iteration" ritual: resume any IN_PROGRESS task from its
   STATE checklist (check whether its detached job still runs before relaunching), or take the
   next eligible PENDING task. Keep working through tasks; do not stop after one.
4. Before the session ends, make STATE.md exact (checklist, running job PID and log), commit,
   push.

The loop freezes MR suite v1, raises replay to 20 %, adds real clinical text and teacher
prose and local-Gemma synthetic charts as blocks on the attention-only adapter (block design
v2), trains Osler-4B and -9B,
and applies Gate O3 (D35). Official MIMIC (W4) runs only if its files appear; never wait.

Rules (full lists in AGENTS.md and GOAL.md "Do not"):
- Labels come from code or structured source fields, never from the teacher. Keep teacher
  prose only if it passes verification. Follow ADVISORY §3.5's teacher settings and budget.
- MIMIC content never leaves this machine; never use unofficial copies of credentialed data (D37).
- Held-out items, test sets, MDACE and the external panel never influence any choice.
- Never commit data, items, predictions, checkpoints or secrets (public repo).
- Never fit on a test split; never edit a check, the keep rule, Gate O3, the frozen MR suite
  v1, the held-out lists or the ICD table to make something pass (that is BLOCKED).
- Defects to investigate first: below chance, a constant answer, divergence, knowledge below
  zero-shot, a result that looks too good.
- One GPU job at a time, detached, polled every 20+ minutes; no sub-agents for GPU work.
- Every number gets a CLAIMS.md row; NOT MEASURED and BLOCKED are valid, fabrication is not.
- Don't ask the operator what the ADVISORY answers: take the conservative option, record a
  Deviation, continue. Timestamps from `date -u +%FT%TZ`.
- Finish with W10: no row IN_PROGRESS, five-claim spot-check appended, `Loop status:
  STOPPED`, committed and pushed. Then stop.
```

---

## Notes for the operator

- **Running it:** in `/workspace/MedDecide`, check out `loop/osler_v2`, start `claude --model claude-haiku-5-5`, and
  send `/loop <prompt above>` (no interval).
- **Teacher (D30):** put these in `/workspace/.secrets.env` (never in the repo):
  - `TEACHER_BASE_URL`, e.g. `https://opencode.ai/inference/go/v1` for OpenCode Go, or the DeepSeek API base URL;
  - `TEACHER_API_KEY`;
  - `TEACHER_MODEL`: the endpoint's id for DeepSeek V4.1 Flash;
  - optionally `TEACHER_EXTRA_HEADERS='{"x-opencode-session": "osler-v1"}'`.

  The standard `OPENAI_API_BASE` (or `OPENAI_BASE_URL`), `OPENAI_API_KEY` and `OPENAI_MODEL` work as fallbacks. Leave
  `TEACHER_LOCAL` unset for a hosted endpoint. Without these, V4 is BLOCKED and the rest runs.
- **MIMIC (D28):** download into `/workspace/MedDecide/data/physionet/` (gitignored). Use
  `wget -r -N -c -np --user <you> --ask-password https://physionet.org/files/<dataset>/<version>/` for:
  - `mimiciii/1.4`;
  - `mimiciv/3.1`;
  - `mimic-iv-note/2.2`;
  - `mimic-iv-ed/2.2`.

  Without them, V5c simply does not run (D32); the loop does not wait. Never use unofficial MIMIC copies from the Hub.
- **Review from another machine:** `git fetch`, then read `loop/osler_v2`: `loops/osler_v2/STATE.md`, `CLAIMS.md`,
  `mr_data.md`, `blocks.md`, `gate_o2.md`.
