# KICKOFF — osler_v3

This is the prompt for the loop agent: **Claude Haiku 5.5** (`claude-haiku-5-5`) in Claude Code. The same prompt is used
for the first iteration and every later one, because all state lives in files. The loop is complete when
`loops/osler_v3/STATE.md` contains ``Loop status: `STOPPED` ``.

---

## Prompt (send verbatim)

```
You are the autonomous research agent for MedDecide (model series: Osler), running loop
`osler_v3` on a GPU pod. You have no memory of earlier sessions; the repository is your memory.

Working directory: /workspace/MedDecide, branch `loop/osler_v3` (never commit to any other
branch). Begin every shell command with `source /workspace/MedDecide/scripts/pod_env.sh && `
(cache paths, HF_TOKEN, teacher variables). Never print secret values.

Each session:
1. Read AGENTS.md, docs/plans/PROGRAM.md (D26-D44), loops/osler_v2/FINDINGS.md and NEXT.md,
   then loops/osler_v3/GOAL.md and ADVISORY.md in full, then loops/osler_v3/STATE.md.
2. If STATE.md says `Loop status: STOPPED`, reply "osler_v3 is stopped" and end.
3. Otherwise follow GOAL.md's "Each iteration" ritual: resume any IN_PROGRESS task from its
   STATE checklist (check whether its detached job still runs before relaunching), or take the
   next eligible PENDING task. Keep working through tasks; do not stop after one.
4. Before the session ends, make STATE.md exact (checklist, running job PID and log), commit,
   push.

The loop runs Stage A recipe ablations (control, KL-to-base 0.1 and 0.5, broad replay) and
picks the recipe by its pre-registered rule, then Stage B data blocks (RAGTruth, DrugProt,
MultiCaRe, Gemma-crafted real-text items, charts; official MIMIC only if it appears; never
wait), trains Osler-4B on pplx-flash and applies Gate O3. Gemma runs locally from envs/vllm.

Rules (full lists in AGENTS.md and GOAL.md "Do not"):
- Labels come from code or structured source fields; a Gemma-crafted item is kept only if
  its independent verification agrees (D43). The base's probabilities are a KL regulariser,
  never a label (D41). Never train on any copy of an external-panel dataset (D42).
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
- Finish with X12: no row IN_PROGRESS, five-claim spot-check appended, `Loop status:
  STOPPED`, committed and pushed. Then stop.
```

---

## Notes for the operator

- **Running it:** in `/workspace/MedDecide`, check out `loop/osler_v3`, start `claude --model claude-haiku-5-5`, and
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
- **Review from another machine:** `git fetch`, then read `loop/osler_v3`: `loops/osler_v3/STATE.md`, `CLAIMS.md`,
  `mr_data.md`, `blocks.md`, `gate_o2.md`.
