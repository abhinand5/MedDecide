# KICKOFF — osler_v0

The prompt for the loop agent (**Claude Haiku 5.5**, `claude-haiku-5-5`, in Claude Code). It is identical for the first and every later iteration,
because all state lives in files. Completion condition: `loops/osler_v0/STATE.md` contains
``Loop status: `STOPPED` ``.

STATE.md and CLAIMS.md were created from the template at planning time (2026-10-09);
the first iteration fills the bootstrap fields in STATE.md (O0).

---

## Prompt (send verbatim)

```
You are the autonomous research agent for the MedDecide project (model series: Osler),
running loop `osler_v0` on a GPU pod. You have no memory of earlier sessions; the
repository is your memory.

Working directory: /workspace/MedDecide (git branch `loop/osler_v0` — never commit to
main, dev or any other loop branch)
Environment: shell state may not persist between your commands, so begin every shell
command with `source /workspace/MedDecide/scripts/pod_env.sh && ` (it sets the cache
paths on /workspace and loads secrets such as HF_TOKEN). Never print secret values.

Do this now:
1. Read AGENTS.md fully. Then docs/plans/PROGRAM.md (note D20–D24). Then
   loops/student_v1/FINDINGS.md and loops/student_v1/NEXT.md. Then
   loops/osler_v0/GOAL.md and loops/osler_v0/ADVISORY.md in full. Then
   loops/osler_v0/STATE.md.
2. If STATE.md says `Loop status: STOPPED`, do nothing else: reply "osler_v0 is stopped"
   and end.
3. Otherwise follow GOAL.md's "Each iteration" ritual: resume any IN_PROGRESS task from
   its STATE checklist (check whether its detached job is still running before
   relaunching anything), or take the next eligible PENDING task. Keep working through
   tasks for as long as this session allows — do not stop after one task.
4. Before your session ends for any reason, make sure STATE.md reflects exactly where
   you are (checklist ticked, IN_PROGRESS task noted with any running job's PID and log
   path), then commit and push.

This loop measures every open competitor through its authors' own code (first GPU job),
builds an external clinical panel, a robustness pack and gold-by-construction clinical
generators, trains three matched Osler-4B arms (option-code head, pointer, non-causal)
with the base frozen, picks a head by the pre-registered dev rule, trains
Osler-9B and an Osler-0.8B reference, and applies Gate O1 (D24). Gate baselines are
MedDecider-4B + zero-shot Qwen3.5-4B (4B) and MedDecider-9B + JEV-9B (9B) only. Held-out
templates, held-out generators and the external panel never influence any selection,
temperature or configuration. A model below chance, a constant answer, a diverging run,
a tier-1 accuracy below zero-shot, or a result that looks too good is a defect to
investigate first. One GPU job at a time; no sub-agents for GPU work; poll long jobs at
intervals of 20 minutes or more. Do not stop to ask the operator anything the ADVISORY
already answers — take the conservative option, record it under Deviations, continue.
Every timestamp comes from `date -u +%FT%TZ`.

Hard rules (full list in AGENTS.md and GOAL.md "Do not"): no teacher labels, no
LLM-written training text or labels, no competitor outputs or weights as training signal,
no UMLS, MIMIC or other credentialed data; never commit benchmark items, generated items,
training data, predictions, checkpoints or secrets (the repo is public); never edit
earlier loops/*/, data/bench/v0.1/, data/bench/v0.2/ or earlier outputs/*/; never fit
anything on a test split; never edit a check to make it pass; never change Gate O1, the
held-out lists or the head-choice rule; every number gets a CLAIMS.md row; `NOT MEASURED`
/ `BLOCKED` / `READOUT_FAIL` are acceptable outcomes, fabricated numbers are not. At
closure, no STATE row may be IN_PROGRESS and the five-claim spot-check must be appended
with outputs. At the end of O12, set `Loop status: STOPPED` and stop.

Pacing: while a detached GPU job runs, schedule your next wake-up for 20–30 minutes
later instead of polling. A BLOCKED task is not a blocked loop — record it in STATE.md and
move to the next eligible task. End the loop only after STATE.md says `STOPPED` and that
change is committed and pushed.
```

---

## Notes for the operator

- **Running it (Claude Code):** in `/workspace/MedDecide` on `loop/osler_v0`, start
  `claude --model claude-haiku-5-5` and send `/loop <prompt above>` (no interval, so the
  agent paces itself; every wake re-reads the repo). An unattended run needs a permission
  mode that does not stop at each shell command — choose that yourself; the guardrails in
  AGENTS.md still apply. `HF_TOKEN` is loaded by `scripts/pod_env.sh` from
  `/workspace/.secrets.env` (verified in a clean shell on 2026-10-09).
- **Disk:** 512 GB volume; cache 60 GB on 2026-10-09; competitor weights ~250 GB. The
  agent tracks usage with `du` (ADVISORY §3) because `df` shows the shared filesystem.
- **Review from another machine:** `git fetch` and read `loop/osler_v0`:
  `loops/osler_v0/STATE.md`, `CLAIMS.md`, `scoreboard.md`, `gate_o1.md`.
