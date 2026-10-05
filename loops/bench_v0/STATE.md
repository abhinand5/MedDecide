# bench_v0 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/bench_v0/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (T12), or when no PENDING task can proceed without the operator -->
Run started (UTC): `2026-10-05T20:49:53Z`
Last updated (UTC): `2026-10-05T21:02:40Z`
Iterations so far: `1`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| T0 | Environment verification | no | — | DONE | 2026-10-05T20:49:53Z | 2026-10-05T20:50:24Z |
| T1 | Repo scaffold | smoke | T0 | DONE | 2026-10-05T20:53:18Z | 2026-10-05T20:58:30Z |
| T2 | Item schema | no | T1 | DONE | 2026-10-05T20:59:06Z | 2026-10-05T21:02:40Z |
| T3 | Tier 1 loaders | no | T2 | IN_PROGRESS | 2026-10-05T21:02:40Z | | |
| T5 | Fresh-tier builders | no | T2 | PENDING | | |
| T6 | Eval harness | yes | T2 | PENDING | | |
| T7 | Harness validation vs reference | yes | T3, T6 | PENDING | | |
| T8 | Fresh-template screen | no | T5 | PENDING | | |
| T4 | Tier 1 contamination probe | yes | T3, T6 | PENDING | | |
| T9 | Zero-shot baseline table | yes | T7, T8 | PENDING | | |
| T10 | Teacher pipeline gate | teacher | T3, T5, T6 | PENDING | | |
| T11 | Operator audit page + sample | no | T8 | PENDING | | |
| T12 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (T10 exception in
ADVISORY §6). Never run two GPU tasks at once. A `BLOCKED` task does not block
unrelated work — record it and move on.

---

## 2. Key values discovered during the run

Fill these in as they are measured; later tasks read them from here rather than
recomputing or guessing.

| key | value | source | task |
|---|---|---|---|
| fresh window start | | `docs/benchmark/fresh_window.md` | T5 |
| fresh window end | | | T5 |
| option-letter token variant per model | | | T6 |
| T7 reference agreement (pts) | | | T7 |
| kept / dropped fresh templates | | | T8 |
| teacher logprob API shape | | | T10 |
| teacher items/hour (non-thinking / thinking) | | | T10 |

---

## 3. Current task checklist

**Fill this in before starting any task**, by copying that task's concrete steps out of
the plan as unticked boxes. Tick each one the moment it is finished, not at the end.
This is the only record of progress *within* a task — without it, a compaction mid-task
leaves you unable to tell what you already did.

Clear this section and write the new task's checklist when you start the next task; the
completed checklist goes into the iteration-log entry.

```
Task in flight: T3 (tier 1 loaders)
Working dir:    outputs/bench_v0/T3/

- [ ] loaders: MedQA, MedMCQA, PubMedQA, MMLU medical subsets, SciFact, PubHealth/HealthVer, TREC-COVID + NFCorpus, MedQuAD
- [ ] official splits preserved; MedMCQA test->validation, dev carved from train
- [ ] caps applied (5000 test / 2000 dev per source) with recorded sampling
- [ ] write data/bench/tier1/<source>.jsonl + manifest rows
- [ ] license recorded per source (or UNKNOWN + question to operator)
- [ ] no (source, source_record_id) in more than one split
- [ ] acceptance check run and passed
- [ ] self-audit (R6) written to SELF_AUDIT.md
- [ ] CLAIMS.md rows appended
- [ ] STATE updated, committed, pushed
```

---

## 4. Iteration log (append only, newest last)

<!-- Template for each entry:
### <task-id> — <DONE|BLOCKED> — <UTC timestamp>
- What ran: <command or script>
- Output: <path>
- Headline: <one number or one sentence, with its CLAIMS id>
- Surprises: <anything unexpected, or "none">
- Next: <what this unblocks>
-->

<!-- The Headline line matters beyond this file: these headlines are the raw material
     FINDINGS.md's Summary section is written from at closure. A headline that only
     makes sense with full context ("done, see report") starves the summary. Write
     each one so a reader who has not seen the task can repeat it: what was measured,
     what came out, with what denominator. -->

### T2 — DONE — 2026-10-05T21:02:40Z
- What ran: `uv run pytest` (37), `uv run ruff check .`, `uv run python scripts/t2_acceptance.py`.
- Output: `src/meddecide/bench/schema.py`, `src/meddecide/utils/io.py`,
  `docs/benchmark/schema.md`, `tests/test_schema.py`, `tests/test_manifest.py`,
  `outputs/bench_v0/T2/acceptance.json`, `outputs/bench_v0/T2/SELF_AUDIT.md`.
- Headline: the one item format is defined and enforced — 6 documented invariants each
  backed by a check, deterministic 16-hex ids, a reader that accounts for every line
  (3/3 kept, 0 dropped), a manifest whose sha256 equals the file's
  (`4f74d121…0926`, C011) and whose counts close, and 0 records crossing splits (C010, C012).
- Surprises: whitespace-only `state` passed `min_length=1`; now rejected by an explicit
  blank-text validator (stricter, not looser). Test count 37 vs 35 test functions is
  parametrisation, explained in the audit.
- Next: unblocks T3 (tier 1 loaders), T5 (fresh builders), T6 (harness) — the three
  independent branches of the loop.

### T1 — DONE — 2026-10-05T20:58:30Z
- What ran: `uv sync` (157 resolved / 154 installed), `uv run pytest`, `uv run ruff check .`,
  `uv run python scripts/gpu_smoke.py --model Qwen/Qwen3.5-0.8B`.
- Output: `pyproject.toml`, `uv.lock`, `src/meddecide/{bench,eval,teacher,utils}/`,
  `tests/` (27 tests), `configs/bench_v0.yaml`, `docs/benchmark/schema.md`,
  `outputs/bench_v0/T1/gpu_smoke.txt`, `outputs/bench_v0/T1/SELF_AUDIT.md`.
- Headline: the full GPU stack works on this pod — `Qwen/Qwen3.5-0.8B` (752,393,024 params,
  revision `2fc0636…`) loads in bf16 at capability sm_120 and completes a forward pass
  (C006, C007); `uv run pytest` 27 passed and `uv run ruff check .` clean (C008).
- Surprises: `transformers` 5.x needs `accelerate` for `device_map` (found by the smoke run,
  added, re-ran); two of my own test expectations were arithmetically wrong and were
  corrected with extra stricter cases (see `outputs/bench_v0/T1/SELF_AUDIT.md`).
- Next: unblocks T2 (item schema); T3/T5/T6 all depend on T2.

### T0 — DONE — 2026-10-05T20:50:24Z
- What ran: inline environment probe writing `outputs/bench_v0/T0/env.json` (GPU/CPU/RAM/disk,
  cache paths, `HF_TOKEN` presence, HF `whoami`, gated `medgemma` config fetch, three data
  APIs, `git push --dry-run`, teacher endpoint).
- Output: `outputs/bench_v0/T0/env.json`, `outputs/bench_v0/T0/SELF_AUDIT.md`.
- Headline: 9 of 10 environment checks PASS; the only FAIL is the teacher endpoint
  (`TEACHER_BASE_URL` unset), which blocks T10 only (C002, C003). Pod = RTX PRO 6000
  Blackwell 97887 MiB, driver 595.91.07, CUDA 13.0, 128 CPUs, 2015 GB RAM (C001).
- Surprises: teacher endpoint variables were not in `/workspace/.secrets.env` at loop start
  — T10 will be `BLOCKED` unless the operator provides them. First draft of the audit
  undercounted the checks (9 vs 10); corrected in place (R5).
- Next: unblocks T1 (repo scaffold) and every other task; T10 carries the endpoint question.

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|
| T10 | Teacher pipeline gate | `TEACHER_BASE_URL` and `TEACHER_API_KEY` are not set in the pod environment (`/workspace/.secrets.env` has neither), so the endpoint cannot be reached | Operator adds `export TEACHER_BASE_URL=...` and `export TEACHER_API_KEY=...` to `/workspace/.secrets.env` and brings the self-hosted DeepSeek-V4.1-Flash endpoint up; T10 then needs a `GET $TEACHER_BASE_URL/models` → 200. Not yet marked BLOCKED in the task board — decided when the task is reached. |

---

## 6. Questions for the operator

Anything you could not resolve without a human. Be specific enough to answer without
re-reading the run: state the ambiguity, the options, and which you would pick.

**Q1 (T10, added 2026-10-05T20:50:24Z) — teacher endpoint not configured.**
At T0 the pod had no `TEACHER_BASE_URL` / `TEACHER_API_KEY`, so `GET $TEACHER_BASE_URL/models`
could not be attempted. T10 is the only task that needs it. Options: (a) operator adds both
variables to `/workspace/.secrets.env` and starts the self-hosted DeepSeek-V4.1-Flash
endpoint before T10 is reached — preferred; (b) leave unset, in which case T10 is recorded
as `BLOCKED — teacher endpoint not provided` with no numbers (GOAL.md allows this
explicitly, and a blocked T10 does not block T12). I will re-check the endpoint at the
moment T10 is reached and will not wait for it.

---

## 7. Deviations from the plan

Any place you departed from GOAL/ADVISORY, with the reason. An empty section is the
expected outcome. Editing code or a check to make it pass is never an acceptable
deviation — that is a `BLOCKED`.

---

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| | | | |
