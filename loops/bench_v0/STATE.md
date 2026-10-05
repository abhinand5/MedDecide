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
Last updated (UTC): `2026-10-06T00:00:00Z`
Iterations so far: `3`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| T0 | Environment verification | no | — | DONE | 2026-10-05T20:49:53Z | 2026-10-05T20:50:24Z |
| T1 | Repo scaffold | smoke | T0 | DONE | 2026-10-05T20:53:18Z | 2026-10-05T20:58:30Z |
| T2 | Item schema | no | T1 | DONE | 2026-10-05T20:59:06Z | 2026-10-05T21:02:40Z |
| T3 | Tier 1 loaders | no | T2 | DONE | 2026-10-05T21:02:40Z | 2026-10-05T21:22:10Z |
| T5 | Fresh-tier builders | no | T2 | DONE | 2026-10-05T21:22:10Z | 2026-10-05T21:52:30Z |
| T6 | Eval harness | yes | T2 | DONE | 2026-10-05T22:00:12Z | 2026-10-05T23:20:00Z |
| T7 | Harness validation vs reference | yes | T3, T6 | PENDING | | |
| T8 | Fresh-template screen | no | T5 | DONE | 2026-10-05T23:22:00Z | 2026-10-06T00:00:00Z |
| T4 | Tier 1 contamination probe | yes | T3, T6 | PENDING | | |
| T9 | Zero-shot baseline table | yes | T7, T8 | PENDING | | |
| T10 | Teacher pipeline gate | teacher | T3, T5, T6 | PENDING | | |
| T11 | Operator audit page + sample | no | T8 | IN_PROGRESS | 2026-10-06T00:00:00Z | | |
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
| fresh window start | `2026-09-10` (teacher HF repo creation; latest of all ladder/teacher dates) | `docs/benchmark/fresh_window.md` | T5 |
| fresh window end | build date, `2026-10-05` for this build | `data/bench/fresh/manifest.json` | T5 |
| tier-1 items | 21,202 in 10 templates (test 13,099 / dev 8,103), 0 leaks, 0 dup ids | `data/bench/tier1/audit.json` | T3 |
| fresh items | 41,502 in 12 templates (test 30,870 / dev 10,632), 0 leaks, 0 dup ids | `data/bench/fresh/audit.json` | T5 |
| item id rule | stable hash of source + record + template + option seed + **split** | `src/meddecide/bench/schema.py` | T2/T3 |
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
Task in flight: T11 (operator audit page + sample)
Working dir:    outputs/bench_v0/T11/

- [ ] draw a seeded stratified sample of 150 fresh TEST items (50 per source, across kept templates)
- [ ] write outputs/bench_v0/T11/audit_sample.jsonl (item_id, template_id, source_url, state, question,
      options, gold, and the raw structured field(s) the gold came from)
- [ ] tools/audit/audit.html: self-contained, no network, file picker, one item per screen,
      A accept / R reject / N note, arrow navigation, progress bar, localStorage autosave,
      Export -> audit_v0.jsonl (item_id, verdict, note, timestamp)
- [ ] commit tools/audit/audit.html (no data); do NOT commit the sample
- [ ] validate the page loads a sample and exports valid JSONL (headless check of the export logic)
- [ ] mark the audit itself BLOCKED - awaiting operator, with instructions in STATE questions
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

### session-2 close — 2026-10-05T21:58:00Z
- Tasks completed this session: T0, T1, T2, T3, T5 (all DONE and pushed).
- Not started: T6 (eval harness, GPU) — the session ended before it began; T4/T7 depend on it,
  T8 on T5 (now DONE), so T6 is the next eligible task.
- Nothing is running: no detached jobs, no GPU work in flight (`pgrep` clean for build scripts).
- Raw outputs for this session live under `outputs/bench_v0/{T0,T1,T2,T3,T5}/` (gitignored),
  including SELF_AUDIT.md for each completed task.

### T8 — DONE — 2026-10-06T00:00:00Z
- What ran: `uv run python scripts/bench/screen_templates.py` (regex + TF-IDF/LogReg baselines,
  the latter fitted on dev and scored on test).
- Output: `src/meddecide/eval/screen.py`, `scripts/bench/screen_templates.py`,
  `configs/template_screen_patterns.yaml`, committed `loops/bench_v0/template_screen.md`,
  `outputs/bench_v0/T8/screen.json`, `data/bench/fresh/manifest_screened.json`.
- Headline: 10 of 12 fresh templates kept, 2 dropped — `ct_healthy_volunteers_noul_v1` because
  all 2,900 test items share one gold class, and `pubmed_observational_noul_v1` because the
  bag-of-words baseline reaches 0.985 (majority 0.985, macro 0.500: no skill); the BoW macro
  accuracy is <=0.564 on every template, so the micro numbers are not evidence of difficulty
  (C033-C037).
- Surprises: the screen's most useful output is not the drop list but the per-template note that
  every BoW classifier collapsed toward the majority class; the healthy-volunteers field is
  constant in a 25-day window, which is a window-width problem surfacing as a template problem.
- Next: T11 (audit page + sample) is CPU-only and now unblocked; T9 still waits on T7.

### T6 — DONE — 2026-10-05T23:20:00Z
- What ran: `uv run pytest` (65), `scripts/bench/run_eval.py` twice (0.8B + 0.8B-Base, 50
  MedQA items), `scripts/bench/run_probes.py` (30 items, 3 probes), `scripts/bench/label_token_check.py`.
- Output: `src/meddecide/eval/{readout,harness,probes}.py`, `scripts/bench/{run_eval,run_probes,label_token_check}.py`,
  `tests/test_harness.py`; predictions/summaries/probe JSON under `outputs/bench_v0/T6/`.
- Headline: the harness scores 50 MedQA items on Qwen3.5-0.8B (0.34, majority 0.36, Brier
  0.761, ECE 0.226) and on -Base (0.40), writes complete prediction files, and passes 65 unit
  tests; the option-letter variant is **measured per model** (`bare` for Qwen) rather than
  assumed (C025–C032).
- Surprises: three readout bugs, each caught by a different cross-check — a prompt that made
  the model continue the question instead of choosing, a left-padding index that read the wrong
  token position, and a variant heuristic that read `" A"` while the model emits `A`. The third
  was invisible in accuracy terms but showed up as label mass ~1e-06; after fixing it the mass
  is 0.996 and the best option is in the vocab top-5 on every item. Also replaced an unstable
  `argsort` rank metric with a tie-robust count after it disagreed with direct inspection.
- Next: T7 must validate this harness against lm-evaluation-harness before any T9 number is
  reported; T4 (contamination probe) and T8 (CPU screen) are now both unblocked.

### T5 — DONE — 2026-10-05T21:52:30Z
- What ran: `uv run python scripts/bench/build_fresh.py --window-end 2026-10-05 --pubmed-files 12`
  (twice, to prove determinism), then `scripts/bench/audit_fresh.py`.
- Output: `data/bench/fresh/*.jsonl` + `manifest.json` + `acceptance.json` + `audit.json`
  (gitignored); `docs/benchmark/fresh_window.md` (committed); code in
  `src/meddecide/bench/fresh/`; `outputs/bench_v0/T5/SELF_AUDIT.md`.
- Headline: 41,502 fresh items in 12 templates from ClinicalTrials.gov, openFDA and PubMed,
  every one dated inside the 2026-09-10 → 2026-10-05 window (0 items before the start),
  with 0 split leaks, 0 duplicate ids, 0 straddling question texts and a byte-identical
  rebuild (C018–C024).
- Surprises: the window is only 25 days wide because the teacher's HF repo date is the
  binding cutoff, so openFDA yields 122 labels and its class template only 19 items; the CT.gov
  builder silently produced 1 of 5 templates until field names were corrected against the live
  API; PubMed update files repeat records (one PMID up to 6 times) and the duplicate copies
  were being discarded by the identity guard until record-level merging was added.
- Next: T6 (eval harness) is now unblocked and needs the GPU. T7 depends on T6, T8 on T5,
  T11 on T8.

### T3 — DONE — 2026-10-05T21:22:10Z
- What ran: `uv run python scripts/bench/build_tier1.py` then `scripts/bench/audit_tier1.py`
  (recounts from raw JSONL in a fresh process).
- Output: `data/bench/tier1/*.jsonl` + `manifest.json` + `acceptance.json` + `audit.json`
  (gitignored); code in `src/meddecide/bench/tier1/` (committed);
  `tests/test_tier1_loaders.py` (18 tests); `outputs/bench_v0/T3/SELF_AUDIT.md`.
- Headline: 21,202 tier-1 items from 8 public sources build cleanly — 13,099 test / 8,103
  dev, 0 split leaks, 0 duplicate ids, 0 question texts in two splits, 8/8 file hashes
  matching the manifest (C013–C017).
- Surprises: the self-audit caught three integrity bugs that a counts-only check would have
  missed — item ids collided across splits, MedQuAD's `question_id` is not unique (9,662
  repeats), and MMLU/NFCorpus/MedQuAD repeat content across their official splits, so a stem
  could be seen in dev and scored in test. All three fixed; the fix is a check (content found
  in two splits is forced to test), never a loosened tolerance.
- Next: T5 (fresh tier) is the other half of the benchmark; T6 (harness) can start once T5's
  checklist is parked or in parallel by a later session.

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

**Q4 (T6, added 2026-10-05T23:20:00Z) — the fixed T6 prompt is part of the baseline protocol.**
The harness renders every model through its own chat template with a fixed system prompt and a
fixed "respond with a single letter … the correct option is:" instruction, reads the next-token
distribution over option-letter tokens, and detects per model whether the model emits the bare
letter or a space-prefixed one. ADVISORY forbids tuning prompts per model to raise scores, so
this prompt is fixed for all models and recorded in every run config. Two consequences the
operator should know before T9: (a) base checkpoints (e.g. `Qwen3.5-0.8B-Base`) are scored with
the same chat-template path as instruct models, which is the comparable choice but is not how a
base model is usually evaluated; (b) the readout is a zero-shot letter choice, whereas
lm-evaluation-harness scores option continuations — T7 runs both and reports the gap, and that
gap is the main threat to comparability with published numbers. If the operator wants a
second protocol (continuation scoring) for comparability, that is a T9/T7 decision, not
something I will add unilaterally.

**Q3 (T5, added 2026-10-05T21:52:30Z) — the fresh window is only 25 days wide.**
Window start is `2026-09-10`, the Hugging Face repository creation date of the teacher
(`deepseek-ai/DeepSeek-V4.1-Flash`), which is the latest of all ladder/teacher dates and the
only one that is not documented anywhere. Consequences: ClinicalTrials.gov gives 3,719 studies
in-window (fine), openFDA gives 122 labels (thin — the class template has 19 items), PubMed
gives 44,217 records (ample). Options: (a) keep the conservative 25-day window and report the
thin templates as thin (my pick — it is the only window that is defensible without a
documented cutoff); (b) widen the start to a *documented* date if the operator can supply the
teacher's real training cutoff (e.g. from the DeepSeek-V4.1-Flash tech report), which would
let the window start much earlier and thicken the openFDA templates; (c) widen to the earliest
Qwen3.5 repo date (2026-02-27) and accept a contamination risk for the teacher. I did not
choose (b) or (c) because both require information I do not have, and a benchmark's headline
tier must not rest on a guess. Note also that PubMed's filter field is the Entrez date, so
in-window records include older papers being re-indexed — that is fresh in the record stream,
not necessarily post-cutoff science.

**Q2 (T3, added 2026-10-05T21:22:10Z) — MedQuAD license.**
`lavita/MedQuAD` (used for the tier-1 routing template) declares **no license** on its HF
card; the upstream National Library of Medicine content has no explicit redistribution
statement either. Options: (a) keep it in v0 with `license: UNKNOWN` recorded and decide at
release time (my pick — it is one of 10 templates and can be dropped without touching the
harness or the teacher gate); (b) drop the template now. The full 47,441-row MedQuAD mirror
is gitignored and never committed; only counts/hashes appear in the repo. Related: the same
source carries UMLS fields, which the loader drops by name and a unit test asserts never
reach a serialised item.

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
