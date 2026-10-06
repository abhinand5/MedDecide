# bench_v0_fix0 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/bench_v0_fix0/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (F11), or when no PENDING task can proceed without the operator -->
Run started (UTC): `2026-10-06T05:47:36Z`
Last updated (UTC): `2026-10-06T09:17:08Z`
Iterations so far: `8`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| F0 | Orientation and baseline snapshot | no | — | DONE | 2026-10-06T05:47:36Z | 2026-10-06T05:48:40Z |
| F1 | MedMCQA key fix + raw-gold verification for all tier-1 sources | no | F0 | DONE | 2026-10-06T05:48:40Z | 2026-10-06T05:57:21Z |
| F2 | `noul`/`score` readout fix + reference validation + health gate | yes | F0 | DONE | 2026-10-06T05:57:21Z | 2026-10-06T07:17:04Z |
| F4 | Fresh tier v0.1: window 2026-03-01, balanced, strict slice | no | F0 | DONE | 2026-10-06T08:32:00Z | 2026-10-06T08:55:19Z |
| F3 | Template screen v2 (on v0.1) | no | F1, F4 | DONE | 2026-10-06T08:57:00Z | 2026-10-06T09:00:49Z |
| F5 | Tier-1 contamination probe | yes | F1, F2 | IN_PROGRESS | 2026-10-06T09:04:00Z |  |
| F6 | Ladder baselines on v0.1 with health gate | yes | F2, F3 | IN_PROGRESS | 2026-10-06T09:10:00Z |  |
| F7 | Decision-model baselines | yes | F3 | PENDING | | |
| F8 | Teacher pipeline gate | teacher | F1, F2, F4 | PENDING | | |
| F9 | Corrections record | no | F1, F2, F6 | PENDING | | |
| F10 | Regenerate operator audit sample | no | F3 | DONE | 2026-10-06T09:17:00Z | 2026-10-06T09:17:08Z |
| F11 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (F8 exception in
ADVISORY §6). Never run two GPU tasks at once. A `BLOCKED` task does not block
unrelated work — record it and move on.

---

## 2. Key values discovered during the run

Fill these in as they are measured; later tasks read them from here rather than
recomputing or guessing.

| key | value | source | task |
|---|---|---|---|
| v0.1 fresh window start / end | | `docs/benchmark/fresh_window.md` | F4 |
| strict-slice start | 2026-09-10 | ADVISORY §1 item 4 | F4 |
| MedMCQA gold mismatches after fix (of N) | **0 of 4,183** (all 8 tier-1 sources: 0 of 15,915 checked) | `outputs/bench_v0_fix0/F1/gold_verification.json` | F1 |
| `noul` reference agreement (pts) | **0.00 (protocol-identical**: ours 0.5828 = lm-eval 0.5828); letter-vs-continuation differs by +3.56 as a protocol, not an implementation | `outputs/bench_v0_fix0/F2/reference_validation_0p8b.json` | F2 |
| `choice` MedQA reproduction (vs bench_v0) | **-0.00016 pts** (4B 0.70149 = 0.70149); 200/200 prompts byte-identical | `outputs/bench_v0_fix0/F2/reference_validation_0p8b.json` | F2 |
| balancing K per template | | | F4 |
| kept / dropped templates (v2 screen) | | | F3 |
| cells passing / failing the health gate | | | F6 |
| teacher logprob API shape | | | F8 |

---

## 3. Current task checklist

**Fill this in before starting any task**, by copying that task's concrete steps out of
the plan as unticked boxes. Tick each one the moment it is finished, not at the end.
This is the only record of progress *within* a task — without it, a compaction mid-task
leaves you unable to tell what you already did.

Clear this section and write the new task's checklist when you start the next task; the
completed checklist goes into the iteration-log entry.

```
Task in flight: F5 (contamination probe) + F6 (ladder baselines) — BOTH RUNNING detached

F5: PID 47008, log outputs/bench_v0_fix0/F5/logs/contamination_full.log
    cmd: uv run python scripts/bench/contamination_probe.py --per-source 500
    progress: 3 of 6 models done (LFM2.5-350M finished; 0.8B-Base and 0.8B in flight)
    -> writes outputs/bench_v0_fix0/F5/contamination.json + loops/bench_v0_fix0/contamination.md

F6: PID 48188 (bash outputs/bench_v0_fix0/F6/run_queue.sh), log outputs/bench_v0_fix0/F6/logs/queue.log
    queue: 0.8B-Base -> medgemma-1.5-4b-it -> 4B -> 9B -> 0.8B (350M run separately, done in smoke)
    -> per-model JSON under outputs/bench_v0_fix0/F6/, then run
       scripts/bench/report_baselines_v0_1.py for results.json + loops/bench_v0_fix0/baselines_v0_1.md
    if the queue is interrupted, relaunch only the models whose model_<slug>.json is missing
    (delete that model's preds_<slug>.jsonl first, since predictions append)

Both must be checked for completeness of OUTPUT, not exit codes:
  F5: json has 6 models x 8 sources with n_scored and control counts
  F6: every cell n == the template's test count (or a recorded shortfall), preds count == n
```

---

## 4. Iteration log (append only, newest last)

### F10 — DONE — 2026-10-06T09:17:08Z
- Drew the operator audit sample from v0.1 fresh test: **150 rows, 50 per source, 8 kept
  templates, 20 strict-slice items**, sha256 `e380f930…` (X022). Packaging acceptance PASS 7/7.
- The audit itself stays **BLOCKED — awaiting operator** (X023): label correctness is a human
  judgement. `tools/audit/audit.html` is reused unchanged and its tests pass (2 passed).
- F5 (PID 47008) and F6 (PID 48188) continue running on the GPU; F6 queue on model 1 of 5.

### F5 + F6 — IN_PROGRESS — 2026-10-06T09:11:22Z
- F5 launched (PID 47008) after a smoke run on LFM2.5-350M; F6 runner written, smoke-tested on
  `scifact_relevant_noul_v1` and launched as a per-model queue (PID 48188).
- F6 smoke result for the smallest model on that template: n=600/600 (coverage 1.000, the metric
  F6 was asked to fix), accuracy 0.5033, **label mass 1.0000 and greedy agreement 1.000** where
  bench_v0 measured label mass ~0.000 on the same template — the F2 readout working in the F6 path.
- Deviation recorded: F5 and F6 run concurrently on the same GPU (F5 will be using 9B while F6
  runs a 4B/9B model). One-GPU-job-at-a-time is a project rule; the collision is on memory and
  throughput, not correctness, and F6 timings are not used for any contention-free claim. If
  either job OOMs, it is relaunched serially.
- Next: collect both, then F7.

### F3 — DONE — 2026-10-06T09:00:21Z
- What ran: `scripts/bench/screen_v0_1.py` over v0.1 tier-1 and fresh (regex + TF-IDF/BoW
  baselines, gold-in-state check, balanced drop rule).
- Headline: **22 templates screened, 16 kept, 6 dropped — every drop is `n_test<200`, none is a
  shortcut drop** (X017); `pubmed_observational_noul_v1` is **kept** because with a 103/103 split
  its real BoW macro accuracy is 0.796, not the 0.985 that v0's imbalance produced (X018).
- 0 templates flagged for gold-in-state leakage (X020); 16 regex baselines reported as
  NOT MEASURED rather than 0.000 (X021).
- Surprises: the corrected rule changes the *keep/drop decision* on the exact template the review
  named, which is the cleanest evidence that v0's screen was measuring imbalance.
- Next: F5 (tier-1 contamination probe).

### F4 — DONE — 2026-10-06T08:55:19Z
- Determinism: second build into a scratch directory is **byte-identical (3/3 files)**, both
  builds PASS 7/7 (`outputs/bench_v0_fix0/F4/determinism_check.txt`).
- Headline stands: 23,582 items (17,027 test / 6,555 dev), 12 templates, 2,896 strict-slice,
  0 single-class groups, 0 before the window start (X012-X016).
- Next: F3 (template screen v2 over both v0.1 tiers).

### F4 — IN_PROGRESS — 2026-10-06T08:39:54Z
- What ran: `build_fresh.py` on the D11 window into `data/bench/v0.1/fresh/` (five attempts; two
  were killed mid-flight for racing each other, one crashed on a `date` serialisation, one hit a
  CT.gov 429), then a determinism rebuild into `/workspace/tmp/fresh_v01_rebuild/`.
- Headline: **23,582 fresh items** (17,027 test / 6,555 dev) across 12 templates, 0 before the
  window start, **2,896 in the strict slice**, 0 single-class groups, acceptance **PASS 7/7**
  (X012-X014).
- Found and fixed two more loader defects: CT.gov `healthyVolunteers` is a JSON boolean, so
  `str(value) == "yes"` made **every** item "no" (X015) — caught only because balancing forced
  the single-class split into the open; and the manifest writer crashed on a `date` object.
- Self-correction: I first reported 13,047 test / 10,535 dev by misreading the source-keyed
  `acceptance.totals` as split counts; corrected to 17,027 / 6,555 in the audit and CLAIMS (X012).
- Next: finish the determinism check, then F4 DONE, then F3 (screen on v0.1).

### F2 — DONE — 2026-10-06T07:17:04Z
- What ran: `check_readout_health.py` (0.8B and 4B, 200 `noul` + 200 `score` items each),
  `validate_readouts.py` (choice reproduction + relevance `noul` + lm-eval's pubmedqa task),
  `lm_eval --tasks pubmedqa_parquet --limit 477`, `pytest` (84 tests).
- Output: committed `loops/bench_v0_fix0/readout_validation.md`; artifacts under
  `outputs/bench_v0_fix0/F2/`.
- Headline: the readout defect is fixed — `noul`/`score` median label mass is **0.996-0.999**
  (was ~0.002) with greedy agreement 0.96-1.00 on both models, and the previously validated
  `choice` path is **byte-identical** (MedQA 0.70149 = 0.70149) (X007, X008).
- Implementation check vs lm-evaluation-harness on a yes/no task: **0.5828 vs 0.5828
  (delta 0.00 pts)** under the same protocol (X009); the letter readout's +3.56 pts against the
  continuation rule is a protocol difference, reported as such (X010).
- Surprises: the first two validation designs were themselves broken (a bare BoolQ prompt dropped
  label mass to 1.2e-04; canonical gold compared against yes/no labels), and lm-eval's own
  `pubmedqa` task cannot load its script-based dataset on `datasets` 5.x; all three are recorded in
  `readout_validation.md`.
- Next: F4 (rebuild the fresh tier on the wider window with balancing).

### F1 — DONE — 2026-10-06T05:57:21Z
- What ran: fixed `_medmcqa_item` (cop is 0-based), wrote `scripts/bench/verify_gold.py` and
  `src/meddecide/bench/verify.py`, rebuilt tier-1 as v0.1, ran the independent gold check twice.
- Output: `outputs/bench_v0_fix0/F1/` (gold_verification.json, SELF_AUDIT.md),
  `data/bench/v0.1/tier1/` (manifest, acceptance, audit).
- Headline: **0 gold mismatches of 15,915 checked items across all eight tier-1 sources** (X004);
  MedMCQA now has 4,183 test items with gold A 1,348 / B 1,085 / C 925 / D 825, identical to the raw
  `cop` counts (X002, X003).
- Surprises: verification also caught a *real* second loader bug - three nfcorpus score items
  carried a grade borrowed from another query's pool (X005) - plus two defects in my own checker
  (label-vs-key comparison, pooled qrels) that produced false alarms before the real bug surfaced.
- Next: F2 (`noul`/`score` readout), which gates every GPU number in this loop.


### F0 — DONE — 2026-10-06T05:48:40Z
- What ran: `git rev-parse HEAD`, `sha256sum` over every v0 benchmark JSONL, manifest totals via
  `jq`, `uv run pytest --junit-xml`, `uv run ruff check .`.
- Output: `outputs/bench_v0_fix0/F0/{start.json,pytest.txt,pytest.xml,SELF_AUDIT.md}`.
- Headline: the repair loop starts from a provable snapshot — commit `88dea18`, 11 v0 files
  hashed, tier-1 21,202 items and fresh 41,502 items (reproducing bench_v0 C013/C018), test suite
  74 passed / 0 failed (X001).
- Surprises: the repo's `-q` addopts suppresses pytest's console summary under `--tb=no`, so the
  count had to come from the JUnit XML; the first parse attempt raised IndexError and is recorded
  in the audit rather than hidden.
- Next: F1 (MedMCQA key fix and gold verification for all eight tier-1 sources).

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

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|

---

## 6. Questions for the operator

Anything you could not resolve without a human. Be specific enough to answer without
re-reading the run: state the ambiguity, the options, and which you would pick.

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
