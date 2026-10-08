# student_v1 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/student_v1/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (V10), or when no PENDING task can proceed without the operator -->
Run started (UTC): `2026-10-08T03:59:12Z`
Last updated (UTC): `2026-10-08T05:23:00Z`
Iterations so far: `2`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| V0 | Orientation, snapshot, Unsloth environment | smoke | — | DONE | 2026-10-08T03:59:12Z | 2026-10-08T04:17:00Z |
| V1 | JEV-9B on v0.2 + G1 recomputed per D16 | yes | V0 | DONE | 2026-10-08T04:30:39Z | 2026-10-08T05:28:00Z |
| V2 | Padding fix + measured effect | small | V0 | PENDING | | |
| V3 | Checkpoint trajectory (seen vs held-out) | yes | V2 | PENDING | | |
| V4 | Training data v1 (diversity) | no | V0 | PENDING | | |
| V5 | Unsloth validation + converter + evaluator | small | V0, V4 | PENDING | | |
| V6 | Arm A: pointer head | yes | V2, V4 | PENDING | | |
| V7 | Arm B: letter-readout LoRA | yes | V4 | PENDING | | |
| V8 | Arm C: Clef-style head via Unsloth | yes | V5 | PENDING | | |
| V9 | Evaluation + G1 per arm + arm-vs-arm | yes | V1, V6, V7, V8 | PENDING | | |
| V10 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (exception in ADVISORY §6:
V4 may run on CPU while V1–V3 use the GPU). Never run two GPU jobs at once. A `BLOCKED` task
does not block unrelated work — record it and move on. No row may be left `IN_PROGRESS` at
closure.

---

## 2. Key values discovered during the run

Fill these in as they are measured; later tasks read them from here rather than
recomputing or guessing.

| key | value | source | task |
|---|---|---|---|
| unsloth / torch / transformers versions (unsloth env; main env) | unsloth env: unsloth 2026.10.2, torch 2.14.1+cu130, transformers 5.17.0, peft 0.21.2; main env: torch 2.14.1+cu130, transformers 5.18.0 | `outputs/student_v1/V0/envs.json` | V0 |
| G1 for student_v0 per D16 (seen / held-out) | seen: PASS vs zero-shot (macro +0.1633, CI [0.1530, 0.1741]) and vs JEV-9B on the intersection (n=15,911; macro +0.0435, CI [0.0311, 0.0548]). Held-out: FAIL vs both (vs zero-shot macro −0.0176, CI [−0.0301, −0.0048]; vs JEV-9B macro −0.1553, CI [−0.1661, −0.1435]). Claims V011–V014. | `loops/student_v1/g1_student_v0_rerun.md` | V1 |
| JEV-9B v0.2 coverage | 35,097 of 37,289 kept test items (94.1%); 2,192 excluded by the 8,192-token prompt cap, all on openFDA templates (claims V008) | `outputs/student_v1/V1/jev9b/jev9b.json` | V1 |
| padding effect on student_v0 test accuracy (max per-template Δ) | | `outputs/student_v1/V2/effect.json` | V2 |
| Spearman(step, held-out dev accuracy) with CI | | `loops/student_v1/trajectory.md` | V3 |
| training mix v1: items, distinct templates (vs student_v0), max template share | | `data/train/student_v1/manifest.json` | V4 |
| leakage check result | | | V4 |
| Unsloth typed-decisions validation (before → after) | | | V5 |
| arm A / B / C: selected step, dev macro, wall-clock, grad p95 / max | | | V6–V8 |
| G1 per arm (seen / held-out) | | `loops/student_v1/g1.md` | V9 |

---

## 3. Current task checklist

**Fill this in before starting any task**, by copying that task's concrete steps out of
the plan as unticked boxes. Tick each one the moment it is finished, not at the end.
This is the only record of progress *within* a task — without it, a compaction mid-task
leaves you unable to tell what you already did.

Clear this section and write the new task's checklist when you start the next task; the
completed checklist goes into the iteration-log entry.

```
Task in flight: none (V1 DONE 2026-10-08T05:28:00Z; V2 PENDING, not started)
Working dir:    outputs/student_v1/V1/

- [x] JEV-9B run on v0.2 test (fresh + tier1), detached, log in outputs/student_v1/V1/logs/
- [x] F7 -> harness converter (scripts/bench/jev_to_harness.py) + unit test
- [x] g1.py: verdict baselines zeroshot + jev9b by default; ablations only with --additional; unit test
- [x] JEV tagged as non-letter readout in g1 (D12 label-mass/greedy inapplicable, as for pointer)
- [x] G1 rerun for student_v0 step 1,000 -> loops/student_v1/g1_student_v0_rerun.md
- [x] SELF_AUDIT.md (R6), CLAIMS rows appended (V008–V018), re-derivation script scripts/bench/rederive_g1_headline.py
- [x] ruff green; pytest 294 collected, exit 0; iteration log; commit; push
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

### V0 — 2026-10-08T04:17:00Z — DONE — 2026-10-08T04:17:00Z
- What ran: `scripts/student/v1_snapshot.py` (main + `--env unsloth`), `scripts/student/v1_unsloth_load.py` (in `envs/unsloth/.venv`); `uv sync --project envs/unsloth` (123 packages).
- Output: `outputs/student_v1/V0/{snapshot,envs,unsloth_load}.json`, `SELF_AUDIT.md`.
- Headline: Unsloth 2026.10.2 (first release with `FastDecisionModel`) loads Qwen3.5-0.8B in bf16 with `load_in_4bit=False`: 344 bf16 weight tensors; 251 norm/head tensors fp32; `predict` probabilities finite, sums 1.0 / 0.9999 (V001–V007).
- Surprises: the loader keeps RMSNorm weights in fp32 (arm C precision differs from A/B on norms, see Deviations); `causal_conv1d` is not installed in either env, so the reference kernel is used; the main env has transformers 5.18.0 vs 5.17.0 in the Unsloth env.
- Next: V1 (JEV-9B on v0.2 + D16 G1 rerun). Needs an F7-to-harness row converter and a G1 change that restricts verdict baselines to zeroshot + jev9b.

### V1 — DONE — 2026-10-08T05:28:00Z
- What ran: `scripts/bench/run_jev_baseline.py` (detached, 2,927 s, 19 cells); `scripts/bench/jev_to_harness.py`; `scripts/bench/g1.py` (D16 baselines zeroshot + jev9b; `--out loops/student_v1/g1_student_v0_rerun.md`); `scripts/bench/rederive_g1_headline.py` (fresh-process check). Full pytest exit 0, ruff clean.
- Output: `outputs/student_v1/V1/{jev9b/, preds/preds_jev9b__test.jsonl, g1_student_v0_rerun.json, rederive_g1_headline.txt, SELF_AUDIT.md}`; committed aggregate `loops/student_v1/g1_student_v0_rerun.md`.
- Headline: student_v0 step 1,000, seen fresh templates (17,442 items): G1 PASS vs zero-shot 0.8B (macro +0.163, CI [0.153, 0.174]) and vs JEV-9B on the 15,911-item intersection (macro +0.044, CI [0.031, 0.055]). Held-out (6,326 items): FAIL vs both (vs zero-shot −0.018; vs JEV-9B −0.155). JEV-9B scored 35,097 of 37,289 test items (2,192 long openFDA items excluded by the 8,192-token cap). Claims V008–V018.
- Surprises: six allocator OOM warnings during the JEV run, none fatal; `jev9b_provenance.json` has a 1 s wall-clock bug (true time in `jev9b.json`); the strict-slice seen comparison vs JEV-9B fails (additional, CI lower −0.032).
- Next: V2 (padding-invariant pointer head; trained-checkpoint padding test; effect on student_v0 step 1,000). The JEV coverage gap is an operator question (Section 6).

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|

---

## 6. Questions for the operator

Anything you could not resolve without a human. Be specific enough to answer without
re-reading the run: state the ambiguity, the options, and which you would pick.

- **JEV-9B coverage on the 2,192 long openFDA test items (V1).** JEV-9B is unscored on 661 `fda_boxed_warning_noul_v1` (held-out), 596 `fda_class_choice_v2`, and 935 `fda_route_claim_noul_v1` items because they exceed the 8,192-token prompt cap. G1 is therefore computed on the intersection, and its vs-JEV direction on those items is unmeasured. Options: (a) accept the intersection as is (labelled); (b) re-run JEV-9B on those items with a memory-safe attention path (the single-item allocations reached 28 GB, so this needs engineering and a raised cap); (c) report them as NOT MEASURED for JEV. I would pick (a) for this loop and (b) only if the program needs JEV on the full set.

---

## 7. Deviations from the plan

- **V1 (JEV-9B via transformers + peft, not vLLM):** `adapter_vllm` head read through transformers and peft because vLLM is not installed here. The card's mean |Δp| 0.0008 between paths is unverified in this environment.
- **V1 (verbalizer readout is non-letter for D12):** JEV-9B's probabilities are a softmax over offered options only, so label mass is 1.0 by construction and the greedy-agreement check is inapplicable. Treated as non-letter (as the pointer head is), so `NO_VERBALIZER_READOUT_REASON` is used. D12 thresholds unchanged.
- **V1 (`g1.py --out` now required):** the old default `loops/student_v0/g1.md` was removed so an earlier loop's output cannot be overwritten.
- **V1 (JEV coverage gap, reported, not hidden):** the F7 protocol's 8,192-token prompt cap leaves 2,192 test items unscored by JEV-9B. G1's vs-JEV comparisons use the item intersection (seen 15,911 of 17,442; held-out 5,665 of 6,326), labelled in the report. The cap was not changed.
- **V1 (provenance file):** `jev9b_provenance.json` shows a 1 s wall clock and empty `models`; the true wall clock is in `jev9b.json`. Recorded in SELF_AUDIT caveat 3; not used as evidence.
- **V0 (norm precision, decision deferred to V8):** Unsloth's `FastDecisionModel` loads the 251 RMSNorm/LayerNorm tensors in fp32; arms A/B load them in bf16. ADVISORY §3 asks for matched bf16 arms. Unresolved; V8 will either cast the norms to bf16 to match or record the difference in the per-arm provenance table.

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
