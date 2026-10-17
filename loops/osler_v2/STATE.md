# osler_v2 — RUN STATE

> **This file is the loop's memory.** The live copy is `loops/osler_v2/STATE.md`
> (committed, so the operator can review it remotely). Read it at the start of every
> iteration, before the plan. If it disagrees with your recollection, **the file
> wins** — your context may have been compacted since the last iteration.
>
> **How to update:** set the task `IN_PROGRESS` with a UTC start time *before* working;
> on completion set `DONE` or `BLOCKED — <reason>`, fill the finished time, and append a
> ≤5-line entry to the iteration log. Never delete a log entry; append only. Timestamps
> are `date -u +%FT%TZ`. Never paste item text, predictions, or secrets into this file.

Loop status: `RUNNING`  <!-- set to STOPPED at the hard stop (W10), or when no PENDING task can proceed without the operator -->
Run started (UTC): `2026-10-17T13:34:35Z`
Last updated (UTC): `2026-10-17T14:06:18Z`
Iterations so far: `1`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| W0 | Orientation | no | — | DONE | 2026-10-17T13:34:35Z | 2026-10-17T14:06:18Z |
| W1 | Audits; MR suite v1 frozen | small | W0 | DONE | 2026-10-17T14:07:22Z | 2026-10-17T14:13:48Z |
| W2 | Replay ≥ 20 %; EHR-Rel | no | W0 | DONE (C3 replay caveat, Q4) | 2026-10-17T14:14:14Z | 2026-10-17T14:19:31Z |
| W3 | ICD prose v2 (optional) | no | W0 | PENDING | | |
| W3b | Whole synthetic charts, local Gemma 4 teacher | yes | W0 | BLOCKED — pilot label-tracking phrase remains after the one allowed prompt fix (chart_icd_support_v1); C3 not built | 2026-10-17T14:19:50Z | 2026-10-17T14:56:32Z |
| W4 | MIMIC (official files only; only when they exist) | no | W0 | BLOCKED — official MIMIC files absent (W0 probe; re-check at W6 and before W7) | 2026-10-17T14:14:08Z | 2026-10-17T14:14:08Z |
| W5 | Fast dev v2, knowledge dev v2, new test templates and baselines | yes | W1, W2, W3b (W3, W4 if built) | PENDING | | |
| W6 | Blocks v2 + keep rule | yes | W5 | PENDING | | |
| W7 | Final Osler-4B (9B deferred, D39) | yes | W6 | PENDING | | |
| W8 | Evaluation + Gate O3 (Osler-4B; 9B NOT MEASURED) | yes | W7 | PENDING | | |
| W9 | Memorisation audit (only with MIMIC) | small | W7 | PENDING | | |
| W10 | Findings and closure — HARD STOP | no | all | PENDING | | |

Rules: take the **first** `PENDING` task whose deps are all `DONE` (exceptions in ADVISORY §5). Never run two GPU jobs at
once. A `BLOCKED` task does not block unrelated work. No row may be left `IN_PROGRESS` at closure.

---

## 2. Key values discovered during the run

| key | value | source | task |
|---|---|---|---|
| teacher allowance / DeepSeek balance; MIMIC files | OpenCode Go reachable (HTTP 200, 37 ids, chat OK; 5-hour and weekly allowances NOT MEASURED, not exposed to the client); DeepSeek balance USD 9.88 (loop cap $9.00); MIMIC core 0 of 5 files | `loops/osler_v2/CLAIMS.md` W003–W005; `outputs/osler_v2/W0/env_probe.json` | W0 |
| too-good ICD probe result; NB screen per template; ceiling statistic; MR suite v1 verdict templates | Probe 0.612 (339/554); overlap with B1 0 exact / 0 near-dup; NB dev macro 1.000 (page tabular), 0.796 (indicator), 0.804 (ICD), 1.000 (synthea page), 0.576 (synthea indicator), 0.590 (MTSamples), 0.520 (PMC); ceiling 1.000 / 0.695 / 0.710 / 0.995 / 0.788 / 0.629 / 0.780; verdict 13,103 test items, reported only 6,191; codiesp and mkurman screens NOT MEASURED (no dev) | `loops/osler_v2/mr_suite_v1.md`; `outputs/osler_v2/W1/screens.json`; `icd_audit.json`; CLAIMS W009–W016 | W1 |
| replay share of the largest block; sources and licences; EHR-Rel items | Fixed replay 74,477 rows (ARC 3,370; BoolQ 9,427; SNLI 61,680; CC BY-SA, training only). Shares: B1 14.38 %, C0 22.65 %, C1 21.10 % and C2 20.33 % (upper bounds); C3 keeps 20 % only if its chart rows are ≤ 14,092. EHR-Rel 3,741 rows, 0–3 scale (paper Table 1), split 2,961 / 404 / 376 | `loops/osler_v2/CLAIMS.md` W017–W022; `outputs/osler_v2/W2/replay_manifest.json`, `ehr_rel_manifest.json` | W2 |
| ICD prose v2 pilot pass rate; pairs built; spend | | | W3 |
| charts: pilot keep rate, charts/h, items per template, NB screens | Pilot 1: 179/200 kept, 1,334 charts/h; pilot 2 (after one wording fix): 171/200 kept, 1,358 charts/h; label-tracking phrase remains in chart_icd_support_v1 ("ruled out" 0.667 vs 0.000) → BLOCKED; NB screens NOT MEASURED | `loops/osler_v2/charts.md`; CLAIMS W023–W029 | W3b |
| fast dev v2 partitions; knowledge dev v2 size; zero-shot knowledge dev | | | W5 |
| speed check: real tokens/s and peak per backbone | Qwen 6,802 tok/s, peak 58.4 GB; pplx-flash 6,469 tok/s, peak 57.1 GB (both pass: ≥ 5,500, ≤ 85 GB) | `loops/osler_v2/CLAIMS.md` W007–W008; `outputs/osler_v2/W0/speed_summary` logs | W0 |
| base choice (C0p − C0q, three partitions) and the chosen base | | `loops/osler_v2/blocks.md` | W6 |
| kept blocks and dev deltas | | `loops/osler_v2/blocks.md` | W6 |
| Osler-4B: base, steps, selected step, dev template macro | | | W7 |
| Gate O3 verdict (4B; 9B NOT MEASURED) | | `loops/osler_v2/gate_o3.md` | W8 |

---

## 3. Current task checklist

**Fill this in before starting any task**, by copying that task's concrete steps out of
the plan as unticked boxes. Tick each one the moment it is finished, not at the end.
This is the only record of progress *within* a task — without it, a compaction mid-task
leaves you unable to tell what you already did.

Clear this section and write the new task's checklist when you start the next task; the
completed checklist goes into the iteration-log entry.

```
Task in flight: none (W3b BLOCKED at 2026-10-17T14:56:32Z)
Working dir:    outputs/osler_v2/W3b/

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

### W0 — DONE — 2026-10-17T14:06:18Z
- What ran: `scripts/osler_v2/w0_speed.py --steps 300 --dyn 8192` (Qwen and `--pplx`); `scripts/osler_v2/w0_env_probe.py`; ruff; full pytest gate with the padding strict-xfail deselected on the command line only.
- Output: `outputs/osler_v2/W0/` (speed logs, `env_probe.json`, `pytest_gate.log`, `ruff.log`, `SELF_AUDIT.md`).
- Headline: speed Qwen 6,802 real tok/s (peak 58.4 GB), pplx-flash 6,469 (peak 57.1 GB); both pass the 5,500 / 85 GB rule (CLAIMS W007–W008). Pytest gate: 730 passed, 1 skipped, 1 deselected, exit 0 (W002).
- Surprises: (1) first Qwen speed attempt hit CUDA OOM because my own pytest run (started with the wrong deselect) held about 89 GB; rerun after it exited. (2) that pytest run failed the padding test, which is deselected by osler_v1 practice and was not edited. (3) non-fatal allocator warnings in both speed logs. (4) MIMIC files absent (expected).
- Next: W1 (CPU audits) and W2 (replay and EHR-Rel) are unblocked.

### W1 — DONE — 2026-10-17T14:13:48Z
- What ran: `scripts/osler/o2_run_zeroshot.py` (Qwen3.5-4B, mr_fastdev, 2,397 rows, GPU, 4 min); `scripts/osler_v2/w1_screens.py`; `scripts/osler_v2/w1_icd_audit.py`; `scripts/osler_v2/w1_freeze_suite.py`.
- Output: `outputs/osler_v2/W1/` (screens.json, icd_audit.json, zeroshot/, SELF_AUDIT.md); `data/bench/osler_mr_v1/manifest.json`; `loops/osler_v2/mr_suite_v1.md` (committed).
- Headline: MR suite v1 frozen (D35): 13,103 of 19,294 test items in the verdict; the two page-category templates reported only (ceiling 1.000 and 0.995; NB 1.000). codiesp and mkurman screens NOT MEASURED (no dev items). ICD too-good probe 0.612 with zero training overlap (CLAIMS W015–W016), so it is still unexplained.
- Surprises: (1) the ceiling statistic is on the 400-item fast-dev sample (D-W1-1); (2) a one-item tie in the synthea ceiling (decision unchanged); (3) pmc passes D35 but its word-overlap AUC is high (Q3).
- Next: W2 (replay ≥ 20 %, EHR-Rel) and W3/W4 status; W3b needs the vLLM environment (GPU, ~1 h set-up).

---

### W2 — DONE — 2026-10-17T14:19:31Z
- What ran: `scripts/osler_v2/w2_replay.py` (ARC, BoolQ, SNLI train splits; leakage check against the osler_v0 protected sets; SNLI capped); `scripts/osler_v2/w2_ehr_rel.py` (EHR-Rel TSVs from the repository); unit tests `tests/test_replay_v2.py`, `tests/test_ehr_rel.py` (26 passed).
- Output: `data/train/osler_v2/replay_v2_train.jsonl`, `data/train/osler_v2/ehr_rel_train.jsonl`, `data/gen/osler_v2/ehr_rel/` (gitignored); `outputs/osler_v2/W2/` (manifests, SELF_AUDIT.md).
- Headline: fixed replay of 74,477 new rows (CC BY-SA, training only) on top of B1's 14.38 % replay: C0 22.65 %, C1 21.10 % and C2 20.33 % (upper bounds); C3 keeps ≥ 20 % only if its chart rows are ≤ 14,092 (Q4). EHR-Rel: 3,741 score rows, 0–3 scale, split 2,961 / 404 / 376.
- Surprises: (1) the 8 % per-template cap, not the source count, limits the replay (SNLI); (2) OpenBookQA has no stated licence; (3) EHR-Rel's hash split is 79/11/10.
- Next: W3b (local Gemma charts, GPU) and W5 (dev assembly) — W5 needs W3b first.

---

### W3b — BLOCKED — 2026-10-17T14:56:32Z
- What ran: vLLM 0.31.0 install in `envs/vllm/`; `google/gemma-4-31B-it-qat-w4a16-ct` served with the draft (ADVISORY flags); `scripts/osler_v2/w3b_charts.py --count 200` twice (pilot 1; pilot 2 after one prompt fix); `scripts/osler_v2/w3b_pilot_checks.py` (phrase, number and discharge checks; 10-chart read-through). Server stopped afterwards.
- Output: `loops/osler_v2/charts.md`; `outputs/osler_v2/W3b/` (serve and pilot logs, checks JSON, read-through text, not committed); `data/gen/osler_v2_charts_pilot*` (gitignored).
- Headline: keep rate 89.5 % (pilot 1) and 85.5 % (pilot 2) of 200 encounters; 1,334–1,358 charts per hour. Acceptance FAILS on label-tracking: "ruled out" is in 0.971 (pilot 1) and 0.667 (pilot 2) of gold-none ICD charts and in 0.000 of gold-code charts (CLAIMS W026–W027). BLOCKED.
- Surprises: (1) the label cue is the certainty word of the target diagnosis; (2) a patient-sex cohort difference between gold classes; (3) the first vLLM launch failed on a missing `ninja` on PATH.
- Next: C3 is not built; W5 proceeds without the chart templates (ADVISORY W5: "if W3b built them"); Q5 for the operator.

---

## 5. Blocked items

| id | what is blocked | exact reason | what would unblock it |
|---|---|---|---|
| W3b | Chart templates (C3, W5 chart dev items, chart NB screens) | the pilot's `chart_icd_support_v1` carries a label-tracking phrase ('ruled out', 0.667 vs 0.000) after one allowed prompt fix (CLAIMS W026–W027) | the operator's Q5 decision: a redesign of certainty wording (new loop) or a two-template C3 (deviation). |
| W4 | MIMIC block (C4) and MIMIC in W5 dev | official PhysioNet files are not under `data/physionet/` (env probe, 2026-10-17T14:14:08Z: 0 of 5 core files). D37 forbids unofficial copies. | The operator's PhysioNet approval and the official files under `data/physionet/`. Re-checked at W6 and before W7; never waited for. |

---

## 6. Questions for the operator

Anything you could not resolve without a human. Be specific enough to answer without
re-reading the run: state the ambiguity, the options, and which you would pick.

---

**Q1 (W1, 2026-10-17T14:13:48Z) — mkurman_icd_support_v1 has no dev split and can never be screened on dev.** It is evaluation-only (osler_v1 V5b), so the D35 rules cannot trigger for it, and it stays in the verdict (13,103 verdict test items include its 1,490). Options: (a) keep it in the verdict with its screens NOT MEASURED (current; the literal D35 text); (b) report it only, which removes 1,490 test items where osler_v1's loss sat (V185), and would look like cherry-picking unless the operator decides it. I would keep (a). Please confirm before W6.

**Q2 (W1, 2026-10-17T14:13:48Z) — codiesp_icd_support_v1's screen is measured at W5, after the CodiEsp dev carve-out.** I will apply D35 to its dev items at W5 and record the result before any osler_v2 block (W6). If the operator wants a different rule for this, say so before W5.

**Q3 (W1, 2026-10-17T14:13:48Z) — pmc_similar_patient_v1 (8,100 test items, 42 % of the test items) passes D35 (NB 0.520), but osler_v1 measured word-overlap AUC 0.97 (V151).** The AUC is not a D35 criterion, so the template stays in the verdict; I will report the AUC as an additional row. Confirm that this is the right reading.

**Q4 (W2, 2026-10-17T14:19:31Z) — the replay target for the largest block (ADVISORY W2: replay ≥ 20 % of the largest block).** With the 8 % per-template cap and the listed sources, the fixed replay set (74,477 rows) gives 22.65 % at C0, 21.10 % at C1 and 20.33 % at C2 (upper bounds). C3 keeps 20 % only if its chart rows are at most 14,092. Options: (a) keep the listed sources and let the W6 guard decide C3 (current; the ADVISORY's rule is the guard, so C3 may fail it); (b) add another permissive source (for example CosmosQA, SocialIQA or PIQA, all CC BY 4.0 or AFL-3.0; not in the ADVISORY list), which is a new source and needs its own leakage check; (c) cap the charts. I would take (a) and record it. Please say if you want (b).

  *Update (2026-10-17T14:56:32Z): W3b is BLOCKED, so no chart rows exist. The largest block is C2 (C1 plus prose), and its replay share is 20.33 % (upper bound; recompute at W6 with the EHR-Rel rows). Q4 is closed for this loop.*

**Q5 (W3b, 2026-10-17T14:56:32Z) — the chart templates are BLOCKED on a label-tracking cue.** The pilot kept 85–90 % of charts, but chart_icd_support_v1's discharge diagnosis carries the answer in its certainty word ("ruled out" in 67–97 % of gold-none charts, never in gold-code charts). One allowed prompt fix did not remove it. Options: (a) keep W3b BLOCKED and run this loop without charts (current; C3 not run); (b) build only chart_indicator_evidence_v1 and chart_evidence_page_v1, which passed the phrase check, as a C3 with two templates (a change to the ADVISORY's "all three templates"); (c) redesign the certainty wording in a new loop. I would take (a) now and (c) later. Please say if you want (b) before W6.

## 7. Deviations from the plan

Any place you departed from GOAL/ADVISORY, with the reason. An empty section is the
expected outcome. Editing code or a check to make it pass is never an acceptable
deviation — that is a `BLOCKED`.

---

- **D-W1-1 (W1, 2026-10-17T14:13:48Z):** the ceiling statistic uses each template's 400-item fast-dev sample (the `mr_fastdev` file), not its full dev split: the zero-shot runner reads the bench item schema and the full dev splits are in the generator schema. The NB screens use the full dev splits. The decisions do not depend on it (page-category templates are excluded by NB; the others are 0.69–0.79 on 400 items, far below 0.97). Recorded in `outputs/osler_v2/W1/SELF_AUDIT.md`.
- **D-W1-2 (W1, 2026-10-17T14:13:48Z):** codiesp_icd_support_v1 and mkurman_icd_support_v1 have no dev items, so D35's two rules cannot trigger: both stay in the verdict with their screens NOT MEASURED (the literal D35 text excludes only ceiling and vocabulary-shortcut templates). See question Q1.
- **Tie (W1):** one zero-shot item (synthea_page, 6956ca9e6ee8d55b) has two options with equal maximum probability; the repo's `np.argmax` convention (first index) is used. The decision is unchanged either way (see SELF_AUDIT).

- **D-W2-1 (W2, 2026-10-17T14:19:31Z):** EHR-Rel's split by hash gives 79.2 / 10.8 / 10.1 % (train / dev / test), not exactly 80/10/10 (the rule is the stable hash).
- **D-W2-2 (W2, 2026-10-17T14:19:31Z):** the EHR-Rel rows were not run through the osler_v0 protected-set leakage check (the ADVISORY asks for it on the replay sources). Concept-name pairs are a different source; add it in W5 if wanted.
- **D-W2-3 (W2, 2026-10-17T14:19:31Z):** OpenBookQA not used (no licence on its Hub card; ADVISORY keeps only verifiable licences).
- **D-W2-4 (W2, 2026-10-17T14:19:31Z):** the SNLI sample is capped at 61,680 rows by the 8 % per-template cap at C0 (the fixed set is one set for all blocks). Q4 covers C3.

- **D-W3b-1 (W3b, 2026-10-17T14:56:32Z):** `ninja` was not on PATH for the first vLLM launch (server crash, log kept as `serve_attempt1_ninja_missing.log`); the relaunch added the venv's bin directory to PATH. No model setting changed.
- **D-W3b-2 (W3b, 2026-10-17T14:56:32Z):** one prompt fix in `src/meddecide/mr/charts.py` (the discharge diagnosis instruction in `_narrative`: vary the wording, do not use the labels as headings). It was made after pilot 1 and before pilot 2, as the ADVISORY allows. It did not remove the cue; pilot 2 is the reported pilot. The verifier was not changed.

## 8. Closure summary feed

<!-- Filled at closure, feeding FINDINGS.md's Summary section. One row per major
     outcome, in the order a human should hear them. Each claim cites its CLAIMS id;
     failures and blocked tasks get rows too. -->

| # | outcome (one sentence, plain language) | claims | artifact |
|---|---|---|---|
| | | | |

