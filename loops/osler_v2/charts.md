# Whole synthetic charts (W3b) — pilot only, BLOCKED (osler_v2)

Aggregates only; no chart text is in this file. Written 2026-10-17 after the pilot.

**Status: BLOCKED.** The ADVISORY W3b acceptance requires that no label-tracking phrase is found in the pilot. After one allowed
prompt fix, `chart_icd_support_v1` still carries phrases that track its gold label (below). Per the ADVISORY, W3b is BLOCKED and the
loop continues without charts. C3 (charts) is therefore not run in this loop.

## Set-up (ADVISORY W3b step 1–3)

- vLLM 0.31.0 in `envs/vllm/` (own venv; torch 2.13 cu130; the Blackwell GPU is capability 12.0).
- Served: `google/gemma-4-31B-it-qat-w4a16-ct` with the speculative draft `google/gemma-4-31B-it-assistant`, the ADVISORY's flags,
  `VLLM_USE_RUST_FRONTEND=1`. The first launch failed because `ninja` was not on PATH (a set-up fix, not a model result); the second
  launch served.
- Teacher variables as ADVISORY W3b step 3 (`TEACHER_LOCAL=1`, local URL, cost $0 at the client).

## Pilot (200 encounters each run; ADVISORY W3b step 4)

| run | prompt change | charts kept (of 200) | kept share | items | charts per hour (wall) | local tokens (prompt / completion) |
|---|---|---|---|---|---|---|
| pilot 1 | none | 179 | 89.5 % | 718 | 1,334 | 1,353,596 / 380,112 |
| pilot 2 | one wording fix in `charts.py` (the discharge diagnosis instruction; see below) | 171 | 85.5 % | 686 | 1,358 | 1,356,723 / 377,935 |

Items by template (pilot 2): `chart_indicator_evidence_v1` 342 (yes and no balanced by construction), `chart_evidence_page_v1` 172,
`chart_icd_support_v1` 172.

Drop reasons (pilot 2): admission note 14, progress note 5, discharge summary 9, indicator twin did not flip 1.

The keep-rate part of the acceptance passes (≥ 50 %). The label-tracking part does not.

## Label-tracking check (ADVISORY W3b step 4; `scripts/osler_v2/w3b_pilot_checks.py`)

Narrative n-grams compared by gold class (ratio ≥ 3, ≥ 10 charts, share difference ≥ 0.10):

| run | template | phrase | share of gold-code charts | share of gold-none charts |
|---|---|---|---|---|
| pilot 1 | chart_icd_support_v1 | "ruled out" | 0.000 | 0.971 |
| pilot 1 | chart_icd_support_v1 | "diagnosis confirmed" | 0.483 | 0.000 |
| pilot 2 | chart_icd_support_v1 | "ruled out" | 0.000 | 0.667 |
| pilot 2 | chart_icd_support_v1 | "does have" / "determined does have" | 0.000 | 0.636 / 0.576 |
| pilot 2 | chart_icd_support_v1 | "diagnosis confirmed" | 0.223 | 0.000 |
| pilot 2 | chart_icd_support_v1 | "sex female" (demographic) | 0.108 | 0.333 |

The indicator and evidence-page templates show no flagged phrase in either run.

**Reading.** The phrases are the certainty of the target diagnosis. A discharge summary that says the diagnosis was ruled out (or
confirmed) carries the answer, so the template measures reading a certainty word. The one prompt fix moved the coupling into
synonyms ("does have", "determined does not") instead of removing it. The sex cue is a cohort difference between gold-none and gold-code
charts that a prompt cannot fix. This is the osler_v1 failure mode: a repeated phrase that went with one gold label (osler_v1 found "documented baseline" with gold "no").

## Other checks

- Stated numbers: 157 of 3,017 narrative pages (pilot 2) contain a medication volume ("10 mL", "0.4 mL"), and 15 contain a decimal.
  None is a vital sign or a laboratory value as the prompt defines them; recorded, not a failure.
- Discharge summary naming the gold condition (substring check, pilot 2): gold-code charts 74 of 139; gold-none 0 of 33 (by design:
  none names no condition). The check is a substring match and may undercount.
- Read-through: 10 charts sampled at random (`scripts/osler_v2/w3b_pilot_checks.py --show 10`) and printed for a fresh-process read;
  the narrative text is fluent and consistent with the case facts in the samples read. Not committed.

## Not measured

- The O3 Naive Bayes screen and the leakage check of the chart templates: NOT MEASURED (the build was stopped at the pilot; no
  dev split of a full build exists).
- Build to 20,000 encounters and the 12-hour budget: NOT RUN (BLOCKED at the pilot).
