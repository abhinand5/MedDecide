# osler_v3 — X4 charts (aggregates only; no item text, no phrases)

Written at the X4 closure (2026-10-18, 23:31–23:40 UTC). Generator: osler_v2's `scripts/osler_v2/w3b_charts.py`, unchanged. Label-tracking
check: `scripts/osler_v3/x4_charts_check.py` (the D44(b) rule; the osler_v2 phrase test with its thresholds). Every number below is a
row of `CLAIMS.md` (C057–C071), with its recompute command there. Raw charts are in `data/gen/osler_v3/charts/` (gitignored).

## Outcome

| template | status | why |
|---|---|---|
| `chart_indicator_evidence_v1` (yes/no) | **usable** | no label-tracking phrase flagged (C060); NB 0.6758 on dev, below the 0.90 screen (C063) |
| `chart_icd_support_v1` (choice of codes) | **BLOCKED** | label-tracking phrases beyond the labelled fact, under the extended and the original exemption list (C061; D-X4-2) |
| `chart_evidence_page_v1` (choice of pages) | **BLOCKED** | label-tracking phrases beyond the labelled fact; no exemption applies (C062); the page-section vocabulary is not the labelled fact (as X3, Q4) |

Only the indicator template enters fast dev v3, the new test pool and block B3 (C067).

## Build

- Pilot (osler_v2's rules, 200 charts): 200 attempted, 174 kept, 698 items (C057).
- Build (1,800 charts): 1,800 attempted, 1,541 kept, 6,164 items; rejections by the generator's recorded reason: admission 155,
  progress note 29, discharge summary 74, consult note 1 (C057).
- Total: 2,000 charts attempted, 1,715 kept (85.8 %), 6,862 items (C057).

## Items by split (before the block exclusion)

| template | train | dev | test |
|---|---|---|---|
| chart_indicator_evidence_v1 | 2,916 | 330 | 184 |
| chart_icd_support_v1 | 1,459 | 165 | 92 |
| chart_evidence_page_v1 | 1,459 | 165 | 92 |
| total | 5,834 | 660 | 368 |

(C058, C059, C071.) The indicator train rows are balanced yes 1,458 / no 1,458 (C069).

## Label tracking

- The check compares each gold group with the rest (ratio at least 3, a difference of at least 0.10, at least 10 items per group).
  The record lists at most 30 flagged and 30 exempt candidates per template; the decision uses whether any flagged candidate remains,
  so the cap does not change a verdict.
- Indicator: none flagged (C060).
- ICD: the groups are "none" (319 items) and "code" (1,305 items) (C071). The flagged candidates track the "none" group. The verdict is
  the same under the extended exemption list (30 exempt) and under the original list (8 exempt), both with flagged candidates (C061).
- Evidence pages: the groups are the nine page classes (C071). Class J (2 items) is below the minimum group size and is not compared;
  class I (20 items) is compared, and the flagged candidates track class I. No exemption applies (C062).

## Naive Bayes screen (O3, fitted on train, scored on dev)

| template | dev macro accuracy | reference |
|---|---|---|
| indicator | 0.6758 | 2 classes, chance 0.5; below the 0.90 screen (C063) |
| ICD | 0.1069 | a uniform guess over the offered options averages 0.3048 |
| evidence pages | 0.2259 | a uniform guess over the offered options averages 0.1359 |

The ICD and page values are not a verdict input. They come from a screen that scores each option by its text; the answer depends on
which options are offered, and the screen ignores them. On dev the screen predicts one class for most items (C064), and no dev chart
state is shared with another dev item. Every dev gold option text occurs as a gold text in train, and no prediction falls outside the
offered options (C065), so the low value is a formulation limit, not a leak. The limit is recorded as D-X4-5.

## Leakage and stated numbers

- Leakage: zero hits against every protected set, in train, dev and test (C066).
- Stated numbers (narrative pages that state a unit value; the prompt forbids it), counted only: indicator 1,029 of 15,227 pages; ICD
  516 of 7,619; evidence pages 505 of 7,603 (C068). No block rule applies to them; the question is open (Q6).

## Mix files

Written by `scripts/osler_v3/x4_to_mix.py`, with the blocked templates excluded and counted (C067): train 2,916 (2,918 excluded),
dev 330 (330), test 184 (184). The indicator template is the only chart source in fast dev v3 (330 items), the new test pool (184 items)
and block B3.

## First-500 re-check

The ADVISORY asks for the label-tracking check on the first 500 built charts. The rows carry no build order, and the first files hold
fewer than 500 distinct chart keys, so the selection equals the full check (C070; D-X4-3). The full check covers every built chart,
including the first 500.
