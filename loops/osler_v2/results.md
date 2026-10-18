# osler_v2 — results (aggregates; every number has a CLAIMS row)

Written 2026-10-18. Gate O3 detail: `gate_o3.md`. Findings and the §2 outcome: `FINDINGS.md`. Blocks: `blocks.md`. Baselines: `baselines_v2.md`. MR suite: `mr_suite_v1.md`.

## Gate O3 (Osler-4B; 9B NOT MEASURED, deferred by the operator)

| check | result | rule | verdict contribution |
|---|---|---|---|
| MR suite v1 (13,103 verdict items) vs MedDecider-4B | accuracy +0.2634 [+0.2511, +0.2756]; Brier −0.3080 | accuracy lower > 0; Brier upper < 0 | pass (W053) |
| MR suite v1 vs zero-shot pplx-decider-flash-v1.2 | accuracy +0.1577 [+0.1422, +0.1733]; Brier −0.2428 | same | pass (W054) |
| external panel vs MedDecider-4B (4,141 paired) | accuracy −0.0038 [−0.0199, +0.0112] | lower > −0.02 | pass by 0.0001 (W055) |
| external panel vs zero-shot pplx (4,353) | accuracy −0.0198 [−0.0344, −0.0060] | lower > −0.02 | FAIL (W056) |
| knowledge guard vs zero-shot pplx (5,456) | NOT MEASURED: 21 items with repeated option labels | lower > −0.02 | not measured (W057) |
| **Gate O3 (osler_v2 code verdict)** | **NOT MEASURED; a measured failure stands; gate passes: false** | | (W059) |

## Additional rows (never in the verdict)

| row | result | CLAIMS |
|---|---|---|
| MR verdict, real-text part vs MedDecider-4B (11,244) | +0.2873 [+0.2744, +0.3003] | W060 |
| MR verdict, generated part vs MedDecider-4B (1,748) | +0.2395 [+0.2194, +0.2595] | W060 |
| MR verdict without PMC vs MedDecider-4B (4,892 paired) | +0.2799 [+0.2652, +0.2941] | W060 |
| MR reported-only templates vs zero-shot pplx (6,191) | measured; vs MedDecider-4B NOT MEASURED (no paired items) | W053–W054 (gate JSON) |
| held-out MR test (584; 70 with MedDecider-4B) | measured; see gate JSON | gate_o3.json |
| knowledge guard vs zero-shot pplx, 5,435 paired items | +0.0018 [−0.0063, +0.0110]; passes | W057 |
| knowledge guard vs zero-shot Qwen3.5-4B (5,456) | +0.0687 [+0.0550, +0.0812]; passes | W058 |
| panel vs zero-shot Qwen3.5-4B (4,353; base not chosen) | +0.0700 [+0.0526, …] | gate JSON |
| osler_v1's Osler-4B on the MR verdict vs MedDecider-4B | +0.0805 (osler_v1's own gate figure) | gate JSON |
| d1-3B | NOT MEASURED (not run in this loop) | — |

## Per template (MR verdict test items; Osler-4B; MedDecider-4B; zero-shot pplx)

| template | n | Osler-4B | MedDecider-4B | zero-shot pplx | CLAIMS |
|---|---|---|---|---|---|
| codiesp_icd_support_v1 | 1,654 | 0.8041 | 0.1862 | 0.3416 | W061 |
| mkurman_icd_support_v1 | 1,490 | 0.4604 | 0.3973 | 0.5409 | W061 |
| pmc_similar_patient_v1 | 8,100 | 0.9791 | 0.7983 | 0.8615 | W061 |
| mtsamples_page_category_v1 | 111 | 0.8288 | — | 0.6847 | W061 |
| mr_icd_support_template_v2 (too-good flag) | 554 | 1.0000 | 0.7166 | 0.7762 | W061, W063 |
| mr_indicator_evidence_v1 | 576 | 0.9688 | 0.7083 | 0.7569 | W061 |
| synthea_indicator_evidence_v1 | 618 | 0.9984 | 0.8236 | 0.9741 | W061 |

## Fast dev v2 (training-time dev; full 11,323 items)

| run | accuracy | template macro | Brier | CLAIMS |
|---|---|---|---|---|
| C0q (Qwen3.5-4B; B1 + replay) | 0.7849 | 0.8589 | 0.3136 | W043 |
| C0p (pplx-flash; B1 + replay) | 0.8355 | 0.8702 | 0.2556 | W044 |
| C1 (pplx; + real text) — kept | 0.8602 | 0.8880 | 0.2025 | W048 |
| C2 (pplx; + prose) — not kept | 0.8560 | 0.8783 | 0.2122 | W050 |
| final Osler-4B (C1; 42,200 steps; 250,044 items) | 0.8648 | 0.8985 | 0.2002 | W052 |

## Base choice (D39) and keep decisions (D34)

| decision | result | CLAIMS |
|---|---|---|
| base: pplx vs Qwen (C0p − C0q): MR, general, knowledge | +0.0378 (lower +0.0283); +0.0045 (lower −0.0063); +0.0918 (lower +0.0793) → pplx | W045 |
| C0 knowledge checks | C0q +0.0203 (lower +0.0078, passes); C0p −0.0013 (lower −0.0095, passes) | W046–W047 |
| C1 vs C0p | MR +0.0750 (lower +0.0645); general +0.0032 (lower −0.0043); knowledge +0.0027 (lower −0.0056) → KEPT | W049 |
| C2 vs C1 | MR −0.0042 (lower −0.0121); general −0.0112 (lower −0.0197); knowledge −0.0051 (lower −0.0131) → NOT KEPT | W051 |

## Baselines on the EHR-Rel test (new template; 373 items; chance 0.25)

| model | accuracy | mean abs. level error | CLAIMS |
|---|---|---|---|
| MedDecider-4B | 0.5442 | 1.1765 | W039 |
| zero-shot Qwen3.5-9B | 0.5067 | 1.4323 | W039 |
| MedDecider-9B | 0.4906 | 1.3926 | W039 |
| zero-shot Qwen3.5-4B | 0.3995 | 1.5243 | W039 |
| JEV-9B | 0.2386 | 1.6273 | W039 |

## Knowledge dev v2 (zero-shot; 3,988 items after the 12 repeated-label exclusions)

| model | accuracy | readout | CLAIMS |
|---|---|---|---|
| zero-shot Qwen3.5-4B | 0.6642 | D12 PASS | W037 |
| zero-shot pplx-decider-flash-v1.2 | 0.7728 | reference readout | W038 |

## Speed, replay and data

| item | value | CLAIMS |
|---|---|---|
| training speed (300 real batches): Qwen / pplx | 6,802 / 6,469 real tokens/s; peak 58.4 / 57.1 GB | W007–W008 |
| replay: new share-alike rows; C0 / C1 / C2 replay share | 74,477; 22.65 % / 21.04 % / 20.13 % | W017, W035 |
| EHR-Rel rows (0–3 scale; train / dev / test) | 3,717 (2,942 / 402 / 373) | W031 |
| CodiEsp carve-out | 74 of 493 cases (15.01 %); 400 dev rows | W032 |
| knowledge dev v2 | 4,000 items, zero overlap with blocks or test | W030 |

## Not measured or blocked

- W3 (optional, ICD prose v2): BLOCKED — not run. W3b (charts): BLOCKED on a label-tracking cue (W026–W029). W4 (MIMIC): BLOCKED, files absent. W9 (memorisation audit): BLOCKED — no MIMIC in the final mix.
- Osler-9B: NOT MEASURED (deferred by the operator, D39). d1-3B: NOT MEASURED (not run).
- The codiesp ceiling and the mkurman screens (D35): NOT MEASURED (see FINDINGS §7; the codiesp NB screen was run post hoc, W065).
