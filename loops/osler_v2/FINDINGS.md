# osler_v2 — FINDINGS (closure)

Written 2026-10-18 after W8 and the W10 spot-check. Loop status: STOPPED — a hard stop for operator review. Every number has a row in
`loops/osler_v2/CLAIMS.md` (W001 to W064). Gate O3 detail: `gate_o3.md`. Results table: `results.md`. Proposals: `NEXT.md`.

## Summary

- **Gate O3 does not pass for Osler-4B** (CLAIMS W059). The gate code (`o2_verdict`) reports NOT MEASURED, because the knowledge guard
  against zero-shot pplx cannot be scored on 21 test items. Its measured status is a **failure**: the external panel's
  non-inferiority against zero-shot pplx-decider-flash-v1.2 has an accuracy lower bound of −0.0344 (floor −0.02; W056). A gate with
  a measured failure cannot pass, whatever the unmeasured check shows.
- **The MR verdict set passes against both baselines.** Against MedDecider-4B: accuracy +0.2634 [+0.2511, +0.2756], Brier −0.3080
  (W053). Against zero-shot pplx: +0.1577 [+0.1422, +0.1733], Brier −0.2428 (W054). The real-text part flips sign against MedDecider-4B
  (+0.2873, versus osler_v1's −0.0742; W060), so the real-text block that osler_v1 lacked does what it was meant to do.
- **The panel passes non-inferiority against MedDecider-4B only by 0.0001.** Accuracy −0.0038, lower bound −0.0199 against the −0.02
  floor (W055). That is a pass, but a thin one, and it is stated as such.
- **The knowledge guard passes where it can be measured.** Against zero-shot pplx on the 5,435 items it can score: +0.0018, lower bound
  −0.0063 (additional, W057). On the full set the guard is NOT MEASURED. Against zero-shot Qwen3.5-4B on the full 5,456 items: +0.0687,
  lower +0.0550 (additional, W058). The base retains its knowledge.
- **The base choice (D39) chose pplx-decider-flash-v1.2** on fast dev v2 (MR +0.038, general +0.005, knowledge +0.092; all lower bounds
  pass; W045). The C0 pair was run on the same data and steps (W043, W044).
- **Which ADVISORY §2 outcome applies: outcome 2** ("the real-text block is kept but the gate fails on the panel. The audit skill
  transfers; broad clinical data is the next lever."). C1 (real text) was kept against C0p (W049). The gate fails on the panel (W056).
  Outcome 5 (the base choice) is reported as context below. Outcomes 1, 3 and 4 do not apply.

## 1. What ran

| task | status | what | CLAIMS |
|---|---|---|---|
| W0 | DONE | environment, teacher and MIMIC checks; speed: Qwen 6,802 real tok/s (58.4 GB), pplx 6,469 (57.1 GB) | W001–W008 |
| W1 | DONE | MR suite v1 frozen (D35): 13,103 verdict and 6,191 reported-only test items; codiesp and mkurman screens NOT MEASURED (no dev); the too-good ICD probe 0.612 and zero overlap with B1 | W009–W016 |
| W2 | DONE | fixed share-alike replay: 74,477 rows (ARC, BoolQ, SNLI; CC BY-SA, training only); EHR-Rel score items (3,717 rows; 0–3 scale from the paper) | W017–W022, W031 |
| W3 | BLOCKED — not run (optional) | ICD prose v2 was not started in this loop (ADVISORY §4 optional; the GPU was used by W3b and W5–W8). No prose v2 template is in any block | — |
| W3b | BLOCKED | local Gemma-4 charts: pilot keep rate 89.5 % and 85.5 %, but a label-tracking phrase ("ruled out": 0.971 and 0.667 of gold-none charts, 0.000 of gold-code) remains after the one allowed prompt fix. C3 not built | W023–W029 |
| W4 | BLOCKED | MIMIC: the official files are absent (0 of 5 core files); never waited for; D37 forbids unofficial copies | W005 |
| W5 | DONE | fast dev v2 (11,323 unique items); knowledge dev v2 (4,000 official-train items; zero overlap with blocks and test); CodiEsp carve-out (74 of 493 cases, exactly 15 %); zero-shot knowledge (Qwen 0.6642, pplx 0.7728); five baselines on EHR-Rel test; pplx test-side scoring (MR, panel, knowledge) | W030–W041 |
| W6 | DONE | C0q (0.7849), C0p (0.8355), C1 (0.8602, kept), C2 (0.8560, not kept); base pplx; knowledge checks pass | W043–W051 |
| W7 | DONE | final Osler-4B on C1 (pplx backbone): 42,200 steps, 250,044 realised items, eight evaluations, 0 log mismatches; fast dev v2 0.8648 | W052 |
| W8 | DONE | scoring of 30,621 V10 items; Gate O3 (NOT PASS); additional rows; per-template and CodiEsp leakage diagnostics | W053–W063 |
| W9 | BLOCKED — not applicable | memorisation audit only with MIMIC in the final mix; it is not (W4) | — |
| W10 | DONE | this file, results.md, NEXT.md, the five-claim spot-check (all five match) | W064 |

## 2. Results (aggregates; full table in results.md)

- **Final model vs MedDecider-4B on the MR verdict set (13,103 test items).** Accuracy +0.2634 [+0.2511, +0.2756]; Brier −0.3080 (W053).
- **Per template (Osler-4B; MedDecider-4B; zero-shot pplx):** CodiEsp 0.8041; 0.1862; 0.3416. mkurman 0.4604; 0.3973; 0.5409. PMC 0.9791;
  0.7983; 0.8615. MR ICD v2 1.0000; 0.7166; 0.7762 (the too-good flag, below). Synthea indicator 0.9984; 0.8236. MR indicator 0.9688; 0.7083.
  MTSamples 0.8288; pplx 0.6847 (W061).
- **Panel (4,353).** vs MedDecider-4B: −0.0038 [−0.0199, +0.0112] (passes by 0.0001). vs zero-shot pplx: −0.0198 [−0.0344, −0.0060] (fails).
  vs zero-shot Qwen (not a verdict baseline; additional): +0.0700 [+0.0526, …] (W055–W056, additional).
- **Knowledge (test).** Guard vs zero-shot pplx: NOT MEASURED on 5,456 (21 skipped); +0.0018 on 5,435 (additional). vs Qwen: +0.0687 (W057–W058).

## 3. The §2 question, answered

- **Does real clinical text close the real-text ICD gap without costing knowledge or general skill?** On dev, yes: C1 over C0p gives MR
  +0.075 (lower +0.065), general +0.003 (lower −0.004, passes the −0.01 margin), knowledge +0.003 (lower −0.006) (W049). On the test
  set, the CodiEsp gap that osler_v1 lost to MedDecider-4B (−0.099) is now +0.618 (W061). A held-out CodiEsp carve-out scores 0.7825
  (C1) and 0.8150 (final), and the CodiEsp test has no text overlap with the training rows (W062): the gain generalises across cases,
  and it is not leakage. mkurman, which has no training rows at all, is +0.063 against MedDecider-4B and −0.080 against pplx.
- **Is pplx a better starting point than Qwen3.5-4B?** On dev, yes (D39: W045). On the panel, the zero-shot base is the baseline that
  matters: the final model passes non-inferiority to MedDecider-4B but fails it to zero-shot pplx. Against zero-shot Qwen the panel is
  +0.070 (additional). The base choice therefore decides which baseline the panel check is failed against, not whether the model is
  better than its own base.
- **Does teacher prose help?** No: C2 is not kept against C1 (MR −0.0042, lower −0.0121; general lower −0.0197) (W051). This matches
  osler_v1's B2 finding.
- **Replay at 20 %.** Met at C0 (22.65 %), C1 (21.04 %) and C2 (20.13 %) (W035). The 8 % per-template cap, not the source count, is the
  binding limit on the replay set.

## 4. Defects and cautions (ADVISORY §7 and the loop's own list)

1. **Too good: `mr_icd_support_template_v2` reads 1.0000** on its 554 test items and 1.0000 on the final model's 400 dev items (W063). The
   rule probe (0.612) and the B1 overlap (zero) do not explain it. It is in the verdict set; its removal would not change the MR
   direction. **Not cleared; open.**
2. **In-distribution generators are near ceiling** (synthea indicator 0.998; PMC 0.979). The MR gain is dominated by the in-distribution
   templates and by the PMC template (8,100 items), whose word-overlap AUC was 0.97 in osler_v1 (V151) and whose NB screen is 0.52
   (D35 does not exclude it). Without PMC the MR gain against MedDecider-4B is still +0.2799 (W060), so PMC is not the whole story.
3. **The panel pass against MedDecider-4B is 0.0001 wide** (W055).
4. **The W3b cue is semantic, not wording:** the certainty of the target diagnosis carries the answer in the discharge summary. One
   permitted prompt fix moved the cue into synonyms without removing it (W026–W027). Charts are BLOCKED for this reason.
5. **Unmeasured items:** 111 MR items and 212 panel items have no MedDecider-4B prediction in the V6/O2 files; 21 knowledge items cannot
   be scored by pplx's reference (duplicate option labels). These are reported with their counts, not dropped silently.
6. **Real defects found and fixed during the loop (recorded as deviations):** an EHR-Rel score key error (1..N schema), an EHR-Rel
   duplicate-pair error, a duplicated knowledge id, a pplx replay path bug, and a knowledge check that compared unequal item sets. Each
   superseded output is kept in a `superseded` folder.

## 5. Outcome 5 (the base) — context, not a verdict

The base choice was made on dev before any C1 run (D39), as the ADVISORY requires. On the test panel, the base choice changes the
baseline: pplx is the substitute for zero-shot Qwen in the verdict (D39), and the final model is non-inferior to MedDecider-4B but not to
zero-shot pplx. Zero-shot pplx is 0.0898 more accurate than zero-shot Qwen on the panel (derived from W055–W058), so a base with a
stronger zero-shot panel raises the bar the fine-tuned model must clear. The C0 run of the base not chosen (C0q) is dev-only; it was not
scored on the test sets.

## 6. Five-claim spot-check (W10; `outputs/osler_v2/W10/spot_check.json`)

Re-derived in a fresh process from the raw artifacts; all five match the recorded values:
1. W052 final fast dev v2 accuracy 0.8648 (recomputed 0.8648).
2. W045 base-choice MR lower bound 0.0283 (recomputed 0.0283).
3. W056 panel vs zero-shot pplx lower bound −0.0344 (recomputed −0.0344).
4. W035 C0 replay share 0.2265 (recomputed 0.2265).
5. W052 realised items 250,044 with 0 log mismatches (recomputed 250,044 / 0).

## 7. Not measured

- Osler-9B: NOT MEASURED — deferred by the operator (D39).
- d1-3B (an additional row): NOT MEASURED — not run in this loop.
- The MR verdict's codiesp and mkurman screens (D35): NOT MEASURED before W6. For codiesp, Q2 promised the screen at W5 and it was not
  run then. A post-hoc NB screen (labelled; W065) gives 0.636 on the carve-out dev, below the 0.90 rule, so the vocabulary rule would not
  have excluded it; the ceiling was not measured post hoc. mkurman has no dev items; its screens stay NOT MEASURED.
- MIMIC, MDACE, the memorisation audit: NOT MEASURED (absent; W4, W9).
- The chart templates, their NB screens and their leakage checks: NOT MEASURED (W3b BLOCKED).
