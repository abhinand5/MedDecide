# ADVISORY — bench_v0_fix0: repair the benchmark and harness defects found in review, then re-measure

**Read `AGENTS.md` first** (repo root). It carries the evidence rules (R1–R10),
guardrails, and loop mechanics. This document is the work plan. Read it in full at the
start of every loop iteration.

**Status:** authored 2026-10-06 by the advisor after reviewing loop `bench_v0`
(operator approved). It supersedes nothing; `loops/bench_v0/` is retained read-only as
evidence. This loop continues on branch `loop/bench_v0`.

---

## 1. Where things stand

Loop `bench_v0` (artifacts in `loops/bench_v0/`, read its `FINDINGS.md` first) built
MedDecide-Bench v0 (21,202 tier-1 + 41,502 fresh items), an eval harness, and 264,478
zero-shot predictions over six ladder models. Its infrastructure is sound: integrity
audits pass (C014, C021), the fresh build is deterministic (C022), and the
**multiple-choice** readout agrees with lm-evaluation-harness within 0.5 points on
MedQA (C041).

The advisor review found defects that invalidate part of its findings. Each was verified
against raw data on the pod on 2026-10-06:

1. **MedMCQA answer key is off by one.** `src/meddecide/bench/tier1/mcq.py::_medmcqa_item`
   computes `gold_index = int(cop) - 1`, but `openlifescienceai/medmcqa` stores `cop` as
   **0-based** (validation split: `cop` counts 0→1,348, 1→1,085, 2→925, 3→825; n=4,183).
   Every gold label is shifted one letter back; the 1,348 items whose answer is "A" were
   silently dropped (4,183 − 1,348 = 2,835 = the test count used), and "D" can never be
   gold. The same bug affects MedMCQA **dev** (carved from train). Consequence: the
   bench_v0 finding "MedMCQA collapse is positional bias, proven by shuffle" (C048, C051)
   is **false** — Qwen3.5-4B/9B scoring 0.12 (below the 0.25 chance level) were mostly
   answering correctly against a shifted key.
2. **`noul` (yes/no) items are mis-rendered and mis-read.** `render_prompt` prints options
   as `yes. Yes` / `no. No` but the instruction says "Respond with a single letter: A, B,
   C, D…", and the readout scores the tokens of the option keys (`yes`/`no`). Measured
   label mass on `noul` templates is ~0.002–0.2 (Qwen3.5-9B: 0.002 on NFCorpus) versus
   ~0.99 on `choice` templates. Every `noul` result in `loops/bench_v0/baselines.md` is
   invalid, and the `score` templates (label mass 0.002–0.22) are almost certainly affected
   the same way. The bench_v0 finding "relevance judging is at chance for every model"
   (C050 rows for NFCorpus/SciFact) is a **harness artefact**, not a measurement.
   Only the `choice` path was ever validated (T7 used MedQA).
3. **Template screen gaps.** `pubmed_observational_noul_v1` was dropped as a BoW shortcut,
   but its majority class is 0.985 — it is an imbalance problem, misattributed. Kept
   templates are also dominated by one class: `pubmed_pubtype_choice_v1` (majority 0.926),
   `pubmed_humans_noul_v1` (0.883), `fda_boxed_warning_noul_v1` (0.787),
   `ct_randomised_noul_v1` (0.693). Every `noul` regex baseline reads `0.000` because none
   was written — those cells must read `NOT MEASURED`. No check exists for "the gold
   label's text appears in the state" (the MeSH leak, C047, was found only after T9).
4. **Fresh window narrower than necessary.** bench_v0 bounded the window by the teacher's
   repo date (2026-09-10), leaving openFDA templates with 16 / 73 / 94 test items.
   **Operator decision D11 (2026-10-06):** the window start is bounded by the models whose
   scores we report as *ours or our starting points* — the ladder — whose latest date is
   2026-02-28 (Qwen3.5 repo creation; `docs/benchmark/fresh_window.md`). New start:
   **2026-03-01**. Rationale: the teacher never labels benchmark items, so its training
   data cannot leak gold; baseline models released later (27B decision models, Sept 2026)
   could only be *advantaged* by having seen records, which makes our comparison
   conservative. Every fresh item also records whether it falls in the **strict slice**
   (first date ≥ 2026-09-10, after the teacher and the 27B baselines), so results can be
   reported on both.
5. **Smaller defects.** T4 (contamination probe) was never run and left `PENDING`. The
   `coverage` column in `baselines.md` is "template items / source items", not "share of
   the template's items the model could take" — misleading. Some STATE timestamps were not
   taken from `date -u` (e.g. T7 logged 00:30–00:45 but committed 22:04; T9 finished before
   it started). No decision-model baselines were run.

Everything else from bench_v0 stands and is reused: schema, loaders (once fixed), fresh
builders, harness code, audit page, reporting scripts, the 52 CLAIMS rows not listed
above.

## 2. The question this loop answers

Does MedDecide-Bench, with these defects repaired, give numbers we can build loop 1 on —
for every question type, not only `choice`?

- If every (model, template) cell passes the readout-health gate (§3) and both the
  `choice` and `noul` readouts agree with an independent reference → loop 1 (gold-only
  student) starts on this benchmark.
- If `noul` or `score` cannot be read reliably for some models → those question types are
  excluded from loop 1's headline set until a readout that passes exists, and loop 1's
  plan changes accordingly.
- If the rebuilt fresh tier is still too thin or too imbalanced for a template → that
  template is dropped from the headline set, with numbers.

## 3. Ground rules specific to this loop

- **Branch:** keep working on `loop/bench_v0`. Never commit to `main`.
- **Write paths:** `src/`, `scripts/` (add files; do not change `run_loop.sh` or
  `pod_env.sh`), `tests/`, `configs/`, `tools/`, `docs/benchmark/`,
  `loops/bench_v0_fix0/`, `outputs/bench_v0_fix0/`, `data/`, and `pyproject.toml`,
  `uv.lock`, `.gitignore` (append only).
- **Read-only:** `AGENTS.md` (except this loop's "Started" field), `docs/plans/`, and
  everything in `loops/bench_v0/` — bench_v0's records are evidence; corrections to them
  go in `loops/bench_v0_fix0/CORRECTIONS.md`, never into the old files.
- **Benchmark versioning (R9):** the repaired benchmark is **v0.1**. Write it to
  `data/bench/v0.1/{tier1,fresh}/`; leave `data/bench/tier1/` and `data/bench/fresh/`
  (v0) untouched so before/after comparisons stay possible. Configuration for v0.1 goes in
  a new `configs/bench_v0_1.yaml`; `configs/bench_v0.yaml` stays as it was.
- **Readout-health gate (new, permanent — applies to every number this loop reports):**
  for each (model, template) cell, compute on the evaluated items:
  - median label mass (full-vocabulary probability of the option tokens);
  - accuracy with its bootstrap 95% CI, against chance (1/number of options);
  - greedy agreement: on 50 seeded items, the readout's argmax option equals the option
    letter the model generates greedily (1 token, same prompt).

  The cell **passes** if median label mass ≥ 0.5, greedy agreement ≥ 0.9, and the CI's
  upper bound is not below chance. A cell that fails is reported as
  `READOUT_FAIL — <which check>`, never as an accuracy, and is investigated. Write it as a
  reusable function with unit tests; the baseline report must call it.
- **Timestamps:** every STATE timestamp comes from `date -u +%FT%TZ` at the moment it is
  written. Never type or estimate a time.
- **No training.** Inference only.
- **No credentialed data** (no MIMIC, no UMLS).
- **Test splits are evaluation-only.**
- **Teacher endpoint** as in bench_v0: `TEACHER_BASE_URL` / `TEACHER_API_KEY`; if unset when
  reached, the teacher task is `BLOCKED` and the loop continues.
- **One GPU job at a time; timebox** ~2 h per baseline-model integration.
- **Hard stop at F11.** Do not start loop 1.

## 4. Tasks

Each task: mark IN_PROGRESS (timestamp from `date -u`), fill the STATE checklist, do the
work, run the acceptance check, self-audit (R6) into
`outputs/bench_v0_fix0/<task>/SELF_AUDIT.md`, append CLAIMS rows to
`loops/bench_v0_fix0/CLAIMS.md`, mark DONE/BLOCKED, log, commit, push.

### F0 — Orientation and baseline snapshot (no GPU, ~30 min)

**Why:** this loop repairs another loop's work; it must start from that loop's record.

**Do:** fill "Run started" in STATE.md and "Started" in AGENTS.md. Read
`loops/bench_v0/FINDINGS.md`, `NEXT.md`, `baselines.md`, `template_screen.md`, and this
document's §1. Record in `outputs/bench_v0_fix0/F0/start.json`: git commit, the sha256 of
every v0 benchmark file, and the v0 manifest totals. Re-run `uv run pytest -q` and record
the result.

**Acceptance:** `start.json` exists; test result recorded.

### F1 — Fix the MedMCQA answer key, and prove every tier-1 gold against its raw source (CPU, ~2 h)

**Why:** one silent off-by-one invalidated a whole source; the other seven loaders have
never been checked against their raw records either.

**Do:**
1. Fix `_medmcqa_item`: `cop` is 0-based (`gold_index = int(cop)`), with a validation
   that `cop ∈ {0,1,2,3}`; never drop a row without a counted reason.
2. Write `scripts/bench/verify_gold.py` and `src/meddecide/bench/verify.py`: for **every
   tier-1 source**, re-load the raw HF dataset at the pinned revision and, independently of
   the loader code (a separate, minimal mapping written from the dataset card), recompute
   the gold option **text** for every item; compare with the loader's gold option text.
   Also compare the gold-class distribution with the raw answer-field distribution.
3. Add unit tests: MedMCQA `cop=0` → option A; a tiny fixture per source.
4. Build tier-1 v0.1 into `data/bench/v0.1/tier1/` and run the tier-1 audit on it.

**Acceptance:** `verify_gold.py` reports **0 mismatches of N** for every source (N = all
items), MedMCQA test n = 4,183 before caps (or the cap, with the drop reasons counted),
gold distribution on MedMCQA test within ±1 item of the raw `cop` counts
(1,348 / 1,085 / 925 / 825 before caps), tier-1 v0.1 audit passes.
Artifact: `outputs/bench_v0_fix0/F1/gold_verification.json`.

### F2 — Fix the `noul` and `score` readout (GPU, ~3 h)

**Why:** two of three question types were never readable.

**Do:**
1. Render **every** item's options with letter labels (A, B, C, …) in the prompt and read
   the option-letter tokens, exactly as the validated `choice` path does. `noul`: "A. Yes"
   / "B. No" in a fixed order recorded in config; `score`: levels lowest first, A = lowest.
   The prediction file records the letter → option-key mapping per item.
2. Keep the `choice` path byte-identical (re-run T7's MedQA comparison to prove it).
3. Validate the `noul` path against **lm-evaluation-harness** on a yes/no task it
   implements (e.g. `boolq`, or PubMedQA restricted to yes/no-gold items), same model,
   same items, same scoring rule, as bench_v0 T7 did for `choice`: agreement within ±2
   accuracy points. Record both commands and protocol notes.
4. Implement the **readout-health gate** (§3) with unit tests (cases: high-mass correct,
   low-mass, below-chance, greedy disagreement).

**Acceptance:** `noul` reference agreement within ±2 points (both numbers in CLAIMS);
MedQA `choice` accuracy identical to bench_v0's to 4 decimals for the same model; health
gate tests pass; on 200 `noul` and 200 `score` items for Qwen3.5-0.8B and Qwen3.5-4B,
median label mass ≥ 0.9 and greedy agreement ≥ 0.9.
Artifacts: `outputs/bench_v0_fix0/F2/`, committed `loops/bench_v0_fix0/readout_validation.md`.

### F3 — Template screen v2 (CPU, ~3 h)

**Why:** the v0 screen missed a leak and an imbalance, and reported unwritten baselines
as zero.

**Do:** extend `src/meddecide/eval/screen.py` and `scripts/bench/screen_templates.py`:
1. **Gold-in-state check:** for each item, whether the normalised gold option label occurs
   in the normalised state; and a rule baseline "pick the option whose label occurs in the
   state (ties → first)". Flag a template if that rule's test accuracy ≥ 0.8 **or** exceeds
   the majority baseline by ≥ 0.3.
2. **Regex baselines:** write one per template (≤ 30 min each, before looking at model
   results) or record `NOT MEASURED — not written`. A missing baseline is never `0.000`.
3. **Drop rule:** drop a template if any surface baseline (regex, BoW, gold-in-state rule)
   reaches **macro** accuracy ≥ 0.9, or if after balancing (F4) any split has < 2 classes
   or < 200 test items. Record the specific reason.
4. Re-classify bench_v0's decisions: `pubmed_observational_noul_v1` (imbalance, not
   shortcut), `pubmed_mesh_major_choice_v1` (gold-in-state leak),
   `medquad_routing_v1` (keyword-heavy, C047) — each re-screened under v2 rules.

**Acceptance:** `outputs/bench_v0_fix0/F3/screen_v2.json`; committed
`loops/bench_v0_fix0/template_screen_v2.md` listing every template with all baselines (or
`NOT MEASURED`), its flags, and keep/drop with reason. Run on v0.1 data after F4; the
check functions themselves have unit tests.

### F4 — Rebuild the fresh tier: window 2026-03-01, class-balanced, strict slice (CPU + network, ~4–6 h)

**Why:** D11 widens the window; balancing makes accuracy mean something.

**Do:**
1. New config `configs/bench_v0_1.yaml` with `fresh_window.start: 2026-03-01`. Update
   `docs/benchmark/fresh_window.md` with a new section "v0.1 window (D11)" stating the
   rule (ladder-bounded) and the strict slice; keep the v0 section as history.
2. Every fresh item gets `meta.strict_post_teacher: true|false` (first date ≥ 2026-09-10).
3. **Class balancing** per template and split: sample up to K items per gold class with a
   fixed seed (K chosen so each template targets ≥ 1,000 test items where the window
   allows; record K). Report counts before and after balancing per class.
4. Rebuild into `data/bench/v0.1/fresh/`; run the freshness audit (0 items before
   2026-03-01), integrity audit, and rebuild determinism (two builds, identical hashes).
5. Then run F3's screen on v0.1 (tier 1 and fresh).

**Acceptance:** 0 items before the new window start; determinism hashes equal; per
template: classes, per-class counts, total test items, strict-slice count; screen v2
written. Artifacts: `data/bench/v0.1/fresh/{manifest,audit}.json`,
`outputs/bench_v0_fix0/F4/`.

### F5 — Contamination probe on tier 1 (GPU, ~3–5 h) — carried from bench_v0 T4

**Do / acceptance:** exactly as `loops/bench_v0/ADVISORY.md` T4, on v0.1 tier-1 test.
Committed `loops/bench_v0_fix0/contamination.md`.

### F6 — Re-run the ladder baselines on v0.1 with the health gate (GPU, ~10–16 h)

**Why:** every bench_v0 baseline number either used a broken key, a broken readout, or an
imbalanced template.

**Do:** the six ladder models (LFM2.5-350M, Qwen3.5-0.8B, Qwen3.5-0.8B-Base,
MedGemma-1.5-4b-it, Qwen3.5-4B, Qwen3.5-9B) on v0.1 tier-1 test and v0.1 fresh test (kept
templates), with the F2 readout. Report through the health gate. Fix the `coverage`
column to mean "share of this template's items the model could take". Report the fresh
tier both whole and on the strict slice. Include the option-shuffle flip rate for every
model on ≤ 500 items per template (cap in config).

**Acceptance:** `outputs/bench_v0_fix0/F6/results.json`; committed
`loops/bench_v0_fix0/baselines_v0_1.md`; every cell is a measurement that passed the gate,
`READOUT_FAIL — <check>`, or `NOT MEASURED — <reason>`; prediction count == item count per
(model, template).

### F7 — Decision-model baselines (GPU, ~2 h each, timeboxed)

**Why:** the headline claim compares against decision models; none is measured yet.

**Do:** in this order, each through its own published inference path mapped onto v0.1
items, each under the 2 h timebox: `autotrust/JEV-9B` (vLLM with its decision LoRA, per its
model card), `convaiinnovations/laya-typed-decisions`, `convaiinnovations/laya`,
`SupersonicLabs/Julia-1`, `fastino/GLiNER2.5-Decide`, `rAVEUK/open-jev-deberta-v3-large`.
They read options natively, so the letter-readout health gate does not apply; report their
own probabilities with accuracy, Brier, ECE, coverage, and option-shuffle flip rate.

**Acceptance:** each model is in `baselines_v0_1.md` with numbers or
`NOT MEASURED — <reason after timebox>`; at least JEV-9B and one Laya checkpoint measured,
or the reason they could not be is specific (error, missing artifact, licence).

### F8 — Teacher pipeline gate (teacher endpoint) — carried from bench_v0 T10

**Do / acceptance:** exactly as `loops/bench_v0/ADVISORY.md` T10, on v0.1 dev items, with
the F2 readout for `noul`/`score`. `BLOCKED — teacher endpoint not provided` if unset or
unreachable when reached. Same exception as bench_v0: if the endpoint answers when F6
would start, run F8 first.

### F9 — Corrections record (no GPU, ~1 h)

**Why:** a wrong finding left standing in the record is worse than no finding.

**Do:** write `loops/bench_v0_fix0/CORRECTIONS.md`: for each bench_v0 claim or finding
that this loop invalidates or changes — at least C048, C050 (the `noul`/`score` and
MedMCQA rows), C051, FINDINGS items 3 and 5, the MedMCQA counts inside C013, and the
`pubmed_observational` drop reason in C033 — state the original claim, what was wrong (with
evidence), and the corrected value with its new CLAIMS id in this loop, or "withdrawn".
Do not edit any file in `loops/bench_v0/`.

**Acceptance:** every bench_v0 CLAIMS id whose value depended on the MedMCQA loader, the
`noul`/`score` readout, or the v0 fresh window is listed with a disposition (still valid /
corrected → Cxxx / withdrawn). Check: grep bench_v0 CLAIMS for `medmcqa`, `noul`, `score`,
`nfcorpus`, `scifact`, `trec_covid`, `fresh` and account for every hit.

### F10 — Regenerate the operator audit sample (no GPU, ~30 min)

**Do:** draw a new stratified 150-item sample from v0.1 fresh **test** (kept templates, 50
per source where possible, balanced across templates), same format as bench_v0 T11, to
`outputs/bench_v0_fix0/F10/audit_sample.jsonl`. The page `tools/audit/audit.html` is
reused. The audit itself is `BLOCKED — awaiting operator`, with instructions in
"Questions for the operator".

**Acceptance:** sample file exists with 150 rows (or fewer with the reason); the page loads
it (reuse the existing test).

### F11 — Findings and closure (no GPU, ~2 h) — HARD STOP

**Do:** write `loops/bench_v0_fix0/FINDINGS.md` following the FINDINGS structure (Summary
first — the question, what we did, what we found, what it means, what did not work —
every number with its denominator and CLAIMS id; body covering every summary claim; the
five-claim spot-check re-run in fresh processes). The Summary must state plainly which
bench_v0 findings were wrong and what replaced them. Fill STATE's closure feed. Write
`loops/bench_v0_fix0/NEXT.md` (proposals for loop 1, not decisions). Set
`Loop status: STOPPED`, commit, push. **Then stop.**

**Acceptance:** FINDINGS.md with Summary, body, and spot-check; every STATE task `DONE` or
`BLOCKED — <reason>` (none left `PENDING`); pushed.

## 5. Definition of done

- STATE.md — every task `DONE` or `BLOCKED` with a reason; none `PENDING`.
- CLAIMS.md (this loop) — every reported number, with artifact and recompute command.
- CORRECTIONS.md — every affected bench_v0 claim dispositioned.
- FINDINGS.md — Summary first, full-coverage body, five-claim spot-check.
- Committed: code, tests, `configs/bench_v0_1.yaml`, `docs/benchmark/` updates, this
  loop's markdown reports. Not committed: any item text, predictions, or teacher outputs.

## 6. Task table (seed STATE.md with this)

| id | task | GPU | deps | status |
|---|---|---|---|---|
| F0 | Orientation and baseline snapshot | no | — | PENDING |
| F1 | MedMCQA key fix + raw-gold verification for all tier-1 sources | no | F0 | PENDING |
| F2 | `noul`/`score` readout fix + reference validation + health gate | yes | F0 | PENDING |
| F4 | Fresh tier v0.1: window 2026-03-01, balanced, strict slice | no | F0 | PENDING |
| F3 | Template screen v2 (on v0.1) | no | F1, F4 | PENDING |
| F5 | Tier-1 contamination probe | yes | F1, F2 | PENDING |
| F6 | Ladder baselines on v0.1 with health gate | yes | F2, F3 | PENDING |
| F7 | Decision-model baselines | yes | F3 | PENDING |
| F8 | Teacher pipeline gate | teacher | F1, F2, F4 | PENDING |
| F9 | Corrections record | no | F1, F2, F6 | PENDING |
| F10 | Regenerate operator audit sample | no | F3 | PENDING |
| F11 | Findings and closure — HARD STOP | no | all | PENDING |

The order of the table is the execution order. F2 gates every GPU number: nothing is
reported from the new readout before it passes its reference validation. The F8 exception
from bench_v0 applies (run it early if the endpoint is up). A blocked F7, F8, or F10 never
blocks F11.

## 7. Explicitly out of scope

| item | why |
|---|---|
| Any training, LoRA, pointer-head code | loop 1 |
| MIMIC / UMLS / credentialed data | later loops |
| New fresh sources (PMC case reports, OpenAlex) | benchmark v2 |
| 27B-class baselines | need the 4-GPU window; loop 3 |
| Editing any file in `loops/bench_v0/` | it is evidence; corrections go in CORRECTIONS.md |
| Prompt tuning per model | baselines use the fixed protocol |
| Redesigning the MeSH or MedQuAD templates into new tasks | screen and drop/keep only; redesign is a loop-1 proposal |

## 8. Failure modes to avoid

1. **Explaining a bug as a finding** (bench_v0's central failure). A model scoring below
   chance, or a label mass far below 1, is a pipeline defect until proven otherwise. The
   health gate exists so that this is checked mechanically, not by narrative.
2. **Validating only the path you already trust.** bench_v0 validated `choice` and
   assumed `noul`/`score` worked. Every question type needs its own reference check.
3. **Trusting a loader because its audit passed.** Integrity audits (no leaks, no
   duplicates) say nothing about whether gold is *correct*. F1 compares gold with the raw
   record for every item.
4. **Imbalance hiding as accuracy.** Always show majority and macro accuracy; balance the
   fresh tier.
5. **Reporting an unwritten baseline as 0.000** (R3). It is `NOT MEASURED`.
6. **Hand-written timestamps.** Use `date -u`.
7. **Rewriting history.** bench_v0 files stay as they are; CORRECTIONS.md carries the fix.
8. **Overruled objection (carried from the program):** the teacher was chosen without a
   comparison. If F8 runs and ECE > 0.05 after fitting, report it plainly.
9. **Scope creep into training or into template redesign.**

## 9. If you finish early

1. Add a second `noul` reference task to F2's validation.
2. Raise per-template fresh counts toward 2,000 test items where the window allows.
3. Measure option-shuffle flip rate on the full test set for Qwen3.5-4B.
