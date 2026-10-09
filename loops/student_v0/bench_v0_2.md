# MedDecide-Bench v0.2 — the three template repairs (loop `student_v0`, task S1)

**What this is.** v0.1 left three templates that either could not be answered correctly or did
not discriminate between models. v0.2 replaces those three with `_v2` versions and carries
**everything else through unchanged**, so every earlier prediction on a carried template stays
comparable. Nothing here fits anything: each gold is still a structured field of the source
record.

| superseded (v0.1) | replacement (v0.2) | defect in v0.1 |
|---|---|---|
| `nfcorpus_graded_score_v1` | `nfcorpus_graded_score_v2` | offered "Not relevant", which **no passage in its pool had**: 0 of 429 items could be answered with it (X029) |
| `pubmed_mesh_major_choice_v1` | `pubmed_mesh_major_choice_v2` | distractors were unrelated frequent topics; 0.9872 (0.8B) and 0.9950 (9B) |
| `fda_class_choice_v1` | `fda_class_choice_v2` | distractors were unrelated class names; 0.8400 (0.8B) and 0.9005 (9B) |

## 1. What the repairs were

1. **Graded score: offer only the levels the pool has.** `load_beir_relevance` now decides the
   offered level set from the pool of the queries actually sampled for the split (and records
   it in every item's `meta.offered_levels`). On nfcorpus the test qrels contain grades 1 and 2
   and no grade 0, so the v0.2 template offers exactly two levels — "Relevant" and "Highly
   relevant" — and **every offered level is the gold of at least one built item**, which is
   checked in the build, in the unit tests, and again by the independent verifier on the
   written file.
2. **MeSH distractors are tree siblings.** `pubmed_mesh_major_choice_v2` draws its distractors
   from descriptors sharing the gold topic's **immediate parent tree number** in NLM's public
   2026 MeSH descriptor file (31,108 descriptors; sha256 `ccd4d0d3…4c8716dd`, recorded in every
   item's `meta.mesh`). Two gold-correctness rules the v1 template did not state are enforced:
   **every** major topic of a record is a correct answer, so no major topic of the record — and
   no sibling of one — is ever offered as a distractor.
3. **openFDA class distractors share the mechanism.** `fda_class_choice_v2` draws distractors
   from classes sharing the gold class's `pharm_class_moa` or `pharm_class_pe` value in a
   full-corpus openFDA index (24,296 labels, 499 classes). Two measured facts shaped this:
   `same_route` fires for 99.7 % of classes (median 228 candidates) and is therefore *not* a
   near-miss signal, so it is not used; and hub values such as `Cell-mediated Immunity [PE]`
   (67 classes) are excluded, because a value most of the corpus carries cannot be why two
   classes are near-misses. A label with fewer than three such distractors is **dropped with a
   counted reason** rather than padded with unrelated classes (2,021 labels).
4. **A fourth D12 check: constant answer.** A cell fails `READOUT_FAIL — degenerate` when its
   most-predicted option is the argmax for ≥ 0.90 of items **and** the template's majority
   share is ≤ 0.60 — the LFM2.5-350M pattern (exactly 0.50 on balanced `noul`, one answer every
   time) that the first three checks passed.
5. **Prediction files carry a `run_id`.** New reader `meddecide.eval.predlog` deduplicates by
   `(run_id, item_id)`, keeps the last row per key, and reports raw rows, unique rows, distinct
   items and per-run counts side by side, so an append-only file can no longer be mistaken for
   an item count (bench_v0_fix0 P7).

## 2. The artifact

| quantity | v0.1 | v0.2 |
|---|---|---|
| items | 46,176 | **45,009** (tier 1 22,403 + fresh 22,606) |
| carried through untouched | — | **35,263 items, 0 content mismatches** |
| rebuilt as `_v2` | — | **9,746 items** (score 238; MeSH 5,856; FDA class 3,652) |
| superseded and absent from v0.2 | — | 10,913 items across the three `_v1` templates |
| build | acceptance PASS 7/7, rebuild byte-identical | acceptance **PASS 8/8**, independent verifier **PASS 11/11** |
| strict slice (by record date ≥ 2026-09-10) | 2,896 | 2,905 |

* `data/bench/v0.2/acceptance.json` — the build's own checks (carried identical, no superseded
  template left, every offered score level is a gold in test, no record crosses splits, every
  fresh item inside the window, every `_v2` template ≥ 200 test items, manifest matches files).
* `outputs/student_v0/S1/verify_v0_2.json` — an **independent** re-derivation from the written
  files (a builder checking its own memory can be right about the wrong thing). 11/11 PASS.
* The build took 672 s; the window is unchanged (2026-03-01 → 2026-10-05) and the PubMed reader
  uses **the same 60 update files v0.1 recorded**, so the `_v2` template covers the same records.
* Shortcut risk measured on the test splits: gold text verbatim in the state is 0.224 for the
  MeSH template (v1: 0.4625), 0.0 for the FDA class template as stored — and **0.102** once the
  source's `[EPC]` tag is stripped, which is the number a copy rule could actually exploit
  (`shortcut_risk` in the verifier; the screen's own check sees only the tagged 0.0).

## 3. Screen (`loops/student_v0/template_screen_v0_2.md`)

22 templates, **16 kept, 6 dropped**, every drop `n_test<200` — the same rule and the same count
as v0.1. All three `_v2` templates are kept:

| template | n_test | majority | BoW macro | gold-in-state | flagged |
|---|---|---|---|---|---|
| `nfcorpus_graded_score_v2` | 238 | 0.500 | NOT MEASURED (score) | 0.004 | no |
| `fda_class_choice_v2` | 3,236 | 0.250 | 0.258 | 0.000 | no |
| `pubmed_mesh_major_choice_v2` | 4,000 | 0.250 | 0.264 | 0.475 | no |

As a reproducibility check, the v0.1 screen was re-derived from the frozen v0.1 files with the
same script: **22 of 22 templates identical, 0 field differences**.

## 4. Zero-shot baselines on the repaired templates (D12 applied)

Protocol: the fixed chat-template letter readout, 0.8B and 9B, identical item sets to the v0.1
runs (per-template `n` below). Full table: `loops/student_v0/bench_v0_2_baselines.md`.

| template | 0.8B v0.2 (v0.1) | 9B v0.2 (v0.1) | gate |
|---|---|---|---|
| `nfcorpus_graded_score_v2` | **0.5966** (v1 unusable) | `READOUT_FAIL — greedy agreement 0.880 < 0.9 (n=50)` | 0.8B PASS, 9B withheld |
| `pubmed_mesh_major_choice_v2` | **0.9580** (0.9872) | **0.9830** (0.9950) | both PASS |
| `fda_class_choice_v2` | **0.7803** (0.8400) | **0.8705** (0.9005) | both PASS |

* All six cells score the full template (n = 238 / 3,236 / 4,000), coverage 1.000, no coverage
  mismatch in the report.
* The 9B score cell is **withheld, not reported**: its letter readout agrees with greedy
  generation on only 0.88 of the sampled items. The 0.8B cell passes at 0.94 with label mass
  0.9953, and its 0.5966 is well above the 0.5 chance level for a two-level item.
* Every `_v2` template separates less than the v1 it replaces, which is what "near-miss
  distractors" was meant to do: FDA class −0.060 (0.8B) and −0.030 (9B); MeSH −0.029 / −0.012.

## 5. The saturation criterion: one pass, one failure, one withheld

The S1 acceptance requires the new templates **not** to be saturated (9B ≤ 0.90):

* `fda_class_choice_v2` — **PASS** (0.8705).
* `nfcorpus_graded_score_v2` — **NOT MEASURED**: the 9B cell fails the D12 gate, so no accuracy
  is reported (the 0.8B is 0.5966 against chance 0.5).
* `pubmed_mesh_major_choice_v2` — **FAIL** (0.9830). Recorded, with the fix investigated below
  rather than the rule loosened.

**Why the MeSH template is still easy, measured.** Restricting the *scored items* to those where
the gold topic does not appear in the abstract — i.e. removing the copy shortcut entirely —
leaves 0.8B accuracy at **0.9491** (n = 3,104 of 4,000; the copy subset scores 0.9888). The
task is not solved by string matching; a 0.8B model can identify the indexing topic of an
abstract among four siblings.

**Additional analysis (labelled as such): does raising the option count help?** An 8-option
variant (7 near-miss siblings) was built offline from the frozen v0.2 items plus the MeSH index
(`outputs/student_v0/S1/extra/options8/`, template id `…_options8`; not part of v0.2) and scored
on the **same 1,838 test records**:

| model | 4 options | 8 options | Δ | chance drop 0.25 → 0.125 |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0.9576 | **0.8667** | −0.0909 | −0.125 |
| Qwen3.5-9B | 0.9788 | **0.9614** | −0.0174 | −0.125 |

More options help the 0.8B (below 0.90) but not the 9B: both models stay far above chance, so
the template measures a capability the model has rather than a shortcut. 3,086 of 5,856 records
could not produce 7 safe siblings at all, so an 8-option variant is also only constructible on
about half the pool.

**Proposal for the next loop (not a decision):** retire `pubmed_mesh_major_choice_v2` from the
headline set — or keep it as a reported *ceiling* template, clearly labelled — and do not spend
another design cycle on it. The two repairs that were expected to matter (score levels, FDA
class) both behave as intended; the MeSH-topic question is intrinsically easy for models of this
size. Evidence: this section, `verify_v0_2.json`, `extra/options8/summary.json`.

## 6. Acceptance, criterion by criterion

| S1 acceptance criterion | result |
|---|---|
| `data/bench/v0.2/` built with a manifest | **PASS** — 45,009 items, `manifest.json` + `acceptance.json` |
| the three `_v2` templates pass the screen (gold-in-state, BoW macro < 0.90, `n_test` ≥ 200) | **PASS** — 3/3 kept, see §3 |
| the `_v1` versions excluded from v0.2 tables, kept in the manifest as `superseded` | **PASS** — 0 left in the files; 3 entries with counts and reasons |
| tests pass | **PASS** — `uv run ruff check .` clean; `uv run pytest` 155 passed at the build commit |
| zero-shot 0.8B and 9B reported on the three `_v2` templates | **PASS** — 5 cells reported, 1 withheld by D12 |
| the new templates must not be saturated (9B ≤ 0.90) | **FAIL for `pubmed_mesh_major_choice_v2` (0.9830)**, NOT MEASURED for the score template (D12), PASS for `fda_class_choice_v2` — investigated and proposed above rather than patched |

## 7. Commands

```bash
# build (v0.1 stays untouched; needs the MeSH index and the openFDA class index in /workspace/tmp)
uv run python scripts/bench/fetch_mesh.py --cache-dir /workspace/tmp/mesh --out /workspace/tmp/mesh/mesh_index.json
uv run python scripts/bench/fetch_fda_class_index.py --out /workspace/tmp/openfda/class_index_full.json --max-labels 24296
uv run python scripts/bench/build_v0_2.py --config configs/bench_v0_2.yaml --pubmed-files 60
# independent verification of the written artifact
uv run python scripts/bench/verify_v0_2.py --out outputs/student_v0/S1/verify_v0_2.json
# screen
uv run python scripts/bench/screen_v0_1.py --tier1 data/bench/v0.2/tier1 --fresh data/bench/v0.2/fresh \
  --out data/bench/v0.2/screen.json --report loops/student_v0/template_screen_v0_2.md
# baselines (one GPU job at a time; per-template so the model files do not overwrite each other)
bash outputs/student_v0/S1/logs/run_s1_baselines.sh
uv run python scripts/bench/report_baselines_v0_1.py --dir outputs/student_v0/S1 --out outputs/student_v0/S1/results.json \
  --report loops/student_v0/bench_v0_2_baselines.md --screen data/bench/v0.2/screen.json \
  --benchmark-label "MedDecide-Bench v0.2 (S1: the three \`_v2\` templates)"
# the 8-option additional analysis
uv run python scripts/bench/mesh_options_experiment.py --n-options 8 --out outputs/student_v0/S1/extra/options8
```

Raw artifacts live under `outputs/student_v0/S1/` (gitignored) and `data/bench/v0.2/` (gitignored);
the committed record is this file, `template_screen_v0_2.md`, `bench_v0_2_baselines.md`,
`loops/student_v0/CLAIMS.md` (S014–S0xx) and `outputs/student_v0/S1/SELF_AUDIT.md`.
