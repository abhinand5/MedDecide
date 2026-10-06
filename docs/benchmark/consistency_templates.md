# Record–claim consistency templates (D15) — design and measurements

**What this family is.** A record–claim consistency item states a fact and asks whether the
*record* supports it, rather than asking the model to infer a hidden field. Real document work
is mostly verification, and this family is the closest public analogue with gold known by
construction.

**Gold** is always a lookup of a structured source field — ClinicalTrials.gov arm types and
design fields, openFDA route values. No model produced any label.

## 1. The three templates

| template | qtype | source | role-binding design | fresh test/dev | pre-window |
|---|---|---|---|---|---|
| `ct_arm_role_noul_v1` | `noul` | ClinicalTrials.gov | arms are listed with their interventions and **all arm types removed**; the claim names an intervention and the answer is the role of the arm it sits in | 2,000 / 1,026 | built (training) |
| `ct_claim_set_choice_v1` | `choice` | ClinicalTrials.gov | four stated fields, exactly one unsupported; the swapped intervention-type value is another intervention's type **in the same record** | 4,000 / 1,936 | built (training) |
| `fda_route_claim_noul_v1` | `noul` | openFDA | the unsupported route word **always occurs verbatim (or as its stem) in the label text**, so "the value is present" cannot separate the classes | 2,000 / 1,036 | built (training) |

**Held out (D14): `ct_arm_role_noul_v1`.** Its role structure — an intervention bound to an arm
whose *type* the state never names — differs most from the other two, which are both "verify a
stated field value". A model that learned field verification from the training templates still
has to handle the arm-role question, so the held-out set measures transfer rather than recall of
a template shape. It is excluded from every training file (the pre-window builder drops it, with
the excluded count recorded).

## 2. How each template satisfies the S2 design rules

**Rule 1 — gold from structured fields only.** `armGroups[].type` (`EXPERIMENTAL` vs any
comparator type); the trial design fields (`phases`, `designInfo.allocation`,
`designInfo.primaryPurpose`, `interventions[].type`, `eligibilityModule.healthyVolunteers`);
`openfda.route`. Each item records its `source_field` in `meta`.

**Rule 2 — role binding, not string matching.** The measured detector is the string-presence
baseline below; each design also removes the obvious role cue:

* *arm role*: the arms block lists `Arm n: <interventions>` with no type, label, title or
  description. An intervention listed in **two** arms is never used (the claim would be true of
  either arm), and an intervention whose *name* is a role word ("Placebo") is never used as a
  claim, because that claim would be answerable without reading the record. Records whose
  comparator arm has no non-role-word intervention are dropped with a counted reason.
* *claim set*: the four stated fields are rendered in the **question**, not the state, so the
  gold option's text is not placed in its own state by construction (an earlier version did
  exactly that and the screen flagged it: gold-in-state 1.000 → the template was dropped).
* *route claim*: an unsupported claim's route word is required to occur in the label text, and —
  after a measured leak — the **same cue is required for the supported half too**. The first
  version selected the unsupported half by "another route word appears in the text", which a
  bag-of-words baseline read at **0.715** macro accuracy; requiring the cue for both classes
  brought that to **0.530**.

**Rule 3 — balanced per template and split.** `noul` templates are 50/50 by construction
(alternating supported/unsupported) and the choice variant rotates which field is swapped. Test
gold counts: `yes/no` = 1,000/1,000 and 1,000/1,000; the choice variant is 1,000 per option key.
Balancing also caps each class at 1,000 per split, which is why the templates land on exactly
2,000 and 4,000 test items.

**Rule 4 — a multi-field variant.** `ct_claim_set_choice_v1` states four fields and asks which
one the record does not support.

**Rule 5 — both windows.** The fresh window (≥ 2026-03-01) is built into
`data/bench/v0.2/fresh/`; the pre-window window (< 2026-03-01, 2023-01-01 → 2026-02-28) is built
into `data/train/student_v0/prewindow_consistency.jsonl` for S6, with the held-out template
excluded (manifest: `prewindow_consistency_manifest.json`).

## 3. Measured: the string-presence baseline (acceptance: ≤ 0.60 macro)

The naive rule the design must defeat — `noul`: answer "yes" iff the claimed value (or its stem)
occurs in the state; `choice`: choose the first option whose value does *not* occur — scored on
the fresh **test** split of the written files:

| template | n | baseline accuracy | cap |
|---|---|---|---|
| `ct_arm_role_noul_v1` | 2,000 | **0.4985** | 0.60 |
| `ct_claim_set_choice_v1` | 4,000 | **0.3138** | 0.60 |
| `fda_route_claim_noul_v1` | 2,000 | **0.4750** | 0.60 |

All three are at or below chance for their option count. The arm-role and route-claim templates
are `noul` (chance 0.5); the choice variant has four options (chance 0.25) and its baseline sits
just above it.

## 4. Measured: the screen (acceptance: gold-in-state, BoW macro < 0.90, n_test ≥ 200)

| template | n_test | majority | BoW macro | regex | gold-in-state | flagged | kept |
|---|---|---|---|---|---|---|---|
| `ct_arm_role_noul_v1` | 2,000 | 0.500 | 0.507 | NOT MEASURED | 0.389 | no | yes |
| `ct_claim_set_choice_v1` | 4,000 | 0.250 | 0.342 | NOT MEASURED | 0.000 | no | yes |
| `fda_route_claim_noul_v1` | 2,000 | 0.500 | 0.530 | NOT MEASURED | 0.595 | no | yes |

No regex patterns are configured for these templates, so that baseline reads `NOT MEASURED`
rather than a misleading 0.000 (the v0.1 screen rule).

## 5. Measured: zero-shot Qwen3.5-0.8B and 9B (D12 applied)

| template | 0.8B | 9B | chance |
|---|---|---|---|
| `ct_arm_role_noul_v1` (held out) | **0.6950** | **0.8780** | 0.50 |
| `ct_claim_set_choice_v1` | **0.2517** | **0.7778** | 0.25 |
| `fda_route_claim_noul_v1` | **0.8875** | *see `outputs/student_v0/S2/`* | 0.50 |

Every cell passes the readout-health gate (label mass 0.987–0.998, greedy agreement 0.94–1.00).

Two readings worth stating plainly, because they are results rather than defects:

* the **multi-field variant is at chance for the 0.8B** (0.2517 against 0.25) and 0.53 above
  chance for the 9B: identifying *which* of four stated fields is unsupported needs more
  capability than the smallest model has, and the template discriminates sharply between the two;
* the **route claim is easy for the 0.8B** (0.8875) while its string-presence baseline is 0.475,
  so the accuracy is earned by reading the label rather than by matching the claimed word.

## 6. What is deliberately not here

* No template asks the model to *produce* a value; every item is a verification question.
* Nothing in this family was trained on in this loop: the pre-window file is training data for
  S9, and the held-out template is in no training file at any window.
* One design from the plan was not built: `ct_outcome_role_noul_v1` (primary vs secondary
  outcome). The family already has three templates across two sources including the required
  multi-field variant, and the outcome-role wording needs a separate answerability check that
  the abstract-only state may not support — recorded as not built rather than silently dropped.

## 7. Commands

```bash
# fresh window (writes into data/bench/v0.2/fresh/ and updates the manifest)
uv run python scripts/bench/build_consistency.py --window fresh
# pre-window training file (held-out template excluded)
uv run python scripts/bench/build_consistency.py --window prewindow
# screen
uv run python scripts/bench/screen_v0_1.py --tier1 data/bench/v0.2/tier1 --fresh data/bench/v0.2/fresh \
  --out data/bench/v0.2/screen.json --report loops/student_v0/template_screen_v0_2.md
# zero-shot baselines on the three templates
bash outputs/student_v0/S2/logs/run_s2_baselines.sh
```

Raw artifacts: `outputs/student_v0/S2/` (gitignored). Claims: `loops/student_v0/CLAIMS.md`
(S027+).
