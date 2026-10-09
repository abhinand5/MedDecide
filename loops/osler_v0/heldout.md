# osler_v0 — held-out generators (fixed before any O6 run)

This list is the O3 lock. It is committed with O3 DONE, and no model has been trained on O3 items. A change
to it is a deviation (STATE §7) and needs the operator's approval. `tests/test_gen.py` checks the list against the
generator registry.

## The list (2 of 9 generators)

| generator | family | task (label space) | train items | dev items (patients) | test items (patients) |
|---|---|---|---|---|---|
| `note_lab_range_v1` | A_note_facts | choice: within / below / above the reference range | 0 | 300 (150) | 300 (150) |
| `policy_triage_v1` | C_triage_policy | score: triage level 1, 2 or 3 under the written policy | 0 | 300 (150) | 300 (150) |
| held out, total | | | **0** | **600** | **600** |

Source: `src/meddecide/gen/families.py` (`HELD_OUT`). Counts: CLAIMS O048–O050, `outputs/osler_v0/O3/recompute.json`.

`policy_triage_v1` is the only generator of family C, so holding it out leaves that family with no training data.
`note_lab_range_v1` is one of four family-A generators; the other three stay in training.

## Rules (binding for every later task)

1. No held-out item enters any training set: the O4 mix, the O6–O8 arms, or O10. The build gives these generators
   a training quota of 0 patients, and `tests/test_gen.py` checks that no held-out row is in `train.jsonl`.
2. Held-out **dev** items are used only for the O3 screen and its audit. They are never used for model selection,
   the head-choice rule (O9), temperatures, thresholds, prompts or templates.
3. Held-out **test** items are for evaluation only. O11 reports them in a separate row and never pools them with
   seen-generator results.
4. The gold rules, the triage policy text and the generator code of these two generators are fixed after this
   commit. Any change is a deviation and must be reported (R8).

## Screen result (O3)

Both generators pass the screen's pass criteria. Read the caveat before relying on the pass.

| check (dev, 300 items each) | note_lab_range_v1 | policy_triage_v1 |
|---|---|---|
| gold label text occurs in the patient record | 0 hits | 0 hits |
| string-presence macro | 0.3333 | 0.3333 |
| Naive Bayes macro | 0.3333 | 0.3333 |
| twin pairs; NB both-correct share | 150; 0.0 | 150; 0.0 |

**Caveat: the two baselines are constant predictors, not measured shortcut baselines** (CLAIMS O056). On dev, the
Naive Bayes predicts one label for every item. For `note_lab_range_v1` it was trained on 12,000 rows of the seen
family-A generators, but their gold labels (yes/no) never match the offered options, so every option scores undefined
and the model returns the first option. For `policy_triage_v1` no seen generator is in family C, so the Naive Bayes
trains on 0 rows. A macro of 0.3333 is what a constant predictor scores over three classes.

The held-out pass therefore rests on two things: the gold label never appears in the patient text (0 hits), and each
twin flips the gold label by construction (checked for 40 patients per generator in `tests/test_gen.py`). It does
not show that a learned shortcut is absent. `screen.json` also names the triage Naive Bayes source "seen generators
of family C", which is inaccurate (STATE §7, item 22). `outputs/osler_v0/O3/recompute.json` has the accurate
description.

## Revisions to held-out generator code before this lock

- `policy_triage_v1`: when chest pain and sweating are both present and systolic BP is below 90, the base now
  draws a systolic BP of 90 or more. The earlier version had no twin that changed the level in that case and raised
  `StopIteration` in the first screen run. The triage policy text and the level rule are unchanged.
- `note_lab_range_v1`: the HbA1c twin range was corrected during the first screen run. The earlier version is not
  in git (O3 code was first committed at 8794fc8), so the change cannot be diffed.

Both revisions were made before any model result and before this list was committed. The list itself did not change.

## Files

- `data/gen/osler_v0/{train,dev,test}.jsonl` and `manifest.json` (gitignored; SHA-256 in the manifest)
- `outputs/osler_v0/O3/{screen.json, recompute.json, SELF_AUDIT.md}` (gitignored)
