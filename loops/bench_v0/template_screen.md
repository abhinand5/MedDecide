# Fresh-template screen (T8)

**Generated:** 2026-10-05T21:47:00Z  
**Patterns:** `configs/template_screen_patterns.yaml` (sha256 `f951b483acc5807e…`)  
**Thresholds:regex >= 0.95, bag-of-words >= 0.95 on the **test** split → template dropped.

The bag-of-words model (TF-IDF 1-2 grams + logistic regression) is fitted on each
template's **dev** split and scored on its **test** split; the regex baseline is
hand-written per template and never fitted. Nothing here touches a model's predictions.

## Kept templates

| template | source | qtype | n_test | classes | majority | regex (test) | regex macro (test) | BoW (test) | BoW macro (test) | BoW (dev) |
|---|---|---|---|---|---|---|---|---|---|---|
| `ct_intervention_type_choice_v1` | clinicaltrials | choice | 2588 | 11 | 0.273 | 0.302 | 0.213 | 0.530 | 0.235 | 0.879 |
| `ct_phase_choice_v1` | clinicaltrials | choice | 701 | 5 | 0.345 | 0.184 | 0.157 | 0.456 | 0.286 | 0.924 |
| `ct_primary_purpose_choice_v1` | clinicaltrials | choice | 2221 | 9 | 0.596 | 0.451 | 0.160 | 0.597 | 0.112 | 0.611 |
| `ct_randomised_noul_v1` | clinicaltrials | noul | 2221 | 2 | 0.693 | 0.000 | 0.000 | 0.730 | 0.564 | 0.821 |
| `fda_boxed_warning_noul_v1` | openfda | noul | 94 | 2 | 0.787 | 0.000 | 0.000 | 0.787 | 0.500 | 0.875 |
| `fda_class_choice_v1` | openfda | choice | 16 | 4 | 0.438 | 0.000 | n/a | n/a | n/a | n/a |
| `fda_route_choice_v1` | openfda | choice | 73 | 6 | 0.630 | 0.589 | 0.316 | 0.767 | 0.246 | 1.000 |
| `pubmed_humans_noul_v1` | pubmed | noul | 5000 | 2 | 0.883 | 0.000 | 0.000 | 0.884 | 0.501 | 0.891 |
| `pubmed_mesh_major_choice_v1` | pubmed | choice | 5000 | 4 | 0.259 | 0.000 | n/a | 0.255 | 0.257 | 0.997 |
| `pubmed_pubtype_choice_v1` | pubmed | choice | 5000 | 6 | 0.926 | 0.049 | 0.518 | 0.926 | 0.167 | 0.919 |

## Dropped templates

| template | source | n_test | classes | majority | regex (test) | BoW (test) | reason |
|---|---|---|---|---|---|---|---|
| `ct_healthy_volunteers_noul_v1` | clinicaltrials | 2900 | 1 | 1.000 | 0.000 | n/a | single_class_test_split |
| `pubmed_observational_noul_v1` | pubmed | 5000 | 2 | 0.985 | 0.000 | 0.985 | bow_test_accuracy>=0.95 |

## Notes carried with the numbers

- 10 of 12 templates kept; 2 dropped.
- A `noul` template has chance = 0.5, so a BoW score *below* 0.5 is not evidence the
  template is hard — it can mean the classifier collapsed to the majority class.
- Templates whose answer is not stated in the state (major MeSH topic, pharmacologic
  class) have no honest regex cue; their regex rows are 0 by construction and the
  baseline that matters for them is the bag-of-words one.
- Dropping a template removes it from T9's headline table; it stays in the data so the
  decision is reviewable.

- `ct_healthy_volunteers_noul_v1`: the test split has one gold class, so no accuracy baseline can discriminate: the template measures nothing as built; dev split too small or single-class; bag-of-words baseline not fitted
- `ct_intervention_type_choice_v1`: BoW macro accuracy 0.235 <= 0.55: the classifier is close to a majority-class predictor, so its micro accuracy is not evidence of template difficulty either way
- `ct_phase_choice_v1`: BoW macro accuracy 0.286 <= 0.55: the classifier is close to a majority-class predictor, so its micro accuracy is not evidence of template difficulty either way
- `ct_primary_purpose_choice_v1`: BoW macro accuracy 0.112 <= 0.55: the classifier is close to a majority-class predictor, so its micro accuracy is not evidence of template difficulty either way
- `fda_boxed_warning_noul_v1`: BoW macro accuracy 0.500 <= 0.55: the classifier is close to a majority-class predictor, so its micro accuracy is not evidence of template difficulty either way
- `fda_class_choice_v1`: dev split too small or single-class; bag-of-words baseline not fitted
- `fda_route_choice_v1`: BoW macro accuracy 0.246 <= 0.55: the classifier is close to a majority-class predictor, so its micro accuracy is not evidence of template difficulty either way
- `pubmed_humans_noul_v1`: BoW macro accuracy 0.501 <= 0.55: the classifier is close to a majority-class predictor, so its micro accuracy is not evidence of template difficulty either way
- `pubmed_mesh_major_choice_v1`: BoW macro accuracy 0.257 <= 0.55: the classifier is close to a majority-class predictor, so its micro accuracy is not evidence of template difficulty either way
- `pubmed_observational_noul_v1`: majority class is 0.9850 of the test split; BoW macro accuracy 0.500 <= 0.55: the classifier is close to a majority-class predictor, so its micro accuracy is not evidence of template difficulty either way
- `pubmed_pubtype_choice_v1`: majority class is 0.9260 of the test split; BoW macro accuracy 0.167 <= 0.55: the classifier is close to a majority-class predictor, so its micro accuracy is not evidence of template difficulty either way
