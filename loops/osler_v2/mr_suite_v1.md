# MR suite v1 (osler_v2, D35) — frozen at W1

Aggregates only. Recompute: `uv run --frozen python scripts/osler_v2/w1_screens.py` then `uv run --frozen python scripts/osler_v2/w1_freeze_suite.py`. The rules are fixed (D35): zero-shot Qwen3.5-4B dev accuracy ≥ 0.97 (ceiling) and Naive Bayes dev macro ≥ 0.90 (vocabulary shortcut) exclude a template from the verdict; both are reported. Nothing here was chosen after an osler_v2 model result: none exists yet.

| template | dev items | ceiling (zero-shot acc) | NB dev macro | string presence | verdict | test items | held-out items |
|---|---|---|---|---|---|---|---|
| `mr_page_category_tabular_v1` | 300 | 1.0000 (n=300) | 1.0000 | 0.2764 | REPORTED ONLY | 300 | 0 |
| `mr_indicator_evidence_v1` | 550 | 0.6925 (n=400) | 0.7964 | — | IN VERDICT | 576 | 74 |
| `mr_icd_support_template_v2` | 544 | 0.7100 (n=400) | 0.8040 | 0.0256 | IN VERDICT | 554 | 102 |
| `synthea_page_category_v1` | 6962 | 0.9950 (n=400) | 1.0000 | 0.2783 | REPORTED ONLY | 5891 | 0 |
| `synthea_indicator_evidence_v1` | 740 | 0.7875 (n=400) | 0.5757 | — | IN VERDICT | 618 | 1342 |
| `mtsamples_page_category_v1` | 97 | 0.6289 (n=97) | 0.5903 | 0.1476 | IN VERDICT | 111 | 0 |
| `pmc_similar_patient_v1` | 7990 | 0.7800 (n=400) | 0.5202 | — | IN VERDICT | 8100 | 0 |
| `codiesp_icd_support_v1` | 0 | NOT MEASURED | NOT MEASURED | — | IN VERDICT | 1654 | 0 |
| `mkurman_icd_support_v1` | 0 | NOT MEASURED | NOT MEASURED | — | IN VERDICT | 1490 | 0 |

## Reasons

- `mr_page_category_tabular_v1`: ceiling: zero-shot dev accuracy 1.0000 >= 0.97
- `mr_page_category_tabular_v1`: vocabulary shortcut: Naive Bayes dev macro 1.0000 >= 0.9
- `synthea_page_category_v1`: ceiling: zero-shot dev accuracy 0.9950 >= 0.97
- `synthea_page_category_v1`: vocabulary shortcut: Naive Bayes dev macro 1.0000 >= 0.9
- `codiesp_icd_support_v1`: dev screens NOT MEASURED: no dev items yet: the CodiEsp dev carve-out (15 % of train cases) is built in W5
- `codiesp_icd_support_v1`: D35 rules cannot trigger without dev items; the template stays in the verdict (D35's own exclusions are the only drops). Operator question recorded in STATE section 6.
- `mkurman_icd_support_v1`: dev screens NOT MEASURED: no train or dev split: evaluation-only source (osler_v1 V5b), so no dev screen exists
- `mkurman_icd_support_v1`: D35 rules cannot trigger without dev items; the template stays in the verdict (D35's own exclusions are the only drops). Operator question recorded in STATE section 6.

Test items: total 19294; in the verdict 13103; reported only 6191 (sum checked in code).

