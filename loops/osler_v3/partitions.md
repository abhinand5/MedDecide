# osler_v3 — X6 partitions (aggregates only; fixed before any Stage A run)

Built 2026-10-18T21:10:32Z by `scripts/osler_v3/x6_partitions.py` (git 6aa0023283). No item text. Recompute: `uv run --frozen python scripts/osler_v3/x6_partitions.py`.

## Clinical-general (CG) dev proxy

The dev proxy for the external panel (the panel itself never enters). Used for the Stage A choice and the monitoring of every run.

| part | items |
|---|---|
| knowledge dev v2 (MedQA 1,000 and MedMCQA 3,000 train hold-outs) | 4000 |
| fast dev v2 clinical, non-domain family items (medqa, medmcqa, pubmedqa, mmlu_medical, medquad, pubhealth, nfcorpus, trec_covid) | 4989 |
| of which dropped: normalised text already in knowledge dev v2 (same question under another item id) | 4122 |
| of which dropped: duplicate text within the CG | 2 |
| of which kept | 865 |
| **CG total** | **4865** (the ADVISORY requires at least 4,000) |

Templates (fast dev v2 family): `medqa_usmle4_v1`, `medmcqa_4opt_v1`, `pubmedqa_ynm_v1`, `mmlu_mc_v1`, `medquad_routing_v1`, `pubhealth_claim_true_noul_v1`, `nfcorpus_relevant_noul_v1`, `trec_covid_graded_score_v1`, `trec_covid_relevant_noul_v1`.

## Fast dev v3 (ADVISORY X6 step 1)

Fast dev v2 (11,323 items, all kept) plus up to 400 dev items per new template (X1, X3, X4), drawn in a stable-hash order.
The new-template dev pools are not built yet (the restyle and chart builds are running); this section is filled when they are.

## Notes

- The CG and fast dev v3 are fixed before any Stage A run; their manifests are in `outputs/osler_v3/X6/` (gitignored item files).
- No item of the external panel, the MR suite test or held-out sets, or the knowledge test enters any partition.
