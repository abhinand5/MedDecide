# synth_v1 MIMIC-grounded claims (aggregates only)

id | claim | value | artifact | recompute command | task
--- | --- | --- | --- | --- | ---
S1 | kept items, mimic_icd_specificity_v1 | 3441 of 4400 | outputs/synth_v1/manifest_mimic_v1.json | `jq` on the manifest `templates` key | synth_v1 build
S2 | leakage probe recovery, mimic_icd_specificity_v1 | 0.983 vs chance 0.173 (n=300) | outputs/synth_v1/manifest_mimic_v1.json | uv run python scripts/synth/mimic_v1_build.py probe | synth_v1 probe
S3 | kept items, mimic_msdrg_v1 | 3664 of 4400 | outputs/synth_v1/manifest_mimic_v1.json | `jq` on the manifest `templates` key | synth_v1 build
S4 | leakage probe recovery, mimic_msdrg_v1 | 0.527 vs chance 0.167 (n=300) | outputs/synth_v1/manifest_mimic_v1.json | uv run python scripts/synth/mimic_v1_build.py probe | synth_v1 probe
S5 | kept items, mimic_drg_cc_level_v1 | 3664 of 4398 | outputs/synth_v1/manifest_mimic_v1.json | `jq` on the manifest `templates` key | synth_v1 build
S6 | leakage probe recovery, mimic_drg_cc_level_v1 | 0.447 vs chance 0.333 (n=300) | outputs/synth_v1/manifest_mimic_v1.json | uv run python scripts/synth/mimic_v1_build.py probe | synth_v1 probe
S7 | kept items, mimic_apr_severity_v1 | 3437 of 4400 | outputs/synth_v1/manifest_mimic_v1.json | `jq` on the manifest `templates` key | synth_v1 build
S8 | leakage probe recovery, mimic_apr_severity_v1 | 0.240 vs chance 0.250 (n=300) | outputs/synth_v1/manifest_mimic_v1.json | uv run python scripts/synth/mimic_v1_build.py probe | synth_v1 probe
S9 | kept items, mimic_ed_disposition_v1 | 2376 of 6000 | outputs/synth_v1/manifest_mimic_v1.json | `jq` on the manifest `templates` key | synth_v1 build
S10 | leakage probe recovery, mimic_ed_disposition_v1 | 0.397 vs chance 0.333 (n=300) | outputs/synth_v1/manifest_mimic_v1.json | uv run python scripts/synth/mimic_v1_build.py probe | synth_v1 probe
S11 | total kept items | 16582 of 23598 | outputs/synth_v1/manifest_mimic_v1.json | uv run python scripts/synth/mimic_v1_build.py probe | synth_v1 build
