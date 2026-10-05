# bench_v0 — CLAIMS

Every number reported anywhere in this loop (STATE headlines, reports, FINDINGS) has a
row here (AGENTS.md R1). Append only. IDs are `C001`, `C002`, …

`recompute` must be a command that reproduces the value in a fresh process from a
committed script and an artifact under `outputs/bench_v0/` (paths relative to repo root).

| id | claim | value | artifact | recompute | task |
|---|---|---|---|---|---|
| C001 | Pod GPU / CPU / RAM as verified at loop start | `NVIDIA RTX PRO 6000 Blackwell Server Edition`, 97887 MiB, driver 595.91.07, CUDA 13.0; 128 CPUs; 2015 GB RAM | `outputs/bench_v0/T0/env.json` | `jq '{hardware:{gpu:.hardware.gpu,gpu_memory_total_mib:.hardware.gpu_memory_total_mib,driver:.hardware.driver_version,cuda:.hardware.cuda_toolkit,cpu_count:.hardware.cpu_count,ram_total_gb:.hardware.ram_total_gb}}' outputs/bench_v0/T0/env.json` | T0 |
| C002 | Every T0 environment check except the teacher endpoint passed | 9 PASS / 1 FAIL of 10 checks (2026-10-05) | `outputs/bench_v0/T0/env.json` | `jq '.checks | to_entries | group_by(.value.status) | map({status:.[0].value.status,n:length})' outputs/bench_v0/T0/env.json` | T0 |
| C003 | Teacher endpoint reachability at loop start | `FAIL` — `TEACHER_BASE_URL` unset (blocks T10 only) | `outputs/bench_v0/T0/env.json` | `jq -r '.checks.teacher_endpoint_models.status + " — " + .checks.teacher_endpoint_models.reason' outputs/bench_v0/T0/env.json` | T0 |
| C004 | Gated-model access: `google/medgemma-1.5-4b-it` config reachable with pod `HF_TOKEN` | HTTP 200 (2026-10-05) | `outputs/bench_v0/T0/env.json` | `jq -r '.checks.hf_gated_medgemma_config.status + " " + .checks.hf_gated_medgemma_config.reason' outputs/bench_v0/T0/env.json` | T0 |
| C005 | Disk headroom on the pod at loop start | `/workspace` 992 TB total / 327 TB avail; `/` 20 GB total / 19 GB avail | `outputs/bench_v0/T0/env.json` | `jq '.hardware.workspace_fs, .hardware.root_fs' outputs/bench_v0/T0/env.json` | T0 |
