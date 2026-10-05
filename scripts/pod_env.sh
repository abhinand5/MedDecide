#!/usr/bin/env bash
# Source this on the pod before anything else:  source scripts/pod_env.sh
# Keeps every cache on the /workspace volume (the root disk is only 20 GB).
export HF_HOME=/workspace/.hf_home
export UV_CACHE_DIR=/workspace/.uv_cache
export UV_PYTHON_INSTALL_DIR=/workspace/.uv_python
export TMPDIR=/workspace/tmp
export TORCH_HOME=/workspace/.torch
export VLLM_CACHE_ROOT=/workspace/.vllm_cache
export TRITON_CACHE_DIR=/workspace/.triton_cache
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$TMPDIR" "$TORCH_HOME" \
         "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"

# Secrets: put them in /workspace/.secrets.env (never inside the repo), one per line:
#   export HF_TOKEN=...
#   export TEACHER_BASE_URL=...      # optional until T10
#   export TEACHER_API_KEY=...       # optional until T10
#   export DEEPSEEK_API_KEY=...      # or OPENROUTER_API_KEY, for the agent harness
# `set -a` exports every variable the file defines, with or without `export`.
if [ -f /workspace/.secrets.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /workspace/.secrets.env
  set +a
fi
