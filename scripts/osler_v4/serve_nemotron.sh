#!/usr/bin/env bash
# Nemotron 3.5 Super (D45) on one RTX PRO 6000; tested by the advisor 2026-10-21 (startup ~4 min, KV cache 382,713 tokens).
# Port 8001 is taken by the pod nginx. ninja must be on PATH for the torch.compile build.
source /workspace/MedDecide/scripts/pod_env.sh
export PATH=/workspace/MedDecide/envs/vllm/.venv/bin:$PATH
exec /workspace/MedDecide/envs/vllm/.venv/bin/vllm serve nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 \
  --served-model-name nemotron --host 127.0.0.1 --port 8011 --tensor-parallel-size 1 \
  --max-model-len 16384 --trust-remote-code --language-model-only --reasoning-parser nemotron_v3 \
  --speculative-config.method mtp --speculative-config.num_speculative_tokens 5 \
  --gpu-memory-utilization 0.95 --max-num-seqs 64
