# MedDecide & Osler

**Osler** is a family of open, calibrated **general medical decision models** ("System One" single-pass) that combine broad clinical decision-making with **stronger EHR and clinical documentation understanding than much larger models (including 27B)**. Given any clinical state (an EHR note, discharge summary, clinical trial record, FDA drug label, or medical question) and typed questions (`noul` yes/no, `choice` over 2–255 named options, `score` over 2–10 ordered levels), they return a calibrated probability distribution over every offered option in **one prefill pass** — no text generation.

This repository contains the **MedDecide** codebase (`src/meddecide/`), evaluation harness, benchmark schemas, training/calibration/audit scripts (`scripts/`, `configs/`), and unit tests (`tests/`).

## Released Models (Apache-2.0 — Standalone Merged Weights)

Both models are published on Hugging Face as **standalone merged safetensors checkpoints** with bundled runtimes and built-in dev-only per-type calibration (no external library required):

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("abhinand/Osler-9B", trust_remote_code=True)  # or "abhinand/Osler-4B"
result = model.system_one(
    state="68yo M with fever 38.9C, BP 84/52, lactate 4.2 mmol/L, on norepinephrine for E. coli bacteremia.",
    questions={
        "septic_shock": {"type": "noul", "instructions": "Does the note support septic shock?"},
    },
)
```

For production HTTP serving of **Osler-9B**, use **vLLM Semantic Router** (`vllm-sr serve -e abhinand/Osler-9B`) or the bundled `d3_server.py`.

| Model | Hugging Face | Base Backbone | Params | 5-Set Macro | General Medical Avg | EHR & Clinical Docs Avg | Batch-1 p50 | Peak VRAM |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Osler-9B** | [`abhinand/Osler-9B`](https://huggingface.co/abhinand/Osler-9B) | `vllm-sr/d3-flash` | 8.39B | **0.7801** | 0.7419 | **0.8375** | 33.71 ms | 17.32 GB |
| **Osler-4B** | [`abhinand/Osler-4B`](https://huggingface.co/abhinand/Osler-4B) | `perplexity-ai/pplx-decider-flash-v1.2` | 4.55B | 0.7530 | 0.7038 | 0.8269 | 26.54 ms | 9.60 GB |
| *`pplx-27B` (27B ceiling)* | `perplexity-ai/pplx-decider-v1.2-27b` | — | 26.09B | 0.7600 | **0.7725** | 0.7411 | 90.96 ms | 52.92 GB |

- **Global 2-axis Pareto frontier (General Medical QA vs. EHR & Clinical Docs):** **Osler-9B** is the only ≤10B model on the global front alongside `pplx-27B`, achieving the highest 5-set macro of all 12 evaluated models and **+9.6 pts stronger EHR & clinical documentation understanding than `pplx-27B`** at **2.7× lower latency** and **3.1× lower VRAM**, while strictly dominating every comparable ≤10B model (`Osler-4B`, `pplx-flash`, `MedDecider-9B/4B`, `Clef-Flash`) across all five headline sets.
- **Privacy & safety:** Both models cleared a local MIMIC-IV PHI/memorisation audit (visited-member ΔAUC `+0.0468` 9B / `+0.0248` 4B vs. zero-shot controls, `0/300` 4B verbatim matches). **Osler-9B** carries no `lm_head` (architecturally non-generative).
- **Honest limitations:** `Osler-9B` ties its `d3-flash` base on OOD `panel` and `knowledge` and trails `pplx-27B` on hard multi-step QA; `Osler-4B` trades ~1 pt on OOD `panel`/`knowledge` for large EHR and clinical documentation gains; dev-fitted temperatures leave both models under-confident on fresh `confirm_general`; and both models fail a held-out synthetic indicator template (`synthea_indicator_evidence_v1`).

## Repository Layout

- `src/meddecide/` — Core schema, model adapters, trainer, calibration, evaluation, and synthetic data modules
- `scripts/` — Training, evaluation, calibration, latency, PHI audit, and release scripts
- `configs/` — Experiment and model configurations
- `tests/` — Unit test suite
- `docs/leaderboard.md` — 12-model × 9-frozen-set evaluation leaderboard
- `docs/benchmark/` — MedDecide-Bench item schema and dataset documentation
