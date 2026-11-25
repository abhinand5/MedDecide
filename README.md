# MedDecide & Osler

**Osler** is a family of open, calibrated **medical decision models** ("System One" single-pass): given a clinical state (a discharge note, trial record, FDA label, or medical question) and a list of typed questions (`noul` yes/no, `choice` over 2–255 options, `score` over 2–10 ordered levels), they return a calibrated probability distribution over all allowed options in **one prefill pass** — no text generation.

Alongside the models, this repository provides **MedDecide-Bench**, a unified 9-set (35,769-item) evaluation harness combining established medical exams, a 7-dataset out-of-domain external panel, a contamination-resistant post-cutoff tier (`≥ 2026-03-01` structured-field gold), clinical record audits, and MIMIC-IV payer-audit tasks.

## Released Models (Apache-2.0 — Standalone Merged Weights)

Both models are published as **complete, merged safetensors checkpoints** with bundled runtimes and built-in dev-only per-type calibration — **no `meddecide` library is needed**.

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

| Model | Hugging Face | Base Backbone | Params | 5-Set Macro | OOD Avg | Clinical Audit Avg | Batch-1 p50 | Peak VRAM |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Osler-9B** | [`abhinand/Osler-9B`](https://huggingface.co/abhinand/Osler-9B) | `vllm-sr/d3-flash` | 8.39B | **0.7801** | 0.7419 | **0.8375** | 33.71 ms | 17.32 GB |
| **Osler-4B** | [`abhinand/Osler-4B`](https://huggingface.co/abhinand/Osler-4B) | `perplexity-ai/pplx-decider-flash-v1.2` | 4.55B | 0.7530 | 0.7038 | 0.8269 | 26.54 ms | 9.60 GB |
| *`pplx-27B` (27B ceiling)* | `perplexity-ai/pplx-decider-v1.2-27b` | — | 26.09B | 0.7600 | **0.7725** | 0.7411 | 90.96 ms | 52.92 GB |

- **Global 2-axis Pareto frontier (OOD General vs. Clinical Audit):** **Osler-9B** is the only ≤10B model on the global front alongside `pplx-27B`, achieving the highest 5-set macro of all 12 evaluated models at **2.7× lower latency** and **3.1× lower VRAM**, and strictly dominating every comparable ≤10B model (`Osler-4B`, `pplx-flash`, `MedDecider-9B/4B`, `Clef-Flash`) across all five headline sets.
- **Privacy & safety:** Both models cleared a mandatory local MIMIC-IV PHI/memorisation audit (`loops/osler_rc0/AUDIT.md`); **Osler-9B** carries no `lm_head` (architecturally non-generative), and **Osler-4B** yielded `0/300` verbatim matches and `20/20` token-identical base generation when its adapter is disabled.
- **Honest limitations:** `Osler-9B` achieves parity (not a gain) against its `d3-flash` base on OOD `panel` and `knowledge` and trails `pplx-27B` on hard multi-step QA; `Osler-4B` trades ~1 pt on OOD `panel`/`knowledge` for large clinical audit gains; dev-fitted temperatures leave both models under-confident on fresh `confirm_general`; and both models fail a held-out synthetic indicator template (`synthea_indicator_evidence_v1`). See [`docs/paper/PAPER_GUIDE.md`](docs/paper/PAPER_GUIDE.md) and the model cards for full details.

## Key Documentation & Artifacts

- **Paper Drafting Guide (self-contained):** [docs/paper/PAPER_GUIDE.md](docs/paper/PAPER_GUIDE.md)
- **Reporting Database (DuckDB, 21 aggregate tables):** [docs/paper/osler_rc0_reporting.duckdb](docs/paper/osler_rc0_reporting.duckdb)
- **Unified 12-Model Leaderboard:** [docs/leaderboard.md](docs/leaderboard.md) and [loops/osler_rc0/LEADERBOARD.md](loops/osler_rc0/LEADERBOARD.md)
- **Pareto Frontier & Paired Comparisons:** [loops/osler_rc0/PARETO.md](loops/osler_rc0/PARETO.md)
- **MIMIC PHI / Memorisation Audit (D28):** [loops/osler_rc0/AUDIT.md](loops/osler_rc0/AUDIT.md)
- **RC0 Synthesis & Model Cards:** [loops/osler_rc0/FINDINGS.md](loops/osler_rc0/FINDINGS.md), [MODEL_CARD_9B.md](loops/osler_rc0/MODEL_CARD_9B.md), [MODEL_CARD_4B.md](loops/osler_rc0/MODEL_CARD_4B.md)
- **Benchmark Schema & Program Plan:** [docs/benchmark/schema.md](docs/benchmark/schema.md), [docs/plans/PROGRAM.md](docs/plans/PROGRAM.md)
- **Operating Guide:** [AGENTS.md](AGENTS.md)
