# ADVISORY — osler_v7 (pre-registered plan)

Status: **PROPOSED, not yet started.** The numbers in §4–§7 are proposals from the advisor review of 2026-11-12. §0 lists the
decisions the operator must settle before T6 (the first training arm). Once the loop starts, this file is frozen (GOAL.md).

## 0. Open decisions (operator)
1. **Gate limits (§7).** Confirm or change the margins. Proposed: non-inferiority margin −0.01, audit gain floor 0.
2. **KL weight (§3).** Proposed 0.1 (osler_v3's value that narrowed the panel gap; 0.5 is run only if 0.1 fails).
3. **Confirmation set (§8).** Who builds it and from which post-window source. Proposed: DeepSeek builds it from records dated
   after 2026-03-01 with structured gold, reviewed by the operator before first use.
4. **Declared dependency.** `peft` is imported by `src/meddecide/train/trainer.py` and `src/meddecide/model/pplx.py` but is not
   declared in `pyproject.toml`. Declaring it changes `uv.lock`. Needs a decision before T6.
5. **Teacher (§2).** Confirmed by the operator: pplx-decider-v1.2-27b, local. d1-ultra is not used for MIMIC items.

## 1. Targets and bases
- **4B:** `perplexity-ai/pplx-decider-flash-v1.2` (Qwen3.5-4B lineage; noncausal full attention on full-attention layers;
  255-row option-code readout at the marker after each question).
- **9B:** `vllm-sr/d3-flash` (rev 67e03e1bc157961c582ee7c6a35d9b15fada3dec; Qwen3.5-9B lineage; the student parity files record
  `noncausal_full_attention`; 255-row readout at the last prompt position, per the d3 runtime).
- The two protocols differ in readout position. Each model keeps its own protocol. No cross-protocol claim is made.
- Each model is gated against **its own zero-shot base**, never against the other model or a sibling arm (D16, D24, D31).

## 2. Data cells (labels by construction; training splits only)
All cells come from training splits or from records used only for training. Every filter returns counts and reasons.

| Cell | Content | Gold | Size (target) |
|---|---|---|---|
| C1 many-option diagnosis | MIMIC presentation (chief complaint, HPI) as state; 5–50 options; same-chapter distractors | primary ICD chapter/code from structured fields | 20,000 |
| C2 two-option equivalence | balanced yes/no pairs on clinical text: paraphrase and minimal-edit | from the construction (edit type fixes the label) | 10,000 |
| C3 code and concept | ICD, ATC, procedure dictionaries: code-to-description, hierarchy, sibling distractors | from the dictionary | 20,000 |
| C4 option-count augmentation | existing 4-option train items expanded to 8–10 options; gold unchanged | from the source item | 20,000 |
| C0 general replay | official general training splits, existing v6 mix | from the source | as in osler_v6 mix |

- Teacher labels: pplx-decider-v1.2-27b, run locally, on C1–C4 and the unlabelled KL pool (§3). Never on dev, test, panel or
  held-out items.
- MIMIC-derived cells: local processing only; patient-level splits; the PHI-leak audit precedes any release (D28).
- Balance rule: each family's share of the loss gradient is capped at 25 % (per-family weighting, logged in the mix manifest).

## 3. Lever 1: keep the base in training (KL to base)
- Loss = CE(gold) + λ_KL · KL(student ‖ zero-shot base) on the unlabelled format-diverse pool. The base is the same model before
  training. Proposed λ_KL = 0.1; λ_KL = 0.5 only if 0.1 fails the kill rule (§9).
- The unlabelled pool is the C1–C4 prompts without gold. It never contains dev, test or panel items.

## 4. Lever 2: interpolation at selection
- Candidate weights: θ(s) = θ_base + s · (θ_trained − θ_base), for s ∈ {0.25, 0.5, 0.75, 1.0}. Scale applies to LoRA and readout
  deltas together (or separately under Lever 3's result).
- s is chosen on format-matched dev only (§6).

## 5. Lever 3: locate the damage (diagnostic, inference only)
- On the osler_v6 final (the v6 checkpoint at 63,304 is the only available trained model): grid LoRA scale {0.5, 0.75, 1} ×
  readout scale {0, 0.5, 1} on the format-matched dev set (§6).
- Purpose: tell whether the two-option and two-letter losses sit in the readout. If base readout plus trained LoRA recovers them,
  the readout is anchored in T6–T8 (readout frozen or KL-anchored). The grid does not select a model.
- Reported as a diagnostic with its own CLAIMS rows. No gate uses it.

## 6. Format-matched dev set (non-panel; never trained on)
- Built from: MIMIC held-out patients (patient-level split), official training splits held out for dev, and synthetic-free sources.
- Must cover: many-option diagnosis (10–50), two-option pairs, code items, 4-option and 10-option MCQ, the MR verdict template
  family, and the MIMIC audit templates.
- Size target ≥ 3,000 items. Held-out codes and patients (never seen in training).
- Frozen before T6 (manifest with SHA-256 and counts).

## 7. Selection and gate (proposed; §0.1 to confirm)
- **Selection (dev only):** a candidate (base × s × recipe) is eligible if its paired difference against its own zero-shot base on
  the format-matched dev set has a lower bound > −0.01 (non-inferiority). Among eligible candidates, choose the highest dev S (the
  S composite already defined in `loops/osler_v6/ADVISORY.md`, §2.7). If none is eligible, the arm is reported and stops.
- **Gate v7 (on the confirmation set, §8; run once per model):**
  - G-a: paired difference against the model's own zero-shot base, on the confirmation set's general items: lower bound > −0.01.
  - G-b: paired difference on the audit items (MR verdict and MIMIC templates): lower bound > 0 (audit gain kept).
  - G-c: knowledge items: lower bound > −0.02.
- The panel is reported as an additional row only, never as a gate input.
- Any check on a set that does not exist yet is **NOT MEASURED**. A model with an unmeasured check does not pass the gate.

## 8. Confirmation set (fresh, tier 2)
- Built from records dated after 2026-03-01 with gold from structured source fields only (the program's tier-2 rule).
- Built and manifested before any arm is trained (T2). It is used once, at the gate. Its items are never used for selection.
- Until it exists, Gate v7 is **NOT MEASURED**, and no frontier claim is made.

## 9. Kill rules and stops
- If neither 4B arm is eligible on the format-matched dev set (§7), stop the 4B track and report the best scaled checkpoint.
- If the 9B arm is not eligible, the 9B track stops the same way.
- A 63,304-step final is run only for an arm that is eligible and has a confirmation set (T11). It costs about 54 hours at
  3.1 seconds a step.
- If an arm fails the positional-health check that osler_v6 used (all three orders), it is BLOCKED, not chosen.

## 10. Budget (estimates; measured in T5 and T6)
- T4 grid: inference only, hours.
- T5 labelling of the pool with pplx-27B: unknown until measured (the d3 teacher labelled 200,000 items in about 2 hours at
  26 items/s on the batched path; the 27B's rate is measured in T5).
- T6 arm (4B) and T8 arm (9B): about 14,000 steps each, about 12 hours at the osler_v6 pace.
- T11 finals: 54 hours each, only for a passing arm.
