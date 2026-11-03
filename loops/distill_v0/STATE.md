# distill_v0 — state

Loop status: `RUNNING` (autonomous; operator go 2026-11-04 in chat; hard stop after Gate O7)
Run started (UTC): `2026-11-03T19:30:00Z`
Last updated (UTC): `2026-11-03T19:30:00Z`
Iterations so far: `0`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| D0 | Pre-registration: GOAL, ADVISORY, STATE, CLAIMS; AGENTS current-loop pointer | no | — | DONE | 2026-11-03T19:30Z | 2026-11-03T19:40Z |
| D1 | Teacher ceiling on all frozen sets; labeler, cost ledger, freeze schema | no | D0 | PENDING | | |
| D2 | Basket manifest; label run (cap $15); QA; frozen label file | no | D1 | PENDING | | |
| D3 | Clef-Flash port check (3 h timebox); fallback decision | yes | D0 | PENDING | | |
| D4 | Distillation driver + λ calibration + 4B run (63,304 steps) | yes | D2 | PENDING | | |
| D5 | 9B run (63,304 steps), same labels and recipe | yes | D3, D4 | PENDING | | |
| D6 | Gate O7 for both students; scoreboard; SELF_AUDIT | yes | D4, D5 | PENDING | | |
| D7 | FINDINGS.md; closure; hard stop | no | D6 | PENDING | | |

Rules: take the **first** `PENDING` task whose dependencies are all `DONE`. Never run two GPU jobs at once. A `BLOCKED` task does not block unrelated work. No row may be left `IN_PROGRESS` at closure.

---

## 2. Key values

**Operator confirmations (2026-11-04, chat; not edited by the loop):**
- Two students: `pplx-decider-flash-v1.2` (4B) and `clef-flash` (9B): yes.
- Objective: best open model under 10B overall (medical ≤10B Pareto frontier): yes.
- Frozen test items never used for training: yes.
- No new teacher-only synthetic task templates: yes.
- `d1-ultra` primary teacher; local 27B backup only if the API fails: yes.
- API budget: balance $23.44; hard cap $15.00; never above $50: yes.

**Model revisions**
- `perplexity-ai/pplx-decider-flash-v1.2` — `6d771ad7b5ecff5e93b8fc7b90d4c5ccc341954d` (4B base)
- `Cloudflare/clef-flash` — `fde727a287004204b7518dcc983fe64379776712` (9B base)
- `Qwen/Qwen3.5-9B` — `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (9B fallback)
- `perplexity-ai/pplx-decider-v1.2-27b` — `cce0f3f80033e9d89d80831d5281fd72124aacce` (teacher fallback)
- `liquid/d1-ultra-20261015` — pinned teacher id (the API resolves the alias to this)

**Costs**
- Measured teacher price: `$0.042 / 1M input tokens` (1.4M tokens).
- OpenRouter balance at start: `$23.44` (credits endpoint, 2026-11-03).
- Hard cap: `$15.00`; estimated core basket: `$7.0` + ceiling `$1.5`.

**Frozen-set sizes** (evaluation only): knowledge 5,456; panel 4,353; MR verdict 13,103; MIMIC suite 4,253; MedNLI 1,422; pairs 500; MIMIC held-out 110; MR reported 6,191; MR held-out 584.

---

## 3. D0 checklist

- [x] `loops/distill_v0/GOAL.md` written (mission, design, gate, tasks, hard stop)
- [x] `loops/distill_v0/ADVISORY.md` written (pre-registered design: teacher, basket, loss, training, gate)
- [x] `loops/distill_v0/CLAIMS.md` header written
- [x] `loops/distill_v0/STATE.md` written (this file)
- [x] `AGENTS.md` "Current loop" replaced with distill_v0
- [x] branch `loop/distill_v0` committed and pushed

---

## 4. Iteration log

- `2026-11-03T19:30Z` — D0 started. Branch created from `loop/osler_v5` at `23a0362` (the tree with `corrected_mix`, `option_augment`, `osler_arm`, `z5_*`). Teacher and basket measured before the loop: d1-ultra panel 0.750 (n=420), knowledge 0.792 (n=120), MIMIC suite 0.683 (n=360), MR 0.707 (n=420); local 4B final panel 0.597, knowledge 0.702, MIMIC 0.800, MR 0.843. These are sample estimates, not gate results.
- `2026-11-03T19:40Z` — D0 DONE. Pre-registration committed: GOAL, ADVISORY, STATE, CLAIMS; AGENTS current-loop pointer moved to distill_v0. The $15 API cap and the frozen-test exclusion are in force. Next: D1 (teacher ceiling + labeler).

---

## 5. Questions for the operator

None. All decisions needed for D0–D7 were answered in chat on 2026-11-04. Anything that arises is recorded under §6 and handled with the conservative reading.

---

## 6. Deviations from the plan

None yet.

---

## 7. Open items / risks

- **Teacher determinism:** labels are frozen once; a rerun will not reproduce them exactly. The frozen file hash is the provenance.
- **API budget:** the $15 cap may end the basket before the optional rows. The labeler prioritises knowledge/general first.
- **Clef port:** the text tower is Qwen3.5-9B-shaped (`model.language_model.*` keys); the port loads `Qwen3_5ForConditionalGeneration` and takes `.model`. If it is not clean in 3 h, fall back to `Qwen/Qwen3.5-9B` (recorded).
- **9B memory:** batch 8 × 8,192 tokens with LoRA + checkpointing is budgeted but not yet measured on the 9B. If OOM, reduce `max_batch_tokens` first (documented), not the steps.
- **No 9B control:** the 9B has no no-distill sibling in this loop; its comparison is to Clef-Flash zero-shot and the 4B. Any causal claim about distillation is a hypothesis (R10).
