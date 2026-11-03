# distill_v0 — state

Loop status: `RUNNING` (autonomous; operator go 2026-11-04 in chat; hard stop after Gate O7)
Run started (UTC): `2026-11-03T19:30:00Z`
Last updated (UTC): `2026-11-03T20:46:00Z`
Iterations so far: `3`

---

## 1. Task board

Statuses: `PENDING` → `IN_PROGRESS` → `DONE` | `BLOCKED — reason`

| id | task | GPU | deps | status | started (UTC) | finished (UTC) |
|---|---|---|---|---|---|---|
| D0 | Pre-registration: GOAL, ADVISORY, STATE, CLAIMS; AGENTS current-loop pointer | no | — | DONE | 2026-11-03T19:30Z | 2026-11-03T19:40Z |
| D1 | Teacher ceiling on all frozen sets; labeler, cost ledger, freeze schema | no | D0 | DONE | 2026-11-03T19:38Z | 2026-11-03T19:59Z |
| D2 | Basket manifest; label run (cap $15); QA; frozen label file | no | D1 | IN_PROGRESS | 2026-11-03T20:02Z | |
| D3 | Clef-Flash port check (3 h timebox); fallback decision | yes | D0 | DONE | 2026-11-03T20:28Z | 2026-11-03T20:45Z |
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

## 3b. D1 checklist

Acceptance (GOAL §4): 9 sets scored; cost, model id, date, token counts recorded; parity test.

- [x] Reusable teacher client in `src/meddecide/distill/d1_teacher.py` and `d1_client.py` (request construction, response validation, freeze schema, cost ledger, HTTP retries) + unit tests (37)
- [x] Hosted ceiling run on all 9 frozen sets. The first five (knowledge, panel, mr_verdict, mr_reported, mr_heldout_test) were labelled 19:41–19:58Z; the four MIMIC-derived sets (mednli_test, mimic_suite, pairs_test, mimic_heldout) were labelled 19:55–19:58Z after the operator's authorization of 2026-11-03 (§6)
- [x] Cost ledger + manifest: model id, endpoint, date, git commit, input/output token counts, per-file SHA-256, failures and skips with reasons (`allow_protected` and the authorization text recorded)
- [x] Parity test: a fixed 40-item panel sample labelled twice; 37/40 argmax agreement, max |Δp| 0.2083, mean 0.0183
- [x] Ceiling table per set (micro, template macro, n, per qtype) and per-template counts (`ceiling.json`)
- [x] SELF_AUDIT.md (R6): 20 checks re-derived in a fresh process; denominators; majority baselines; flagged items (mr_reported 1.000 cue-solvability, mr_heldout below majority, the one http_422, score-mode definition)
- [x] CLAIMS rows C-D1-01…13; STATE + iteration log

MIMIC note: on 2026-11-03 the operator wrote in chat that MIMIC-derived datasets may be used 100 % and that the provider's email confirmations are in hand. The hosted guard therefore takes an explicit `allow_protected` flag, and the D1 manifest records the authorization; MIMIC content is still never committed. Training keeps the audit family gold-only (measured decision, GOAL §3.3), so the authorization widens the D1 ceiling only, not the T basket.

---

## 3c. D3 checklist

Acceptance (GOAL §4): forward + readout init works, or fallback recorded.

- [x] Load the pinned Clef snapshot's text tower (`Qwen3_5ForConditionalGeneration.model.language_model` + `lm_head`) as a causal LM, in memory (no 18 GB disk copy: `src/meddecide/model/clef.py`)
- [x] `MedDecideModel` accepts a caller-built backbone (`base_override`); readout rows are initialised from `lm_head` — checked equal (`rows_equal_lm_head: true`), 255 codes
- [x] 20-item dev smoke on fastdev v4: probabilities sum to 1 (min 0.99999993), 6 distinct argmax keys, 11/20 correct (a code-path check, not a result)
- [x] Backward pass: LoRA gradient norm 130.2, option-code head 71.0; peak GPU 28.1 GB
- [x] Verdict `PORT_OK` at 2026-11-03T20:44:39Z, wall 15.7 s after load, GPU RTX PRO 6000 Blackwell
- [x] Fallback not needed (`Qwen/Qwen3.5-9B` recorded only as the pre-registered alternative)

---

## 4. Iteration log

- `2026-11-03T19:30Z` — D0 started. Branch created from `loop/osler_v5` at `23a0362` (the tree with `corrected_mix`, `option_augment`, `osler_arm`, `z5_*`). Teacher and basket measured before the loop: d1-ultra panel 0.750 (n=420), knowledge 0.792 (n=120), MIMIC suite 0.683 (n=360), MR 0.707 (n=420); local 4B final panel 0.597, knowledge 0.702, MIMIC 0.800, MR 0.843. These are sample estimates, not gate results.
- `2026-11-03T19:40Z` — D0 DONE. Pre-registration committed: GOAL, ADVISORY, STATE, CLAIMS; AGENTS current-loop pointer moved to distill_v0. The $15 API cap and the frozen-test exclusion are in force. Next: D1 (teacher ceiling + labeler).
- `2026-11-03T19:38Z` — D1 started: hosted client (`src/meddecide/distill/d1_teacher.py`, `d1_client.py`), unit tests, ceiling driver, parity driver, re-derivation audit.
- `2026-11-03T19:41Z` — D1 ceiling run launched (5 authorised sets, 29,687 rows); completed 19:58Z with 1 `http_422` failure and $0.6998 spend.
- `2026-11-03T19:55Z` — operator authorization received in chat: MIMIC-derived datasets allowed 100 %, email confirmations obtained. D1 resumed with `--allow-protected` and the authorization recorded; the 4 MIMIC-derived sets (6,285 rows) labelled by 19:58Z.
- `2026-11-03T19:58Z` — D1 DONE. Ceiling (micro, n): knowledge 0.8167 (5,456), panel 0.6825 (4,353), mr_verdict 0.7387 (13,102), mednli 0.8812 (1,422), mr_reported 1.0000 (6,191), mr_heldout 0.8322 (584), mimic_suite 0.6901 (4,253), pairs 0.9360 (500), mimic_heldout 0.7364 (110). Total spend $1.27091, 30.26M tokens, 1 failure. SELF_AUDIT 20/20 PASS; CLAIMS C-D1-01…13. mr_heldout is below its majority baseline (0.8510) and mr_reported is cue-solvable (1.0000) — both flagged.
- `2026-11-03T19:59Z` — D1 parity: 37/40 argmax agreement between two identical calls, max |Δp| 0.2083, mean 0.0183.
- `2026-11-03T20:28Z` — D3 started (Clef port). The first design (convert the text tower to a `Qwen3_5ForCausalLM` checkpoint on disk) hit the workspace quota (18 GB copy, `EDQUOT`); replaced by an in-memory loader (`meddecide.model.clef`) plus a `base_override` path in `MedDecideModel`. Two PEFT attribute probes (`base_model_prefix`, `prepare_inputs_for_generation`) and one output-shape mismatch were fixed; `PORT_OK` at 20:44:39Z (255 codes, readout rows equal lm_head, 20-item smoke, backward grads, 28.1 GB peak).
- `2026-11-03T20:02Z` — D2 started (basket + label run). Plan found: pool 1,268,815 (matches); rule-lookup drop 187,623 (matches); T basket 494,803 rows (pre-registration says 508,516: nine templates hold fewer rows than listed, two CT templates are below the 10,000 cap; teacher share 39.0 % is inside the 38–45 % band — deviation recorded). Pool has 5,232 duplicated item ids (Winogrande twins, 27,340 rows) so the training basket is keyed by text key (added field `text_key` on D2 freeze rows); 1,826 duplicate text keys / 8,478 rows, of which 71 keys carry conflicting gold (reported, inherited from the mix).

---

## 5. Questions for the operator

None. All decisions needed for D0–D7 were answered in chat on 2026-11-04. Anything that arises is recorded under §6 and handled with the conservative reading.

---

## 6. Deviations from the plan

1. **MIMIC and the hosted teacher (2026-11-03, operator instruction in chat).** The operator wrote that MIMIC-derived datasets may be used 100 % and that the provider's email confirmations are in hand. D60 required the operator to confirm the BAA before any MIMIC item is sent; this instruction is that confirmation. The `hosted_guard` gained an explicit `allow_protected` flag (default off); the D1 manifest records `allow_protected: true` and the authorization text. MIMIC content is still never written into a committed file. The **training** basket is unchanged: the audit family stays gold-only on the measured evidence (teacher 0.6901 vs tuned student 0.800 on the MIMIC suite).
2. **D1 ceiling on 9 sets, not 6 (scope widened by the same authorization).** The pre-registered D1 was "9 sets scored"; the rule as then written allowed only 5. The authorization restored the pre-registered scope rather than changing it.
3. **D2 basket counts (2026-11-03, recorded, not silent).** The composed pool matches the pre-registered 1,268,815 rows and the 187,623-row rule-lookup drop exactly, but the T basket is **494,803 rows, not 508,516** (delta −13,713). Cause: the GOAL §3.3 per-template counts are stale for nine templates — `ct_intervention_type_choice_v1` holds 6,318 rows and `ct_primary_purpose_choice_v1` 4,237 (below the 10,000/template cap), and seven broad replay templates are smaller than listed (`broad_aqua_rat_v1` 69,980 vs 72,469, etc.). Every available row of every listed template is in the basket; nothing is topped up from other families. Teacher share 39.0 % (band 38–45 %) and MIMIC share 39.3 % (band 37–42 %), so the dose is inside the pre-registered bands.
4. **D2 freeze schema adds `text_key`.** Winogrande twins share an `item_id` while their state text differs (5,232 ids / 27,340 pool rows), so an id-keyed label file would collapse distinct rows. D2 rows carry the normalised text key (the repository's existing disambiguator, osler_v4 Y5) and the labels attach by it; D1's ceiling rows keep the eight-field schema where ids are unique.
5. **D1's resumed ledger.** The MIMIC extension seeded the ledger from the already-frozen labels, so the manifest's cost/token totals are whole-task totals rather than the second run's only. The manifest reports `attempted_this_run: 6,286` separately.

---

## 7. Open items / risks

- **Workspace quota (2026-11-03).** The `/workspace` volume is near its quota (≈450 GB used, ~4.5 GB free before cleanup); a 6 GB write failed with `EDQUOT`. `/workspace/tmp` was cleaned (24 GB → 1 GB, allowed scratch) and the failed D3 18 GB copy removed. Free space is now ~32 GB. Large intermediate copies are avoided (D3 loads in memory); check free space before any new large artifact and keep the allowed scratch clean.
- **Pre-existing strict-xfail in the suite.** `tests/test_padding_trained.py::test_trained_checkpoint_logits_are_padding_invariant` is marked `xfail(strict=True)` (V2 BLOCKED) and now XPASSES in this environment, which pytest counts as a failure. It loads a student_v0 checkpoint and does not import the trainer; it is not caused by this loop's code and was not edited (R8). The rest of the suite passes.

- **Teacher determinism:** labels are frozen once; a rerun will not reproduce them exactly. The frozen file hash is the provenance.
- **API budget:** the $15 cap may end the basket before the optional rows. The labeler prioritises knowledge/general first.
- **Clef port (resolved 2026-11-03T20:44Z):** the text tower loads in memory from the pinned snapshot (`Qwen3_5ForConditionalGeneration.model.language_model` + `lm_head`) and the readout is initialised from `lm_head`; verdict `PORT_OK`. No fallback needed. The 9B config uses `backbone: clef`.
- **9B memory:** batch 8 × 8,192 tokens with LoRA + checkpointing is budgeted but not yet measured on the 9B. If OOM, reduce `max_batch_tokens` first (documented), not the steps.
- **No 9B control:** the 9B has no no-distill sibling in this loop; its comparison is to Clef-Flash zero-shot and the 4B. Any causal claim about distillation is a hypothesis (R10).

---

## 8. Next actions (for a resumed/compacted context)

1. **D2 label run in progress** (`outputs/distill_v0/D2/logs/run3.log`). Resume command if the process is gone:
   `source /workspace/.secrets.env && setsid nohup uv run --frozen python scripts/distill_v0/d2_basket.py --out outputs/distill_v0/D2 --threads 48 --resume --allow-protected --authorization "operator chat 2026-11-03: MIMIC-derived datasets allowed 100%; provider email confirmations obtained" > outputs/distill_v0/D2/logs/run3.log 2>&1 < /dev/null &`
   Then `uv run --frozen python scripts/distill_v0/d2_rederive.py --out outputs/distill_v0/D2`, SELF_AUDIT, CLAIMS, STATE, commit.
2. **D4 (4B)** after D2 completes: `bash scripts/distill_v0/watchdog.sh configs/distill_v0/student_4b.yaml outputs/distill_v0/D4 63304` (GPU; ~13–14 h). A `--dry-run` first checks the pool hash, label hash, attachment and item conversion without loading a model.
3. **D5 (9B)** after D4: same watchdog with `configs/distill_v0/student_9b.yaml` and `outputs/distill_v0/D5` (~25 h). Uses the in-memory Clef text tower (D3 `PORT_OK`).
4. **D6**: write `scripts/distill_v0/d6_gate.py` (score both students on the 9 y9 sets with their saved temperatures, `gate_o4.pair_rows`/`comparison` for checks i–iv, Gate O7 verdict, ≤10B frontier table, SELF_AUDIT). Baselines: `outputs/osler_v4/Y9/baselines/{pplx,final,md4b,md9b,jev9b,pplx27b,osler_v2}_<set>/predictions.jsonl`; Clef-Flash and Qwen zero-shot in `outputs/osler_v5/Z10/`.
5. **D7**: FINDINGS.md, CLAIMS, commit + push, hard stop.
