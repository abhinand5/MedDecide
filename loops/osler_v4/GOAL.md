# GOAL — osler_v4 (MIMIC at scale, Nemotron-crafted minimal pairs, the panel lever, a longer final)

Osler is an open, calibrated medical decision model. osler_v2's Osler-4B (pplx-decider-flash-v1.2 base) leads the
record-audit suite by +15.8 points over its base and +26.3 over MedDecider-4B. It failed Gate O3 on one check:
the external panel against its own base (−0.020 [−0.034, −0.006], limit −0.02 on the lower bound). osler_v3 kept no new block and chose the control recipe.

This loop aims to make Osler-4B **the best open decision model of its size** on real medical-record audit. It has five
parts:
- **MIMIC:** official MIMIC-IV records become payer-audit training data. Labels come from database fields: ICD-10 coding,
  MS-DRG and CC/MCC, APR severity levels, ICD-10-PCS, lab-based clinical validation, eMAR, and radiology exams.
- **Minimal pairs:** the local Nemotron 3.5 Super removes the documentation of a coded condition from a real note, with
  independent verification.
- **The panel lever:** KL 0.1 plus broad replay, and (if the operator confirms D48) distillation from pplx's 27B on general
  items, chosen on a stronger clinical-general dev proxy.
- **Keep rule v2:** blocks that teach a new skill without harm are kept.
- **The final:** a run 1.5× longer than osler_v2's, then **Gate O4** with a frozen MIMIC audit suite. It ends with a hard
  stop.

## Read first, in this order

1. `AGENTS.md` (repo root): project rules, R1–R10, loop mechanics, guardrails.
2. `docs/plans/PROGRAM.md`, note D41–D50.
3. `loops/osler_v3/FINDINGS.md` and `loops/osler_v3/NEXT.md`.
4. `loops/osler_v4/ADVISORY.md`: this loop's work plan. Read it in full at the start of every iteration.
5. `loops/osler_v4/STATE.md`: what has already been done. If it still contains `<fill in bootstrap>`, start with Y0.

Do not improvise the plan. Where STATE.md disagrees with your recollection, the file wins.

## Each iteration

1. Read STATE.md.
2. Take the **first** `PENDING` task whose dependencies are `DONE` (order exceptions in ADVISORY §5).
3. Mark it `IN_PROGRESS` with a timestamp from `date -u +%FT%TZ` and copy its steps into the STATE checklist.
4. Execute it, run its acceptance check, and self-audit (R6).
5. Append to `loops/osler_v4/CLAIMS.md`.
6. Mark it `DONE` or `BLOCKED — <reason>` and log it.
7. Commit (`loop(osler_v4): Y<n> <DONE|BLOCKED> — <headline>`) and push `loop/osler_v4`.

A `BLOCKED` task does not block unrelated work. Record it, write the question into STATE.md's questions section, and
move on. When a judgement call is needed, take the conservative option, record it under Deviations, and continue.

**Long GPU jobs:**
- launch them detached, with logs under `outputs/osler_v4/<task>/logs/`;
- record the PID in STATE;
- poll at intervals of 20 minutes or more;
- run one GPU job at a time; the Nemotron server **is** a GPU job, so stop it before any training or scoring;
- check outputs for completeness, not just exit codes.

## The contract

Every number carries its artifact and a recompute command (CLAIMS.md). `NOT MEASURED — <reason>` and
`BLOCKED — <reason>` are valid results; a fabricated number is a critical failure.

Gate O4's baselines (D49) are exactly MedDecider-4B and zero-shot pplx-decider-flash-v1.2. Every other model, every Stage
A or Stage B run, osler_v2's Osler-4B and pplx's 27B are additional rows. The 9B is deferred.

Held-out items, test sets, the MIMIC audit suite, MDACE and the external panel never influence any choice. If a result is
unwelcome, report it and name the ADVISORY §2 outcome it implies.

## Do not

- Send MIMIC text, values or identifiers to any non-local service. The teacher client's loopback guard (Y0) must stay on.
  Never point MIMIC work at `OPENAI_*` or `TEACHER_*` hosted endpoints.
- Commit, or paste into STATE, CLAIMS or any committed file: MIMIC text, values, ids or snippets; benchmark items;
  generated or teacher items; training data; predictions; checkpoints; keys.
- Use any unofficial copy of MIMIC or other credentialed data (D37).
- Touch `/workspace/download-manager/` or its process: no stop, start or edit. Read `./dm status` only.
- Train on any test split, the external panel or any copy of a panel dataset (e.g. `MaziyarPanahi/MedExQA-clean`),
  MDACE, the knowledge dev v2, CG items or held-out items:
  - the ICD family `heart_failure`;
  - the indicator `aki_creatinine_rise` and any AKI clinical-validation item;
  - osler_v0's held-out templates and generators.
- Use a teacher's answer, option distribution or correctness judgement as a label. The only exceptions are D41 (base KL)
  and, if confirmed, D48 (27B soft labels on general items only).
- Keep a Nemotron-crafted item that failed any verification check, or relabel one.
- Use CPT descriptors, 3M APR-DRG descriptions, InterQual or MCG content, or anything from a payment-integrity vendor.
- Change D35, D47, D49 (or the frozen MR suite v1 and MIMIC audit suite v1), the Stage A choice rule, the KL anchor
  prefixes, the held-out lists or the ICD table after any run, or to make something pass (that is `BLOCKED`).
- Change the head, the attention, the base model or the adapter (D26, D33).
- Run two GPU jobs at once, or spawn sub-agents for GPU work.
- Edit earlier `loops/*/`, `data/bench/v0.1/`, `data/bench/v0.2/`, `data/bench/v0.3_ext/`, earlier `outputs/*/`,
  `AGENTS.md` (except Y0's "Started" field) or `docs/plans/`.
- Delete, stop or reconfigure pods, volumes, HF repos or remote services; upload anything; push to the HF Hub.
- Fall back to Gemma or any hosted teacher without the operator.
- Start the next loop, or anything in ADVISORY §6.

## Done when

- Every task is `DONE` or `BLOCKED — <reason>`; none is left `IN_PROGRESS`.
- `mimic.md`, `crafted.md`, `stage_a.md`, `blocks.md`, `gate_o4.md`, `results.md` and `audit.md` exist, as far as their
  tasks ran.
- `FINDINGS.md` exists: Summary first, and the five-claim spot-check appended with its re-run outputs.
- Every reported number has a CLAIMS row.
- `NEXT.md` lists proposals.
- Everything committable is committed and pushed.

Then set `Loop status: STOPPED` at the top of STATE.md, commit, push, **stop**, and report:
- which tasks completed;
- which blocked and why;
- the Gate O4 verdict;
- what you would do next.

Also set `Loop status: STOPPED` (with FINDINGS for what ran) when every remaining `PENDING` task is blocked on the
operator.
