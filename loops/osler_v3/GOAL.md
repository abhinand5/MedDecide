# GOAL — osler_v3 (keep the base's general skill with KL-to-base; real sources + Gemma-crafted real-text items)

Osler is an open, calibrated medical decision model. osler_v2's Osler-4B, on the pplx-decider-flash-v1.2 base, won the
record-audit suite by +15.8 points over its base and +26.3 over MedDecider-4B. It failed Gate O3 only on the external panel
against its own base (−2.0), which points to forgetting.

This loop decides by experiment:
- **Stage A:** four matched recipe runs (control; KL-to-base at 0.1 and 0.5; broader replay), with the recipe chosen by a
  pre-registered rule on a clinical-general dev proxy.
- **Stage B:** data blocks on that recipe: real sources (RAGTruth, DrugProt, MultiCaRe), Gemma-crafted items over real
  text, synthetic charts, and official MIMIC if it arrives.

Then it trains the final Osler-4B and applies Gate O3. It ends with a hard stop.

## Read first, in this order

1. `AGENTS.md` (repo root): project rules, R1–R10, loop mechanics, guardrails.
2. `docs/plans/PROGRAM.md`, note D33–D44.
3. `loops/osler_v2/FINDINGS.md` and `loops/osler_v2/NEXT.md`.
4. `loops/osler_v3/ADVISORY.md`: this loop's work plan. Read it in full at the start of every iteration.
5. `loops/osler_v3/STATE.md`: what has already been done. If it still contains `<fill in bootstrap>`, start with X0.

Do not improvise the plan. Where STATE.md disagrees with your recollection, the file wins.

## Each iteration

1. Read STATE.md.
2. Take the **first** `PENDING` task whose dependencies are `DONE` (order exceptions in ADVISORY §6).
3. Mark it `IN_PROGRESS` with a timestamp from `date -u +%FT%TZ` and copy its steps into the STATE checklist.
4. Execute it, run its acceptance check, and self-audit (R6).
5. Append to `loops/osler_v3/CLAIMS.md`.
6. Mark it `DONE` or `BLOCKED — <reason>` and log it.
7. Commit (`loop(osler_v3): X<n> <DONE|BLOCKED> — <headline>`) and push `loop/osler_v3`.

A `BLOCKED` task does not block unrelated work. Record it, write the question into STATE.md's questions section, and
move on. When a judgement call is needed, take the conservative option, record it under Deviations, and continue.

**Long GPU jobs:**
- launch them detached, with logs under `outputs/osler_v3/<task>/logs/`;
- record the PID in STATE;
- poll at intervals of 20 minutes or more;
- run one GPU job at a time;
- check outputs for completeness, not just exit codes.

## The contract

Every number carries its artifact and a recompute command (CLAIMS.md). `NOT MEASURED — <reason>` and
`BLOCKED — <reason>` are valid results; a fabricated number is a critical failure.

Gate O3's baselines for Osler-4B are exactly MedDecider-4B and zero-shot pplx-decider-flash-v1.2 (D39: pplx was chosen
in osler_v2). The 9B is deferred. Every other model and every Stage A or Stage B run is an additional row.

Held-out items, MR test sets, MDACE and the external panel never influence any choice. If a result is unwelcome, report
it and name the ADVISORY §2 outcome it implies.

## Do not

- Train on `MaziyarPanahi/MedExQA-clean` or any copy of an external-panel dataset, or use JEV's predictions as labels (D42).
- Keep a Gemma-crafted item whose independent verification disagrees, or relabel one (D43).
- Change the KL anchor prefixes (ADVISORY §3.2), the Stage A choice rule, or the frozen MR suite v1 after any run.
- Use any unofficial copy of MIMIC or other credentialed data, e.g. `lucky9-cyou/MIMIC-IV` or
  `ricardosantoss/ehrcomplete_icdfiltered` (D37).
- Train on any test split, the external panel, MDACE, the NHS synthetic notes, the clinical-case ICD-10 set, the knowledge dev v2, or held-out items (indicator `aki_creatinine_rise`, ICD family
  `heart_failure`, and osler_v0's held-out templates and generators).
- Use a teacher's answer, option distribution or correctness judgement as a label, or keep teacher prose that failed
  verification.
- Send MIMIC text, values or identifiers to any hosted service, or set `TEACHER_LOCAL=1` for a hosted endpoint.
- Commit, or paste into STATE, CLAIMS or any committed file: MIMIC text, values or identifiers; benchmark items;
  generated or teacher items; training data; predictions; checkpoints; keys.
- Use CPT descriptors, InterQual or MCG content, or anything from a payment-integrity vendor.
- Change D24, D31, D35 (or the frozen MR suite v1), the keep rule, the held-out lists or the curated ICD table to make something pass (that is
  `BLOCKED`).
- Change the head, the attention, the base models or the adapter (D26, D33).
- Run two GPU jobs at once or spawn sub-agents for GPU work.
- Edit earlier `loops/*/`, `data/bench/v0.1/`, `data/bench/v0.2/`, `data/bench/v0.3_ext/`, earlier `outputs/*/`,
  `AGENTS.md` (except V0's "Started" field) or `docs/plans/`.
- Delete, stop or reconfigure pods, volumes, HF repos or remote services; upload anything; push to the HF Hub.
- Use a hosted teacher (none is planned; Gemma is local).
- Start the next loop or anything in ADVISORY §7.

## Done when

- Every task is `DONE` or `BLOCKED — <reason>`; none is left `IN_PROGRESS`.
- `sources.md`, `stage_a.md`, `blocks.md`, `gate_o3.md` and `results.md` exist, as far as their tasks ran.
- `FINDINGS.md` exists: Summary first, and the five-claim spot-check appended with its re-run outputs.
- Every reported number has a CLAIMS row.
- `NEXT.md` lists proposals.
- Everything committable is committed and pushed.

Then set `Loop status: STOPPED` at the top of STATE.md, commit, push, **stop**, and report:
- which tasks completed;
- which blocked and why;
- the Gate O3 verdicts;
- what you would do next.

Also set `Loop status: STOPPED` (with FINDINGS for what ran) when
every remaining `PENDING` task is blocked on the operator.
