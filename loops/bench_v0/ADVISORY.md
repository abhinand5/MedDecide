# ADVISORY — bench_v0: build MedDecide-Bench v0, a validated harness, zero-shot baselines, and the teacher gate

**Read `AGENTS.md` first** (repo root). It carries the evidence rules (R1–R10),
guardrails, and loop mechanics. This document is the work plan. Read it in full at the
start of every loop iteration.

**Status:** authored 2026-10-06 from the planning session between the operator and the
advisor. Supersedes nothing. Program context: `docs/plans/PROGRAM.md`.

---

## 1. Where things stand

The repo is new: no code, no data, no results. This is loop 0 of the MedDecide program.

What exists outside the repo and what this loop builds on:

- **Base and baseline checkpoints exist on Hugging Face** (verified by the advisor on
  2026-10-06 with `hf models info` / `hf models ls`): `LiquidAI/LFM2.5-350M`,
  `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-0.8B-Base`, `Qwen/Qwen3.5-4B`,
  `Qwen/Qwen3.5-9B`, `google/medgemma-1.5-4b-it` (gated), `convaiinnovations/laya`,
  `convaiinnovations/laya-typed-decisions`, `SupersonicLabs/Julia-1`,
  `fastino/GLiNER2.5-Decide`, `rAVEUK/open-jev-deberta-v3-large`, `autotrust/JEV-9B`,
  `autotrust/JEV-27B`, `junma/MedJev-Qwen3.5-0.8B`, `perplexity-ai/pplx-decider-v1-27b`,
  `autotrust/GEV-26B-Decide`. **Existence only** — every performance number published
  for them is **unverified** until measured here.
- **Teacher:** `deepseek-ai/DeepSeek-V4.1-Flash` (763B params, ~510 GB weights,
  verified on HF). The operator self-hosts it on a separate 4×RTX PRO 6000 machine and
  exposes an OpenAI-compatible endpoint via `TEACHER_BASE_URL` / `TEACHER_API_KEY`. It
  may not be up when this loop starts — only T10 needs it.
- **Lesson carried from the predecessor project (MedColBERT v1):** its models scored
  0.754 nDCG@10 on an eval built by the same synthetic pipeline that made the training
  data, then lost to BM25 on every public benchmark (e.g. NFCorpus 0.151 vs BM25
  0.321). An eval the pipeline itself produced proves nothing. Hence: validate the
  harness against an independent reference first (T7), and make the headline tier
  fresh, with gold from structured fields (T5).
- **Lesson carried from MedColBERT v1 eval runs:** a run on a 244k-document corpus died
  silently at ~23 GB RAM with exit code 0 and no results. Check output completeness, not
  exit codes.

## 2. The question this loop answers

Is there a trustworthy measurement foundation for MedDecide, and does the teacher
pipeline work? Specifically:

- If the harness reproduces an independent reference **and** the benchmark is valid
  **and** the teacher gate passes → loop 1 trains MedDecide-0.8B on gold, with bulk
  teacher labelling in non-thinking mode planned for loop 2.
- If the teacher's ECE stays > 0.05 after temperature fitting → loop 2 labels in
  thinking mode at reduced volume, or trains on gold + UMLS synthetic only.
- If baselines already score > 95% on a fresh-tier template → that template is too
  easy; it is redesigned before training.
- If the harness cannot reproduce a reference → nothing downstream starts until it does.

## 3. Ground rules specific to this loop

- **Write paths:** `src/`, `scripts/` (add files; do not change `run_loop.sh` or
  `pod_env.sh`), `tests/`, `configs/`, `tools/`, `docs/benchmark/`, `loops/bench_v0/`,
  `outputs/bench_v0/`, `data/`, and the root files `pyproject.toml`, `uv.lock`,
  `.gitignore` (append only). Read-only: `AGENTS.md` (except T0's "Started" field),
  `docs/plans/`, everything in `loops/bench_v0/` except STATE.md, CLAIMS.md,
  FINDINGS.md and files you create.
- **No training of any kind.** Inference only.
- **No credentialed data** (no MIMIC, no UMLS) in this loop.
- **Benchmark data stays out of git.** Items live in `data/bench/` (gitignored).
  Commit only manifests (counts, hashes, IDs) and aggregate result tables.
- **Splits:** test splits are evaluation-only. Teacher temperature fitting (T10) uses
  dev/train items only.
- **Teacher endpoint:** read `TEACHER_BASE_URL` and `TEACHER_API_KEY` from the
  environment. If unset or unreachable when T10 is reached, T10 is
  `BLOCKED — teacher endpoint not provided`. Do not wait for it.
- **One GPU job at a time.** The pod has one RTX PRO 6000 Blackwell (96 GB), 128 CPU
  cores, ~2 TB RAM, a 20 GB root disk, and a large network volume at `/workspace`.
- **Timebox:** ~2 hours of effort per baseline-model integration or data-source
  integration; past that, `NOT MEASURED — <reason>` and move on.
- **Hard stop at T12.** Do not start loop 1.

## 4. Tasks

Each task: mark IN_PROGRESS, fill the STATE checklist, do the work, run the acceptance
check, self-audit (R6) into `outputs/bench_v0/<task>/SELF_AUDIT.md`, append CLAIMS
rows, mark DONE/BLOCKED, log, commit.

### T0 — Environment verification (no GPU work, ~30 min)

**Why:** a missing token or unreachable endpoint discovered mid-run wastes the night.

**Do:**
- Fill "Run started" in STATE.md and "Started" in AGENTS.md's Current loop.
- Record into `outputs/bench_v0/T0/env.json`: `nvidia-smi` GPU name + memory + driver,
  CUDA version, Python version, `uv --version`, free space on `/workspace` and `/`,
  RAM, CPU count, git commit.
- Check and record PASS/FAIL for: `HF_HOME`, `UV_CACHE_DIR`, `TMPDIR` set to paths on
  `/workspace`; `HF_TOKEN` set; HF access to `google/medgemma-1.5-4b-it` (download its
  `config.json` only); outbound HTTP 200 from `https://clinicaltrials.gov/api/v2/version`,
  `https://api.fda.gov/drug/label.json?limit=1`,
  `https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/`; `git push --dry-run` to origin;
  teacher endpoint (`GET $TEACHER_BASE_URL/models`) — FAIL is allowed here, it only
  blocks T10.

**Acceptance:** `env.json` exists with every check marked PASS or FAIL + reason. Any
FAIL other than the teacher endpoint goes into STATE.md "Questions for the operator";
the dependent tasks are marked BLOCKED, unrelated tasks proceed.

### T1 — Repo scaffold (no GPU, ~1 h)

**Why:** everything after this needs a package, an environment, and tests.

**Do:**
- `pyproject.toml` (Python 3.12, `uv`), package `src/meddecide/` with subpackages
  `bench/` (schema, loaders, builders), `eval/` (readout, metrics, probes),
  `teacher/` (teacher client), `utils/` (io, hashing, provenance).
- Dependencies: torch with CUDA 13 wheels that run on Blackwell (sm_120), transformers,
  datasets, pyarrow, pandas, numpy, scipy, pydantic, httpx, tqdm, rich; dev: pytest,
  ruff. Add others (vllm, model-specific packages) when a task needs them, and record
  each addition in the iteration log.
- `ruff` config, `tests/` with a smoke test, `.gitignore` additions: `outputs/`,
  `data/`, `*.parquet`, `*.jsonl` outside `docs/`, `.env`.
- `configs/bench_v0.yaml`: paths, seeds, caps (defaults from this document).
- A GPU smoke test: load `Qwen/Qwen3.5-0.8B` in bf16 and run one forward pass.

**Acceptance:** `uv run pytest` passes; `uv run ruff check .` passes; GPU smoke test
output recorded in `outputs/bench_v0/T1/gpu_smoke.txt`.

### T2 — Item schema (no GPU, ~1 h)

**Why:** every source, harness, and model in this program speaks one item format.

**Do:** define the item model (pydantic) and document it in
`docs/benchmark/schema.md`. Fields at minimum:
`item_id` (stable hash of source + record id + template id + option order seed),
`tier` (`established|fresh`), `source`, `source_record_id`, `source_url`,
`source_license`, `record_date` (ISO), `split` (`train|dev|test`), `template_id`,
`skill`, `qtype` (`noul|choice|score`), `state` (text), `question` (text),
`options` (list of `{key, label, description?}`; for `score`, ordered lowest first),
`gold` (option key, or for `noul` `yes|no`), `meta` (dict). Plus JSONL read/write
helpers with validation and a `manifest` builder (counts per source × split × qtype ×
template, sha256 of each file).

**Acceptance:** unit tests cover round-trip, deterministic IDs (same input → same
`item_id`), and validation failures (gold not in options; `score` with < 2 levels).

### T3 — Tier 1 loaders (CPU, ~3 h)

**Why:** comparability with published work, and gold dev data for teacher calibration.

**Do:** loaders that turn public datasets into items, preserving official splits and
recording the dataset id, revision, and license:
- MedQA (USMLE, 4-option) — `choice`.
- MedMCQA — `choice`. Its official test labels are not public: use the official
  **validation** split as tier-1 test, and carve a dev split from train (record this).
- PubMedQA (expert-labelled) — `choice` over yes/no/maybe.
- MMLU medical subsets (anatomy, clinical_knowledge, college_medicine,
  college_biology, medical_genetics, professional_medicine) — `choice`.
- SciFact claim verification (SUPPORT/CONTRADICT/NOT ENOUGH INFO) — `choice`.
- PubHealth / HealthVer if available on HF with a clear license — `choice`.
- Relevance: TREC-COVID and NFCorpus via BEIR-format qrels — `noul` (relevant?) and
  `score` where graded labels exist; sample balanced positives/negatives per query
  with a fixed seed; record the sampling.
- MedQuAD question-type routing — `choice`.

Caps: keep full official test sets up to 5,000 items per source (random sample with a
fixed seed above that; record the cap). Dev sets: up to 2,000 per source.

**Acceptance:** `data/bench/tier1/<source>.jsonl` per source; manifest rows for each;
a check that no `(source, source_record_id)` appears in more than one split; license
recorded per source (or `UNKNOWN` + a question to the operator).

### T4 — Tier 1 contamination probe (GPU, ~3–5 h) — deps T3, T6

**Why:** a reviewer will discount tier-1 numbers for models that memorised the test
sets. We measure it instead of arguing about it.

**Do:** for each ladder model and each tier-1 test source, on a fixed sample of ≤500
items: (a) a Min-K%-Prob style memorisation score on the question text (K=20%), and
(b) the same score on a matched control (the same items with the question paraphrased
by deterministic templating — e.g. option order permuted and the stem's sentences
reordered where possible). Report the per-model × source table and the score gap.
State clearly that these are probes, not proofs.

**Acceptance:** `outputs/bench_v0/T4/contamination.json` + a committed table in
`loops/bench_v0/contamination.md`.

### T5 — Fresh-tier builders (CPU + network, ~6–8 h)

**Why:** this is the headline tier. It must be uncontaminated and its gold must not
come from any model.

**Do:**
1. **Date window.** For each ladder model and the teacher, find its documented
   training-data cutoff (model card, tech report). Where none is documented, use the
   model's HF repo creation date as a conservative upper bound. Window start = the
   latest of these dates. Record every date and its source in
   `docs/benchmark/fresh_window.md`. Window end = the build date.
2. **Builders** (scripts in `scripts/bench/`, logic in `src/meddecide/bench/fresh/`),
   each taking `--start` and `--end` dates and rebuilding identically for the same
   arguments:
   - **ClinicalTrials.gov API v2:** filter by **first-posted date** (not last-update)
     inside the window. Candidate templates (state = brief summary + detailed
     description, **with the title and any field the question asks about removed**):
     phase (`choice`); allocation randomised? (`noul`); primary purpose (`choice`);
     intervention type (`choice`); accepts healthy volunteers? (`noul`, state =
     condition + summary, not the eligibility text).
   - **openFDA `drug/label`:** filter by the label's first effective date inside the
     window (exclude labels whose `set_id` existed before the window — they are
     updates). Candidate templates (state = selected sections, with the answer's
     section removed): established pharmacologic class (`choice`, distractors =
     other classes from the same build, seeded); has a boxed warning? (`noul`, state
     excludes the boxed-warning section); route of administration (`choice`).
   - **PubMed update files:** records with publication/entry date inside the window.
     Candidate templates (state = title + abstract): publication type
     (RCT / systematic review / meta-analysis / case report / observational /
     other — `choice`); humans vs animals check tag (`noul`); major MeSH topic among
     seeded distractors from the same MeSH tree level (`choice`).
3. Distractors and option order are deterministic (seeded) and recorded.
4. Target ≥500 items per template where the window allows; record the actual count.
   Split fresh items into `dev` (20%) and `test` (80%) by record id hash.

**Acceptance:** `data/bench/fresh/<source>.jsonl`; manifest rows; a check script proving
**zero** records dated before the window start (by the filter field) and
rebuild-determinism (two builds with the same arguments produce identical hashes).

### T6 — Eval harness (GPU, ~4–6 h)

**Why:** one harness scores every model the same way; bugs here invalidate everything.

**Do:** in `src/meddecide/eval/`:
- **Zero-shot verbalizer readout** with HF transformers (bf16, batched): render the
  item through the model's chat template (system: "You are a medical decision model.
  Answer with the letter of the correct option only."; user: state, question, options
  as `A. ...`; assistant prefix up to the answer position), with any thinking mode
  **disabled** (e.g. `enable_thinking=False`) and a closed think block where the
  template requires one. Read next-token logits for the option letter tokens.
- **Label checks:** assert each option letter is a single token for the tokenizer
  (try the `" A"` and `"A"` variants; record which one is used); fail loudly otherwise.
- **Outputs per item:** probabilities over options (softmax over option logits),
  **label mass** (full-vocabulary probability of the option tokens), argmax, latency.
  For `score` items also the expected level.
- **Metrics:** accuracy (micro, macro over templates), majority-class baseline, Brier,
  ECE (15 equal-width bins), label-mass summary, p50/p95 latency, bootstrap 95% CIs
  (1,000 resamples).
- **Probes:** option-shuffle (a second seeded permutation; flip rate = share of items
  whose argmax *content* changes); candidate-count scaling on items where extra
  seeded distractors are possible (2/4/8/16 options); none-of-the-above abstention
  (a variant with the gold option removed and "None of the above" added; report
  detection rate on those, false-rejection rate on the unmodified items).
- A results writer: one JSONL of per-item predictions per (model, source) in
  `outputs/bench_v0/<task>/preds/`, and a summary JSON.

**Acceptance:** unit tests on metric functions with hand-computed cases (Brier, ECE,
flip rate); label-token check passes for every ladder model; a 50-item end-to-end run
on Qwen3.5-0.8B completes and writes predictions + summary.

### T7 — Harness validation against an independent reference (GPU, ~2–3 h)

**Why:** the cheapest test that could invalidate every later number.

**Do:** pick a tier-1 task that exists in **lm-evaluation-harness** with a
log-likelihood multiple-choice protocol (e.g. `medqa_4options`, `medmcqa`,
`mmlu_clinical_knowledge`). Run lm-evaluation-harness and our harness on the **same
model and items** (e.g. Qwen3.5-0.8B and one more ladder model), zero-shot. The
protocols differ slightly (lm-eval scores option continuations; we score option
letters), so first configure lm-eval to the closest letter-scoring variant if
available; record exactly which protocol each used. Alternatively reproduce a number
published in a model card or paper whose protocol is documented as letter
log-likelihood.

**Acceptance:** agreement within **±2 accuracy points** on at least one (model, task)
pair, with both numbers, both commands, and the protocol notes in CLAIMS. If no pair
agrees within ±2 after fixing genuine bugs, T7 is `BLOCKED — harness disagrees with
reference by X pts` and T9 must not be reported as results (R8: do not loosen the
tolerance).

### T8 — Fresh-template screen (CPU, ~1–2 h)

**Why:** a template answerable from surface cues measures pattern-matching, not
medicine.

**Do:** for each fresh template: majority-class accuracy; a hand-written regex/keyword
baseline (≤30 minutes per template, written **before** looking at model results); a
bag-of-words logistic regression trained on the template's **dev** split, scored on
test. Drop templates where the regex **or** the BoW baseline reaches ≥95% accuracy on
test; record each drop with its numbers.

**Acceptance:** `outputs/bench_v0/T8/screen.json`; committed `loops/bench_v0/template_screen.md`
listing kept and dropped templates with counts; the manifest marks dropped templates.

### T9 — Zero-shot baseline table (GPU, ~10–20 h)

**Why:** the bar MedDecide must clear, measured on our items.

**Do:** run the harness (T6 protocol) on tier-1 test + fresh test (kept templates) for:
- **Ladder (verbalizer readout):** LFM2.5-350M, Qwen3.5-0.8B, Qwen3.5-0.8B-Base,
  MedGemma-1.5-4b-it, Qwen3.5-4B, Qwen3.5-9B.
- **Open decision models that fit one GPU**, each through its own published inference
  path (package or serving recipe from its model card), mapped onto our item format:
  Laya (`laya`, `laya-typed-decisions`), Julia-1, GLiNER2.5-Decide,
  open-jev-deberta-v3-large, JEV-9B (vLLM + its LoRA decision path). MedJev is
  trained for a fixed 11-variable schema: run it only if its inference path accepts
  arbitrary questions; otherwise `NOT APPLICABLE — fixed schema`.
- Respect each model's option limits; where a model cannot take an item (too many
  options, context too long), record it as unsupported for that item and report
  coverage beside accuracy.

**Acceptance:** `outputs/bench_v0/T9/results.json`; committed
`loops/bench_v0/baselines.md` with, per model × source: n, coverage, accuracy (± CI),
majority baseline, Brier, ECE, label mass (where defined), p50 latency, shuffle flip
rate, abstention detection / false rejection. Every empty cell reads
`NOT MEASURED — <reason>`.

### T10 — Teacher pipeline gate (teacher endpoint; ~3–6 h)

**Why:** loop 2's data moat depends on cheap, calibratable teacher probabilities.

**Do:** with the teacher at `TEACHER_BASE_URL`:
- Confirm the endpoint returns **log-probabilities at the answer position** for the
  option tokens (completions API with `logprobs`/`top_logprobs` large enough to cover
  every option, or a prompt-scoring endpoint). Record exactly which API shape works.
  If only top-k is returned and an option is missing, treat its probability as the
  residual mass divided across missing options and **report how often this happens**.
- **Non-thinking mode:** run ≥2,000 **dev** items stratified across tier-1 sources and
  fresh templates. Compute accuracy, Brier, ECE raw; fit one temperature per `qtype`
  on half of these items, evaluate ECE on the other half.
- **Thinking mode:** on ≥200 of the same items, let the teacher reason, then read the
  answer distribution. Record accuracy, ECE, and cost (output tokens/item, seconds/item).
- Throughput: items/hour for each mode at the concurrency the endpoint sustains.

**Gate:** PASS if non-thinking ECE ≤ 0.05 after temperature fitting (held-out half).
Report accuracy vs the best T9 baseline on the same items regardless of pass/fail.

**Acceptance:** `outputs/bench_v0/T10/teacher_gate.json` and committed
`loops/bench_v0/teacher_gate.md` with the verdict and numbers; or
`BLOCKED — teacher endpoint not provided` if unset/unreachable when reached.

### T11 — Operator audit page + sample (no GPU, ~2 h)

**Why:** the operator validates that each fresh template means what it claims. It
must be fast for a human.

**Do:**
- Draw a stratified sample of **150** fresh **test** items: 50 per source, spread
  across kept templates, seeded. Write `outputs/bench_v0/T11/audit_sample.jsonl`
  (each row: item_id, template_id, source_url, state, question, options, gold, and the
  raw structured field(s) the gold was derived from).
- Write `tools/audit/audit.html`: one self-contained file, no network, no external
  libraries. The operator opens it locally, loads the JSONL with a file picker, and
  sees one item per screen: left = source record excerpt + link, right = question,
  options, gold highlighted. Keys: `A` accept, `R` reject, `N` add note, `←/→`
  navigate. Progress bar. Autosaves to `localStorage`. "Export" downloads
  `audit_v0.jsonl` (item_id, verdict, note, timestamp).
- Commit `tools/audit/audit.html` (it contains no data). Do **not** commit the sample.

**Acceptance:** the page loads a sample file and exports a valid JSONL (test with a
headless browser or a small JS/Python check of the export logic). The audit itself is
`BLOCKED — awaiting operator` in STATE.md with instructions in "Questions for the
operator" (where the sample file is; how to copy it off the pod; where to put
`audit_v0.jsonl` when done: `outputs/bench_v0/T11/audit_v0.jsonl`).

### T12 — Findings and closure (no GPU, ~2 h) — HARD STOP

**Why:** the loop's output is only as good as its record. This task writes the artifact
the next loop and the operator actually read.

**Do:** write `loops/bench_v0/FINDINGS.md` following the FINDINGS structure (Summary
first: the question, what we did, what we found, what it means, what did not work —
plain language, every number with its denominator and CLAIMS id, honest about
failures and blocked tasks). Every summary claim must exist, with more detail, in the
body. Then pick five claims spanning the loop's major results, re-run their recompute
commands in fresh processes, and append the spot-check table. Fill STATE.md's closure
summary feed. Also write `loops/bench_v0/NEXT.md`: what you would put in loop 1, as
proposals for the operator — not decisions. Commit and push. **Then stop.** Do not
start loop 1.

**Acceptance:** FINDINGS.md exists with (1) a Summary a non-specialist can follow,
(2) a body covering every summary claim, (3) the five-claim spot-check with actual
re-run outputs; every STATE task is DONE or BLOCKED with a reason; pushed to origin.

## 5. Definition of done

- STATE.md — every task `DONE` or `BLOCKED` with a reason.
- CLAIMS.md — every reported number, with artifact and recompute command.
- FINDINGS.md — written per T12: mandatory Summary first, full-coverage body, five-claim
  spot-check appended.
- Per-task directories under `outputs/bench_v0/` with reports, SELF_AUDIT.md, artifacts.
- Committed: code, tests, `docs/benchmark/`, `tools/audit/audit.html`, the loop's
  markdown reports. Not committed: any item text, predictions, or teacher outputs.

## 6. Task table (seed STATE.md with this)

| id | task | GPU | deps | status |
|---|---|---|---|---|
| T0 | Environment verification | no | — | PENDING |
| T1 | Repo scaffold | smoke | T0 | PENDING |
| T2 | Item schema | no | T1 | PENDING |
| T3 | Tier 1 loaders | no | T2 | PENDING |
| T5 | Fresh-tier builders | no | T2 | PENDING |
| T6 | Eval harness | yes | T2 | PENDING |
| T7 | Harness validation vs reference | yes | T3, T6 | PENDING |
| T8 | Fresh-template screen | no | T5 | PENDING |
| T4 | Tier 1 contamination probe | yes | T3, T6 | PENDING |
| T9 | Zero-shot baseline table | yes | T7, T8 | PENDING |
| T10 | Teacher pipeline gate | teacher | T3, T5, T6 | PENDING |
| T11 | Operator audit page + sample | no | T8 | PENDING |
| T12 | Findings and closure — HARD STOP | no | all | PENDING |

The order of the table is the execution order. T7 gates T9: no baseline numbers are
reported from an unvalidated harness. One exception to "first PENDING task wins": T10 does not
depend on T9, so if the teacher endpoint answers `GET $TEACHER_BASE_URL/models` at the
moment T9 would start, run T10 first (it is shorter and the teacher machine is costly
to keep up), then T9. Otherwise run T9 and re-check the endpoint when T9 finishes.
A blocked T10 or T11 never blocks T12.

## 7. Explicitly out of scope

Do not spend time on these. Each is deliberately excluded.

| item | why |
|---|---|
| Any training, LoRA, pointer-head code | loop 1, after gate G0 |
| MIMIC / PhysioNet data, MedDecide-clinical | DUA + training pending; later loop |
| UMLS-derived items or synthetic data | loop 2 |
| Fresh sources beyond ClinicalTrials.gov, openFDA, PubMed (PMC case reports, OpenAlex citations) | deferred to benchmark v2 |
| 27B-class baselines (JEV-27B, pplx-decider-v1-27b, GEV-26B-Decide) | need the 4-GPU window; loop 3 |
| Joint schema head, RLCD, adaptive thinking | later ablations, gated on evidence |
| Retrieval track (MedColBERT / MedEmbed) | separate repo and track |
| Prompt engineering to raise baseline scores | baselines use the fixed T6 protocol; tuning prompts per model biases the comparison |
| Naming, paper writing, public release | operator decisions after results |

## 8. Failure modes to avoid

1. **Fresh-tier leakage through updates.** Old ClinicalTrials.gov trials and FDA labels
   get updated inside the window and look fresh. Filter on first-posted / first
   effective dates and exclude set_ids/NCT ids that existed earlier. Detection: any
   record whose first-posted date precedes the window.
2. **Answer leaks in the state.** The trial title says "Phase 2", the label's boxed
   warning section is still in the state, the abstract says "randomised". Remove the
   answer-bearing field from the state; T8's regex and BoW screens are the detector.
3. **Readout bugs.** Multi-token option labels, an unclosed think block, the wrong
   assistant prefix, top-k truncation. Detection: label mass ≪ 1 for a model, accuracy
   below chance, T7 disagreement.
4. **Overruled objection — teacher fixed without comparison.** DeepSeek-V4.1-Flash was
   chosen without a bake-off. If T10's held-out ECE stays > 0.05 after temperature
   fitting, report it plainly; that is this objection materialising, and it is the
   operator's decision what follows.
5. **Trusting an eval the pipeline produced** (MedColBERT v1's failure). Fresh-tier
   gold comes only from structured fields; no LLM ever labels a benchmark item.
6. **Silent eval death** (MedColBERT v1's failure). A run can exit 0 with no results.
   Check predictions count == items count for every (model, source).
7. **Class imbalance hiding weakness.** Always show the majority baseline and macro
   accuracy beside micro accuracy.
8. **Relabelling an unwelcome result (R4).** If baselines are weak, strong, or
   inconsistent, report them as measured. Do not tune prompts per model.
9. **Citing artifacts that do not exist (R2).** Check the path before writing it.
10. **Impossible arithmetic (R5).** Coverage × n must match evaluated counts; split
    counts must sum to totals.
11. **Scope creep into training.** Any training code is out of scope for this loop.

## 9. If you finish early

In priority order, without starting anything in §7:
1. Raise fresh-tier counts toward ≥1,000 items per kept template if the window allows.
2. Add a second ladder model to T7's validation.
3. Profile harness throughput (items/s per model) and record it for loop 1 planning.
