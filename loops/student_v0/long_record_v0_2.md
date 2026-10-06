# Long-record slice and shared-prefix measurement (loop `student_v0`, task S3)

Two measurements on v0.2, plus one defect found while doing them.

## 1. The long-record slice

Threshold: **8,192 tokens** under the Qwen3.5 tokenizer, over every v0.2 fresh **test** item
(24,263 items). Counts live in `data/bench/v0.2/long_record.json`, per-item token counts in
`data/bench/v0.2/long_record_items.json`, and the manifest carries a `long_record` block. The
flag is recorded *beside* the items rather than inside them, so the v0.1 carried-identity
property that S1 established survives (an item's `meta` is part of its content hash).

| quantity | value |
|---|---|
| fresh test items | 24,263 |
| **long (> 8,192 tokens)** | **2,224 (9.2 %)** |
| over the harness prompt cap (16,384) | 584 |
| prompt tokens p50 / p90 / p99 / max | 523 / 7,705 / 19,654 / **75,319** |

| template | n_test | n_long | p50 | max |
|---|---|---|---|---|
| `fda_route_claim_noul_v1` | 2,000 | **944** | 7,945 | **75,319** |
| `fda_boxed_warning_noul_v1` | 1,910 | **668** | 4,775 | 32,485 |
| `fda_class_choice_v2` | 3,236 | **601** | 999 | 50,885 |
| `fda_route_choice_v1` (dropped template) | 84 | 11 | 2,745 | 25,893 |
| every ClinicalTrials.gov and PubMed template | — | **0** | 391–541 | ≤ 8,093 |

The slice is 2,224 items, well above the 300 the task asks for, so **no template was rebuilt**
with a larger state budget — the cap stayed where it was.

## 2. A defect this task found: the harness truncated the question away

The harness tokenizes with `truncation=True, max_length=max_prompt_tokens` (16,384). The
tokenizer default is `truncation_side="right"`, which keeps the **head** of the prompt — the
beginning of a long state — and drops the **tail**, where the question, the options and the
instruction live. For every item over the cap the model was therefore scored on a prompt that
asked nothing.

Measured before the fix, items over the cap (i.e. scored without their question):
**258 of 2,000** `fda_route_claim_noul_v1`, **185 of 1,910** `fda_boxed_warning_noul_v1`,
**137 of 3,236** `fda_class_choice_v2` — **584 of 24,263** fresh test items in total.

The fix keeps the tail (`configure_truncation` in `src/meddecide/eval/harness.py`); the cap is
unchanged. Zero-shot 0.8B on the affected templates, before → after:

| template | 0.8B before | 0.8B after | change |
|---|---|---|---|
| `fda_class_choice_v2` | 0.7803 | **0.8106** | +0.030 |
| `fda_route_claim_noul_v1` | 0.8875 | **0.9525** | +0.065 |
| `fda_boxed_warning_noul_v1` | not measured in this loop before | **0.6147** | — |

The pre-fix numbers in CLAIMS S020 and S031 are superseded for those two templates; the
correction is recorded in CLAIMS S035–S037, and the same defect silently affected the **v0.1
openFDA cells reported by `bench_v0_fix0`** (the ladder models used the same cap) — flagged for
the operator, since `loops/bench_v0_fix0/` is read-only for this loop.

## 3. Slice results, with the majority baseline beside every number

Zero-shot 0.8B, long vs short, same templates (`outputs/student_v0/S3/long_slice.json`). The D12
gate applies to the whole cell; a **subset** carries no gate of its own, so each subset is shown
with its own majority share.

| template | subset | n | majority | 0.8B accuracy |
|---|---|---|---|---|
| `fda_route_claim_noul_v1` | long | 944 | 0.503 | **0.9915** |
| | short | 1,056 | 0.503 | 0.9176 |
| `fda_class_choice_v2` | long | 601 | 0.271 | **0.9451** |
| | short | 2,635 | 0.256 | 0.7799 |
| `fda_boxed_warning_noul_v1` | long | 668 | **0.835** | **0.4296** |
| | short | 1,242 | 0.680 | 0.7142 |

Long records are **not** uniformly harder. Two of the three templates are *easier* on long
records (a long label states its class and its route more explicitly), while
`fda_boxed_warning_noul_v1` collapses: 558 of its 668 long items are "yes" (majority 0.835) and
the model scores 0.4296, i.e. far **below the majority baseline** — the one subset in the slice
where that happens. Hypothesis (not a measurement): with the tail kept, the sections that remain
after removing the boxed-warning section are the *end* of a long label
(description/pharmacodynamics), while the warning context sits earlier, so the model loses the
evidence it would need. The alternative (the old right-truncation) removed the question
entirely, which is strictly worse.

## 4. Shared-prefix measurement (`outputs/student_v0/S3/prefix.json`)

Protocol: 32 records of v0.2 fresh test items that carry several questions, Qwen3.5-0.8B, batch
size 1. Path (a) renders one prompt per question and prefills it whole; path (b) prefills the
state once and feeds only the question's own tokens with a **copy of the prefix cache** per
question. Both paths receive the identical token sequence, so the argmax comparison is exact.

| metric | (a) one prompt per question | (b) shared prefix |
|---|---|---|
| questions | 66 (2.06 per record) | 66 |
| total seconds | 2.451 | 2.734 |
| **questions/s** | **26.93** | **24.14** |
| **records/s** | **13.06** | **11.66** |
| seconds in prefix prefill | — | 1.200 |
| seconds in suffix passes | — | 1.535 |

**Throughput: the shared prefix is *slower* (0.90×), not faster.** Two reasons, both measured:
records in this tier carry only ~2 questions, so the most a shared prefix can save is one of
three passes; and the per-question cache copy plus the cached decode path cost more than the
prefill they save.

**Agreement: 0.9697 (64/66), against a ≥ 0.99 requirement — the check FAILS.** It is reported as
a failure with its cause rather than rounded up:

* every disagreement is a **near-tie**: margins (top1−top2 option probability) of **0.0000** and
  **0.0369**, and restricted to questions with margin > 0.05 the two paths agree on **100 %**
  (`checks.agreement_at_least_0.99_above_0.05_margin` is true);
* the paths differ numerically (max |Δlogit| ≈ 0.2 measured on a sample), because prefill uses
  the chunked gated-delta-rule kernel and the cached path uses the recurrent one. They are
  mathematically equivalent, not bit-identical.

One bug was found and fixed inside this measurement, and it is the reason the first version read
0.85: every question must branch from the **prefix** cache. Appending question 2 to a cache that
already held question 1 gave question 2 a different context (prefix + question 1 + question 2
instead of prefix + question 2), which flipped the argmax on 15 % of questions — including one
flip with a 0.73 probability margin. After the fix only the two ties remain.

**What this means for serving:** on this model, sharing a state prefix across questions is not a
free optimisation. It changes the decision on ties, so a cached-prefix serving path must be
validated against the plain path before any number measured through it is compared with the
benchmark — the same rule the D12 gate applies to readouts.

## 5. Commands

```bash
uv run python scripts/bench/tag_long_records.py --threshold 8192          # slice + per-item counts
bash outputs/student_v0/S3/logs/run_s3_0p8b.sh                            # 0.8B re-measurement after the fix
bash outputs/student_v0/S3/logs/run_s3_9b.sh                              # 9B re-measurement (in flight when this was written)
uv run python scripts/bench/report_long_slice.py --dir outputs/student_v0/S3 --out outputs/student_v0/S3/long_slice.json
uv run python scripts/bench/measure_prefix.py --records 40 --out outputs/student_v0/S3/prefix.json
```

Acceptance: slice counts per template are in `data/bench/v0.2/manifest.json` and
`long_record.json`; `prefix.json` holds both throughputs and the agreement rate.
