# NEXT — proposals for the next loop (not decisions)

Written at the close of `bench_v0_fix0`. Everything below is a **proposal**; the operator decides.
Each one names the evidence that motivates it in this loop's artifacts.

## P1 — Fix the graded-score template's option set (highest priority, small)

`nfcorpus_graded_score_v1` offers a lowest level (`Not relevant`) that **no item in its pool has**,
so answering it is always wrong. JEV-9B scores **0.2634** three-way (chance 0.3333) but **0.5804**
restricted to the two levels present (chance 0.5), and the 4B/9B ladder cells are withheld by the
health gate as below chance — the only two ladder cells the gate suppresses on CI grounds.
Evidence: `outputs/bench_v0_fix0/F7/SCORE_TEMPLATE_FLAW.md`, X029, X035.
**Proposal:** build score options from the levels present in the sampled pool, or make grade-0 items
a non-empty part of the same pool; either way re-run the affected cells. This is a v0.2 builder
change and invalidates the current v0.1 numbers for that template only.

## P2 — Provide the teacher endpoint (blocks a whole task)

`TEACHER_BASE_URL` and `TEACHER_API_KEY` have been unset for two loops, so the teacher gate
(F8/T10) has never run. The procedure is fully specified in `loops/bench_v0/ADVISORY.md` T10 and
needs no design work: agreement on v0.1 **dev** items, the F2 readout for `noul`/`score`, a drift
check, and refusal rates. Evidence: X024, `outputs/bench_v0_fix0/F8/BLOCKED.md`.

## P3 — Run the human label audit

150 v0.1 fresh test items are staged (50 per source, 8 kept templates, sha256 recorded) and the
audit page is reused unchanged. Whether the gold labels are correct is the one thing this loop
could not measure about its own benchmark. Evidence: X022, X023.

## P4 — Install the fused kernels before the next large run (cheap, large payoff)

`causal_conv1d` and `flash-linear-attention` are absent, so Qwen3.5's hybrid layers fall back to
reference PyTorch. That cost this loop (a) two full re-runs of the 9B baseline (~2 h), (b) a 30 GB
allocation on long prompts that forced an 8,192-token cap on JEV-9B's two openFDA cells, leaving
**661 of 1,910** and **846 of 4,000** items unmeasured, and (c) roughly 3× slower inference
throughout. Evidence: X030 coverage columns; the F6/F7 memory defects in STATE's iteration log.

## P5 — Tier-1 templates the next loop should reconsider

`bench_v0`'s FINDINGS item 4 flagged two "good for the wrong reason" templates. v0.1 measures this
rather than argues it, and the picture is now concrete:

| template | what the measurement shows | proposal |
|---|---|---|
| `medquad_routing_v1` | every model 0.91–0.99; gold text in state 0.588 (F3) | keep but never quote as skill; report beside the copy baseline |
| `pubmed_mesh_major_choice_v1` | gold text in state only 0.462 (< 0.5 majority), BoW macro 0.246, JEV 0.996 | **keep** — the check clears it; the state is title+abstract only |
| `ct_phase_choice_v1` | best model 0.47–0.50 against 0.20 majority; BoW 0.503 | keep; it discriminates |
| `fda_route_choice_v1`, `ct_intervention_type_choice_v1`, `ct_primary_purpose_choice_v1`, `pubmed_pubtype_choice_v1` | dropped by F3 for `n_test<200` | either widen the window per template or leave them out of the headline table |

Evidence: `template_screen_v0_1.md`, X017, X020, X032.

## P6 — Decide what the strict slice is for

v0.1 reports 2,896 strict-slice items (≥ 2026-09-10). They are currently a *reported subset*, and
no decision has been made about whether the headline tier-2 number should be the full window or the
slice. The full window is the conservative choice for our own baselines (later models may have seen
in-window records, which only helps them); the slice is the defensible choice against a
contamination challenge. **Proposal:** quote the full window as the primary number and the slice
beside it, and say which is which.

## P7 — Benchmark hygiene: dedupe prediction files

Prediction JSONL files are append-only across re-runs, so row counts can exceed item counts (the
JEV file has 5,092 MedQA rows for 1,273 items). The reports dedupe, but a future consumer might not.
**Proposal:** write predictions per run with a run tag, or dedupe by `item_id` on read and record
the raw row count beside the unique count.

## P8 — What this loop did *not* re-measure, and probably should

* `C024` (share of fresh states containing a study-design term) — superseded but not re-derived for
  v0.1; the templates changed.
* Throughput per model (`C053`) — F6 records per-cell wall times, but no consolidated table.
* Adaptive-thinking and verbalizer-readout ablations: out of scope here, still unmeasured.
