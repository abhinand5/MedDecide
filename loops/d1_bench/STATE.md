# d1_bench — STATE (bench/d1-2b)

Branch `bench/d1-2b` (worktree `/workspace/MedDecide-d1-2b`, created from `dev` at 1a188ba). Not pushed, not merged.
Status: **DONE** for steps 1–5. Results: `loops/d1_bench/results.md` (aggregates only). Raw predictions are private in
`outputs/d1_bench/` (gitignored).

## 1. Model and license

- Revision: `LiquidAI/d1-2B` @ `10e7671d44e4b37700020b1eed3f594c5f2af0b4` (downloaded to HF_HOME, 3.6 GB; not committed).
- Card metadata: `license: other`, `license_name: lfm1.0`. The full LICENSE file is the **LFM Open License v1.0**.
- Clauses (quoted):
  - Commercial Use (definition): "any use of the Work for direct or indirect commercial advantage or monetary compensation."
  - Non-Commercial or Research Purposes: "purposes that do not involve any use of the Work or a Derivative Work for Commercial Use."
  - Section 2 (grant): "a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce,
    prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work ..., subject to ... the
    Commercial Use limitation set forth in Section 5."
  - Section 5(a): "The rights granted under this License for Commercial Use are conditioned upon You or Your Legal Entity not
    exceeding the Threshold." 5(b): "Any Commercial Use ... by a Legal Entity that exceeds the Threshold is not licensed under this Agreement."
  - Threshold: "annual revenue of 10 million United States dollars ($10,000,000) or more."
  - Section 11: "This License will terminate automatically and immediately if You fail to comply with any of its terms and conditions."
- Reading for this run: the license has no separate evaluation clause. Running frozen sets locally for a research benchmark
  is inside the Section 2 grant (reproduce / prepare derivatives) and is not Commercial Use as defined. No weights or derivative
  works are redistributed. Only aggregates are committed (Section 4 redistribution duties not triggered).
- **Operator question:** whether MedDecide's use counts as "indirect commercial advantage" (an open model program, a public
  benchmark) and whether the operator's entity exceeds the $10M threshold. I did not decide this; I proceeded as research evaluation.

## 2. Loader and validation

- `src/meddecide/model/d1.py` follows the card's "Inference" section: `AutoModel.from_pretrained(REPO, revision=REVISION,
  trust_remote_code=True, dtype=bfloat16)`, then `model.system_one(state, {q: question})`, one question per call.
  Question dicts: noul (P(yes)), choice (criteria keys = our option keys, so single letters stay letters), score (levels lowest first).
- Custom code was read before any load (`modeling_d1.py`, `api.py`, `runner.py`, `prompt.py`); grep found no network, exec or
  pickle calls in the model path.
- Load under the repo's transformers **5.18.0** failed first with `ImportError: This modeling file requires the following
  packages that were not found in your environment: torchvision`. After installing torchvision 0.29.1+cu130 and pillow 13.0.0
  into the worktree venv only, the load succeeded (424/424 weights). Not BLOCKED.
- Validation (`scripts/bench/d1_validate.py`, `outputs/d1_bench/validate/validation.json`): 7 hand-made items, all four checks
  true: every vector sums to one (|sum-1| <= 1e-4), argmax answers not constant (5 distinct keys), noul separates a positive
  from a negative case, all argmaxes match the intended answer. The card's quick-start example reproduces its printed output
  (refund P(yes) 0.9968 ≈ 1.0, team billing, topics billing+delivery, urgency medium). The checkpoint's calibration.json is applied.
- Unit tests: `tests/test_d1.py` (17 pure tests, pass).

## 3. Scoring

- Frozen sets: `data/bench/osler_v4/y9/*.jsonl` (main checkout, gitignored). Counts match the brief for all five sets
  (knowledge 5,456; MedNLI 1,422; panel 4,353; MR verdict 13,103; MIMIC suite 4,253). Every item was scored; none skipped.
- Metric and gate: `meddecide.eval.gate_o4.normalise_row`, `meddecide.eval.readout_gate.gate_by_template` (D12 constant-answer and
  below-chance checks; D21 variant). Definitions as `loops/osler_v4/results.md`.
- Run: `scripts/bench/d1_chain.sh` (one GPU job at a time, all five sets, exit 0 each). Wall time: MedNLI 15 s, knowledge 58 s,
  panel 47 s, MR verdict 264 s, MIMIC 170 s.
- Gate result: four sets PASS; **MIMIC suite READOUT_FAIL on `mimic_drg_cc_level_v1`** (constant answer C on 98.0 % of its 498 items).
  Its accuracy is not reported as a result (D12).

## 4. Diagnostics (additional, labelled, no verdict)

- MR verdict, option J: the model picks J ("None of these codes is supported") 1,080 times; gold J appears 15 times. The
  excess is in the `codiesp_icd_support_v1` (525 picks, 0 gold) and `mkurman_icd_support_v1` (549 picks, 0 gold) templates. Real model
  behaviour, checked against the item keys and gold; the cells still pass the gate.
- Knowledge (4 options): predicted letters are near uniform (A 1,426; B 1,358; C 1,300; D 1,372). The 0.4555 micro accuracy is a weak
  result, not a readout collapse.
- Score items (MIMIC, 494): the argmax of the distribution differs from the card's median level on 172 items (`results.md` section 4).
  Our harness scores by argmax of the distribution, as for every other model; the median readout is not used for accuracy.

## 5. Deviations and caveats

- transformers 5.18.0 in use (card asks >=5.19). The model loads and the card's example reproduces; not re-run on 5.19.
- torchvision 0.29.1+cu130 and pillow 13.0.0 and peft 0.21.2 are installed into the worktree venv by hand (`uv pip install`), not in
  `uv.lock`. The worktree `uv run --frozen --with ...` overlay resolved torch 2.15 and hit a disk quota, so it was not used.
- `ruff check .` passes. Full `pytest -m "not gpu"`: the first run had 23 `ModuleNotFoundError: peft` failures in 7 files that
  need peft, which I installed in the worktree venv; those 7 files then exit 0. Other files passed in the first run.
- Comparators: `loops/osler_v5/results` does not exist on `loop/osler_v5`; comparator values come from
  `loops/osler_v5/CLAIMS.md` (C090–C096) and `docs/paper/EVIDENCE.md` §5.2 on that branch. Missing values are NOT MEASURED. The
  pplx-flash micro figure on MR and MIMIC (osler_v4 C087, 0.624) is outside the permitted sources and is not used.
- Knowledge set: EVIDENCE quotes 5,435 paired items for pplx; this run scored all 5,456 items.
- Claims: `results.md` section 5 (D1-1 to D1-5) carries the R1 rows; `loops/d1_bench` has no CLAIMS.md (not requested).

## 6. Questions for the operator

1. License: is this research evaluation of a non-commercial kind under LFM 1.0 §1 and §5, or should the MedDecide program be checked
   against the "indirect commercial advantage" wording first?
2. transformers >=5.19 (the card's floor) versus the repo's 5.18.0: should the loader be re-run on 5.19 in a separate environment?
3. Should `torchvision`, `pillow` and `peft` be declared in `pyproject.toml` and `uv.lock` (a change to shared envs), or stay manual?
