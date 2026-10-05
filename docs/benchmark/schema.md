# MedDecide-Bench item schema (v0)

**Status:** authored 2026-10-05 (loop `bench_v0`, task T2).
**Code:** `src/meddecide/bench/schema.py` — this document is the human-readable contract;
the pydantic model is the executable one. Where they disagree, the model wins and this
document is wrong.

One item format is spoken by every source loader, the eval harness, and every model. The
same schema covers the established tier and the fresh tier, so a baseline number is
always produced from the same object.

---

## 1. Question types

| `qtype` | shape | answer readout |
|---|---|---|
| `noul` | yes / no | `p(yes)` |
| `choice` | 2–255 named options | distribution over options |
| `score` | 2–10 ordered levels | distribution + expected level |

`noul` items are stored as ordinary items with exactly two options whose keys are `yes`
and `no` (in that order), so every question type flows through the same scoring path.
`score` levels are keyed `"1"` … `"N"` in ascending order, lowest first.

## 2. Fields

| field | type | meaning |
|---|---|---|
| `item_id` | str (16 hex) | Stable hash of `source` + `source_record_id` + `template_id` + `option_order_seed`. Same logical item → same id on any machine. |
| `tier` | `established` \| `fresh` | Tier 1 (public test sets) or tier 2 (post-cutoff, structured-field gold). |
| `source` | str | Short source name, e.g. `clinicaltrials`, `openfda`, `pubmed`, `medqa`. |
| `source_record_id` | str | The upstream identifier (`NCT…`, `set_id`, PMID, dataset row id). |
| `source_url` | str | http(s) URL a human can open to check the record. |
| `source_license` | str | License of the source, or `UNKNOWN`. |
| `record_date` | date | The date the *filter* uses (first posted, first effective, entry date). |
| `split` | `train` \| `dev` \| `test` | Official split. Test splits are evaluation-only. |
| `template_id` | str | Which template produced the item; templates are screened in T8. |
| `skill` | str | Skill label (`medical_knowledge`, `evidence`, `relevance`, `trial_design`, …). |
| `qtype` | `noul` \| `choice` \| `score` | See §1. |
| `state` | str | The text the model conditions on. Answer-bearing fields are removed. |
| `question` | str | The question asked about the state. |
| `options` | list of `{key, label, description?}` | Allowed answers, in display order. |
| `gold` | str | One of the option keys. Derived from a structured source field, never from a model. |
| `meta` | dict | Free-form per-template metadata (seed, distractor source, filter counts). |

### Option keys

* `choice` / `score` templates number options `A`, `B`, `C`, … (single uppercase letters).
* `noul` items use `yes` / `no`.
* A key must match `[A-Za-z0-9]{1,8}` — the harness reads it as one token, so punctuation
  in keys would break the readout.

## 3. Invariants (enforced by validation, not by convention)

1. `gold` is one of the option keys.
2. `choice`: 2–255 options; option keys unique.
3. `score`: 2–10 options, keys exactly `1..N` ascending, `gold` a level key.
4. `noul`: keys exactly `["yes", "no"]`, `gold` ∈ {`yes`, `no`}.
5. `source_url` starts with `http://` or `https://`.
6. Unknown fields are rejected (`extra="forbid"`) — a typo in a loader is an error, not a
   silently dropped field.

## 4. Splits

* **Tier 1:** the official split is preserved. Where a source publishes no usable test
  labels (e.g. MedMCQA's official test), the official validation split is used as tier-1
  test and a dev split is carved from train; the choice is recorded in the manifest.
* **Fresh tier:** items are assigned `dev` (20 %) / `test` (80 %) by a hash of
  `source_record_id` (`split_by_record_hash`), never by item. All items derived from one
  source record therefore land in the same split, so a template cannot leak a record
  across the dev/test boundary.
* No `(source, source_record_id)` pair may appear in more than one split within a tier
  (checked by the loaders' acceptance check).

## 5. Manifest

Each build writes `data/bench/<tier>/manifest.json`:

* per file: path, sha256, row count;
* counts nested by `source → split → qtype → template_id`;
* `count_by_source` / `count_by_split` / `count_by_qtype` totals.

Every count is checkable: the source counts sum to the file total (the manifest builder
raises if they do not). No filter is allowed to drop a record silently — loaders return
counts and reasons (`LoadReport`), and dropped records are listed with their reason.

## 6. What is *not* in an item

* No answer-bearing text: the field the question asks about is removed from `state`.
* No model output of any kind. Gold is a deterministic lookup of a structured field.
* No source text that carries a redistribution restriction beyond its license.

## 7. Reproducing an item

```bash
uv run python -c "
from meddecide.bench.schema import compute_item_id
print(compute_item_id('clinicaltrials', 'NCT00000001', 'ct_phase_v1', 7))
"
```

The id is a truncated sha256 over canonical JSON (sorted keys, NFC-normalised text), so
it is stable across platforms and processes.
