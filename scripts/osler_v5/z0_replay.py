"""osler_v5 Z0 step 6 (ADVISORY Z0.2): replay osler_v4's Y8 batches on the CPU and measure the MIMIC share of the gradient under
per_step and per_item weighting, and the closed-list counts that the D52a expectation needs. Counts only: no item text, label,
identifier or value is printed or written (the mix holds MIMIC-derived rows that never leave this machine).

What is replayed: the same mix (composed in memory, composition 9d25e30f…), the same encoder (the pplx tokenizer), the same seed-0
epoch-0 order and the same batch former as scripts/osler_v4/y8_train.py. The batch sizes, real tokens and padded tokens of every step
are compared with the Y8 replay record (63,304 steps); a mismatch is a failure, not a note. The MIMIC and closed-list counts of each
batch are then read off the replay. The logged steps of the Y8 run (every 50th step) carry their template ids and pre-clip gradient
norms; their counts are compared with the replay, and the norms are used for the clip-aware shares.

Shares (per-item gradient magnitudes are taken as equal; the replay has no gradients; this is the gradient-coefficient share the
ADVISORY's 0.59 and 0.34 refer to):
- per_step, full run: mean over steps of (MIMIC items / items in the step); the applied coefficient of an item is 1/n.
- per_item, full run: MIMIC items / all items (each item carries 1/batch_size of the clipped update).
- clip-aware, logged steps (the primary check): the applied norm of a step is min(G_t, clip), G_t being the logged pre-clip norm.
  per_step: sum_t (m_t/n_t) a_t / sum_t a_t. per_item: sum_t (m_t/B) a_t / sum_t (n_t/B) a_t.

Usage (CPU; about 30 minutes; detached):
    uv run --frozen python scripts/osler_v5/z0_replay.py
"""

from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.bench.schema import Item
from meddecide.train.config import load_config
from meddecide.train.data import iter_batches, shuffled_order
from meddecide.train.final_mix import compose_final_mix
from meddecide.train.mix_items import mix_row_to_item
from meddecide.train.option_augment import CLOSED_LIST
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "configs" / "osler_v3" / "block_arm.yaml"
Y8 = REPO / "outputs" / "osler_v4" / "Y8"
OUT = REPO / "outputs" / "osler_v5" / "Z0" / "replay"
MIMIC_PREFIX = "mimic_"
STEPS = 63_304
PROGRESS_EVERY = 5_000


def _module(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LazyItems(Sequence):
    """The training items of the composed mix, converted on access (the same conversion the Y8 run applies to every row)."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Item:  # type: ignore[override]
        return mix_row_to_item(self.rows[index])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    t0 = time.time()
    y8_mix = _module("y8_final_mix", "scripts/osler_v4/y8_final_mix.py")
    rp = _module("w6_replay", "scripts/osler_v2/w6_replay.py")
    decisions = {name: y8_mix.entering(name) for name in ("B1", "B2", "B3", "B4")}
    family = {name: y8_mix.add_files(name) for name in y8_mix.FAMILY if decisions[name][0]}
    other = y8_mix.add_files("B4") if decisions["B4"][0] else []
    rows, composition = compose_final_mix(y8_mix.BASE, family, other, y8_mix.BROAD)
    print(f"composed {len(rows)} rows in {round(time.time() - t0)} s", flush=True)

    config = load_config(RECIPE)
    config = dataclasses.replace(config, backbone="pplx", kl_to_base=0.0, distill_weight=0.0)
    batch_size = int(config.batch_size)
    items = LazyItems(rows)
    encoder = rp.encoder_for("pplx", config)
    order = shuffled_order(len(items), seed=config.seed, epoch=0)
    batches = iter_batches(encoder, items, batch_size=config.batch_size, max_batch_tokens=config.max_batch_tokens,
                           max_prompt_tokens=config.max_prompt_tokens, order=order, augment_options=config.shuffle_options,
                           epoch=0, seed=config.seed)
    n_items = np.zeros(STEPS, dtype=np.int32)
    real = np.zeros(STEPS, dtype=np.int64)
    padded = np.zeros(STEPS, dtype=np.int64)
    mimic = np.zeros(STEPS, dtype=np.int32)
    closed = np.zeros(STEPS, dtype=np.int32)
    t1 = time.time()
    for t, (_, batch) in enumerate(itertools.islice(batches, STEPS)):
        n_items[t], real[t], padded[t] = rp.batch_record(batch)
        mimic[t] = sum(str(e.template_id).startswith(MIMIC_PREFIX) for e in batch)
        closed[t] = sum(str(e.template_id) in CLOSED_LIST for e in batch)
        if (t + 1) % PROGRESS_EVERY == 0:
            print(f"step {t + 1} ({round(time.time() - t1)} s)", flush=True)
    batches.close()
    drawn = int(np.count_nonzero(n_items))
    if drawn != STEPS:
        raise SystemExit(f"the replay drew {drawn} batches for {STEPS} steps")
    print(f"replayed {STEPS} steps in {round(time.time() - t1)} s", flush=True)

    # 1. the replay against the Y8 record: every step's size, real tokens and padded tokens
    y8_replay = json.loads((Y8 / "replays" / "final_mix_replay.json").read_text(encoding="utf-8"))
    mismatch = {
        "items": int(np.count_nonzero(n_items != np.asarray(y8_replay["sizes"]))),
        "real_tokens": int(np.count_nonzero(real != np.asarray(y8_replay["real_tokens"]))),
        "padded_tokens": int(np.count_nonzero(padded != np.asarray(y8_replay["padded_tokens"]))),
    }
    # 2. the logged steps of the Y8 run: template counts and the pre-clip norms
    logged = []
    with (Y8 / "logs" / "train.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("event") == "step":
                logged.append(record)
    log_check = {"logged_steps": len(logged), "n_items_mismatch": 0, "mimic_count_mismatch": 0, "closed_count_mismatch": 0}
    for record in logged:
        t = int(record["step"]) - 1
        ids = [str(x) for x in record["template_ids"]]
        log_check["n_items_mismatch"] += int(len(ids) != int(n_items[t]))
        log_check["mimic_count_mismatch"] += int(sum(i.startswith(MIMIC_PREFIX) for i in ids) != int(mimic[t]))
        log_check["closed_count_mismatch"] += int(sum(i in CLOSED_LIST for i in ids) != int(closed[t]))
    logged_t = np.array([int(r["step"]) - 1 for r in logged])
    grad_norm = np.array([float(r["grad_norm"]) for r in logged])
    clip = float(config.grad_clip)
    applied = np.minimum(grad_norm, clip)  # the clipped norm a step applies (per_step and per_item clip the same gradient)

    n_t = n_items.astype(np.float64)
    m_t = mimic.astype(np.float64)
    c_t = closed.astype(np.float64)
    share_ps_full = float(np.mean(m_t / n_t))
    share_pi_full = float(m_t.sum() / n_t.sum())
    nl, ml = n_t[logged_t], m_t[logged_t]
    share_ps_logged = float(np.sum(ml / nl * applied) / np.sum(applied))
    share_pi_logged = float(np.sum(ml / batch_size * applied) / np.sum(nl / batch_size * applied))
    horizons = {}
    for h in (13_992, 63_304):
        horizons[str(h)] = {
            "steps": h, "realised_items": int(n_t[:h].sum()), "mimic_items": int(m_t[:h].sum()),
            "closed_items": int(c_t[:h].sum()), "closed_share": float(c_t[:h].sum() / n_t[:h].sum()),
        }
    summary = {
        "started_utc": started, "finished_utc": utcnow(), "git_commit": git_commit(REPO), "composition_sha256": composition["composition_sha256"],
        "steps": STEPS, "batch_size": batch_size, "clip": clip, "realised_items": int(n_t.sum()),
        "mimic_items": int(m_t.sum()), "mimic_share_of_realised_items": share_pi_full,
        "replay_vs_y8_mismatches": mismatch, "logged_steps_vs_replay": log_check,
        "shares": {
            "per_step_full_run_unclipped": share_ps_full,
            "per_item_full_run_unclipped": share_pi_full,
            "per_step_clip_aware_logged": share_ps_logged,
            "per_item_clip_aware_logged": share_pi_logged,
            "logged_steps": len(logged), "logged_clipped_share": float(np.mean(grad_norm > clip)),
        },
        "horizons": horizons,
        "composition_counts": {k: v for k, v in composition.items() if k in ("rows", "base_rows", "replay_rows", "topup_rows")},
    }
    write_json(OUT / "summary.json", summary)
    np.savez(OUT / "steps.npz", n_items=n_items, real_tokens=real, padded_tokens=padded, mimic_items=mimic, closed_items=closed,
             logged_step_index=logged_t, logged_grad_norm=grad_norm)
    print(json.dumps({k: summary[k] for k in ("replay_vs_y8_mismatches", "logged_steps_vs_replay", "shares", "horizons")},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
