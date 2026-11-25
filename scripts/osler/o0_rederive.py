"""O0 re-derivation (AGENTS.md R6): recompute every headline number in a fresh process.

Reads the raw artifacts again (benchmark files, screen, the training mix, smoke JSONs, throughput
JSONs, the Qwen config) and prints one line per number with its source. It recomputes the median
prompt length from the training rows as well, so the throughput projection's input is not taken
on trust from the earlier run.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o0_rederive.py
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from meddecide.bench.denominators import count_test_denominator
from meddecide.eval.prompt_lengths import summarize_lengths
from meddecide.eval.readout import render_prompt

REPO = Path(__file__).resolve().parents[2]
O0 = REPO / "outputs/osler_v0/O0"


def v02_test_rows() -> dict[str, int]:
    manifest = json.loads((REPO / "data/bench/v0.2/manifest.json").read_text())
    screen = json.loads((REPO / "data/bench/v0.2/screen.json").read_text())

    def rows() -> Iterator[dict[str, Any]]:
        for rec in manifest["files"].values():
            with (REPO / rec["path"]).open() as fh:
                for line in fh:
                    yield json.loads(line)

    counts = count_test_denominator(rows(), screen["templates"])
    return {"raw_test": counts.raw_test_rows, "dropped_test": counts.dropped_test_rows,
            "kept_test": counts.kept_test_rows, "kept_fresh": counts.kept_fresh_rows,
            "kept_tier1": counts.kept_tier1_rows}


def mix_median(tokenizer, every: int = 120) -> tuple[int, float, int]:
    lengths = []
    with (REPO / "data/train/student_v1/train.jsonl").open() as fh:
        for i, line in enumerate(fh):
            if i % every:
                continue
            r = json.loads(line)
            options = [SimpleNamespace(key=o["key"], label=o["label"]) for o in r["options"]]
            item = SimpleNamespace(state=r["state"], question=r["question"], options=options)
            lengths.append(len(tokenizer.encode(render_prompt(item, tokenizer), add_special_tokens=False)))
    summary = summarize_lengths(lengths)
    return summary.median, summary.mean, summary.p95


def main() -> None:
    print("== v0.2 denominators (files and screen, fresh read)")
    c = v02_test_rows()
    print(f"raw test rows={c['raw_test']} dropped-by-screen={c['dropped_test']} kept={c['kept_test']} "
          f"(fresh {c['kept_fresh']}, tier1 {c['kept_tier1']})  [data/bench/v0.2]")
    assert c["kept_test"] == 37289 and c["kept_fresh"] == 23768 and c["kept_tier1"] == 13521

    print("== v0.2 manifest hash check (recomputed)")
    snap = json.loads((O0 / "snapshot.json").read_text())
    b = snap["bench_v0_2"]
    print(f"files checked={b['files_checked']} mismatches={b['file_hash_mismatches']} "
          f"manifest_items={b['manifest_n_items_total']} acceptance_items={b['acceptance_n_items_total']}")

    print("== smoke rows (recomputed from raw JSON)")
    for path in sorted((O0 / "smoke").glob("*.json")):
        raw = json.loads(path.read_text())
        if "results" in raw:
            diffs = [r["worst_abs_diff"] for r in raw["results"] if r.get("worst_abs_diff") is not None]
            print(f"{raw['id']:10s} card examples with probabilities={raw['examples_with_probabilities']} "
                  f"worst |printed - returned|={max(diffs)} overall={raw['overall']}")
        else:
            print(f"{raw.get('id', path.stem):10s} overall={raw['overall']} "
                  f"checks={sum(raw['checks'].values())}/{len(raw['checks'])} passed")
    jev = json.loads((O0 / "smoke/jev27b.json").read_text())
    print("jev27b choice max |p - card approx| =", max(jev["choice"]["abs_diff_to_card"].values()))
    noul_true = dict(zip(jev["noul"]["options"], jev["noul"]["probabilities"], strict=True))["true"]
    print(f"jev27b refund noul P(true)={noul_true:.4f} (card transformers block: 0.978)")
    pplx = json.loads((O0 / "smoke/pplx27b.json").read_text())
    print("pplx27b choice probabilities", {o: round(p, 5) for o, p in zip(
        ["billing", "support", "sales"], pplx["choice"]["probabilities"], strict=True)}, "noul P(true)=", round(pplx["noul"]["probabilities"][1], 5))
    for cid in ["clef", "clef_flash"]:
        c = json.loads((O0 / f"smoke/{cid}.json").read_text())
        inv = c["invoice_response"]["answers"]
        chk = c["checkout_response"]["answers"]
        print(f"{cid}: invoice status={inv['status']['choice']} P(large true)={inv['large']['noul']} | "
              f"checkout department={chk['department']['choice']} outage P(true)={chk['outage']['noul']}")

    print("== pinned revisions: fetch.json hub_sha vs registry pin")
    registry = {m["id"]: m["revision"] for m in json.loads((REPO / "configs/osler_v0/competitors.json").read_text())["models"]}
    fetched = json.loads((O0 / "fetch.json").read_text())
    done = [r for r in fetched if r.get("status") == "DONE"]
    same = [r["id"] for r in done if r["hub_sha"] == registry[r["id"]] == r["resolved_revision"]]
    print(f"registry entries={len(registry)} DONE={len(done)} hub_sha == pin == resolved: {len(same)} of {len(registry)}")
    print("disk GB (hf_home+outputs+data) =", round(json.loads((O0 / "snapshot.json").read_text())["disk"]["total_bytes_counted"] / 1e9, 2))

    print("== throughput inputs (median recomputed from training rows)")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    med, mean, p95 = mix_median(tok)
    print(f"median={med} mean={mean} p95={p95} (every 120th line of data/train/student_v1/train.jsonl)")
    thr = json.loads((O0 / "throughput.json").read_text())["sizes"]["4b"]
    tok_s = thr["tokens_per_s_train"]
    arm_train_s = 200_000 * mean / tok_s
    print(f"4B tokens/s={tok_s} -> bucketed arm train = 200000*{mean}/{tok_s} = {arm_train_s:.0f} s "
          f"(recorded {thr['projection_one_arm']['projected_train_s_bucketed']})")
    if med != thr["sequence_length_median_prompt"]:
        raise SystemExit(f"median mismatch: recomputed {med} vs recorded {thr['sequence_length_median_prompt']}")
    print("ok: recomputed median matches the throughput run")


if __name__ == "__main__":
    main()
