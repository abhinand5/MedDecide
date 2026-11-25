"""d3_bench step 3: draw the fixed seeded subsamples of three frozen Y9 sets and write their id lists and hashes.

- knowledge (MedQA + MedMCQA test items): 1,000 items, simple random sample, seed 0;
- panel (external clinical panel): 500 items, stratified by source dataset (7), proportional allocation, seed 0;
- mr_verdict (MR verdict v1): 500 items, simple random sample, seed 0.

Only these three set files are read; no MIMIC-derived file is opened. The output holds item ids, counts and hashes only.

Usage:
    uv run --frozen python scripts/d3_bench/make_subsamples.py --out outputs/d3_bench/subsamples.json
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from meddecide.bench.quick_bench import id_list_sha256, seeded_sample, stratified_sample
from meddecide.utils.io import iter_jsonl, write_json

REPO = Path(__file__).resolve().parents[2]
Y9 = REPO / "data" / "bench" / "osler_v4" / "y9"
if not Y9.exists():  # a git worktree shares data/ with the main checkout
    Y9 = REPO.parents[2] / "data" / "bench" / "osler_v4" / "y9"
SEED = 0
PLAN = {"knowledge": ("simple", 1000), "panel": ("stratified:source", 500), "mr_verdict": ("simple", 500)}
EXPECTED_SHA256 = {  # the frozen files as distill_v0 D1 recorded them
    "knowledge": "aa3eb27d637a40dfe3602840565a5a22569cc0d0efb775dda9485d6b2983e082",
    "panel": "84836287043543c864527c9714780622282794be3fc210a30fd1e51aac30be99",
    "mr_verdict": "dbb178c64cd1515b00fc8c8678deb3b59a73c708379a8a834116723af5e217e8",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("outputs/d3_bench/subsamples.json"))
    args = parser.parse_args()
    result: dict = {"seed": SEED, "y9_dir": str(Y9), "sets": {}}
    for name, (method, k) in PLAN.items():
        path = Y9 / f"{name}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != EXPECTED_SHA256[name]:
            raise SystemExit(f"{name}: sha256 {digest} is not the frozen file's")
        items = list(iter_jsonl(path))
        allocation = None
        if method == "simple":
            ids = seeded_sample([str(r["item_id"]) for r in items], k, SEED)
        else:
            ids, allocation = stratified_sample(items, lambda r: str(r["source"]), k, SEED)
        chosen = set(ids)
        by_template = Counter(str(r["template_id"]) for r in items if str(r["item_id"]) in chosen)
        result["sets"][name] = {
            "file": f"data/bench/osler_v4/y9/{name}.jsonl", "file_sha256": digest, "set_items": len(items),
            "method": method, "k": k, "allocation": allocation, "by_template": dict(sorted(by_template.items())),
            "ids_sha256": id_list_sha256(ids), "item_ids": ids,
        }
        print(f"{name}: {len(ids)} of {len(items)} ({method}); ids sha256 {id_list_sha256(ids)[:16]}", flush=True)
    write_json(args.out, result)


if __name__ == "__main__":
    main()
