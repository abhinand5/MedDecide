"""osler_v2 POST-HOC check (labelled; not part of any verdict): the D35 Naive Bayes screen on the CodiEsp dev carve-out.

Q2 (STATE §6) promised the D35 screen for codiesp_icd_support_v1 at W5, before any osler_v2 block. It was not run then. This runs
it now, on the same rule (NB dev macro >= 0.90 excludes), with the template's C1 training rows as the NB training split and the
400 carve-out dev rows of fast dev v2 as the dev split. The result is reported; it does not change any verdict by itself.

Writes outputs/osler_v2/W8/codiesp_screen_posthoc.json.

Usage: uv run --frozen python scripts/osler_v2/w8_codiesp_screen_posthoc.py
"""

from __future__ import annotations

from pathlib import Path

from meddecide.gen.screen import screen_generator
from meddecide.utils.io import iter_jsonl, write_json

REPO = Path(__file__).resolve().parents[2]
C1 = REPO / "data" / "train" / "osler_v2" / "block_C1.jsonl"
FASTDEV = REPO / "data" / "bench" / "osler_v2" / "fastdev_v2.jsonl"
OUT = REPO / "outputs" / "osler_v2" / "W8" / "codiesp_screen_posthoc.json"
TEMPLATE = "codiesp_icd_support_v1"


def main() -> None:
    # CodiEsp rows have no twins: each row is its own pair (the screen reads pair_id for the twin statistic only)
    train = [{**r, "pair_id": r["item_id"]} for r in iter_jsonl(C1) if r["template_id"] == TEMPLATE]
    dev = [{**r, "pair_id": r["item_id"]} for r in iter_jsonl(FASTDEV) if r["template_id"] == TEMPLATE]
    rep = screen_generator(TEMPLATE, False, dev, train, "C1 codiesp train rows (post hoc)", threshold=0.90)
    write_json(OUT, {"kind": "osler_v2_posthoc_codiesp_screen", "label": "POST HOC (Q2 promised this at W5; not run then)",
                     "n_train": len(train), "n_dev": len(dev), "report": rep.__dict__, "rule": "D35 (b): NB dev macro >= 0.90"})
    print(f"codiesp post-hoc NB dev macro {rep.naive_bayes_macro} (n_dev {len(dev)}, n_train {len(train)}); "
          f"passes {rep.passes}", flush=True)


if __name__ == "__main__":
    main()
