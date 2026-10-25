"""osler_v4 Y9 (defect screen before the MR verdict is read; D35; CPU): the shortcut screen of the two MR verdict templates that
osler_v1 flagged as too good and did not explain: pmc_similar_patient_v1 (open since V180 to V183 of loops/osler_v1/CLAIMS.md; the
Naive Bayes screen was never run) and mr_icd_support_template_v2 (1.0000 on 554 test items; NB 0.804 at build).

For each template: the naive-Bayes macro accuracy on its fast dev v4 items, the naive Bayes fit on the template's own rows of the
osler_v2 C1 block (the base of the osler_v4 final mix), the string-presence macro (choice items), and the share of dev items whose
gold label occurs in the record text. The D35 limit is 0.90 for the two macro screens. Prints counts and figures only; no item text.
Writes outputs/osler_v4/Y9/mr_screen.json.

Usage (CPU):
    CUDA_VISIBLE_DEVICES= uv run --frozen python scripts/osler_v4/y9_mr_screen.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from meddecide.gen.screen import (
    NaiveBayes,
    gold_in_state,
    gold_label,
    macro_accuracy,
    string_presence_predict,
)
from meddecide.utils.io import write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "data" / "train" / "osler_v2" / "block_C1.jsonl"
DEV = REPO / "data" / "bench" / "osler_v4" / "fastdev_v4.jsonl"
OUT = REPO / "outputs" / "osler_v4" / "Y9" / "mr_screen.json"
TEMPLATES = ("pmc_similar_patient_v1", "mr_icd_support_template_v2")
D35_LIMIT = 0.90


def stream_templates(path: Path, templates: tuple[str, ...]) -> list[dict]:
    """The rows of the given templates, read line by line (the C1 block is several gigabytes)."""
    needles = tuple(f'"template_id":{sp}"{t}"' for t in templates for sp in ("", " "))
    kept: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if any(n in line for n in needles):
                kept.append(json.loads(line))
    return kept


def main() -> None:
    train: dict[str, list[dict]] = collections.defaultdict(list)
    for row in stream_templates(TRAIN, TEMPLATES):
        train[str(row["template_id"])].append(row)
    dev: dict[str, list[dict]] = collections.defaultdict(list)
    for row in stream_templates(DEV, TEMPLATES):
        dev[str(row["template_id"])].append(row)
    figures: dict[str, object] = {"generated_utc": utcnow(), "d35_limit": D35_LIMIT, "templates": {}}
    for template in TEMPLATES:
        items = dev[template]
        nb = NaiveBayes().fit(train[template])
        nb_macro = macro_accuracy([(gold_label(r), nb.predict(r)) for r in items])
        majority = collections.Counter(gold_label(r) for r in train[template]).most_common(1)[0][0]
        choice = [r for r in items if r["qtype"] != "noul"]
        sp_macro = macro_accuracy([(gold_label(r), string_presence_predict(r, majority)) for r in choice]) if choice else None
        in_state = sum(1 for r in items if gold_in_state(r))
        figures["templates"][template] = {
            "train_rows": len(train[template]), "dev_items": len(items), "nb_macro_dev": round(nb_macro, 4),
            "nb_passes_d35": nb_macro < D35_LIMIT,
            "string_presence_macro_dev": None if sp_macro is None else round(sp_macro, 4),
            "gold_in_record_text": in_state,
        }
        sp_text = "NOT APPLICABLE (yes/no items only)" if sp_macro is None else f"{sp_macro:.4f}"
        print(f"{template}: train {len(train[template])}, dev {len(items)}, NB macro dev {nb_macro:.4f}, "
              f"string-presence macro dev {sp_text}, gold in record text {in_state} of {len(items)}", flush=True)
    write_json(OUT, figures)


if __name__ == "__main__":
    main()
