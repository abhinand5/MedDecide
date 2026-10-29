"""synth_v1: convert the MIMIC-grounded items into osler_v4/v5 training rows, drop the mix exclusions, and check the caps.

Reads outputs/synth_v1/items_mimic_v1.jsonl (code only; no item text is printed). Writes, under outputs/synth_v1/train/:
  grounded_rows.jsonl     the training rows (gitignored; MIMIC-derived text)
  grounded_report.json    aggregates only: counts, drop reasons, per-template counts, held-out and split checks

The question text per template is the osler_v4 mimic_items.py text. Option keys follow the osler_v4 convention (letters for the
DRG, CC and disposition templates; digits for APR). Gold is mapped to the key of the grounded label; the grounded options are not
changed in content.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
import string
from pathlib import Path
from typing import Any

from meddecide.mix.rows import make_row
from meddecide.utils.io import write_json

ITEMS = Path("/workspace/MedDecide/outputs/synth_v1/items_mimic_v1.jsonl")
MANIFEST = Path("/workspace/MedDecide/outputs/synth_v1/manifest_mimic_v1.json")
OUT = Path("/workspace/MedDecide/outputs/synth_v1/train")
SYNTH_SOURCE = "synth_mimic_v1"
LICENCE = "synthetic training row generated from MIMIC-IV-derived structured facts (training split only)"
CAP_BLOCK = 0.20
CAP_TEMPLATE = 0.08
LETTERS = string.ascii_uppercase

QUESTIONS = {
    "mimic_msdrg_v1": ("mimic_msdrg", "choice", "Which MS-DRG was assigned to this admission?"),
    "mimic_drg_cc_level_v1": ("mimic_drg_cc_level", "choice",
                              "Which severity level (MCC, CC or neither) is in this admission's MS-DRG title?"),
    "mimic_apr_severity_v1": ("mimic_apr_severity", "score",
                              "What is the APR-DRG severity of illness level for this admission (1 to 4)?"),
    "mimic_ed_disposition_v1": ("mimic_ed_disposition", "choice", "What was the disposition of this emergency department visit?"),
}
# osler_v4 key convention per template (position for msdrg; named map for the others; identity for apr)
KEY_MAP = {
    "mimic_drg_cc_level_v1": {"MCC": "A", "CC": "B", "neither": "C"},
    "mimic_ed_disposition_v1": {"admitted": "A", "home": "B", "other": "C"},
}
HELD_OUT_TEXT = re.compile(r"heart failure|acute kidney|kidney injury|\bCHF\b", re.IGNORECASE)
HELD_OUT_CODE = re.compile(r"\b(I50|N17)")


def _load_exclusions() -> tuple[dict[str, str], Any]:
    module = importlib.import_module("meddecide.synth.mimic_families")  # PYTHONPATH: the synth code at commit 8c1c7bb
    return dict(module.MIX_EXCLUSIONS), module.in_training_mix, module.HELD_OUT_TEXT


def convert(item: dict[str, Any], in_mix: Any, held_text: Any = HELD_OUT_TEXT) -> tuple[dict[str, Any] | None, str]:
    template = item["template"]
    if not in_mix(template):
        return None, "mix_exclusion"
    if held_text.search(str(item["text"])) or HELD_OUT_CODE.search(str(item["text"])):
        return None, "held_out_text_match"
    if template not in QUESTIONS:
        return None, "unknown_template"
    skill, qtype, question = QUESTIONS[template]
    keys = [str(k) for k in item["options"]]
    labels = [str(d) for d in item["display"]]
    if len(keys) != len(labels) or len(set(keys)) != len(keys):
        return None, "option_key_or_label_mismatch"
    if template in KEY_MAP:
        mapping = KEY_MAP[template]
        if any(k not in mapping for k in keys):
            return None, "unmapped_option_key"
        new_keys = [mapping[k] for k in keys]
    elif template == "mimic_apr_severity_v1":
        if keys != ["1", "2", "3", "4"]:
            return None, "unexpected_apr_keys"
        new_keys = keys
    else:  # mimic_msdrg_v1: positional letters, as in osler_v4
        if len(keys) > len(LETTERS):
            return None, "too_many_options"
        new_keys = list(LETTERS[: len(keys)])
    label = str(item["label"])
    if label not in keys:
        return None, "gold_not_offered"
    gold = new_keys[keys.index(label)]
    try:
        row = make_row(source=SYNTH_SOURCE, template_id=template, skill=skill, qtype=qtype, state=str(item["text"]),
                       question=question, options=list(zip(new_keys, labels, strict=True)), gold=gold, record_id=str(item["case_id"]),
                       record_url="", licence=LICENCE, record_date="", split="train", tier="mimic",
                       meta={"family": item["family"], "style": item["style"], "target_words": item["target_words"],
                             "word_count": item["word_count"]})
    except ValueError:
        return None, "make_row_refused"
    return row, "kept"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    exclusions, in_mix, held_text = _load_exclusions()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = [json.loads(line) for line in ITEMS.read_text(encoding="utf-8").splitlines() if line.strip()]
    drops: collections.Counter[str] = collections.Counter()
    per_template_in: collections.Counter[str] = collections.Counter(i["template"] for i in items)
    rows: list[dict[str, Any]] = []
    for item in items:
        row, reason = convert(item, in_mix, held_text)
        drops[reason] += 1
        if row is not None:
            rows.append(row)
    ids = [r["item_id"] for r in rows]
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        raise SystemExit(f"{duplicates} duplicate item ids after conversion; refusing to write")
    per_template = collections.Counter(r["template_id"] for r in rows)
    # held-out and split checks (counts only)
    held_text_in = sum(1 for i in items if held_text.search(str(i["text"])) or HELD_OUT_CODE.search(str(i["text"])))
    held_text_rows = sum(1 for r in rows if held_text.search(str(r["state"])) or HELD_OUT_CODE.search(str(r["state"])))
    not_train = sum(1 for r in rows if r["split"] != "train")
    report = {
        "items_in": len(items),
        "items_per_template_in": dict(sorted(per_template_in.items())),
        "drops_by_reason": dict(sorted(drops.items())),
        "mix_exclusions_from_code": dict(sorted(exclusions.items())),
        "mix_exclusion_items": drops.get("mix_exclusion", 0),
        "rows_out": len(rows),
        "per_template_rows": dict(sorted(per_template.items())),
        "held_out_text_matches_in_items_before_drop": held_text_in,
        "held_out_text_matches_in_rows_after_drop": held_text_rows,
        "rows_not_train_split": not_train,
        "manifest_split": manifest["config"]["split"],
        "manifest_has_mix_exclusions_key": "mix_exclusions" in manifest,
        "duplicate_item_ids": duplicates,
    }
    if held_text_rows or not_train:
        raise SystemExit(f"held-out or split check failed: {held_text_rows} held-out text matches, {not_train} non-train rows")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "grounded_rows.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    write_json(args.out / "grounded_report.json", report)
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
