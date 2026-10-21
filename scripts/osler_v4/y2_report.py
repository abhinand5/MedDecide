"""Writes loops/osler_v4/mimic.md §2 (the Y2 templates) from the Y2 count files, and checks leakage across splits.

Counts, screens, drop reasons and hash overlaps only: no item text is written anywhere committed. Keeps §1 (Y1) as it is.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPORT = Path("loops/osler_v4/mimic.md")
ITEMS = Path("data/mimic_build/items")
OUT = Path("outputs/osler_v4/Y2")
SPLITS = ("train", "dev", "test")
ADDITIONAL = {"mednli_v1"}   # the MedNLI test split is an additional row, not in any verdict
MIMIC_TEMPLATES = ("mimic_icd_support_v1", "mimic_icd_specificity_v1", "mimic_principal_dx_v1", "mimic_msdrg_v1",
                   "mimic_drg_cc_level_v1", "mimic_apr_severity_v1", "mimic_apr_mortality_v1", "mimic_pcs_support_v1",
                   "mimic_rad_exam_v1", "mimic_ed_acuity_v1", "mimic_ed_disposition_v1", "mimic_lab_validation_v1",
                   "mimic_med_given_v1")


def split_keys(path: Path) -> tuple[set[str], set[str]]:
    """(state-text hashes, record ids) of one items file. Text is hashed, never written anywhere."""
    if not path.exists():
        return set(), set()
    hashes: set[str] = set()
    records: set[str] = set()
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            hashes.add(hashlib.sha256(row["state"].encode("utf-8")).hexdigest())
            records.add(row["source_record_id"])
    return hashes, records


def overlap_check(template: str) -> dict[str, dict[str, int]]:
    keys = {s: split_keys(ITEMS / f"{template}__{s}.jsonl") for s in SPLITS}
    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    return {
        "state": {f"{a}-{b}": len(keys[a][0] & keys[b][0]) for a, b in pairs},
        "record": {f"{a}-{b}": len(keys[a][1] & keys[b][1]) for a, b in pairs},
    }


def main() -> None:
    counts = {}
    for name in ("counts.json", "counts_lab_med.json"):
        path = OUT / name
        if path.exists():
            counts.update(json.loads(path.read_text())["templates"])
    text = REPORT.read_text() if REPORT.exists() else ""
    head = text.split("\n## 2.")[0].rstrip() + "\n" if "\n## 2." in text else text.rstrip() + "\n"
    lines = ["", "## 2. MIMIC structured-gold templates and MedNLI (osler_v4 Y2)", "",
             "Every label is a database field or a code-fixed rule (ADVISORY Y2; the coder's codes are not proof of documentation,",
             "ADVISORY §7.1). Test rows are late-anchor patients only. Held-out families (heart failure; AKI, ADVISORY §3.4) are in a",
             "held-out file and in no split. MedNLI's test split is an additional row. Screens (D35, dev): Naive Bayes macro accuracy",
             "trained on train, and the majority-label rate; a Naive Bayes score of 0.90 or more is a shortcut.", "",
             "| template | train items | dev items | test items | held-out items | NB macro dev | majority dev | state-text overlap (train-dev / train-test / dev-test) | record-id overlap (train-dev / train-test / dev-test) |",
             "|---|---|---|---|---|---|---|---|---|"]
    details = []
    for template in [*MIMIC_TEMPLATES, "mednli_v1"]:
        entry = counts.get(template)
        if entry is None:
            lines.append(f"| {template} | NOT MEASURED | | | | | | | |")
            continue
        splits = entry["splits"]
        items = {s: splits.get(s, {}).get("items", 0) for s in SPLITS}
        held = sum(splits.get(s, {}).get("heldout_items", 0) for s in SPLITS)
        screens = entry.get("screens", {})
        leak = overlap_check(template)
        label = f"{template} (additional test)" if template in ADDITIONAL else template
        st, rec = leak["state"], leak["record"]
        lines.append(f"| {label} | {items['train']} | {items['dev']} | {items['test']} | {held} | "
                     f"{screens.get('nb_macro_accuracy_dev', '-')} | {screens.get('majority_rate_dev', '-')} | "
                     f"{st['train-dev']} / {st['train-test']} / {st['dev-test']} | "
                     f"{rec['train-dev']} / {rec['train-test']} / {rec['dev-test']} |")
        for s in SPLITS:
            d = splits.get(s)
            if not d:
                continue
            drops = "; ".join(f"{k} {v}" for k, v in sorted(d["drops"].items(), key=lambda kv: -kv[1])) or "none"
            short = "; ".join(f"{k} {v}" for k, v in sorted(d["shortage"].items())) or "none"
            details.append(f"- {template} {s}: drops: {drops}. Shortage: {short}. Truncated states: {d['truncated_states']}.")
    lines += ["", "Drop reasons and shortages (counts; the screens are on dev):", ""]
    lines += [*details, ""]
    REPORT.write_text(head + "\n".join(lines) + "\n")
    print(f"wrote §2 to {REPORT}")


if __name__ == "__main__":
    main()
