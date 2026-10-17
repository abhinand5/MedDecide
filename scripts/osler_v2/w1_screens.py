"""W1 step 2 and 3 (osler_v2 ADVISORY W1): O3 Naive Bayes screen per MR test template on its dev items, and the
ceiling statistic (zero-shot Qwen3.5-4B accuracy on each template's dev items).

Dev and train rows come from the osler_v1 generated and real-text files. Held-out rows are never read into a screen.
Templates without a dev split are reported as NOT MEASURED with the reason (no silent drops).

Usage (after the zero-shot dev run has written its predictions):
    uv run --frozen python scripts/osler_v2/w1_screens.py
Writes outputs/osler_v2/W1/screens.json.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.gen.screen import screen_generator

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "data" / "gen" / "osler_v1"
SYN = REPO / "data" / "gen" / "osler_v1_synthea"
REAL = GEN / "realtext"
ZS_PREDICTIONS = REPO / "outputs" / "osler_v2" / "W1" / "zeroshot" / "qwen35-4b" / "predictions.jsonl"
OUT = REPO / "outputs" / "osler_v2" / "W1" / "screens.json"
CEILING_THRESHOLD = 0.97  # D35 (a)
SCREEN_THRESHOLD = 0.90  # D35 (b)

# MR test templates of osler_v1's evaluation file, with where their train and dev rows live.
# (template, train file(s), dev file(s), generator field value or None for "all rows of the file")
SOURCES: dict[str, tuple[list[Path], list[Path], str | None]] = {
    "mr_page_category_tabular_v1": ([GEN / "train.jsonl"], [GEN / "dev.jsonl"], "mr_page_category_tabular_v1"),
    "mr_indicator_evidence_v1": ([GEN / "train.jsonl"], [GEN / "dev.jsonl"], "mr_indicator_evidence_v1"),
    "mr_icd_support_template_v2": ([GEN / "train.jsonl"], [GEN / "dev.jsonl"], "mr_icd_support_template_v2"),
    "synthea_page_category_v1": ([SYN / "train.jsonl"], [SYN / "dev.jsonl"], "synthea_page_category_v1"),
    "synthea_indicator_evidence_v1": ([SYN / "train.jsonl"], [SYN / "dev.jsonl"], "synthea_indicator_evidence_v1"),
    "mtsamples_page_category_v1": ([REAL / "mtsamples_page_category_v1.jsonl"], [REAL / "mtsamples_page_category_v1.jsonl"],
                                   "mtsamples_page_category_v1"),
    "pmc_similar_patient_v1": ([REAL / "pmc_similar_patient_v1.jsonl"], [REAL / "pmc_similar_patient_v1.jsonl"],
                               "pmc_similar_patient_v1"),
    "codiesp_icd_support_v1": ([REAL / "codiesp_icd_support_v1.jsonl"], [], "codiesp_icd_support_v1"),
    "mkurman_icd_support_v1": ([], [], "mkurman_icd_support_v1"),
}
NOT_MEASURED_REASON = {
    "codiesp_icd_support_v1": "no dev items yet: the CodiEsp dev carve-out (15 % of train cases) is built in W5",
    "mkurman_icd_support_v1": "no train or dev split: evaluation-only source (osler_v1 V5b), so no dev screen exists",
}
SPLIT_FIELD = "split"


def read_rows(paths: list[Path], generator: str | None, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if generator is not None and row.get("generator") != generator:
                    continue
                if row.get(SPLIT_FIELD) != split:
                    continue
                if row.get("held_out"):
                    raise ValueError(f"held-out row in {split} for {generator}: {path}")
                rows.append(row)
    return rows


def ceiling_from_predictions(template: str) -> dict[str, Any] | None:
    """Zero-shot accuracy on the template's dev items (predictions written by o2_run_zeroshot.py)."""
    if not ZS_PREDICTIONS.exists():
        return None
    hits = total = 0
    for line in ZS_PREDICTIONS.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("template_id") != template or row.get("set_name") != "mr_fastdev":
            continue
        if row.get("status") != "scored":
            continue
        keys = row["option_keys"]
        pred = keys[max(range(len(keys)), key=lambda i: row["probs"][i])]
        hits += pred == row["gold"]
        total += 1
    return {"n": total, "accuracy": None if total == 0 else round(hits / total, 4)}


def main() -> None:
    report: dict[str, Any] = {"thresholds": {"ceiling": CEILING_THRESHOLD, "screen": SCREEN_THRESHOLD},
                              "templates": {}}
    for template, (train_paths, dev_paths, generator) in SOURCES.items():
        entry: dict[str, Any] = {}
        train = read_rows(train_paths, generator, "train") if train_paths else []
        dev = read_rows(dev_paths, generator, "dev") if dev_paths else []
        entry["n_train"] = len(train)
        entry["n_dev"] = len(dev)
        entry["dev_gold_counts"] = dict(Counter(r["gold"] for r in dev))
        if not dev or not train:
            entry["screen"] = {"status": "NOT MEASURED", "reason": NOT_MEASURED_REASON[template]}
            entry["ceiling"] = {"status": "NOT MEASURED", "reason": NOT_MEASURED_REASON[template]}
            report["templates"][template] = entry
            print(template, "NOT MEASURED", flush=True)
            continue
        rep = screen_generator(template, False, dev, train, "own train split", threshold=SCREEN_THRESHOLD)
        entry["screen"] = {"status": "MEASURED", **rep.__dict__}
        ceiling = ceiling_from_predictions(template)
        entry["ceiling"] = ceiling if ceiling is not None else {"status": "NOT MEASURED",
                                                                "reason": "zero-shot dev predictions not present"}
        entry["screen_drop"] = rep.naive_bayes_macro is not None and rep.naive_bayes_macro >= SCREEN_THRESHOLD
        report["templates"][template] = entry
        print(template, "nb", rep.naive_bayes_macro, "sp", rep.string_presence_macro, "ceiling", entry["ceiling"],
              flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
