"""Y4 step 1 (ADVISORY Y4 §1): fast dev v4 = osler_v3's fast dev v3, minus the items it labels as train, plus up to 400 dev
items per new template (Y2 MIMIC and MedNLI, Y3 pairs). Fixed before any Stage A run.

Why the train-labelled items are removed: fast dev v3 holds 400 items with split "train" (codiesp_icd_support_v1, source
mr_codiesp, partition fastdev_v2). They are identical to items in data/train/osler_v1/block_B3.jsonl, a block osler_v1 did not
keep; no kept training mix holds them (checked by item id over all 30 training files). A dev proxy must hold dev items only, so
fast dev v4 drops them and records the deviation (STATE §7). The osler_v3 partitions themselves are not edited.

Writes data/bench/osler_v4/fastdev_v4.jsonl (gitignored; item text) and outputs/osler_v4/Y4/fastdev_v4_manifest.json (counts,
hashes and partitions only; committed as loops/osler_v4/fastdev_v4.md by the report step).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.mr import mimic_items as mi
from meddecide.panel import training_overlap as to
from meddecide.utils.io import write_json

FASTDEV_V3 = Path("data/bench/osler_v3/fastdev_v3.jsonl")
MIMIC_ITEMS = Path("data/mimic_build/items")
PAIRS_DEV = Path("data/mimic_build/pairs/build_off/rows_dev.jsonl")
OUT = Path("data/bench/osler_v4/fastdev_v4.jsonl")
MANIFEST = Path("outputs/osler_v4/Y4/fastdev_v4_manifest.json")
PER_TEMPLATE_NEW = 400          # ADVISORY Y4 §1: up to 400 dev items per new template
MIMIC_TEMPLATES = (
    "mimic_icd_support_v1", "mimic_icd_specificity_v1", "mimic_principal_dx_v1", "mimic_msdrg_v1",
    "mimic_drg_cc_level_v1", "mimic_apr_severity_v1", "mimic_apr_mortality_v1", "mimic_pcs_support_v1",
    "mimic_rad_exam_v1", "mimic_ed_acuity_v1", "mimic_ed_disposition_v1", "mimic_lab_validation_v1",
    "mimic_med_given_v1",
)


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _digest(rows: list[dict[str, Any]]) -> str:
    blob = hashlib.sha256()
    for row in rows:
        blob.update(row["item_id"].encode("utf-8"))
    return blob.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base_all = _read(FASTDEV_V3)
    base = [r for r in base_all if r["split"] == "dev"]
    removed_train = [r for r in base_all if r["split"] != "dev"]
    removed_by_template = Counter(r["template_id"] for r in removed_train)

    new_rows: list[dict[str, Any]] = []
    new_counts: dict[str, dict[str, int]] = {}
    sources: dict[str, list[dict[str, Any]]] = {}
    for template in MIMIC_TEMPLATES:
        path = MIMIC_ITEMS / f"{template}__dev.jsonl"
        sources[template] = _read(path)
    sources["mimic_doc_support_v1"] = [r for r in _read(PAIRS_DEV) if r["split"] == "dev"]
    sources["mednli_v1"] = _read(MIMIC_ITEMS / "mednli_v1__dev.jsonl")
    for template, rows in sources.items():
        ordered = mi.stable_order(rows, args.seed, f"fastdev_v4:{template}", lambda r: r["item_id"])
        kept = ordered[:PER_TEMPLATE_NEW]
        partition = "mednli" if template == "mednli_v1" else "mimic"
        for row in kept:
            new_rows.append({**row, "fastdev_partition": partition})
        new_counts[template] = {"available": len(rows), "kept": len(kept), "partition": partition}

    candidates = base + new_rows
    ids = [r["item_id"] for r in candidates]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate item ids in fast dev v4: stop (no silent drops)")
    # a dev proxy must not share text or source records with training input (every .jsonl under data/train/ except the
    # dev files): the osler mixes contain fast dev v2 records (pubmed, clinical trials, openFDA, MedMCQA), so matches are
    # removed and counted by partition, template and reason
    matches = to.match_training(candidates, to.training_files())
    final = [r for r in candidates if str(r["item_id"]) not in matches]
    removed_for_training: Counter[str] = Counter()
    for row in candidates:
        match = matches.get(str(row["item_id"]))
        if match is not None:
            reason = "text" if match.text and not match.record else "record" if match.record and not match.text else "text and record"
            removed_for_training[f"{row['fastdev_partition']} | {row['template_id']} | {reason}"] += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as handle:
        for row in final:
            handle.write(json.dumps(row) + "\n")
    partitions = Counter(r["fastdev_partition"] for r in final)
    manifest = {
        "fastdev_v3_rows": len(base_all),
        "fastdev_v3_dev_rows_kept": len(base),
        "fastdev_v3_train_labelled_rows_removed": len(removed_train),
        "fastdev_v3_train_labelled_removed_by_template": dict(removed_by_template),
        "new_templates": new_counts,
        "partitions": dict(partitions),
        "removed_for_training_overlap": dict(removed_for_training),
        "removed_for_training_overlap_total": sum(removed_for_training.values()),
        "training_files_checked": len(to.training_files()),
        "total_items": len(final),
        "item_id_digest_sha256": _digest(final),
        "per_template_new_cap": PER_TEMPLATE_NEW,
        "seed": args.seed,
    }
    write_json(MANIFEST, manifest)
    print(json.dumps({k: manifest[k] for k in ("fastdev_v3_dev_rows_kept", "fastdev_v3_train_labelled_rows_removed",
                                              "partitions", "total_items")}, indent=2))


if __name__ == "__main__":
    main()
