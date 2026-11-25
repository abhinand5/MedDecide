"""Y4 ceiling input (ADVISORY Y4 §3; D35): the MIMIC dev items of the 13 Y2 templates, in one file for the zero-shot screen.

Every row of every template's dev shard is kept; the counts per template are written to the manifest (no silent drops).
Writes data/bench/osler_v4/mimic_dev_all.jsonl (item text: private, gitignored) and
outputs/osler_v4/Y4/mimic_dev_all_manifest.json (counts and digests only).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from meddecide.utils.io import write_json

ITEMS = Path("data/mimic_build/items")
OUT = Path("data/bench/osler_v4/mimic_dev_all.jsonl")
MANIFEST = Path("outputs/osler_v4/Y4/mimic_dev_all_manifest.json")
MIMIC_TEMPLATES = (
    "mimic_icd_support_v1", "mimic_icd_specificity_v1", "mimic_principal_dx_v1", "mimic_msdrg_v1",
    "mimic_drg_cc_level_v1", "mimic_apr_severity_v1", "mimic_apr_mortality_v1", "mimic_pcs_support_v1",
    "mimic_rad_exam_v1", "mimic_ed_acuity_v1", "mimic_ed_disposition_v1", "mimic_lab_validation_v1",
    "mimic_med_given_v1",
)


def main() -> None:
    per_template: dict[str, int] = {}
    ids: list[str] = []
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for template in MIMIC_TEMPLATES:
            path = ITEMS / f"{template}__dev.jsonl"
            count = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["split"] != "dev" or row["template_id"] != template:
                    raise SystemExit(f"{path}: row {row['item_id']} is not a dev row of {template}: stop")
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                ids.append(str(row["item_id"]))
                count += 1
            per_template[template] = count
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate item ids across the MIMIC dev shards: stop")
    write_json(MANIFEST, {
        "items": len(ids),
        "per_template": per_template,
        "item_id_digest_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        "file": str(OUT),
    })
    print(json.dumps({"items": len(ids), "per_template": per_template}, indent=2))


if __name__ == "__main__":
    main()
