"""Y4 step 3 (ADVISORY Y4 §3; D49): the MIMIC audit suite v1, frozen before any Stage A run.

The suite is the late-anchor test items of the D46 structured-gold templates (Y2 test splits; test rows are late-anchor by
construction), minus any template that fails D35's screens on dev:
- Naive Bayes macro accuracy on dev at or above 0.90 (from outputs/osler_v4/Y2/counts*.json);
- zero-shot Qwen3.5-4B ceiling at or above 0.97 on dev (from the GPU ceiling screen, outputs/osler_v4/Y4/ceiling_qwen35_4b.json).
Excluded templates are listed with their reason and are never in the verdict. Not in the suite (additional rows, never in a
verdict): the held-out items (heart failure and AKI), the Y3 test pairs (mimic_doc_support_v1) and the MedNLI test split.

Refuses to write without the ceiling file: a suite frozen without the ceiling screen would not follow D35.
Writes the suite and its manifest under data/bench/osler_mimic_v1/ (items, gitignored), makes the suite read-only, and writes
outputs/osler_v4/Y4/mimic_audit_suite_v1_manifest.json (counts, digests and exclusions only).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from meddecide.panel import training_overlap as to
from meddecide.utils.io import write_json

ITEMS = Path("data/mimic_build/items")
SUITE_DIR = Path("data/bench/osler_mimic_v1")
SUITE = SUITE_DIR / "mimic_audit_suite_v1.jsonl"
HELDOUT = SUITE_DIR / "mimic_heldout_v1.jsonl"
MANIFEST_PRIVATE = SUITE_DIR / "manifest.json"
MANIFEST = Path("outputs/osler_v4/Y4/mimic_audit_suite_v1_manifest.json")
COUNTS = [Path("outputs/osler_v4/Y2/counts.json"), Path("outputs/osler_v4/Y2/counts_lab_med.json")]
CEILING = Path("outputs/osler_v4/Y4/ceiling_qwen35_4b.json")
MIMIC_TEMPLATES = (
    "mimic_icd_support_v1", "mimic_icd_specificity_v1", "mimic_principal_dx_v1", "mimic_msdrg_v1",
    "mimic_drg_cc_level_v1", "mimic_apr_severity_v1", "mimic_apr_mortality_v1", "mimic_pcs_support_v1",
    "mimic_rad_exam_v1", "mimic_ed_acuity_v1", "mimic_ed_disposition_v1", "mimic_lab_validation_v1",
    "mimic_med_given_v1",
)
NB_LIMIT = 0.90
CEILING_LIMIT = 0.97


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not CEILING.exists():
        raise SystemExit(f"{CEILING} is missing: the zero-shot ceiling screen has not run; the suite is not frozen")
    if SUITE.exists():
        raise SystemExit(f"{SUITE} exists: the suite is frozen; a new suite needs a new loop, not an edit (ADVISORY §3.7)")

    counts: dict[str, Any] = {}
    for path in COUNTS:
        counts.update(json.loads(path.read_text())["templates"])
    ceiling = json.loads(CEILING.read_text())

    included: list[dict[str, Any]] = []
    excluded: dict[str, str] = {}
    per_template: dict[str, dict[str, Any]] = {}
    for template in MIMIC_TEMPLATES:
        screens = counts[template]["screens"]
        nb = screens.get("nb_macro_accuracy_dev")
        entry = ceiling.get(template)
        if entry is None:
            raise SystemExit(f"ceiling for {template} is missing: the suite is not frozen")
        ceil = entry.get("macro_accuracy_dev")
        if nb is not None and nb >= NB_LIMIT:
            excluded[template] = f"Naive Bayes dev {nb} >= {NB_LIMIT} (D35 shortcut)"
            continue
        if ceil is None:
            excluded[template] = f"ceiling not measured: readout gate failed ({entry.get('readout_status')})"
            continue
        if ceil >= CEILING_LIMIT:
            excluded[template] = f"zero-shot Qwen3.5-4B dev macro {ceil} >= {CEILING_LIMIT} (D35 ceiling)"
            continue
        rows = [json.loads(line) for line in (ITEMS / f"{template}__test.jsonl").read_text().splitlines() if line.strip()]
        rows = [{**r, "suite": "mimic_audit_suite_v1"} for r in rows]
        included.extend(rows)
        per_template[template] = {"items": len(rows), "ceiling_dev": ceil, "nb_dev": nb}

    held = []
    for template in MIMIC_TEMPLATES:
        path = ITEMS / f"{template}__heldout.jsonl"
        if path.exists():
            held.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())

    # the suite and the held-out items must not occur in any training input (text or source record): checked at freeze time
    matches = to.match_training(included + held, to.training_files())
    if matches:
        raise SystemExit(f"{len(matches)} suite or held-out items occur in training input: the suite is not frozen")
    SUITE_DIR.mkdir(parents=True, exist_ok=True)
    included.sort(key=lambda r: (r["template_id"], r["item_id"]))
    with SUITE.open("w") as handle:
        for row in included:
            handle.write(json.dumps(row) + "\n")
    with HELDOUT.open("w") as handle:
        for row in held:
            handle.write(json.dumps(row) + "\n")
    for path in (SUITE, HELDOUT):
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    manifest_private = {"suite": str(SUITE), "items": len(included), "sha256": _digest(SUITE),
                        "per_template": per_template, "excluded": excluded,
                        "heldout": {"items": len(held), "sha256": _digest(HELDOUT)}}
    write_json(MANIFEST_PRIVATE, manifest_private)
    write_json(MANIFEST, {"suite_items": len(included), "suite_sha256": manifest_private["sha256"],
                          "training_overlap_suite_and_heldout": len(matches),
                          "training_files_checked": len(to.training_files()),
                          "items_by_template": {t: v["items"] for t, v in per_template.items()},
                          "excluded_templates": excluded,
                          "heldout_items_not_in_suite": len(held),
                          "heldout_sha256": manifest_private["heldout"]["sha256"],
                          "rules": {"nb_limit": NB_LIMIT, "ceiling_limit": CEILING_LIMIT}})
    print(json.dumps({"items": len(included), "excluded": excluded, "heldout": len(held)}, indent=2))


if __name__ == "__main__":
    main()
