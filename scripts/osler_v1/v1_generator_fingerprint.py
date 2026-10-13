"""Fingerprint of the MR generators' output for 300 patient ids each (osler_v1 V1 self-audit; R6).

Used to show that a refactor leaves the emitted items unchanged: the sha256 of the canonical JSON of every emitted row.
Prints the counts and hashes; writes nothing.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v1_generator_fingerprint.py
"""

from __future__ import annotations

import hashlib
import json

from meddecide.mr.generators import (
    mr_icd_support_template_v1,
    mr_indicator_evidence_v1,
    mr_page_category_tabular_v1,
)

GENERATORS = (("ind", mr_indicator_evidence_v1), ("icd", mr_icd_support_template_v1),
              ("page", mr_page_category_tabular_v1))


def fingerprint() -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for name, make in GENERATORS:
        rows = [item.as_dict() for k in range(300) for item in make(f"ref-{k:04d}")]
        blob = json.dumps(rows, sort_keys=True, ensure_ascii=False)
        out[name] = {"rows": len(rows), "sha256": hashlib.sha256(blob.encode()).hexdigest()}
    return out


def main() -> None:
    print(json.dumps(fingerprint(), indent=1))


if __name__ == "__main__":
    main()
