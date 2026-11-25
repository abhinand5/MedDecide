"""V2: verify the curated ICD-10-CM families against the CMS tabular order file (osler_v1 ADVISORY V2).

Downloads the CMS "code descriptions in tabular order" zip for the fiscal year (FY2026 is the acceptance year; the
ADVISORY names it), unpacks it under data/cms/icd10cm/<year>/ (gitignored), and checks every curated code with
``meddecide.mr.icd.verify_codes``: each must exist and be billable. FY2027 (effective 2026-10-01) is checked the same way
as an additional row; it never decides V2.

Writes outputs/osler_v1/V2/verify_fy<year>.json with the file hashes, the row count and the verdict. Exits non-zero when
the FY2026 check fails (V2 is then BLOCKED; the curated table is never edited to pass).

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v2_icd_verify.py
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Any

import httpx

from meddecide.mr import icd
from meddecide.utils.hashing import file_sha256
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v1" / "V2"
ZIP_URL = "https://www.cms.gov/files/zip/{year}-code-descriptions-tabular-order.zip"
ACCEPTANCE_YEAR = "2026"
ADDITIONAL_YEARS = ("2027",)


def fetch_zip(year: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"icd10cm-tabular-order-{year}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        return zip_path
    url = ZIP_URL.format(year=year)
    tmp = zip_path.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=180.0) as response:
        response.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in response.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.rename(zip_path)
    return zip_path


def unpack(zip_path: Path, dest_dir: Path) -> list[str]:
    names: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            target = (dest_dir / member).resolve()
            if not target.is_relative_to(dest_dir.resolve()):
                raise ValueError(f"unsafe member name in {zip_path.name}: {member!r}")
        archive.extractall(dest_dir)
        names = archive.namelist()
    return names


def order_file_in(dest_dir: Path, year: str) -> Path:
    """The tabular order file for the year (``icd10cm_order_<year>.txt``). The addenda file that the same zip carries is
    not the order file and is never used for the verification."""
    hits = sorted(p for p in dest_dir.rglob(f"icd10cm_order_{year}.txt"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one icd10cm_order_{year}.txt under {dest_dir}, found {[p.name for p in hits]}")
    return hits[0]


def verify_year(year: str) -> dict[str, Any]:
    dest_dir = REPO / "data" / "cms" / "icd10cm" / year
    zip_path = fetch_zip(year, dest_dir)
    members = unpack(zip_path, dest_dir)
    order_path = order_file_in(dest_dir, year)
    rows = icd.load_order_file(order_path)
    verdict = icd.verify_codes(rows)
    curated = [code for fam in icd.FAMILIES for code in fam.codes]
    billable = {code: rows[code].billable for code in curated if code in rows}
    result = {
        "year": year,
        "written_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "source_url": ZIP_URL.format(year=year),
        "zip_file": str(zip_path.relative_to(REPO)),
        "zip_sha256": file_sha256(zip_path),
        "zip_members": len(members),
        "order_file": str(order_path.relative_to(REPO)),
        "order_file_sha256": file_sha256(order_path),
        "order_rows_parsed": len(rows),
        "curated_families": len(icd.FAMILIES),
        "curated_codes": len(curated),
        "missing": verdict["missing"],
        "not_billable": verdict["not_billable"],
        "passed": not verdict["missing"] and not verdict["not_billable"],
        "billable_by_code": billable,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / f"verify_fy{year}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    accepted = verify_year(ACCEPTANCE_YEAR)
    print(f"FY{ACCEPTANCE_YEAR} (acceptance): {accepted['curated_codes']} curated codes, "
          f"{len(accepted['missing'])} missing, {len(accepted['not_billable'])} not billable, "
          f"{accepted['order_rows_parsed']} order rows, passed={accepted['passed']}", flush=True)
    for year in ADDITIONAL_YEARS:
        extra = verify_year(year)
        print(f"FY{year} (additional, not a verdict): missing {extra['missing']}, not billable "
              f"{extra['not_billable']}, passed={extra['passed']}", flush=True)
    if not accepted["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
