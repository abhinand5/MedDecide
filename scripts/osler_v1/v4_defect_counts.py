"""V4 defect counts (osler_v1 ADVISORY V4, step 3): how many verified teacher pages show each defect type found in the spot read.

Counts pages per defect type over every verified row of each generator. Aggregates only: no teacher text is printed or written.
Writes outputs/osler_v1/V4/defect_counts.json. The patterns are the ones the spot read identified; a count is a measurement of
the pattern, not a judgement of the page.

Usage:
    uv run --frozen python scripts/osler_v1/v4_defect_counts.py
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
TEACHER = REPO / "data" / "gen" / "osler_v1" / "teacher"
OUT = REPO / "outputs" / "osler_v1" / "V4" / "defect_counts.json"

# SNOMED CT semantic tags that the cleaning of V4-D3 does not remove from every fact (the spot read found one)
SNOMED_TAGS = (
    "disorder|finding|morphologic abnormality|procedure|situation|event|substance|product|organism|body structure|"
    "qualifier value|observable entity|record artifact|regime/therapy|specimen|assessment scale|physical object|"
    "staging scale|cell|social context|linkage concept|navigational concept|tumor staging|ethnic group|occupation|"
    "religion/philosophy|life style|person|geographic location|attribute"
)
PATTERNS: dict[str, re.Pattern[str]] = {
    "unrounded_decimal_value": re.compile(r"\d+\.\d{9,}"),
    "documented_baseline_phrase": re.compile(r"documented baseline", re.IGNORECASE),
    "prior_admission_phrase": re.compile(r"prior admission|previous admission", re.IGNORECASE),
    "bracketed_placeholder": re.compile(r"\[(?:Invented|Redacted|De-identified|Name|Date|Time|Signature|Physician|Clinician|"
                                        r"Hospital|Patient|Provider|Attending)[^\]]{0,40}\]", re.IGNORECASE),
    "tall_man_lettering": re.compile(r"\b[A-Z]{2,}[a-z]{3,}\b"),
    "residual_snomed_semantic_tag": re.compile(rf"\((?:{SNOMED_TAGS})\)", re.IGNORECASE),
    "outpatient_prescription_status_phrase": re.compile(r"administered as an outpatient prescription"),
    "synthetic_mrn_4471902": re.compile(r"\b4471902\b"),
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def gold_association(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Whether a pattern predicts the gold label (a shortcut check). Indicator: share of gold yes with and without the phrase
    and the synthetic MRN. Narrative: the largest difference between the class shares with and without the MRN."""
    if name == "indicator":
        out: dict[str, Any] = {}
        for key in ("documented_baseline_phrase", "synthetic_mrn_4471902"):
            hit = [r for r in rows if PATTERNS[key].search(str(r["state"]))]
            miss = [r for r in rows if not PATTERNS[key].search(str(r["state"]))]
            share = lambda group: round(sum(1 for r in group if r["gold"] == "yes") / len(group), 4) if group else None  # noqa: E731
            out[key] = {"pages_with": len(hit), "gold_yes_share_with": share(hit),
                        "pages_without": len(miss), "gold_yes_share_without": share(miss)}
        return out
    hit = [r for r in rows if PATTERNS["synthetic_mrn_4471902"].search(str(r["state"]))]
    miss = [r for r in rows if not PATTERNS["synthetic_mrn_4471902"].search(str(r["state"]))]
    classes = sorted({str(r["gold"]) for r in rows})
    def shares(group: list[dict[str, Any]]) -> dict[str, float]:
        counts = Counter(str(r["gold"]) for r in group)
        return {c: counts.get(c, 0) / len(group) for c in classes} if group else {}
    a, b = shares(hit), shares(miss)
    gap = max((abs(a.get(c, 0.0) - b.get(c, 0.0)) for c in classes), default=0.0)
    return {"synthetic_mrn_4471902": {"pages_with": len(hit), "pages_without": len(miss),
                                      "largest_class_share_gap": round(gap, 4)}}


def main() -> None:
    report: dict[str, Any] = {"written_utc": utcnow(), "git_commit": git_commit(REPO), "generators": {}}
    for name in ("indicator", "narrative"):
        rows = read_rows(TEACHER / f"{name}.jsonl")
        pages: Counter[str] = Counter()
        matches: Counter[str] = Counter()
        tags: Counter[str] = Counter()  # SNOMED semantic tag names only (public vocabulary words)
        tall_man: set[str] = set()  # distinct tokens are counted, never listed (drug names are RxNorm-derived strings)
        for row in rows:
            text = str(row["state"])
            for key, pattern in PATTERNS.items():
                found = pattern.findall(text)
                if found:
                    pages[key] += 1
                    matches[key] += len(found)
                if key == "residual_snomed_semantic_tag":
                    tags.update(f.strip("()").lower() for f in found)
                if key == "tall_man_lettering":
                    tall_man.update(found)
        digests = Counter(hashlib.sha256(str(row["state"]).encode("utf-8")).hexdigest() for row in rows)
        report["generators"][name] = {
            "rows": len(rows),
            "pages_with_pattern": dict(pages),
            "matches": dict(matches),
            "residual_snomed_tags_by_name": dict(tags.most_common()),
            "tall_man_distinct_tokens": len(tall_man),
            "gold_association": gold_association(rows, name),
            "distinct_state_texts": len(digests),
            "duplicated_state_texts": sum(1 for count in digests.values() if count > 1),
            "rows_in_duplicated_texts": sum(count for count in digests.values() if count > 1),
        }
    write_json(OUT, report)
    for name, entry in report["generators"].items():
        print(f"{name}: {entry['rows']} rows; distinct state texts {entry['distinct_state_texts']}; "
              f"duplicated texts {entry['duplicated_state_texts']} ({entry['rows_in_duplicated_texts']} rows)", flush=True)
        for key in PATTERNS:
            print(f"  {key}: {entry['pages_with_pattern'].get(key, 0)} pages, {entry['matches'].get(key, 0)} matches", flush=True)
    print(f"wrote {OUT.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
