"""Manifest of a generator run: counts, keep rates and rejection reasons per family. Never item text.

The manifest is built from verdicts only. ``Verdict`` carries fact values in ``missing_facts`` and those are
deliberately not copied here: the manifest holds integers, ratios, reason codes and a run name.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from meddecide.synth.verify import Verdict

MANIFEST_SCHEMA = "meddecide.synth.manifest.v1"


@dataclass(frozen=True)
class FamilySummary:
    family: str
    generated: int
    kept: int
    rejected: int
    keep_rate: float | None
    rejected_by_primary_reason: dict[str, int] = field(default_factory=dict)
    rejected_by_any_reason: dict[str, int] = field(default_factory=dict)


def _keep_rate(kept: int, generated: int) -> float | None:
    return kept / generated if generated else None


def summarize(verdicts: Iterable[Verdict]) -> dict[str, FamilySummary]:
    """Group verdicts by family. Checks that primary-reason counts sum to the rejections (R5)."""
    groups: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        groups.setdefault(verdict.family, []).append(verdict)

    summaries: dict[str, FamilySummary] = {}
    for family in sorted(groups):
        items = groups[family]
        kept = sum(v.kept for v in items)
        rejected_items = [v for v in items if not v.kept]
        for v in rejected_items:
            if v.primary_reason is None:
                raise ValueError("rejected verdict without a reason")
        primary = Counter(v.primary_reason for v in rejected_items)
        any_reason = Counter(code for v in rejected_items for code in v.reasons)
        if sum(primary.values()) != len(rejected_items):
            raise ValueError("primary-reason counts do not sum to rejections")
        summaries[family] = FamilySummary(
            family=family,
            generated=len(items),
            kept=kept,
            rejected=len(rejected_items),
            keep_rate=_keep_rate(kept, len(items)),
            rejected_by_primary_reason=dict(sorted(primary.items())),
            rejected_by_any_reason=dict(sorted(any_reason.items())),
        )
    return summaries


def build_manifest(
    verdicts: Iterable[Verdict],
    *,
    run_name: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Manifest dict: schema, run name, config (caller supplies JSON-safe values without item text), totals, families."""
    items = list(verdicts)
    families = summarize(items)
    generated = sum(s.generated for s in families.values())
    kept = sum(s.kept for s in families.values())
    return {
        "schema": MANIFEST_SCHEMA,
        "run": run_name,
        "config": dict(config or {}),
        "item_text_included": False,
        "totals": {
            "generated": generated,
            "kept": kept,
            "rejected": generated - kept,
            "keep_rate": _keep_rate(kept, generated),
        },
        "families": {name: asdict(summary) for name, summary in families.items()},
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    """Write the manifest as sorted, indented JSON. Refuses any manifest that claims to include item text."""
    if manifest.get("item_text_included") is not False:
        raise ValueError("manifest must declare item_text_included=False")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path
