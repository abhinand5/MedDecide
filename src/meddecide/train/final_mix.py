"""osler_v4 final mix (operator ruling, Stage B, 2026-10-23, item 3): the caps as pure functions.

The final mix is the base mix plus the added sources of the MIMIC-family blocks that enter it (and of any other kept block), with
three caps: the MIMIC-family share at most 35 % (every row of a family block counts, MedNLI included), the replay share at least
20 %, and no template above 8 %. The replay top-up is the smallest number of replay rows that meets the replay share, given the
rows already in the mix. A family above its cap is subsampled proportionally across templates.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from fractions import Fraction
from math import ceil, floor
from pathlib import Path
from typing import Any

from meddecide.utils.hashing import stable_hash

FAMILY_CAP = Fraction(35, 100)  # ruling item 3: the MIMIC-family share of the final mix
REPLAY_MIN = Fraction(1, 5)  # D36 and ADVISORY X8 / Y7: the replay share of every block
TEMPLATE_CAP = Fraction(8, 100)  # ADVISORY Y7: no template above 8 % of a block


def topup_count(rows: int, replay: int, minimum: Fraction = REPLAY_MIN) -> int:
    """The smallest R with (replay + R) / (rows + R) at least the minimum; 0 when the share already holds."""
    if Fraction(replay) >= minimum * rows:
        return 0
    return ceil((minimum * rows - replay) / (1 - minimum))


def family_limit(base_rows: int, base_replay: int) -> int:
    """The largest family row count F for which, with the replay top-up that F needs, the family share stays at or below the cap.

    base_rows counts every non-family row (the base mix and any non-family added source, replay rows included); base_replay counts
    the replay rows among them. With R the minimal top-up, F / (base_rows + F + R) <= FAMILY_CAP solves to
    F <= k (base_rows - base_replay) / ((1 - REPLAY_MIN) - k REPLAY_MIN), where k = FAMILY_CAP / (1 - FAMILY_CAP).
    """
    k = FAMILY_CAP / (1 - FAMILY_CAP)
    return floor(k * (base_rows - base_replay) / ((1 - REPLAY_MIN) - k * REPLAY_MIN))


def proportional_quota(counts: Mapping[str, int], target: int) -> dict[str, int]:
    """Per-template quotas that sum to target and never exceed a template's count (largest remainder; ties by template id)."""
    total = sum(counts.values())
    if not 0 <= target <= total:
        raise ValueError(f"target {target} is outside 0..{total}")
    if total == 0:
        return dict.fromkeys(counts, 0)
    exact = {t: Fraction(n * target, total) for t, n in counts.items()}
    quota = {t: floor(x) for t, x in exact.items()}
    short = target - sum(quota.values())
    for template in sorted(counts, key=lambda t: (-(exact[t] - quota[t]), t))[:short]:
        quota[template] += 1
    return quota


GENERAL_REPLAY = ("snli", "winogrande", "hellaswag", "commonsenseqa", "boolq", "qasc", "arc_easy", "arc_challenge")  # as in y7


def is_replay(row: Mapping[str, Any]) -> bool:
    """A general replay row: a general replay source, or a broad replay template (the osler_v3 rule)."""
    return str(row["source"]) in GENERAL_REPLAY or str(row["template_id"]).startswith("broad_")


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def scan_rows(path: Path) -> tuple[set[str], Counter[str], int]:
    """Item ids, template counts and replay rows of a JSONL file of mix rows, in one streamed pass."""
    ids: set[str] = set()
    templates: Counter[str] = Counter()
    replay = 0
    for row in iter_rows(path):
        ids.add(str(row["item_id"]))
        templates[str(row["template_id"])] += 1
        replay += is_replay(row)
    return ids, templates, replay


def build_final_mix(base: Path, family: Mapping[str, Sequence[Path]], other: Sequence[Path], broad: Path, out: Path,
                    salt: str = "final_mix") -> dict[str, Any]:
    """Write the final mix: the base rows, the other rows, the kept family rows and the replay top-up; check every cap first.

    family maps each entering MIMIC-family block to its added files; its rows count toward the family cap (MedNLI included). The
    family is subsampled proportionally across templates, by stable-hash rank within a template, when it exceeds the cap. The top-up
    is drawn from the broad pool in stable-hash order, base and added ids excluded. A breached cap raises before anything is written.
    """
    base_ids, base_templates, base_replay = scan_rows(base)
    base_rows = sum(base_templates.values())
    taken: set[str] = set(base_ids)
    other_templates: Counter[str] = Counter()
    other_replay = 0
    for path in other:
        ids, templates, replay = scan_rows(path)
        if ids & taken:
            raise ValueError(f"{path.name}: item ids repeat across the mix (ids are not printed)")
        taken |= ids
        other_templates += templates
        other_replay += replay
    other_rows = sum(other_templates.values())
    fam_pairs: list[tuple[str, str]] = []
    for paths in family.values():
        for path in paths:
            for row in iter_rows(path):
                item = str(row["item_id"])
                if item in taken:
                    raise ValueError(f"{path.name}: an item id repeats across the mix (ids are not printed)")
                taken.add(item)
                fam_pairs.append((str(row["template_id"]), item))
    fam_counts = Counter(template for template, _ in fam_pairs)
    target = min(len(fam_pairs), family_limit(base_rows + other_rows, base_replay + other_replay))
    quotas = proportional_quota(fam_counts, target)
    ranked: dict[str, list[str]] = defaultdict(list)
    for template, item in fam_pairs:
        ranked[template].append(item)
    kept: set[str] = set()
    for template, items in ranked.items():
        items.sort(key=lambda item: stable_hash({"salt": salt, "item": item}, length=16))
        kept.update(items[: quotas[template]])
    rows_before = base_rows + other_rows + target
    replay_before = base_replay + other_replay
    need = topup_count(rows_before, replay_before)
    pool = [row for row in iter_rows(broad) if str(row["item_id"]) not in taken]
    if need > len(pool):
        raise ValueError(f"the replay top-up needs {need} rows but the pool holds {len(pool)}")
    topup = sorted(pool, key=lambda row: stable_hash({"salt": salt, "item": str(row["item_id"])}, length=16))[:need]
    total = rows_before + need
    templates = base_templates + other_templates + Counter({t: q for t, q in quotas.items() if q})
    templates += Counter(str(row["template_id"]) for row in topup)
    replay_total = replay_before + sum(is_replay(row) for row in topup)
    largest = max(templates, key=lambda t: templates[t])
    checks = {
        "family_share_within_cap": target <= FAMILY_CAP * total,
        "replay_share_at_least_minimum": replay_total >= REPLAY_MIN * total,
        "largest_template_within_cap": templates[largest] <= TEMPLATE_CAP * total,
    }
    report: dict[str, Any] = {
        "rows": total, "base_rows": base_rows, "base_replay_rows": base_replay, "other_rows": other_rows,
        "family_available": len(fam_pairs), "family_limit": family_limit(base_rows + other_rows, base_replay + other_replay),
        "family_kept": target, "family_quota": dict(sorted(quotas.items())), "topup_rows": need, "topup_pool_rows": len(pool),
        "replay_rows": replay_total, "replay_share": replay_total / total, "family_share": target / total,
        "largest_template": largest, "largest_template_rows": templates[largest], "largest_template_share": templates[largest] / total,
        "templates": len(templates), "checks": checks,
    }
    if not all(checks.values()):
        raise ValueError(f"the final mix breaks a cap: {checks}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for path in (base, *other):
            with path.open(encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        fh.write(line if line.endswith("\n") else line + "\n")
        per_file: dict[str, int] = {}
        for paths in family.values():
            for path in paths:
                written = 0
                for row in iter_rows(path):
                    if str(row["item_id"]) in kept:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        written += 1
                per_file[path.name] = written
        for row in topup:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["family_kept_per_file"] = per_file
    return report
