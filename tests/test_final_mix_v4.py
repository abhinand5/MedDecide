"""osler_v4 final mix caps (operator ruling 2026-10-23, item 3): the family limit, the replay top-up and the proportional quota."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from meddecide.train.final_mix import (
    FAMILY_CAP,
    REPLAY_MIN,
    build_final_mix,
    family_limit,
    proportional_quota,
    topup_count,
)

BASE_ROWS = 830_076  # osler_v2 C1 (A0's mix), C123 and C140
BASE_REPLAY = 174_609


def test_proportional_quota_sums_to_target_and_never_exceeds_a_count() -> None:
    quota = proportional_quota({"a": 3, "b": 7}, 5)
    assert quota == {"a": 2, "b": 3}  # exact shares 1.5 and 3.5: the tie goes to the smaller template id
    assert sum(quota.values()) == 5
    assert all(quota[t] <= n for t, n in {"a": 3, "b": 7}.items())


def test_proportional_quota_extremes_keep_or_drop_everything() -> None:
    counts = {"a": 3, "b": 7}
    assert proportional_quota(counts, 10) == counts
    assert proportional_quota(counts, 0) == {"a": 0, "b": 0}
    with pytest.raises(ValueError):
        proportional_quota(counts, 11)


def test_topup_count_is_the_smallest_top_up_that_meets_the_replay_share() -> None:
    rows = BASE_ROWS + 509_807
    r = topup_count(rows, BASE_REPLAY)
    assert r == 116_710
    assert Fraction(BASE_REPLAY + r, rows + r) >= REPLAY_MIN
    assert Fraction(BASE_REPLAY + r - 1, rows + r - 1) < REPLAY_MIN


def test_topup_count_is_zero_when_the_share_already_holds() -> None:
    assert topup_count(10, 5) == 0


def test_family_limit_is_the_largest_family_within_the_cap() -> None:
    limit = family_limit(BASE_ROWS, BASE_REPLAY)
    assert limit == 509_807

    def share(family: int) -> Fraction:
        total = BASE_ROWS + family
        return Fraction(family, total + topup_count(total, BASE_REPLAY))

    assert share(limit) <= FAMILY_CAP
    assert share(limit + 1) > FAMILY_CAP


def _mix_file(path: Path, specs: list[tuple[str, str, str]]) -> Path:
    """Write synthetic mix rows (item_id, template_id, source) as JSONL."""
    with path.open("w", encoding="utf-8") as fh:
        for item, template, source in specs:
            fh.write(json.dumps({"item_id": item, "template_id": template, "source": source}) + "\n")
    return path


def test_build_final_mix_subsamples_the_family_and_tops_up_the_replay(tmp_path: Path) -> None:
    base = _mix_file(tmp_path / "base.jsonl", [(f"b{i}", f"base_{i % 20}", "snli" if i < 30 else "other") for i in range(100)])
    family = _mix_file(tmp_path / "family.jsonl", [(f"f{i}", f"fam_{i % 6}", "mimic_iv") for i in range(60)])
    broad = _mix_file(tmp_path / "broad.jsonl", [(f"r{i}", f"broad_{i % 30}", "broad") for i in range(300)])
    out = tmp_path / "out" / "final_mix.jsonl"
    report = build_final_mix(base, {"B1": [family]}, [], broad, out)
    assert report["base_rows"] == 100 and report["base_replay_rows"] == 30
    assert report["family_available"] == 60 and report["family_limit"] == 54
    assert report["family_kept"] == 54 and sum(report["family_quota"].values()) == 54
    assert report["topup_rows"] == 1
    assert report["rows"] == 155 and report["replay_rows"] == 31
    assert all(report["checks"].values())
    assert sum(1 for _ in out.open(encoding="utf-8")) == 155


def test_compose_matches_the_written_mix_and_is_deterministic(tmp_path: Path) -> None:
    from meddecide.train.final_mix import compose_final_mix, composition_sha256

    base = _mix_file(tmp_path / "base.jsonl", [(f"b{i}", f"base_{i % 20}", "snli" if i < 30 else "other") for i in range(100)])
    family = _mix_file(tmp_path / "family.jsonl", [(f"f{i}", f"fam_{i % 6}", "mimic_iv") for i in range(60)])
    broad = _mix_file(tmp_path / "broad.jsonl", [(f"r{i}", f"broad_{i % 30}", "broad") for i in range(300)])
    rows, report = compose_final_mix(base, {"B1": [family]}, [], broad)
    again, _ = compose_final_mix(base, {"B1": [family]}, [], broad)
    assert [r["item_id"] for r in rows] == [r["item_id"] for r in again]
    assert len(rows) == report["rows"] == 155
    written = tmp_path / "out" / "final_mix.jsonl"
    build_final_mix(base, {"B1": [family]}, [], broad, written)
    file_ids = [json.loads(line)["item_id"] for line in written.open(encoding="utf-8")]
    assert file_ids == [r["item_id"] for r in rows]
    assert report["composition_sha256"] == composition_sha256(file_ids)
