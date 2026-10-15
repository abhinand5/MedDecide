"""osler_v1 block runs (ADVISORY V8): the step budget, the dev partitions and the pre-registered keep rule.

Pure functions. The scripts read and write files; anything that decides a number lives here so it can be unit-tested. The
margins below are the pre-registered keep rule and are not changed after any result.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from meddecide.eval.o2_metrics import score_row
from meddecide.eval.paired import (
    Difference,
    paired_mean_difference_stratified,
    stratified_macro_difference,
)

REALISED_ITEM_TARGET = 80_000  # ADVISORY V8: the step count gives this many realised items on B0
EVAL_QUARTERS = 4  # evaluations at 25, 50, 75 and 100 % of the steps
MR_SOURCES = frozenset({"mr_v1", "mr_synthea", "mr_teacher", "mr_mtsamples", "mr_codiesp", "mr_pmc"})
KNOWLEDGE_SOURCES = frozenset({"medqa", "medmcqa"})
BOOTSTRAP_RESAMPLES = 1000  # ADVISORY V8: paired bootstrap within templates, 1,000 resamples, seed 0
BOOTSTRAP_SEED = 0
# keep rule: a block X is kept iff the lower bound of X minus the last kept block C is above each margin
KEEP_MARGINS = {"mr": 0.0, "general": -0.01, "knowledge": -0.02}
KNOWLEDGE_CHECK_MARGIN = -0.02  # B0's knowledge dev minus zero-shot Qwen3.5-4B on the same items

Pair = tuple[Mapping[str, Any], Mapping[str, Any]]


def steps_for_items(batch_sizes: Iterable[int], target_items: int, multiple: int = EVAL_QUARTERS) -> int:
    """The smallest step count, a multiple of ``multiple``, whose batches hold at least ``target_items`` items.

    ``batch_sizes`` is consumed lazily (the items of each batch, in the trainer's order). The default multiple of four keeps
    the block runs' evaluations at 25, 50, 75 and 100 % on whole steps; the final run uses eight (ADVISORY V9).
    """
    if target_items < 1:
        raise ValueError("target_items must be positive")
    if multiple < 1:
        raise ValueError("multiple must be positive")
    total = 0
    steps = 0
    for size in batch_sizes:
        steps += 1
        total += int(size)
        if total >= target_items and steps % multiple == 0:
            return steps
    raise ValueError(f"the batches hold {total} items after {steps} steps, fewer than the target {target_items}")


def correct_flag(row: Mapping[str, Any]) -> float:
    """1.0 for a correct prediction row (``correct`` is a boolean), 0.0 otherwise."""
    return 1.0 if row["correct"] else 0.0


def pair_by_item(rows_x: Sequence[Mapping[str, Any]], rows_c: Sequence[Mapping[str, Any]]) -> list[Pair]:
    """Two prediction sets over the same items, paired by ``item_id`` and in the order of ``rows_x``.

    Raises on a duplicate id or on a set of ids that differs between the two sides; nothing is dropped silently.
    """
    index_c = {row["item_id"]: row for row in rows_c}
    if len(index_c) != len(rows_c):
        raise ValueError("duplicate item ids on the comparison side")
    ids_x = [row["item_id"] for row in rows_x]
    if len(set(ids_x)) != len(ids_x):
        raise ValueError("duplicate item ids on the candidate side")
    if set(ids_x) != set(index_c):
        raise ValueError(f"the two sides cover different items: {len(set(ids_x) - set(index_c))} only in the candidate, "
                         f"{len(set(index_c) - set(ids_x))} only in the comparison")
    return [(row, index_c[row["item_id"]]) for row in rows_x]


def partition_pairs(pairs: Sequence[Pair]) -> dict[str, list[Pair]]:
    """The three dev parts of the keep rule, by the candidate's source.

    ``mr`` is the MR dev part; ``general`` is every other (mix v2) item; ``knowledge`` is the tier-1 items (MedQA and
    MedMCQA), which are also general items.
    """
    parts: dict[str, list[Pair]] = {"mr": [], "general": [], "knowledge": []}
    for x, c in pairs:
        if x["source"] in MR_SOURCES:
            parts["mr"].append((x, c))
            continue
        parts["general"].append((x, c))
        if x["source"] in KNOWLEDGE_SOURCES:
            parts["knowledge"].append((x, c))
    return parts


def _template_groups(pairs: Sequence[Pair]) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    groups_x: dict[str, list[float]] = {}
    groups_c: dict[str, list[float]] = {}
    for x, c in pairs:
        groups_x.setdefault(str(x["template_id"]), []).append(correct_flag(x))
        groups_c.setdefault(str(x["template_id"]), []).append(correct_flag(c))
    return groups_x, groups_c


def macro_difference(pairs: Sequence[Pair]) -> Difference:
    """Template macro accuracy, candidate minus comparison, with the bootstrap resampling within templates."""
    groups_x, groups_c = _template_groups(pairs)
    return stratified_macro_difference(groups_x, groups_c, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)


def micro_difference(pairs: Sequence[Pair]) -> Difference:
    """Item accuracy (micro), candidate minus comparison, with the bootstrap resampling within templates."""
    return paired_mean_difference_stratified(
        [correct_flag(x) for x, _ in pairs],
        [correct_flag(c) for _, c in pairs],
        [str(x["template_id"]) for x, _ in pairs],
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )


def keep_statistics(pairs: Sequence[Pair]) -> dict[str, Difference]:
    """The three differences of the keep rule (candidate minus last kept block): MR and general by template macro, knowledge
    by micro accuracy."""
    parts = partition_pairs(pairs)
    return {
        "mr": macro_difference(parts["mr"]),
        "general": macro_difference(parts["general"]),
        "knowledge": micro_difference(parts["knowledge"]),
    }


def keep_verdict(lower_bounds: Mapping[str, float]) -> dict[str, Any]:
    """Kept iff every lower bound is strictly above its margin (ADVISORY V8). Returns each check and the decision."""
    missing = set(KEEP_MARGINS) - set(lower_bounds)
    if missing:
        raise ValueError(f"no lower bound for {sorted(missing)}")
    checks = {part: float(lower_bounds[part]) > margin for part, margin in KEEP_MARGINS.items()}
    return {"margins": dict(KEEP_MARGINS), "lower_bounds": {p: float(lower_bounds[p]) for p in KEEP_MARGINS},
            "checks": checks, "kept": all(checks.values())}


def knowledge_check_passes(lower_bound: float) -> bool:
    """B0's knowledge check against zero-shot Qwen3.5-4B: the lower bound must be above the margin (ADVISORY V8)."""
    return float(lower_bound) > KNOWLEDGE_CHECK_MARGIN


def starting_block(b0_passes: bool | None, b0b_passes: bool | None) -> str | None:
    """The block C before any keep decision (ADVISORY V8): B0, unless B0 fails its knowledge check and B0b passes it.

    None while the check that decides it is still pending (B0 not measured, or B0 failed and B0b not measured).
    """
    if b0_passes is None:
        return None
    if b0_passes:
        return "B0"
    if b0b_passes is None:
        return None
    return "B0b" if b0b_passes else "B0"


def keep_comparison(x_rows: Sequence[Mapping[str, Any]], c_rows: Sequence[Mapping[str, Any]], *, candidate: str,
                    current: str) -> dict[str, Any]:
    """The keep rule for a candidate block against the last kept block, from their fast dev predictions (aggregates only)."""
    pairs = pair_by_item(x_rows, c_rows)
    parts = partition_pairs(pairs)
    stats = keep_statistics(pairs)
    return {
        "candidate": candidate, "current": current,
        "items": {p: len(parts[p]) for p in parts},
        "templates": {p: len({str(x["template_id"]) for x, _ in parts[p]}) for p in parts},
        "statistics": {p: s.as_dict() for p, s in stats.items()},
        "verdict": keep_verdict({p: s.lo for p, s in stats.items()}),
        "margins": dict(KEEP_MARGINS), "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
    }


def knowledge_comparison(block_rows: Sequence[Mapping[str, Any]], zero_shot_rows: Sequence[Mapping[str, Any]], *,
                         block: str, comparison: str, health: Mapping[str, str]) -> dict[str, Any]:
    """A block's knowledge items minus a zero-shot model on the same items (micro accuracy), after the D12 gate.

    ``zero_shot_rows`` are O2 prediction rows; ``health`` maps each zero-shot template to its readout-health status. A failing
    cell means no zero-shot accuracy is reported: the check is NOT MEASURED, and the failing cells are named. Skipped
    zero-shot items raise, so nothing is dropped silently.
    """
    knowledge = [r for r in block_rows if r["source"] in KNOWLEDGE_SOURCES]
    base: dict[str, Any] = {
        "block": block, "comparison": comparison, "items_block": len(knowledge), "items_zero_shot": len(zero_shot_rows),
        "zero_shot_readout_health": dict(sorted(health.items())), "margin": KNOWLEDGE_CHECK_MARGIN,
    }
    skipped = sorted({str(r.get("reason")) for r in zero_shot_rows if r["status"] == "skipped"})
    if skipped:
        raise ValueError(f"the zero-shot run skipped knowledge items ({', '.join(skipped)}): decide before comparing")
    failed = {t: s for t, s in health.items() if s.startswith("READOUT_FAIL")}
    if failed:
        named = "; ".join(f"{t}: {s}" for t, s in sorted(failed.items()))
        return {**base, "status": f"NOT MEASURED — D12 readout gate failed for the zero-shot cells: {named}"}
    zero_shot = [{"item_id": s.item_id, "template_id": s.template_id, "correct": s.correct}
                 for s in (score_row(r) for r in zero_shot_rows)]
    diff = micro_difference(pair_by_item(knowledge, zero_shot))
    return {**base, "status": "MEASURED", "statistic": diff.as_dict(), "passes": knowledge_check_passes(diff.lo)}
