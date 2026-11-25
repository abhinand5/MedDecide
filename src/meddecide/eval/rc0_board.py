"""RC0 unified leaderboard core (osler_rc0 R7).

Loads frozen evaluation sets and prediction files from any of the three on-disk schemas, normalises every row onto the
frozen item universe through the ``raw_id``/``recomp_id`` bijection (zero silent drops), and computes cell metrics,
paired differences, readout-health checks and Pareto frontiers.

The item-ID rule (verified 2026-11-24, ADVISORY §3): for every frozen file the raw ``row["item_id"]`` and the recomputed
``mix_row_to_item(row).item_id`` are disjoint and form a 1-to-1 bijection. A prediction file uses exactly one scheme;
:func:`load_cell` requires 100 % of its scored rows to map through one scheme and raises otherwise.

Schemas normalised here:
* **A** (`t10_score_d3.py`, `score_items.py`): ``gold_index`` + optional ``probs`` (option order = frozen order).
* **B** (`y4_pplx_flash.py`, `y4_meddecider.py`): ``gold_key`` / ``predicted_key`` (A..Z, yes/no, 1..n) + ``probs``.
* **C** (`Z10`, `Y9` JEV, `d1-2B`): ``status`` / ``gold`` / ``option_keys`` / ``probs``.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from meddecide.eval.metrics import ece_from_confidence
from meddecide.train.mix_items import mix_row_to_item

PROB_SUM_TOLERANCE = 1e-3
DEFAULT_MIN_MEDIAN_LABEL_MASS = 0.5
DEFAULT_MAX_MODAL_SHARE = 0.90
DEFAULT_DEGENERATE_MAJORITY_CEILING = 0.60


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _gold_index_from_key(key: str, option_keys: Sequence[str]) -> int:
    if key not in option_keys:
        raise ValueError(f"gold key {key!r} is not among the offered option keys {list(option_keys)[:8]}")
    return list(option_keys).index(key)


@dataclass(frozen=True)
class FrozenItem:
    """One item of a frozen evaluation set, with the two id schemes and the canonical fields."""

    index: int
    raw_id: str
    recomp_id: str
    source: str
    template_id: str
    qtype: str
    option_keys: tuple[str, ...]
    gold_index: int

    @property
    def n_options(self) -> int:
        return len(self.option_keys)


@dataclass
class FrozenSet:
    """A frozen item file plus the two id indexes used to normalise prediction rows."""

    name: str
    path: Path
    items: tuple[FrozenItem, ...]
    raw_index: dict[str, int]
    recomp_index: dict[str, int]

    def __len__(self) -> int:
        return len(self.items)

    def templates(self, index: np.ndarray | None = None) -> np.ndarray:
        positions = range(len(self.items)) if index is None else index
        return np.asarray([self.items[i].template_id for i in positions], dtype=object)

    def sources(self, index: np.ndarray | None = None) -> np.ndarray:
        positions = range(len(self.items)) if index is None else index
        return np.asarray([self.items[i].source for i in positions], dtype=object)


def load_frozen(name: str, path: Path) -> FrozenSet:
    """Read a frozen item file and verify the raw<->recomp bijection it must have."""
    rows = read_jsonl(path)
    items = []
    raw_index: dict[str, int] = {}
    recomp_index: dict[str, int] = {}
    for position, row in enumerate(rows):
        raw = str(row["item_id"])
        recomp = mix_row_to_item(row).item_id
        option_keys = tuple(str(option["key"]) for option in row["options"])
        if len(set(option_keys)) != len(option_keys):
            raise ValueError(f"{name}: duplicate option keys in item {raw}")
        gold = str(row["gold"])
        item = FrozenItem(index=position, raw_id=raw, recomp_id=recomp, source=str(row["source"]),
                          template_id=str(row["template_id"]), qtype=str(row["qtype"]),
                          option_keys=option_keys, gold_index=_gold_index_from_key(gold, option_keys))
        if raw in raw_index:
            raise ValueError(f"{name}: duplicate raw item id {raw}")
        if recomp in recomp_index:
            raise ValueError(f"{name}: duplicate recomputed item id {recomp}")
        raw_index[raw] = position
        recomp_index[recomp] = position
        items.append(item)
    overlap = set(raw_index) & set(recomp_index)
    if overlap:
        raise ValueError(f"{name}: {len(overlap)} ids occur in both the raw and the recomputed universes")
    if len(items) != len(set(raw_index.values())):
        raise ValueError(f"{name}: raw id universe is not the frozen file")
    return FrozenSet(name=name, path=path, items=tuple(items), raw_index=raw_index, recomp_index=recomp_index)


def _schema_of(row: Mapping[str, Any]) -> str:
    if "gold_index" in row:
        return "A"
    if "gold_key" in row or "predicted_key" in row:
        return "B"
    if "option_keys" in row and "gold" in row:
        return "C"
    raise ValueError(f"unknown prediction schema, keys: {sorted(row.keys())}")


@dataclass
class Cell:
    """One (model, set) prediction cell normalised onto the frozen item universe."""

    model: str
    set_name: str
    scheme: str
    schema: str
    n_universe: int
    index: np.ndarray
    predicted: np.ndarray
    correct: np.ndarray
    gold: np.ndarray
    probs: list[list[float]] | None
    label_mass: np.ndarray | None
    skipped: int
    path: str

    @property
    def scored(self) -> int:
        return len(self.index)

    @property
    def coverage(self) -> float:
        return self.scored / self.n_universe if self.n_universe else 0.0

    def position(self, frozen_index: int) -> int:
        hits = np.where(self.index == frozen_index)[0]
        if hits.size != 1:
            raise KeyError(f"frozen index {frozen_index} not in cell")
        return int(hits[0])


def load_cell(model: str, frozen: FrozenSet, path: Path) -> Cell | None:
    """Load one prediction file onto the frozen universe; ``None`` when the file does not exist."""
    if not path.is_file():
        return None
    rows = read_jsonl(path)
    if not rows:
        return None
    scored_rows = [row for row in rows if row.get("status", "scored") == "scored"]
    skipped = len(rows) - len(scored_rows)
    if not scored_rows:
        return None
    schema = _schema_of(scored_rows[0])
    ids = [str(row["item_id"]) for row in scored_rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{model}/{frozen.name}: duplicate item ids in {path}")
    raw_hits = sum(1 for item_id in ids if item_id in frozen.raw_index)
    recomp_hits = sum(1 for item_id in ids if item_id in frozen.recomp_index)
    if recomp_hits == len(ids) and raw_hits == 0:
        scheme = "recomp"
    elif raw_hits == len(ids) and recomp_hits == 0:
        scheme = "raw"
    else:
        raise ValueError(f"{model}/{frozen.name}: {len(ids)} rows map raw={raw_hits}, recomp={recomp_hits}; "
                         f"expected 100% through exactly one scheme")
    universe = frozen.raw_index if scheme == "raw" else frozen.recomp_index

    index = np.empty(len(scored_rows), dtype=np.int64)
    predicted = np.empty(len(scored_rows), dtype=np.int64)
    correct = np.empty(len(scored_rows), dtype=bool)
    gold_array = np.empty(len(scored_rows), dtype=np.int64)
    probs: list[list[float]] = []
    masses: list[float] = []
    masses_available = schema == "C"
    has_probs = True
    for position, row in enumerate(scored_rows):
        idx = universe[str(row["item_id"])]
        item = frozen.items[idx]
        gold_index = item.gold_index
        if schema == "A":
            declared = int(row["gold_index"])
            if not 0 <= declared < item.n_options:
                raise ValueError(f"{model}/{frozen.name}: item {row['item_id']} gold_index {declared} out of range")
            # the d3 path renders noul as (no, yes) whatever the row stores (d3_train.render_order), so its gold
            # index is the frozen one or its swap; every other qtype renders in the row's order. All reported
            # metrics (accuracy, Brier, ECE) are invariant to a per-item option permutation.
            allowed = {gold_index}
            if item.qtype == "noul" and item.option_keys == ("yes", "no"):
                allowed = {gold_index, 1 - gold_index}
            if declared not in allowed:
                raise ValueError(f"{model}/{frozen.name}: item {row['item_id']} gold_index {declared} is not a "
                                 f"permutation of the frozen gold index {gold_index}")
            gold_index = declared
            probs_row = row.get("probs")
            if isinstance(probs_row, list) and len(probs_row) == item.n_options:
                array = np.asarray(probs_row, dtype=np.float64)
                _check_probs(array, model, frozen.name, row["item_id"])
                predicted_index = int(array.argmax())
                if "predicted_index" in row and int(row["predicted_index"]) != predicted_index:
                    raise ValueError(f"{model}/{frozen.name}: argmax(probs) != predicted_index on {row['item_id']}")
                probs.append(_normalise(array).tolist())
            else:
                has_probs = False
                if "predicted_index" not in row:
                    raise ValueError(f"{model}/{frozen.name}: no probs and no predicted_index on {row['item_id']}")
                predicted_index = int(row["predicted_index"])
        elif schema == "B":
            key = str(row["gold_key"])
            if _gold_index_from_key(key, item.option_keys) != gold_index:
                raise ValueError(f"{model}/{frozen.name}: item {row['item_id']} gold_key {key!r} disagrees with frozen")
            if "predicted_key" not in row or "probs" not in row:
                raise ValueError(f"{model}/{frozen.name}: schema B row without predicted_key/probs on {row['item_id']}")
            array = np.asarray(row["probs"], dtype=np.float64)
            _check_probs(array, model, frozen.name, row["item_id"])
            predicted_index = _gold_index_from_key(str(row["predicted_key"]), item.option_keys)
            if int(array.argmax()) != predicted_index:
                raise ValueError(f"{model}/{frozen.name}: argmax(probs) != predicted_key on {row['item_id']}")
            probs.append(_normalise(array).tolist())
        else:  # schema C
            keys = tuple(str(k) for k in row["option_keys"])
            if keys != item.option_keys:
                raise ValueError(f"{model}/{frozen.name}: option_keys disagree with the frozen order on {row['item_id']}")
            if "probs" not in row or not isinstance(row["probs"], list):
                raise ValueError(f"{model}/{frozen.name}: schema C row without probs on {row['item_id']}")
            array = np.asarray(row["probs"], dtype=np.float64)
            _check_probs(array, model, frozen.name, row["item_id"])
            predicted_index = int(array.argmax())
            key_index = _gold_index_from_key(str(row["gold"]), keys)
            if key_index != gold_index:
                raise ValueError(f"{model}/{frozen.name}: item {row['item_id']} gold disagrees with frozen")
            probs.append(_normalise(array).tolist())
            extras = row.get("extras") or {}
            if "label_mass" in extras:
                masses.append(float(extras["label_mass"]))
            else:
                masses_available = False
        index[position] = idx
        predicted[position] = predicted_index
        gold_array[position] = gold_index
        correct[position] = predicted_index == gold_index
    if not has_probs:
        probs = None
        masses_available = False
    return Cell(model=model, set_name=frozen.name, scheme=scheme, schema=schema, n_universe=len(frozen),
                index=index, predicted=predicted, correct=correct, gold=gold_array,
                probs=probs if has_probs else None,
                label_mass=(np.asarray(masses, dtype=np.float64)
                            if masses_available and masses and has_probs else None),
                skipped=skipped, path=str(path))


def _check_probs(array: np.ndarray, model: str, set_name: str, item_id: str) -> None:
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{model}/{set_name}: item {item_id} probs are not a vector of at least 2 options")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{model}/{set_name}: item {item_id} has a non-finite probability")
    if np.any(array < 0):
        raise ValueError(f"{model}/{set_name}: item {item_id} has a negative probability")
    if abs(float(array.sum()) - 1.0) > PROB_SUM_TOLERANCE:
        raise ValueError(f"{model}/{set_name}: item {item_id} probs sum to {array.sum():.6f}")


def _normalise(array: np.ndarray) -> np.ndarray:
    return array / array.sum()


def bootstrap_ci(values: Sequence[float], *, n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def stratified_macro_ci(templates: np.ndarray, correct: np.ndarray, *,
                        n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI of template-macro accuracy, resampling items within each template."""
    groups = [np.where(templates == template)[0] for template in sorted(set(templates.tolist()))]
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        means = np.empty(len(groups), dtype=np.float64)
        for g, members in enumerate(groups):
            draws = rng.integers(0, members.size, size=members.size)
            means[g] = correct[members[draws]].mean()
        samples[b] = means.mean()
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def template_macro(templates: np.ndarray, correct: np.ndarray) -> float:
    groups: dict[str, list[float]] = defaultdict(list)
    for template, value in zip(templates.tolist(), correct.tolist(), strict=True):
        groups[str(template)].append(float(value))
    return float(np.mean([np.mean(values) for values in groups.values()]))


def per_template(frozen: FrozenSet, cell: Cell) -> list[dict[str, Any]]:
    """Per-template accuracy, majority baseline and constant-answer diagnostics."""
    out = []
    templates = frozen.templates(cell.index)
    predicted = cell.predicted
    for template in sorted(set(templates.tolist())):
        mask = templates == template
        n = int(mask.sum())
        gold = np.asarray([frozen.items[i].gold_index for i in cell.index[mask]], dtype=np.int64)
        majority_share = float(np.bincount(gold).max() / n) if n else 0.0
        modal_share = float(np.bincount(predicted[mask]).max() / n) if n else 0.0
        out.append({"template_id": str(template), "n": n, "accuracy": float(cell.correct[mask].mean()),
                    "majority_share": majority_share, "modal_share": modal_share,
                    "n_options": int(frozen.items[cell.index[mask][0]].n_options) if n else 0})
    return out


def readout_health(frozen: FrozenSet, cell: Cell, *, mode: str,
                   min_median_label_mass: float = DEFAULT_MIN_MEDIAN_LABEL_MASS,
                   max_modal_share: float = DEFAULT_MAX_MODAL_SHARE,
                   degenerate_majority_ceiling: float = DEFAULT_DEGENERATE_MAJORITY_CEILING) -> dict[str, Any]:
    """Readout-health status per template from the checks computable from prediction rows.

    ``mode`` is ``d12`` for a zero-shot letter readout (label mass is a gate), ``d21`` for a trained readout that
    normalises over the offered options (label mass is a diagnostic), or ``diagnostic`` (no gate).
    """
    results = []
    for entry in per_template(frozen, cell):
        mask = frozen.templates(cell.index) == entry["template_id"]
        failures: list[str] = []
        notes: list[str] = []
        if mode == "d12":
            if cell.label_mass is None:
                notes.append("label mass NOT MEASURED")
            else:
                median_mass = float(np.median(cell.label_mass[mask]))
                entry["median_label_mass"] = median_mass
                if median_mass < min_median_label_mass:
                    failures.append(f"median label mass {median_mass:.4f} < {min_median_label_mass}")
        else:
            notes.append("label mass is a diagnostic for a trained readout (D21)")
            if cell.label_mass is not None:
                entry["median_label_mass"] = float(np.median(cell.label_mass[mask]))
        lo, hi = bootstrap_ci(cell.correct[mask].astype(float), n_resamples=1000, seed=0)
        entry["accuracy_ci95"] = [lo, hi]
        chance = 1.0 / entry["n_options"] if entry["n_options"] else 0.0
        entry["chance"] = chance
        if hi < chance:
            failures.append(f"accuracy CI upper {hi:.4f} < chance {chance:.4f}")
        if entry["modal_share"] >= max_modal_share and entry["majority_share"] <= degenerate_majority_ceiling:
            failures.append(f"constant answer: modal share {entry['modal_share']:.3f} >= {max_modal_share} on a "
                            f"template whose majority share is {entry['majority_share']:.3f}")
        if failures:
            status = "READOUT_FAIL — " + "; ".join(failures)
        elif mode == "d21":
            status = "PASS (D21: label mass diagnostic only)"
        elif mode == "d12" and cell.label_mass is None:
            status = "NOT MEASURED — label mass unavailable"
        else:
            status = "PASS"
        results.append({"template_id": entry["template_id"], "n": entry["n"], "status": status,
                        "failures": failures, "notes": notes, **{k: v for k, v in entry.items()
                                                                if k not in ("template_id", "n")}})
    overall = "PASS"
    failing = [r for r in results if r["status"].startswith("READOUT_FAIL")]
    if failing:
        overall = f"READOUT_FAIL — {len(failing)} of {len(results)} templates"
    return {"overall": overall, "templates": results}


def cell_metrics(frozen: FrozenSet, cell: Cell, *, n_resamples: int = 2000, seed: int = 0,
                 health_mode: str = "diagnostic") -> dict[str, Any]:
    """Every metric the leaderboard reports for one (model, set) cell."""
    if cell.scored == 0:
        return {"status": "NOT MEASURED — no scored rows", "universe": cell.n_universe}
    templates = frozen.templates(cell.index)
    correct = cell.correct.astype(np.float64)
    micro = float(correct.mean())
    micro_ci = bootstrap_ci(correct, n_resamples=n_resamples, seed=seed)
    macro = template_macro(templates, correct)
    macro_ci = stratified_macro_ci(templates, correct, n_resamples=n_resamples, seed=seed)
    out: dict[str, Any] = {
        "status": "MEASURED",
        "path": cell.path,
        "scheme": cell.scheme,
        "schema": cell.schema,
        "universe": cell.n_universe,
        "scored": cell.scored,
        "skipped": cell.skipped,
        "coverage": round(cell.coverage, 6),
        "accuracy": round(micro, 6),
        "accuracy_ci95": [round(v, 6) for v in micro_ci],
        "macro_accuracy": round(macro, 6),
        "macro_accuracy_ci95": [round(v, 6) for v in macro_ci],
        "templates": len(set(templates.tolist())),
    }
    if cell.probs is None:
        out["brier"] = "NOT MEASURED — no probs in prediction file"
        out["ece"] = "NOT MEASURED — no probs in prediction file"
    else:
        gold = cell.gold.astype(np.int64, copy=False)
        # option counts differ inside a set (e.g. 3-option and 5-option score items), so the rows are ragged:
        # the Brier score is computed item by item and the top-label confidence is the row maximum.
        brier_items = np.empty(len(gold), dtype=np.float64)
        confidence = np.empty(len(gold), dtype=np.float64)
        for position, (probs_row, gold_index) in enumerate(zip(cell.probs, gold.tolist(), strict=True)):
            row = np.asarray(probs_row, dtype=np.float64)
            brier_items[position] = float(((row - np.eye(row.size)[gold_index]) ** 2).sum())
            confidence[position] = float(row.max())
        out["brier"] = round(float(brier_items.mean()), 6)
        out["brier_ci95"] = [round(v, 6) for v in bootstrap_ci(brier_items, n_resamples=n_resamples, seed=seed)]
        out["ece"] = round(float(ece_from_confidence(confidence, cell.correct, n_bins=15)), 6)
    out["readout_health"] = readout_health(frozen, cell, mode=health_mode)
    return out


def paired_difference(frozen: FrozenSet, a: Cell, b: Cell, *, n_resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Paired accuracy (and Brier, when both cells carry probs) difference ``a - b`` on the items both scored."""
    common = np.intersect1d(a.index, b.index)
    if common.size == 0:
        return {"n": 0, "status": "NOT MEASURED — no common items"}
    a_pos = {int(idx): i for i, idx in enumerate(a.index)}
    b_pos = {int(idx): i for i, idx in enumerate(b.index)}
    ia = np.asarray([a_pos[int(idx)] for idx in common], dtype=np.int64)
    ib = np.asarray([b_pos[int(idx)] for idx in common], dtype=np.int64)
    acc_diff = a.correct[ia].astype(np.float64) - b.correct[ib].astype(np.float64)
    acc_ci = bootstrap_ci(acc_diff, n_resamples=n_resamples, seed=seed)
    out: dict[str, Any] = {"n": int(common.size), "accuracy_diff": round(float(acc_diff.mean()), 6),
                           "accuracy_diff_ci95": [round(v, 6) for v in acc_ci]}
    if a.probs is not None and b.probs is not None:
        def item_brier(cell: Cell, positions: np.ndarray) -> np.ndarray:
            out = np.empty(len(positions), dtype=np.float64)
            gold = cell.gold[np.asarray(positions, dtype=np.int64)]
            for k, (p, gold_index) in enumerate(zip(positions, gold.tolist(), strict=True)):
                row = np.asarray(cell.probs[int(p)], dtype=np.float64)
                out[k] = float(((row - np.eye(row.size)[gold_index]) ** 2).sum())
            return out

        brier_diff = item_brier(a, ia) - item_brier(b, ib)
        brier_ci = bootstrap_ci(brier_diff, n_resamples=n_resamples, seed=seed)
        out["brier_diff"] = round(float(brier_diff.mean()), 6)
        out["brier_diff_ci95"] = [round(v, 6) for v in brier_ci]
    else:
        out["brier_diff"] = "NOT MEASURED — one cell lacks probs"
    return out


def restricted_to_n_options(cell: Cell, frozen: FrozenSet, max_options: int) -> Cell:
    """The same cell restricted to items with at most ``max_options`` offered options (paired MedDecider tables)."""
    keep = np.asarray([frozen.items[i].n_options <= max_options for i in cell.index], dtype=bool)
    return Cell(model=cell.model, set_name=cell.set_name, scheme=cell.scheme, schema=cell.schema,
                n_universe=sum(1 for item in frozen.items if item.n_options <= max_options),
                index=cell.index[keep], predicted=cell.predicted[keep], correct=cell.correct[keep],
                gold=cell.gold[keep],
                probs=[p for p, k in zip(cell.probs or [], keep.tolist(), strict=True) if k] if cell.probs is not None else None,
                label_mass=cell.label_mass[keep] if cell.label_mass is not None else None,
                skipped=cell.skipped, path=cell.path)


def exact_dominates(a: Mapping[str, float | None], b: Mapping[str, float | None], keys: Sequence[str]) -> bool:
    """True when ``a`` is at least as good as ``b`` on every measured key and strictly better on one."""
    comparable = [key for key in keys if a.get(key) is not None and b.get(key) is not None]
    if len(comparable) != len(keys):
        return False
    if not all(a[key] >= b[key] for key in keys):  # type: ignore[operator]
        return False
    return any(a[key] > b[key] for key in keys)  # type: ignore[operator]


def pareto_front(points: Mapping[str, tuple[float, float]]) -> list[str]:
    """Names not dominated on a 2-D maximisation problem (both axes higher is better)."""
    front = []
    for name, (x, y) in points.items():
        dominated = any((x2 >= x and y2 >= y) and (x2 > x or y2 > y)
                        for other, (x2, y2) in points.items() if other != name)
        if not dominated:
            front.append(name)
    return sorted(front)


def mean_of(values: Iterable[float | None]) -> float | None:
    present = [float(v) for v in values if isinstance(v, (int, float))]
    return float(np.mean(present)) if present else None


def fmt(value: Any, digits: int = 4) -> str:
    """Markdown cell: numbers to ``digits`` places, NOT MEASURED strings passed through verbatim."""
    if value is None:
        return "NOT MEASURED"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NOT MEASURED"
        return f"{value:.{digits}f}"
    return str(value)
