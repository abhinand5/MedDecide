"""openFDA established-pharmacologic-class index for **near-miss** distractors.

``fda_class_choice_v1`` drew its distractors from an openFDA-wide pool of class names, so
almost any model could pick the gold class without reading the label. The ``_v2`` template
needs distractors that are *close* to the gold class: classes that share the gold class's
mechanism of action (``openfda.pharm_class_moa``), its physiologic effect
(``openfda.pharm_class_pe``), or at least a route of administration.

This module builds the index that makes that possible:

* :func:`fetch_class_index` pages through ``drug/label`` for every label carrying an
  established pharmacologic class and records, per class, the MoA / PE / route values seen
  on the labels that carry it.
* :func:`near_miss_candidates` ranks every other class by how strongly it relates to a gold
  class (``shared_moa`` > ``shared_pe`` > ``same_route``).
* :func:`save_index` / :func:`load_index` are a deterministic JSON round-trip.

Everything is sorted before it leaves a function: the same API pages give byte-identical
files, and the same index gives the same candidate order. Nothing is dropped silently — the
meta block carries the fetch counters, and a label that carries no class is counted in
``n_labels_without_class`` rather than ignored.

The raw openFDA class names keep their vocabulary tag (``"Biguanides [EPC]"``), because the
tag says which vocabulary the string came from and the near-miss rules compare raw values.
:func:`find_class` / :func:`strip_class_tag` accept the untagged spelling too, so a caller
may name a class either way.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypedDict

API = "https://api.fda.gov/drug/label.json"
LICENSE = "public-domain"  # openFDA: public domain (see https://open.fda.gov/license/)
QUERY = "_exists_:openfda.pharm_class_epc"
PAGE_SIZE = 100
MAX_LABELS = 6000
MAX_CANDIDATES = 50

#: Near-miss rules, strongest first. The tuple order *is* the ranking order.
RULES = ("shared_moa", "shared_pe", "same_route")
RULE_RANK = {rule: rank for rank, rule in enumerate(RULES)}


class ClassEntry(TypedDict):
    """Values seen on the labels carrying one established-pharmacologic-class name."""

    moa: list[str]
    pe: list[str]
    routes: list[str]
    n_labels: int


class IndexMeta(TypedDict):
    """Fetch counters for an index (every number is measured, none is estimated)."""

    n_labels_fetched: int
    n_labels_without_class: int
    n_classes: int
    pages: int
    query: str
    api_total: int


class ClassIndex(TypedDict):
    """``{"classes": {name: ClassEntry}, "meta": IndexMeta}``."""

    classes: dict[str, ClassEntry]
    meta: IndexMeta


Candidate = TypedDict(
    "Candidate",
    {"class": str, "rule": str, "shared": list[str], "n_labels": int},
)

_TAG_RE = re.compile(r"\s*\[[A-Za-z]{2,4}\]\s*$")


def strip_class_tag(name: str) -> str:
    """Drop a trailing openFDA vocabulary tag (``[EPC]``, ``[MoA]``, ...) from a name."""
    return _TAG_RE.sub("", str(name).strip()).strip()


def find_class(name: str, index: ClassIndex) -> str | None:
    """Resolve ``name`` to an index key: exact match first, then untagged/case-insensitive.

    ``"Biguanides"`` resolves to ``"Biguanides [EPC]"``. Returns ``None`` when the index
    holds no such class, so a caller can report the miss instead of silently building items
    with no distractor pool.
    """
    classes = index.get("classes") or {}
    if name in classes:
        return str(name)
    target = strip_class_tag(name).casefold()
    matches = sorted(key for key in classes if strip_class_tag(key).casefold() == target)
    return matches[0] if matches else None


def _strings(value: Any) -> list[str]:
    """Non-empty stripped strings from an openFDA field (list, scalar, or absent)."""
    if isinstance(value, list):
        items: list[Any] = value
    elif isinstance(value, str):
        items = [value]
    else:
        return []
    return [text for text in (str(item).strip() for item in items) if text]


def _accumulate(
    raw: dict[str, dict[str, Any]], label: dict[str, Any]
) -> bool:
    """Fold one label's class/MoA/PE/route values into ``raw``; ``False`` if it has no class.

    A label with several classes contributes to each of them. A class named twice on one
    label counts once for ``n_labels`` (that counter counts labels, not occurrences).
    """
    openfda = label.get("openfda") or {}
    names = _strings(openfda.get("pharm_class_epc"))
    if not names:
        return False
    moa = set(_strings(openfda.get("pharm_class_moa")))
    pe = set(_strings(openfda.get("pharm_class_pe")))
    routes = {route.upper() for route in _strings(openfda.get("route"))}
    for name in set(names):
        entry = raw.setdefault(name, {"moa": set(), "pe": set(), "routes": set(), "n_labels": 0})
        entry["moa"] |= moa
        entry["pe"] |= pe
        entry["routes"] |= routes
        entry["n_labels"] += 1
    return True


def _freeze(raw: dict[str, dict[str, Any]]) -> dict[str, ClassEntry]:
    """Turn the accumulating sets into sorted lists, keyed in sorted order."""
    return {
        name: {
            "moa": sorted(raw[name]["moa"]),
            "pe": sorted(raw[name]["pe"]),
            "routes": sorted(raw[name]["routes"]),
            "n_labels": int(raw[name]["n_labels"]),
        }
        for name in sorted(raw)
    }


def fetch_class_index(
    *,
    max_labels: int = MAX_LABELS,
    page_size: int = PAGE_SIZE,
    client: Any | None = None,
) -> ClassIndex:
    """Fetch the class index for the first ``max_labels`` labels that carry an EPC class.

    Pages with ``search``/``skip``/``limit``; a 404 is openFDA's "no matches" answer and ends
    the walk. ``api_total`` records what the API said the full corpus holds, so a capped fetch
    is visibly capped rather than silently partial.
    """
    import httpx

    if page_size < 1:
        raise ValueError(f"page_size must be >= 1, got {page_size}")
    own_client = client is None
    http = client or httpx.Client(timeout=120.0)
    raw: dict[str, dict[str, Any]] = {}
    n_labels = 0
    n_without_class = 0
    pages = 0
    api_total = 0
    try:
        skip = 0
        while n_labels < max_labels:
            limit = min(page_size, max_labels - n_labels)
            response = http.get(API, params={"search": QUERY, "limit": limit, "skip": skip})
            if response.status_code == 404:  # openFDA returns 404 when nothing matches
                break
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("results") or []
            if not batch:
                break
            if pages == 0:
                api_total = int(((payload.get("meta") or {}).get("results") or {}).get("total") or 0)
            pages += 1
            for label in batch:
                if not _accumulate(raw, label):
                    n_without_class += 1
            n_labels += len(batch)
            skip += len(batch)
            if api_total and skip >= api_total:
                break
    finally:
        if own_client:
            http.close()
    classes = _freeze(raw)
    return {
        "classes": classes,
        "meta": {
            "n_labels_fetched": n_labels,
            "n_labels_without_class": n_without_class,
            "n_classes": len(classes),
            "pages": pages,
            "query": QUERY,
            "api_total": api_total,
        },
    }


def near_miss_candidates(
    gold_class: str,
    index: ClassIndex,
    *,
    exclude: set[str] | None = None,
    max_candidates: int = MAX_CANDIDATES,
) -> list[Candidate]:
    """Classes related to ``gold_class``, strongest rule first, then class name.

    Each candidate appears once, carrying its strongest rule and the shared value(s). Every
    name in ``exclude`` is dropped before ranking, and ``gold_class`` is never returned — it
    is resolved through :func:`find_class` first, so naming it untagged (``"Biguanides"``
    against key ``"Biguanides [EPC]"``) still removes it. An unknown ``gold_class`` has no
    MoA/PE/route values to share, so the result is empty rather than arbitrary.
    """
    classes = index.get("classes") or {}
    gold_key = find_class(gold_class, index)
    gold = classes.get(gold_key) if gold_key is not None else None
    if gold is None:
        return []
    blocked = set(exclude or ()) | {str(gold_class), gold_key}
    gold_moa = set(gold["moa"])
    gold_pe = set(gold["pe"])
    gold_routes = set(gold["routes"])
    candidates: list[Candidate] = []
    for name in classes:
        if name in blocked:
            continue
        entry = classes[name]
        shared_moa = sorted(gold_moa & set(entry["moa"]))
        shared_pe = sorted(gold_pe & set(entry["pe"]))
        shared_routes = sorted(gold_routes & set(entry["routes"]))
        if shared_moa:
            rule, shared = RULES[0], shared_moa
        elif shared_pe:
            rule, shared = RULES[1], shared_pe
        elif shared_routes:
            rule, shared = RULES[2], shared_routes
        else:
            continue
        candidates.append(
            {"class": name, "rule": rule, "shared": shared, "n_labels": int(entry["n_labels"])}
        )
    candidates.sort(key=lambda c: (RULE_RANK[c["rule"]], c["class"]))
    return candidates[:max_candidates]


def rule_counts(candidates: list[Candidate]) -> dict[str, int]:
    """How many candidates carry each rule; every rule is present, zero when unused."""
    counts: dict[str, int] = dict.fromkeys(RULES, 0)
    for candidate in candidates:
        rule = str(candidate["rule"])
        counts[rule] = counts.get(rule, 0) + 1
    return counts


def save_index(index: ClassIndex, path: Path) -> Path:
    """Write ``index`` as sorted-key JSON (deterministic bytes); returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_index(path: Path) -> ClassIndex:
    """Read an index written by :func:`save_index`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
