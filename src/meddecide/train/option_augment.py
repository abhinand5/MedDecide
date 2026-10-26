"""Option-count augmentation of closed-set choice rows (osler_v5 D52a).

A choice row whose template is on the closed list is rebuilt, with probability p, as its gold option plus distractors drawn from the
template's training label set: the count is uniform in [max(original, min_options), min(max_options, label-set size)], the distractors
are frequency-weighted (one draw per training row, so a label is drawn in proportion to its gold frequency), deduplicated by code and
by title and never equal to the gold, and the options are shuffled. A row whose count would fall below max(original, min_options) is
left unchanged. The gold follows its content, so the label is still the one the row's source field names.

The closed list is fixed by code, with one reason per template (the reasons also appear in loops/osler_v5/STATE.md §2). A template is
on the list only when its other offered labels cannot be true for the same record. Every other template is excluded.

Rows are augmented before they become training items, because the code and title of a label are the row's metadata and the item
type does not carry them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from meddecide.utils.hashing import stable_hash

MIN_OPTIONS = 8
MAX_OPTIONS = 64
KEY_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_RNG_TAG = 5252  # the augmentation's own stream in the seed vector (independent of the option-order streams)

# template id -> why it is on the list: its other offered labels cannot be true for the same record
CLOSED_LIST: dict[str, str] = {
    "mimic_principal_dx_v1": (
        "One principal diagnosis per admission (database sequence number 1 is unique). A distractor is another admission's "
        "principal code, deduplicated by code and by title, so it cannot be this admission's principal diagnosis."
    ),
    "mimic_msdrg_v1": (
        "One MS-DRG per admission. A distractor is another MS-DRG, deduplicated by DRG number and by title, so it cannot be this "
        "admission's DRG."
    ),
}

# named exclusions and why (ADVISORY Z0.3 and §7.1); every template that is not on the list is excluded
EXCLUDED_NAMED: dict[str, str] = {
    "mimic_rad_exam_v1": "Named by ADVISORY Z0.3. Other offered exam labels can describe the same study; not checked further.",
    "mimic_icd_support_v1": (
        "The offered codes are not exclusive answers by construction (a note can support more than one code); excluded "
        "conservatively (ADVISORY §7.1, when unsure, exclude)."
    ),
    "fda_route_choice_v1": (
        "Defect (ADVISORY Z0.3; checked in src/meddecide/bench/fresh/openfda.py): gold is routes[0] of a label that can list more "
        "than one route, so the other routes of that label are also true answers. Reported to the operator as a finding."
    ),
}
GENERIC_EXCLUSION = (
    "Not on the closed list: the offered labels are not a closed set of exclusive answers by construction, or that was not "
    "verified (excluded conservatively, ADVISORY §7.1)."
)


def validate_templates(templates: Iterable[str]) -> tuple[str, ...]:
    """The requested templates, checked against the closed list; an unknown name raises (nothing is augmented silently)."""
    names = tuple(templates)
    unknown = sorted(set(names) - set(CLOSED_LIST))
    if unknown:
        raise ValueError(f"templates not on the closed list: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError("a closed-list template is named twice")
    return names


def option_keys(n: int) -> list[str]:
    """Keys for n options in display order: A-Z, then the two-letter codes in alphabetical order (the readout's code names)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    names = list(KEY_ALPHABET[: min(n, 26)])
    for first in KEY_ALPHABET:
        for second in KEY_ALPHABET:
            if len(names) >= n:
                break
            names.append(first + second)
    return names[:n]


def gold_label(row: Mapping[str, Any]) -> str:
    gold = str(row["gold"])
    return next(str(o["label"]) for o in row["options"] if str(o["key"]) == gold)


def code_and_title(template_id: str, row: Mapping[str, Any], label: str) -> tuple[str, str]:
    """The identity of an offered label on a closed template: its code and its title (the label without a code prefix)."""
    meta = row.get("meta") or {}
    if template_id == "mimic_principal_dx_v1":
        code = str(meta["principal"])
        title = label[len(code):].strip() if label.startswith(code) else label.strip()
        return code, title
    if template_id == "mimic_msdrg_v1":
        return str(meta["drg"]), label.strip()
    raise ValueError(f"{template_id} is not on the closed list")


@dataclass(frozen=True)
class DistractorPool:
    """The training label set of one closed template: one (code, title, label) entry per training row, so a draw is frequency-weighted."""

    template_id: str
    entries: tuple[tuple[str, str, str], ...]

    @property
    def label_set_size(self) -> int:
        return len({(code, title) for code, title, _ in self.entries})


def build_pools(rows: Iterable[Mapping[str, Any]], templates: Sequence[str]) -> dict[str, DistractorPool]:
    """Distractor pools for the given closed templates, from training rows only (a dev or test row raises)."""
    names = validate_templates(templates)
    entries: dict[str, list[tuple[str, str, str]]] = {t: [] for t in names}
    for row in rows:
        template = str(row["template_id"])
        if template not in entries:
            continue
        if str(row["split"]) != "train":
            raise ValueError("distractor pools are built from training rows only")
        label = gold_label(row)
        code, title = code_and_title(template, row, label)
        entries[template].append((code, title, label))
    return {t: DistractorPool(template_id=t, entries=tuple(v)) for t, v in entries.items()}


def draw_entry(pool: DistractorPool, rng: np.random.Generator) -> tuple[str, str, str]:
    """One distractor draw: uniform over the pool's entries, one per training row, so a label is drawn in proportion to its gold
    frequency (the code, title and label of the entry)."""
    return pool.entries[int(rng.integers(len(pool.entries)))]


def augment_row(row: Mapping[str, Any], pool: DistractorPool, *, seed: int, probability: float,
                min_options: int = MIN_OPTIONS, max_options: int = MAX_OPTIONS) -> tuple[dict[str, Any], str]:
    """One row, rebuilt with probability ``probability``. Returns the row (a new dict when rebuilt) and the outcome:
    ``augmented``, ``not_selected`` (the draw said no) or ``fallback`` (the count would fall below max(original, min_options),
    or the pool cannot supply enough distinct distractors)."""
    template = str(row["template_id"])
    if template != pool.template_id:
        raise ValueError(f"row template {template} does not match the pool {pool.template_id}")
    if str(row["qtype"]) != "choice":
        raise ValueError("option-count augmentation applies to choice rows only")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    rng = np.random.default_rng([int(seed), _RNG_TAG, int(stable_hash({"item": str(row["item_id"])}, length=8), 16)])
    if rng.random() >= probability:
        return dict(row), "not_selected"
    original = len(row["options"])
    floor = max(original, min_options)
    ceiling = min(max_options, pool.label_set_size)
    if ceiling < floor:
        return dict(row), "fallback"
    count = int(rng.integers(floor, ceiling + 1))
    gold_text = gold_label(row)
    gold_code, gold_title = code_and_title(template, row, gold_text)
    used_codes, used_titles = {gold_code}, {gold_title}
    distractors: list[tuple[str, str, str]] = []
    attempts, max_attempts = 0, 200 * count + 1000
    while len(distractors) < count - 1:
        if attempts >= max_attempts:
            return dict(row), "fallback"
        attempts += 1
        code, title, label = draw_entry(pool, rng)
        if code in used_codes or title in used_titles:
            continue
        used_codes.add(code)
        used_titles.add(title)
        distractors.append((code, title, label))
    # the gold first, then the drawn distractors; the shuffle gives the display order
    sources = [(gold_code, gold_title, gold_text), *distractors]
    order = [int(i) for i in rng.permutation(count)]
    keys = option_keys(count)
    options = [{"key": keys[position], "label": sources[source][2], "description": None}
               for position, source in enumerate(order)]
    gold_position = order.index(0)
    # the identities of the offered options in display order, kept in memory for the audit (training never reads meta)
    offered = [[sources[source][0], sources[source][1]] for source in order]
    meta = {**dict(row.get("meta") or {}), "offered": offered}
    return {**dict(row), "options": options, "gold": keys[gold_position], "meta": meta}, "augmented"


@dataclass(frozen=True)
class AugmentReport:
    """Counts of what the augmentation did, by template. Every pooled row is counted once (no silent drops)."""

    seen: Counter = field(default_factory=Counter)
    augmented: Counter = field(default_factory=Counter)
    not_selected: Counter = field(default_factory=Counter)
    fallback: Counter = field(default_factory=Counter)
    added_options: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {
            name: {t: int(c) for t, c in sorted(getattr(self, name).items())}
            for name in ("seen", "augmented", "not_selected", "fallback", "added_options")
        }


def augment_rows(rows: Iterable[Mapping[str, Any]], pools: Mapping[str, DistractorPool], *, seed: int,
                 probability: float) -> tuple[list[dict[str, Any]], AugmentReport]:
    """Every row in order: pooled rows are augmented, the rest pass through unchanged. Returns the rows and the counts."""
    report = AugmentReport()
    out: list[dict[str, Any]] = []
    for row in rows:
        template = str(row["template_id"])
        pool = pools.get(template)
        if pool is None:
            out.append(dict(row))
            continue
        new_row, outcome = augment_row(row, pool, seed=seed, probability=probability)
        report.seen[template] += 1
        if outcome == "augmented":
            report.augmented[template] += 1
            report.added_options[template] += len(new_row["options"]) - len(row["options"])
        elif outcome == "not_selected":
            report.not_selected[template] += 1
        else:
            report.fallback[template] += 1
        out.append(new_row)
    return out, report


def expected_two_letter_golds(n_eligible: int, probability: float, *, min_options: int = MIN_OPTIONS,
                              max_options: int = MAX_OPTIONS, label_set_size: int | None = None) -> dict[int, float]:
    """Expected training items whose gold lands on option position r (0-based), for every two-letter position r in 26..max-1, from
    n_eligible eligible rows at probability p. The count is uniform over its range; the gold is uniform over the offered positions."""
    ceiling = max_options if label_set_size is None else min(max_options, label_set_size)
    floor = min_options
    counts = range(floor, ceiling + 1)
    width = len(counts)
    out: dict[int, float] = {}
    for r in range(26, max_options):
        p_gold = sum(1.0 / c for c in counts if c > r) / width
        out[r] = n_eligible * probability * p_gold
    return out
