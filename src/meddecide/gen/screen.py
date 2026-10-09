"""Screen for the generators (osler_v0 O3 acceptance): gold-in-state check and shortcut baselines.

A generator passes when, on its dev split:
  - no gold label occurs verbatim in the patient-specific part of the state (labels of two or more characters);
  - the string-presence baseline has macro accuracy below 0.90;
  - the bag-of-words Naive Bayes baseline has macro accuracy below 0.90 (trained on seen train items only);
  - minimal-pair twins defeat single-token rules: the share of pairs where the Naive Bayes baseline gets both
    twins right is reported.
Held-out generators are screened too, but their baseline is trained on the seen generators of the same family
(never on their own items), and a failure is reported, not repaired (their definition is fixed).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from meddecide.utils.hashing import normalize_text

FIXED_BLOCKS = ("Code list:", "Triage policy:", "Trial criterion:")
TOKEN = re.compile(r"[a-z0-9]+")


def record_text(state: str) -> str:
    """The patient-specific part of a state: the note or presentation, without the fixed instruction blocks."""
    kept = []
    for block in state.split("\n\n"):
        if block.startswith(FIXED_BLOCKS):
            continue
        kept.append(block)
    return normalize_text("\n".join(kept)).lower()


def gold_label(item: Mapping[str, Any]) -> str:
    return next(o["label"] for o in item["options"] if o["key"] == item["gold"])


def gold_in_state(item: Mapping[str, Any]) -> bool:
    """Not applicable to yes/no items: "yes" and "no" occur in ordinary text, so the check is only meaningful
    for phrase and code labels (choice and score items)."""
    if item["qtype"] == "noul":
        return False
    label = normalize_text(gold_label(item)).lower()
    if len(label) < 2:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(label) + r"(?![a-z0-9])", record_text(item["state"])) is not None


def string_presence_predict(item: Mapping[str, Any], majority: str) -> str:
    """Predict the one option whose label occurs in the record text; the training majority label otherwise.
    Applied to choice and score items only (see gold_in_state)."""
    text = record_text(item["state"])
    hits = [o["label"] for o in item["options"]
            if re.search(r"(?<![a-z0-9])" + re.escape(normalize_text(o["label"]).lower()) + r"(?![a-z0-9])", text)]
    if len(hits) == 1:
        return hits[0]
    return majority if majority in [o["label"] for o in item["options"]] else item["options"][0]["label"]


class NaiveBayes:
    """Multinomial Naive Bayes over tokens of the record text and the question, classes = gold label text."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_counts: Counter[str] = Counter()
        self.word_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def _tokens(self, item: Mapping[str, Any]) -> list[str]:
        return TOKEN.findall(record_text(item["state"]) + " " + normalize_text(item["question"]).lower())

    def fit(self, items: Iterable[Mapping[str, Any]]) -> NaiveBayes:
        for it in items:
            label = normalize_text(gold_label(it)).lower()
            self.class_counts[label] += 1
            toks = self._tokens(it)
            self.word_counts[label].update(toks)
            self.vocab.update(toks)
        return self

    def predict(self, item: Mapping[str, Any]) -> str:
        toks = self._tokens(item)
        total = sum(self.class_counts.values())
        best, best_score = None, -math.inf
        offered = [normalize_text(o["label"]).lower() for o in item["options"]]
        for label in offered:
            if label not in self.class_counts:
                score = -math.inf
            else:
                counts = self.word_counts[label]
                denom = sum(counts.values()) + self.alpha * len(self.vocab)
                score = math.log(self.class_counts[label] / total)
                score += sum(math.log((counts[t] + self.alpha) / denom) for t in toks)
            if score > best_score:
                best, best_score = label, score
        if best is None:
            return item["options"][0]["label"]
        return next(o["label"] for o in item["options"] if normalize_text(o["label"]).lower() == best)


def macro_accuracy(pairs: Sequence[tuple[str, str]]) -> float | None:
    """Mean over gold classes of per-class recall (the project's macro, as in metrics.macro_accuracy)."""
    if not pairs:
        return None
    by_gold: dict[str, list[bool]] = defaultdict(list)
    for gold, pred in pairs:
        by_gold[gold].append(gold == pred)
    return sum(sum(v) / len(v) for v in by_gold.values()) / len(by_gold)


@dataclass
class ScreenReport:
    generator: str
    held_out: bool
    n_dev: int
    gold_in_state_hits: int
    string_presence_macro: float | None
    naive_bayes_macro: float | None
    naive_bayes_source: str
    twin_pairs: int
    twin_both_correct_share: float | None
    passes: bool
    failures: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def screen_generator(name: str, held_out: bool, dev: Sequence[Mapping[str, Any]], train_for_nb: Sequence[Mapping[str, Any]],
                     nb_source: str, threshold: float = 0.90) -> ScreenReport:
    gold_hits = sum(1 for it in dev if gold_in_state(it))
    majority = Counter(gold_label(it) for it in train_for_nb).most_common(1)[0][0] if train_for_nb else ""
    phrase_dev = [it for it in dev if it["qtype"] != "noul"]
    sp_pairs = [(gold_label(it), string_presence_predict(it, majority)) for it in phrase_dev]
    sp = macro_accuracy(sp_pairs)
    nb = NaiveBayes().fit(train_for_nb)
    nb_pairs = [(gold_label(it), nb.predict(it)) for it in dev]
    nbm = macro_accuracy(nb_pairs)
    # twins: fraction of pairs where the NB baseline is right on both twins
    by_pair: dict[str, list[bool]] = defaultdict(list)
    for it, (g, p) in zip(dev, nb_pairs, strict=True):
        by_pair[it["pair_id"]].append(g == p)
    pairs_full = [v for v in by_pair.values() if len(v) == 2]
    both = (sum(1 for v in pairs_full if all(v)) / len(pairs_full)) if pairs_full else None
    failures = []
    if gold_hits:
        failures.append(f"gold label appears in the record text for {gold_hits} dev items")
    if sp is not None and sp >= threshold:
        failures.append(f"string-presence baseline macro {sp:.3f} >= {threshold}")
    if nbm is not None and nbm >= threshold:
        failures.append(f"Naive Bayes baseline macro {nbm:.3f} >= {threshold}")
    return ScreenReport(
        generator=name, held_out=held_out, n_dev=len(dev), gold_in_state_hits=gold_hits,
        string_presence_macro=None if sp is None else round(sp, 4),
        naive_bayes_macro=None if nbm is None else round(nbm, 4), naive_bayes_source=nb_source,
        twin_pairs=len(pairs_full), twin_both_correct_share=None if both is None else round(both, 4),
        passes=not failures, failures=failures,
    )
