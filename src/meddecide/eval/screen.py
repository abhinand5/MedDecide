"""Fresh-template screen: regex and bag-of-words baselines (task T8).

A template that a hand-written regex or a bag-of-words classifier can already answer is
measuring surface pattern-matching, not medicine. The screen trains on the template's **dev**
split and scores on **test** (never the other way round), and drops a template when either
baseline reaches ``screen.regex_drop_threshold`` / ``screen.bow_drop_threshold`` from
``configs/bench_v0.yaml`` (0.95).

Patterns live in ``configs/template_screen_patterns.yaml`` so they are dated and reviewable
rather than buried in code.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from meddecide.bench.schema import Item, QuestionType


@dataclass
class TemplateScreenResult:
    """Everything measured about one template."""

    template_id: str
    source: str
    qtype: str
    n_dev: int
    n_test: int
    majority_label: str
    majority_test_accuracy: float
    regex_train_accuracy: float | None
    regex_test_accuracy: float | None
    regex_test_macro_accuracy: float | None
    regex_patterns: int
    bow_test_accuracy: float | None
    bow_test_macro_accuracy: float | None
    bow_test_majority: float | None
    bow_dev_accuracy: float | None
    n_test_classes: int
    single_class_test_split: bool
    drop: bool
    drop_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "source": self.source,
            "qtype": self.qtype,
            "n_dev": self.n_dev,
            "n_test": self.n_test,
            "majority_label": self.majority_label,
            "majority_test_accuracy": self.majority_test_accuracy,
            "regex_train_accuracy": self.regex_train_accuracy,
            "regex_test_accuracy": self.regex_test_accuracy,
            "regex_test_macro_accuracy": self.regex_test_macro_accuracy,
            "n_regex_patterns": self.regex_patterns,
            "bow_test_accuracy": self.bow_test_accuracy,
            "bow_test_macro_accuracy": self.bow_test_macro_accuracy,
            "bow_dev_accuracy": self.bow_dev_accuracy,
            "n_test_classes": self.n_test_classes,
            "single_class_test_split": self.single_class_test_split,
            "bow_test_majority": self.bow_test_majority,
            "drop": self.drop,
            "drop_reasons": self.drop_reasons,
            "notes": self.notes,
        }


def majority_baseline(items: Sequence[Item]) -> tuple[str, float]:
    """Most frequent gold key and its share."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item.gold] = counts.get(item.gold, 0) + 1
    label = max(sorted(counts), key=lambda k: counts[k])
    return label, counts[label] / len(items)


def regex_predict(item: Item, patterns: dict[str, list[str]]) -> str | None:
    """First matching pattern wins; patterns are checked in the order the config lists them.

    A pattern maps a gold option *key* to a list of regular expressions. Case-insensitive.
    Returns ``None`` when nothing matches, which counts as a wrong (or abstained) answer.
    """
    text = f"{item.state}\n{item.question}"
    for key, regexes in patterns.items():
        for pattern in regexes:
            if re.search(pattern, text, re.IGNORECASE):
                return key
    return None


def score_regex(
    items: Sequence[Item], patterns: dict[str, list[str]]
) -> dict[str, Any]:
    """Accuracy and macro accuracy of the regex baseline, with the counts that produce them.

    Macro accuracy (unweighted mean per-class recall) is reported beside micro accuracy because
    several templates are heavily imbalanced: on `pubmed_observational_noul_v1` the majority
    class is 98.5 % of the test split, so micro accuracy cannot distinguish a real baseline from
    "always say no".
    """
    from meddecide.eval.metrics import macro_accuracy

    if not items:
        return {"n": 0, "n_correct": 0, "accuracy": None, "macro_accuracy": None, "n_matched": 0}
    predictions = [regex_predict(item, patterns) or "" for item in items]
    gold = [item.gold for item in items]
    correct = sum(1 for p, g in zip(predictions, gold, strict=True) if p == g)
    matched = sum(1 for p in predictions if p)
    return {
        "n": len(items),
        "n_correct": correct,
        "accuracy": correct / len(items),
        # a baseline that never predicts a label has no macro accuracy: "0.0" would imply it
        # predicted and was wrong on every class
        "macro_accuracy": macro_accuracy(predictions, gold) if matched else None,
        "n_matched": matched,
        "match_rate": matched / len(items),
    }


def bow_predict(
    train_items: Sequence[Item],
    test_items: Sequence[Item],
    *,
    max_features: int = 20000,
    seed: int = 0,
) -> dict[str, Any]:
    """TF-IDF + logistic regression trained on dev, scored on test.

    Deterministic (fixed seed, no shuffling), and refuses to run when the dev split has fewer
    than two classes — that would be a degenerate fit, not a baseline.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    if len(train_items) < 4 or len({i.gold for i in train_items}) < 2:
        return {
            "n_train": len(train_items),
            "n_test": len(test_items),
            "test_accuracy": None,
            "test_macro_accuracy": None,
            "dev_accuracy": None,
            "test_majority": None,
            "note": "dev split too small or single-class; bag-of-words baseline not fitted",
        }
    texts_train = [f"{i.state}\n{i.question}" for i in train_items]
    texts_test = [f"{i.state}\n{i.question}" for i in test_items]
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1,
                                 sublinear_tf=True)
    x_train = vectorizer.fit_transform(texts_train)
    x_test = vectorizer.transform(texts_test)
    labels = [i.gold for i in train_items]
    classifier = LogisticRegression(max_iter=2000, random_state=seed, n_jobs=1)
    classifier.fit(x_train, labels)
    dev_predictions = classifier.predict(x_train)
    test_predictions = classifier.predict(x_test)
    dev_accuracy = float(np.mean([p == g for p, g in zip(dev_predictions, labels, strict=True)]))
    test_accuracy = float(
        np.mean([p == i.gold for p, i in zip(test_predictions, test_items, strict=True)])
    )
    from meddecide.eval.metrics import macro_accuracy

    majority_label, majority_share = majority_baseline(test_items)
    return {
        "n_train": len(train_items),
        "n_test": len(test_items),
        "n_features": int(x_train.shape[1]),
        "dev_accuracy": dev_accuracy,
        "test_accuracy": test_accuracy,
        "test_macro_accuracy": macro_accuracy(
            list(test_predictions), [i.gold for i in test_items]
        ),
        "test_majority": majority_share,
        "majority_label": majority_label,
        "classes": sorted(set(labels)),
    }


def screen_template(
    template_id: str,
    dev_items: Sequence[Item],
    test_items: Sequence[Item],
    *,
    patterns: dict[str, list[str]],
    regex_threshold: float = 0.95,
    bow_threshold: float = 0.95,
    seed: int = 0,
) -> TemplateScreenResult:
    """Run all three baselines for one template and decide whether to drop it."""
    if not test_items:
        return TemplateScreenResult(
            template_id=template_id,
            source=dev_items[0].source if dev_items else "unknown",
            qtype=str(dev_items[0].qtype) if dev_items else "unknown",
            n_dev=len(dev_items),
            n_test=0,
            majority_label="",
            majority_test_accuracy=0.0,
            regex_train_accuracy=None,
            regex_test_accuracy=None,
            regex_test_macro_accuracy=None,
            regex_patterns=len(patterns),
            bow_test_accuracy=None,
            bow_test_macro_accuracy=None,
            bow_test_majority=None,
            bow_dev_accuracy=None,
            n_test_classes=0,
            single_class_test_split=True,
            drop=True,
            drop_reasons=["no_test_items"],
        )
    source = test_items[0].source
    qtype = str(test_items[0].qtype)
    majority_label, majority_share = majority_baseline(test_items)
    regex_dev = score_regex(dev_items, patterns) if dev_items else {"accuracy": None}
    regex_test = score_regex(test_items, patterns)
    bow = bow_predict(dev_items, test_items, seed=seed)

    reasons: list[str] = []
    notes: list[str] = []
    n_test_classes = len({i.gold for i in test_items})
    if n_test_classes < 2:
        reasons.append("single_class_test_split")
        notes.append(
            "the test split has one gold class, so no accuracy baseline can discriminate: "
            "the template measures nothing as built"
        )
    if regex_test["accuracy"] is not None and regex_test["accuracy"] >= regex_threshold:
        reasons.append(f"regex_test_accuracy>={regex_threshold}")
    if bow["test_accuracy"] is not None and bow["test_accuracy"] >= bow_threshold:
        reasons.append(f"bow_test_accuracy>={bow_threshold}")
    if n_test_classes >= 2 and majority_share >= 0.90:
        notes.append(f"majority class is {majority_share:.4f} of the test split")
    if bow.get("test_macro_accuracy") is not None and bow["test_macro_accuracy"] <= 0.55:
        notes.append(
            f"BoW macro accuracy {bow['test_macro_accuracy']:.3f} <= 0.55: the classifier is "
            "close to a majority-class predictor, so its micro accuracy is not evidence of "
            "template difficulty either way"
        )
    if (
        qtype == QuestionType.NOUL.value
        and n_test_classes >= 2
        and max(regex_test["accuracy"] or 0.0, bow["test_accuracy"] or 0.0) <= 0.5
    ):
        notes.append("noul: chance is 0.5 and both baselines are at or below chance")
    return TemplateScreenResult(
        template_id=template_id,
        source=source,
        qtype=qtype,
        n_dev=len(dev_items),
        n_test=len(test_items),
        majority_label=majority_label,
        majority_test_accuracy=majority_share,
        regex_train_accuracy=regex_dev.get("accuracy"),
        regex_test_accuracy=regex_test["accuracy"],
        regex_test_macro_accuracy=regex_test["macro_accuracy"],
        regex_patterns=len(patterns),
        bow_test_accuracy=bow["test_accuracy"],
        bow_test_macro_accuracy=bow.get("test_macro_accuracy"),
        bow_dev_accuracy=bow["dev_accuracy"],
        bow_test_majority=bow["test_majority"],
        n_test_classes=n_test_classes,
        single_class_test_split=n_test_classes < 2,
        drop=bool(reasons),
        drop_reasons=reasons,
        notes=[*notes, bow.get("note", "")] if bow.get("note") else notes,
    )
