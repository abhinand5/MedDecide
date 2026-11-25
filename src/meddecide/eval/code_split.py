"""Option-code diagnostics for a readout with more than 26 options: accuracy and probability mass by the gold code type.

Codes are the positional letters of ``meddecide.eval.readout.option_letter``: index 0 to 25 is A to Z, and index 26 and up is a
two-letter base-26 code (BA for 26, BW for 48, the last of the 49 DDXPlus options). The first two-letter code is BA, because the
digits run A to Z with no zero digit, so no AA exists. The split is by the gold index: below 26 is a single-letter gold, 26 and
up a two-letter gold.

These are additional diagnostics (the Y8 steer of 2026-10-24). They change no verdict or choice, and a readout cell that fails
the D12 gate is not reported as an accuracy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from meddecide.eval.readout import option_letter

SINGLE_LETTERS = 26
SECTION_BEGIN = "<!-- y8-code-split:begin -->"
SECTION_END = "<!-- y8-code-split:end -->"


def letter_index(code: str) -> int:
    """The 0-based option index of a canonical option code (A is 0, Z is 25, BA is 26). Any other string raises ``ValueError``."""
    if not code or any(not ("A" <= ch <= "Z") for ch in code):
        raise ValueError(f"not an option code: {code!r}")
    index = 0
    for ch in code:
        index = index * 26 + (ord(ch) - ord("A"))
    if option_letter(index) != code:
        raise ValueError(f"not a canonical option code: {code!r}")
    return index


def option_index(key: str) -> int:
    """The 0-based option index of a run's option key.

    The trainer keys options A, B, ... (``letter_index``). The zero-shot runner keys them by number, 1 for the first option of a score
    item or of a 49-option DDXPlus item, and keys a yes/no item yes and no, with yes the first option (the trainer's A). The script
    that uses this mapping checks it against the trainer's gold index for every item.
    """
    if key in ("yes", "no"):
        return 0 if key == "yes" else 1
    if key.isdigit():
        if str(int(key)) != key or int(key) < 1:
            raise ValueError(f"not a numbered option key: {key!r}")
        return int(key) - 1
    return letter_index(key)


def two_letter_mass(probs: Sequence[float]) -> float:
    """One item's probability on the two-letter codes (option index 26 and up)."""
    return float(sum(probs[SINGLE_LETTERS:]))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _group(items: Sequence[tuple[bool, Sequence[float]]]) -> dict[str, Any]:
    correct = sum(flag for flag, _ in items)
    return {
        "items": len(items),
        "correct": correct,
        "accuracy": correct / len(items) if items else None,
        "two_letter_mass_mean": _mean([two_letter_mass(probs) for _, probs in items]),
    }


def split_by_gold_code(items: Sequence[tuple[int, bool, Sequence[float]]]) -> dict[str, Any]:
    """One readout's figures split by the gold code type. Each item is (gold index, correct flag, probability vector).

    An empty group has accuracy None and mass None. The overall two-letter mass is the mean over every item.
    """
    if not items:
        raise ValueError("no items to split")
    single: list[tuple[bool, Sequence[float]]] = []
    two_letter: list[tuple[bool, Sequence[float]]] = []
    for gold, correct, probs in items:
        if not 0 <= gold < len(probs):
            raise ValueError("the gold index is outside the item's option vector")
        (two_letter if gold >= SINGLE_LETTERS else single).append((bool(correct), probs))
    return {
        "items": len(items),
        "single_gold": _group(single),
        "two_letter_gold": _group(two_letter),
        "two_letter_mass_mean": _mean([two_letter_mass(probs) for _, _, probs in items]),
    }


def _accuracy(value: float | None, status: str) -> str:
    if status == "NOT RUN":
        return "NOT RUN"
    if not status.startswith("PASS"):
        return "READOUT_FAIL"
    return "—" if value is None else f"{value:.4f}"


def _count(value: int) -> str:
    return f"{value:,}"


def _mass(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def render_code_section(result: Mapping[str, Any]) -> str:
    """The additional section of results.md, from the JSON that ``scripts/osler_v4/y8_code_split.py`` writes.

    Figures only. A cell whose readout status is not PASS reads READOUT_FAIL and shows no accuracy (D12).
    """
    labels: Mapping[str, str] = result["labels"]
    ddx: Mapping[str, Mapping[str, Any]] = result["ddxplus"]
    lines = [
        SECTION_BEGIN,
        "## Additional rows: CG v2 by option-code type, and per template (dev only)",
        "",
        "Source: `outputs/osler_v4/Y8/code_split.json` (written by `scripts/osler_v4/y8_code_split.py`; aggregates only). "
        "CG v2 is a development set: every item has split `dev`. These are additional, labelled rows; no verdict or choice uses them.",
        "",
        "### CG v2 DDXPlus accuracy by gold code, and the mean probability mass on two-letter codes",
        "",
        "| model | gold A-Z: items | accuracy | gold two-letter (BA-BW): items | accuracy | mean mass on two-letter codes, all items "
        "| mean mass, A-Z gold | mean mass, two-letter gold |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, label in labels.items():
        entry = ddx[key]
        split = entry["split"]
        status = entry["status"]
        single, two = split["single_gold"], split["two_letter_gold"]
        lines.append(f"| {label} | {_count(single['items'])} | {_accuracy(single['accuracy'], status)} "
                     f"| {_count(two['items'])} | {_accuracy(two['accuracy'], status)} | {_mass(split['two_letter_mass_mean'])} "
                     f"| {_mass(single['two_letter_mass_mean'])} | {_mass(two['two_letter_mass_mean'])} |")
    lines += [
        "",
        f"Two-letter codes are option indexes 26 and up (keys BA to BW for the 49 options). The steer's label \"AA+\" is this split: "
        f"no AA key exists, because the base-26 letters run A to Z. Share of the DDXPlus gold answers that are two-letter: "
        f"{result['ddx_gold_two_letter_share']:.4f} ({_count(result['ddx_items'])} items).",
        "",
        "### CG v2 accuracy per template (micro accuracy on the items each model scored)",
        "",
        "| template | items | " + " | ".join(labels.values()) + " |",
        "|---|---|" + "---|" * len(labels),
    ]
    for row in result["templates"]:
        reference_items = row["items"][result["reference"]]
        cells = []
        for key in labels:
            n = row["items"][key]
            cell = _accuracy(row["accuracy"][key], row["status"][key])
            cells.append(cell if n == reference_items else f"{cell} (n = {_count(n)})")
        lines.append(f"| {row['template']} | {_count(reference_items)} | " + " | ".join(cells) + " |")
    unscored = result["unscored"]
    lines.append("")
    lines.append(f"The items column is the count of the reference run, {result['reference_name']}; a model with another count shows it in its cell.")
    if unscored["items"]:
        by_template = ", ".join(f"{count} {template}" for template, count in unscored["by_template"].items())
        lines.append(
            f"{labels[unscored['run']]} skips {unscored['items']} CG v2 items that the trainer-based runs keep "
            f"({unscored['repeated_label_items']} of them have repeated option labels; C086): {by_template}."
        )
    lines += [SECTION_END, ""]
    return "\n".join(lines)


def place_section(text: str, section: str) -> str:
    """Replaces the marked section of a results file, or appends it when the file has no marked section."""
    if SECTION_BEGIN in text and SECTION_END in text:
        head, _, rest = text.partition(SECTION_BEGIN)
        _, _, tail = rest.partition(SECTION_END)
        tail = tail.lstrip("\n")
        return head + section + ("\n" + tail if tail else "")
    return text.rstrip("\n") + "\n\n" + section
