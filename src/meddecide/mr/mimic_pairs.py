"""Minimal pairs over MIMIC discharge notes (osler_v4 Y3; ADVISORY §Y3; D43, D45, D46). Pure functions, no network.

The teacher never writes note text. It returns the numbers of the sentences that document a condition; code deletes them.
Every surviving sentence is kept byte-identical, so the edit is a pure deletion. Verification and the string rules are
checked by code (``check_pure_deletion``, ``leaked_keywords``) and by the caller's independent yes/no calls.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from meddecide.mr.mimic import section_header

MAX_SELECTED_SHARE = 0.30          # ADVISORY Y3 §3: more than 30 % of sentences selected -> drop the pair
SENTENCE_BREAK = re.compile(r"((?<=[.!?]) +)")
STOP_WORDS = frozenset({
    "with", "without", "from", "that", "this", "than", "into", "over", "under", "due", "other", "unspecified", "specified",
    "disorder", "disease", "type", "site", "organism", "acute", "chronic", "and", "the", "for", "not", "elsewhere",
    "classified", "episode", "stage", "primary", "secondary", "history", "status", "encounter",
})


@dataclass
class Line:
    """One line of a note: a section header (kept as is), a blank line, or a line of sentences with their separators."""

    kind: str                                 # "header", "blank" or "text"
    raw: str
    sentences: list[list[str]] = field(default_factory=list)   # [[sentence, separator_before], ...] for kind "text"


@dataclass(frozen=True)
class SentenceRef:
    number: int                               # the 1-based number shown to the teacher
    line: int                                 # index into the note's lines
    slot: int                                 # index of the sentence inside its line
    text: str


def parse_note(note: str) -> tuple[list[Line], list[SentenceRef]]:
    """Lines of the note and the numbered sentences. Section headers (known MIMIC-IV headers) are lines of their own."""
    lines: list[Line] = []
    refs: list[SentenceRef] = []
    for index, raw in enumerate(note.split("\n")):
        if section_header(raw) is not None:
            lines.append(Line("header", raw))
            continue
        if not raw.strip():
            lines.append(Line("blank", raw))
            continue
        pieces = SENTENCE_BREAK.split(raw)          # sentence, separator, sentence, ...
        entries: list[list[str]] = []
        for k in range(0, len(pieces), 2):
            sep = pieces[k - 1] if k > 0 else ""
            entries.append([pieces[k], sep])
        line = Line("text", raw, entries)
        lines.append(line)
        for slot, (sentence, _) in enumerate(entries):
            refs.append(SentenceRef(len(refs) + 1, index, slot, sentence))
    return lines, refs


def numbered_view(lines: Sequence[Line]) -> str:
    """The note as the teacher sees it: one sentence per numbered line; headers as unnumbered lines; blank lines dropped."""
    out: list[str] = []
    counter = 0
    for line in lines:
        if line.kind == "header":
            out.append(line.raw.strip())
        elif line.kind == "text":
            for sentence, _ in line.sentences:
                counter += 1
                out.append(f"[{counter}] {sentence}")
    return "\n".join(out)


def delete_sentences(lines: Sequence[Line], numbers: Iterable[int]) -> str:
    """The note without the numbered sentences. A section whose sentences are all deleted loses its header too.
    Surviving sentences are emitted unchanged, with their original separators (minus a leading separator)."""
    drop = set(numbers)
    counter = 0
    out: list[str] = []
    pending_header: tuple[int, int] | None = None     # (index in out, number of sentences in the section)
    section_sentences = 0
    section_kept = 0

    def close_section() -> None:
        nonlocal pending_header
        if pending_header is not None and section_sentences > 0 and section_kept == 0:
            del out[pending_header[0]]
        pending_header = None

    for line in lines:
        if line.kind == "header":
            close_section()
            pending_header = (len(out), 0)
            out.append(line.raw)
            section_sentences = 0
            section_kept = 0
        elif line.kind == "blank":
            out.append(line.raw)
        else:
            kept_here: list[str] = []
            for sentence, sep in line.sentences:
                counter += 1
                section_sentences += 1
                if counter in drop:
                    continue
                section_kept += 1
                kept_here.append(sentence if not kept_here else sep + sentence)
            if kept_here:
                out.append("".join(kept_here))
    close_section()
    return "\n".join(out)


def check_pure_deletion(original: str, edited: str) -> bool:
    """True when every sentence of the edit is an original sentence, in the original order, byte-identical (ADVISORY iii)."""
    orig = [s for line in parse_note(original)[0] if line.kind == "text" for s, _ in line.sentences if s.strip()]
    edit = [s for line in parse_note(edited)[0] if line.kind == "text" for s, _ in line.sentences if s.strip()]
    it = iter(orig)
    return all(any(s == o for o in it) for s in edit)


def parse_selection(answer: str, n_sentences: int) -> list[int] | None:
    """The list of sentence numbers in the teacher's JSON answer, or None when it is not a list of distinct valid numbers."""
    match = re.search(r"\[[^\[\]]*\]", answer)
    if match is None:
        return None
    try:
        values = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(values, list) or not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return None
    if len(set(values)) != len(values) or any(v < 1 or v > n_sentences for v in values):
        return None
    return sorted(values)


def parse_yes_no(answer: str) -> str | None:
    """'yes' or 'no' when the answer begins with it (case and punctuation ignored); None for anything else."""
    words = re.findall(r"[a-z]+", answer.lower())
    if not words:
        return None
    return words[0] if words[0] in ("yes", "no") else None


def condition_keywords(title: str, family_terms: Iterable[str] = ()) -> tuple[str, ...]:
    """Keywords that must not survive in the edit: content words of the ICD-10-CM title (four letters or more, stop words
    removed) plus the curated family's condition words (ADVISORY §Y3 step 5(ii))."""
    words = {w for w in re.findall(r"[a-z]+", title.lower()) if len(w) >= 4 and w not in STOP_WORDS}
    words |= {w.lower() for term in family_terms for w in re.findall(r"[A-Za-z]+", term) if len(w) >= 4}
    return tuple(sorted(words - STOP_WORDS))


def leaked_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    """The keywords that still occur, as whole words, in the text (case-insensitive)."""
    found = []
    for word in keywords:
        if re.search(rf"\b{re.escape(word)}", text, flags=re.IGNORECASE):
            found.append(word)
    return found


def selection_share(refs: Sequence[SentenceRef], numbers: Sequence[int]) -> float:
    return len(numbers) / len(refs) if refs else 0.0


def other_codes(codes: Sequence[str], excluded: str) -> list[str]:
    """The distinct coded diagnoses other than the target, in the order given."""
    return [c for c in dict.fromkeys(codes) if c != excluded]


# ---- the pair pipeline (one admission, one coded secondary diagnosis) --------------------------------

MIN_SENTENCES = 5
SELECT_SYSTEM = "You are a clinical documentation reviewer. You never write or rewrite text."
VERIFY_SYSTEM = "You check whether a clinical record documents a given diagnosis. Answer with yes or no only."
YES_NO_OPTIONS = (("yes", "Yes"), ("no", "No"))


@dataclass(frozen=True)
class Attempt:
    """The outcome of one admission: ``outcome`` is "kept" or the reason it was dropped (no silent drops)."""

    outcome: str
    n_sentences: int = 0
    n_selected: int = 0
    calls: int = 0
    original: str = ""
    edited: str = ""
    others_checked: int = 0
    others_skipped: int = 0


def select_messages(numbered: str, title: str, code: str) -> list[dict[str, str]]:
    user = (f"Condition: {title} (ICD-10-CM {code}).\n\nThe numbered text below is a discharge note. Return only a JSON list "
            "of the numbers of the sentences that state or evidence this condition, including treatments given only for it. "
            "Return [] if none. Do not write any text.\n\n" + numbered)
    return [{"role": "system", "content": SELECT_SYSTEM}, {"role": "user", "content": user}]


def verify_messages(note: str, title: str, code: str) -> list[dict[str, str]]:
    user = f"Record:\n{note}\n\nDoes this record document {title} (ICD-10-CM {code})? Answer yes or no."
    return [{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": user}]


def question_text(title: str, dotted_code: str) -> str:
    return f"Is ICD-10-CM {dotted_code} {title} supported by the documentation in this note?"


def build_pair(note: str, code: str, title: str, family_terms: Sequence[str], others: Sequence[tuple[str, str]],
               chat: Callable[[list[dict[str, str]], str], str], rng: random.Random) -> Attempt:
    """One pair for one coded secondary diagnosis (ADVISORY Y3 steps 3-5). ``others`` are the admission's other coded
    (code, title) pairs; two of them are re-checked on the edit when the original documents them."""
    note = note.replace("\r\n", "\n").replace("\r", "\n")
    lines, refs = parse_note(note)
    if len(refs) < MIN_SENTENCES:
        return Attempt("too few sentences", n_sentences=len(refs))
    calls = 0
    selected: list[int] | None = None
    for _ in range(2):   # an answer that is not a valid list of distinct numbers is asked for once more
        calls += 1
        selected = parse_selection(chat(select_messages(numbered_view(lines), title, code), "select"), len(refs))
        if selected is not None:
            break
    if selected is None:
        return Attempt("selection not a valid list", len(refs), 0, calls)
    if not selected:
        return Attempt("nothing selected", len(refs), 0, calls)
    if selection_share(refs, selected) > MAX_SELECTED_SHARE:
        return Attempt("more than 30 % of sentences selected", len(refs), len(selected), calls)
    edited = delete_sentences(lines, selected)
    if not edited.strip() or not check_pure_deletion(note, edited):
        return Attempt("edit is not a pure deletion", len(refs), len(selected), calls)
    if leaked_keywords(edited, condition_keywords(title, family_terms)):
        return Attempt("string rule: a keyword is left in the edit", len(refs), len(selected), calls)
    calls += 1
    if parse_yes_no(chat(verify_messages(note, title, code), "verify")) != "yes":
        return Attempt("original not read as documented", len(refs), len(selected), calls)
    calls += 1
    if parse_yes_no(chat(verify_messages(edited, title, code), "verify")) != "no":
        return Attempt("edit still read as documented", len(refs), len(selected), calls)
    pool = list(others)
    rng.shuffle(pool)
    checked = skipped = 0
    for other_code, other_title in pool:
        if checked == 2:
            break
        calls += 1
        if parse_yes_no(chat(verify_messages(note, other_title, other_code), "verify")) != "yes":
            skipped += 1
            continue
        calls += 1
        if parse_yes_no(chat(verify_messages(edited, other_title, other_code), "verify")) != "yes":
            return Attempt("other coded diagnosis lost by the edit", len(refs), len(selected), calls, others_checked=checked,
                           others_skipped=skipped)
        checked += 1
    return Attempt("kept", len(refs), len(selected), calls, note, edited, checked, skipped)
