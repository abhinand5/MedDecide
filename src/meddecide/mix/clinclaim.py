"""Claim-grounding items over real clinical text (osler_v3 X3; D43; ADVISORY X3 step 1): ``clinclaim_grounding_v1``.

For a passage of real text (1-4 paragraphs), code draws a label (supported, contradicted, not mentioned) and, for contradicted,
an edit kind (number, laterality, negation, drug, temporal order). A local Gemma writes one claim following the label. An
independent verification call (a different system prompt, temperature 0) classifies the claim against the passage. An item is
kept only if the verifier's class equals the drawn label, and, for contradicted items, a string rule finds the edited element in
the passage and different in the claim. For supported and not-mentioned items, a claim that copies more than 80 % of a passage
sentence verbatim is rejected. Labels come from the code, never from the writer or the verifier.

Pure functions only; the generation and the assembly are in scripts/osler_v3/x3_generate.py and x3_assemble.py.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from meddecide.mix.rows import make_row
from meddecide.mix.sentences import sentence_spans
from meddecide.utils.hashing import stable_hash

TEMPLATE = "clinclaim_grounding_v1"
SKILL = "clinical_grounding"
LABELS = ("supported", "contradicted", "not_mentioned")
LABEL_TEXT = {
    "supported": "Supported by the passage",
    "contradicted": "Contradicted by the passage",
    "not_mentioned": "Not mentioned in the passage",
}
EDIT_KINDS = ("number", "laterality", "negation", "drug", "temporal")
EDIT_INSTRUCTION = {
    "number": "change one number (a value, a count, a dose, a size or a duration) to a different number",
    "laterality": "change one laterality (left or right, or bilateral) to the opposite side",
    "negation": "turn one finding that the passage states as present into one stated as absent, or the reverse",
    "drug": "replace one drug name with a different drug name",
    "temporal": "reverse or change one temporal order (for example before and after, or the sequence of two events)",
}
COPY_LIMIT = 0.80  # ADVISORY X3: reject supported or not-mentioned claims copying more than 80 % of a sentence
MAX_CLAIM_WORDS = 30
LATERAL = ("left", "right", "bilateral")
NEGATION = ("no", "not", "without", "denies", "denied", "negative", "absent", "absence", "none")
TEMPORAL = ("before", "after", "prior", "previously", "subsequently", "then", "followed", "following", "preceding", "earlier",
            "later", "since")
# common generic drug names (public vocabulary) for the drug edit kind
DRUGS = (
    "aspirin", "warfarin", "heparin", "enoxaparin", "clopidogrel", "metoprolol", "atenolol", "carvedilol", "lisinopril",
    "enalapril", "losartan", "amlodipine", "furosemide", "spironolactone", "hydrochlorothiazide", "digoxin", "amiodarone",
    "atorvastatin", "simvastatin", "rosuvastatin", "metformin", "insulin", "glipizide", "prednisone", "methylprednisolone",
    "dexamethasone", "hydrocortisone", "ceftriaxone", "cefazolin", "vancomycin", "piperacillin", "meropenem", "azithromycin",
    "ciprofloxacin", "levofloxacin", "doxycycline", "metronidazole", "clindamycin", "gentamicin", "fluconazole", "acyclovir",
    "morphine", "hydromorphone", "fentanyl", "oxycodone", "tramadol", "acetaminophen", "ibuprofen", "naproxen", "ketorolac",
    "gabapentin", "pregabalin", "levetiracetam", "phenytoin", "lorazepam", "midazolam", "diazepam", "haloperidol",
    "quetiapine", "olanzapine", "sertraline", "fluoxetine", "citalopram", "trazodone", "omeprazole", "pantoprazole",
    "famotidine", "ondansetron", "metoclopramide", "albuterol", "tiotropium", "montelukast", "levothyroxine", "allopurinol",
    "colchicine", "tamsulosin", "finasteride", "oxybutynin", "potassium chloride", "magnesium sulfate", "sodium bicarbonate",
    "norepinephrine", "epinephrine", "vasopressin", "dopamine", "dobutamine", "propofol", "ketamine", "rocuronium",
)
SYSTEM_WRITE = (
    "You write one short clinical claim about a passage of real clinical text. You follow the instruction exactly and reply "
    "with the claim only: one sentence, no quotation marks, no explanation."
)
SYSTEM_VERIFY = (
    "You are a strict clinical fact checker. You judge one claim against one passage, using the passage alone. You reply with "
    "exactly one of three words: SUPPORTED, CONTRADICTED or NOT_MENTIONED."
)
_VERDICT = re.compile(r"\b(SUPPORTED|CONTRADICTED|NOT[ _]MENTIONED)\b")
_VERDICT_LABEL = {"SUPPORTED": "supported", "CONTRADICTED": "contradicted", "NOT_MENTIONED": "not_mentioned"}


def applicable_kinds(passage: str) -> tuple[str, ...]:
    """The edit kinds this passage can carry: a number edit needs a number, a laterality edit a side word, a drug edit a listed
    drug, a temporal edit a temporal cue. A negation edit can always be made (a finding can be stated as absent)."""
    kinds = []
    if numbers(passage):
        kinds.append("number")
    if cue_set(passage, LATERAL):
        kinds.append("laterality")
    kinds.append("negation")
    if cue_set(passage, TEMPORAL):
        kinds.append("temporal")
    if cue_set(passage, DRUGS):
        kinds.append("drug")
    return tuple(k for k in EDIT_KINDS if k in kinds)


def draw(doc_key: str, index: int, seed: int, passage: str) -> tuple[str, str | None]:
    """The code's label and, for contradicted, its edit kind among the kinds the passage can carry (stable hash)."""
    label = LABELS[int(stable_hash({"clinclaim": doc_key, "index": index, "seed": seed, "field": "label"}, length=8), 16)
                   % len(LABELS)]
    kind = None
    if label == "contradicted":
        kinds = applicable_kinds(passage)
        kind = kinds[int(stable_hash({"clinclaim": doc_key, "index": index, "seed": seed, "field": "kind"}, length=8), 16)
                     % len(kinds)]
    return label, kind


# one style rule for every label, so that no wording tracks the label (the one prompt fix allowed for this template, D-X3-1)
STYLE_RULE = ("Do not begin with 'The patient', and do not use the word 'history'. Vary the subject of the sentence.")


def writer_messages(passage: str, label: str, kind: str | None) -> list[dict[str, str]]:
    if label == "supported":
        instruction = ("Write a claim that the passage states. Paraphrase one fact the passage states in your own words; do not "
                       "copy a whole sentence. " + STYLE_RULE)
    elif label == "not_mentioned":
        instruction = ("Write a plausible clinical claim about this case that the passage does NOT state and does NOT imply. "
                       "Choose a fact related to the passage's topic that the passage leaves unaddressed (for example a test not "
                       "reported, a medication not named, a date or a family or social detail not given). Do not name a disease "
                       "the passage does not mention. " + STYLE_RULE)
    else:
        if kind not in EDIT_KINDS:
            raise ValueError(f"unknown edit kind {kind!r}")
        instruction = ("Write a claim that states one fact the passage states, but " + EDIT_INSTRUCTION[kind] +
                       ", so that the claim contradicts the passage. Do not say that anything was changed. " + STYLE_RULE)
    user = (f"PASSAGE:\n{passage}\n\nINSTRUCTION: {instruction}\nWrite one sentence of at most {MAX_CLAIM_WORDS} words. "
            "Reply with the claim only.")
    return [{"role": "system", "content": SYSTEM_WRITE}, {"role": "user", "content": user}]


def verifier_messages(passage: str, claim: str) -> list[dict[str, str]]:
    user = (f"PASSAGE:\n{passage}\n\nCLAIM: {claim}\n\nDoes the passage SUPPORT the claim, CONTRADICT it, or NOT_MENTION it? "
            "Reply with one word: SUPPORTED, CONTRADICTED or NOT_MENTIONED.")
    return [{"role": "system", "content": SYSTEM_VERIFY}, {"role": "user", "content": user}]


def clean_claim(text: str) -> str:
    """The first non-empty line of the writer's reply, without a leading label and without quotes."""
    for line in str(text).splitlines():
        line = re.sub(r"^\s*(claim|answer|sentence)\s*:\s*", "", line.strip(), flags=re.IGNORECASE)
        line = line.strip().strip('"').strip("'").strip()
        if line:
            return line
    return ""


def parse_verdict(text: str) -> str | None:
    """The verifier's class, or None when the reply names no class or more than one."""
    found = {_VERDICT_LABEL[m.group(1).replace(" ", "_")] for m in _VERDICT.finditer(str(text).upper())}
    return next(iter(found)) if len(found) == 1 else None


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text).lower())


def passage_sentences(passage: str) -> list[str]:
    return [passage[a:b] for a, b in sentence_spans(passage)]


def cue_set(text: str, cues: Sequence[str]) -> frozenset[str]:
    low = str(text).lower()
    return frozenset(c for c in cues if re.search(rf"(?<![a-z0-9]){re.escape(c)}(?![a-z0-9])", low))


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", str(text)))


def best_sentence(claim: str, passage: str) -> str | None:
    """The passage sentence sharing the most word types with the claim (the first one on a tie), or None."""
    claim_types = set(tokens(claim))
    best, best_score = None, 0.0
    for sentence in passage_sentences(passage):
        types = set(tokens(sentence))
        if not types:
            continue
        score = len(claim_types & types) / len(claim_types | types)
        if score > best_score:
            best, best_score = sentence, score
    return best


def edit_ok(kind: str, claim: str, passage: str) -> bool:
    """The string rule for a contradicted claim: the edited element exists in the passage and differs in the claim."""
    if kind == "number":
        in_passage, in_claim = numbers(passage), numbers(claim)
        return bool(in_passage) and bool(in_claim - in_passage)
    if kind == "laterality":
        c, p = cue_set(claim, LATERAL), cue_set(passage, LATERAL)
        return bool(c) and bool(p) and c != p
    sentence = best_sentence(claim, passage)
    if sentence is None:
        return False
    if kind == "negation":
        return bool(cue_set(claim, NEGATION)) != bool(cue_set(sentence, NEGATION))
    if kind == "temporal":
        c = cue_set(claim, TEMPORAL)
        return bool(c) and c != cue_set(sentence, TEMPORAL)
    if kind == "drug":
        c, p = cue_set(claim, DRUGS), cue_set(passage, DRUGS)
        return bool(c - p) and bool(p - c)
    raise ValueError(f"unknown edit kind {kind!r}")


def longest_run(a: Sequence[str], b: Sequence[str]) -> int:
    """The length of the longest contiguous token run common to both sequences."""
    best = 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def copy_share(claim: str, passage: str) -> float:
    """The largest share of the claim's words that form one contiguous run inside a single passage sentence."""
    claim_tokens = tokens(claim)
    if not claim_tokens:
        return 1.0
    runs = [longest_run(claim_tokens, tokens(s)) for s in passage_sentences(passage)]
    return max(runs, default=0) / len(claim_tokens)


def keep(label: str, kind: str | None, claim: str, passage: str, verdict: str | None) -> tuple[bool, str]:
    """Whether a generated item is kept, and the reason (every drop has a reason)."""
    if not claim:
        return False, "empty_claim"
    if len(tokens(claim)) > MAX_CLAIM_WORDS:
        return False, "claim_too_long"
    if verdict is None:
        return False, "verifier_no_single_class"
    if verdict != label:
        return False, "verifier_disagrees"
    if label == "contradicted":
        if kind is None or not edit_ok(kind, claim, passage):
            return False, "edit_rule_failed"
        return True, "kept"
    if copy_share(claim, passage) > COPY_LIMIT:
        return False, "copies_passage_sentence"
    return True, "kept"


def option_order(record_id: str) -> list[str]:
    """The three labels in an order that depends only on the record id (ADVISORY X3: options in random order)."""
    return sorted(LABELS, key=lambda lab: stable_hash({"order": record_id, "label": lab}, length=12))


def build_row(record_id: str, passage: str, claim: str, label: str, meta: dict[str, Any], *, licence: str,
              record_url: str, split: str) -> dict[str, Any]:
    """The mix-schema row for one kept item."""
    order = option_order(record_id)
    keys = dict(zip(order, "ABC", strict=True))
    options = [(keys[lab], LABEL_TEXT[lab]) for lab in order]
    return make_row(source=str(meta["source"]), template_id=TEMPLATE, skill=SKILL, qtype="choice", state=passage,
                    question=f"Is this claim supported, contradicted or not mentioned by the passage? Claim: {claim}",
                    options=options, gold=keys[label], record_id=record_id, record_url=record_url, licence=licence,
                    record_date="", split=split, tier="osler_v3_source", meta=dict(meta, label=label))


def balance_claims(rows: list[dict[str, Any]], salt: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Equal numbers for the three labels (ADVISORY X3: balance the three labels). The edit kinds are not balanced: the kinds a
    passage can carry differ in frequency, and their agreement and keep rates are reported instead (ADVISORY §7.3). Deterministic
    (stable hash)."""
    def pick(items: list[dict[str, Any]], n: int, tag: str) -> list[dict[str, Any]]:
        return sorted(items, key=lambda r: stable_hash({"salt": salt, "key": r["key"], "tag": tag}, length=16))[:n]

    groups = {lab: [r for r in rows if r["label"] == lab] for lab in LABELS}
    n = min(len(v) for v in groups.values()) if rows else 0
    kept = [r for lab in LABELS for r in pick(groups[lab], n, lab)]
    report = {"before": {"labels": dict(Counter(r["label"] for r in rows)),
                         "contradicted_kinds": dict(Counter(r["kind"] for r in rows if r["label"] == "contradicted"))},
              "per_label_target": n, "after": len(kept),
              "after_contradicted_kinds": dict(Counter(r["kind"] for r in kept if r["label"] == "contradicted"))}
    return kept, report
