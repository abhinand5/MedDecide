"""The real-world look of a payer's medical record: fax banners, page furniture and OCR-style character noise.

Records reach a payer as faxed or scanned PDFs run through OCR. The noise here imitates that without touching the
facts a label depends on: digits, and the characters next to digits, are never changed (``protect_digits``), so a
value cannot cross a threshold by noise; words are only perturbed by visually confusable substitutions.
"""

from __future__ import annotations

import random

CONFUSABLE: dict[str, tuple[str, ...]] = {
    "l": ("1", "I"), "I": ("l", "1"), "O": ("0",), "o": ("0",), "m": ("rn",), "rn": ("m",), "S": ("5",),
    "B": ("8",), "e": ("c",), "c": ("e",), "h": ("b",), "t": ("f",), "a": ("o",), "i": ("l",),
}


def fax_banner(rng: random.Random, page: int, pages: int) -> str:
    sender = rng.choice(("HIM DEPT", "MEDICAL RECORDS", "RELEASE OF INFO", "HIM/ROI", "PATIENT ACCOUNTS"))
    phone = f"({rng.randint(200, 989)}) {rng.randint(200, 989)}-{rng.randint(1000, 9999)}"
    stamp = f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/2026 {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"
    style = rng.randrange(3)
    if style == 0:
        return f"{stamp}  FROM: {sender} {phone}  TO: CLAIMS REVIEW   P.{page:03d}/{pages:03d}"
    if style == 1:
        return f"Received {stamp} via fax  |  {sender}  |  Page {page} of {pages}"
    return f"{sender} FAX {phone}   {stamp}   #{rng.randint(1000, 9999)} P. {page}"


def page_footer(rng: random.Random, page: int, pages: int) -> str:
    style = rng.randrange(3)
    if style == 0:
        return f"Page {page} of {pages}"
    if style == 1:
        return f"CONFIDENTIAL - Contains protected health information        {page}/{pages}"
    return f"Printed {rng.randint(1, 12)}/{rng.randint(1, 28)}/2026 by HIM    Page {page}"


def _near_digit(text: str, i: int, width: int) -> bool:
    lo, hi = max(0, i - 1), min(len(text), i + width + 1)
    return any(ch.isdigit() for ch in text[lo:hi])


def ocr_noise(text: str, rng: random.Random, rate: float = 0.01, *, protect_digits: bool = True) -> str:
    """Substitute visually confusable characters at about ``rate`` per character, drop a few spaces, and break a few
    lines. Digits (and characters touching a digit) are never changed when ``protect_digits`` is on."""
    if rate <= 0:
        return text
    out: list[str] = []
    i = 0
    while i < len(text):
        two = text[i:i + 2]
        if two in CONFUSABLE and rng.random() < rate and not (protect_digits and _near_digit(text, i, 2)):
            out.append(rng.choice(CONFUSABLE[two]))
            i += 2
            continue
        ch = text[i]
        if ch in CONFUSABLE and rng.random() < rate and not (protect_digits and _near_digit(text, i, 1)):
            out.append(rng.choice(CONFUSABLE[ch]))
        elif ch == " " and rng.random() < rate / 2 and not (protect_digits and _near_digit(text, i, 1)):
            pass  # dropped space
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def dress_page(body: str, rng: random.Random, *, page: int | None = None, pages: int | None = None,
               fax_p: float = 0.5, footer_p: float = 0.7, noise_rate: float | None = None) -> str:
    """Add a fax banner, a footer and OCR noise to one rendered page. The page number is random when not given."""
    pages = pages or rng.randint(4, 60)
    page = page or rng.randint(1, pages)
    parts = []
    if rng.random() < fax_p:
        parts.append(fax_banner(rng, page, pages))
    parts.append(body)
    if rng.random() < footer_p:
        parts.append(page_footer(rng, page, pages))
    text = "\n".join(parts)
    rate = rng.choice((0.0, 0.0, 0.005, 0.01, 0.02)) if noise_rate is None else noise_rate
    return ocr_noise(text, rng, rate)
