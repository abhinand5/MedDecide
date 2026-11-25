"""W3b pilot checks (osler_v2 ADVISORY W3b step 4): the acceptance reading of the chart pilot.

1. Stated numbers in narrative pages (the prompt forbids vital signs and lab values in narrative text).
2. Label-tracking phrases: narrative word n-grams whose presence differs by gold label (ratio ≥ 3 and ≥ 10 items).
3. Discharge wording against the gold code: how often the discharge summary names the gold condition, by gold class.
4. Ten charts, narrative pages only, shortened, printed for a fresh-process read (never written to a committed file).

Writes aggregates to outputs/osler_v2/W3b/pilot_checks.json. Nothing is written into the repository.

Usage: uv run --frozen python scripts/osler_v2/w3b_pilot_checks.py --dir data/gen/osler_v2_charts_pilot
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v2" / "W3b" / "pilot_checks.json"
NARRATIVE_WORDS = ("note", "discharge summary", "consult", "admission", "nutrition", "progress")
CODE_WORDS = ("medication", "lab", "mar", "bill", "flowsheet", "chemistry", "test |", "date ", "continued")
UNIT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mmol|mL|L|bpm|mmHg|%|g/dL|mg/dL|/min|°C|kg|cm|mEq|U/L|ng/mL|x10)",
                  re.IGNORECASE)
DECIMAL = re.compile(r"\b\d+\.\d+\b")
STOP = {"the", "and", "of", "to", "a", "in", "with", "for", "was", "is", "on", "at", "by", "as", "or", "an", "be",
        "this", "that", "from", "has", "had", "no", "not", "patient", "page", "his", "her", "he", "she"}


def pages(state: str) -> list[tuple[str, str]]:
    """(header, body) for every page; the header is the page's third line when the page is narrative."""
    out = []
    for block in re.split(r"(?==== Page )", state):
        lines = block.strip().split("\n")
        if not lines or not lines[0].startswith("=== Page"):
            continue
        header = lines[2].strip() if len(lines) > 2 else ""
        out.append((header, "\n".join(lines[1:])))
    return out


def load(dirpath: Path) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "dev", "test"):
        with (dirpath / f"{split}.jsonl").open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh)
    return rows


def is_narrative(header: str) -> bool:
    """Narrative pages by title (bold markers and capitalisation ignored); code pages are never narrative."""
    h = header.strip().strip("*").strip().lower()
    if any(re.search(r"\b" + re.escape(w.strip()) + r"\b", h) for w in CODE_WORDS):
        return False
    return any(w in h for w in NARRATIVE_WORDS)


def options_of(row: dict[str, Any]) -> list[dict[str, str]]:
    value = row["options"]
    return ast.literal_eval(value) if isinstance(value, str) else value


def narrative_text(row: dict[str, Any]) -> str:
    return "\n".join(body for header, body in pages(row["state"]) if is_narrative(header))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/gen/osler_v2_charts_pilot")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()
    rows = load(REPO / args.dir)
    headers = Counter(h for r in rows for h, _ in pages(r["state"]))
    result: dict[str, Any] = {"items": len(rows), "page_headers": dict(headers.most_common(30))}

    # 1. stated numbers in narrative pages
    n_pages = n_units = n_decimal = 0
    samples: list[str] = []
    for r in rows:
        for header, body in pages(r["state"]):
            if not is_narrative(header):
                continue
            n_pages += 1
            hits = UNIT.findall(body)
            n_units += bool(hits)
            n_decimal += bool(DECIMAL.search(body))
            if hits and len(samples) < 5:
                samples.append(hits[0])
    result["narrative_pages"] = n_pages
    result["narrative_pages_with_unit_number"] = n_units
    result["narrative_pages_with_decimal"] = n_decimal
    result["unit_number_examples_count_only"] = len(samples)

    # 2. label-tracking phrases, by template and gold class
    by_group: dict[str, dict[str, list[set[str]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        text = narrative_text(r).lower()
        words = [w for w in re.findall(r"[a-z]+", text) if w not in STOP]
        grams = {" ".join(words[i:i + 2]) for i in range(len(words) - 1)}
        grams |= {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}
        group = str(r["gold"])
        if r["generator"] == "chart_icd_support_v1":  # ICD: gold "none" against gold code (all code letters together)
            gold_label = next(o["label"] for o in options_of(r) if o["key"] == r["gold"])
            group = "none" if gold_label.startswith("None") else "code"
        by_group[r["generator"]][group].append(grams)
    tracking: dict[str, list[dict[str, Any]]] = {}
    for gen, groups in by_group.items():
        if len(groups) < 2:
            continue
        counts: dict[str, Counter[str]] = {g: Counter() for g in groups}
        for g, sets in groups.items():
            for s in sets:
                counts[g].update(s)
        totals = {g: len(sets) for g, sets in groups.items()}
        top: list[dict[str, Any]] = []
        keys = set().union(*(c.keys() for c in counts.values()))
        golds = sorted(groups)
        if len(golds) == 2:  # yes/no items, and the ICD none/code split
            a, b = golds
            for phrase in keys:
                ra = counts[a][phrase] / totals[a]
                rb = counts[b][phrase] / totals[b]
                if max(counts[a][phrase], counts[b][phrase]) < 10:
                    continue
                ratio = (max(ra, rb) + 1e-9) / (min(ra, rb) + 1e-9)
                if ratio >= 3 and abs(ra - rb) >= 0.10:
                    top.append({"phrase": phrase, f"share_gold_{a}": round(ra, 3), f"share_gold_{b}": round(rb, 3)})
        tracking[gen] = sorted(top, key=lambda x: -abs(list(x.values())[1] - list(x.values())[2]))[:15]
    result["label_tracking_phrases"] = tracking

    # 3. discharge wording against gold for chart_icd_support_v1
    icd = [r for r in rows if r["generator"] == "chart_icd_support_v1"]
    named = defaultdict(lambda: [0, 0])
    for r in icd:
        discharge = "\n".join(body for header, body in pages(r["state"]) if "discharge" in header.lower())
        opts = {o["key"]: o["label"] for o in options_of(r) if not o["label"].startswith("None")}
        gold_label = opts.get(r["gold"])
        gold_class = "none" if gold_label is None else "code"
        if gold_label is None:
            named[gold_class][1] += 1
            continue
        condition = gold_label.split(" - ", 1)[-1].split(",")[0].lower()
        named[gold_class][1] += 1
        named[gold_class][0] += condition[:25] in discharge.lower()
    result["icd_discharge_names_gold_condition"] = {k: {"named": v[0], "n": v[1]} for k, v in named.items()}

    # 4. ten charts, narrative pages only, shortened (for the read-through)
    rng = random.Random(0)
    shown = rng.sample(rows, min(args.show, len(rows)))
    print("== read-through (narrative pages, truncated) ==")
    for r in shown:
        print(f"--- {r['generator']} gold={r['gold']} split={r['split']}")
        for header, body in pages(r["state"]):
            if is_narrative(header):
                print(f"[{header}] " + " ".join(body.split())[:600])
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print("== counts ==")
    print(json.dumps({k: v for k, v in result.items() if k not in ("page_headers",)}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
