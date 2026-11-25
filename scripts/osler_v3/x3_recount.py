"""osler_v3 X3 recount (ADVISORY X3; R6): re-derive the X3 numbers in a fresh process and compare them with build.json.

Reads the candidate files written by x3_generate.py (data/gen/osler_v3/x3_<task>_<split>.jsonl) and recounts: written, kept and
rejection reasons per split; the restyle verifier agreement and its cross-tab with the keep flag; the claims label-tracking share of
"family" (kept and all candidates); and the balanced restyle sets, using the assembler's own balancing, screen and phrase check and
the same protected-paragraph filter. Every recounted number is compared with outputs/osler_v3/X3/build.json. The log is
outputs/osler_v3/X3/recount.log; any mismatch is printed and the exit status is 1.

Usage:  source scripts/pod_env.sh && uv run --frozen python scripts/osler_v3/x3_recount.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.gen.screen import gold_in_state
from meddecide.mix import restyle as rs
from meddecide.mix.protected import (
    hit_total,
    leakage_counts,
    load_protected,
    paragraph_key,
    protected_paragraph_keys,
)
from meddecide.mix.substrate import paragraphs
from meddecide.mr.realtext import PAGE_LETTER
from meddecide.utils.io import read_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "data" / "gen" / "osler_v3"
OUT = REPO / "outputs" / "osler_v3" / "X3"
SPLITS = ("train", "dev", "test")
FAMILY = re.compile(r"\bfamily\b")


def assembler() -> Any:
    """The X3 assembler, loaded by path (scripts/ is not a package); its balancing, screen and phrase check are reused."""
    spec = importlib.util.spec_from_file_location("x3_assemble", REPO / "scripts" / "osler_v3" / "x3_assemble.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def share(numerator: int, denominator: int) -> str:
    return f"{numerator} of {denominator} ({100 * numerator / max(1, denominator):.1f} %)"


def main() -> int:
    asm = assembler()
    build = read_json(OUT / "build.json")
    checks: list[tuple[str, Any, Any]] = []
    info: list[str] = []
    records: dict[str, list[dict[str, Any]]] = {}
    for task, prefix in (("claims", "x3_claims"), ("restyle", "x3_restyle")):
        records[task] = []
        for split in SPLITS:
            rows = asm.read_candidates(GEN / f"{prefix}_{split}.jsonl")
            if any(not isinstance(r.get("kept"), bool) for r in rows):
                raise SystemExit(f"{prefix}_{split}: a candidate has no boolean keep flag")
            records[task].extend(rows)
            key = f"{task}/{split}"
            reported = build["candidates"].get(key, {})
            reasons = dict(Counter(r.get("reason", "missing") for r in rows))
            kept = sum(r["kept"] for r in rows)
            checks.append((f"{key} written", len(rows), reported.get("written")))
            if task == "restyle":
                checks.append((f"{key} kept", kept, reported.get("kept")))
                checks.append((f"{key} reasons", reasons, reported.get("reasons")))
                checks.append((f"{key} reasons sum to written", sum(reasons.values()), len(rows)))
            else:
                info.append(f"INFO {key} (not in build.json; claims blocked): kept {share(kept, len(rows))}, reasons {reasons}")

    restyle = records["restyle"]
    agrees = [r.get("verified_letter") == PAGE_LETTER[r["page_class"]] for r in restyle]
    table = build["agreement"]["restyle"]
    checks.append(("restyle candidates (agreement table)", len(restyle), table["candidates"]))
    checks.append(("restyle verifier agrees", sum(agrees), table["verifier_agrees"]))
    checks.append(("restyle kept (agreement table)", sum(r["kept"] for r in restyle), table["kept"]))
    cross = Counter((agree, r["kept"], r.get("reason", "missing")) for agree, r in zip(agrees, restyle, strict=True))
    info.append(f"INFO restyle cross-tab (verifier agrees, kept, reason): {sorted(cross.items(), key=str)}")
    info.append(f"INFO claims verifier agreement by label and kind (not in build.json): {asm.agreement(records['claims'])}")

    claims = records["claims"]
    for name, pool in (("kept", [r for r in claims if r["kept"]]), ("all candidates", claims)):
        not_mentioned = [r for r in pool if r["label"] == "not_mentioned"]
        others = [r for r in pool if r["label"] != "not_mentioned"]
        fam_nm = sum(bool(FAMILY.search(r["claim"].lower())) for r in not_mentioned)
        fam_other = sum(bool(FAMILY.search(r["claim"].lower())) for r in others)
        info.append(f"INFO claims 'family' ({name}): not_mentioned {share(fam_nm, len(not_mentioned))}; "
                    f"supported or contradicted {share(fam_other, len(others))}")

    protected = protected_paragraph_keys(REPO)
    balanced: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        kept = [r for r in restyle if r["split"] == split and r["kept"]]
        kept = [r for r in kept
                if not any(paragraph_key(p) in protected for p in paragraphs(r.get("passage") or r.get("page") or "")
                           if len(p) >= 60)]
        picked, bal = asm.balance_pages(kept, salt=f"restyle:{split}")
        balanced[split] = asm.page_rows(picked)
        reported = build["balance"][f"restyle/{split}"]
        checks.append((f"restyle/{split} balanced per class", bal["per_class_target"], reported["per_class_target"]))
        checks.append((f"restyle/{split} balanced rows", len(balanced[split]), reported["after"]))

    screen = asm.screen(balanced["train"], balanced["dev"], rs.TEMPLATE)
    reported_screen = next((s for s in build["checks"]["screens"] if s.get("template") == rs.TEMPLATE), {})
    checks.append(("restyle NB macro on dev (diagnostic; blocked)", screen.get("naive_bayes_macro"),
                   reported_screen.get("naive_bayes_macro")))
    info.append(f"INFO restyle NB: train rows {screen.get('train_rows')}, dev rows {screen.get('dev_rows')}, "
                f"majority-class dev accuracy {screen.get('majority_dev_accuracy')}")

    page_texts = {cls: [r["state"] for r in balanced["train"] if r["meta"]["page_class"] == cls] for cls in rs.RESTYLE_CLASSES}
    page_exempt = {w for label in rs.PAGE_LABEL.values() for w in re.findall(r"[a-z]+", label.lower())}
    phrases = asm.phrase_check(page_texts, page_exempt)
    checks.append(("restyle phrase flags (train)", phrases["flagged_count"], build["checks"]["phrase_pages"]["flagged_count"]))
    for split in SPLITS:
        checks.append((f"restyle gold_in_state {split}", sum(gold_in_state(r) for r in balanced[split]),
                       build["checks"]["gold_in_state"][split]))
    every = [r for split in SPLITS for r in balanced[split]]
    text, bare, qualified = load_protected(REPO)
    checks.append(("leakage hits, balanced rows", hit_total(leakage_counts(every, text, bare, qualified)),
                   hit_total(build["checks"]["leakage_all_rows"])))
    checks.append(("paragraph hits in protected states", sum(
        1 for r in every for p in paragraphs(r["state"]) if len(p) >= 60 and paragraph_key(p) in protected),
        build["checks"]["paragraph_hits_in_protected_states"]))

    lines = [f"x3 recount {utcnow()} (fresh process; candidates in data/gen/osler_v3; compared with build.json)"]
    failed = 0
    for name, got, want in checks:
        ok = got == want
        failed += 0 if ok else 1
        lines.append(f"{'OK' if ok else 'MISMATCH'} {name}: recount {got} | build.json {want}")
    lines.extend(info)
    lines.append(f"checks {len(checks)}, agree {len(checks) - failed}, mismatch {failed}")
    (OUT / "recount.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
