"""osler_v4 Y9 (R6 self-audit; CPU): the Gate O4 point estimates re-derived from the raw prediction files in a fresh process, without
the gate library. Accuracy = the template macro of the paired differences (final minus baseline), paired by item id; a set's items
that the baseline did not score are dropped from the comparison, as the pre-registration's pairing rule states (D44(a)). Compared
with the point estimates in outputs/osler_v4/Y9/gate.json to 1e-4. The bootstrap intervals are not re-derived here. Prints counts and
figures only. Writes outputs/osler_v4/Y9/rederive.json.

Usage (CPU):
    CUDA_VISIBLE_DEVICES= uv run --frozen python scripts/osler_v4/y9_rederive.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from meddecide.utils.io import write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "outputs" / "osler_v4" / "Y9" / "baselines"
GATE = REPO / "outputs" / "osler_v4" / "Y9" / "gate.json"
OUT = REPO / "outputs" / "osler_v4" / "Y9" / "rederive.json"
TOLERANCE = 1e-4
# (check, set, final run, baseline run, the gate.json key of the point estimate, baseline key)
CHECKS = (
    ("i mr_verdict vs pplx_flash", "mr_verdict", "final_mr_verdict", "pplx_mr_verdict", "pplx_flash"),
    ("iii panel vs pplx_flash", "panel", "final_panel", "pplx_panel", "pplx_flash"),
    ("iv knowledge vs pplx_flash", "knowledge", "final_knowledge", "pplx_knowledge", "pplx_flash"),
    ("i mr_verdict vs meddecider_4b", "mr_verdict", "final_mr_verdict", "md4b_mr_verdict", "meddecider_4b"),
    ("iv knowledge vs meddecider_4b", "knowledge", "final_knowledge", "md4b_knowledge", "meddecider_4b"),
)
GATE_KEY = {"i mr_verdict vs pplx_flash": "i", "iii panel vs pplx_flash": "iii", "iv knowledge vs pplx_flash": "iv",
            "i mr_verdict vs meddecider_4b": "i", "iv knowledge vs meddecider_4b": "iv"}


def load(run: str) -> dict[str, dict]:
    path = RUNS / run / "predictions.jsonl"
    if not path.exists():  # the jev and 27B runs use other names; not needed here
        return {}
    with path.open(encoding="utf-8") as fh:
        return {str(r["item_id"]): r for r in (json.loads(line) for line in fh if line.strip())}


def macro_difference(final: dict[str, dict], base: dict[str, dict]) -> tuple[float, int, int]:
    """Template macro of the paired differences; returns (figure, paired items, templates)."""
    per: dict[str, list[float]] = collections.defaultdict(list)
    for item_id, row in final.items():
        other = base.get(item_id)
        if other is None:
            continue
        per[str(row["template_id"])].append(float(bool(row["correct"])) - float(bool(other["correct"])))
    paired = sum(len(v) for v in per.values())
    macro = sum(sum(v) / len(v) for v in per.values()) / len(per)
    return macro, paired, len(per)


def main() -> None:
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    results: dict[str, object] = {"generated_utc": utcnow(), "checks": {}}
    mismatches = []
    for name, _set_name, final_run, base_run, base_key in CHECKS:
        final, base = load(final_run), load(base_run)
        if not final or not base:
            results["checks"][name] = {"status": "NOT RUN"}
            continue
        figure, paired, templates = macro_difference(final, base)
        recorded = gate["gate"][base_key]["checks"][GATE_KEY[name]]["accuracy"]["point"]
        ok = abs(figure - recorded) <= TOLERANCE
        if not ok:
            mismatches.append(name)
        results["checks"][name] = {"paired_items": paired, "templates": templates, "macro_difference": round(figure, 6),
                                   "gate_point": round(recorded, 6), "match": ok}
        print(f"{name}: paired {paired}, templates {templates}, macro difference {figure:+.6f} vs gate {recorded:+.6f} "
              f"({'match' if ok else 'MISMATCH'})", flush=True)
    results["mismatches"] = mismatches
    write_json(OUT, results)
    print(f"fresh-process re-derivation: {len(CHECKS)} checks, {len(mismatches)} mismatches", flush=True)
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
