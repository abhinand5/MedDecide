"""Self-audit of V5a, the Synthea items (osler_v1 ADVISORY V5a; R6).

Re-reads data/gen/osler_v1_synthea/{train,dev,test,heldout_test}.jsonl in a fresh process and checks:
- the splits (each row's split and its patient's hash split), no patient in two splits;
- the held-out indicator (aki_creatinine_rise) appears only in heldout_test;
- yes and no balance per indicator in each split (the build down-samples the majority in train; dev and test are reported);
- the O3 screen on dev for each template (meddecide.gen.screen), with the same threshold as V1;
- chance and majority on dev, and the page-class counts of the page-category template.

Writes outputs/osler_v1/V5a/screen.json and outputs/osler_v1/V5a/SELF_AUDIT.md (aggregates only).

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v5a_audit.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.gen.core import split_of
from meddecide.gen.screen import screen_generator
from meddecide.mr.generators import HELD_OUT_INDICATORS
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "gen" / "osler_v1_synthea"
OUT_DIR = REPO / "outputs" / "osler_v1" / "V5a"
SPLITS = ("train", "dev", "test", "heldout_test")


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def main() -> None:
    rows: dict[str, list[dict[str, Any]]] = {s: list(iter_rows(DATA / f"{s}.jsonl")) for s in SPLITS}
    report: dict[str, Any] = {"written_utc": utcnow(), "git_commit": git_commit(REPO), "templates": {}}
    problems: Counter[str] = Counter()
    patients: dict[str, set[str]] = {s: {r["patient_id"] for r in rows[s]} for s in SPLITS}
    report["patients_per_split"] = {s: len(patients[s]) for s in SPLITS}
    report["patients_in_two_splits"] = sum(len(patients[a] & patients[b]) for i, a in enumerate(SPLITS)
                                           for b in SPLITS[i + 1:])
    # training integrity: train shares no patient with any other split; dev and test share none with each other.
    # The held-out pool (heldout_test) shares patients with dev and test by design (V5a-D2): it is evaluated only, and no
    # training item belongs to one of its patients.
    report["patients_train_vs_other_splits"] = len(patients["train"] & (patients["dev"] | patients["test"]
                                                                        | patients["heldout_test"]))
    report["patients_dev_vs_test"] = len(patients["dev"] & patients["test"])
    report["patients_heldout_vs_dev_or_test"] = len(patients["heldout_test"] & (patients["dev"] | patients["test"]))
    for s in SPLITS:
        for r in rows[s]:
            if split_of(r["patient_id"]) != r["split"]:
                problems["split disagrees with the patient hash"] += 1
            if s != "heldout_test" and r["meta"].get("indicator") in HELD_OUT_INDICATORS:
                problems["held-out indicator in a training split"] += 1
            if s == "heldout_test" and r["meta"].get("indicator") not in HELD_OUT_INDICATORS and r["generator"] == "synthea_indicator_evidence_v1":
                problems["non-held-out indicator in heldout_test"] += 1
    report["problems"] = dict(problems)

    screen_rows: dict[str, Any] = {}
    for template in ("synthea_page_category_v1", "synthea_indicator_evidence_v1"):
        per_split = {s: [r for r in rows[s] if r["generator"] == template] for s in SPLITS}
        info: dict[str, Any] = {"rows": {s: len(per_split[s]) for s in SPLITS}}
        balance: dict[str, dict[str, dict[str, int]]] = {}
        for s in ("train", "dev", "test", "heldout_test"):
            if template == "synthea_indicator_evidence_v1":
                table: dict[str, Counter[str]] = defaultdict(Counter)
                for r in per_split[s]:
                    table[r["meta"]["indicator"]][r["gold"]] += 1
                balance[s] = {k: dict(v) for k, v in sorted(table.items())}
        if balance:
            info["balance_by_indicator"] = balance
        dev = per_split["dev"]
        gold_counts = Counter(r["gold"] for r in dev)
        info["dev_majority_gold_share"] = round(max(gold_counts.values()) / len(dev), 4) if dev else None
        info["dev_uniform_chance"] = round(sum(1.0 / len(r["options"]) for r in dev) / len(dev), 4) if dev else None
        if template == "synthea_page_category_v1":
            info["page_class_counts"] = {s: dict(sorted(Counter(r["meta"]["page_class"] for r in per_split[s]).items()))
                                         for s in ("train", "dev", "test")}
        rep = screen_generator(template, False, dev, per_split["train"], "own train split")
        info["screen"] = {"string_presence_macro": rep.string_presence_macro,
                          "naive_bayes_macro": rep.naive_bayes_macro, "gold_in_state_hits": rep.gold_in_state_hits,
                          "passes": rep.passes, "failures": rep.failures}
        report["templates"][template] = info
        screen_rows[template] = info["screen"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "screen.json", {"written_utc": utcnow(), "reports": screen_rows})
    write_json(OUT_DIR / "audit.json", report)
    lines = ["# V5a self-audit (osler_v1 ADVISORY V5a; R6)", "",
             "Re-derived from data/gen/osler_v1_synthea in a fresh process by `scripts/osler_v1/v5a_audit.py`.", "",
             f"Patients per split: {report['patients_per_split']}.",
             f"Train patients shared with any other split: {report['patients_train_vs_other_splits']} (must be 0); "
             f"dev and test shared: {report['patients_dev_vs_test']} (must be 0); held-out pool shared with dev or test: "
             f"{report['patients_heldout_vs_dev_or_test']} (documented, V5a-D2; never used for training).",
             f"Problems found (counts): {report['problems'] or 'none'}.", "",
             "| template | rows train / dev / test / heldout | dev uniform chance | dev majority gold share | O3 screen (dev) |",
             "|---|---|---|---|---|"]
    for name, info in report["templates"].items():
        r = info["rows"]
        sc = info["screen"]
        verdict = "PASS" if sc["passes"] else "FAIL: " + "; ".join(sc["failures"])
        lines.append(f"| {name} | {r['train']} / {r['dev']} / {r['test']} / {r['heldout_test']} | "
                     f"{info['dev_uniform_chance']} | {info['dev_majority_gold_share']} | {verdict} |")
    (OUT_DIR / "SELF_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"patients_in_two_splits": report["patients_in_two_splits"], "problems": report["problems"],
                      "screens": screen_rows}, indent=1))


if __name__ == "__main__":
    main()
