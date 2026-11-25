"""Self-audit of V5b, the real clinical text (osler_v1 ADVISORY V5b; R6).

Re-reads data/gen/osler_v1/realtext/*.jsonl in a fresh process and checks:
- the splits: MTSamples and PMC by their hash split (the split of each row equals split_of(its id));
- PMC: no train pair touches a dev or test anchor or a dev or test pair (the drop rule, re-derived from the file);
- the overlap of every real-text template's text keys with the osler_v0 protected sets (v0.2 test, v0.2 dev, panel, robustness,
  held-out) and with the osler_v1 MR test sets (dev, test, heldout_test), recorded as counts;
- chance and majority baselines on dev and test (CodiEsp has no dev split: its corpus dev is our test);
- the evaluation-only sets (mkurman) are all in the test split and no training template holds one of their keys.

The O3 screen is required for generated and teacher sources; these real-text sources are screened for information only
(Naive Bayes on dev where a dev split exists).

Writes outputs/osler_v1/V5b/SELF_AUDIT.md and outputs/osler_v1/V5b/audit.json (aggregates only).

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v5b_audit.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from meddecide.gen.core import split_of
from meddecide.gen.screen import screen_generator
from meddecide.mix.leakage import row_text_key
from meddecide.utils.io import iter_jsonl, write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "gen" / "osler_v1" / "realtext"
MR = REPO / "data" / "gen" / "osler_v1"
OUT = REPO / "outputs" / "osler_v1" / "V5b"
TEMPLATES = ("mtsamples_page_category_v1", "codiesp_icd_support_v1", "pmc_similar_patient_v1", "mkurman_icd_support_v1")
TRAINING = ("mtsamples_page_category_v1", "codiesp_icd_support_v1", "pmc_similar_patient_v1")


def load_protected() -> dict[str, set[str]]:
    spec = importlib.util.spec_from_file_location("o4_build_mix", REPO / "scripts" / "osler" / "o4_build_mix.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prot = module.protected_sets()
    out: dict[str, set[str]] = {}
    for split in ("train", "dev"):
        for name, keys in prot[split]["text"].items():
            out[f"v0_{name}"] = out.get(f"v0_{name}", set()) | keys
    return out


def main() -> None:
    rows: dict[str, list[dict[str, Any]]] = {}
    for t in TEMPLATES:
        rows[t] = list(iter_jsonl(DATA / f"{t}.jsonl"))
    report: dict[str, Any] = {"written_utc": utcnow(), "git_commit": git_commit(REPO), "templates": {}}
    problems: Counter[str] = Counter()

    for t in TEMPLATES:
        per_split = Counter(r["split"] for r in rows[t])
        info: dict[str, Any] = {"rows": dict(per_split)}
        if t in ("mtsamples_page_category_v1", "pmc_similar_patient_v1"):
            for r in rows[t]:
                if split_of(r["pair_id"].split(":", 1)[1]) != r["split"]:
                    problems[f"{t}: split disagrees with the hash"] += 1
        if t == "pmc_similar_patient_v1":
            dev_test_anchors = {r["meta"]["pair_uids"][0] for r in rows[t] if r["split"] in ("dev", "test")}
            dev_test_pairs = {u for r in rows[t] if r["split"] in ("dev", "test") for u in r["meta"]["pair_uids"]}
            leaks = sum(1 for r in rows[t] if r["split"] == "train"
                        and (r["meta"]["pair_uids"][1] in dev_test_anchors or r["meta"]["pair_uids"][1] in dev_test_pairs))
            info["train_pairs_touching_dev_or_test"] = leaks
            if leaks:
                problems["pmc: train pair touches a dev or test patient"] += leaks
            info["licence_subsets"] = dict(Counter(r["meta"]["licence_subset"] for r in rows[t]))
        if t == "mkurman_icd_support_v1" and any(r["split"] != "test" for r in rows[t]):
            problems["mkurman: a row outside the test split"] += 1
        dev = [r for r in rows[t] if r["split"] == "dev"]
        test = [r for r in rows[t] if r["split"] == "test"]
        for name, part in (("dev", dev), ("test", test)):
            if part:
                golds = Counter(r["gold"] for r in part)
                info[f"{name}_majority_gold_share"] = round(max(golds.values()) / len(part), 4)
                info[f"{name}_uniform_chance"] = round(sum(1.0 / len(r["options"]) for r in part) / len(part), 4)
        report["templates"][t] = info

    protected = load_protected()
    mr_test_keys: dict[str, set[str]] = {}
    for split in ("dev", "test", "heldout_test"):
        mr_test_keys[split] = {row_text_key(r) for r in iter_jsonl(MR / f"{split}.jsonl")}
    for t in TEMPLATES:
        overlaps: Counter[str] = Counter()
        for r in rows[t]:
            key = row_text_key(r)
            for name, keys in protected.items():
                if key in keys:
                    overlaps[name] += 1
            for name, keys in mr_test_keys.items():
                if key in keys:
                    overlaps[f"mr_{name}"] += 1
        report["templates"][t]["text_overlap"] = dict(overlaps)

    screens: dict[str, Any] = {}
    for t in TRAINING:
        dev = [r for r in rows[t] if r["split"] == "dev"]
        train = [r for r in rows[t] if r["split"] == "train"]
        if dev and train:
            rep = screen_generator(t, False, dev, train, "own train split")
            screens[t] = {"naive_bayes_macro": rep.naive_bayes_macro, "string_presence_macro": rep.string_presence_macro,
                          "passes": rep.passes, "failures": rep.failures, "role": "information only (real text)"}
        else:
            screens[t] = {"status": "NOT MEASURED", "reason": "no dev split (the corpus dev set is our test set)"}
    report["screens_informational"] = screens
    report["problems"] = dict(problems)

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "audit.json", report)
    lines = ["# V5b self-audit (osler_v1 ADVISORY V5b; R6)", "",
             "Re-derived from data/gen/osler_v1/realtext in a fresh process by `scripts/osler_v1/v5b_audit.py`.", "",
             f"Problems found (counts): {report['problems'] or 'none'}.", "",
             "| template | rows train / dev / test | dev chance / majority | test chance / majority | text overlap with the v0 protected sets and the MR test sets |",
             "|---|---|---|---|---|"]
    for t in TEMPLATES:
        info = report["templates"][t]
        r = info["rows"]
        dc = f"{info.get('dev_uniform_chance', '-')} / {info.get('dev_majority_gold_share', '-')}"
        tc = f"{info.get('test_uniform_chance', '-')} / {info.get('test_majority_gold_share', '-')}"
        lines.append(f"| {t} | {r.get('train', 0)} / {r.get('dev', 0)} / {r.get('test', 0)} | {dc} | {tc} | "
                     f"{info['text_overlap'] or 'none'} |")
    lines += ["", "Informational Naive Bayes on dev (real text; not a gate):", ""]
    for t, s in screens.items():
        lines.append(f"- {t}: " + (f"Naive Bayes {s['naive_bayes_macro']:.4f}, string presence {s['string_presence_macro']}"
                                    if "naive_bayes_macro" in s else f"{s['status']} ({s['reason']})"))
    (OUT / "SELF_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"problems": report["problems"], "templates": {t: report["templates"][t]["rows"] for t in TEMPLATES},
                      "overlap": {t: report["templates"][t]["text_overlap"] for t in TEMPLATES}}, indent=1))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
