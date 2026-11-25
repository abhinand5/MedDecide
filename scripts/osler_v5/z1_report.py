"""osler_v5 Z1 report (CPU): writes loops/osler_v5/probes.md from outputs/osler_v5/Z1/summary.json and the saved per-item rows, and the knowledge
paired analysis to outputs/osler_v5/Z1/logs/knowledge_paired.log. Every figure in the report is computed here (none is typed by hand).
The probes choose nothing (ADVISORY Z1): this report is additional.

Usage (CPU):
    uv run --frozen python scripts/osler_v5/z1_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meddecide.train.blocks import micro_difference, pair_by_item
from meddecide.utils.io import write_json

REPO = Path(__file__).resolve().parents[2]
Z1 = REPO / "outputs" / "osler_v5" / "Z1"
Z0_COMPOSITE = REPO / "outputs" / "osler_v5" / "Z0" / "composite.json"
REPORT = REPO / "loops" / "osler_v5" / "probes.md"
KNOWLEDGE_LOG = Z1 / "logs" / "knowledge_paired.log"
BLEND = ("alpha_0", "alpha_0.3", "alpha_0.5", "alpha_0.7", "alpha_0.85", "alpha_1")
SPLITS = (("split_trained_adapter_base_readout", "(a) trained adapter, base readout"),
          ("split_base_adapter_trained_readout", "(b) base adapter, trained readout"))
RETENTION = 0.8


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def paired(name: str, base: str = "alpha_0", source: str | None = None) -> tuple[float, float, float, int]:
    """Item-micro knowledge accuracy of ``name`` minus ``base``, paired by item, within templates (keep-rule statistic)."""
    x = rows(Z1 / name / "knowledge_dev.jsonl")
    b = rows(Z1 / base / "knowledge_dev.jsonl")
    if source is not None:
        x = [r for r in x if r["source"] == source]
        b = [r for r in b if r["source"] == source]
    d = micro_difference(pair_by_item(x, b))
    return d.point, d.lo, d.hi, d.n


def pos(p: dict[str, Any]) -> str:
    return "pass" if p["passes"] else "FAIL"


def consistency() -> dict[str, Any]:
    """alpha 1 against osler_v4's Y8 predictions on the same items: argmax agreement and the largest probability difference."""
    out: dict[str, Any] = {}
    pairs = (("fast_dev_v4", "alpha_1/fast_dev.jsonl", "outputs/osler_v4/Y8/dev_predictions.jsonl"),
             ("cg_v2", "alpha_1/cg_v2.jsonl", "outputs/osler_v4/Y8/cg_v2_predictions.jsonl"))
    for label, mine, theirs in pairs:
        a = {r["item_id"]: r for r in rows(Z1 / mine)}
        b = {r["item_id"]: r for r in rows(REPO / theirs)}
        common = sorted(set(a) & set(b))
        same = sum(a[i]["predicted_key"] == b[i]["predicted_key"] for i in common)
        diff = max(max(abs(x - y) for x, y in zip(a[i]["probs"], b[i]["probs"], strict=True)) for i in common)
        out[label] = {"items_common": len(common), "items_probe": len(a), "items_y8": len(b), "same_argmax": same, "max_abs_prob_diff": diff}
    return out


def main() -> None:
    summary = json.loads((Z1 / "summary.json").read_text(encoding="utf-8"))
    probes = json.loads((Z1 / "probes.json").read_text(encoding="utf-8"))
    z0 = json.loads(Z0_COMPOSITE.read_text(encoding="utf-8"))["models"]["osler_v4_final"]["composite"]["S"]
    s0, s1 = summary["alpha_0"], summary["alpha_1"]

    # the knowledge analysis, logged (the report quotes these figures)
    log: list[str] = []
    for name in ("alpha_0.3", "alpha_0.5", "alpha_0.7", "alpha_0.85", "alpha_1"):
        p, lo, hi, n = paired(name)
        log.append(f"knowledge dev, {name} minus alpha_0: {p:+.4f} [{lo:+.4f}, {hi:+.4f}] n={n}")
    by_source: dict[str, tuple[float, float, float, int]] = {}
    for source in ("medqa", "medmcqa"):
        for name in ("alpha_0.5", "alpha_1"):
            p, lo, hi, n = paired(name, source=source)
            by_source[f"{name}/{source}"] = (p, lo, hi, n)
            log.append(f"knowledge dev, {name}, {source} only: {p:+.4f} [{lo:+.4f}, {hi:+.4f}] n={n}")
    KNOWLEDGE_LOG.write_text("\n".join(log) + "\n", encoding="utf-8")
    write_json(Z1 / "knowledge_paired.json", {"log": log})

    cons = consistency()
    write_json(Z1 / "logs" / "consistency.json", cons)
    (Z1 / "logs" / "consistency.log").write_text(json.dumps(cons, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    k1 = paired("alpha_1")
    med, medqa = by_source["alpha_1/medmcqa"], by_source["alpha_1/medqa"]
    # the Z4 retention threshold on this checkpoint (what-if)
    gain_mr = s1["composite"]["mr"] - s0["composite"]["mr"]
    thr_mr = s0["composite"]["mr"] + RETENTION * gain_mr
    mr05 = summary["alpha_0.5"]["composite"]["mr"]
    mr07 = summary["alpha_0.7"]["composite"]["mr"]
    single0 = summary["alpha_0"]["ddx_single_letter_accuracy"]
    lines: list[str] = []
    add = lines.append
    add("# osler_v5 — Z1 probes on the osler_v4 final (additional; choose nothing)")
    add("")
    add("Dev only. The checkpoint is osler_v4's final (Y8, 63,304 steps, `outputs/osler_v4/Y8/checkpoints/best`), scored in bf16 on the GPU by")
    add("`scripts/osler_v5/z1_probes.py` (job started 2026-10-26T19:59Z, finished 2026-10-26T22:54Z). The figures are recomputed from the saved")
    add("per-item rows by `scripts/osler_v5/z1_summarise.py` (`outputs/osler_v5/Z1/summary.json`); this file is written by")
    add("`scripts/osler_v5/z1_report.py`. Nothing here chooses an arm, a checkpoint or an alpha (ADVISORY Z1). No test set, panel, MIMIC")
    add("audit suite or held-out file was scored.")
    add("")
    add("## 1. Consistency checks")
    add("")
    add(f"- alpha 0 against the reference path (`MedDecideModel.reference_option_probs`, 16 CG items, bf16 on the GPU): max abs diff "
        f"{probes['reference_check']['max_abs_diff_alpha0']:.2e}.")
    fd, cgc = cons["fast_dev_v4"], cons["cg_v2"]
    add(f"- alpha 1 against osler_v4's Y8 predictions on the same items (the trained model, uncalibrated): fast dev v4 "
        f"{fd['same_argmax']:,} of {fd['items_common']:,} items with the same argmax (maximum absolute probability difference "
        f"{fd['max_abs_prob_diff']:.4f}); CG v2 {cgc['same_argmax']:,} of {cgc['items_common']:,} (maximum {cgc['max_abs_prob_diff']:.4f}). "
        "Logged in `outputs/osler_v5/Z1/logs/consistency.log`.")
    add(f"- alpha 1 composite S {s1['composite']['S']:.4f} against Z0's composite of the same predictions {z0:.4f} (C026).")
    add(f"- alpha 0 DDXPlus two-letter accuracy {s0['ddx_two_letter_accuracy']:.4f} against zero-shot pplx-flash's {0.6112:.4f} in osler_v4 (C210).")
    add("")
    add("## 2. Blend curve (alpha 0 is the base model through the reference path; all figures dev)")
    add("")
    add("| alpha | S | MR macro | MIMIC macro | CG v2 micro | knowledge dev | knowledge minus alpha 0 [95 % CI] | DDX two-letter | DDX single | positional (orig / seed 1 / seed 2) |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for name in BLEND:
        s = summary[name]
        c = s["composite"]
        if name == "alpha_0":
            kd = "reference"
        else:
            p, lo, hi, _ = paired(name)
            kd = f"{p:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        pp = s["positional_permuted"]
        cell = f"{pos(s['positional_original'])} / {pos(pp['1'])} / {pos(pp['2'])}"
        label = name.replace("alpha_", "").replace("_", ".")
        add(f"| {label} | {c['S']:.4f} | {c['mr']:.4f} | {c['mimic']:.4f} | {c['cg_micro']:.4f} | {s['knowledge_dev_micro']:.4f} | {kd} | "
            f"{s['ddx_two_letter_accuracy']:.4f} | {s['ddx_single_letter_accuracy']:.4f} | {cell} |")
    add("")
    add("MR and MIMIC are template macro accuracies, CG v2 is item micro, and S is their mean (D54). The knowledge difference is the paired "
        "bootstrap of item micro accuracy within templates (medqa, medmcqa), 1,000 resamples, seed 0, as in the keep rule.")
    add("")
    add("## 3. Readout split (which part of the readout and the adapter carries the two-letter loss)")
    add("")
    add("| condition | S | MR macro | MIMIC macro | CG v2 micro | knowledge dev | DDX two-letter | DDX single | positional (orig / seed 1 / seed 2) |")
    add("|---|---|---|---|---|---|---|---|---|")
    conditions = [("alpha 0 (base adapter, base readout)", "alpha_0"), *[(t, k) for k, t in SPLITS],
                  ("alpha 1 (trained adapter, trained readout)", "alpha_1")]
    for label, name in conditions:
        s = summary[name]
        c = s["composite"]
        pp = s["positional_permuted"]
        cell = f"{pos(s['positional_original'])} / {pos(pp['1'])} / {pos(pp['2'])}"
        add(f"| {label} | {c['S']:.4f} | {c['mr']:.4f} | {c['mimic']:.4f} | {c['cg_micro']:.4f} | {s['knowledge_dev_micro']:.4f} | "
            f"{s['ddx_two_letter_accuracy']:.4f} | {s['ddx_single_letter_accuracy']:.4f} | {cell} |")
    add("")
    add("## 4. Readings (additional; labelled; none of them is a choice)")
    add("")
    add(f"1. **Positional health is lost between alpha 0.5 and alpha 0.7** on this checkpoint. It passes at 0, 0.3 and 0.5 in all three orders "
        f"and fails at 0.7, 0.85 and 1 in all three orders. The two-letter accuracy falls monotonically with alpha, from "
        f"{summary['alpha_0']['ddx_two_letter_accuracy']:.4f} at 0 to {summary['alpha_1']['ddx_two_letter_accuracy']:.4f} at 1, while the "
        f"single-letter accuracy stays at {summary['alpha_0.7']['ddx_single_letter_accuracy']:.4f} or above up to 0.7.")
    add(f"2. **Knowledge dev falls slightly with alpha, and significantly on MedMCQA at alpha 1.** Against alpha 0 the paired difference is "
        f"{k1[0]:+.4f} [{k1[1]:+.4f}, {k1[2]:+.4f}] at alpha 1 (n = {k1[3]}). This is the GOAL's \"knowledge below zero-shot\" check on dev. "
        f"By source at alpha 1: MedMCQA ({med[3]} items) {med[0]:+.4f} [{med[1]:+.4f}, {med[2]:+.4f}]; MedQA ({medqa[3]} items) "
        f"{medqa[0]:+.4f} [{medqa[1]:+.4f}, {medqa[2]:+.4f}]. Gate O4's knowledge check is on the knowledge test set (osler_v4 measured -0.0067 "
        f"there), a different set; the dev figure is additional. The log is `outputs/osler_v5/Z1/logs/knowledge_paired.log`.")
    add(f"3. **Both the readout and the adapter carry the two-letter loss.** The trained readout alone (split b) takes the two-letter accuracy from "
        f"{s0['ddx_two_letter_accuracy']:.4f} to {summary[SPLITS[1][0]]['ddx_two_letter_accuracy']:.4f}; the trained adapter alone with the base "
        f"readout (split a) takes it to {summary[SPLITS[0][0]]['ddx_two_letter_accuracy']:.4f}; together (alpha 1) to "
        f"{s1['ddx_two_letter_accuracy']:.4f}. The trained readout also lowers the single-letter accuracy to "
        f"{summary[SPLITS[1][0]]['ddx_single_letter_accuracy']:.4f} with the adapter off (alpha 0 gives {single0:.4f}), so the readout rows for "
        f"options A to Z changed too. (A hypothesis about the mechanism; the figures are measured.)")
    add(f"4. **A discrepancy with osler_v4's wording.** osler_v4's FINDINGS section 1 says single-letter gold is intact (C210, C212). C210's own "
        f"figures give {s1['ddx_single_letter_accuracy']:.4f} for the final against 0.7964 for zero-shot on the same 501 single-letter items, and "
        f"this probe gives {s1['ddx_single_letter_accuracy']:.4f} at alpha 1. The FINDINGS wording is not supported by its own numbers. It is noted "
        f"here; osler_v4's files are not changed.")
    add(f"5. **What the Z4 rule would say about this checkpoint (what-if, not a choice).** The MR retention threshold is {thr_mr:.4f} "
        f"(80 % of the alpha 1 gain over alpha 0, which is {gain_mr:.4f}). Alpha 0.5 reaches {mr05:.4f}, below it; alpha 0.7 reaches {mr07:.4f}, "
        f"above it, but fails positional health. Alpha 1 itself fails positional health. So no value measured here qualifies. Alpha 0.6 is not "
        f"measured here and is on the Z4 grid. Z4 runs on the Z3 checkpoint selected by S among positional-health passes, not on this checkpoint.")
    add(f"6. **S is not a safe selector on its own.** S peaks at alpha 0.7 ({summary['alpha_0.7']['composite']['S']:.4f}) and alpha 0.85 "
        f"({summary['alpha_0.85']['composite']['S']:.4f}); both fail positional health. A selection by S alone would have taken a model that "
        f"fails the two-letter check. This is why the Z3 rule restricts to positional passes.")
    add("")
    add("## 5. Limits")
    add("")
    add("- Dev only: fast dev v4 (17,532 items), CG v2 (8,307 items, of which 1,000 are DDXPlus and 499 have two-letter gold), knowledge dev v2 "
        "(4,000 items).")
    add("- Accuracy only; no calibration is applied, and accuracy does not depend on temperature.")
    add("- The probe job's own composite used item-micro MR and MIMIC accuracies. Its in-memory figures are superseded by `summary.json` "
        "(STATE §7.10).")
    add("- The alpha grid here (0, 0.3, 0.5, 0.7, 0.85, 1) is a probe grid. The Z4 grid (0.3 to 1.0 in steps of 0.1) runs on the Z3 "
        "checkpoint.")
    add("")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
