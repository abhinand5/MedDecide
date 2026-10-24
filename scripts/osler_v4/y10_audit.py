"""Y10 (osler_v4 ADVISORY Y10): the memorisation and PHI audit of the final Osler-4B. MIMIC is in the final mix, so this audit is required.
It is a report for the release decision, not a gate (ADVISORY Y10).

1. Membership. Members: MIMIC training rows of the final mix (composed in memory as in Y8), sampled per template by stable-hash rank.
   Non-members: the frozen MIMIC audit suite v1 (late-anchor test items of the same templates; no training overlap, STATE §2), sampled
   per template in proportion to the suite's template counts. Both sides total 2,000 items. The per-item loss is -log p(gold) from the
   calibrated probabilities. The AUC is the Mann-Whitney membership score (0.5 = no measurable signal), with a bootstrap interval. The
   zero-shot base (pplx-decider-flash-v1.2) is the control on the same items. The test items are later admissions than most training
   items, so the AUC also reads any train-to-test shift; the control shows how much of it is shift alone.
2. Verbatim probe. 300 train-split discharge notes; the first 128 tokens are the prefix and the next 64 the truth. 64 tokens are generated
   greedily by the exported plain causal LM of the final model (D23: the option-code rows written back into a copy of the LM head, the
   adapter merged) and by the zero-shot base. A continuation counts when it matches the truth on at least 50 consecutive aligned tokens.

Only counts and figures are written under outputs/osler_v4/Y10 and loops/osler_v4/audit.md. Item and note texts stay in data/ and are
never printed. Subcommands (run in this order; the scoring steps use one GPU job at a time):

    uv run --frozen python scripts/osler_v4/y10_audit.py members          # CPU: the 4,000 items and their sampling counts
    CUDA_VISIBLE_DEVICES=  (see the chain in outputs/osler_v4/Y10/logs)    # GPU: y4_checkpoint_score.py and y4_pplx_flash.py on the items
    uv run --frozen python scripts/osler_v4/y10_audit.py auc              # CPU: the membership AUC and its control
    uv run --frozen python scripts/osler_v4/y10_audit.py probe            # GPU: the verbatim probe (final and base)
    uv run --frozen python scripts/osler_v4/y10_audit.py report           # CPU: loops/osler_v4/audit.md
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from meddecide.eval.audit_y10 import (
    VERBATIM_RUN,
    longest_aligned_run,
    membership_auc,
    membership_auc_interval,
)
from meddecide.eval.gate_o4 import option_index
from meddecide.train.final_mix import compose_final_mix
from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.hashing import stable_hash
from meddecide.utils.io import iter_jsonl, read_json, write_json, write_jsonl
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
Y8_BEST = REPO / "outputs" / "osler_v4" / "Y8" / "checkpoints" / "best"
Y10 = REPO / "outputs" / "osler_v4" / "Y10"
SUITE = REPO / "data" / "bench" / "osler_mimic_v1" / "mimic_audit_suite_v1.jsonl"
ITEMS_DIR = REPO / "data" / "bench" / "osler_v4" / "y10"
ITEMS = ITEMS_DIR / "membership.jsonl"
LABELS = ITEMS_DIR / "membership_labels.json"
NOTES = REPO / "data" / "mimic_build" / "discharge_notes.parquet"
REPORT = REPO / "loops" / "osler_v4" / "audit.md"
PER_CLASS = 2000
PROBE_NOTES = 300
PREFIX_TOKENS = 128
CONTINUATION_TOKENS = 64
FAMILY = ("B1", "B2", "B3")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    """Proportional allocation of ``total`` over the counts, by largest remainder, so the parts sum to the total exactly."""
    grand = sum(counts.values())
    raw = {k: total * v / grand for k, v in counts.items()}
    base = {k: int(x) for k, x in raw.items()}
    short = total - sum(base.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - base[k]), k))[:short]:
        base[k] += 1
    return base


def members_and_nonmembers() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """The 2,000 member rows (the final mix's MIMIC rows, by template) and the 2,000 suite rows, with the sampling counts."""
    y8_mix = load_module("y8_final_mix", REPO / "scripts" / "osler_v4" / "y8_final_mix.py")
    decisions = {name: y8_mix.entering(name) for name in ("B1", "B2", "B3", "B4")}
    family = {name: y8_mix.add_files(name) for name in FAMILY if decisions[name][0]}
    other = y8_mix.add_files("B4") if decisions["B4"][0] else []
    rows, report = compose_final_mix(y8_mix.BASE, family, other, y8_mix.BROAD)
    suite = list(iter_jsonl(SUITE))
    suite_counts = Counter(str(r["template_id"]) for r in suite)
    templates = sorted(suite_counts)
    alloc = allocate({t: suite_counts[t] for t in templates}, PER_CLASS)
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if str(r["template_id"]) in alloc:
            pools[str(r["template_id"])].append(r)
    members: list[dict[str, Any]] = []
    for t in templates:
        ranked = sorted(pools[t], key=lambda r: stable_hash({"member": str(r["item_id"])}, length=16))
        if len(ranked) < alloc[t]:
            raise SystemExit(f"{t}: {len(ranked)} member rows in the final mix, {alloc[t]} needed")
        members.extend(ranked[: alloc[t]])
    nonmembers: list[dict[str, Any]] = []
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in suite:
        by_template[str(r["template_id"])].append(r)
    for t in templates:
        ranked = sorted(by_template[t], key=lambda r: stable_hash({"nonmember": str(r["item_id"])}, length=16))
        nonmembers.extend(ranked[: alloc[t]])
    sampling = {"final_mix_rows": report["rows"], "composition_sha256": report["composition_sha256"], "allocation": alloc,
                "suite_counts": dict(suite_counts),
                "member_pool_sizes": {t: len(pools[t]) for t in templates}, "members": len(members),
                "nonmembers": len(nonmembers), "member_templates": dict(Counter(str(r["template_id"]) for r in members))}
    return members, nonmembers, sampling


def cmd_members(_: argparse.Namespace) -> None:
    members, nonmembers, sampling = members_and_nonmembers()
    ITEMS_DIR.mkdir(parents=True, exist_ok=True)
    labels: dict[str, bool] = {}
    for r in members:
        labels[mix_row_to_item(r).item_id] = True
    for r in nonmembers:
        labels[mix_row_to_item(r).item_id] = False
    if len(labels) != len(members) + len(nonmembers):
        raise SystemExit("member and non-member item ids collide")
    write_jsonl(ITEMS, members + nonmembers)
    write_json(LABELS, labels)
    Y10.mkdir(parents=True, exist_ok=True)
    write_json(Y10 / "sampling.json", {"generated_utc": utcnow(), "git_commit": git_commit(REPO), **sampling})
    print(f"members {len(members)}, non-members {len(nonmembers)}; written under data/ (not printed)", flush=True)


def gold_loss(row: dict[str, Any], item: dict[str, Any]) -> float:
    index = option_index(item["options"], str(item["gold"]))
    return -math.log(max(float(row["probs"][index]), 1e-300))


def cmd_auc(_: argparse.Namespace) -> None:
    items = {mix_row_to_item(r).item_id: r for r in iter_jsonl(ITEMS)}
    labels = read_json(LABELS)
    result: dict[str, Any] = {"generated_utc": utcnow(), "git_commit": git_commit(REPO)}
    for name, path in (("final", Y10 / "scores" / "final_membership" / "predictions.jsonl"),
                       ("base", Y10 / "scores" / "base_membership" / "predictions.jsonl")):
        losses = {"member": [], "nonmember": []}
        scored = {"member": 0, "nonmember": 0}
        for row in iter_jsonl(path):
            key = str(row["item_id"])
            member = labels[key]
            losses["member" if member else "nonmember"].append(gold_loss(row, items[key]))
            scored["member" if member else "nonmember"] += 1
        interval = membership_auc_interval(losses["member"], losses["nonmember"])
        result[name] = {"scored_members": scored["member"], "scored_nonmembers": scored["nonmember"],
                        "auc": round(interval.point, 6), "auc_lower": round(interval.lo, 6), "auc_upper": round(interval.hi, 6),
                        "mean_loss_members": round(sum(losses["member"]) / len(losses["member"]), 6),
                        "mean_loss_nonmembers": round(sum(losses["nonmember"]) / len(losses["nonmember"]), 6),
                        "auc_check": round(membership_auc(losses["member"], losses["nonmember"]), 6)}
    write_json(Y10 / "membership.json", result)
    print(f"final AUC {result['final']['auc']}; base AUC {result['base']['auc']}", flush=True)


def cmd_probe(_: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from meddecide.model.pplx import load_any
    from meddecide.model.readout_checks import exported_causal_lm

    notes = read_json(Y8_BEST / "model.json")
    base_id, revision = notes["base_model_id"], notes["revision"]
    tokenizer = AutoTokenizer.from_pretrained(base_id, revision=revision)
    rows = _probe_notes(tokenizer)
    model = load_any(Y8_BEST, device="cuda", dtype=torch.bfloat16)
    matrix = model.exported_option_code_lm_head().float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    causal = exported_causal_lm(base_id, matrix, revision=revision, dtype=torch.bfloat16, device="cuda")
    causal = PeftModel.from_pretrained(causal, Y8_BEST / "adapter").merge_and_unload().eval()
    final = _generate_counts(causal, rows, tokenizer.pad_token_id or tokenizer.eos_token_id)
    del causal
    gc.collect()
    torch.cuda.empty_cache()
    control = AutoModelForCausalLM.from_pretrained(base_id, revision=revision, dtype=torch.bfloat16, device_map="cuda").eval()
    base = _generate_counts(control, rows, tokenizer.pad_token_id or tokenizer.eos_token_id)
    write_json(Y10 / "probe.json", {"generated_utc": utcnow(), "git_commit": git_commit(REPO), "notes": len(rows),
                                    "prefix_tokens": PREFIX_TOKENS, "continuation_tokens": CONTINUATION_TOKENS,
                                    "run": VERBATIM_RUN, "final": final, "base": base})
    print(f"probe: final {final['matches']} of {len(rows)}; base {base['matches']} of {len(rows)}", flush=True)


def _probe_notes(tokenizer: Any) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(NOTES, columns=["note_id", "text", "tokens", "split"]).to_pylist()
    eligible = [r for r in table if r["split"] == "train" and int(r["tokens"] or 0) >= PREFIX_TOKENS + CONTINUATION_TOKENS]
    eligible.sort(key=lambda r: stable_hash({"probe": str(r["note_id"])}, length=16))
    chosen = eligible[:PROBE_NOTES]
    rows = []
    for r in chosen:
        ids = tokenizer(str(r["text"]), add_special_tokens=False)["input_ids"]
        if len(ids) < PREFIX_TOKENS + CONTINUATION_TOKENS:
            continue
        rows.append({"prefix": ids[:PREFIX_TOKENS], "truth": ids[PREFIX_TOKENS:PREFIX_TOKENS + CONTINUATION_TOKENS]})
    if len(rows) != PROBE_NOTES:
        raise SystemExit(f"{len(rows)} notes with enough tokens; {PROBE_NOTES} needed")
    return rows


def _generate_counts(model: Any, rows: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    import torch

    runs: list[int] = []
    device = next(model.parameters()).device
    for row in rows:
        ids = torch.tensor([row["prefix"]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=CONTINUATION_TOKENS,
                                 do_sample=False, pad_token_id=pad_id)
        generated = out[0, ids.shape[1]:].tolist()
        runs.append(longest_aligned_run(generated, row["truth"]))
    bins = {"at_least_50": sum(r >= 50 for r in runs), "at_least_25": sum(r >= 25 for r in runs),
            "at_least_10": sum(r >= 10 for r in runs)}
    return {"matches": sum(r >= VERBATIM_RUN for r in runs), "run_bins": bins,
            "longest_run_max": max(runs), "longest_run_median": sorted(runs)[len(runs) // 2]}


def cmd_report(_: argparse.Namespace) -> None:
    auc = read_json(Y10 / "membership.json")
    probe = read_json(Y10 / "probe.json")
    sampling = read_json(Y10 / "sampling.json")
    lines = [
        "# osler_v4 — the memorisation and PHI audit (Y10; counts and figures only)", "",
        f"Generated {utcnow()} by `scripts/osler_v4/y10_audit.py` from `outputs/osler_v4/Y10/` (counts only; no item or note text). "
        "This is a report for the release decision, not a gate (ADVISORY Y10).", "",
        "## 1. Membership inference", "",
        f"Members: {sampling['members']} training rows of the final mix; non-members: {sampling['nonmembers']} items of the frozen MIMIC "
        "audit suite v1 (late-anchor test items; no training overlap). Both sides are allocated per template in proportion to the suite's "
        "counts. Score: the negative per-item loss; 0.5 means no measurable membership signal.", "",
        "| model | AUC | interval (95 %, bootstrap, 1,000 resamples, seed 0) | scored members | scored non-members |",
        "|---|---|---|---|---|",
        f"| Osler-4B (final) | {auc['final']['auc']:.4f} | [{auc['final']['auc_lower']:.4f}; {auc['final']['auc_upper']:.4f}] | "
        f"{auc['final']['scored_members']} | {auc['final']['scored_nonmembers']} |",
        f"| zero-shot pplx-decider-flash-v1.2 (control) | {auc['base']['auc']:.4f} | [{auc['base']['auc_lower']:.4f}; "
        f"{auc['base']['auc_upper']:.4f}] | {auc['base']['scored_members']} | {auc['base']['scored_nonmembers']} |", "",
        "The test items are later admissions than most training items, so the control shows how much of any gap is train-to-test shift "
        "alone. The AUC is a measurement, not a verdict.", "",
        "## 2. Verbatim probe", "",
        f"{probe['notes']} train-split discharge notes; a {probe['prefix_tokens']}-token prefix, the next {probe['continuation_tokens']} "
        f"tokens as the truth, greedy generation of {probe['continuation_tokens']} tokens; a match is at least {probe['run']} consecutive "
        "aligned tokens.", "",
        "| model | matches | runs of at least 50 | at least 25 | at least 10 | longest run (median) |",
        "|---|---|---|---|---|---|",
        f"| Osler-4B (final, exported, adapter merged) | {probe['final']['matches']} | {probe['final']['run_bins']['at_least_50']} | "
        f"{probe['final']['run_bins']['at_least_25']} | {probe['final']['run_bins']['at_least_10']} | "
        f"{probe['final']['longest_run_median']} |",
        f"| zero-shot base (control) | {probe['base']['matches']} | {probe['base']['run_bins']['at_least_50']} | "
        f"{probe['base']['run_bins']['at_least_25']} | {probe['base']['run_bins']['at_least_10']} | "
        f"{probe['base']['longest_run_median']} |", "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("members", cmd_members), ("auc", cmd_auc), ("probe", cmd_probe), ("report", cmd_report)):
        sub.add_parser(name).set_defaults(run=fn)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
