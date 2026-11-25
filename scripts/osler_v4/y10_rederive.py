"""osler_v4 Y10 (R6 self-audit; CPU): the membership AUC and the verbatim counts re-derived in a fresh process from the raw files.

The AUC is a pairwise count over the raw prediction files: a member's loss below a non-member's wins the pair, and ties count half.
The per-item loss is -log of the gold option's probability. The gold position is found in the membership item's own option keys; each
prediction must agree with it (the trainer-based rows by gold_index, the pplx rows by gold_key; the count of disagreements is reported).
Membership labels come from
membership_labels.json. The only library call is the item-id scheme (mix_row_to_item), which the labels and the scorers also use.
The verbatim counts are recounted from the per-note runs in probe.json. Compared with membership.json and probe.json to 1e-6.
Writes outputs/osler_v4/Y10/rederive.json. Counts and figures only; no item or note text is printed.

Usage (CPU): CUDA_VISIBLE_DEVICES= uv run --frozen python scripts/osler_v4/y10_rederive.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from meddecide.train.mix_items import mix_row_to_item
from meddecide.utils.io import write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
Y10 = REPO / "outputs" / "osler_v4" / "Y10"
ITEMS = REPO / "data" / "bench" / "osler_v4" / "y10" / "membership.jsonl"
LABELS = REPO / "data" / "bench" / "osler_v4" / "y10" / "membership_labels.json"
OUT = Y10 / "rederive.json"
TOLERANCE = 1e-6
RUN = 50  # ADVISORY Y10: at least 50 consecutive matching tokens


def option_keys_and_gold(path: Path) -> dict[str, tuple[list[str], str]]:
    """Each membership item's option keys (in order) and gold key, keyed by the id the scorers write."""
    out: dict[str, tuple[list[str], str]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(mix_row_to_item(row).item_id)] = ([str(o["key"]) for o in row["options"]], str(row["gold"]))
    return out


def losses(path: Path, labels: dict[str, bool], golds: dict[str, tuple[list[str], str]]) -> tuple[dict[str, list[float]], int]:
    """The negative log of the gold option's probability, split by membership label; and the number of gold keys that disagree."""
    out: dict[str, list[float]] = {"member": [], "nonmember": []}
    disagree = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["item_id"])
            if key not in labels or key not in golds:
                raise SystemExit(f"{path.name}: a scored item has no membership label or options")
            keys, gold = golds[key]
            position = keys.index(gold)
            if "gold_index" in row:  # the trainer-based scorer keys options by position; its gold_index is that position
                agrees = int(row["gold_index"]) == position
            else:  # the pplx scorer keys options by the item's own keys
                agrees = str(row["gold_key"]) == gold
            if not agrees:
                disagree += 1
            if len(row["probs"]) != len(keys):
                raise SystemExit(f"{path.name}: a row's probabilities do not match its item's options")
            prob = max(float(row["probs"][position]), 1e-300)
            out["member" if labels[key] else "nonmember"].append(-math.log(prob))
    return out, disagree


def pairwise_auc(member: list[float], nonmember: list[float]) -> float:
    """The share of member-nonmember pairs in which the member's loss is the lower one; ties count half."""
    m = np.asarray(member, dtype=np.float64)[:, None]
    n = np.asarray(nonmember, dtype=np.float64)[None, :]
    wins = float(np.count_nonzero(m < n)) + 0.5 * float(np.count_nonzero(m == n))
    return wins / (len(member) * len(nonmember))


def recount(runs: list[int]) -> dict[str, int]:
    """The verbatim counts from the per-note longest aligned runs."""
    return {
        "matches": sum(r >= RUN for r in runs),
        "at_least_50": sum(r >= 50 for r in runs),
        "at_least_25": sum(r >= 25 for r in runs),
        "at_least_10": sum(r >= 10 for r in runs),
        "longest_run_max": max(runs),
        "longest_run_median": sorted(runs)[len(runs) // 2],
    }


def main() -> None:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    golds = option_keys_and_gold(ITEMS)
    membership = json.loads((Y10 / "membership.json").read_text(encoding="utf-8"))
    probe = json.loads((Y10 / "probe.json").read_text(encoding="utf-8"))
    mismatches: list[str] = []
    membership_checks: dict[str, object] = {}
    for name in ("final", "base"):
        got, disagree = losses(Y10 / "scores" / f"{name}_membership" / "predictions.jsonl", labels, golds)
        auc = pairwise_auc(got["member"], got["nonmember"])
        recorded = membership[name]
        ok = (
            disagree == 0
            and abs(auc - recorded["auc"]) <= TOLERANCE
            and len(got["member"]) == recorded["scored_members"]
            and len(got["nonmember"]) == recorded["scored_nonmembers"]
            and abs(float(np.mean(got["member"])) - recorded["mean_loss_members"]) <= TOLERANCE
            and abs(float(np.mean(got["nonmember"])) - recorded["mean_loss_nonmembers"]) <= TOLERANCE
        )
        if not ok:
            mismatches.append(f"membership {name}")
        membership_checks[name] = {
            "scored_members": len(got["member"]),
            "scored_nonmembers": len(got["nonmember"]),
            "gold_key_disagreements": disagree,
            "auc_pairwise": round(auc, 6),
            "auc_recorded": recorded["auc"],
            "match": ok,
        }
        print(f"membership {name}: scored {len(got['member'])} and {len(got['nonmember'])}; gold-key disagreements {disagree}; "
              f"pairwise AUC {auc:.6f} vs recorded {recorded['auc']:.6f} ({'match' if ok else 'MISMATCH'})", flush=True)
    verbatim_checks: dict[str, object] = {}
    for name in ("final", "base"):
        runs = [int(r) for r in probe[name]["runs"]]
        counts = recount(runs)
        recorded = {
            "matches": probe[name]["matches"],
            "at_least_50": probe[name]["run_bins"]["at_least_50"],
            "at_least_25": probe[name]["run_bins"]["at_least_25"],
            "at_least_10": probe[name]["run_bins"]["at_least_10"],
            "longest_run_max": probe[name]["longest_run_max"],
            "longest_run_median": probe[name]["longest_run_median"],
        }
        ok = counts == recorded and len(runs) == probe["notes"]
        if not ok:
            mismatches.append(f"verbatim {name}")
        verbatim_checks[name] = {"notes": len(runs), "matches": counts["matches"], "match": ok}
        print(f"verbatim {name}: {counts['matches']} of {len(runs)} notes match on at least {RUN} tokens "
              f"({'match' if ok else 'MISMATCH'})", flush=True)
    write_json(OUT, {"generated_utc": utcnow(), "membership": membership_checks, "verbatim": verbatim_checks,
                     "mismatches": mismatches})
    print(f"fresh-process re-derivation: {len(mismatches)} mismatches", flush=True)
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
