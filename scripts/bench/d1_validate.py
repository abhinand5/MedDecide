"""d1-2B loader validation (bench/d1-2b): hand-made synthetic items through the card's API (GPU, one call per question).

Checks, each recorded in the output JSON:
- every probability vector is finite, in [0, 1] and sums to one (validate_probabilities), per question;
- the answers are not constant: the argmax answers across the synthetic items take at least two distinct values, and the
  noul answers separate a positive from a negative case;
- the card's own worked example (its "Transformers quick start": refund, team, topics, urgency) is run through the card's
  call directly, so the loader's answers can be compared with the card's printed output.

Writes outputs/d1_bench/validate/validation.json. The synthetic items are written here, not taken from any benchmark.

Usage (one GPU job at a time):
    uv run --frozen --with pillow==12.3.0 --with "torchvision==0.29.*" python scripts/bench/d1_validate.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import transformers

from meddecide.model.d1 import REPO, REVISION, D1Model, validate_probabilities
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, gpu_name, utcnow

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "outputs" / "d1_bench" / "validate" / "validation.json"

NOUL_YN = [("yes", "Yes"), ("no", "No")]
TEAMS = [("A", "Charges, refunds, invoices"), ("B", "Delivery and parcels"), ("C", "App or site faults")]
URGENCY = [("A", "low"), ("B", "medium"), ("C", "high")]

# (name, qtype, state, question, options, expected)  -- expected: the key the case should read as most likely (choice/score),
# or "yes"/"no" for noul. These are our own sentences, written for this check.
SYNTHETIC = [
    ("anticoag_pos", "noul", "Discharge summary. Medications: warfarin 5 mg nightly for atrial fibrillation.",
     "Is the patient taking an anticoagulant?", NOUL_YN, "yes"),
    ("anticoag_neg", "noul", "Discharge summary. Medications: metformin 500 mg twice daily for type 2 diabetes.",
     "Is the patient taking an anticoagulant?", NOUL_YN, "no"),
    ("ticket_billing", "choice", "I was charged twice for my order and I want one charge refunded.",
     "Which team should handle this ticket?", TEAMS, "A"),
    ("ticket_shipping", "choice", "My parcel was due on Monday and the tracking has not moved for two weeks.",
     "Which team should handle this ticket?", TEAMS, "B"),
    ("ticket_technical", "choice", "The app closes every time I open the settings screen on Android.",
     "Which team should handle this ticket?", TEAMS, "C"),
    ("urgency_high", "score", "Patient reports crushing chest pain spreading to the left arm, now sweating and short of breath.",
     "How urgent is this case?", URGENCY, "C"),
    ("urgency_low", "score", "Patient asks for a copy of last year's vaccination record for a school form next term.",
     "How urgent is this case?", URGENCY, "A"),
]


def card_example(model: Any) -> dict[str, Any]:
    """The card's quick start, run through the card's own call (not through the loader)."""
    questions = {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Charges, refunds, invoices", "shipping": "Delivery and parcels",
                              "technical": "App or site faults"}},
        "topics": {"type": "choice", "multiple": True, "instructions": "Which topics does the message raise?",
                   "criteria": {"billing": "Payments and charges", "delivery": "Shipping and parcels",
                                "login": "Signing in to the account"}},
        "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                    "criteria": ["low", "medium", "high"]},
    }
    out = model.system_one("I was charged twice and my parcel never arrived. Please refund one charge.", questions)
    a = out["answers"]
    return {"refund_p_yes": round(float(a["refund"]["noul"]), 4), "team_choice": a["team"]["choice"],
            "topics_choices": a["topics"]["choices"], "urgency_legend": a["urgency"]["legend"][str(int(a["urgency"]["score"]))],
            "card_printed_output": "1.0 billing ['billing', 'delivery'] medium"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    started = time.time()
    loaded = D1Model.load()
    load_s = round(time.time() - started, 1)
    results = []
    for name, qtype, state, question, options, expected in SYNTHETIC:
        t0 = time.time()
        probs, meta = loaded.answer(state, qtype, question, options)
        validate_probabilities(probs)
        keys = [k for k, _ in options]
        argmax = keys[max(range(len(probs)), key=lambda i: probs[i])]
        results.append({"name": name, "qtype": qtype, "option_keys": keys, "probs": [round(p, 6) for p in probs],
                        "sum": round(sum(probs), 9), "argmax": argmax, "expected": expected,
                        "agrees_with_expected": argmax == expected, "answer_type": meta["answer_type"],
                        "input_tokens": meta["input_tokens"], "latency_s": round(time.time() - t0, 4)})
    argmaxes = {r["argmax"] for r in results}
    noul = {r["name"]: r["probs"][r["option_keys"].index("yes")] for r in results if r["qtype"] == "noul"}
    checks = {
        "every_vector_sums_to_one": all(abs(r["sum"] - 1.0) <= 1e-4 for r in results),
        "answers_not_constant": len(argmaxes) >= 2,
        "noul_separates_pos_neg": noul["anticoag_pos"] > 0.5 > noul["anticoag_neg"],
        "all_expected_argmax": all(r["agrees_with_expected"] for r in results),
    }
    payload = {
        "task": "bench/d1-2b loader validation", "utc": utcnow(), "git_commit": git_commit(REPO_ROOT),
        "model": REPO, "revision": REVISION, "transformers": transformers.__version__, "torch": torch.__version__,
        "gpu": gpu_name(), "load_s": load_s, "calibration_applied": loaded.calibrated,
        "synthetic": results, "checks": checks, "argmax_values": sorted(argmaxes),
        "card_example": card_example(loaded.model),
        "note": "agrees_with_expected is a check on hand-made sentences, not a benchmark result",
    }
    write_json(args.out, payload)
    print(f"checks={checks}", flush=True)
    print(f"written {args.out}", flush=True)
    if not all(checks.values()):
        raise SystemExit("a validation check failed; see the JSON")


if __name__ == "__main__":
    main()
