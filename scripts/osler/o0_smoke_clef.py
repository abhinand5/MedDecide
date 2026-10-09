"""O0 smoke test for Cloudflare Clef / Clef-Flash through the authors' joint_schema_model.py.

The two requests are the card's own examples (the invoice record and the checkout-errors
triage record), answered through the authors' `systemone` endpoint function. The card prints no
expected numbers, so the criteria are the obvious answers the records state:

  1. both requests return a SystemOne body with one answer per question;
  2. every choice/score answer's probabilities are finite, non-negative and sum to 1;
  3. invoice record: status argmax is `overdue` (the invoice is overdue), and P(large) > 0.5
     (1,250 USD is above 1,000 USD);
  4. checkout record: department argmax is `technical` (checkout errors are a technical problem).

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o0_smoke_clef.py \
        --path <snapshot> --id clef --out outputs/osler_v0/O0/smoke/clef.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sane(p: dict[str, float]) -> bool:
    values = list(p.values())
    return all(math.isfinite(v) and v >= 0 for v in values) and abs(sum(values) - 1.0) < 1e-3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.path))
    import joint_schema_model as authors  # the authors' module, from the checkpoint

    t0 = time.time()
    model, processor = authors.load_release_model(args.path, device="cuda")
    load_s = round(time.time() - t0, 1)

    invoice = {
        "model": "clef",
        "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
        "questions": {
            "status": {"type": "choice", "instructions": "What is the invoice status?",
                       "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.",
                                    "draft": "Not sent."}},
            "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
        },
    }
    checkout = {
        "model": "clef",
        "state": "Our checkout started returning errors and orders are blocked.",
        "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle the message?",
                           "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
            "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
            "outage": {"type": "noul", "instructions": "Is a service down?"},
        },
    }
    t1 = time.time()
    r_invoice = authors.systemone(model, processor, invoice)
    r_checkout = authors.systemone(model, processor, checkout)
    answer_s = round(time.time() - t1, 3)

    a_inv, a_chk = r_invoice["answers"], r_checkout["answers"]
    checks = {
        "invoice_one_answer_per_question": set(a_inv) == {"status", "large"},
        "checkout_one_answer_per_question": set(a_chk) == {"department", "urgency", "outage"},
        "invoice_status_probabilities_sane": sane(a_inv["status"]["probabilities"]),
        "checkout_department_probabilities_sane": sane(a_chk["department"]["probabilities"]),
        "checkout_urgency_probabilities_sane": sane(a_chk["urgency"]["probabilities"]),
        "invoice_status_argmax_is_overdue": a_inv["status"]["choice"] == "overdue",
        "invoice_large_true_above_half": a_inv["large"]["noul"] > 0.5,
        "checkout_department_argmax_is_technical": a_chk["department"]["choice"] == "technical",
    }
    overall = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "id": args.id,
        "task": "O0 smoke (card examples through the authors' systemone)",
        "utc": utcnow(),
        "path": str(args.path),
        "authors_code": "joint_schema_model.py (systemone, load_release_model)",
        "env": "envs/clef (pins follow the card's tested torch 2.11 / transformers 5.10.2)",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "load_s": load_s,
        "answer_s_two_requests": answer_s,
        "invoice_response": r_invoice,
        "checkout_response": r_checkout,
        "checks": checks,
        "overall": overall,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"{args.id}: overall={overall} failed={[k for k, v in checks.items() if not v]}")
    print(f"invoice status={a_inv['status']['choice']} large={round(a_inv['large']['noul'], 4)}")
    print(f"checkout department={a_chk['department']['choice']} urgency_score={round(a_chk['urgency']['score'], 3)} "
          f"outage={round(a_chk['outage']['noul'], 4)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
