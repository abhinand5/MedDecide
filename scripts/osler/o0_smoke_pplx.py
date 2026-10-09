"""O0 smoke test for perplexity-ai/pplx-decider-v1.1-27b through its authors' code.

Runs inside envs/pplx27b (the authors' pinned lock, uv.lock sha256 cba79e0f...). The authors'
package is imported from the checkpoint's own source/src directory, as their README does. This
script imports nothing from meddecide: the env deliberately does not install the project.

Checks (the smoke criterion for this row):
  1. the README's choice example loads on the GPU and returns finite, non-negative probabilities
     that sum to 1 over the options;
  2. its argmax is "support" (an integration failure belongs to the support team, which is what
     the README's example expects);
  3. a noul question on a refund request puts more than half its mass on "true";
  4. the loaded checkpoint is the saved noncausal full-attention decision checkpoint, with the
     temperature from decision_config.json (not a default).
Every value is written to --out; nothing is asserted silently.

Usage:
    envs/pplx27b/.venv/bin/python -I scripts/osler/o0_smoke_pplx.py --checkpoint <snapshot> \
        --out outputs/osler_v0/O0/smoke/pplx27b.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch

    src = args.checkpoint / "source" / "src"
    sys.path.insert(0, str(src))
    from autojev.model import DecisionModel, answer  # authors' code, from the checkpoint

    t0 = time.time()
    model = DecisionModel(args.checkpoint, device="cuda")
    load_s = round(time.time() - t0, 1)
    decision_cfg = json.loads((args.checkpoint / "decision_config.json").read_text())

    question = {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "Charges and refunds",
            "support": "Technical integration errors",
            "sales": "Questions about buying a product",
        },
    }
    row = {"state": "My integration keeps failing. Please help.", "question": question}
    t1 = time.time()
    probs_choice = model.predict([row])[0]
    choice_s = round(time.time() - t1, 3)
    ans_choice = answer(question, probs_choice)

    noul = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
    row_noul = {"state": "Customer says the parcel arrived damaged and wants their money back.",
                "question": noul}
    probs_noul = model.predict([row_noul])[0]
    ans_noul = answer(noul, probs_noul)

    def sane(p: list[float]) -> bool:
        return all(math.isfinite(x) and x >= 0 for x in p) and abs(sum(p) - 1.0) < 1e-3

    keys = list(question["criteria"])
    argmax_choice = keys[max(range(len(probs_choice)), key=lambda i: probs_choice[i])]
    checks = {
        "choice_probabilities_sane": sane(probs_choice),
        "noul_probabilities_sane": sane(probs_noul),
        "choice_argmax_is_support": argmax_choice == "support",
        "noul_true_mass_above_half": probs_noul[1] > 0.5,
        "attention_mode_is_saved_noncausal": model.attention_mode == "noncausal_full_attention",
        "temperature_from_decision_config": abs(model.temperature - decision_cfg["temperature"]) < 1e-12,
    }
    overall = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "id": "pplx27b",
        "checkpoint": str(args.checkpoint),
        "authors_code": "source/src/autojev/model.py (DecisionModel.predict, answer)",
        "env": "envs/pplx27b (authors' uv.lock)",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "load_s": load_s,
        "choice_predict_s": choice_s,
        "choice": {"probabilities": probs_choice, "answer": ans_choice, "argmax": argmax_choice},
        "noul": {"probabilities": probs_noul, "answer": ans_noul},
        "temperature": model.temperature,
        "attention_mode": model.attention_mode,
        "pooling": model.pooling,
        "checks": checks,
        "overall": overall,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"pplx27b: overall={overall} checks={checks}")
    print(f"choice probs={probs_choice} argmax={argmax_choice}; noul probs={probs_noul}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
