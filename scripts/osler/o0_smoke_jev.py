"""O0 smoke test for an autotrust JEV checkpoint through its card's decision-head protocol.

The protocol is the one the card documents (adapter_vllm + decision_head.json + calibration.json)
and the one the F7 baseline (scripts/bench/run_jev_baseline.py) implements: one prefill, the
full-vocabulary log-softmax at the last position, the head's per-slot bias, the per-kind
temperature, and a softmax over the offered verbalizer tokens. The helpers are imported from F7
so that this smoke and the scoreboard share one implementation (the card's vLLM route is not
run; see the deviation note in the output).

Targets come from the JEV-27B card (the vLLM route, `decide()` in its README):
    choice example: "≈ {'issue_warning': 0.25, 'renegotiate': 0.13, 'dual_source': 0.62, 'maintain': 0.001}"
PASS needs: valid probabilities for both examples; the choice argmax is `dual_source`; the choice
probabilities are within 0.03 of the card's approximate values (when the card gives them); and the
refund noul example puts more than half its mass on "true". The 27B card's own transformers block
prints {'false': 0.022, 'true': 0.978} for that noul example through a different head file
(head.safetensors); the value is recorded for comparison, not used as a pass criterion.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o0_smoke_jev.py \
        --repo autotrust/JEV-27B --card-values --out outputs/osler_v0/O0/smoke/jev27b.json
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import time
from pathlib import Path
from typing import Any

import torch

from meddecide.utils.io import write_json
from meddecide.utils.provenance import gpu_name, utcnow

REPO = Path(__file__).resolve().parents[2]
F7_PATH = REPO / "scripts/bench/run_jev_baseline.py"
CARD_CHOICE = {"issue_warning": 0.25, "renegotiate": 0.13, "dual_source": 0.62, "maintain": 0.001}
CARD_NOUL_TRANSFORMERS_BLOCK = {"false": 0.022, "true": 0.978}


def load_f7():
    spec = importlib.util.spec_from_file_location("run_jev_baseline", F7_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--card-values", action="store_true",
                    help="compare the choice example with the 27B card's approximate values")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    f7 = load_f7()
    head, temperatures = f7.load_head(args.repo)
    tok = AutoTokenizer.from_pretrained(args.repo)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(args.repo, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, args.repo, subfolder="adapter_vllm")
    model.eval()
    load_s = round(time.time() - t0, 1)

    def decide(kind: str, state: str, question: str, options: list[str]) -> dict[str, Any]:
        n = len(options) if kind == "choice" else (2 if kind == "noul" else len(options))
        labels = options if kind != "noul" else ["false", "true"]
        prompt = f7.build_prompt(kind, state, question, labels)
        ids = f7.verbalizer_ids(head, kind, n)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits[0, -1, :]
        probs = f7.head_probabilities(logits, ids, head, kind, temperatures[kind], n)
        return {"prompt_tokens": int(enc["input_ids"].shape[1]), "verbalizer_ids": ids,
                "options": labels if kind != "score" else [str(i) for i in range(n)],
                "probabilities": probs}

    choice_opts = ["issue_warning", "renegotiate", "dual_source", "maintain"]
    t1 = time.time()
    choice = decide("choice", "SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.",
                    "Supplier response for this scenario.", choice_opts)
    choice_s = round(time.time() - t1, 3)
    noul = decide("noul", "Customer says the parcel arrived damaged and wants their money back.",
                  "Is the customer asking for a refund?", ["false", "true"])

    def sane(p: list[float]) -> bool:
        return all(math.isfinite(x) and x >= 0 for x in p) and abs(sum(p) - 1.0) < 1e-3

    choice_map = dict(zip(choice["options"], choice["probabilities"], strict=True))
    argmax = max(choice_map, key=choice_map.get)
    noul_map = dict(zip(noul["options"], noul["probabilities"], strict=True))
    checks = {
        "choice_probabilities_sane": sane(choice["probabilities"]),
        "noul_probabilities_sane": sane(noul["probabilities"]),
        "choice_argmax_is_dual_source": argmax == "dual_source",
        "noul_true_mass_above_half": noul_map["true"] > 0.5,
    }
    card_diffs = None
    if args.card_values:
        card_diffs = {k: round(abs(choice_map[k] - v), 4) for k, v in CARD_CHOICE.items()}
        checks["choice_within_0.03_of_card"] = max(card_diffs.values()) <= 0.03
    overall = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "task": "O0 smoke (card decision-head protocol)",
        "utc": utcnow(),
        "repo": args.repo,
        "protocol": "adapter_vllm via PEFT + decision_head.json + calibration.json (F7 helpers); "
                    "the card's vLLM route is NOT run in this smoke",
        "deviation": "vLLM not installed in the main env; transformers+PEFT path, as F7",
        "gpu": gpu_name(),
        "torch": torch.__version__,
        "load_s": load_s,
        "choice_s": choice_s,
        "temperatures": temperatures,
        "choice": {**choice, "argmax": argmax, "card_values": CARD_CHOICE if args.card_values else None,
                   "abs_diff_to_card": card_diffs},
        "noul": {**noul, "card_transformers_block_values": CARD_NOUL_TRANSFORMERS_BLOCK
                 if args.card_values else None},
        "checks": checks,
        "overall": overall,
    }
    write_json(args.out, payload)
    print(f"{args.repo}: overall={overall} checks={checks}")
    print(f"choice={ {k: round(v, 4) for k, v in choice_map.items()} }")
    print(f"noul={ {k: round(v, 4) for k, v in noul_map.items()} }")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
