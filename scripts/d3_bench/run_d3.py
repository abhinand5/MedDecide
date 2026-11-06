"""d3_bench steps 2-3 (GPU): vllm-sr/d3-ultra through the card's Quickstart (AutoModel, trust_remote_code=True), local only.

Protocol (the model card's System One call, one question per request, as osler_v5's Clef runner):
    model.system_one(state=item["state"], questions={"q": question})
with ``{"type": "choice", "instructions": item["question"], "criteria": {key: label}}`` for a choice item (our keys, our order)
and ``{"type": "noul", "instructions": item["question"]}`` for a yes/no item (the answer's ``noul`` is p(yes)). No option-order
averaging (the card's default). The answer is mapped back to our option order (meddecide.bench.quick_bench.probs_from_answer).

Local only: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set before any import, the model loads from the pinned local snapshot,
and requests carry text only (no image or URL inputs), so nothing leaves the machine.

Modes:
    --validate   7 hand-made synthetic items (no benchmark data): sums, non-constant answers, argmax matches intent
    (default)    the seeded subsamples in --subsamples, resumable; rows go to <out>/<set>.jsonl (gitignored)

Usage (GPU; detached; worktree venv, so that uv does not resync it):
    setsid nohup .venv/bin/python scripts/d3_bench/run_d3.py --snapshot <dir> --validate > outputs/d3_bench/logs/validate.log 2>&1 &
    setsid nohup .venv/bin/python scripts/d3_bench/run_d3.py --snapshot <dir> > outputs/d3_bench/logs/score.log 2>&1 &
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import json
import math
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from meddecide.bench.quick_bench import probs_from_answer

REPO = Path(__file__).resolve().parents[2]
Y9 = REPO / "data" / "bench" / "osler_v4" / "y9"
if not Y9.exists():  # a git worktree shares data/ with the main checkout
    Y9 = REPO.parents[2] / "data" / "bench" / "osler_v4" / "y9"
SETS = ("knowledge", "panel", "mr_verdict")  # never a MIMIC-derived set

# 7 hand-made synthetic items: (state, question, qtype, options {key: label}, intended key)
SYNTHETIC = [
    ("A 54-year-old man has crushing chest pain radiating to the left arm; ECG shows ST elevation in II, III and aVF.",
     "What is the most likely diagnosis?", "choice",
     {"A": "Inferior myocardial infarction", "B": "Migraine", "C": "Ankle sprain", "D": "Seasonal allergy"}, "A"),
    ("A child has a barking cough, inspiratory stridor and a low-grade fever after a cold.",
     "Which diagnosis fits best?", "choice",
     {"A": "Otitis externa", "B": "Croup (laryngotracheobronchitis)", "C": "Gout", "D": "Carpal tunnel syndrome"}, "B"),
    ("Fasting plasma glucose 7.9 mmol/L on two occasions; HbA1c 7.4 %.",
     "Which condition do these results indicate?", "choice",
     {"A": "Hypothyroidism", "B": "Iron deficiency", "C": "Diabetes mellitus", "D": "Normal glucose tolerance"}, "C"),
    ("Discharge note: the patient was given amoxicillin and developed an urticarial rash and wheeze within 20 minutes.",
     "Does the note describe an allergic reaction to the drug?", "noul", {"yes": "Yes", "no": "No"}, "yes"),
    ("Discharge note: routine follow-up; blood pressure 118/76, no complaints, no medication changes.",
     "Does the note describe an acute emergency?", "noul", {"yes": "Yes", "no": "No"}, "no"),
    ("A woman reports polyuria, polydipsia, weight loss and fruity breath; blood ketones are high and pH is 7.12.",
     "What is the most likely diagnosis?", "choice",
     {"A": "Diabetic ketoacidosis", "B": "Tension headache", "C": "Plantar fasciitis", "D": "Common cold",
      "E": "Contact dermatitis"}, "A"),
    ("Thyroid panel: TSH 0.01 mU/L (low), free T4 high; the patient has tremor, heat intolerance and weight loss.",
     "Which thyroid state do these findings show?", "choice",
     {"A": "Hypothyroidism", "B": "Euthyroid", "C": "Hyperthyroidism"}, "C"),
]


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def question_for(qtype: str, question: str, options: dict[str, str]) -> dict:
    if qtype == "choice":
        return {"type": "choice", "instructions": question, "criteria": dict(options)}
    if qtype == "noul":
        return {"type": "noul", "instructions": question}
    raise ValueError(f"unsupported qtype {qtype!r}")


def ask(model, state: str, qtype: str, question: str, options: dict[str, str]) -> tuple[list[float] | None, dict]:
    response = model.system_one(state=state, questions={"q": question_for(qtype, question, options)})
    answer = response["answers"]["q"]
    if "error" in answer:
        return None, {"error": answer.get("error"), "reason": answer.get("reason"), "usage": response.get("usage")}
    probs = probs_from_answer(list(options), qtype, answer)
    if not all(math.isfinite(p) for p in probs):
        raise RuntimeError("non-finite probabilities")
    return probs, {"usage": response.get("usage")}


def load(snapshot: Path):
    import torch
    from transformers import AutoModel

    t0 = time.time()
    model = AutoModel.from_pretrained(str(snapshot), trust_remote_code=True, dtype=torch.bfloat16, device="cuda:0")
    return model, time.time() - t0


def environment(snapshot: Path) -> dict:
    import torch
    import transformers

    try:
        import fla
        fla_version = getattr(fla, "__version__", "?")
    except Exception as exc:
        fla_version = f"missing ({type(exc).__name__})"
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"git_commit": commit, "snapshot": str(snapshot), "revision": snapshot.name, "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__, "fla": fla_version,
            "gpu": torch.cuda.get_device_name(0), "offline": os.environ.get("HF_HUB_OFFLINE")}


def validate(model, out: Path) -> None:
    rows = []
    for i, (state, question, qtype, options, intent) in enumerate(SYNTHETIC):
        probs, extra = ask(model, state, qtype, question, options)
        keys = list(options)
        argmax = keys[max(range(len(keys)), key=lambda j: probs[j])] if probs else None
        rows.append({"item": i, "qtype": qtype, "options": len(keys), "intent": intent, "argmax": argmax,
                     "sum": sum(probs) if probs else None, "probs": probs, **extra})
        print(f"synthetic {i}: qtype={qtype} argmax={argmax} intent={intent} sum={sum(probs) if probs else None}", flush=True)
    sums_ok = all(r["sum"] is not None and abs(r["sum"] - 1.0) <= 1e-3 for r in rows)
    non_constant = len({r["argmax"] for r in rows}) > 1
    intent_ok = sum(r["argmax"] == r["intent"] for r in rows)
    verdict = "PASS" if sums_ok and non_constant and intent_ok == len(rows) else "FAIL"
    result = {"rows": rows, "sums_within_1e-3": sums_ok, "non_constant": non_constant,
              "distinct_argmax": len({r["argmax"] for r in rows}), "argmax_matches_intent": intent_ok,
              "items": len(rows), "verdict": verdict, "finished_utc": utcnow()}
    (out / "validate.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"VALIDATE {verdict}: sums_ok={sums_ok} non_constant={non_constant} intent={intent_ok}/{len(rows)}", flush=True)


def score(model, subsamples: Path, out: Path) -> None:
    plan = json.loads(subsamples.read_text(encoding="utf-8"))
    for set_name in SETS:
        wanted = set(plan["sets"][set_name]["item_ids"])
        path = out / f"{set_name}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(line)["item_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        started = time.time()
        n = 0
        with path.open("a", encoding="utf-8") as fh, (Y9 / f"{set_name}.jsonl").open(encoding="utf-8") as src:
            for line in src:
                item = json.loads(line)
                item_id = str(item["item_id"])
                if item_id not in wanted or item_id in done:
                    continue
                options = {str(o["key"]): str(o["label"]) for o in item["options"]}
                t1 = time.time()
                try:
                    probs, extra = ask(model, item["state"], item["qtype"], item["question"], options)
                except ValueError as exc:  # a request the runtime refuses: recorded, never dropped
                    probs, extra = None, {"error": "exception", "reason": str(exc)[:200]}
                row = {"item_id": item_id, "template_id": item["template_id"], "source": item.get("source"),
                       "qtype": item["qtype"], "option_keys": list(options), "gold": item["gold"],
                       "status": "scored" if probs is not None else "skipped", "probs": probs,
                       "latency_s": round(time.time() - t1, 4), **extra}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n += 1
                if n % 100 == 0:
                    print(f"[{utcnow()}] {set_name}: {n} new rows, {n / (time.time() - started):.2f} items/s", flush=True)
        print(f"[{utcnow()}] {set_name}: done, {n} new rows in {time.time() - started:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--subsamples", type=Path, default=REPO / "outputs" / "d3_bench" / "subsamples.json")
    parser.add_argument("--out", type=Path, default=REPO / "outputs" / "d3_bench" / "predictions")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    t0 = time.time()
    model, load_s = load(args.snapshot.resolve())
    env = environment(args.snapshot.resolve())
    print(f"[{utcnow()}] loaded in {load_s:.1f}s: {json.dumps(env)}", flush=True)
    if args.validate:
        validate(model, args.out)
    else:
        score(model, args.subsamples, args.out)
    import torch

    run = {"mode": "validate" if args.validate else "score", "started_utc": started, "finished_utc": utcnow(),
           "wall_s": round(time.time() - t0, 1), "load_s": round(load_s, 1),
           "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), **env}
    (args.out / f"run_{run['mode']}.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(f"[{utcnow()}] RUN DONE {json.dumps(run)}", flush=True)


if __name__ == "__main__":
    main()
