"""W3b: whole synthetic charts from Synthea encounters, narrative pages by the local Gemma 4 teacher (ADVISORY W3b).

Needs the local vLLM server (ADVISORY W3b step 1) and these variables (never hosted credentials):
    TEACHER_BASE_URL=http://127.0.0.1:8000/v1  TEACHER_MODEL=google/gemma-4-31B-it-qat-w4a16-ct  TEACHER_LOCAL=1
    TEACHER_THINKING=none  TEACHER_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'

Encounters come from osler_v1's Synthea extract (inpatient first, then emergency). Splits are by Synthea patient
(``split_of("synthea-<patient>")``, the same split as osler_v1's Synthea items), so a patient's charts never cross
splits. Resumable: encounters already in the progress file are skipped.

Usage:
    uv run --frozen python scripts/osler_v2/w3b_charts.py --count 200 --out data/gen/osler_v2_charts_pilot
    uv run --frozen python scripts/osler_v2/w3b_charts.py --count 20000
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from meddecide.gen.core import split_of
from meddecide.mr.charts import build_chart
from meddecide.mr.synthea import build_encounter
from meddecide.teacher.client import TeacherClient

REPO = Path(__file__).resolve().parents[2]
TABLES = ("observations", "medications", "procedures", "conditions")


def load_encounters(src: Path, limit: int):
    patients = {r["Id"]: r for r in pd.read_parquet(src / "patients.parquet").to_dict("records")}
    enc = pd.read_parquet(src / "encounters.parquet")
    enc = pd.concat([enc[enc["ENCOUNTERCLASS"] == "inpatient"], enc[enc["ENCOUNTERCLASS"] == "emergency"]])
    grouped = {t: defaultdict(list) for t in TABLES}
    for t in TABLES:
        for r in pd.read_parquet(src / f"{t}.parquet").to_dict("records"):
            grouped[t][r["ENCOUNTER"]].append(r)
    out = []
    for e in enc.to_dict("records"):
        p = patients.get(e["PATIENT"])
        if p is None or not grouped["observations"][e["Id"]]:
            continue  # a chart needs measured values for its code pages
        out.append(build_encounter(e, p, *(grouped[t][e["Id"]] for t in TABLES)))
        if len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200, help="encounters to attempt")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--src", default="data/synthea/osler_v1")
    ap.add_argument("--out", default="data/gen/osler_v2_charts")
    args = ap.parse_args()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    client = TeacherClient.from_env(timeout_s=600)
    if not client.local:
        raise SystemExit("W3b runs on the local teacher only (TEACHER_LOCAL=1)")
    probe = client.list_models()
    if not probe["ok"]:
        raise SystemExit(f"local teacher not reachable: {probe['status_code']}")
    progress = out / "progress.jsonl"
    done = {json.loads(line)["encounter"] for line in progress.open()} if progress.exists() else set()
    encounters = [e for e in load_encounters(REPO / args.src, args.count) if e.encounter_id not in done]
    print(f"{len(encounters)} encounters to build ({len(done)} already done)", flush=True)
    lock = threading.Lock()
    files = {s: (out / f"{s}.jsonl").open("a", encoding="utf-8") for s in ("train", "dev", "test")}
    reasons: Counter[str] = Counter()
    started = time.time()

    def one(enc):
        res = build_chart(client, enc, f"chart-{enc.patient_id}:{enc.encounter_id}")
        split = split_of(f"synthea-{enc.patient_id}")
        with lock:
            for it in res.items:
                row = it.as_dict()
                row["split"] = split
                row["patient_id"] = f"synthea-{enc.patient_id}"
                files[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            with progress.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"encounter": enc.encounter_id, "items": len(res.items),
                                     "reasons": list(res.reasons)}) + "\n")
            for r in res.reasons or (("kept",) if res.items else ("no items",)):
                reasons[r.split(":")[0] if ":" in r else r] += 1
        return len(res.items)

    n_items = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(one, e) for e in encounters]), 1):
            n_items += fut.result()
            if i % 50 == 0 or i == len(encounters):
                rate = i / max(1e-9, time.time() - started)
                print(f"{i}/{len(encounters)} charts, {n_items} items, {rate * 3600:.0f} charts/h, "
                      f"tokens {client.usage.as_dict()}", flush=True)
    for fh in files.values():
        fh.close()
    summary = {"charts_attempted": len(encounters), "items": n_items, "reasons": dict(reasons),
               "usage": client.usage.as_dict(), "wall_s": round(time.time() - started, 1)}
    (out / f"summary_{int(time.time())}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
