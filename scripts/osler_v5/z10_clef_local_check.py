"""osler_v5 Z10 data rule (ADVISORY §10 Z10.2): do Cloudflare Clef-Flash's weights run locally?

Runs in envs/clef (the authors' stack; no meddecide import). Three checks, written to outputs/osler_v5/Z10/clef_local_check.json:
1. The snapshot is the pinned revision (configs/osler_v0/competitors.json: the folder name under snapshots/ is the commit).
2. The authors' Python code in the snapshot (joint_schema_model.py and every other .py file there) is scanned for network clients
   (requests, httpx, urllib, http.client, socket, aiohttp, openai, anthropic, huggingface_hub, websockets, grpc) and for http(s) URLs.
   Each hit is listed with its file and line and the line's text (authors' code; no item data is involved).
3. The model is loaded and used with the network disabled: HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE are set before any import, and
   two public panel items are scored by the authors' systemone function on the GPU. Only the first two panel rows are read.

local_only is true only when check 1 matches, check 3 succeeds, and the scan has no hit. A hit does not fail the run by itself: it
sets local_only to false until the hit is read and an allowlist entry, with its reason, is added here (ALLOWLIST holds the one hit read).
Only when local_only is true may o2-style Z10 items go to Clef (scripts/osler_v5/z10_clef.py refuses otherwise).

Usage (GPU; one job; detached, after the Z9 job has exited):
    source scripts/pod_env.sh && envs/clef/.venv/bin/python -I scripts/osler_v5/z10_clef_local_check.py \
        --path <snapshot> > outputs/osler_v5/Z10/logs/clef_local_check.log 2>&1 < /dev/null
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

# the network is switched off before any library import: a load or a score that needs it fails here
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

REPO = Path(__file__).resolve().parents[2]
COMPETITORS = REPO / "configs" / "osler_v0" / "competitors.json"
PANEL = REPO / "data" / "bench" / "osler_v4" / "y9" / "panel.jsonl"
OUT = REPO / "outputs" / "osler_v5" / "Z10" / "clef_local_check.json"
NETWORK_PATTERN = re.compile(
    r"\b(?:import|from)\s+(?:requests|httpx|urllib|http\.client|socket|aiohttp|openai|anthropic|huggingface_hub|websockets|grpc)\b"
    r"|https?://",
)
# (file suffix, line text fragment, reason) for hits read by hand. The one hit found on 2026-10-28T20:23Z is the authors' loader's
# import of snapshot_download. load_release_model calls it only when the path given is not a local directory; this check passes the
# snapshot directory, and HF_HUB_OFFLINE is set, so the call is not reached. systemone sends no request and does no I/O for text items.
ALLOWLIST: tuple[tuple[str, str, str], ...] = (
    ("joint_schema_model.py", "from huggingface_hub import snapshot_download",
     "authors' loader calls snapshot_download only when the path is not a local directory; the check passes a directory"),
)


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def pinned_revision(model_id: str) -> str:
    for entry in json.loads(COMPETITORS.read_text(encoding="utf-8"))["models"]:
        if entry["id"] == model_id:
            return str(entry["revision"])
    raise KeyError(model_id)


def scan_code(snapshot: Path) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    for path in sorted(snapshot.rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if NETWORK_PATTERN.search(line):
                accepted = any(path.name.endswith(suffix) and fragment in line for suffix, fragment, _ in ALLOWLIST)
                hits.append({"file": str(path.relative_to(snapshot)), "line": number, "text": line.strip()[:200],
                             "allowlisted": accepted})
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--path", type=Path, required=True, help="the local snapshot directory of Cloudflare/clef-flash")
    args = parser.parse_args()
    snapshot = args.path.resolve()
    pinned = pinned_revision("clef_flash")
    record: dict[str, object] = {"checked_utc": utcnow(), "model": "Cloudflare/clef-flash", "snapshot": str(snapshot),
                                 "pinned_revision": pinned, "snapshot_revision": snapshot.name}
    record["revision_matches"] = snapshot.name == pinned
    hits = scan_code(snapshot)
    record["network_hits"] = hits
    try:
        import torch

        sys.path.insert(0, str(snapshot))
        import joint_schema_model as authors  # the authors' module, from the snapshot

        model, processor = authors.load_release_model(snapshot, device="cuda")
        panel_rows = []
        with PANEL.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    panel_rows.append(json.loads(line))
                if len(panel_rows) == 2:
                    break
        scored = []
        for row in panel_rows:
            keys = [o["key"] for o in row["options"]]
            if row["qtype"] == "choice":
                question = {"type": "choice", "instructions": row["question"],
                            "criteria": {o["key"]: o["label"] for o in row["options"]}}
            elif row["qtype"] == "noul":
                question = {"type": "noul", "instructions": row["question"]}
            else:
                question = {"type": "score", "instructions": row["question"], "criteria": [o["label"] for o in row["options"]]}
            response = authors.systemone(model, processor, {"model": "clef_flash", "state": row["state"],
                                                            "questions": {"q": question}})
            answer = response["answers"]["q"]
            if row["qtype"] == "noul":
                values = [float(answer["noul"])]  # P(true); the probability_sum field is then P(true)
            elif row["qtype"] == "choice":
                values = [float(answer["probabilities"][k]) for k in keys]
            else:
                values = [float(answer["probabilities"][str(i)]) for i in range(len(keys))]
            scored.append({"answer_type": answer["type"], "option_count": len(keys), "probability_sum": round(sum(values), 4),
                           "finite": all(math.isfinite(v) for v in values)})
        record["offline_load_and_score"] = {"ok": True, "device": torch.cuda.get_device_name(0),
                                            "class": type(model).__name__, "items": scored}
    except Exception as exc:  # the failure is recorded, not hidden: the Z10 gate then refuses
        record["offline_load_and_score"] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    unexplained = [h for h in hits if not h["allowlisted"]]
    load_ok = bool(record["offline_load_and_score"]["ok"])
    record["local_only"] = bool(record["revision_matches"] and load_ok and not unexplained)
    record["rule"] = ("local_only = revision matches the pin, the offline load and two panel scores succeed, and no network-client "
                      "or URL line is in the authors' code outside the allowlist (ALLOWLIST: one entry, read on 2026-10-28).")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"revision matches {record['revision_matches']}; network hits {len(hits)} (unexplained {len(unexplained)}); "
          f"offline load and score ok {load_ok}; local_only {record['local_only']}", flush=True)


if __name__ == "__main__":
    main()
