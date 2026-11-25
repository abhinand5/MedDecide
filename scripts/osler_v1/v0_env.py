"""V0: record the pod environment, teacher reachability and MIMIC file presence (osler_v1 ADVISORY V0 steps 2-3).

Writes outputs/osler_v1/V0/env.json. Never writes the endpoint URL or any key: the teacher is described by the names of
the variables it was read from, its model ids, and the status and usage of one harmless chat call. The chat prompt is a
fixed string; no MIMIC or benchmark text is ever sent.

MIMIC presence is checked by file name and size only (never opened).

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v0_env.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from meddecide.teacher.client import TeacherClient
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v1" / "V0" / "env.json"
PHYSIONET = REPO / "data" / "physionet"

# MIMIC files named in ADVISORY V0 step 3; the operator's wget may keep the physionet.org/files/<dataset>/<version>/ prefix.
MIMIC_FILES = {
    "mimic3_noteevents": "NOTEEVENTS.csv.gz",
    "mimic4_diagnoses_icd": "diagnoses_icd.csv.gz",
    "mimic4_patients": "patients.csv.gz",
    "mimic4_admissions": "admissions.csv.gz",
    "mimic4_note_discharge": "discharge.csv.gz",
    "mimic4_note_radiology": "radiology.csv.gz",
}
MIMIC_ED_GLOB = "ed/*.csv.gz"


def gpu_info() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"present": False}
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout.strip().splitlines()
    gpus = []
    for line in out:
        name, total, used, util = (p.strip() for p in line.split(","))
        gpus.append({"name": name, "memory_total_mib": int(total), "memory_used_mib": int(used),
                     "utilization_pct": int(util)})
    return {"present": True, "gpus": gpus}


def disk_info() -> dict[str, Any]:
    result = {}
    for label, path in (("workspace", "/workspace"), ("root", "/")):
        usage = shutil.disk_usage(path)
        result[label] = {"total_gb": round(usage.total / 1e9, 1), "free_gb": round(usage.free / 1e9, 1)}
    return result


def teacher_info() -> dict[str, Any]:
    client = TeacherClient.from_env()
    source = {
        "base_url_from": next((v for v in ("TEACHER_BASE_URL", "OPENAI_API_BASE", "OPENAI_BASE_URL")
                               if os.environ.get(v)), None),
        "api_key_from": next((v for v in ("TEACHER_API_KEY", "OPENAI_API_KEY") if os.environ.get(v)), None),
        "model_from": next((v for v in ("TEACHER_MODEL", "OPENAI_MODEL") if os.environ.get(v)), None),
    }
    info: dict[str, Any] = {"configured": client.configured, "model_id": client.model or None,
                            "local": client.local, "thinking": client.thinking, "variables": source}
    if not client.configured:
        info["status"] = "not configured"
        return info
    listing = client.list_models()
    body = listing["body"] if isinstance(listing["body"], dict) else {}
    model_ids = [m.get("id") for m in body.get("data", []) if isinstance(m, dict)]
    info["models_endpoint"] = {"status_code": listing["status_code"], "ok": listing["ok"],
                               "n_models": len(model_ids), "model_ids": model_ids[:50]}
    if client.model:
        try:
            reply = client.chat([{"role": "user", "content": "Reply with the single word OK."}],
                                temperature=0.0, max_tokens=16)
            info["chat"] = {"ok": True, "reply_length": len(reply), "reply": reply.strip()[:20]}
        except Exception as exc:
            info["chat"] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    info["usage"] = client.usage.as_dict()
    info["status"] = "reachable" if listing["ok"] else "models endpoint not 200"
    client.close()
    return info


def mimic_info() -> dict[str, Any]:
    found: dict[str, Any] = {}
    for key, name in MIMIC_FILES.items():
        hits = sorted(PHYSIONET.rglob(name)) if PHYSIONET.exists() else []
        found[key] = {"present": bool(hits), "bytes": [p.stat().st_size for p in hits]}
    ed = sorted(PHYSIONET.rglob(MIMIC_ED_GLOB)) if PHYSIONET.exists() else []
    found["mimic4_ed"] = {"present": bool(ed), "n_files": len(ed), "bytes": [p.stat().st_size for p in ed]}
    complete = all(found[k]["present"] for k in ("mimic3_noteevents", "mimic4_diagnoses_icd", "mimic4_patients",
                                                 "mimic4_admissions", "mimic4_note_discharge"))
    return {"physionet_dir_exists": PHYSIONET.exists(), "files": found, "core_set_complete": complete}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    payload = {
        "task": "V0",
        "written_utc": utcnow(),
        "git_commit": git_commit(REPO),
        "gpu": gpu_info(),
        "disk": disk_info(),
        "teacher": teacher_info(),
        "mimic": mimic_info(),
    }
    write_json(OUT, payload)
    print(f"wrote {OUT}")
    print(f"teacher status: {payload['teacher'].get('status')}; MIMIC core complete: {payload['mimic']['core_set_complete']}")


if __name__ == "__main__":
    main()
