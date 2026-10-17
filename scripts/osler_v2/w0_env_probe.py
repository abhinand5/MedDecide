"""W0 environment probe (osler_v2 ADVISORY W0 steps 2-4): teacher, DeepSeek balance, MIMIC files, GPU and disk.

Writes outputs/osler_v2/W0/env_probe.json. Never prints or stores a key: the teacher is read through the
TEACHER_* / OPENAI_* variables and the DeepSeek key through DEEPSEEK_API_KEY, both from the environment.

Usage (after `source scripts/pod_env.sh`, with DEEPSEEK_API_KEY exported from ~/.bashrc):
    uv run --frozen python scripts/osler_v2/w0_env_probe.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import httpx

from meddecide.teacher.client import TeacherClient

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "osler_v2" / "W0" / "env_probe.json"
PHYSIONET = REPO / "data" / "physionet"
# The five core MIMIC files of osler_v1's V0 check (official PhysioNet files only, D28/D37).
MIMIC_CORE = ("NOTEEVENTS.csv.gz", "diagnoses_icd.csv.gz", "patients.csv.gz", "admissions.csv.gz", "discharge.csv.gz")


def teacher_probe() -> dict:
    client = TeacherClient.from_env()
    info: dict = {"configured": client.configured, "model": client.model or None,
                  "base_url_var": next((v for v in ("TEACHER_BASE_URL", "OPENAI_API_BASE")
                                        if os.environ.get(v)), None)}
    if not client.configured:
        return info | {"status": "not configured"}
    listing = client.list_models()
    body = listing["body"] if isinstance(listing["body"], dict) else {}
    ids = [m.get("id") for m in body.get("data", []) if isinstance(m, dict)]
    info |= {"models_status": listing["status_code"], "models_ok": listing["ok"], "n_models": len(ids)}
    try:
        reply = client.chat([{"role": "user", "content": "Reply with the single word OK."}],
                            temperature=0.0, max_tokens=16)
        info |= {"chat_ok": True, "chat_reply_chars": len(reply)}
    except Exception as exc:  # the probe records the failure instead of stopping
        info |= {"chat_ok": False, "chat_error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    info["client_usage"] = client.usage.as_dict()
    client.close()
    return info


def deepseek_balance() -> dict:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return {"status": "DEEPSEEK_API_KEY not in environment"}
    resp = httpx.get("https://api.deepseek.com/user/balance", headers={"Authorization": f"Bearer {key}"},
                     timeout=30)
    body = resp.json() if resp.status_code == 200 else {}
    return {"status_code": resp.status_code, "is_available": body.get("is_available"),
            "balances": [{"currency": b.get("currency"), "total_balance": b.get("total_balance")}
                         for b in body.get("balance_infos", [])]}


def gpu_and_disk() -> dict:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout.strip()
    disks = {}
    for label, path in (("root", "/"), ("workspace", "/workspace")):
        usage = shutil.disk_usage(path)
        disks[label] = {"total_gb": round(usage.total / 1e9, 1), "free_gb": round(usage.free / 1e9, 1)}
    return {"gpu": gpu, "disk": disks}


def main() -> None:
    payload = {
        "teacher": teacher_probe(),
        "deepseek": deepseek_balance(),
        "mimic_core_found": {name: bool(list(PHYSIONET.rglob(name))) if PHYSIONET.exists() else False
                             for name in MIMIC_CORE},
        **gpu_and_disk(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
