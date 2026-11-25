"""Fetch the osler_v0 model registry into HF_HOME at pinned revisions (O0; ADVISORY §3 disk rule).

For every entry in configs/osler_v0/competitors.json:
  1. ask the Hub for the current commit sha; refuse to download if it differs from the pin;
  2. refuse to download if the counted disk (hf_home + outputs + data) plus the repo's size would
     exceed the 450 GB budget — that row is recorded as BLOCKED — disk;
  3. snapshot_download at the pinned revision (resumable; already-cached files are skipped);
  4. record the local snapshot path and the byte total of the snapshot's files.

Usage (detached, from the repo root):
    source scripts/pod_env.sh && setsid nohup uv run --frozen python scripts/osler/o0_fetch.py \
        --only all > outputs/osler_v0/O0/logs/fetch.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from meddecide.utils.io import write_json
from meddecide.utils.provenance import utcnow

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "configs/osler_v0/competitors.json"
OUT = REPO / "outputs/osler_v0/O0/fetch.json"
BUDGET_BYTES = 450 * 10**9


def counted_bytes() -> int:
    paths = ["/workspace/.hf_home", str(REPO / "outputs"), str(REPO / "data")]
    out = subprocess.run(["du", "-sb", *paths], capture_output=True, text=True, check=True)
    return sum(int(line.split()[0]) for line in out.stdout.splitlines())


def snapshot_bytes(path: Path) -> int:
    """Bytes of the files a snapshot resolves to (symlinks followed into the blob store)."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="all", help="comma-separated registry ids, or 'all'")
    args = ap.parse_args()

    registry = json.loads(REGISTRY.read_text())
    wanted = None if args.only == "all" else set(args.only.split(","))
    api = HfApi()
    records: list[dict[str, Any]] = json.loads(OUT.read_text()) if OUT.exists() else []
    done_ids = {r["id"] for r in records if r.get("status") == "DONE"}

    for entry in registry["models"]:
        mid = entry["id"]
        if wanted is not None and mid not in wanted:
            continue
        if mid in done_ids:
            print(f"[{utcnow()}] {mid}: already DONE in fetch.json, skipping", flush=True)
            continue
        rec: dict[str, Any] = {"id": mid, "repo": entry["repo"], "pinned_revision": entry["revision"],
                               "started_utc": utcnow()}
        info = api.model_info(entry["repo"], files_metadata=True)
        rec["hub_sha"] = info.sha
        if info.sha != entry["revision"]:
            rec.update(status="BLOCKED — revision mismatch", finished_utc=utcnow())
            print(f"[{utcnow()}] {mid}: BLOCKED revision mismatch hub={info.sha}", flush=True)
        else:
            repo_bytes = sum((s.size or 0) for s in info.siblings)
            rec["repo_bytes_hub"] = repo_bytes
            now = counted_bytes()
            rec["counted_bytes_before"] = now
            if now + repo_bytes > BUDGET_BYTES:
                rec.update(status="BLOCKED — disk", finished_utc=utcnow())
                print(f"[{utcnow()}] {mid}: BLOCKED disk now={now} + repo={repo_bytes} > budget", flush=True)
            else:
                print(f"[{utcnow()}] {mid}: downloading {entry['repo']}@{entry['revision'][:12]} "
                      f"({repo_bytes / 1e9:.1f} GB; counted now {now / 1e9:.1f} GB)", flush=True)
                t0 = time.time()
                local: Path | None = None
                last_error = ""
                for attempt in range(1, 4):
                    try:
                        local = Path(snapshot_download(entry["repo"], revision=entry["revision"]))
                        break
                    except Exception as exc:  # recorded, retried, then BLOCKED
                        last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                        print(f"[{utcnow()}] {mid}: attempt {attempt} failed: {last_error}", flush=True)
                        time.sleep(30 * attempt)
                if local is None:
                    rec.update(status=f"BLOCKED — download error after 3 attempts ({last_error})",
                               finished_utc=utcnow())
                else:
                    rec.update(local_path=str(local), resolved_revision=local.name,
                               snapshot_bytes=snapshot_bytes(local),
                               wall_s=round(time.time() - t0, 1),
                               status="DONE", finished_utc=utcnow())
                    print(f"[{utcnow()}] {mid}: DONE in {rec['wall_s']} s -> {local}", flush=True)
                    rec["counted_bytes_after"] = counted_bytes()
        records = [r for r in records if r["id"] != mid] + [rec]
        write_json(OUT, records)

    print(f"[{utcnow()}] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
