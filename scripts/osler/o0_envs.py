"""O0 aggregator: envs.json, throughput.json and smoke_summary.json, recomputed from raw outputs.

Nothing here is typed by hand. Each smoke status is re-derived from its raw smoke JSON (the
`overall` field and its `checks`), each throughput figure is read from the two throughput runs,
and the environment record asks each env's own interpreter for its versions.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler/o0_envs.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

REPO = Path(__file__).resolve().parents[2]
O0 = REPO / "outputs/osler_v0/O0"
SMOKE = O0 / "smoke"
VERSIONS = (
    "import importlib.metadata as md, json, sys\n"
    "out = {'python': sys.version.split()[0]}\n"
    "for p in ['torch','transformers','peft','accelerate','flash-linear-attention',"
    "'causal-conv1d','pillow','torchvision','safetensors']:\n"
    "    try:\n"
    "        out[p] = md.version(p)\n"
    "    except md.PackageNotFoundError:\n"
    "        out[p] = None\n"
    "print(json.dumps(out))\n"
)


def env_versions(python: Path) -> dict[str, Any]:
    out = subprocess.run([str(python), "-I", "-c", VERSIONS], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def smoke_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(SMOKE.glob("*.json")):
        raw = json.loads(path.read_text())
        checks = raw.get("checks") or {}
        if "results" in raw:  # card-runner output: one result per card example
            status = "PASS" if raw.get("overall") == "PASS" and all(
                r.get("status") in ("PASS", "NO_PROBABILITIES") for r in raw["results"]) else "FAIL"
            detail = {
                "examples_with_probabilities": raw.get("examples_with_probabilities"),
                "worst_abs_diff": max((r["worst_abs_diff"] for r in raw["results"]
                                       if r.get("worst_abs_diff") is not None), default=None),
                "gold_matches": sum(1 for r in raw["results"] if r.get("argmax_matches_card_gold")),
                "gold_checked": sum(1 for r in raw["results"] if r.get("argmax_matches_card_gold") is not None),
            }
        else:
            status = "PASS" if raw.get("overall") == "PASS" and all(checks.values()) else "FAIL"
            detail = {"checks": checks}
        rows.append({"id": raw.get("id", path.stem), "status": status, "file": str(path.relative_to(REPO)),
                     "overall_field": raw.get("overall"), "detail": detail})
    return rows


def throughput_block() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for size in ("4b", "9b"):
        raw = json.loads((O0 / f"throughput_{size}.json").read_text())
        out[size] = {
            "model": raw["model"],
            "sequence_length_median_prompt": raw["prompt_length_tokens"]["median"],
            "mean_prompt_tokens": raw["prompt_length_tokens"]["mean"],
            "p95_prompt_tokens": raw["prompt_length_tokens"]["p95"],
            "sec_per_step_mean_timed": raw["timing"]["sec_per_step_mean_timed"],
            "tokens_per_s_train": raw["timing"]["tokens_per_s_train"],
            "eval_items_per_s_forward_b8": raw["timing"]["eval_items_per_s_forward_b8"],
            "eval_items_per_s_forward_b32": raw["timing"]["eval_items_per_s_forward_b32"],
            "peak_memory_gb": raw["memory_gb"]["peak_incl_eval"],
            "projection_one_arm": raw["projection_one_arm"],
            "file": str((O0 / f"throughput_{size}.json").relative_to(REPO)),
        }
    return out


def main() -> None:
    snapshot_dirs = sorted(Path("/workspace/.hf_home/hub/models--Qwen--Qwen3.5-4B/snapshots").glob("*/config.json"))
    qwen4 = json.loads(snapshot_dirs[0].read_text())
    text_cfg = qwen4.get("text_config", qwen4)
    layer_types = text_cfg["layer_types"]
    full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    linear = [i for i, t in enumerate(layer_types) if t != "full_attention"]

    main_env = REPO / ".venv/bin/python"
    envs = {
        "task": "O0",
        "utc": utcnow(),
        "git_commit": git_commit(REPO),
        "main_env": {"path": ".venv (uv --frozen)", **env_versions(main_env)},
        "qwen35_4b_layers": {
            "snapshot": str(snapshot_dirs[0].parent.name),
            "num_hidden_layers": text_cfg["num_hidden_layers"],
            "hidden_size": text_cfg["hidden_size"],
            "vocab_size": text_cfg["vocab_size"],
            "tie_word_embeddings": text_cfg.get("tie_word_embeddings", qwen4.get("tie_word_embeddings")),
            "full_attention_layers": full,
            "linear_attention_layers_count": len(linear),
            "full_attention_interval": text_cfg.get("full_attention_interval"),
        },
        "competitor_envs": {
            "envs/pplx27b": {"path": "envs/pplx27b/.venv (authors' uv.lock, cba79e0f...)",
                             **env_versions(REPO / "envs/pplx27b/.venv/bin/python")},
            "envs/clef": {"path": "envs/clef/.venv (card's tested torch 2.11 / transformers 5.10.2)",
                          **env_versions(REPO / "envs/clef/.venv/bin/python")},
        },
        "note": "main env has fla and causal_conv1d (fast linear-attention kernels). envs/pplx27b has fla but "
                "not causal_conv1d; envs/clef has neither. Where a kernel is missing the reference PyTorch path "
                "runs (correct, slower), so latency comparisons need that caveat",
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, driver 595.91.07",
    }
    write_json(O0 / "envs.json", envs)

    throughput = {
        "task": "O0",
        "utc": utcnow(),
        "git_commit": git_commit(REPO),
        "recipe": "LoRA r=32 alpha=32, bf16 base frozen, batch 8, median prompt length of the student_v1 mix, "
                  "50 steps (first 5 excluded)",
        "sizes": throughput_block(),
    }
    write_json(O0 / "throughput.json", throughput)

    rows = smoke_rows()
    summary = {
        "task": "O0",
        "utc": utcnow(),
        "git_commit": git_commit(REPO),
        "rows": rows,
        "n_pass": sum(r["status"] == "PASS" for r in rows),
        "n_fail": sum(r["status"] == "FAIL" for r in rows),
    }
    write_json(O0 / "smoke_summary.json", summary)
    print(f"envs.json, throughput.json, smoke_summary.json written; smoke PASS={summary['n_pass']} "
          f"FAIL={summary['n_fail']}")
    for r in rows:
        print(f"  {r['id']:12s} {r['status']}")


if __name__ == "__main__":
    main()
