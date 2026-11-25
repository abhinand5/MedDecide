"""Self-audit of V0 (osler_v1 ADVISORY R6): re-derive the V0 headline numbers from the raw artifacts in a fresh process.

Checks, each re-derived from the file it cites:
- the speed summary of the chosen configuration, recomputed from its stored step records (not from the stored summary);
- the speed choice, recomputed from the five configuration files by the rule in meddecide.train.speed;
- the padding-test and pytest lines, read from their logs;
- the teacher and MIMIC fields of env.json;
- the denominators: the mix v2 training rows and the steps and items in the chosen window.

Writes outputs/osler_v1/V0/SELF_AUDIT.md and prints the checks. Exits non-zero if any check fails.

Usage:
    source scripts/pod_env.sh && uv run --frozen python scripts/osler_v1/v0_self_audit.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from meddecide.train.speed import MEMORY_CAP_GB, WARMUP_STEPS, choose_config, summarise
from meddecide.utils.io import read_json

REPO = Path(__file__).resolve().parents[2]
V0 = REPO / "outputs" / "osler_v1" / "V0"
CONFIGS = ("a", "b", "c", "d", "e")
TARGET_STEPS = 150
MIX_V2_TRAIN_ROWS = 573_611


def check(name: str, ok: bool, detail: str, results: list[tuple[str, bool, str]]) -> None:
    results.append((name, ok, detail))


def main() -> None:
    results: list[tuple[str, bool, str]] = []

    # 1. the chosen configuration, re-summarised from its step records
    chosen_file = read_json(V0 / "speed" / "a.json")
    records = chosen_file["records"]
    fresh = summarise(records)
    stored = chosen_file["summary"]
    same = all(abs(float(fresh[k]) - float(stored[k])) < 1e-6 for k in ("items_per_s", "real_tokens_per_s",
                                                                       "loss_last_window_mean", "peak_reserved_gb"))
    check("speed a: stored summary equals a re-summary of its step records", same,
          f"items/s {fresh['items_per_s']:.4f} (stored {stored['items_per_s']:.4f}); steps {fresh['n_steps']}", results)
    check("speed a: 150 steps, 140 steady-state steps after the 10-step warm-up",
          fresh["n_steps"] == TARGET_STEPS and fresh["window_steps"] == TARGET_STEPS - WARMUP_STEPS,
          f"n_steps {fresh['n_steps']}, window_steps {fresh['window_steps']}", results)
    check("speed a: peak reserved memory within the 85 GB cap", fresh["peak_reserved_gb"] <= MEMORY_CAP_GB,
          f"peak reserved {fresh['peak_reserved_gb']:.2f} GB", results)
    first = read_json(V0 / "speed" / "a_run1.json")["summary"] if (V0 / "speed" / "a_run1.json").exists() else None
    if first is not None:
        rel = abs(first["items_per_s"] - fresh["items_per_s"]) / first["items_per_s"]
        check("speed a: the first run and the re-run agree within 10 % (timing noise)", rel <= 0.10,
              f"first {first['items_per_s']:.4f}, re-run {fresh['items_per_s']:.4f}, relative difference {rel:.3%}", results)

    # 2. the choice, recomputed from the five configuration files
    summaries: dict[str, dict[str, Any]] = {}
    for name in CONFIGS:
        path = V0 / "speed" / f"{name}.json"
        if not path.exists():
            continue
        res = read_json(path)
        if res.get("oom"):
            summaries[name] = {"oom": True, "oom_at_step": res["oom_at_step"], "n_steps": res["summary"].get("n_steps"),
                               "peak_reserved_gb": res["summary"].get("peak_reserved_gb")}
        else:
            summaries[name] = dict(res["summary"])
    summaries["a"] = fresh  # the re-summary replaces the stored summary of the chosen configuration
    chosen, reason = choose_config(summaries, target_steps=TARGET_STEPS)
    recorded = read_json(V0 / "speed.json")["chosen"]
    check("speed choice recomputed from the five configurations equals the recorded choice", chosen == recorded == "a",
          f"recomputed {chosen}, recorded {recorded}: {reason[:120]}", results)

    # 3. the test-suite lines
    pytest_log = (V0 / "logs" / "pytest_full.log").read_text(encoding="utf-8")
    m = re.search(r"(\d+) passed, (\d+) deselected", pytest_log)
    check("pytest: 670 passed with one deselected (the padding test)", bool(m) and m.group(1) == "670"
          and m.group(2) == "1" and "PYTEST_EXIT=0" in pytest_log,
          f"{m.group(0) if m else 'no summary line'}", results)
    padding = (V0 / "logs" / "padding_observed.log").read_text(encoding="utf-8")
    check("padding test observed once: XFAIL, not XPASS (recorded, not edited)", "XFAIL" in padding and "XPASS" not in padding,
          "XFAIL line present", results)

    # 4. the environment and teacher fields
    env = read_json(V0 / "env.json")
    teacher = env["teacher"]
    check("teacher endpoint answered 200 and the chat call returned", teacher.get("models_endpoint", {}).get(
        "status_code") == 200 and teacher.get("chat", {}).get("ok") is True,
          f"model {teacher.get('model_id')}, models status {teacher.get('models_endpoint', {}).get('status_code')}", results)
    check("no key material in env.json", "Bearer" not in json.dumps(env) and "sk-" not in json.dumps(env),
          "scanned for the bearer prefix and the key prefix", results)
    check("MIMIC core set absent (recorded)", env["mimic"]["core_set_complete"] is False,
          "core_set_complete false", results)

    # 5. denominators
    manifest = read_json(REPO / "loops" / "osler_v0" / "mix_v2_manifest.json")
    check("mix v2 training rows equal 573,611 (the osler_v0 manifest)", manifest["train_rows"] == MIX_V2_TRAIN_ROWS,
          f"train_rows {manifest['train_rows']}", results)
    check("steady-state window contains items (a positive count; the batch sizes are not re-counted here)",
          fresh["window_items"] > 0,
          f"window items {fresh['window_items']}, items per batch {fresh['items_per_batch']:.3f}", results)

    lines = ["# V0 self-audit (osler_v1 ADVISORY R6)", "",
             "Re-derived in a fresh process from the raw files named in each row.", "",
             "| check | result | detail |", "|---|---|---|"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |" for name, ok, detail in results]
    (V0 / "SELF_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, ok, detail in results:
        print(("PASS " if ok else "FAIL ") + name + " | " + detail)
    if not all(ok for _, ok, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
