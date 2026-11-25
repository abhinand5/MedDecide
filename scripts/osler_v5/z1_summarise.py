"""osler_v5 Z1 summary (CPU; a fresh process): every probe figure recomputed from the saved per-item rows in outputs/osler_v5/Z1/<setting>/,
with the composite as defined in meddecide.eval.composite (MR and MIMIC template macro, CG v2 micro). The probe job's own composite used
the earlier item-micro definition (STATE §7.10), so its composite figures are not reported; these are.

Writes outputs/osler_v5/Z1/summary.json: per setting, the composite and its parts, knowledge dev accuracy, positional health (original
order and the two permutations), the DDXPlus two-letter and single-letter accuracies, and the CG v2 micro per template.

Usage (CPU):
    uv run --frozen python scripts/osler_v5/z1_summarise.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meddecide.eval.composite import accuracy, composite, positional_health
from meddecide.utils.io import write_json

REPO = Path(__file__).resolve().parents[2]
Z1 = REPO / "outputs" / "osler_v5" / "Z1"
DDX_PREFIX = "cg_v2_ddxplus"
SETTINGS = ("alpha_0", "alpha_0.3", "alpha_0.5", "alpha_0.7", "alpha_0.85", "alpha_1",
            "split_trained_adapter_base_readout", "split_base_adapter_trained_readout")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarise(folder: Path) -> dict[str, Any]:
    fast = read_jsonl(folder / "fast_dev.jsonl")
    cg = read_jsonl(folder / "cg_v2.jsonl")
    know = read_jsonl(folder / "knowledge_dev.jsonl")
    seeds = {seed: read_jsonl(folder / f"positional_seed{seed}.jsonl") for seed in (1, 2)}
    parts = composite(fast, cg)
    ddx_original = [r for r in cg if str(r["template_id"]).startswith(DDX_PREFIX)]
    original = positional_health(ddx_original)
    by_template: dict[str, list[bool]] = {}
    for r in cg:
        by_template.setdefault(str(r["template_id"]), []).append(bool(r["correct"]))
    return {
        "composite": parts.to_dict(),
        "knowledge_dev_micro": accuracy(know),
        "knowledge_dev_items": len(know),
        "positional_original": original.to_dict(),
        "positional_permuted": {str(seed): positional_health(rows).to_dict() for seed, rows in seeds.items()},
        "positional_passes_all_orders": bool(original.passes and all(positional_health(rows).passes for rows in seeds.values())),
        "ddx_two_letter_accuracy": original.two_letter_accuracy,
        "ddx_single_letter_accuracy": original.single_letter_accuracy,
        "cg_micro_per_template": {t: {"accuracy": sum(v) / len(v), "items": len(v)} for t, v in sorted(by_template.items())},
    }


def main() -> None:
    result: dict[str, Any] = {}
    for name in SETTINGS:
        folder = Z1 / name
        if not (folder / "fast_dev.jsonl").exists():
            result[name] = "NOT MEASURED — the probe job has not written this setting"
            continue
        result[name] = summarise(folder)
        s = result[name]
        print(f"{name}: S {s['composite']['S']:.4f} (MR {s['composite']['mr']:.4f}, MIMIC {s['composite']['mimic']:.4f}, CG micro "
              f"{s['composite']['cg_micro']:.4f}); knowledge {s['knowledge_dev_micro']:.4f}; DDX two-letter "
              f"{s['ddx_two_letter_accuracy']:.4f} / single {s['ddx_single_letter_accuracy']:.4f}; positional all orders "
              f"{s['positional_passes_all_orders']}", flush=True)
    write_json(Z1 / "summary.json", result)


if __name__ == "__main__":
    main()
