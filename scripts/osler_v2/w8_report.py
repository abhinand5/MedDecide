"""osler_v2 W8: write loops/osler_v2/gate_o3.md from outputs/osler_v2/W8/gate_o3.json (aggregates only; generated, not typed by hand).

Usage: uv run --frozen python scripts/osler_v2/w8_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "outputs" / "osler_v2" / "W8" / "gate_o3.json"
MD = REPO / "loops" / "osler_v2" / "gate_o3.md"


def fmt(x: float | None, digits: int = 4) -> str:
    return "—" if x is None else f"{x:+.{digits}f}"


def row(name: str, d: dict[str, Any]) -> str:
    if "not_measured" in d:
        return f"| {name} | NOT MEASURED - {d['not_measured']} | | | | |"
    a, b = d["accuracy"], d["brier"]
    return (f"| {name} | {d['n_paired']} | {fmt(a['point'])} [{fmt(a['lo'])}, {fmt(a['hi'])}] | "
            f"{fmt(b['point'])} [{fmt(b['lo'])}, {fmt(b['hi'])}] | "
            f"{'yes' if d['accuracy_lower_above_zero'] else 'no'} | {'yes' if d['accuracy_lower_above_panel_floor'] else 'no'} |")


def main() -> None:
    g = json.loads(GATE.read_text(encoding="utf-8"))
    lines = [
        "# Gate O3 (osler_v2): Osler-4B - verdict",
        "",
        f"Generated from `outputs/osler_v2/W8/gate_o3.json` by `scripts/osler_v2/w8_report.py` ({g['written_utc']}).",
        "Aggregates only; every number has a row in `loops/osler_v2/CLAIMS.md` (W053-W062).",
        "",
        f"**Verdict (as the gate code states it): {g['verdict']}** - {g['reason']}",
        "",
        f"**Measured failures:** {'; '.join(g['measured_failures']) or 'none'}.",
        f"**Not measured:** {'; '.join(g['unmeasured']) or 'none'}.",
        f"**Gate passes:** {g['gate_passes']}. The gate passes only when every required comparison passes; a measured failure "
        "stands whatever the unmeasured checks show. The 9B verdict is NOT MEASURED (deferred by the operator, D39).",
        "",
        "## Scopes (items)",
        "",
        "| scope | items |",
        "|---|---|",
    ]
    for k, v in g["scope_sizes"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Primary checks (decide the verdict)", "",
              "Accuracy and Brier are Osler-4B minus the baseline, paired by item; intervals are bootstrapped within templates "
              f"({g['n_resamples']} resamples, seed {g['seed']}).", "",
              "| check | paired items | accuracy diff [95 % CI] | Brier diff [95 % CI] | accuracy lower > 0 | panel floor passed |",
              "|---|---|---|---|---|---|"]
    for name, d in g["primary"]["mr_suite_v1"]["vs"].items():
        lines.append(row(f"MR suite v1 (verdict templates) vs {name}", d))
    for name, d in g["primary"]["external_panel"]["vs"].items():
        lines.append(row(f"external panel vs {name}", d))
    kg = g["primary"]["knowledge_guard"]
    lines.append(f"| knowledge guard (test) vs {kg['baseline']} | "
                 + ("NOT MEASURED - " + kg["not_measured"] + " |  |  |  |" if "not_measured" in kg else
                    f"{kg['n_paired']} | {fmt(kg['accuracy']['point'])} [{fmt(kg['accuracy']['lo'])}, "
                    f"{fmt(kg['accuracy']['hi'])}] | | {'yes' if kg['passes'] else 'no'} | |"))
    lines += ["", "## Additional rows (labelled; never in the verdict)", "",
              "| row | paired items | accuracy diff [95 % CI] | Brier diff [95 % CI] | accuracy lower > 0 | panel floor passed |",
              "|---|---|---|---|---|---|"]
    add = g["additional"]
    for name, d in add["mr_verdict_generated_part"].items():
        lines.append(row(f"MR verdict, generated part vs {name}", d))
    for name, d in add["mr_verdict_real_text_part"].items():
        lines.append(row(f"MR verdict, real-text part vs {name}", d))
    for name, d in add["mr_verdict_without_pmc"].items():
        lines.append(row(f"MR verdict without PMC vs {name}", d))
    for name, d in add["reported_only_mr_templates"].items():
        lines.append(row(f"MR reported-only templates vs {name}", d))
    for name, d in add["held_out_test"].items():
        if name != "excluded_readout_fail_templates_for_MedDecider_4B":
            lines.append(row(f"held-out MR test vs {name}", d))
    for name, d in add["vs_zero_shot_Qwen_not_in_verdict"].items():
        lines.append(row(f"MR verdict vs {name} (not a verdict baseline under D39)", d))
    lines.append(row("external panel vs zero-shot Qwen3.5-4B (not a verdict baseline under D39)",
                     add["external_panel_vs_zero_shot_Qwen_not_in_verdict"]))
    g_q = add["knowledge_guard_vs_zero_shot_Qwen_full_set"]
    lines.append(f"| knowledge test vs zero-shot Qwen3.5-4B (full set) | {g_q.get('n_paired', '—')} | "
                 f"{fmt(g_q['accuracy']['point']) if 'accuracy' in g_q else '—'} "
                 f"[{fmt(g_q['accuracy']['lo']) if 'accuracy' in g_q else '—'}, "
                 f"{fmt(g_q['accuracy']['hi']) if 'accuracy' in g_q else '—'}] | | "
                 f"{'yes' if g_q.get('passes') else 'no'} | |")
    g_p = add["knowledge_guard_vs_zero_shot_pplx_paired_items_only"]
    lines.append(f"| knowledge test vs zero-shot pplx, items pplx can score ({g_p.get('n_skipped_by_pplx')} skipped) | "
                 f"{g_p.get('n_paired', '—')} | {fmt(g_p['accuracy']['point'])} [{fmt(g_p['accuracy']['lo'])}, "
                 f"{fmt(g_p['accuracy']['hi'])}] | | {'yes' if g_p.get('passes') else 'no'} | |")
    if "osler_v1_osler_4b_mr_verdict" in add:
        for name, d in add["osler_v1_osler_4b_mr_verdict"].items():
            lines.append(row(f"MR verdict, osler_v1's Osler-4B (V10) vs {name}", d))
    lines += ["", "d1-3B: NOT MEASURED - not run in this loop (ADVISORY W8 lists it as an additional row).", ""]
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
