"""Thin CLI: MIMIC-grounded synth_v1 items (build facts, generate with local Nemotron, verify, probe for leakage).

Stages (run in this order; one GPU job at a time):
    generate   build facts from training-split structured fields, render, call the local server, verify, write items
    probe      leakage probe on a seeded sample of kept items per template; then the families note and claims

Outputs (gitignored item text stays in outputs/synth_v1/): items file, manifest (counts, reasons, rates, no item text),
mimic_families.md and claims_mimic.md (aggregates only). Nothing is printed except counts and rates.

Example:
    uv run python scripts/synth/mimic_v1_build.py generate --per-template 7000 --concurrency 48
    uv run python scripts/synth/mimic_v1_build.py probe
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from meddecide.synth.client import DEFAULT_BASE_URL, NemotronClient, ServerError
from meddecide.synth.driver import Generated, run_plan, verdicts_of
from meddecide.synth.families import STYLES, PlannedCase
from meddecide.synth.leakage import (
    PROBE_SYSTEM,
    ProbeOutcome,
    parse_letter,
    probe_user_prompt,
    sample_items,
    summarize_probe,
)
from meddecide.synth.manifest import build_manifest, write_manifest
from meddecide.synth.mimic_families import (
    EXCLUDED_TEMPLATES,
    MIX_EXCLUSIONS,
    TEMPLATES,
    GroundedCase,
    apr_cases,
    drg_cc_cases,
    ed_disposition_cases,
    family_of,
    icd_specificity_cases,
    label_counts,
    msdrg_cases,
    select,
)
from meddecide.synth.mimic_source import load_admissions, load_ed_stays, load_icd10_titles

OUT = Path("outputs/synth_v1")
ITEMS = OUT / "items_mimic_v1.jsonl"
MANIFEST = OUT / "manifest_mimic_v1.json"
FAMILIES_NOTE = OUT / "mimic_families.md"
CLAIMS = OUT / "claims_mimic.md"
SEED = 20261029
PROBE_N = 300
# Planned cases per template before verification. The ED template is balanced three ways and keeps about half.
PLAN_SIZE: dict[str, int] = {"mimic_ed_disposition_v1": 6000}


def build_all(seed: int) -> dict[str, Any]:
    """Every template's candidate cases from training-split structured fields, with the builder drop counts."""
    admissions = load_admissions()
    titles = load_icd10_titles()
    billable: dict[str, list[str]] = {}
    for code in titles:
        billable.setdefault(code[:3], []).append(code)
    train_drg: dict[str, str] = {}
    for row in admissions:
        if row["split"] == "train" and row["ms_drg_code"] and row["ms_drg_title"]:
            train_drg.setdefault(row["ms_drg_code"], row["ms_drg_title"])
    stays, dx_by_stay = load_ed_stays()
    built = {
        "mimic_icd_specificity_v1": icd_specificity_cases(admissions, titles, billable, seed=seed),
        "mimic_msdrg_v1": msdrg_cases(admissions, titles, train_drg, seed=seed),
        "mimic_drg_cc_level_v1": drg_cc_cases(admissions, titles),
        "mimic_apr_severity_v1": apr_cases(admissions, titles),
        "mimic_ed_disposition_v1": ed_disposition_cases(stays, dx_by_stay),
    }
    return {"built": built, "admission_rows": len(admissions), "ed_stays": len(stays)}


def plan_for(selected: list[GroundedCase], start_index: int, seed: int) -> list[PlannedCase]:
    planned: list[PlannedCase] = []
    for offset, grounded in enumerate(selected):
        rng = random.Random(f"{seed}:style:{grounded.case.case_id}")
        style = STYLES[rng.randrange(len(STYLES))]
        target_words = rng.randrange(60, 161)
        planned.append(PlannedCase(start_index + offset, grounded.case, grounded.spec, style, target_words))
    return planned


def cmd_generate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    built_all = build_all(args.seed)
    grounded_by_index: dict[int, GroundedCase] = {}
    plan: list[PlannedCase] = []
    template_info: dict[str, Any] = {}
    for template in TEMPLATES:
        built = built_all["built"][template]
        target = PLAN_SIZE.get(template, args.per_template)
        selected, shortage = select(built.cases, template, target, seed=args.seed)
        template_info[template] = {
            "family": family_of(template),
            "eligible": len(built.cases),
            "build_drops": dict(sorted(built.drops.items())),
            "selection_shortage": dict(sorted(shortage.items())),
            "planned": len(selected),
            "planned_label_counts": label_counts(selected),
        }
        planned = plan_for(selected, len(plan), args.seed)
        for case_plan, grounded in zip(planned, selected, strict=True):
            grounded_by_index[case_plan.index] = grounded
        plan.extend(planned)
    client = NemotronClient(args.base_url, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    gen_started = time.monotonic()
    results: list[Generated] = run_plan(plan, client, concurrency=args.concurrency)
    wall_s = time.monotonic() - gen_started
    output_tokens = sum(r.output_tokens for r in results)

    OUT.mkdir(parents=True, exist_ok=True)
    items_path = Path(args.items)
    kept_by_template: Counter[str] = Counter()
    with items_path.open("w", encoding="utf-8") as handle:
        for result in results:
            if not result.kept or result.text is None:
                continue
            grounded = grounded_by_index[result.planned.index]
            kept_by_template[grounded.template] += 1
            record = {
                "template": grounded.template,
                "family": grounded.case.family,
                "index": result.planned.index,
                "case_id": grounded.case.case_id,
                "label": grounded.case.label,
                "options": list(grounded.keys),
                "display": list(grounded.display),
                "style": result.planned.style.name,
                "target_words": result.planned.target_words,
                "word_count": result.verdict.word_count,
                "text": result.text,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = build_manifest(
        verdicts_of(results),
        run_name="synth_v1_mimic_grounded",
        config={
            "seed": args.seed,
            "per_template_planned": {t: PLAN_SIZE.get(t, args.per_template) for t in TEMPLATES},
            "templates": list(TEMPLATES),
            "excluded_templates": dict(sorted(EXCLUDED_TEMPLATES.items())),
            "split": "train (hospital admissions and ED patients by the patient-level rule)",
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "chat_template_kwargs": {"enable_thinking": False},
            "source_rows": {"admissions": built_all["admission_rows"], "ed_stays": built_all["ed_stays"]},
            "timing": {
                "generate_wall_clock_s": round(wall_s, 2),
                "output_tokens": output_tokens,
                "output_tokens_per_s": round(output_tokens / wall_s, 2) if wall_s > 0 else None,
                "total_wall_clock_s": round(time.monotonic() - started, 2),
            },
            "length_ratio_median": round(statistics.median(
                r.verdict.word_count / r.planned.target_words for r in results if r.text is not None), 3)
            if any(r.text is not None for r in results) else None,
        },
    )
    manifest["templates"] = {}
    for template in TEMPLATES:
        info = template_info[template]
        family = info["family"]
        fam = manifest["families"].get(family, {"generated": 0, "kept": 0, "rejected": 0, "keep_rate": None,
                                                "rejected_by_primary_reason": {}, "rejected_by_any_reason": {}})
        label_gen: Counter[str] = Counter()
        label_kept: Counter[str] = Counter()
        for result in results:
            grounded = grounded_by_index[result.planned.index]
            if grounded.template != template:
                continue
            label_gen[grounded.case.label] += 1
            label_kept[grounded.case.label] += int(result.kept)
        manifest["templates"][template] = {
            **info,
            "generated": fam["generated"],
            "kept": fam["kept"],
            "keep_rate": fam["keep_rate"],
            "rejected_by_primary_reason": fam["rejected_by_primary_reason"],
            "rejected_by_any_reason": fam["rejected_by_any_reason"],
            "label_generated": dict(sorted(label_gen.items())),
            "label_kept": dict(sorted(label_kept.items())),
        }
    manifest["excluded_templates"] = {t: {"reason": r} for t, r in sorted(EXCLUDED_TEMPLATES.items())}
    manifest["mix_exclusions"] = {t: {"reason": r} for t, r in sorted(MIX_EXCLUSIONS.items())}
    write_manifest(Path(args.manifest), manifest)
    print(f"items={items_path} kept={sum(kept_by_template.values())} generated={len(results)}")
    for template in TEMPLATES:
        print(f"{template}: planned={template_info[template]['planned']} kept={kept_by_template[template]}")
    print(f"wall_clock_s={wall_s:.1f} output_tokens={output_tokens}")
    print(f"manifest={args.manifest}")


def cmd_probe(args: argparse.Namespace) -> None:
    items = [json.loads(line) for line in Path(args.items).read_text(encoding="utf-8").splitlines() if line.strip()]
    client = NemotronClient(args.base_url, model=args.model, temperature=0.0, max_tokens=16)
    leakage: dict[str, Any] = {}

    def ask(item: dict[str, Any]) -> ProbeOutcome | None:
        user = probe_user_prompt(item["text"], item["display"])
        try:
            reply = client.complete(PROBE_SYSTEM, user).text
        except ServerError:
            return None
        index = parse_letter(reply, len(item["options"]))
        correct = None if index is None else item["options"][index] == item["label"]
        return ProbeOutcome(item["template"], len(item["options"]), correct)

    for template in TEMPLATES:
        pool = [item for item in items if item["template"] == template]
        sample = sample_items(pool, PROBE_N, seed=args.seed)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool_exec:
            answers = list(pool_exec.map(ask, sample))
        outcomes = [a for a in answers if a is not None]
        errors = len(answers) - len(outcomes)
        summary = summarize_probe(outcomes)
        summary["probe_errors"] = errors
        summary["pool_kept"] = len(pool)
        summary["chance_basis"] = "mean of 1/(number of offered options) over probed items"
        leakage[template] = summary
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["leakage_probe"] = {
        "question": "Given only this text, which option is the answer? (options listed, reasoning off)",
        "sample_per_template": PROBE_N,
        "seed": args.seed,
        "by_template": leakage,
    }
    write_manifest(manifest_path, manifest)
    write_notes(manifest, manifest_path.parent)
    for template in TEMPLATES:
        s = leakage[template]
        print(f"{template}: probed={s['probed']} unparsed={s['unparsed']} recovered={s['answer_recovered_rate']} "
              f"chance={s['chance']}")


def write_notes(manifest: dict[str, Any], out_dir: Path) -> None:
    """Families note and CLAIMS rows: counts, rates and reason codes only (no item text, values or ids)."""
    lines = ["# synth_v1 MIMIC-grounded families (aggregates only)", "",
             "Generated by `scripts/synth/mimic_v1_build.py`. Labels are structured MIMIC fields read by code. Nemotron writes "
             "text from code-built facts and never decides a label. Training split only.", "",
             "## Templates used", "", "| template | eligible | planned | generated | kept | keep rate |",
             "|---|---|---|---|---|---|"]
    tmpl = manifest["templates"]
    for name in TEMPLATES:
        t = tmpl[name]
        rate = "n/a" if t["keep_rate"] is None else f"{t['keep_rate']:.3f}"
        lines.append(f"| {name} | {t['eligible']} | {t['planned']} | {t['generated']} | {t['kept']} | {rate} |")
    lines += ["", "## Excluded from the training mix", ""]
    for name, reason in sorted(MIX_EXCLUSIONS.items()):
        lines.append(f"- `{name}`: reason `{reason}` (items generated and counted; the mix builder skips them via in_training_mix).")
    lines += ["", "## Excluded templates", ""]
    for name, info in manifest["excluded_templates"].items():
        lines.append(f"- `{name}`: reason `{info['reason']}`. Its option names are near variants of one another (exam names of "
                     "the same modality), so the label is a surface match rather than a finding. Not generated.")
    lines += ["", "- Held-out families (`heart_failure`, AKI) and codes with prefixes I50 and N17, and ICD-9 428 and 584 in "
              "ED diagnoses: any record carrying one is dropped (`held_out_condition`). No held-out template is used.",
              "- Test split: every admission and ED stay outside the training split is dropped (`not_train_split`).",
              "", "## Build drops by reason (counts)", ""]
    for name in TEMPLATES:
        drops = tmpl[name]["build_drops"]
        lines.append(f"- {name}: " + (", ".join(f"`{k}` {v}" for k, v in drops.items()) or "none"))
    lines += ["", "## Selection shortage (counts)", ""]
    for name in TEMPLATES:
        short = tmpl[name]["selection_shortage"]
        lines.append(f"- {name}: " + (", ".join(f"`{k}` {v}" for k, v in short.items()) or "none"))
    (out_dir / FAMILIES_NOTE.name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = ["# synth_v1 MIMIC-grounded claims (aggregates only)", "",
            "id | claim | value | artifact | recompute command | task", "--- | --- | --- | --- | --- | ---"]
    probe = manifest.get("leakage_probe", {}).get("by_template", {})
    cmd = "uv run python scripts/synth/mimic_v1_build.py probe"
    artifact = "outputs/synth_v1/manifest_mimic_v1.json"
    n = 0
    for name in TEMPLATES:
        t = tmpl[name]
        n += 1
        rows.append(f"S{n} | kept items, {name} | {t['kept']} of {t['generated']} | {artifact} | "
                    "`jq` on the manifest `templates` key | synth_v1 build")
        if name in probe:
            p = probe[name]
            rate = p["answer_recovered_rate"]
            n += 1
            rows.append(f"S{n} | leakage probe recovery, {name} | "
                        f"{'n/a' if rate is None else f'{rate:.3f}'} vs chance {p['chance']:.3f} (n={p['probed']}) | "
                        f"{artifact} | {cmd} | synth_v1 probe")
    total_kept = sum(tmpl[name]["kept"] for name in TEMPLATES)
    total_gen = sum(tmpl[name]["generated"] for name in TEMPLATES)
    rows.append(f"S{n + 1} | total kept items | {total_kept} of {total_gen} | {artifact} | {cmd} | synth_v1 build")
    (out_dir / CLAIMS.name).write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--per-template", type=int, default=7000, help="cases planned per template (before verification)")
    gen.add_argument("--concurrency", type=int, default=48)
    gen.add_argument("--items", type=Path, default=ITEMS)
    gen.add_argument("--manifest", type=Path, default=MANIFEST)
    gen.add_argument("--seed", type=int, default=SEED)
    gen.add_argument("--base-url", default=DEFAULT_BASE_URL)
    gen.add_argument("--model", default="nemotron")
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--max-tokens", type=int, default=1200)
    gen.set_defaults(func=cmd_generate)
    probe = sub.add_parser("probe")
    probe.add_argument("--items", type=Path, default=ITEMS)
    probe.add_argument("--manifest", type=Path, default=MANIFEST)
    probe.add_argument("--concurrency", type=int, default=32)
    probe.add_argument("--seed", type=int, default=SEED)
    probe.add_argument("--base-url", default=DEFAULT_BASE_URL)
    probe.add_argument("--model", default="nemotron")
    probe.set_defaults(func=cmd_probe)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
