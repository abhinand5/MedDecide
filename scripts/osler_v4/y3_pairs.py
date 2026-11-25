"""Y3: Nemotron-crafted minimal pairs, ``mimic_doc_support_v1`` (osler_v4 ADVISORY Y3; D43, D45, D46). Credentialed data.

Every request carries MIMIC note text, so every teacher call is flagged ``contains_mimic=True`` and goes through the loopback
guard (osler_v4 Y0): the teacher must be the local Nemotron server on 127.0.0.1:8011. Nothing is sent elsewhere.

Subcommands
  pilot --mode off|on --attempts 200     train admissions only; reasoning in the selection call off or on (ADVISORY Y3 §7)
  build --mode off|on --hours 8          dev 250 pairs, then test 250 pairs (late-anchor), then train up to 40,000 pairs
  cue                                    Naive Bayes on note text (edited vs original): ROC AUC on dev (ADVISORY Y3 §6)

Outputs (private): attempts as JSONL and the training rows under outputs/osler_v4/Y3/<run>/ and data/mimic_build/pairs/<run>/.
The loop reads only aggregates from them (loops/osler_v4/crafted.md).
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from meddecide.mix.rows import make_row
from meddecide.mr import icd, mimic
from meddecide.mr import mimic_pairs as mp
from meddecide.mr.generators import HELD_OUT_ICD_FAMILIES
from meddecide.teacher.client import TeacherClient
from meddecide.utils.hashing import stable_hash
from meddecide.utils.io import write_json
from meddecide.utils.provenance import git_commit, utcnow

BUILD = Path("data/mimic_build")
PAIRS = BUILD / "pairs"
OUT = Path("outputs/osler_v4/Y3")
BASE_URL = "http://127.0.0.1:8011/v1"
MODEL = "nemotron"
WORKERS = 64
PROMPT_CAP = 8192
CHAPTERS = frozenset("EDFGJKNRI")                 # ADVISORY Y3 §1; I50 and N17 are excluded below
HELD_OUT_FAMILIES = HELD_OUT_ICD_FAMILIES | {"aki"}  # heart failure (held out) and AKI (held out, indicator)
LICENCE = "PhysioNet Credentialed Health Data License 1.5.0"
RECORD_URL = "https://physionet.org/content/mimic-iv-note/2.2/"
TARGETS = {"dev": 250, "test": 250, "train": 40_000}  # kept pairs per stage; the evaluation sets come first
TOKENIZER = "perplexity-ai/pplx-decider-flash-v1.2"   # the prompt cap is counted with the pplx tokenizer (ADVISORY §3.8)


def dotted(code: str) -> str:
    return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"


def targets_of(codes: list[str], principal: str | None, titles: dict[str, str]) -> list[str]:
    """The coded secondary diagnoses that may be the target: not principal, in the listed chapters, not I50 or N17, not in a
    held-out family, and with a title (ADVISORY Y3 step 1)."""
    out = []
    for code in dict.fromkeys(codes):
        if code == principal or code[:1] not in CHAPTERS or code[:3] in ("I50", "N17") or code not in titles:
            continue
        family = mimic.CODE_TO_FAMILY.get(dotted(code))
        if family in HELD_OUT_FAMILIES:
            continue
        out.append(code)
    return out


class Chats:
    """The teacher calls of the pair pipeline. Each worker thread owns its clients (and their usage meters); the selection
    call may reason (pilot "on"), the verification calls never do (ADVISORY Y3 step 5)."""

    def __init__(self, reasoning_select: bool, select_max_tokens: int) -> None:
        self.reasoning_select = reasoning_select
        self.select_max_tokens = select_max_tokens
        self._local = threading.local()
        self._clients: list[TeacherClient] = []
        self._lock = threading.Lock()

    def _thread_clients(self) -> tuple[TeacherClient, TeacherClient]:
        if not hasattr(self._local, "select"):
            select = TeacherClient(base_url=BASE_URL, model=MODEL, local=True, thinking="none", timeout_s=1800.0,
                                   extra_body={"chat_template_kwargs": {"enable_thinking": self.reasoning_select}})
            verify = TeacherClient(base_url=BASE_URL, model=MODEL, local=True, thinking="none", timeout_s=1800.0,
                                   extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            self._local.select, self._local.verify = select, verify
            with self._lock:
                self._clients += [select, verify]
        return self._local.select, self._local.verify

    def __call__(self, messages: list[dict[str, str]], kind: str) -> str:
        select, verify = self._thread_clients()
        if kind == "select":
            return select.chat(messages, temperature=0.0, max_tokens=self.select_max_tokens, contains_mimic=True)
        return verify.chat(messages, temperature=0.0, max_tokens=8, contains_mimic=True)

    def usage(self) -> dict[str, int]:
        total: Counter[str] = Counter()
        with self._lock:
            for client in self._clients:
                for key, value in client.usage.as_dict().items():
                    if key != "cost_usd":
                        total[key] += int(value)
        return dict(total)


class Corpus:
    """Admissions, discharge notes and ICD titles from the Y1 tables (private)."""

    def __init__(self) -> None:
        from transformers import AutoTokenizer

        self.admissions = pd.read_parquet(BUILD / "admissions.parquet")
        self.admissions = self.admissions[self.admissions["coding"].eq("icd10") & self.admissions["has_discharge_note"]]
        notes = pd.read_parquet(BUILD / "discharge_notes.parquet", columns=["note_id", "text"])
        self.notes = dict(zip(notes["note_id"], notes["text"], strict=True))
        titles = pd.read_parquet(BUILD / "icd10cm_titles.parquet")
        self.titles = {str(c): str(t) for c, t in zip(titles["icd_code"], titles["long_title"], strict=True)}
        self.family_terms = {f.key: f.condition for f in icd.FAMILIES}
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def pool(self, stage: str, seed: int) -> list[dict[str, Any]]:
        frame = self.admissions
        if stage == "train":
            frame = frame[frame["split"].eq("train")]
        elif stage == "dev":
            frame = frame[frame["split"].eq("dev")]
        else:
            frame = frame[frame["split"].eq("test") & frame["late_anchor"]]
        rows = frame[["hadm_id", "subject_id", "split", "discharge_note_id", "discharge_tokens", "icd10_codes",
                      "principal_icd10"]].to_dict("records")
        rows.sort(key=lambda r: stable_hash({"seed": seed, "hadm": int(r["hadm_id"])}, length=16))
        return rows

    def state_of(self, adm: dict[str, Any]) -> tuple[str, bool]:
        text = self.notes[adm["discharge_note_id"]].replace("\r\n", "\n").replace("\r", "\n")
        if int(adm["discharge_tokens"]) <= PROMPT_CAP:
            return text, False
        return mimic.prioritised_text(mimic.sections(text), PROMPT_CAP, self.count)


def process(adm: dict[str, Any], stage: str, corpus: Corpus, chats: Chats, seed: int, mode: str) -> tuple[dict, list[dict]]:
    hadm = int(adm["hadm_id"])
    rng = random.Random(stable_hash({"seed": seed, "hadm": hadm, "stage": stage}, length=16))
    codes = [str(c) for c in adm["icd10_codes"]]
    principal = None if adm["principal_icd10"] is None or pd.isna(adm["principal_icd10"]) else str(adm["principal_icd10"])
    targets = targets_of(codes, principal, corpus.titles)
    base = {"hadm_id": hadm, "split": adm["split"], "stage": stage, "mode": mode}
    if not targets:
        return {**base, "outcome": "no eligible target code", "calls": 0}, []
    code = rng.choice(targets)
    state, cut = corpus.state_of(adm)
    family = mimic.CODE_TO_FAMILY.get(dotted(code))
    terms = (corpus.family_terms[family],) if family else ()
    others = [(c, corpus.titles[c]) for c in codes if c not in (code, principal) and c in corpus.titles]
    started = time.monotonic()
    try:
        attempt = mp.build_pair(state, code, corpus.titles[code], terms, others, chats, rng)
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        return {**base, "code": dotted(code), "outcome": f"teacher error: {type(exc).__name__}", "calls": 0}, []
    record = {**base, "code": dotted(code), "outcome": attempt.outcome, "n_sentences": attempt.n_sentences,
              "n_selected": attempt.n_selected, "calls": attempt.calls, "others_checked": attempt.others_checked,
              "others_skipped": attempt.others_skipped, "cut": cut, "seconds": round(time.monotonic() - started, 2)}
    if attempt.outcome != "kept":
        return record, []
    pair_id = stable_hash({"hadm": hadm, "code": code, "mode": mode}, length=16)
    question = mp.question_text(corpus.titles[code], dotted(code))
    meta = {"pair_id": pair_id, "code": dotted(code), "family": family, "n_sentences": attempt.n_sentences,
            "n_selected": attempt.n_selected, "share_selected": round(attempt.n_selected / attempt.n_sentences, 4),
            "others_checked": attempt.others_checked, "cut": cut, "mode": mode}
    rows = []
    for role, text, gold in (("original", attempt.original, "yes"), ("edit", attempt.edited, "no")):
        rows.append(make_row(source="mimic_iv_note", template_id="mimic_doc_support_v1", skill="mimic_doc_support",
                             qtype="noul", state=text, question=question, options=list(mp.YES_NO_OPTIONS), gold=gold,
                             record_id=f"hadm:{hadm}:{code}", record_url=RECORD_URL, licence=LICENCE, record_date="",
                             split=str(adm["split"]), tier="mimic", meta={**meta, "role": role}, variant=role))
    return record, rows


def stage_run(name: str, pool: Iterator[dict[str, Any]], target: int | None, max_attempts: int | None, deadline: float,
              corpus: Corpus, chats: Chats, seed: int, mode: str, attempts_path: Path, rows_path: Path,
              done: set[int]) -> dict[str, Any]:
    """Run admissions from ``pool`` through the pair pipeline until ``target`` pairs are kept (build), ``max_attempts``
    admissions have been attempted (pilot), the deadline passes, or the pool is exhausted."""
    kept = 0
    submitted = 0
    outcomes: Counter[str] = Counter()
    errors = 0
    limit = max_attempts if max_attempts is not None else 10**12
    with attempts_path.open("a") as att, rows_path.open("a") as rows_out, ThreadPoolExecutor(WORKERS) as pool_exec:
        futures: set[Future] = set()
        exhausted = False
        while True:
            while (not exhausted and len(futures) < 2 * WORKERS and submitted < limit
                   and (target is None or kept < target) and time.time() < deadline):
                adm = next(pool, None)
                if adm is None:
                    exhausted = True
                    break
                if int(adm["hadm_id"]) in done:
                    continue
                submitted += 1
                futures.add(pool_exec.submit(process, adm, name, corpus, chats, seed, mode))
            if not futures:
                break
            finished, futures = wait(futures, return_when=FIRST_COMPLETED)
            for fut in finished:
                record, rows = fut.result()
                if record["outcome"].startswith("teacher error"):
                    errors += 1
                if rows and target is not None and kept >= target:
                    record = {**record, "outcome": "not used: stage target reached"}
                    rows = []
                if rows:
                    kept += 1
                    for row in rows:
                        rows_out.write(json.dumps(row) + "\n")
                outcomes[record["outcome"]] += 1
                att.write(json.dumps(record) + "\n")
            att.flush()
            rows_out.flush()
            if errors > 20 and errors * 20 > sum(outcomes.values()):
                raise SystemExit(f"teacher errors above 5 % in stage {name}: stopping")
    return {"stage": name, "kept": kept, "target": target, "attempts": sum(outcomes.values()), "outcomes": dict(outcomes)}


def read_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {int(json.loads(line)["hadm_id"]) for line in path.read_text().splitlines() if line.strip()}


def cmd_pilot(args: argparse.Namespace) -> None:
    run = OUT / f"pilot_{args.mode}"
    run.mkdir(parents=True, exist_ok=True)
    (PAIRS / f"pilot_{args.mode}").mkdir(parents=True, exist_ok=True)
    attempts_path = run / "attempts.jsonl"
    rows_path = PAIRS / f"pilot_{args.mode}" / "rows.jsonl"
    done = read_done(attempts_path)
    corpus = Corpus()
    chats = Chats(reasoning_select=args.mode == "on", select_max_tokens=4096 if args.mode == "on" else 1024)
    started = time.time()
    pool = iter(corpus.pool("train", seed=args.seed))
    summary = stage_run("pilot", pool, None, args.attempts - len(done), time.time() + 6 * 3600, corpus, chats,
                        args.seed, args.mode, attempts_path, rows_path, done)
    wall = time.time() - started
    summary |= {"mode": args.mode, "wall_s": round(wall, 1), "usage": chats.usage(), "git_commit": git_commit(),
                "finished_utc": utcnow()}
    write_json(run / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def cmd_build(args: argparse.Namespace) -> None:
    run = OUT / f"build_{args.mode}"
    run.mkdir(parents=True, exist_ok=True)
    (PAIRS / f"build_{args.mode}").mkdir(parents=True, exist_ok=True)
    corpus = Corpus()
    chats = Chats(reasoning_select=args.mode == "on", select_max_tokens=4096 if args.mode == "on" else 1024)
    deadline = time.time() + args.hours * 3600
    started = time.time()
    results = []
    excluded = set()
    for pilot_attempts in (OUT / "pilot_off" / "attempts.jsonl", OUT / "pilot_on" / "attempts.jsonl"):
        excluded |= read_done(pilot_attempts)
    attempts_path = run / "attempts.jsonl"
    done = read_done(attempts_path) | excluded
    for stage in ("dev", "test", "train"):
        rows_path = PAIRS / f"build_{args.mode}" / f"rows_{stage}.jsonl"
        pool = iter(corpus.pool(stage, seed=args.seed))
        results.append(stage_run(stage, pool, TARGETS[stage], None, deadline, corpus, chats, args.seed, args.mode,
                                 attempts_path, rows_path, done))
        done = read_done(attempts_path) | excluded
    wall = time.time() - started
    summary = {"mode": args.mode, "stages": results, "wall_s": round(wall, 1), "gpu_hours": round(wall / 3600, 3),
               "usage": chats.usage(), "git_commit": git_commit(), "finished_utc": utcnow()}
    write_json(run / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def cmd_pilot_report(args: argparse.Namespace) -> None:
    """The pre-registered mode choice (ADVISORY Y3 §7): among modes whose keep rate is at least 40 %, the one with more kept
    pairs per GPU-hour. The keep rate counts only admissions that had a target code (the others are not candidates)."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline

    report: dict[str, Any] = {}
    for mode in ("off", "on"):
        run = OUT / f"pilot_{mode}"
        rows = [json.loads(line) for line in (run / "attempts.jsonl").read_text().splitlines() if line.strip()]
        summary = json.loads((run / "summary.json").read_text())
        candidates = [r for r in rows if r["outcome"] != "no eligible target code"]
        kept = [r for r in rows if r["outcome"] == "kept"]
        shares = [r["n_selected"] / r["n_sentences"] for r in kept]
        pilot_rows = [json.loads(line) for line in (PAIRS / f"pilot_{mode}" / "rows.jsonl").read_text().splitlines()
                      if line.strip()]
        texts = [r["state"] for r in pilot_rows]
        labels = [1 if r["meta"]["role"] == "edit" else 0 for r in pilot_rows]
        cue = None
        if min(labels.count(0), labels.count(1)) >= 5:
            # the two members of a pair share most of their text: folds must keep each pair together, or the model is
            # tested on the twin of a note it was trained on (a leak that drives the AUC below 0.5)
            groups = [r["meta"]["pair_id"] for r in pilot_rows]
            model = make_pipeline(CountVectorizer(min_df=2), MultinomialNB())
            cue = float(np.mean(cross_val_score(model, texts, labels, groups=groups, cv=GroupKFold(n_splits=5),
                                                scoring="roc_auc")))
        gpu_hours = summary["wall_s"] / 3600
        report[mode] = {
            "attempts": len(rows), "candidates": len(candidates), "kept": len(kept),
            "keep_rate": round(len(kept) / len(candidates), 4) if candidates else 0.0,
            "outcomes": dict(Counter(r["outcome"] for r in rows)), "wall_s": summary["wall_s"],
            "gpu_hours": round(gpu_hours, 4), "kept_per_gpu_hour": round(len(kept) / gpu_hours, 1) if gpu_hours else 0.0,
            "share_selected_mean": round(float(np.mean(shares)), 4) if shares else None,
            "share_selected_median": round(float(np.median(shares)), 4) if shares else None,
            "calls_per_attempt": round(sum(r.get("calls", 0) for r in rows) / len(rows), 2),
            "usage": summary["usage"], "pilot_cue_auc_5fold": None if cue is None else round(cue, 4),
            "pilot_pair_rows": len(pilot_rows),
        }
    qualifying = [m for m in ("off", "on") if report[m]["keep_rate"] >= 0.40]
    report["rule"] = "modes with keep rate >= 0.40; the higher kept pairs per GPU-hour"
    report["qualifying_modes"] = qualifying
    report["choice"] = max(qualifying, key=lambda m: report[m]["kept_per_gpu_hour"]) if qualifying else None
    write_json(OUT / "pilot_report.json", report)
    print(json.dumps(report, indent=2))


def cmd_cue(args: argparse.Namespace) -> None:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import roc_auc_score
    from sklearn.naive_bayes import MultinomialNB

    def load(path: Path) -> tuple[list[str], list[int]]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [r["state"] for r in rows], [1 if r["meta"]["role"] == "edit" else 0 for r in rows]

    train_x, train_y = load(PAIRS / args.build / "rows_train.jsonl")
    dev_x, dev_y = load(PAIRS / args.build / "rows_dev.jsonl")
    vectorizer = CountVectorizer(lowercase=True, min_df=3, max_features=50_000)
    model = MultinomialNB(alpha=1.0).fit(vectorizer.fit_transform(train_x), train_y)
    scores = model.predict_proba(vectorizer.transform(dev_x))[:, 1]
    auc = float(roc_auc_score(dev_y, scores))
    report = {"build": args.build, "train_items": len(train_x), "dev_items": len(dev_x), "cue_auc_dev": round(auc, 4),
              "gate": "AUC < 0.75 on dev (ADVISORY Y3 §6)", "passes": auc < 0.75,
              "dev_positive_share": round(float(np.mean(dev_y)), 4)}
    write_json(OUT / args.build / "cue_check.json", report)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--mode", choices=("off", "on"), required=True)
    p.add_argument("--attempts", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_pilot)
    b = sub.add_parser("build")
    b.add_argument("--mode", choices=("off", "on"), required=True)
    b.add_argument("--hours", type=float, default=8.0)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_build)
    r = sub.add_parser("pilot-report")
    r.set_defaults(func=cmd_pilot_report)
    rep = sub.add_parser("report")
    rep.add_argument("--mode", choices=("off", "on"), required=True)
    rep.set_defaults(func=cmd_report)
    c = sub.add_parser("cue")
    c.add_argument("--build", required=True)
    c.set_defaults(func=cmd_cue)
    args = parser.parse_args()
    args.func(args)



def cmd_report(args: argparse.Namespace) -> None:
    """Write loops/osler_v4/crafted.md from the pilot, build and cue artifacts (aggregates only; no text)."""
    pilot = json.loads((OUT / "pilot_report.json").read_text()) if (OUT / "pilot_report.json").exists() else None
    build_dir = OUT / f"build_{args.mode}"
    build = json.loads((build_dir / "summary.json").read_text()) if (build_dir / "summary.json").exists() else None
    cue = json.loads((build_dir / "cue_check.json").read_text()) if (build_dir / "cue_check.json").exists() else None
    lines = [
        "# Nemotron-crafted minimal pairs — osler_v4 Y3 (aggregates only)",
        "",
        "Generated by `scripts/osler_v4/y3_pairs.py report` from the pilot, build and cue artifacts. Counts and rates only: no MIMIC",
        "text, values or identifiers. The teacher is the local Nemotron 3.5 Super server (D45); every call carried MIMIC text and",
        "went through the loopback guard. Pairs are `mimic_doc_support_v1`: an original note (documented) and its edit (not).",
        "",
        "## 1. Pilot (ADVISORY Y3 §7; 200 train admissions per mode, the same admissions in both modes)",
        "",
    ]
    if pilot:
        lines += ["| mode | attempts | with a target | kept | keep rate | kept per GPU-hour | GPU-hours | share of sentences deleted (mean / median, kept) | calls per attempt | pilot cue AUC (5-fold, grouped by pair) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for mode in ("off", "on"):
            p = pilot[mode]
            lines.append(f"| {mode} | {p['attempts']} | {p['candidates']} | {p['kept']} | {p['keep_rate']:.4f} | "
                         f"{p['kept_per_gpu_hour']} | {p['gpu_hours']} | {p['share_selected_mean']} / {p['share_selected_median']} | "
                         f"{p['calls_per_attempt']} | {p['pilot_cue_auc_5fold']} |")
        lines += ["", "Outcomes per mode (no silent drops; every attempt has one outcome):", ""]
        for mode in ("off", "on"):
            lines.append(f"- {mode}: " + "; ".join(f"{k} {v}" for k, v in sorted(pilot[mode]["outcomes"].items(), key=lambda kv: -kv[1])))
        lines += ["", f"Rule (pre-registered): {pilot['rule']}. Modes qualifying: {pilot['qualifying_modes'] or 'none'}. "
                  f"Choice by the rule: {pilot['choice'] or 'none (the rule is silent)'}.", ""]
    lines += ["## 2. Mode used for the build", "", f"Mode: **{args.mode}** (reasoning in the selection call: {'on' if args.mode == 'on' else 'off'}; "
              "verification calls are always reasoning-off). See STATE §7 for the deviation: no mode reached the 40 % keep rate, and "
              "reasoning off has the higher keep rate and the higher kept pairs per GPU-hour.", ""]
    if build:
        lines += ["## 3. Build", "", "| stage | target | kept | attempts | outcomes |", "|---|---|---|---|---|"]
        for s in build["stages"]:
            lines.append(f"| {s['stage']} | {s['target']} | {s['kept']} | {s['attempts']} | "
                         + "; ".join(f"{k} {v}" for k, v in sorted(s["outcomes"].items(), key=lambda kv: -kv[1])) + " |")
        lines += ["", f"Wall-clock {build['wall_s']} s = {build['gpu_hours']} GPU-hours. Usage: {build['usage']}.", ""]
    if cue:
        lines += ["## 4. Cue check (ADVISORY Y3 §6)", "",
                  f"Naive Bayes on note text (no question), trained on train pairs (edit vs original), ROC AUC on dev pairs: "
                  f"**{cue['cue_auc_dev']}** on {cue['dev_items']} dev items (gate: AUC < 0.75). "
                  f"{'Passes.' if cue['passes'] else 'FAILS: the template is BLOCKED unless one fix passes.'}", ""]
    write_json(OUT / "crafted_report.json", {"pilot": pilot, "build": build, "cue": cue})
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT}")


REPORT = Path("loops/osler_v4/crafted.md")


if __name__ == "__main__":
    main()
