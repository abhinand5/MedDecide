"""Unit tests for the S9 training CLI (``scripts/bench/train_student.py``).

Every test is CPU-only: no GPU, no checkpoint, no real training. The acceptance-relevant
behaviour is factored into pure functions plus :class:`DevSelector`, which takes any object with
``evaluate``/``save_checkpoint``/``model.train_mode`` — so the selection rule, the dev-eval log,
the best-checkpoint artifact and the "a failed eval must not kill the run" path are all exercised
against a fake trainer.

Covered:

* the selection rule is Brier-first (macro accuracy is the tie-break, then the earlier step) and
  the trajectory fold agrees with it;
* the dev sample digest is order-independent and content-sensitive;
* the loader refuses a file whose items are not of the expected split (training reads no test);
* per-template metrics sum to the total and Brier is weighted correctly across option counts;
* applying a temperature changes Brier but not accuracy/macro accuracy;
* only verdicts marked ``fitted`` are applied;
* the S9 recipe is pinned: batch 8, 8k prompt cap, LoRA r=16, LoRA lr 2e-4, one epoch, per-step
  logging, dev eval every 500 steps;
* ``--dry-run`` runs the whole data path (read, split check, stratified sample, pidfile) and exits
  without touching the model.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from meddecide.bench.schema import Split, Tier, make_item
from meddecide.model.meddecide_model import ScoredItems

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "bench" / "train_student.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("train_student", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


ts = _load_cli()


# --------------------------------------------------------------------------- fixtures/helpers
class FakeItem:
    """The item attributes the CLI's metrics and split check read (no schema validation)."""

    def __init__(
        self,
        item_id: str,
        *,
        template_id: str = "t_choice",
        source: str = "clinicaltrials",
        qtype: str = "choice",
        split: Split = Split.DEV,
    ) -> None:
        self.item_id = item_id
        self.template_id = template_id
        self.source = source
        self.qtype = qtype
        self.split = split


def scored_items(rows: list[tuple[str, str, str, str, list[str], list[float], int]]) -> ScoredItems:
    """Build a :class:`ScoredItems` from ``(item_id, template, source, qtype, keys, probs, gold)``."""
    items, keys, probs, gold = [], [], [], []
    for item_id, template, source, qtype, option_keys, option_probs, gold_index in rows:
        items.append(
            FakeItem(item_id, template_id=template, source=source, qtype=qtype, split=Split.DEV)
        )
        keys.append(list(option_keys))
        probs.append(np.asarray(option_probs, dtype=np.float64))
        gold.append(int(gold_index))
    return ScoredItems(
        items=items,
        option_keys=keys,
        original_option_keys=[list(k) for k in keys],
        logits=[np.log(np.asarray(p, dtype=np.float64)) for p in probs],
        probs=probs,
        gold_indices=gold,
        prompt_tokens=[10] * len(items),
        latency_s=[0.01] * len(items),
        wall_clock_s=0.1,
        batch_sizes=[len(items)],
    )


def metrics_row(*, step: int, brier: float, macro: float, accuracy: float = 0.5) -> dict:
    return {
        "step": step,
        "metrics": {
            "n": 545,
            "n_correct": int(accuracy * 545),
            "accuracy": accuracy,
            "macro_accuracy": macro,
            "brier": brier,
            "mean_nll": 0.5,
            "majority_baseline": 0.3,
        },
    }


class FakeModel:
    def __init__(self) -> None:
        self.train_mode_calls = 0

    def train_mode(self) -> None:
        self.train_mode_calls += 1


class FakeTrainer:
    """Just enough Trainer for :class:`DevSelector`: a metrics script and a save log."""

    def __init__(self, metrics: list[dict], *, fail_on: int | None = None,
                 save_error: str | None = None) -> None:
        self.metrics = list(metrics)
        self.fail_on = fail_on
        self.save_error = save_error
        self.model = FakeModel()
        self.n_evals = 0
        self.saved: list[str] = []

    def evaluate(self, items, batch_size=None) -> dict:
        index = self.n_evals
        self.n_evals += 1
        if self.fail_on is not None and index == self.fail_on:
            raise ValueError("synthetic eval failure")
        return self.metrics[min(index, len(self.metrics) - 1)]

    def save_checkpoint(self, path) -> Path:
        if self.save_error:
            raise OSError(self.save_error)
        self.saved.append(str(path))
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)


def selector_for(tmp_path: Path, trainer: FakeTrainer, **kwargs) -> object:
    return ts.DevSelector(
        trainer,
        [FakeItem("a"), FakeItem("b")],
        eval_every=kwargs.pop("eval_every", 500),
        evals_path=tmp_path / "logs" / "dev_evals.jsonl",
        best_dir=tmp_path / "best",
        best_json_path=tmp_path / "best.json",
        eval_sample={"per_qtype": 256, "path": str(tmp_path / "dev_eval_ids.json"),
                     "sha256": "deadbeef"},
        model_id="meddecide-0.8b-lora-pointer",
        print_fn=lambda _: None,
        **kwargs,
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------- the selection rule
def test_brier_is_the_primary_key_not_macro_accuracy() -> None:
    worse_brier_better_macro = {"brier": 0.31, "macro_accuracy": 0.90}
    better_brier_worse_macro = {"brier": 0.30, "macro_accuracy": 0.40}
    assert ts.is_better(better_brier_worse_macro, worse_brier_better_macro)
    assert not ts.is_better(worse_brier_better_macro, better_brier_worse_macro)


def test_tie_on_brier_breaks_on_macro_accuracy() -> None:
    assert ts.is_better({"brier": 0.30, "macro_accuracy": 0.60}, {"brier": 0.30, "macro_accuracy": 0.50})
    assert not ts.is_better(
        {"brier": 0.30, "macro_accuracy": 0.50}, {"brier": 0.30, "macro_accuracy": 0.60}
    )


def test_exact_tie_keeps_the_earlier_step() -> None:
    assert ts.best_eval([]) is None
    rows = [
        metrics_row(step=500, brier=0.30, macro=0.50),
        metrics_row(step=1000, brier=0.30, macro=0.50),
    ]
    best = ts.best_eval(rows)
    assert best is not None and best["step"] == 500


def test_best_eval_matches_a_manual_minimum() -> None:
    rows = [
        metrics_row(step=500, brier=0.40, macro=0.50),
        metrics_row(step=1000, brier=0.35, macro=0.45),
        metrics_row(step=1500, brier=0.35, macro=0.55),
        metrics_row(step=2000, brier=0.36, macro=0.99),
    ]
    best = ts.best_eval(rows)
    manual = min(rows, key=lambda r: ts.selection_key(r["metrics"]))
    assert best is not None and best["step"] == manual["step"] == 1500


# --------------------------------------------------------------------------- sample identity
def test_sample_digest_is_order_independent_and_content_sensitive() -> None:
    a = [FakeItem("x"), FakeItem("y")]
    assert ts.sample_digest(a) == ts.sample_digest(list(reversed(a)))
    assert ts.sample_digest(a) != ts.sample_digest([FakeItem("x"), FakeItem("z")])


def test_check_split_refuses_a_test_item_in_the_training_file() -> None:
    with pytest.raises(ValueError, match="never reads a test split"):
        ts.check_split([FakeItem("a"), FakeItem("b", split=Split.TEST)], "train", path="train.jsonl")
    ts.check_split([FakeItem("a", split=Split.TRAIN)], "train", path="train.jsonl")
    with pytest.raises(ValueError, match="expected only split='dev'"):
        ts.check_split([FakeItem("a", split=Split.TRAIN)], "dev", path="dev.jsonl")


# --------------------------------------------------------------------------- dev metrics
def test_grouped_metrics_denominators_and_brier_weighting() -> None:
    # template A: 3 items, 2 options; template B: 3 items, two of them with 4 options
    rows = [
        ("a1", "tA", "s1", "noul", ["no", "yes"], [0.8, 0.2], 0),
        ("a2", "tA", "s1", "noul", ["no", "yes"], [0.7, 0.3], 1),
        ("a3", "tA", "s1", "noul", ["no", "yes"], [0.6, 0.4], 0),
        ("b1", "tB", "s2", "choice", ["w", "x", "y", "z"], [0.4, 0.3, 0.2, 0.1], 0),
        ("b2", "tB", "s2", "choice", ["w", "x", "y", "z"], [0.1, 0.2, 0.3, 0.4], 3),
        ("b3", "tB", "s2", "choice", ["no", "yes"], [0.5, 0.5], 1),
    ]
    scored = scored_items(rows)
    grouped = ts.grouped_metrics(scored, group_by="template_id")
    assert sorted(grouped) == ["tA", "tB"]
    assert sum(v["n"] for v in grouped.values()) == len(rows)  # parts sum to the total (R5)
    assert grouped["tA"]["n"] == 3 and grouped["tA"]["n_correct"] == 2
    assert grouped["tA"]["accuracy"] == pytest.approx(2 / 3)
    # Brier for tB: two 4-option items and one 2-option item, weighted explicitly
    wide = np.stack([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
    wide_brier = float(np.mean(np.sum((wide - np.eye(4)[[0, 3]]) ** 2, axis=1)))
    narrow_brier = float(np.sum((np.asarray([0.5, 0.5]) - np.eye(2)[1]) ** 2))
    assert grouped["tB"]["brier"] == pytest.approx((2 * wide_brier + narrow_brier) / 3)
    assert grouped["tB"]["n_options_max"] == 4
    assert grouped["tB"]["qtype"] == "choice" and grouped["tB"]["source"] == "s2"
    assert 0.0 <= grouped["tB"]["ece"] <= 1.0


def test_grouped_metrics_rejects_an_unknown_grouping() -> None:
    with pytest.raises(ValueError, match="no attribute"):
        ts.grouped_metrics(scored_items([
            ("a1", "tA", "s1", "noul", ["no", "yes"], [0.5, 0.5], 0),
        ]), group_by="not_a_field")


def test_temperature_changes_brier_but_not_accuracy() -> None:
    rows = [
        ("a1", "tA", "s1", "noul", ["no", "yes"], [0.9, 0.1], 0),
        ("a2", "tA", "s1", "noul", ["no", "yes"], [0.6, 0.4], 0),
        ("a3", "tA", "s2", "noul", ["no", "yes"], [0.2, 0.8], 1),
    ]
    scored = scored_items(rows)
    flat = ts.dev_report(scored, temperature={})
    sharp = ts.dev_report(scored, temperature={"noul": 0.5})
    assert flat["overall"]["accuracy"] == sharp["overall"]["accuracy"]
    assert flat["overall"]["macro_accuracy"] == sharp["overall"]["macro_accuracy"]
    assert sharp["overall"]["brier"] < flat["overall"]["brier"]  # T<1 sharpens the right answers
    assert sum(v["n"] for v in sharp["per_template"].values()) == sharp["overall"]["n"]
    assert sum(v["n"] for v in sharp["per_qtype"].values()) == sharp["overall"]["n"]


def test_only_fitted_temperatures_are_applied() -> None:
    verdict = {
        "per_qtype": {
            "choice": {"temperature": 0.91, "fitted": True},
            "noul": {"temperature": 0.04, "fitted": False, "status": "NOT FITTED — lower bound"},
            "score": {"temperature": None, "fitted": False, "status": "NOT FITTED — too few"},
        }
    }
    assert ts.applied_temperatures(verdict) == {"choice": 0.91}


# --------------------------------------------------------------------------- the S9 recipe
def test_s9_recipe_is_pinned() -> None:
    args = ts.parse_args(["--config", str(REPO / "configs" / "student_v0.yaml"),
                          "--eval-every", "500"])
    config = ts.build_config(args)
    assert config.batch_size == 8
    assert config.max_batch_tokens == 8192
    assert config.max_prompt_tokens == 8192
    assert config.epochs == 1
    assert config.lora.r == 16 and config.lora.alpha == 32
    assert config.lora_lr == 2.0e-4  # the brief's "lr 2e-4" is the decision-path/adapter LR
    assert config.lr == 1.0e-3  # the head's own LR, unchanged from the measured S7 config
    assert config.head.option_state == "key_end"
    assert config.lambda_brier == 1.0
    assert config.shuffle_options is True
    assert config.log_every == 1  # a per-step loss/lr/tokens/s/memory log
    assert config.eval_every == 500
    assert config.eval_items == 256
    assert config.fit_temperature is True and config.temperature_per_qtype is True
    assert config.train_path.endswith("data/train/student_v0/train.jsonl")
    assert config.dev_path.endswith("data/train/student_v0/dev.jsonl")


def test_cli_defaults() -> None:
    args = ts.parse_args([])
    assert args.out == Path("outputs/student_v0/S9")
    assert args.eval_every == 500 and args.eval_per_qtype == 256
    assert args.max_seconds is None and args.train_limit is None
    assert args.dry_run is False


def test_throughput_of_reports_items_tokens_and_peak_memory() -> None:
    from types import SimpleNamespace

    history = [
        SimpleNamespace(elapsed_s=1.0, n_items=8, real_tokens=1000, padded_tokens=1200,
                        gpu_peak_gb=20.0),
        SimpleNamespace(elapsed_s=1.0, n_items=8, real_tokens=2000, padded_tokens=2200,
                        gpu_peak_gb=32.1),
    ]
    rates = ts.throughput_of(history)
    assert rates["items_per_s"] == pytest.approx(8.0)
    assert rates["tokens_per_s"] == pytest.approx(1500.0)
    assert rates["gpu_peak_memory_gb"] == 32.1
    assert ts.throughput_of([]) == {"steps": 0}


# --------------------------------------------------------------------------- DevSelector
def test_dev_selector_keeps_the_best_by_brier_and_writes_its_artifacts(tmp_path: Path) -> None:
    trainer = FakeTrainer([
        {"n": 545, "accuracy": 0.50, "macro_accuracy": 0.50, "brier": 0.40, "mean_nll": 0.7},
        {"n": 545, "accuracy": 0.55, "macro_accuracy": 0.55, "brier": 0.35, "mean_nll": 0.6},
        {"n": 545, "accuracy": 0.54, "macro_accuracy": 0.54, "brier": 0.36, "mean_nll": 0.61},
        {"n": 545, "accuracy": 0.56, "macro_accuracy": 0.60, "brier": 0.35, "mean_nll": 0.59},
    ])
    selector = selector_for(tmp_path, trainer)

    selector(500)
    selector(501)  # not a multiple of eval_every: no eval
    selector(1000)
    selector(1500)
    selector(2000)

    # steps 500, 1000 and 2000 improve on the run so far (1500 is worse); each one is saved
    assert trainer.saved == [str(tmp_path / "best")] * 3
    assert selector.best_step == 2000
    assert trainer.model.train_mode_calls == 4  # the optimiser loop is handed back in train mode
    rows = read_jsonl(tmp_path / "logs" / "dev_evals.jsonl")
    evals = [r for r in rows if r["event"] == "dev_eval"]
    assert [r["step"] for r in evals] == [500, 1000, 1500, 2000]
    assert [r["selected"] for r in evals] == [True, True, False, True]
    saved_rows = [r for r in rows if r["event"] == "checkpoint_saved"]
    assert [r["step"] for r in saved_rows] == [500, 1000, 2000]  # one row per improvement
    best = json.loads((tmp_path / "best.json").read_text(encoding="utf-8"))
    assert best["step"] == 2000
    assert best["selection_rule"] == ts.SELECTION_RULE
    assert best["dev_metrics_at_selection"]["brier"] == 0.35
    assert best["dev_eval"]["eval_every"] == 500
    assert best["additional_analysis"]["macro_accuracy_best_step"] == 2000
    trajectory = selector.trajectory()
    assert [t["step"] for t in trajectory] == [500, 1000, 1500, 2000]


def test_dev_selector_records_a_failed_eval_and_keeps_training(tmp_path: Path) -> None:
    trainer = FakeTrainer(
        [{"n": 545, "accuracy": 0.5, "macro_accuracy": 0.5, "brier": 0.4, "mean_nll": 0.7}],
        fail_on=0,
    )
    selector = selector_for(tmp_path, trainer)
    row = selector.evaluate(500)  # `__call__` is the hook; `evaluate` returns the row
    assert row["event"] == "dev_eval_error" and "synthetic eval failure" in row["error"]
    assert selector.best_step is None and selector.n_errors == 1
    assert trainer.saved == []
    assert trainer.model.train_mode_calls == 1  # restored even on the failure path
    assert not (tmp_path / "best.json").exists()
    # a later eval still selects normally
    selector(1000)
    assert selector.best_step == 1000 and selector.n_errors == 1


def test_dev_selector_records_a_checkpoint_save_failure(tmp_path: Path) -> None:
    trainer = FakeTrainer(
        [{"n": 545, "accuracy": 0.5, "macro_accuracy": 0.5, "brier": 0.4, "mean_nll": 0.7}],
        save_error="disk full",
    )
    selector = selector_for(tmp_path, trainer)
    selector(500)
    assert selector.best_step == 500  # the step is still the best one measured
    assert selector.save_error is not None and "disk full" in selector.save_error
    best = json.loads((tmp_path / "best.json").read_text(encoding="utf-8"))
    assert best["save_error"] is not None and "disk full" in best["save_error"]


# --------------------------------------------------------------------------- the data path
def _write_split(path: Path, n: int, split: Split, *, template_id: str) -> None:
    rows = []
    for i in range(n):
        item = make_item(
            tier=Tier.ESTABLISHED,
            source="clinicaltrials",
            source_record_id=f"rec-{split.value}-{i}",
            source_url="https://example.org/record",
            source_license="CC0",
            record_date="2025-01-01",
            split=split,
            template_id=template_id,
            skill="trial",
            qtype="noul",
            state=f"state {i}",
            question="question?",
            options=[{"key": "yes", "label": "yes"}, {"key": "no", "label": "no"}],
            gold="yes",
            option_order_seed=0,
        )
        rows.append(item.model_dump_json())
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_main_dry_run_covers_the_data_path_without_a_model(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    _write_split(train_path, 6, Split.TRAIN, template_id="t_a")
    _write_split(dev_path, 5, Split.DEV, template_id="t_b")
    out = tmp_path / "S9"
    code = ts.main([
        "--config", str(REPO / "configs" / "student_v0.yaml"),
        "--train-path", str(train_path),
        "--dev-path", str(dev_path),
        "--out", str(out),
        "--eval-per-qtype", "2",
        "--dry-run",
    ])
    assert code == 0
    started = json.loads((out / "dev_eval_ids.json").read_text(encoding="utf-8"))
    assert started["n_items"] == 2 and started["sha256"]
    assert started["item_ids"] == sorted(started["item_ids"])  # written in a stable order
    assert not (out / "run.json").exists()
    assert not (out / "best.json").exists()
    assert not (out / "train.pid").exists()  # the pidfile is not left behind


def test_main_dry_run_refuses_a_test_item_in_the_training_file(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    _write_split(train_path, 3, Split.TEST, template_id="t_a")
    dev_path = tmp_path / "dev.jsonl"
    _write_split(dev_path, 2, Split.DEV, template_id="t_b")
    with pytest.raises(ValueError, match="never reads a test split"):
        ts.main([
            "--config", str(REPO / "configs" / "student_v0.yaml"),
            "--train-path", str(train_path),
            "--dev-path", str(dev_path),
            "--out", str(tmp_path / "S9b"),
            "--dry-run",
        ])
