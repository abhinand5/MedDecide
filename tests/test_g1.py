"""Unit tests for the S11 G1 evaluation machinery (``scripts/bench/g1.py``, decision D16).

Every test is CPU-only, tiny, and hand-checkable: a miniature benchmark (two templates, a few
dated items) and miniature prediction logs are written to ``tmp_path``, so the statistics,
the intersection rule, the D12 exclusion and the item-set construction are all exercised
without touching the real benchmark or any model.

Covered (the S11 acceptance list):

* paired bootstrap where one model dominates every item gives a CI strictly above 0;
* two identical prediction files give a difference of exactly 0 and a CI containing 0;
* the paired CI is narrower than the unpaired one when the two models' errors are correlated;
* the intersection rule drops items one model did not score, and the count is stated;
* a D12-failing cell is reported as ``READOUT_FAIL — <check>`` and excluded from the verdict;
* the pointer head's letter-readout-only checks are ``NOT APPLICABLE``, never passed, while
  the constant-answer check is applied;
* the strict-slice and held-out splits are computed from dates and template ids;
* missing inputs produce ``NOT MEASURED — <reason>`` instead of a crash or a fabricated number.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "g1.py"


def _load_g1():
    spec = importlib.util.spec_from_file_location("g1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


g1 = _load_g1()

HELD_OUT = "ct_arm_role_noul_v1"  # a real D14 held-out template id
_ITEM_SEQ = [0]


# --------------------------------------------------------------------------- helpers
def make_item(
    template_id: str,
    record_date: str,
    *,
    gold: str = "A",
    split: str = "test",
    tier: str = "fresh",
    n_options: int = 2,
    item_id: str | None = None,
) -> dict:
    _ITEM_SEQ[0] += 1
    keys = [chr(ord("A") + i) for i in range(n_options)]
    return {
        "item_id": item_id or f"item{_ITEM_SEQ[0]:04d}",
        "tier": tier,
        "source": "unit-test",
        "source_record_id": "r",
        "source_url": "https://example.invalid",
        "source_license": "public-domain",
        "record_date": record_date,
        "split": split,
        "template_id": template_id,
        "skill": "unit",
        "qtype": "choice",
        "state": "state",
        "question": "question",
        "options": [{"key": k, "label": k, "description": None} for k in keys],
        "gold": gold,
        "meta": {"strict_post_teacher": record_date >= "2026-09-10"},
    }


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def make_row(
    item: dict,
    *,
    correct: bool,
    argmax: str | None = None,
    label_mass: float = 1.0,
    variant: str = "pointer-head",
    run_id: str = "run1",
    model_id: str = "unit-model",
) -> dict:
    keys = [o["key"] for o in item["options"]]
    gold = item["gold"]
    argmax = argmax or (gold if correct else next(k for k in keys if k != gold))
    if correct:
        option_probs = [1.0 if k == gold else 0.0 for k in keys]
    else:
        option_probs = [
            0.0 if k == gold else 1.0 / (len(keys) - 1) for k in keys
        ]
    return {
        "item_id": item["item_id"],
        "model_id": model_id,
        "run_id": run_id,
        "source": item["source"],
        "template_id": item["template_id"],
        "split": item["split"],
        "qtype": item["qtype"],
        "option_keys": keys,
        "original_option_keys": keys,
        "option_probs": option_probs,
        "label_mass": label_mass,
        "argmax_key": argmax,
        "gold_key": gold,
        "correct": bool(correct),
        "expected_level": None,
        "latency_s": 0.01,
        "variant": variant,
        "prompt_tokens": 10,
        "transform": {"readout": variant} if variant == "pointer-head" else {},
    }


def write_bench(
    root: Path,
    items: list[dict],
    *,
    strict_start: str = "2026-09-10",
    screen_drop: tuple[str, ...] = (),
) -> Path:
    """A miniature v0.2-shaped benchmark tree; returns the ``--bench-dir`` path."""
    bench = root / "bench"
    fresh = bench / "fresh"
    write_jsonl(fresh / "items.jsonl", items)
    templates = sorted({item["template_id"] for item in items})
    screen = {
        "templates": [
            {
                "template_id": template,
                "tier": "fresh",
                "source": "unit-test",
                "qtype": "choice",
                "n_test": sum(1 for i in items if i["template_id"] == template),
                "majority_share": 0.5,
                "drop": template in screen_drop,
                "drop_reasons": ["unit-test drop"] if template in screen_drop else [],
            }
            for template in templates
        ]
    }
    (bench / "screen.json").write_text(json.dumps(screen), encoding="utf-8")
    (fresh / "acceptance.json").write_text(
        json.dumps({"verdict": "PASS", "strict_slice_start": strict_start}), encoding="utf-8"
    )
    return bench


def run_cli(tmp_path: Path, items: list[dict], model_rows: dict[str, list[dict]], **kwargs) -> str:
    """Write the miniature benchmark + one prediction file per model, run ``g1.main``."""
    bench = write_bench(tmp_path, items, **kwargs.pop("bench_kwargs", {}))
    argv = ["--bench-dir", str(bench), "--out", str(tmp_path / "g1.md"), "--resamples", "200"]
    for label, rows in model_rows.items():
        argv += ["--models", f"{label}={write_jsonl(tmp_path / f'preds_{label}.jsonl', rows)}"]
    argv += kwargs.pop("argv", [])
    g1.main(argv)
    return (tmp_path / "g1.md").read_text(encoding="utf-8")


def make_vectors(rows: list[tuple[str, str, bool, float]]) -> g1.Vectors:
    """``rows`` = ``(item_id, template_id, correct, p_gold)``; 2-option items, gold index 0."""
    probs = [np.asarray([p, 1.0 - p], dtype=np.float64) for _, _, _, p in rows]
    briers = np.asarray([2.0 * (1.0 - p) ** 2 for _, _, _, p in rows], dtype=np.float64)
    ids = [r[0] for r in rows]
    return g1.Vectors(
        item_ids=ids,
        template_ids=[r[1] for r in rows],
        correct=np.asarray([1.0 if r[2] else 0.0 for r in rows]),
        item_brier=briers,
        probs=probs,
        gold_index=[0] * len(rows),
        predicted=["A" if r[2] else "B" for r in rows],
        gold=["A"] * len(rows),
        n_options=[2] * len(rows),
        index={item_id: i for i, item_id in enumerate(ids)},
        dropped={},
    )


def compare(ref: g1.Vectors, base: g1.Vectors, *, seed: int = 0, n_resamples: int = 200) -> dict:
    ids = list(ref.item_ids)
    return g1.paired_comparison(
        set_name="unit",
        reference="ref",
        baseline="base",
        set_ids=ids,
        common_ids=ids,
        ref=ref,
        base=base,
        seed=seed,
        n_resamples=n_resamples,
    )


def _mini_corpus() -> list[dict]:
    """Two templates, balanced golds, dates on both sides of the strict start."""
    return [
        make_item("t_seen", "2026-05-01", gold="A", item_id="s1"),
        make_item("t_seen", "2026-05-02", gold="B", item_id="s2"),
        make_item("t_seen", "2026-09-15", gold="A", item_id="s3"),
        make_item(HELD_OUT, "2026-05-03", gold="A", item_id="h1"),
        make_item(HELD_OUT, "2026-09-16", gold="B", item_id="h2"),
    ]


def _alternating_base_rows(items: list[dict], *, model_id: str = "base") -> list[dict]:
    """A baseline whose argmax alternates A/B (never a constant-answer cell) and is often right."""
    rows = []
    for i, item in enumerate(items):
        argmax = "A" if i % 2 == 0 else "B"
        rows.append(make_row(item, correct=(argmax == item["gold"]), argmax=argmax, model_id=model_id))
    return rows


# --------------------------------------------------------------------------- statistics
def test_item_brier_matches_metrics_brier_score():
    from meddecide.eval.metrics import brier_score

    probs = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    expected = float(np.mean([g1.item_brier(probs[0], 0), g1.item_brier(probs[1], 2)]))
    assert expected == pytest.approx(brier_score(probs, [0, 2]))


def test_macro_accuracy_resamples_identity_draw_is_the_point_estimate():
    rows = [("i1", "t1", True, 1.0), ("i2", "t1", False, 0.0), ("i3", "t2", True, 1.0)]
    vectors = make_vectors(rows)
    # positions into the comparison order: t1 = [0, 1], t2 = [2]
    identity = {"t1": np.tile(np.asarray([0, 1]), (5, 1)), "t2": np.tile(np.asarray([2]), (5, 1))}
    stats = g1.macro_accuracy_resamples(vectors.correct, identity, ["t1", "t2"])
    point = g1.point_stats(vectors.correct, vectors.item_brier, vectors.template_ids, ["t1", "t2"])
    assert stats.tolist() == pytest.approx([point["macro_accuracy"]] * 5)


def test_mean_brier_resamples_identity_draw_is_the_point_estimate():
    rows = [("i1", "t1", True, 1.0), ("i2", "t1", False, 0.0), ("i3", "t2", True, 0.5)]
    vectors = make_vectors(rows)
    identity = {"t1": np.tile(np.asarray([0, 1]), (4, 1)), "t2": np.tile(np.asarray([2]), (4, 1))}
    stats = g1.mean_brier_resamples(vectors.item_brier, identity, ["t1", "t2"])
    assert stats.tolist() == pytest.approx([float(vectors.item_brier.mean())] * 4)


def test_paired_dominance_gives_ci_strictly_above_zero():
    """A dominates every item: accuracy difference is 1.0 on every resample, Brier diff < 0."""
    rows = [(f"i{i}", "t1" if i < 5 else "t2", True, 1.0) for i in range(10)]
    ref = make_vectors(rows)
    base = make_vectors([(r[0], r[1], False, 0.0) for r in rows])
    comparison = compare(ref, base)
    accuracy = comparison["metrics"]["macro_accuracy"]
    brier = comparison["metrics"]["mean_brier"]
    assert comparison["n_items"] == 10
    assert accuracy["point"] == pytest.approx(1.0)
    assert accuracy["ci95"][0] > 0
    assert accuracy["frac_gt_zero"] == 1.0 and accuracy["frac_lt_zero"] == 0.0
    assert brier["point"] < 0 and brier["ci95"][1] < 0
    verdict, _ = g1.comparison_verdict(comparison)
    assert verdict == "PASS"


def test_identical_models_give_zero_difference_and_ci_containing_zero():
    rows = [
        ("i1", "t1", True, 0.9),
        ("i2", "t1", False, 0.2),
        ("i3", "t2", True, 0.6),
        ("i4", "t2", False, 0.4),
    ]
    ref = make_vectors(rows)
    base = make_vectors(rows)
    comparison = compare(ref, base)
    for key in ("macro_accuracy", "mean_brier"):
        metric = comparison["metrics"][key]
        assert metric["point"] == 0.0
        assert metric["ci95"] == [0.0, 0.0]
        assert metric["frac_eq_zero"] == 1.0
    # equality is not dominance: D16 needs the accuracy CI lower bound strictly above 0
    verdict, reason = g1.comparison_verdict(comparison)
    assert verdict == "FAIL"
    assert "accuracy CI lower bound" in reason


def test_paired_ci_is_narrower_than_unpaired_when_errors_are_correlated():
    """Both models fail on the same items, so the paired difference barely moves."""
    rows: list[tuple[str, str, bool, float]] = []
    for template in ("t1", "t2"):
        for i in range(100):
            if i < 50:  # both wrong on the same 50 items
                rows.append((f"{template}-{i}", template, False, 0.0))
            elif i < 90:  # both right on the same 40
                rows.append((f"{template}-{i}", template, True, 1.0))
            else:  # A right, B wrong on the last 10
                rows.append((f"{template}-{i}", template, True, 1.0))
    ref = make_vectors(rows)
    base = make_vectors(
        [(r[0], r[1], r[2] and int(r[0].split("-")[1]) < 90, r[3]) for r in rows]
    )
    comparison = compare(ref, base)
    accuracy = comparison["metrics"]["macro_accuracy"]
    paired_width = accuracy["ci95"][1] - accuracy["ci95"][0]
    unpaired_width = accuracy["unpaired_ci95"][1] - accuracy["unpaired_ci95"][0]
    assert accuracy["point"] == pytest.approx(0.1)
    assert paired_width < unpaired_width
    assert accuracy["frac_gt_zero"] == 1.0


def test_stratified_resample_indices_are_reproducible_and_within_template():
    templates = ["t2", "t1", "t1", "t2", "t2"]
    draw = g1.stratified_resample_indices(templates, n_resamples=7, seed=3)
    again = g1.stratified_resample_indices(templates, n_resamples=7, seed=3)
    assert set(draw) == {"t1", "t2"}
    assert draw["t1"].shape == (7, 2) and draw["t2"].shape == (7, 3)
    assert np.array_equal(draw["t2"], again["t2"])
    # t1 occupies positions 1 and 2; t2 positions 0, 3 and 4
    assert set(np.unique(draw["t1"])) <= {1, 2}
    assert set(np.unique(draw["t2"])) <= {0, 3, 4}


def test_overall_verdict_fail_dominates_a_missing_baseline():
    comparisons = {
        "s": {
            "b1": {
                "n_items": 4,
                "metrics": {
                    "macro_accuracy": {"ci95": [-0.1, 0.2]},
                    "mean_brier": {"ci95": [-0.3, 0.1]},
                },
            }
        }
    }
    verdict, reason = g1.overall_verdict(comparisons, ["b1", "b2"], "s")
    assert verdict == "FAIL"
    assert "b1" in reason and "b2" in reason
    verdict, _ = g1.overall_verdict({}, ["b1"], "s")
    assert verdict == "NOT MEASURED"


# --------------------------------------------------------------------------- D12
def test_pointer_head_letter_checks_not_applicable_and_constant_answer_applied():
    items = [make_item("t1", "2026-05-01", gold="A" if i % 2 == 0 else "B") for i in range(20)]
    rows = [make_row(item, correct=(i % 2 == 0), argmax="A") for i, item in enumerate(items)]
    cell = g1.gate_cell(
        model="m", tier="fresh", template_id="t1", rows=rows, majority_share=0.5,
        seed=0, n_resamples=200,
    )
    assert cell.readout == "pointer-head"
    assert cell.status == "READOUT_FAIL — constant_answer"
    assert cell.failing_checks == ["constant_answer"]
    assert "constant_answer" in cell.applied_checks
    assert "median_label_mass" not in cell.applied_checks
    assert "median_label_mass" in cell.inapplicable_checks
    assert "greedy_agreement" in cell.inapplicable_checks
    assert cell.inapplicable_checks["median_label_mass"].startswith("NOT APPLICABLE")
    assert "median_label_mass: PASS" not in json.dumps(cell.to_dict())


def test_letter_readout_cell_applies_the_mass_check_and_flags_a_low_mass():
    items = [make_item("t1", "2026-05-01", gold="A" if i % 2 == 0 else "B") for i in range(20)]
    rows = [
        make_row(item, correct=(i % 2 == 0), label_mass=(0.2 if i < 15 else 0.99), variant="bare")
        for i, item in enumerate(items)
    ]
    cell = g1.gate_cell(
        model="m", tier="fresh", template_id="t1", rows=rows, majority_share=0.5,
        seed=0, n_resamples=200,
    )
    assert cell.readout == "letter"
    assert cell.median_label_mass == pytest.approx(0.2)
    assert "median_label_mass" in cell.failing_checks
    assert "READOUT_FAIL — median_label_mass" in cell.status
    assert "greedy_agreement" in cell.not_measured_checks
    assert "greedy_agreement" not in cell.applied_checks


def test_below_chance_cell_fails_the_accuracy_ci_check():
    items = [make_item("t1", "2026-05-01", n_options=4) for _ in range(6)]
    rows = [make_row(item, correct=False) for item in items]
    cell = g1.gate_cell(
        model="m", tier="fresh", template_id="t1", rows=rows, majority_share=0.25,
        seed=0, n_resamples=200,
    )
    assert "accuracy_ci_vs_chance" in cell.failing_checks


def test_gate_agrees_with_evaluate_cell_on_the_shared_checks():
    from meddecide.eval.health import evaluate_cell

    items = [make_item("t1", "2026-05-01", gold="A" if i % 2 == 0 else "B") for i in range(10)]
    rows = [
        make_row(item, correct=(i < 5), argmax="A" if i < 5 else "B", variant="bare")
        for i, item in enumerate(items)
    ]
    cell = g1.gate_cell(
        model="m", tier="fresh", template_id="t1", rows=rows, majority_share=0.5,
        seed=0, n_resamples=200,
    )
    report = evaluate_cell(
        model_id="m",
        template_id="t1",
        label_masses=[r["label_mass"] for r in rows],
        correct=[r["correct"] for r in rows],
        greedy_matches=None,
        n_options=2,
        predicted_labels=[r["argmax_key"] for r in rows],
        majority_share=0.5,
        seed=0,
        n_resamples=200,
    )
    assert cell.accuracy == pytest.approx(report.accuracy)
    assert cell.modal_share == pytest.approx(report.modal_share)
    assert cell.median_label_mass == pytest.approx(report.median_label_mass)
    assert cell.failures == report.failures


# --------------------------------------------------------------------------- item sets
def _index(items: list[dict]) -> dict[str, g1.BenchItem]:
    return {
        item["item_id"]: g1.BenchItem(
            item_id=item["item_id"],
            tier=item["tier"],
            template_id=item["template_id"],
            source=item["source"],
            qtype=item["qtype"],
            record_date=date.fromisoformat(item["record_date"]),
            split=item["split"],
            gold_key=item["gold"],
            n_options=2,
            strict_flag=item["meta"]["strict_post_teacher"],
        )
        for item in items
    }


def test_item_sets_computed_from_dates_and_template_ids():
    items = [
        make_item("t_seen", "2026-05-01", item_id="a"),
        make_item("t_seen", "2026-09-15", item_id="b"),
        make_item(HELD_OUT, "2026-05-02", item_id="c"),
        make_item(HELD_OUT, "2026-09-20", item_id="d"),
        make_item("t_seen", "2026-05-03", split="dev", item_id="e"),
    ]
    index = _index(items)
    screen = {"t_seen": {"drop": False}, HELD_OUT: {"drop": False}}
    sets = g1.build_item_sets(
        index, screen=screen, strict_start=date(2026, 9, 10), include_strict=True, include_tier1=False
    )
    assert sets.sets["fresh"] == ["a", "b", "c", "d"]
    assert sets.sets["fresh_seen"] == ["a", "b"]
    assert sets.sets["fresh_heldout"] == ["c", "d"]
    assert sets.sets["fresh_strict"] == ["b", "d"]
    assert sets.sets["fresh_strict_seen"] == ["b"]
    assert sets.sets["fresh_strict_heldout"] == ["d"]
    assert sets.n_excluded_dev == 1


def test_strict_slice_can_be_switched_off_and_superseded_templates_are_excluded():
    items = [
        make_item("t_seen", "2026-09-15", item_id="a"),
        make_item("pubmed_mesh_major_choice_v1", "2026-09-15", item_id="b"),
        make_item("t_dropped", "2026-09-15", item_id="c"),
    ]
    index = _index(items)
    screen = {
        "t_seen": {"drop": False},
        "pubmed_mesh_major_choice_v1": {"drop": False},
        "t_dropped": {"drop": True},
    }
    sets = g1.build_item_sets(
        index, screen=screen, strict_start=date(2026, 9, 10), include_strict=False, include_tier1=False
    )
    assert sets.sets["fresh"] == ["a"]
    assert sets.sets["fresh_strict"] == []
    assert sets.n_excluded_superseded == 1
    assert sets.n_excluded_not_kept_by_screen == 1
    assert "--no-strict-slice" in " ".join(sets.notes)


# --------------------------------------------------------------------------- end to end
def test_intersection_rule_drops_items_one_model_did_not_score(tmp_path):
    items = _mini_corpus()
    by_id = {i["item_id"]: i for i in items}
    ref_rows = [make_row(item, correct=True) for item in items]
    # the baseline did not score s3, so the comparison must land on the 4 shared items
    base_rows = [
        make_row(by_id[item_id], correct=True, argmax=by_id[item_id]["gold"], model_id="base")
        for item_id in ("s1", "s2", "h1", "h2")
    ]
    text = run_cli(tmp_path, items, {"meddecide": ref_rows, "zeroshot": base_rows, "jev9b": base_rows})
    assert "item count: **n = 4**" in text
    assert "reference scored 5 of the set's items, baseline 4" in text


def test_d12_failing_cell_is_excluded_from_the_verdict(tmp_path):
    items = [
        make_item("t_seen", "2026-05-01", gold="A", item_id="s1"),
        make_item("t_seen", "2026-05-02", gold="B", item_id="s2"),
        make_item("t_seen", "2026-05-03", gold="A", item_id="s3"),
        make_item("t_seen", "2026-05-04", gold="B", item_id="s4"),
        make_item(HELD_OUT, "2026-05-05", gold="A", item_id="h1"),
        make_item(HELD_OUT, "2026-05-06", gold="B", item_id="h2"),
    ]
    # reference: every t_seen answer is "A" (constant answer on a balanced template),
    # correct on both held-out items with different answers (so that cell passes D12)
    ref_rows = []
    for item in items:
        if item["template_id"] == "t_seen":
            ref_rows.append(make_row(item, correct=(item["gold"] == "A"), argmax="A"))
        else:
            ref_rows.append(make_row(item, correct=True))
    base_rows = _alternating_base_rows(items)
    text = run_cli(tmp_path, items, {"meddecide": ref_rows, "zeroshot": base_rows, "jev9b": base_rows})
    assert "READOUT_FAIL — constant_answer" in text
    assert "Cells excluded from the verdict" in text
    assert "D12 failing cell — READOUT_FAIL — constant_answer" in text
    # the whole t_seen template leaves the intersection: only the two held-out items remain
    assert "item count: **n = 2**" in text
    assert "| **G1 on seen templates** (decides the branch of §2) | **NOT MEASURED** |" in text


def test_missing_baseline_file_is_not_measured_and_the_reason_is_named(tmp_path):
    items = _mini_corpus()
    ref_rows = [make_row(item, correct=True) for item in items]
    bench = write_bench(tmp_path, items)
    argv = [
        "--bench-dir", str(bench),
        "--out", str(tmp_path / "g1.md"),
        "--resamples", "100",
        "--models", f"meddecide={write_jsonl(tmp_path / 'preds_ref.jsonl', ref_rows)}",
        "--models", f"jev9b={tmp_path / 'does_not_exist.jsonl'}",
    ]
    g1.main(argv)
    text = (tmp_path / "g1.md").read_text(encoding="utf-8")
    assert "prediction file does not exist" in text
    assert "| **G1 on seen templates** (decides the branch of §2) | **NOT MEASURED** |" in text
    assert "| **NOT MEASURED** |" in text


def test_two_identical_prediction_files_give_exactly_zero_difference(tmp_path):
    """The literal acceptance case: same file twice -> difference 0, CI containing 0."""
    items = _mini_corpus()
    rows = [make_row(item, correct=True) for item in items]
    base_rows = [dict(row, model_id="base") for row in rows]
    text = run_cli(tmp_path, items, {"meddecide": rows, "zeroshot": base_rows, "jev9b": base_rows})
    assert "| macro accuracy | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] |" in text
    assert "| mean Brier (lower is better) | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] |" in text
    assert "accuracy CI lower bound 0.0000 is not > 0" in text


def _dominating_corpus() -> list[dict]:
    """Two templates x 10 four-option items: enough headroom for a wide, unambiguous win."""
    items = []
    for template in ("t_seen", HELD_OUT):
        for i in range(10):
            keys = ["A", "B", "C", "D"]
            items.append(
                make_item(
                    template,
                    "2026-05-01",
                    gold=keys[(i // 2) % 4],
                    n_options=4,
                    item_id=f"{template}-{i}",
                )
            )
    return items


def _weak_but_healthy_base(items: list[dict]) -> list[dict]:
    """Accuracy 0.3 on 4-option items (above the 0.25 chance), with a varied argmax."""
    keys = ["A", "B", "C", "D"]
    rows = []
    for i, item in enumerate(items):
        argmax = keys[i % 4]
        rows.append(
            make_row(item, correct=(argmax == item["gold"]), argmax=argmax, model_id="base")
        )
    return rows


def test_report_is_written_end_to_end_with_all_required_sections(tmp_path):
    items = _dominating_corpus()
    ref_rows = [make_row(item, correct=True) for item in items]
    base_rows = _weak_but_healthy_base(items)
    text = run_cli(tmp_path, items, {"meddecide": ref_rows, "zeroshot": base_rows, "jev9b": base_rows})
    for needle in (
        "## Verdict (D16)",
        "G1 on seen templates",
        "G1 on held-out templates",
        "## Paired comparisons",
        "paired 95% CI",
        "unpaired 95% CI",
        "frac > 0",
        "## D12 readout health",
        "## Method",
        "NOT APPLICABLE — pointer head reads the option states directly",
        "## Item sets",
        "2026-09-10",
        "| `fresh_strict` |",
    ):
        assert needle in text, needle
    # a dominating reference passes on both seen and held-out sets
    assert "| **G1 on seen templates** (decides the branch of §2) | **PASS** |" in text
    assert "| G1 on held-out templates (D14, reported separately) | **PASS** |" in text


def test_strict_slice_membership_appears_in_the_report(tmp_path):
    items = _mini_corpus()
    ref_rows = [make_row(item, correct=True) for item in items]
    base_rows = [make_row(item, correct=False, model_id="base") for item in items]
    text = run_cli(tmp_path, items, {"meddecide": ref_rows, "zeroshot": base_rows, "jev9b": base_rows})
    assert "strict slice start: 2026-09-10" in text
    # strict slice = s3 + h2; seen strict = s3
    assert "| `fresh_strict` | headline fresh set, strict slice" in text


def test_two_runs_in_one_file_are_reported_and_the_larger_is_analysed(tmp_path):
    items = _mini_corpus()
    rows = [make_row(item, correct=True, run_id="run-big") for item in items]
    rows += [make_row(items[0], correct=False, run_id="run-small")]
    path = write_jsonl(tmp_path / "preds.jsonl", rows)
    source = g1.load_model_source("m", path)
    assert source.run_id == "run-big"
    assert source.n_items == len(items)
    assert "run-small" in source.note


def test_prediction_rows_outside_the_benchmark_are_dropped_with_a_reason(tmp_path):
    items = _mini_corpus()
    ref_rows = [make_row(item, correct=True) for item in items]
    ref_rows.append(dict(ref_rows[0], item_id="not-in-benchmark"))
    text = run_cli(tmp_path, items, {"meddecide": ref_rows})
    assert "item not in the v0.2 fresh/tier-1 index" in text


def test_file_with_only_unreadable_rows_is_not_measured(tmp_path):
    items = _mini_corpus()
    ref_rows = [make_row(item, correct=True) for item in items]
    broken = tmp_path / "preds_broken.jsonl"
    broken.write_text("{not json}\n", encoding="utf-8")
    bench = write_bench(tmp_path, items)
    argv = [
        "--bench-dir", str(bench),
        "--out", str(tmp_path / "g1.md"),
        "--resamples", "100",
        "--models", f"meddecide={write_jsonl(tmp_path / 'preds_ref.jsonl', ref_rows)}",
        "--models", f"zeroshot={broken}",
    ]
    g1.main(argv)
    text = (tmp_path / "g1.md").read_text(encoding="utf-8")
    assert "prediction file has no readable rows" in text
    assert "| **G1 on seen templates** (decides the branch of §2) | **NOT MEASURED** |" in text


def test_ablation_passed_without_additional_is_rejected():
    labels = ["meddecide", "zeroshot", "jev9b", "base"]
    with pytest.raises(ValueError, match="--additional"):
        g1.resolve_baselines(labels, "zeroshot,jev9b,base", additional=False)


def test_additional_ablation_is_compared_but_never_in_the_verdict():
    labels = ["meddecide", "zeroshot", "jev9b", "base"]
    verdict, additional = g1.resolve_baselines(labels, "zeroshot,jev9b,base", additional=True)
    assert verdict == ["zeroshot", "jev9b"]
    assert additional == ["base"]


def test_default_baselines_are_the_d16_pair_even_with_an_ablation_loaded():
    labels = ["meddecide", "zeroshot", "jev9b", "base"]
    assert g1.resolve_baselines(labels, None, additional=False) == (["zeroshot", "jev9b"], [])


def test_verdict_baselines_must_include_both_d16_baselines():
    with pytest.raises(ValueError, match="D16 baselines"):
        g1.resolve_baselines(["meddecide", "zeroshot"], "zeroshot", additional=False)


def test_a_d16_baseline_without_a_prediction_file_is_still_a_verdict_baseline():
    verdict, additional = g1.resolve_baselines(["meddecide", "zeroshot"], None, additional=False)
    assert verdict == ["zeroshot", "jev9b"]
    assert additional == []


def test_verbalizer_rows_form_a_non_letter_readout_and_mixing_is_named():
    verbalizer = {"variant": "verbalizer-head", "transform": {"readout": "verbalizer-head"}}
    letter = {"variant": "bare", "transform": {}}
    assert g1.readout_kind([verbalizer, verbalizer]) == "verbalizer-head"
    assert g1.readout_kind([letter, letter]) == "letter"
    assert g1.readout_kind([verbalizer, letter]) == "mixed"
