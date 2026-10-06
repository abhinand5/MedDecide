"""Unit tests for the readout-health gate (D12), with hand-computed cases."""

from __future__ import annotations

import pytest

from meddecide.eval.health import evaluate_cell


def _cell(**overrides):
    kwargs = {
        "model_id": "m",
        "template_id": "t",
        "label_masses": [0.9, 0.9, 0.9, 0.9],
        "correct": [True, True, False, True],
        "greedy_matches": [True, True, True, True],
        "n_options": 4,
        "n_resamples": 200,
    }
    kwargs.update(overrides)
    return evaluate_cell(**kwargs)


def test_high_mass_correct_cell_passes() -> None:
    report = _cell()
    assert report.passed, report.failures
    assert report.status == "PASS"
    assert report.median_label_mass == pytest.approx(0.9)
    assert report.greedy_agreement == pytest.approx(1.0)
    assert report.accuracy == pytest.approx(0.75)
    assert report.chance == pytest.approx(0.25)


def test_low_label_mass_fails_with_the_measured_value() -> None:
    report = _cell(label_masses=[0.002, 0.002, 0.002, 0.002])
    assert not report.passed
    assert "median label mass" in report.failures[0]
    assert "0.0020" in report.failures[0]
    assert report.status.startswith("READOUT_FAIL")
    # the accuracy is still computed and reported beside the failure, never instead of it
    assert report.accuracy == pytest.approx(0.75)


def test_below_chance_cell_fails_even_with_high_mass() -> None:
    report = _cell(correct=[False] * 40, label_masses=[0.9] * 40, greedy_matches=[True] * 40)
    assert not report.passed
    assert any("below-chance" in f for f in report.failures)
    # a cell at chance (not below) does not trip the CI check
    at_chance = _cell(
        correct=[True, False, False, False] * 10, label_masses=[0.9] * 40, greedy_matches=[True] * 40
    )
    assert at_chance.passed, at_chance.failures


def test_greedy_disagreement_fails() -> None:
    report = _cell(greedy_matches=[False] * 10)
    assert not report.passed
    assert any("greedy agreement" in f for f in report.failures)


def test_greedy_not_measured_is_recorded_not_passed_silently() -> None:
    report = _cell(greedy_matches=None)
    assert report.greedy_agreement is None
    assert report.greedy_sample_size == 0
    assert any("NOT MEASURED" in n for n in report.notes)
    assert report.passed  # the other two checks pass, but the note says what was not run


def test_thresholds_are_parameters_and_are_respected() -> None:
    strict = _cell(label_masses=[0.6] * 4, min_median_label_mass=0.95)
    assert not strict.passed
    lenient = _cell(label_masses=[0.05] * 4, min_median_label_mass=0.01)
    assert lenient.passed
    greedy_strict = _cell(greedy_matches=[True] * 9 + [False], min_greedy_agreement=1.0)
    assert not greedy_strict.passed


def test_chance_scales_with_option_count() -> None:
    two = _cell(n_options=2)
    assert two.chance == pytest.approx(0.5)
    seventeen = _cell(n_options=17, correct=[False] * 40, label_masses=[0.9] * 40,
                      greedy_matches=[True] * 40)
    assert seventeen.chance == pytest.approx(1 / 17)
    # 0 correct of 40 with a 1/17 chance is not below chance by the CI rule, so only the
    # explicit below-chance case fails
    assert seventeen.passed or any("below-chance" in f for f in seventeen.failures)


def test_mismatched_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        _cell(label_masses=[0.9, 0.9], correct=[True])
    with pytest.raises(ValueError, match="non-empty"):
        _cell(label_masses=[], correct=[])
    with pytest.raises(ValueError, match="n_options"):
        _cell(n_options=1)


def test_report_serialises_all_measured_values() -> None:
    payload = _cell().to_dict()
    for key in ("median_label_mass", "greedy_agreement", "accuracy", "accuracy_ci95", "status",
                "failures", "chance"):
        assert key in payload
    assert payload["passed"] is True


# ---------------------------------------------------------------------------
# constant-answer check (added in loop 1, S1)
# ---------------------------------------------------------------------------
def test_constant_answer_on_a_balanced_template_fails_as_degenerate() -> None:
    """The LFM2.5-350M pattern: exactly chance accuracy, one answer every time.

    A balanced 2-option template (majority 0.5) answered "yes" on every item scores 0.50,
    which is not below chance and carries high label mass, so the original three checks pass
    it. It is a readout defect, not a result.
    """
    n = 200
    report = _cell(
        n_options=2,
        label_masses=[0.99] * n,
        correct=[True] * (n // 2) + [False] * (n // 2),
        greedy_matches=[True] * n,
        predicted_labels=["yes"] * n,
        majority_share=0.50,
    )
    assert not report.passed
    assert any("degenerate" in f for f in report.failures), report.failures
    assert any("yes" in f and "1.000" in f for f in report.failures)
    assert report.modal_option == "yes"
    assert report.modal_share == pytest.approx(1.0)
    assert report.accuracy == pytest.approx(0.5)


def test_constant_answer_is_not_flagged_when_the_template_really_is_dominated() -> None:
    """A template whose majority share is above the ceiling may legitimately get one answer."""
    n = 100
    report = _cell(
        n_options=2,
        label_masses=[0.99] * n,
        correct=[True] * 90 + [False] * 10,
        greedy_matches=[True] * n,
        predicted_labels=["yes"] * n,
        majority_share=0.90,
    )
    assert report.passed, report.failures
    assert report.modal_share == pytest.approx(1.0)


def test_constant_answer_below_the_modal_threshold_passes() -> None:
    n = 100
    labels = ["yes"] * 85 + ["no"] * 15  # modal share 0.85 < 0.90
    report = _cell(
        n_options=2,
        label_masses=[0.99] * n,
        correct=[True] * 60 + [False] * 40,
        greedy_matches=[True] * n,
        predicted_labels=labels,
        majority_share=0.50,
    )
    assert report.passed, report.failures


def test_constant_answer_check_reports_not_measured_without_majority_share() -> None:
    report = _cell(predicted_labels=["yes"] * 4)
    assert report.passed
    assert any("constant-answer check NOT MEASURED" in note for note in report.notes)


def test_predicted_labels_length_is_validated() -> None:
    with pytest.raises(ValueError, match="predicted_labels"):
        _cell(predicted_labels=["yes", "no"])
