"""Option-code diagnostics (additional, dev only): the letter inverse, the split by gold code, and the marked results section."""

import pytest

from meddecide.eval.code_split import (
    SECTION_BEGIN,
    letter_index,
    option_index,
    place_section,
    render_code_section,
    split_by_gold_code,
    two_letter_mass,
)
from meddecide.eval.readout import option_letter


def test_letter_index_inverts_option_letter() -> None:
    for index in range(300):
        assert letter_index(option_letter(index)) == index


def test_option_index_reads_the_runner_keys() -> None:
    assert option_index("A") == 0
    assert option_index("BA") == 26
    assert option_index("yes") == 0
    assert option_index("no") == 1
    assert option_index("1") == 0
    assert option_index("49") == 48


@pytest.mark.parametrize("key", ["0", "01", "1.0"])
def test_option_index_refuses_other_keys(key: str) -> None:
    with pytest.raises(ValueError):
        option_index(key)


def test_two_letter_codes_start_at_ba() -> None:
    assert option_letter(26) == "BA"
    assert letter_index("BA") == 26
    assert option_letter(48) == "BW"


@pytest.mark.parametrize("code", ["", "AA", "a", "A1", "AB-"])
def test_non_canonical_codes_are_refused(code: str) -> None:
    with pytest.raises(ValueError):
        letter_index(code)


def _split_items() -> list[tuple[int, bool, list[float]]]:
    high = [0.9 / 26] * 26 + [0.1 / 23] * 23  # 0.1 of the mass on the two-letter codes
    low = [0.5 / 26] * 26 + [0.5 / 23] * 23  # 0.5 of the mass on the two-letter codes
    return [
        (0, True, high),  # single-letter gold, correct
        (2, False, high),  # single-letter gold, wrong
        (26, False, low),  # two-letter gold, wrong
        (30, True, low),  # two-letter gold, correct
    ]


def test_two_letter_mass_is_the_probability_from_index_26_up() -> None:
    assert two_letter_mass([0.5 / 26] * 26 + [0.5 / 23] * 23) == pytest.approx(0.5)
    assert two_letter_mass([1 / 26] * 26) == 0.0


def test_split_by_gold_code_counts_accuracy_and_mass() -> None:
    split = split_by_gold_code(_split_items())
    assert split["items"] == 4
    assert split["single_gold"]["items"] == 2
    assert split["single_gold"]["correct"] == 1
    assert split["single_gold"]["accuracy"] == pytest.approx(0.5)
    assert split["single_gold"]["two_letter_mass_mean"] == pytest.approx(0.1)
    assert split["two_letter_gold"]["items"] == 2
    assert split["two_letter_gold"]["correct"] == 1
    assert split["two_letter_gold"]["accuracy"] == pytest.approx(0.5)
    assert split["two_letter_gold"]["two_letter_mass_mean"] == pytest.approx(0.5)
    assert split["two_letter_mass_mean"] == pytest.approx((0.1 + 0.1 + 0.5 + 0.5) / 4)


def test_an_empty_group_has_no_accuracy_and_no_mass() -> None:
    probs = [1 / 26] * 26
    split = split_by_gold_code([(0, True, probs), (1, False, probs)])
    assert split["two_letter_gold"] == {"items": 0, "correct": 0, "accuracy": None, "two_letter_mass_mean": None}


def test_a_gold_index_outside_the_options_is_refused() -> None:
    with pytest.raises(ValueError):
        split_by_gold_code([(26, True, [0.5, 0.5])])


def _result(ddx_status: str) -> dict:
    split = split_by_gold_code([(0, True, [1 / 49] * 49), (30, False, [1 / 49] * 49)])
    return {
        "labels": {"a0": "A0 control", "y8_final": "Y8 final"},
        "reference": "a0",
        "reference_name": "A0",
        "unscored": {"run": "a0", "items": 0, "repeated_label_items": 0, "by_template": {}},
        "ddx_items": 2,
        "ddx_gold_two_letter_share": 0.5,
        "ddxplus": {"a0": {"status": "PASS", "split": split}, "y8_final": {"status": ddx_status, "split": split}},
        "templates": [
            {"template": "t1", "items": {"a0": 10, "y8_final": 10}, "accuracy": {"a0": 0.5, "y8_final": 0.7},
             "status": {"a0": "PASS", "y8_final": "PASS"}},
            {"template": "t2", "items": {"a0": 4, "y8_final": 4}, "accuracy": {"a0": 0.25, "y8_final": 0.25},
             "status": {"a0": "PASS", "y8_final": "READOUT_FAIL — constant answer (A) on 0.950 of items"}},
        ],
    }


def test_a_failing_cell_reads_readout_fail_and_shows_no_accuracy() -> None:
    text = render_code_section(_result("PASS"))
    assert "| t2 | 4 | 0.2500 | READOUT_FAIL |" in text
    assert "| t1 | 10 | 0.5000 | 0.7000 |" in text
    failing = render_code_section(_result("READOUT_FAIL — below chance (CI upper 0.0300 < chance 0.0204)"))
    assert "| Y8 final | 1 | READOUT_FAIL | 1 | READOUT_FAIL |" in failing
    assert "| A0 control | 1 | 1.0000 | 1 | 0.0000 |" in text


def test_place_section_appends_once_and_replaces_in_place() -> None:
    section = render_code_section(_result("PASS"))
    once = place_section("# Scoreboard\n", section)
    assert once.startswith("# Scoreboard\n\n" + SECTION_BEGIN)
    assert place_section(once, section).count(SECTION_BEGIN) == 1
    later = place_section(once + "\n## Later\n", section.replace("A0 control", "A0 renamed"))
    assert "A0 renamed" in later
    assert "A0 control" not in later
    assert later.endswith("\n\n## Later\n")
