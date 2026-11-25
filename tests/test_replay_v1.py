"""General replay converters of osler_v1 (ADVISORY V3): the answer mapping, the drop reasons and the licence table."""

from __future__ import annotations

from meddecide.mix.replay_v1 import CANDIDATES, hellaswag_row, winogrande_row


def _hellaswag(label: str = "2", endings: int = 4) -> dict:
    return {"ind": 17, "split": "train", "activity_label": "Cooking", "ctx": "She puts the pan on the stove.",
            "endings": [f"ending {i}" for i in range(endings)], "label": label}


def test_hellaswag_maps_the_label_to_its_letter_and_keeps_the_ending_text() -> None:
    row, reason = hellaswag_row(_hellaswag("2"), "train")
    assert reason == "kept" and row is not None
    assert row["source"] == "hellaswag" and row["split"] == "train" and row["qtype"] == "choice"
    assert row["gold"] == "C"
    gold_label = next(o["label"] for o in row["options"] if o["key"] == row["gold"])
    assert gold_label == "ending 2"
    assert row["state"] == "She puts the pan on the stove."


def test_hellaswag_drops_a_row_without_four_endings_or_a_label() -> None:
    assert hellaswag_row(_hellaswag(endings=3), "train") == (None, "not_four_endings")
    assert hellaswag_row(_hellaswag(label=""), "train") == (None, "label_missing")


def _winogrande(answer: str = "1") -> dict:
    return {"qID": "3DRS9DEEZ5DI3TNMHGEQV1UBBWEJC2_1", "sentence": "The trophy does not fit in the suitcase because _ is big.",
            "option1": "trophy", "option2": "suitcase", "answer": answer}


def test_winogrande_maps_answer_one_to_option_a_and_keeps_the_blank_question() -> None:
    row, reason = winogrande_row(_winogrande("1"), "train")
    assert reason == "kept" and row is not None
    assert row["gold"] == "A"
    assert next(o["label"] for o in row["options"] if o["key"] == "A") == "trophy"
    assert "Which option fills the blank (_)?" in row["state"]


def test_winogrande_answer_two_is_option_b_and_bad_rows_are_counted() -> None:
    row, _ = winogrande_row(_winogrande("2"), "train")
    assert row is not None and row["gold"] == "B"
    assert winogrande_row(_winogrande(""), "train") == (None, "answer_missing")
    no_blank = {**_winogrande("1"), "sentence": "No gap here."}
    assert winogrande_row(no_blank, "train") == (None, "no_blank")


def test_only_permissive_candidates_are_kept_and_each_is_recorded() -> None:
    kept = sorted(k for k, v in CANDIDATES.items() if v[0] == "kept")
    assert kept == ["hellaswag", "winogrande"]
    for name in ("ai2_arc", "boolq", "snli"):
        decision, licence, _ = CANDIDATES[name]
        assert decision == "dropped" and "SA" in licence  # share-alike licences are not permissive (D20)
    assert CANDIDATES["openbookqa"][0] == "dropped"  # its own licence is not stated (ADVISORY: drop an unclear licence)
