"""osler_v3 X2 broad replay converters (CPU, synthetic records shaped like the real sources): the labels come from the answer key,
and no rationale, annotation or formula text reaches an item."""

from __future__ import annotations

from meddecide.mix import broad_replay as br

SECRET = "SECRET-RATIONALE-TEXT"


def test_mathqa_options_parse_into_lettered_pairs() -> None:
    text = "a ) rs . 400 , b ) rs . 300 , c ) rs . 500 , d ) rs . 350 , e ) none of these"
    assert br.parse_options(text) == [("A", "rs . 400"), ("B", "rs . 300"), ("C", "rs . 500"), ("D", "rs . 350"),
                                      ("E", "none of these")]
    assert br.parse_options("no options here") is None


def test_strategyqa_gold_comes_from_the_answer_and_no_facts_are_read() -> None:
    records = [{"qid": "q1", "term": "Sun", "description": "star", "question": "Is the Sun a star?", "answer": "True",
                "facts": SECRET},
               {"qid": "q2", "term": "Moon", "description": "", "question": "Is the Moon hot?", "answer": "False",
                "facts": SECRET},
               {"qid": "q3", "term": "X", "description": "", "question": "?", "answer": "maybe", "facts": SECRET}]
    rows, counts = br.strategyqa_rows(records)
    assert [r["gold"] for r in rows] == ["A", "B"]
    assert counts["dropped_answer_not_yes_or_no"] == 1
    assert all(SECRET not in str(r) for r in rows)
    assert rows[1]["state"] == "Topic: Moon."


def test_quartz_keeps_the_offered_labels_and_the_answer_key() -> None:
    record = {"id": "Q1", "question": "Which is scarcer?", "para": "Water is scarce here.", "answerKey": "B",
              "choices": {"text": ["plentiful", "scarce"], "label": ["A", "B"]}, "para_anno": SECRET}
    rows, counts = br.quartz_rows([record])
    assert counts["kept"] == 1 and rows[0]["gold"] == "B"
    assert [o["label"] for o in rows[0]["options"]] == ["plentiful", "scarce"]
    assert SECRET not in str(rows[0])


def test_aqua_rat_parses_options_and_never_reads_the_rationale() -> None:
    record = {"question": "Two plus two?", "options": ["A)3", "B)4", "C)5"], "rationale": SECRET, "correct": "B"}
    rows, _ = br.aqua_rat_rows([record])
    assert rows[0]["gold"] == "B" and [o["label"] for o in rows[0]["options"]] == ["3", "4", "5"]
    assert SECRET not in str(rows[0])


def test_a_tokenised_copy_of_a_problem_has_the_same_key_and_is_dropped() -> None:
    aqua = {"question": "Two friends plan to walk 43-km, starting at opposite ends.", "options": ["A)21", "B)22"],
            "rationale": SECRET, "correct": "A"}
    math = {"Problem": "two friends plan to walk 43 - km , starting at opposite ends .", "options": "a ) 21 , b ) 22",
            "correct": "a", "Rationale": SECRET}
    aqua_rows, _ = br.aqua_rat_rows([aqua])
    math_rows, _ = br.math_qa_rows([math])
    assert br.problem_key(aqua["question"]) == br.problem_key(math["Problem"])
    kept, dropped = br.drop_rows_with_problem_in(math_rows, frozenset({br.problem_key(aqua["question"])}))
    assert kept == [] and dropped == 1
    kept_aqua, dropped_aqua = br.drop_rows_with_problem_in(aqua_rows, frozenset({"unrelated"}))
    assert len(kept_aqua) == 1 and dropped_aqua == 0


def test_math_qa_drops_problems_whose_answer_is_not_offered() -> None:
    good = {"Problem": "Find x.", "options": "a ) 1 , b ) 2 , c ) 3 , d ) 4 , e ) 5", "correct": "c",
            "Rationale": SECRET, "category": "algebra", "linear_formula": SECRET}
    bad = {"Problem": "Find y.", "options": "a ) 1 , b ) 2", "correct": "e", "Rationale": SECRET}
    rows, counts = br.math_qa_rows([good, bad])
    assert [r["gold"] for r in rows] == ["C"]
    assert counts["dropped_answer_not_offered"] == 1
    assert SECRET not in str(rows)


def test_social_iqa_maps_the_one_based_label_to_a_letter() -> None:
    records = [{"context": "C", "question": "Q?", "answerA": "a", "answerB": "b", "answerC": "c"}]
    rows, _ = br.social_iqa_rows(records, ["3"])
    assert rows[0]["gold"] == "C"


def test_cosmos_qa_maps_the_zero_based_label_to_a_letter() -> None:
    record = {"id": "x", "context": "C", "question": "Q?", "answer0": "a", "answer1": "b", "answer2": "c", "answer3": "d",
              "label": "1"}
    rows, _ = br.cosmos_qa_rows([record])
    assert rows[0]["gold"] == "B" and len(rows[0]["options"]) == 4


def test_bigbench_gold_is_the_one_option_scored_one() -> None:
    record = {"inputs": "Q: pick", "multiple_choice_targets": "['(a)', '(b)', '(c)']", "multiple_choice_scores": "[0, 1, 0]",
              "idx": "7"}
    rows, counts = br.bigbench_rows("snarks", [record])
    assert rows[0]["gold"] == "B" and counts["kept"] == 1
    tie = dict(record, multiple_choice_scores="[1, 1, 0]")
    rows2, counts2 = br.bigbench_rows("snarks", [tie])
    assert rows2 == [] and counts2["dropped_not_exactly_one_correct"] == 1


def test_drop_duplicates_keeps_the_first_of_each_text() -> None:
    record = {"inputs": "Q: same", "multiple_choice_targets": "['(a)', '(b)']", "multiple_choice_scores": "[1, 0]", "idx": "1"}
    rows, _ = br.bigbench_rows("snarks", [record, dict(record, idx="2")])
    kept, dropped = br.drop_duplicates(rows)
    assert len(kept) == 1 and dropped == 1
