import pytest

from meddecide.bench.quick_bench import (
    aggregate,
    id_list_sha256,
    probs_from_answer,
    proportional_allocation,
    seeded_sample,
    stratified_sample,
)


def test_seeded_sample_is_deterministic_and_order_free():
    ids = [f"id{i:03d}" for i in range(100)]
    a = seeded_sample(ids, 10, 0)
    b = seeded_sample(list(reversed(ids)), 10, 0)
    assert a == b and len(set(a)) == 10 and a == sorted(a)
    assert seeded_sample(ids, 10, 1) != a


def test_seeded_sample_rejects_duplicates_and_bad_k():
    with pytest.raises(ValueError):
        seeded_sample(["a", "a", "b"], 1, 0)
    with pytest.raises(ValueError):
        seeded_sample(["a", "b"], 3, 0)


def test_proportional_allocation_closes():
    alloc = proportional_allocation({"a": 1000, "b": 1000, "c": 940, "d": 781, "e": 295, "f": 212, "g": 125}, 500)
    assert sum(alloc.values()) == 500
    assert alloc["a"] == alloc["b"]
    assert all(v > 0 for v in alloc.values())


def test_stratified_sample():
    items = [{"item_id": f"x{i}", "s": "big" if i < 90 else "small"} for i in range(100)]
    ids, alloc = stratified_sample(items, lambda r: r["s"], 20, 0)
    assert alloc == {"big": 18, "small": 2}
    assert len(ids) == 20 and sum(1 for i in ids if int(i[1:]) >= 90) == 2
    assert stratified_sample(items, lambda r: r["s"], 20, 0)[0] == ids


def test_id_list_hash_is_order_free():
    assert id_list_sha256(["b", "a"]) == id_list_sha256(["a", "b"])


def test_probs_from_answer():
    assert probs_from_answer(["A", "B"], "choice", {"probabilities": {"B": 0.3, "A": 0.7}}) == [0.7, 0.3]
    assert probs_from_answer(["no", "yes"], "noul", {"noul": 0.8}) == pytest.approx([0.2, 0.8])
    with pytest.raises(ValueError):
        probs_from_answer(["A", "B"], "choice", {"probabilities": {"A": 1.0}})


def test_aggregate_micro_macro_and_gate():
    rows = []
    for i in range(10):  # template t1: 10 items, 9 correct, varied answers
        rows.append({"template_id": "t1", "gold_key": "AB"[i % 2], "predicted_key": "AB"[i % 2] if i else "B",
                     "correct": bool(i), "probs": [0.5, 0.5]})
    for i in range(10):  # template t2: constant answer on a balanced template -> READOUT_FAIL
        rows.append({"template_id": "t2", "gold_key": "AB"[i % 2], "predicted_key": "A", "correct": i % 2 == 0,
                     "probs": [0.5, 0.5]})
    out = aggregate(rows)
    assert out["items"] == 20 and out["correct"] == 14
    assert out["accuracy_micro"] == pytest.approx(0.7)
    assert out["accuracy_template_macro"] == pytest.approx((0.9 + 0.5) / 2)
    assert out["gate"] == "READOUT_FAIL" and list(out["failing_templates"]) == ["t2"]
