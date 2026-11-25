"""osler_v1 V7-D4: bare and dataset-qualified record collisions with the protected sets (pure functions)."""

from __future__ import annotations

from meddecide.mix.leakage import bare_collisions, qualified_collisions


def test_a_bare_id_shared_by_two_datasets_is_a_bare_collision_but_not_a_qualified_one() -> None:
    row = {"source": "pubhealth", "source_record_id": "280"}  # a PubHealth claim
    protected_bare = {"panel": {"280"}, "robustness": {"280"}}  # an MMLU-Pro item that happens to share the number
    protected_qualified = {"panel": {"mmlu_pro_health/280"}}
    assert bare_collisions(row, protected_bare) == ["panel", "robustness"]
    assert qualified_collisions(row, protected_qualified) == []


def test_the_same_dataset_and_id_is_a_qualified_collision() -> None:
    row = {"source": "medqa", "source_record_id": 17}  # an integer id is compared as text
    assert qualified_collisions(row, {"v02_test": {"medqa/17"}, "panel": {"other/17"}}) == ["v02_test"]


def test_collision_names_are_sorted_and_empty_when_nothing_matches() -> None:
    row = {"source": "pubmed", "source_record_id": "12"}
    assert bare_collisions(row, {"b": {"12"}, "a": {"12"}, "c": {"13"}}) == ["a", "b"]
    assert bare_collisions(row, {"c": {"13"}}) == []
