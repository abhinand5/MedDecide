"""Unit tests for the openFDA class index and its near-miss rules.

Everything here runs on a synthetic in-memory index or a fake HTTP client: no network.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from meddecide.bench.fresh import fda_classes


def _entry(moa: list[str], pe: list[str], routes: list[str], n_labels: int) -> dict[str, Any]:
    return {"moa": moa, "pe": pe, "routes": routes, "n_labels": n_labels}


def _index() -> fda_classes.ClassIndex:
    """Five classes covering every rule, both precedence directions, and one unrelated class.

    Gold is ``Alpha Agonists [EPC]``:
    * ``Beta Blockers`` shares a MoA *and* a PE value -> ``shared_moa`` (precedence)
    * ``Gamma``/``Zeta`` share only a PE value -> ``shared_pe``, ordered by name
    * ``Delta`` shares only a route -> ``same_route``
    * ``Epsilon`` shares nothing -> never a candidate
    """
    return {
        "classes": {
            "Alpha Agonists [EPC]": _entry(
                ["Alpha Adrenergic Agonism [MoA]", "Common Pathway [MoA]"],
                ["Vasoconstriction [PE]"],
                ["ORAL", "TOPICAL"],
                10,
            ),
            "Beta Blockers [EPC]": _entry(
                ["Common Pathway [MoA]"], ["Vasoconstriction [PE]", "Bradycardia [PE]"], ["ORAL"], 7
            ),
            "Gamma Blockers [EPC]": _entry([], ["Vasoconstriction [PE]"], ["ORAL", "INTRAVENOUS"], 3),
            "Zeta Blockers [EPC]": _entry([], ["Vasoconstriction [PE]"], ["SUBCUTANEOUS"], 2),
            "Delta Blockers [EPC]": _entry([], [], ["ORAL"], 4),
            "Epsilon Blockers [EPC]": _entry([], [], ["INTRAVENOUS"], 1),
        },
        "meta": {
            "n_labels_fetched": 27,
            "n_labels_without_class": 0,
            "n_classes": 6,
            "pages": 1,
            "query": fda_classes.QUERY,
            "api_total": 27,
        },
    }


GOLD = "Alpha Agonists [EPC]"


# --------------------------------------------------------------------------- rules


def test_rule_precedence_moa_beats_pe() -> None:
    """A class sharing both a MoA and a PE value is reported under the stronger rule."""
    candidates = fda_classes.near_miss_candidates(GOLD, _index())
    beta = next(c for c in candidates if c["class"] == "Beta Blockers [EPC]")
    assert beta["rule"] == "shared_moa"
    # only the strongest rule's values, not the PE values it also shares
    assert beta["shared"] == ["Common Pathway [MoA]"]
    assert beta["n_labels"] == 7


def test_rule_precedence_pe_beats_route() -> None:
    candidates = fda_classes.near_miss_candidates(GOLD, _index())
    gamma = next(c for c in candidates if c["class"] == "Gamma Blockers [EPC]")
    assert gamma["rule"] == "shared_pe"
    assert gamma["shared"] == ["Vasoconstriction [PE]"]


def test_shared_values_are_sorted_and_complete() -> None:
    index = _index()
    index["classes"]["Beta Blockers [EPC]"]["moa"] = [
        "Common Pathway [MoA]",
        "Alpha Adrenergic Agonism [MoA]",
    ]
    beta = next(c for c in fda_classes.near_miss_candidates(GOLD, index)
                if c["class"] == "Beta Blockers [EPC]")
    assert beta["shared"] == ["Alpha Adrenergic Agonism [MoA]", "Common Pathway [MoA]"]


def test_every_candidate_carries_exactly_one_rule() -> None:
    candidates = fda_classes.near_miss_candidates(GOLD, _index())
    assert [c["rule"] for c in candidates] == ["shared_moa", "shared_pe", "shared_pe", "same_route"]


def test_unrelated_class_is_not_a_candidate() -> None:
    names = {c["class"] for c in fda_classes.near_miss_candidates(GOLD, _index())}
    assert "Epsilon Blockers [EPC]" not in names


def test_unknown_gold_class_returns_empty() -> None:
    assert fda_classes.near_miss_candidates("No Such Class [EPC]", _index()) == []


# ------------------------------------------------------------- exclusion / gold


def test_gold_class_is_never_returned() -> None:
    names = {c["class"] for c in fda_classes.near_miss_candidates(GOLD, _index())}
    assert GOLD not in names


def test_untagged_gold_class_is_still_removed() -> None:
    """Naming the gold class without its ``[EPC]`` tag must not resurrect it as a candidate."""
    names = {c["class"] for c in fda_classes.near_miss_candidates("Alpha Agonists", _index())}
    assert GOLD not in names
    assert "Beta Blockers [EPC]" in names


def test_exclude_drops_candidates_before_ranking() -> None:
    """Excluded names are gone before the cap, so the cap fills with the next class."""
    candidates = fda_classes.near_miss_candidates(
        GOLD, _index(), exclude={"Beta Blockers [EPC]"}, max_candidates=1
    )
    assert [c["class"] for c in candidates] == ["Gamma Blockers [EPC]"]


def test_exclude_can_name_the_gold_class() -> None:
    candidates = fda_classes.near_miss_candidates(GOLD, _index(), exclude={GOLD})
    assert GOLD not in {c["class"] for c in candidates}


# ------------------------------------------------------------------- determinism


def test_determinism_same_input_same_order() -> None:
    index = _index()
    first = fda_classes.near_miss_candidates(GOLD, index)
    second = fda_classes.near_miss_candidates(GOLD, index)
    assert first == second


def test_determinism_independent_of_dict_insertion_order() -> None:
    index = _index()
    shuffled = {**index, "classes": dict(reversed(list(index["classes"].items())))}
    assert (
        fda_classes.near_miss_candidates(GOLD, index)
        == fda_classes.near_miss_candidates(GOLD, shuffled)
    )


def test_max_candidates_caps_the_list() -> None:
    assert len(fda_classes.near_miss_candidates(GOLD, _index(), max_candidates=2)) == 2
    assert fda_classes.near_miss_candidates(GOLD, _index(), max_candidates=0) == []
    assert len(fda_classes.near_miss_candidates(GOLD, _index(), max_candidates=50)) == 4


# ------------------------------------------------------------------ rule_counts


def test_rule_counts_counts_each_rule_and_keeps_zeros() -> None:
    counts = fda_classes.rule_counts(fda_classes.near_miss_candidates(GOLD, _index()))
    assert counts == {"shared_moa": 1, "shared_pe": 2, "same_route": 1}


def test_rule_counts_of_empty_list_is_all_zero() -> None:
    assert fda_classes.rule_counts([]) == {"shared_moa": 0, "shared_pe": 0, "same_route": 0}


# --------------------------------------------------------------------- name tags


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Biguanides [EPC]", "Biguanides"),
        ("Proton Pump Inhibitors [MoA]", "Proton Pump Inhibitors"),
        ("  Biguanides  ", "Biguanides"),
        ("No Tag", "No Tag"),
    ],
)
def test_strip_class_tag(raw: str, expected: str) -> None:
    assert fda_classes.strip_class_tag(raw) == expected


def test_find_class_exact_then_untagged_then_miss() -> None:
    index = _index()
    assert fda_classes.find_class(GOLD, index) == GOLD
    assert fda_classes.find_class("Biguanides", index) is None
    assert fda_classes.find_class("delta blockers", index) == "Delta Blockers [EPC]"


# ------------------------------------------------------------------ json round trip


def test_save_load_round_trip(tmp_path: Any) -> None:
    index = _index()
    path = fda_classes.save_index(index, tmp_path / "nested" / "class_index.json")
    assert path.is_file()
    assert fda_classes.load_index(path) == index


def test_save_is_byte_deterministic(tmp_path: Any) -> None:
    index = _index()
    first = fda_classes.save_index(index, tmp_path / "a.json").read_text(encoding="utf-8")
    second = fda_classes.save_index(index, tmp_path / "b.json").read_text(encoding="utf-8")
    assert first == second
    assert json.loads(first)["classes"][GOLD]["routes"] == ["ORAL", "TOPICAL"]


def test_loaded_index_still_ranks(tmp_path: Any) -> None:
    path = fda_classes.save_index(_index(), tmp_path / "class_index.json")
    candidates = fda_classes.near_miss_candidates(GOLD, fda_classes.load_index(path))
    assert [c["rule"] for c in candidates] == ["shared_moa", "shared_pe", "shared_pe", "same_route"]


# ------------------------------------------------------------- fetch (fake client)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"unexpected HTTP {self.status_code}")


class _FakeClient:
    """Serves ``labels`` in pages; a request past the end answers 404, like openFDA."""

    def __init__(self, labels: list[dict[str, Any]], api_total: int | None = None) -> None:
        self.labels = labels
        self.api_total = len(labels) if api_total is None else api_total
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any]) -> _FakeResponse:
        self.calls.append({"url": url, **params})
        skip, limit = int(params["skip"]), int(params["limit"])
        batch = self.labels[skip : skip + limit]
        if not batch:
            return _FakeResponse(404, {})
        return _FakeResponse(
            200,
            {
                "meta": {"results": {"skip": skip, "limit": limit, "total": self.api_total}},
                "results": batch,
            },
        )


def _label(*, epc: list[str], moa: list[str] | None = None, pe: list[str] | None = None,
           route: list[str] | None = None) -> dict[str, Any]:
    openfda: dict[str, Any] = {"pharm_class_epc": epc}
    if moa is not None:
        openfda["pharm_class_moa"] = moa
    if pe is not None:
        openfda["pharm_class_pe"] = pe
    if route is not None:
        openfda["route"] = route
    return {"openfda": openfda}


def test_fetch_pages_until_404_and_sorts_everything() -> None:
    client = _FakeClient(
        [
            _label(epc=["Zeta [EPC]"], moa=["Second [MoA]", "First [MoA]"], route=["oral", " Topical "]),
            _label(epc=["Alpha [EPC]", "Zeta [EPC]"], pe=["Effect [PE]"]),
            _label(epc=["Alpha [EPC]"]),
        ]
    )
    index = fda_classes.fetch_class_index(max_labels=10, page_size=2, client=client)
    assert index["meta"] == {
        "n_labels_fetched": 3,
        "n_labels_without_class": 0,
        "n_classes": 2,
        "pages": 2,
        "query": fda_classes.QUERY,
        "api_total": 3,
    }
    assert list(index["classes"]) == ["Alpha [EPC]", "Zeta [EPC]"]
    assert index["classes"]["Zeta [EPC]"] == {
        "moa": ["First [MoA]", "Second [MoA]"],  # sorted
        # the second label carries both classes, so its PE value lands on both
        "pe": ["Effect [PE]"],
        "routes": ["ORAL", "TOPICAL"],  # upper-cased and stripped
        "n_labels": 2,  # the second label names Zeta twice, but it is one label
    }
    assert index["classes"]["Alpha [EPC]"]["n_labels"] == 2
    assert client.calls[0]["search"] == fda_classes.QUERY
    # api_total says 3 and 3 were fetched, so the walk stops without a further request
    assert [call["skip"] for call in client.calls] == [0, 2]


def test_fetch_404_mid_walk_ends_the_pages() -> None:
    """openFDA can answer 404 while meta still claims a larger total: the 404 wins."""
    client = _FakeClient([_label(epc=["Alpha [EPC]"]), _label(epc=["Beta [EPC]"])], api_total=99)
    index = fda_classes.fetch_class_index(max_labels=10, page_size=2, client=client)
    assert [call["skip"] for call in client.calls] == [0, 2]
    assert index["meta"]["pages"] == 1
    assert index["meta"]["n_labels_fetched"] == 2
    assert index["meta"]["api_total"] == 99  # the discrepancy stays visible


def test_fetch_counts_labels_without_a_class() -> None:
    client = _FakeClient([_label(epc=[]), _label(epc=["Alpha [EPC]"])])
    index = fda_classes.fetch_class_index(max_labels=10, page_size=10, client=client)
    assert index["meta"]["n_labels_fetched"] == 2
    assert index["meta"]["n_labels_without_class"] == 1
    assert index["meta"]["n_classes"] == 1


def test_fetch_respects_max_labels_across_a_short_last_page() -> None:
    client = _FakeClient([_label(epc=[f"C{i} [EPC]"]) for i in range(5)])
    index = fda_classes.fetch_class_index(max_labels=3, page_size=2, client=client)
    assert index["meta"]["n_labels_fetched"] == 3
    assert index["meta"]["pages"] == 2
    assert [call["limit"] for call in client.calls] == [2, 1]
    assert index["meta"]["api_total"] == 5  # the cap is visible, not silent


def test_fetch_404_on_first_page_is_an_empty_index() -> None:
    index = fda_classes.fetch_class_index(max_labels=10, page_size=10, client=_FakeClient([]))
    assert index["classes"] == {}
    assert index["meta"]["pages"] == 0
    assert index["meta"]["api_total"] == 0


def test_fetch_stops_at_api_total() -> None:
    client = _FakeClient([_label(epc=["Alpha [EPC]"])], api_total=1)
    index = fda_classes.fetch_class_index(max_labels=10, page_size=1, client=client)
    assert index["meta"]["pages"] == 1
    assert len(client.calls) == 1


def test_fetch_rejects_a_bad_page_size() -> None:
    with pytest.raises(ValueError, match="page_size"):
        fda_classes.fetch_class_index(page_size=0, client=_FakeClient([]))
