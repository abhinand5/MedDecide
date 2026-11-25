"""Tests for the public MeSH descriptor index (``meddecide.bench.mesh``).

The fixture is a made-up tree — no real MeSH content beyond a handful of descriptor names — so
the tests exercise the parser and the sibling logic without the 313 MB NLM download.

The tree (8 descriptors), with each descriptor's immediate parents on the right::

    Alpha   X1                      —
    Beta    X1.100                  X1
    Gamma   X1.100.200              X1.100
    Delta   X1.100.300              X1.100
    Epsilon X1.100.400              X1.100
    Zeta    X1.100.500              X1.100
    Eta     X1.100.500.900          X1.100.500
    Theta   X1.100.500.950, X1.100.600   X1.100.500, X1.100

So ``X1.100`` has five children (Gamma, Delta, Epsilon, Zeta, Theta) and ``X1.100.500`` has
two (Eta via ``X1.100.500.900``, Theta via ``X1.100.500.950``).
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from meddecide.bench.mesh import (
    DESC_GZ_URL,
    DESC_XML_URL,
    MESH_YEAR,
    MeshIndex,
    download_descriptors,
    is_gzip,
    load_index,
    parent_tree_number,
    parse_descriptors,
    parse_descriptors_with_stats,
    save_index,
)

FIXTURE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<DescriptorRecordSet LanguageCode="eng">
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000001</DescriptorUI>
    <DescriptorName><String>Alpha</String></DescriptorName>
    <TreeNumberList><TreeNumber>X1</TreeNumber></TreeNumberList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000002</DescriptorUI>
    <DescriptorName><String>Beta</String></DescriptorName>
    <TreeNumberList><TreeNumber>X1.100</TreeNumber></TreeNumberList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000003</DescriptorUI>
    <DescriptorName><String>Gamma</String></DescriptorName>
    <TreeNumberList><TreeNumber>X1.100.200</TreeNumber></TreeNumberList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000004</DescriptorUI>
    <DescriptorName><String>Delta</String></DescriptorName>
    <TreeNumberList><TreeNumber>X1.100.300</TreeNumber></TreeNumberList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000005</DescriptorUI>
    <DescriptorName><String>Epsilon</String></DescriptorName>
    <TreeNumberList><TreeNumber>X1.100.400</TreeNumber></TreeNumberList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000006</DescriptorUI>
    <DescriptorName><String>Zeta</String></DescriptorName>
    <TreeNumberList><TreeNumber>X1.100.500</TreeNumber></TreeNumberList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000007</DescriptorUI>
    <DescriptorName><String>Eta</String></DescriptorName>
    <TreeNumberList><TreeNumber>X1.100.500.900</TreeNumber></TreeNumberList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000008</DescriptorUI>
    <DescriptorName><String>Theta</String></DescriptorName>
    <TreeNumberList>
      <TreeNumber>X1.100.500.950</TreeNumber>
      <TreeNumber>X1.100.600</TreeNumber>
    </TreeNumberList>
  </DescriptorRecord>
</DescriptorRecordSet>
"""

# Unusable records around one good one: no DescriptorName at all, a blank name, and a name with
# an empty TreeNumberList. All three must be skipped *and counted* (no silent drops).
SKIP_FIXTURE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<DescriptorRecordSet LanguageCode="eng">
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000101</DescriptorUI>
    <TreeNumberList><TreeNumber>Y1.100</TreeNumber></TreeNumberList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000102</DescriptorUI>
    <DescriptorName><String>   </String></DescriptorName>
    <TreeNumberList><TreeNumber>Y1.200</TreeNumber></TreeNumberList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000103</DescriptorUI>
    <DescriptorName><String>NoTree</String></DescriptorName>
    <TreeNumberList />
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000104</DescriptorUI>
    <DescriptorName><String>Kept</String></DescriptorName>
    <TreeNumberList><TreeNumber>Y1.100</TreeNumber></TreeNumberList>
  </DescriptorRecord>
</DescriptorRecordSet>
"""

EXPECTED_BY_NAME: dict[str, list[str]] = {
    "Alpha": ["X1"],
    "Beta": ["X1.100"],
    "Gamma": ["X1.100.200"],
    "Delta": ["X1.100.300"],
    "Epsilon": ["X1.100.400"],
    "Zeta": ["X1.100.500"],
    "Eta": ["X1.100.500.900"],
    "Theta": ["X1.100.500.950", "X1.100.600"],
}


@pytest.fixture
def fixture_xml(tmp_path: Path) -> Path:
    path = tmp_path / "desc_fixture.xml"
    path.write_text(FIXTURE_XML, encoding="utf-8")
    return path


@pytest.fixture
def index(fixture_xml: Path) -> MeshIndex:
    return MeshIndex(by_name=parse_descriptors(fixture_xml), meta={"year": MESH_YEAR})


def test_constants_point_at_the_2026_descriptor_files() -> None:
    assert MESH_YEAR == 2026
    assert DESC_XML_URL == (
        "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml"
    )
    assert DESC_GZ_URL == "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.gz"


def test_parent_tree_number_drops_the_last_dotted_component() -> None:
    assert parent_tree_number("C14.280.067") == "C14.280"
    assert parent_tree_number("X1.100.300.400") == "X1.100.300"
    assert parent_tree_number("C14") is None
    assert parent_tree_number("X1") is None


def test_parse_descriptors_maps_names_to_tree_numbers(fixture_xml: Path) -> None:
    by_name = parse_descriptors(fixture_xml)
    assert by_name == EXPECTED_BY_NAME


def test_parse_descriptors_reads_gzipped_files_identically(
    fixture_xml: Path, tmp_path: Path
) -> None:
    gz_path = tmp_path / "desc_fixture.xml.gz"
    with gzip.open(gz_path, "wb") as fh:
        fh.write(FIXTURE_XML.encode("utf-8"))
    assert parse_descriptors(gz_path) == EXPECTED_BY_NAME


def test_parse_descriptors_skips_unusable_records_and_counts_them(tmp_path: Path) -> None:
    path = tmp_path / "skip_fixture.xml"
    path.write_text(SKIP_FIXTURE_XML, encoding="utf-8")
    by_name, stats = parse_descriptors_with_stats(path)
    assert by_name == {"Kept": ["Y1.100"]}
    assert stats["n_records"] == 4
    assert stats["n_kept"] == 1
    assert stats["n_skipped_no_name"] == 2  # a missing name and a blank <String> both count
    assert stats["n_skipped_no_tree_numbers"] == 1  # empty <TreeNumberList />
    assert stats["n_duplicate_names"] == 0
    assert stats["n_tree_numbers"] == 1
    # R5 / no silent drops: every record is accounted for.
    assert stats["n_records"] == (
        stats["n_kept"] + stats["n_skipped_no_name"] + stats["n_skipped_no_tree_numbers"]
    )


def test_index_shape_helpers(index: MeshIndex) -> None:
    assert index.n_descriptors == 8
    assert index.has_descriptor("Theta")
    assert not index.has_descriptor("Kappa")
    assert index.describe("Eta")["parents"] == ["X1.100.500"]
    assert index.describe("Eta")["tree_numbers"] == ["X1.100.500.900"]
    assert index.describe("Kappa")["known"] is False


def test_siblings_use_the_deepest_tree_number_only_when_it_is_enough(index: MeshIndex) -> None:
    """Gamma's deepest (only) parent X1.100 has five children, so no shallower level is added."""
    assert index.siblings("Gamma") == ["Delta", "Epsilon", "Theta", "Zeta"]


def test_siblings_fall_back_to_a_shallower_tree_number(index: MeshIndex) -> None:
    """Theta's deepest parent (X1.100.500) yields only Eta; the next number adds X1.100's group."""
    assert index.siblings("Theta") == ["Delta", "Epsilon", "Eta", "Gamma", "Zeta"]


def test_siblings_stop_at_the_deepest_level_when_min_siblings_is_met(index: MeshIndex) -> None:
    """The same descriptor with ``min_siblings=1`` stops at X1.100.500 and keeps only Eta."""
    assert index.siblings("Theta", min_siblings=1) == ["Eta"]


def test_siblings_never_include_the_descriptor_itself(index: MeshIndex) -> None:
    for name in index.by_name:
        assert name not in index.siblings(name)
    # Theta shares a parent with Eta both at X1.100.500 and at X1.100, the tightest case.
    assert index.siblings("Theta", min_siblings=1) == ["Eta"]
    assert index.siblings("Eta", min_siblings=1) == ["Theta"]


def test_siblings_of_a_narrow_branch_are_exhausted_not_padded(index: MeshIndex) -> None:
    """Eta has one tree number whose parent group holds two descriptors: one sibling, no more."""
    assert index.siblings("Eta") == ["Theta"]


def test_siblings_of_a_top_level_descriptor_are_empty(index: MeshIndex) -> None:
    assert index.siblings("Alpha") == []


def test_siblings_of_an_unknown_descriptor_are_empty(index: MeshIndex) -> None:
    assert index.siblings("Kappa") == []
    assert index.siblings("Kappa", max_siblings=2, min_siblings=0) == []


def test_max_siblings_truncates_the_sorted_list(index: MeshIndex) -> None:
    assert index.siblings("Theta", max_siblings=2) == ["Delta", "Epsilon"]


def test_save_and_load_index_round_trip(index: MeshIndex, tmp_path: Path) -> None:
    index.meta.update({"year": MESH_YEAR, "url": DESC_GZ_URL, "n_descriptors": index.n_descriptors})
    path = tmp_path / "nested" / "mesh_index.json"
    saved = save_index(index, path)
    assert saved == path and path.is_file()
    text = path.read_text(encoding="utf-8")
    assert text.count("\n") == 1  # compact: one line of JSON
    assert text.startswith('{"meta":')
    assert '"by_name"' in text

    loaded = load_index(path)
    assert loaded == index  # dataclass equality covers by_name and meta
    assert loaded.n_descriptors == index.n_descriptors
    assert loaded.siblings("Theta") == index.siblings("Theta")


def test_load_index_rejects_a_file_that_is_not_an_index(tmp_path: Path) -> None:
    path = tmp_path / "not_an_index.json"
    path.write_text('{"meta": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="by_name"):
        load_index(path)


def test_download_descriptors_reuses_an_existing_file_without_network(tmp_path: Path) -> None:
    """A file already in the cache is returned as-is; no client is created, so no network."""
    cached = tmp_path / "desc2026.gz"
    cached.write_bytes(b"\x1f\x8b cached bytes")
    assert is_gzip(cached)
    assert download_descriptors(tmp_path) == cached
    assert list(tmp_path.iterdir()) == [cached]  # reused in place; nothing else written
