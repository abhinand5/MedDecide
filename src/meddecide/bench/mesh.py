"""Public MeSH descriptor index — names and tree numbers — for near-miss distractors.

NLM MeSH is **public domain** — see
https://www.nlm.nih.gov/databases/download/terms_and_conditions.html — so descriptor names and
tree numbers may be committed and shipped with the benchmark.

What this module gives the benchmark builder is *siblings*: descriptors that sit under the
same immediate parent tree number, i.e. the closest lexical/structural neighbours of a gold
MeSH heading. Those make plausible near-miss distractors, which is exactly what a `choice`
template needs. The index is rebuilt from the published NLM file by
``scripts/bench/fetch_mesh.py`` — nothing here is hand-curated.

Layout of the XML: ``DescriptorRecordSet > DescriptorRecord`` with
``DescriptorName/String`` and ``TreeNumberList/TreeNumber``. Records are streamed with
``ElementTree.iterparse`` and cleared as we go; the 2026 descriptor file is ~300 MB
uncompressed and must never be materialised in memory.
"""

from __future__ import annotations

import gzip
import json
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MESH_YEAR = 2026
DESC_XML_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml"
DESC_GZ_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.gz"
SOURCE = "mesh"
LICENSE = "public-domain"  # NLM MeSH: public domain (US government work)

#: Socket-level timeout for the download: MeSH is served from a slow static host, so the
#: read timeout is generous while connect/pool stay short enough to fail fast.
DOWNLOAD_TIMEOUT_S = 600.0
CONNECT_TIMEOUT_S = 30.0
CHUNK_SIZE = 1 << 20  # 1 MiB


def parent_tree_number(tree_number: str) -> str | None:
    """Immediate parent of a tree number: the number with its last dotted part removed.

    ``"C14.280.067" -> "C14.280"``; a top-level number (``"C14"``) has no parent, so ``None``.
    """
    parent, _, _ = tree_number.rpartition(".")
    return parent or None


def _tree_sort_key(tree_number: str) -> tuple[int, int, str]:
    """Sort key ordering tree numbers most-specific (deepest) first, deterministically."""
    return (-tree_number.count("."), -len(tree_number), tree_number)


def is_gzip(path: Path) -> bool:
    """True when the file starts with the gzip magic bytes (cheap integrity check)."""
    with Path(path).open("rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def download_descriptors(cache_dir: Path, *, prefer_gz: bool = True) -> Path:
    """Download the MeSH descriptor file into ``cache_dir`` and return its path.

    The compressed file is preferred (16.8 MB vs 313 MB for 2026). A non-empty file already
    in ``cache_dir`` is reused as-is, so repeated runs are free and the raw download stays
    immutable (R9). The body is streamed to a ``*.part`` file **inside** ``cache_dir`` and
    renamed into place only after the byte count is verified, so an interrupted download can
    never be mistaken for a complete one and nothing is ever written outside ``cache_dir``.

    The remote host serves ``desc2026.gz`` with ``Content-Encoding: gzip`` on a body that is
    *already* the gzip file, so the bytes are taken with ``iter_raw`` — content-decoding the
    response would inflate it back to the 313 MB XML and the stored file would no longer be
    what NLM published. A gzip magic-byte check guards that contract.
    """
    import httpx  # imported here: the module stays importable without the network stack

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = DESC_GZ_URL if prefer_gz else DESC_XML_URL
    target = cache_dir / url.rsplit("/", 1)[-1]
    if target.is_file() and target.stat().st_size > 0:
        return target

    tmp = cache_dir / (target.name + ".part")
    timeout = httpx.Timeout(
        DOWNLOAD_TIMEOUT_S,
        connect=CONNECT_TIMEOUT_S,
        read=DOWNLOAD_TIMEOUT_S,
        write=DOWNLOAD_TIMEOUT_S,
    )
    n_bytes = 0
    try:
        with (
            httpx.Client(follow_redirects=True, timeout=timeout) as http,
            http.stream("GET", url) as response,
        ):
            response.raise_for_status()
            expected = response.headers.get("content-length")
            with tmp.open("wb") as fh:
                for chunk in response.iter_raw(chunk_size=CHUNK_SIZE):
                    fh.write(chunk)
                    n_bytes += len(chunk)
        if expected is not None and n_bytes != int(expected):
            raise RuntimeError(f"short download: got {n_bytes} of {expected} bytes from {url}")
        if n_bytes == 0:
            raise RuntimeError(f"empty download from {url}")
        if prefer_gz and not is_gzip(tmp):
            raise RuntimeError(f"downloaded file is not gzip despite the .gz URL: {url}")
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def parse_descriptors_with_stats(path: Path) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Stream-parse a MeSH descriptor XML (or ``.gz``) into ``{name: [tree numbers]}`` + counts.

    Records with no ``DescriptorName/String`` or no ``TreeNumberList/TreeNumber`` are skipped
    and counted, never silently dropped (AGENTS.md data rules). Tree numbers keep their file
    order; ``stats["n_tree_numbers"]`` counts every tree number kept.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    by_name: dict[str, list[str]] = {}
    # Zero-initialised so every count is present even when nothing was skipped (the CLI
    # builds its meta block from these keys).
    stats: Counter[str] = Counter(
        {
            "n_records": 0,
            "n_kept": 0,
            "n_skipped_no_name": 0,
            "n_skipped_no_tree_numbers": 0,
            "n_duplicate_names": 0,
            "n_tree_numbers": 0,
        }
    )
    root: ET.Element | None = None
    with opener(path, "rb") as fh:
        for event, elem in ET.iterparse(fh, events=("start", "end")):
            if event == "start":
                if root is None:
                    root = elem
                continue
            if elem.tag != "DescriptorRecord":
                continue
            stats["n_records"] += 1
            name = (elem.findtext("DescriptorName/String") or "").strip()
            tree_numbers = [
                (node.text or "").strip() for node in elem.findall("TreeNumberList/TreeNumber")
            ]
            tree_numbers = [t for t in tree_numbers if t]
            if not name:
                stats["n_skipped_no_name"] += 1
            elif not tree_numbers:
                stats["n_skipped_no_tree_numbers"] += 1
            else:
                stored = by_name.get(name)
                if stored is None:
                    by_name[name] = tree_numbers
                    stats["n_tree_numbers"] += len(tree_numbers)
                else:  # not expected: MeSH descriptor names are unique
                    stats["n_duplicate_names"] += 1
                    added = [t for t in tree_numbers if t not in stored]
                    stored.extend(added)
                    stats["n_tree_numbers"] += len(added)
            elem.clear()  # keep only the current record in memory
            if root is not None:  # drop the processed record from the tree as well
                root.remove(elem)
    stats["n_kept"] = len(by_name)
    return by_name, dict(stats)


def parse_descriptors(path: Path) -> dict[str, list[str]]:
    """``{descriptor_name: [tree numbers]}`` for every usable record in a MeSH descriptor file."""
    by_name, _ = parse_descriptors_with_stats(path)
    return by_name


@dataclass
class MeshIndex:
    """A MeSH descriptor index: ``by_name`` maps a descriptor name to its tree numbers."""

    by_name: dict[str, list[str]]
    meta: dict[str, Any] = field(default_factory=dict)
    # Lazily built reverse index: immediate parent tree number -> descriptors under it.
    # `siblings()` without it rescans all 31k descriptors per call (~13 ms), which is fine
    # for a handful of lookups and hopeless for a template builder that sweeps 300k records.
    _children_by_parent: dict[str, list[str]] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def n_descriptors(self) -> int:
        return len(self.by_name)

    def has_descriptor(self, name: str) -> bool:
        return name in self.by_name

    def children_by_parent(self) -> dict[str, list[str]]:
        """``{parent tree number: [descriptor names]}``, built once and cached."""
        if self._children_by_parent is None:
            out: dict[str, list[str]] = {}
            for descriptor, tree_numbers in self.by_name.items():
                for tree_number in tree_numbers:
                    parent = parent_tree_number(tree_number)
                    if parent:
                        out.setdefault(parent, []).append(descriptor)
            for names in out.values():
                names.sort()
            self._children_by_parent = out
        return self._children_by_parent

    def siblings(
        self,
        name: str,
        *,
        max_siblings: int | None = None,
        min_siblings: int = 3,
    ) -> list[str]:
        """Descriptors sharing an immediate parent tree number with ``name``.

        Parent = the tree number with its last dotted component removed, so siblings of
        ``C14.280.067`` are the other descriptors under ``C14.280``. The most specific
        (deepest) tree number of ``name`` is used first; if that level yields fewer than
        ``min_siblings``, the next-deepest tree number is added, and so on — a descriptor
        that is a leaf of a narrow branch still gets distractors from its broader branch.
        ``name`` itself is never included, the result is sorted, and an unknown descriptor
        returns ``[]``.
        """
        tree_numbers = self.by_name.get(name)
        if not tree_numbers:
            return []
        children = self.children_by_parent()
        found: set[str] = set()
        for tree_number in sorted(dict.fromkeys(tree_numbers), key=_tree_sort_key):
            parent = parent_tree_number(tree_number)
            if parent is None:  # a top-level number has no sibling group
                continue
            for other in children.get(parent, ()):  # already sorted
                if other != name:
                    found.add(other)
            if len(found) >= min_siblings:
                break
        ordered = sorted(found)
        return ordered[:max_siblings] if max_siblings is not None else ordered

    def describe(self, name: str) -> dict[str, Any]:
        """Small JSON-friendly summary of one descriptor (tree numbers and their parents)."""
        tree_numbers = list(self.by_name.get(name, []))
        return {
            "name": name,
            "known": bool(tree_numbers),
            "tree_numbers": tree_numbers,
            "parents": sorted({p for t in tree_numbers if (p := parent_tree_number(t))}),
        }


def save_index(index: MeshIndex, path: Path) -> Path:
    """Write the index as compact JSON ``{"meta": {...}, "by_name": {...}}`` (sorted keys)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": index.meta, "by_name": dict(sorted(index.by_name.items()))}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return path


def load_index(path: Path) -> MeshIndex:
    """Read back an index written by :func:`save_index`."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("by_name"), dict):
        raise ValueError(f"not a MeSH index (missing 'by_name'): {path}")
    by_name = {str(name): [str(t) for t in trees] for name, trees in payload["by_name"].items()}
    meta = dict(payload.get("meta") or {})
    return MeshIndex(by_name=by_name, meta=meta)
