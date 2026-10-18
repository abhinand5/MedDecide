"""DrugProt (CC BY 4.0; human chemical-gene relations in PubMed abstracts) as osler_v3 items (D42; ADVISORY X1).

Source: ``OpenMed/drugprot-parquet`` at a pinned revision (train and validation splits). Template ``drugprot_relation_v1``
(choice): for a chemical and a gene marked in an abstract, which relation holds? The options are DrugProt's relation types
(fixed, sorted) plus "no relation". Labels come only from the annotated relations.

- Positive pairs: each (chemical, gene) pair with one annotated relation type. A pair with two or more types is excluded
  (its gold is not one type) and counted.
- Negatives: (chemical, gene) pairs that co-occur in one sentence with no annotated relation, sampled at most 1:1 with the
  positives of the same split (a stable-hash order). Every candidate and every sample is counted.
- Splits: the official train split is train. The validation split is divided into dev and test by PMID (stable hash, one
  half each), as ADVISORY X1 asks; a PMID never crosses splits.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from meddecide.mix.rows import make_row
from meddecide.mix.sentences import sentence_index, sentence_spans
from meddecide.utils.hashing import stable_hash

SOURCE = "drugprot"
TEMPLATE = "drugprot_relation_v1"
SKILL = "relation_extraction"
LICENCE = "CC BY 4.0"
REVISION = "baba427cb0a238e70dd383bf1ec2a70f4ccef078"
REPO_URL = "https://huggingface.co/datasets/OpenMed/drugprot-parquet"
TIER = "osler_v3_source"
NO_RELATION = "no relation"


def pmid_split(pmid: str, official: str) -> str:
    """``train`` for the official train split; ``dev`` or ``test`` (one half each, by stable hash) for the validation split."""
    if official == "train":
        return "train"
    if official != "validation":
        raise ValueError(f"unknown official split {official!r}")
    return "dev" if int(stable_hash({"drugprot_pmid": str(pmid)}, length=8), 16) % 2 == 0 else "test"


def relation_types(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """DrugProt's relation types over the given records, sorted (the fixed option set of the template)."""
    return sorted({str(rel["type"]) for rec in records for rel in rec["relations"]})


def option_keys(types: Sequence[str]) -> dict[str, str]:
    """Letters for the relation types in order, and the last letter for "no relation"."""
    letters = [chr(ord("A") + i) for i in range(len(types) + 1)]
    return {**{name: letters[i] for i, name in enumerate(types)}, NO_RELATION: letters[-1]}


def chem_gene_pairs(record: Mapping[str, Any]) -> tuple[dict[tuple[str, str], set[str]], Counter[str]]:
    """Annotated (chemical, gene) pairs of one abstract: pair -> its relation types. Counts relations that are not chemical-gene."""
    counts: Counter[str] = Counter()
    entities = {str(e["id"]): e for e in record["entities"]}
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for rel in record["relations"]:
        a, b = entities.get(str(rel["arg1"])), entities.get(str(rel["arg2"]))
        if a is None or b is None:
            counts["relation_with_unknown_entity"] += 1
            continue
        if a["type"] == "CHEMICAL" and _is_gene(b):
            pair = (str(a["id"]), str(b["id"]))
        elif _is_gene(a) and b["type"] == "CHEMICAL":
            pair = (str(b["id"]), str(a["id"]))
        else:
            counts["relation_not_chemical_gene"] += 1
            continue
        pairs[pair].add(str(rel["type"]))
    return dict(pairs), counts


def co_sentence_pairs(record: Mapping[str, Any]) -> set[tuple[str, str]]:
    """(chemical, gene) pairs whose two entities lie in the same sentence of the abstract text."""
    text = str(record["text"])
    spans = sentence_spans(text)
    by_sentence: dict[int, dict[str, list[str]]] = defaultdict(lambda: {"chem": [], "gene": []})
    for entity in record["entities"]:
        index = sentence_index(spans, int(entity["start"]))
        if index is None:
            continue
        if entity["type"] == "CHEMICAL":
            by_sentence[index]["chem"].append(str(entity["id"]))
        elif _is_gene(entity):
            by_sentence[index]["gene"].append(str(entity["id"]))
    return {(c, g) for slot in by_sentence.values() for c in slot["chem"] for g in slot["gene"]}


def build(records: Sequence[tuple[str, Mapping[str, Any]]], options: Sequence[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Items by split (``train``, ``dev``, ``test``) from (official split, abstract) pairs, with every count.

    ``options`` is the fixed list of relation types (see :func:`relation_types`); every item offers all of them and "no relation".
    """
    keys = option_keys(options)
    counts: Counter[str] = Counter()
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    pool: dict[str, list[tuple[Mapping[str, Any], str, str]]] = {"train": [], "dev": [], "test": []}
    for official, record in records:
        counts["abstracts"] += 1
        split = pmid_split(str(record["pmid"]), official)
        counts[f"abstracts_{split}"] += 1
        pairs, pair_counts = chem_gene_pairs(record)
        counts.update(pair_counts)
        for (chem, gene), types in sorted(pairs.items()):
            if len(types) > 1:
                counts["dropped_multi_label_pairs"] += 1
                continue
            rows[split].append(_item(record, split, chem, gene, next(iter(types)), keys))
        pool[split].extend((record, chem, gene) for chem, gene in sorted(co_sentence_pairs(record) - set(pairs)))
    for split in ("train", "dev", "test"):
        k = len(rows[split])
        counts[f"positives_{split}"] = k
        ranked = sorted(pool[split], key=lambda c: stable_hash({"negative": f"{c[0]['pmid']}:{c[1]}:{c[2]}"}, length=16))
        for record, chem, gene in ranked[:k]:
            rows[split].append(_item(record, split, chem, gene, NO_RELATION, keys))
        counts[f"negative_candidates_{split}"] = len(pool[split])
        counts[f"negatives_sampled_{split}"] = min(k, len(pool[split]))
    return rows, {"counts": dict(counts), "options": [*options, NO_RELATION], "keys": keys}


def _is_gene(entity: Mapping[str, Any]) -> bool:
    return str(entity["type"]).startswith("GENE")


def _item(record: Mapping[str, Any], split: str, chem: str, gene: str, relation: str, keys: Mapping[str, str]) -> dict[str, Any]:
    entities = {str(e["id"]): e for e in record["entities"]}
    chem_text = str(entities[chem]["text"])
    gene_text = str(entities[gene]["text"])
    offered = [(keys[name], name.lower()) for name in sorted(k for k in keys if k != NO_RELATION)]
    offered.append((keys[NO_RELATION], NO_RELATION))
    return make_row(
        source=SOURCE, template_id=TEMPLATE, skill=SKILL, qtype="choice", state=str(record["text"]),
        question=f"In this abstract, which relation holds between the chemical '{chem_text}' and the gene '{gene_text}'?",
        options=offered, gold=keys[relation], record_id=f"{record['pmid']}:{chem}:{gene}",
        record_url=f"https://pubmed.ncbi.nlm.nih.gov/{record['pmid']}/", licence=LICENCE, record_date="", split=split,
        tier=TIER, meta={"pmid": str(record["pmid"]), "relation": relation, "chemical": chem_text, "gene": gene_text,
                         "revision": REVISION},
    )
