"""Fresh-tier PubMed builder: MeSH headings, publication types and check tags.

Records come from the PubMed update files (``pubmed26nNNNN.xml.gz``), and are selected by
the **Entrez** date inside the build window — the date the record entered PubMed, which is
the structured field that decides freshness. Gold is a lookup of a structured field
(publication type, check tag, MeSH descriptor); nothing here is model-produced.

The templates:

* ``pubmed_pubtype_choice_v1`` — study design from a fixed 6-way label set.
* ``pubmed_humans_noul_v1`` — does the record study humans (check tag ``Humans``)?
* ``pubmed_mesh_major_choice_v1`` — the record's major MeSH topic among seeded distractors.
* ``pubmed_observational_noul_v1`` — is the study observational (no trial/review design)?

Answer-bearing text is removed from the state: the publication-type list, the MeSH
headings, and the check tags never enter the state. Terms such as "randomised" that occur in
the abstract itself cannot be removed without altering the source text; the T8 regex/BoW
screen is what decides whether those make a template too easy.
"""

from __future__ import annotations

import gzip
import random
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from meddecide.bench.schema import QuestionType, Tier, make_item, split_by_record_hash

SOURCE = "pubmed"
DATASET_ID = "pubmed/updatefiles"
LICENSE = "NLM-public-domain"  # NLM: PubMed records are public; see https://www.nlm.nih.gov/databases/download/terms_and_conditions.html

# Fixed label set, in priority order: a record with several publication types takes the
# earliest matching label. The order is part of the template definition and is recorded.
PUBTYPE_LABELS = [
    "Randomized controlled trial",
    "Systematic review",
    "Meta-analysis",
    "Case report",
    "Observational study",
    "Other",
]
PUBTYPE_TAG_MAP = {
    "Randomized Controlled Trial": "Randomized controlled trial",
    "Systematic Review": "Systematic review",
    "Meta-Analysis": "Meta-analysis",
    "Case Reports": "Case report",
    "Observational Study": "Observational study",
}  # + Clinical Trial falls under randomised only if explicitly randomised; handled below
CLINICAL_TRIAL_TAG = "Clinical Trial"
# Check tags that decide the humans/animals template. Drawn from the MeSH descriptor list
# because <CheckTagList> is absent from current PubMed XML.
CHECK_TAG_NAMES = {"Humans", "Animals", "Male", "Female", "Pregnancy", "Mice", "Rats"}
ANSWER_TERM_RE = re.compile(
    r"\b(randomi[sz]ed|systematic review|meta-?analys[ei]s|case report|observational study)\b",
    re.IGNORECASE,
)


@dataclass
class PubmedRecord:
    """The subset of a PubMed record this builder needs (no UMLS/credentialed content)."""

    pmid: str
    entrez_date: date
    title: str
    abstract: str
    pub_types: list[str]
    mesh_major_topics: list[str]
    check_tags: list[str]
    journal: str
    doi: str | None = None

    @property
    def state_text(self) -> str:
        return f"{self.title}\n\n{self.abstract}".strip()


def _text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _date_from(pub_date: ET.Element | None) -> date | None:
    if pub_date is None:
        return None
    year = _text(pub_date.find("Year"))
    month = _text(pub_date.find("Month"))
    day = _text(pub_date.find("Day"))
    if not year:
        return None
    month_num = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }.get(month[:3].lower())
    try:
        return date(int(year), month_num or 1, int(day) if day.isdigit() else 1)
    except ValueError:
        return None


def parse_pubmed_article(article: ET.Element) -> PubmedRecord | None:
    """Parse one ``<PubmedArticle>`` into a :class:`PubmedRecord`, or ``None`` if unusable."""
    citation = article.find("MedlineCitation")
    if citation is None:
        return None
    pmid = _text(citation.find("PMID"))
    art = citation.find("Article")
    if not pmid or art is None:
        return None

    # Entrez date: the date the record entered PubMed (structured, authoritative for freshness)
    entrez: date | None = None
    history = citation.find("History")
    if history is not None:
        for pub_date in history.findall("PubMedPubDate"):
            if pub_date.get("PubStatus") == "entrez":
                entrez = _date_from(pub_date)
                break
    if entrez is None:
        entrez = _date_from(art.find("Journal/JournalIssue/PubDate")) or _date_from(
            citation.find("DateCompleted")
        )
    if entrez is None:
        return None

    title = _text(art.find("ArticleTitle"))
    abstract_parts = [(_text(node)) for node in art.findall("Abstract/AbstractText")]
    abstract = " ".join(p for p in abstract_parts if p)
    pub_types = [_text(node) for node in art.findall("PublicationTypeList/PublicationType")]
    pub_types = [p for p in pub_types if p]
    if not pub_types:
        return None

    mesh_major: list[str] = []
    for heading in citation.findall("MeshHeadingList/MeshHeading"):
        descriptor = heading.find("DescriptorName")
        if descriptor is not None and descriptor.get("MajorTopicYN") == "Y":
            name = _text(descriptor)
            if name:
                mesh_major.append(name)

    # In current PubMed XML the check tags (Humans / Animals / Male / Female …) appear as
    # DescriptorName headings, not as a separate <CheckTagList> (that element no longer
    # occurs in practice — verified on the 2026 update files). Collect both shapes.
    check_tags: list[str] = []
    for descriptor in citation.findall("MeshHeadingList/MeshHeading/DescriptorName"):
        name = _text(descriptor)
        if name in CHECK_TAG_NAMES:
            check_tags.append(name)
    for node in citation.findall(".//CheckTagList/CheckTag"):
        name = _text(node)
        if name and name not in check_tags:
            check_tags.append(name)

    journal = _text(art.find("Journal/Title"))
    doi = None
    for elocation in article.findall(".//ArticleIdList/ArticleId"):
        if elocation.get("IdType") == "doi":
            doi = _text(elocation)
            break
    return PubmedRecord(
        pmid=pmid,
        entrez_date=entrez,
        title=title,
        abstract=abstract,
        pub_types=pub_types,
        mesh_major_topics=mesh_major,
        check_tags=check_tags,
        journal=journal,
        doi=doi,
    )


def iter_records(
    paths: list[Path],
    *,
    limit: int | None = None,
    stop_before: date | None = None,
    max_consecutive_old: int = 20000,
) -> Iterator[PubmedRecord]:
    """Stream records out of gzipped PubMed XML files (no full-file parse in memory).

    ``stop_before`` skips records older than the window and stops reading a file after
    ``max_consecutive_old`` consecutive old records, so a full scan of update files does not
    dominate the build. The guard is deliberately generous: PubMed files are not sorted by
    date, and stopping early must never be able to hide a fresh record.
    """
    count = 0
    for path in paths:
        consecutive_old = 0
        with gzip.open(path, "rb") as fh:
            for _event, element in ET.iterparse(fh, events=("end",)):
                if element.tag != "PubmedArticle":
                    continue
                record = parse_pubmed_article(element)
                element.clear()
                if record is None:
                    continue
                if stop_before is not None and record.entrez_date < stop_before:
                    consecutive_old += 1
                    if consecutive_old >= max_consecutive_old:
                        break
                    continue
                consecutive_old = 0
                yield record
                count += 1
                if limit is not None and count >= limit:
                    return


def pubtype_label(pub_types: list[str]) -> str | None:
    """Map a record's publication types onto the fixed label set, or ``None`` if unusable."""
    for tag in pub_types:
        if tag in PUBTYPE_TAG_MAP:
            return PUBTYPE_TAG_MAP[tag]
    if CLINICAL_TRIAL_TAG in pub_types:
        # A clinical trial that is not explicitly randomised is not an RCT.
        return "Other"
    if "Review" in pub_types:
        return "Other"
    return "Other"


def is_observational(pub_types: list[str]) -> bool:
    """True when the record is an observational study (or an unlabelled non-trial design)."""
    return pubtype_label(pub_types) == "Observational study"


def build_items(
    records: list[PubmedRecord],
    *,
    cfg: dict[str, Any],
    window_start: date,
    window_end: date,
    mesh_distractor_pool: list[str] | None = None,
) -> tuple[list[Any], dict[str, int], list[str]]:
    """Turn PubMed records into fresh-tier items; returns ``(items, drops, notes)``."""
    seed = int(cfg.get("seed", 0))
    drops: dict[str, int] = {}
    notes: list[str] = []

    def drop(reason: str, n: int = 1) -> None:
        drops[reason] = drops.get(reason, 0) + n

    in_window = [r for r in records if window_start <= r.entrez_date <= window_end]
    drop("entrez_date_outside_window", len(records) - len(in_window))

    usable = [r for r in in_window if r.title and r.abstract]
    drop("missing_title_or_abstract", len(in_window) - len(usable))

    # Major MeSH topic pool: the most frequent major topics across the whole window. These
    # are the distractors for the MeSH template; the pool is derived from the same build.
    pool = mesh_distractor_pool or _mesh_pool(usable, min_count=5)
    if len(pool) < 4:
        notes.append(f"major-MeSH pool has only {len(pool)} topics; MeSH template may be thin")

    rng = random.Random(f"{seed}:pubmed:distractors")
    items: list[Any] = []
    n_explicit_design_term = 0

    for record in usable:
        record_date = record.entrez_date
        split = split_by_record_hash(record.pmid, dev_fraction=0.2, salt="pubmed")
        url = f"https://pubmed.ncbi.nlm.nih.gov/{record.pmid}/"
        state = record.state_text
        if ANSWER_TERM_RE.search(state):
            n_explicit_design_term += 1

        # --- publication type (choice, 6-way) ---
        label = pubtype_label(record.pub_types)
        if label is None:
            drop("no_publication_type")
        else:
            items.append(
                make_item(
                    tier=Tier.FRESH,
                    source=SOURCE,
                    source_record_id=record.pmid,
                    source_url=url,
                    source_license=LICENSE,
                    record_date=record_date,
                    split=split,
                    template_id="pubmed_pubtype_choice_v1",
                    skill="evidence",
                    qtype=QuestionType.CHOICE,
                    state=state,
                    question="What study design does this record report?",
                    options=[
                        {"key": chr(ord("A") + i), "label": text}
                        for i, text in enumerate(PUBTYPE_LABELS)
                    ],
                    gold=chr(ord("A") + PUBTYPE_LABELS.index(label)),
                    option_order_seed=seed,
                    meta={"source_field": "PublicationTypeList",
                          "publication_types": record.pub_types,
                          "journal": record.journal,
                          "doi": record.doi},
                )
            )

        # --- humans vs animals (noul, check tags) ---
        tags = set(record.check_tags)
        if "Humans" in tags or "Animals" in tags:
            items.append(
                make_item(
                    tier=Tier.FRESH,
                    source=SOURCE,
                    source_record_id=record.pmid,
                    source_url=url,
                    source_license=LICENSE,
                    record_date=record_date,
                    split=split,
                    template_id="pubmed_humans_noul_v1",
                    skill="evidence",
                    qtype=QuestionType.NOUL,
                    state=state,
                    question="Does this record study human subjects?",
                    options=[{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}],
                    gold="yes" if "Humans" in tags else "no",
                    option_order_seed=seed,
                    meta={"source_field": "CheckTagList", "check_tags": record.check_tags,
                          "journal": record.journal},
                )
            )
        else:
            drop("no_humans_or_animals_check_tag")

        # --- major MeSH topic (choice, seeded distractors) ---
        topic = record.mesh_major_topics[0] if record.mesh_major_topics else None
        if topic and topic in pool:
            others = [t for t in pool if t != topic]
            k = min(3, len(others))
            distractors = rng.sample(others, k)
            labels = [*distractors, topic]
            order = list(range(len(labels)))
            rng.shuffle(order)
            items.append(
                make_item(
                    tier=Tier.FRESH,
                    source=SOURCE,
                    source_record_id=record.pmid,
                    source_url=url,
                    source_license=LICENSE,
                    record_date=record_date,
                    split=split,
                    template_id="pubmed_mesh_major_choice_v1",
                    skill="knowledge",
                    qtype=QuestionType.CHOICE,
                    state=state,
                    question="Which major MeSH topic is this record indexed under?",
                    options=[
                        {"key": chr(ord("A") + i), "label": labels[j]}
                        for i, j in enumerate(order)
                    ],
                    gold=chr(ord("A") + order.index(len(labels) - 1)),
                    option_order_seed=seed,
                    meta={"source_field": "MeshHeadingList/DescriptorName[@MajorTopicYN=Y]",
                          "major_topics": record.mesh_major_topics,
                          "distractor_pool_size": len(pool),
                          "journal": record.journal},
                )
            )
        else:
            drop("no_major_mesh_topic_in_pool")

        # --- observational design (noul) ---
        items.append(
            make_item(
                tier=Tier.FRESH,
                source=SOURCE,
                source_record_id=record.pmid,
                source_url=url,
                source_license=LICENSE,
                record_date=record_date,
                split=split,
                template_id="pubmed_observational_noul_v1",
                skill="evidence",
                qtype=QuestionType.NOUL,
                state=state,
                question="Is this an observational study (rather than a trial, review or case report)?",
                options=[{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}],
                gold="yes" if is_observational(record.pub_types) else "no",
                option_order_seed=seed,
                meta={"source_field": "PublicationTypeList",
                      "publication_types": record.pub_types, "journal": record.journal},
            )
        )

    notes.append(
        f"{n_explicit_design_term} of {len(usable)} records state their design term in the "
        "title/abstract; the T8 regex/BoW screen decides whether the publication-type "
        "template is too easy"
    )
    notes.append(f"major-MeSH distractor pool: {len(pool)} topics")
    return items, drops, notes


def _mesh_pool(records: list[PubmedRecord], *, min_count: int) -> list[str]:
    counts: dict[str, int] = {}
    for record in records:
        for topic in record.mesh_major_topics:
            counts[topic] = counts.get(topic, 0) + 1
    return sorted(t for t, n in counts.items() if n >= min_count)
