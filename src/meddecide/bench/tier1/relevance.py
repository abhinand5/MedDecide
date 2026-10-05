"""Tier-1 relevance loaders: TREC-COVID, NFCorpus, SciFact (BEIR format).

These are ``query + passage`` tasks. Gold comes from the official qrels: a passage counts
as relevant only because the source's annotators said so, never because a model did. The
negative pool is drawn from the same corpus, and every sampling decision is seeded and
recorded.

Grade→level mapping for ``score`` items is fixed and recorded: a passage is level 1
("Not relevant"), 2 ("Relevant") or 3 ("Highly relevant") with the qrels grade 0/1/2
mapped to level 1/2/3. The level labels live in ``GRADE_LEVELS`` and are identical for
every relevance source, so a `score` number means the same thing everywhere.
"""

from __future__ import annotations

import random
from datetime import date
from typing import Any

from meddecide.bench.schema import QuestionType, Split, Tier, make_item, split_by_record_hash
from meddecide.bench.tier1.common import LoadResult, finalize_splits, make_choice_item

TREC_COVID_ID = "BeIR/trec-covid"
NFCORPUS_ID = "BeIR/nfcorpus"
SCIFACT_ID = "BeIR/scifact"
BEIR_LICENSE = "UNKNOWN"  # BEIR redistributes these collections; upstream terms vary

PASSAGE_CHARS = 1200
GRADE_LEVELS = ["Not relevant", "Relevant", "Highly relevant"]  # grade 0 / 1 / 2


def _qrels_by_query(qrels: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in qrels:
        out.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    return out


def _corpus_index(corpus: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    return {
        str(row["_id"]): {
            "title": str(row.get("title") or "").strip(),
            "text": str(row.get("text") or "").strip(),
        }
        for row in corpus
    }


def _passage_text(doc: dict[str, str]) -> str:
    return f"{doc['title']}\n{doc['text']}".strip()[:PASSAGE_CHARS]


def load_beir_relevance(
    *,
    source: str,
    dataset_id: str,
    queries: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    cfg: dict[str, Any],
    revision: str,
    split: Split,
    record_date: date,
    graded: bool,
    query_cap: int | None = None,
) -> LoadResult:
    """Build ``noul`` (relevant?) items and, when ``graded``, graded ``score`` items.

    ``noul``: one relevant (qrels grade > 0) and one non-relevant (grade <= 0) passage per
    sampled query. ``score``: one passage per grade level present in the qrels, plus a
    grade-0 passage from the explicitly non-relevant pool when the qrels contain none.
    """
    seed = int(cfg.get("seed", 0))
    max_items, _ = _caps(cfg)
    result = LoadResult(
        source=source,
        dataset_id=dataset_id,
        dataset_revision=revision,
        license=BEIR_LICENSE,
        split_map={split.value: f"{dataset_id}-qrels (official qrels split)"},
        notes=[
            "gold from official BEIR qrels; non-relevant passages come from the same corpus "
            "and are chosen with a fixed seed",
            f"passage text truncated to {PASSAGE_CHARS} characters",
            f"grade->level mapping: 0/1/2 -> levels 1/2/3 labelled {GRADE_LEVELS}",
        ],
    )
    qrels_map = _qrels_by_query(qrels)
    corpus_index = _corpus_index(corpus)
    query_text = {str(q["_id"]): str(q["text"]).strip() for q in queries}

    # keep only queries that have both a query text and at least one judged passage
    # Every query *passed in* is accounted for. A query the caller did not pass (e.g. a
    # BEIR train query when only the test qrels are in scope) is out of scope, not dropped;
    # a query that was passed but has no text or no judged passage cannot contribute and is
    # counted as dropped.
    in_scope = sorted(query_text)
    usable = [qid for qid in in_scope if any(cid in corpus_index for cid in qrels_map.get(qid, {}))]
    result.drop("query_without_text_or_judged_passage", len(in_scope) - len(usable))
    result.drop("query_not_in_this_qrels_split", sum(1 for q in in_scope if q not in qrels_map))
    cap = query_cap if query_cap is not None else max_items
    picked, over = _subsample(usable, cap, seed, salt=f"{source}:{split.value}:queries")
    result.drop(f"{split.value}_queries_over_cap", over)
    result.notes.append(f"queries sampled: {len(picked)} of {len(usable)} usable")

    rng = random.Random(f"{seed}:{source}:{split.value}:sample")
    judged_cids = {cid for judgements in qrels_map.values() for cid in judgements}
    unjudged = [cid for cid in corpus_index if cid not in judged_cids]
    all_grades = _all_grades(qrels_map)  # computed once, not per query
    grade0 = [cid for cid, grade in all_grades.items() if grade == 0 and cid in corpus_index]
    grade1 = [cid for cid, grade in all_grades.items() if grade == 1 and cid in corpus_index]
    grade2 = [cid for cid, grade in all_grades.items() if grade == 2 and cid in corpus_index]

    n_noul = n_score = 0
    for qid in picked:
        judgements = qrels_map[qid]
        positives = [cid for cid, g in judgements.items() if g > 0 and cid in corpus_index]
        non_relevant = [cid for cid, g in judgements.items() if g <= 0 and cid in corpus_index]
        neg_pool = non_relevant or grade0 or unjudged
        if not positives or not neg_pool:
            result.drop("no_positive_or_negative_pool")
            continue
        pos = positives[rng.randrange(len(positives))]
        neg = neg_pool[rng.randrange(len(neg_pool))]

        for cid, gold in ((pos, "yes"), (neg, "no")):
            result.rows.append(
                make_item(
                    tier=Tier.ESTABLISHED,
                    source=source,
                    source_record_id=f"{qid}:{cid}",
                    source_url=f"https://huggingface.co/datasets/{dataset_id}",
                    source_license=BEIR_LICENSE,
                    record_date=record_date,
                    split=split,
                    template_id=f"{source}_relevant_noul_v1",
                    skill="relevance",
                    qtype=QuestionType.NOUL,
                    state=f"Query: {query_text[qid]}\n\nPassage: {_passage_text(corpus_index[cid])}",
                    question="Is this passage relevant to the query?",
                    options=[{"key": "yes", "label": "Yes"}, {"key": "no", "label": "No"}],
                    gold=gold,
                    option_order_seed=seed,
                    meta={"query_id": qid, "corpus_id": cid, "qrel_grade": judgements.get(cid, -999)},
                )
            )
            n_noul += 1

        if not graded:
            continue
        per_query = {
            0: [cid for cid, g in judgements.items() if g <= 0 and cid in corpus_index] or grade0 or unjudged,
            1: [cid for cid, g in judgements.items() if g == 1 and cid in corpus_index] or grade1,
            2: [cid for cid, g in judgements.items() if g == 2 and cid in corpus_index] or grade2,
        }
        for grade, pool in per_query.items():
            if not pool:
                result.drop(f"score_level_{grade}_unavailable")
                continue
            cid = pool[rng.randrange(len(pool))]
            result.rows.append(
                make_choice_item(
                    source=source,
                    record_id=f"{qid}:{cid}",
                    url=f"https://huggingface.co/datasets/{dataset_id}",
                    license_=BEIR_LICENSE,
                    record_date=record_date,
                    split=split,
                    template_id=f"{source}_graded_score_v1",
                    skill="relevance",
                    state=f"Query: {query_text[qid]}\n\nPassage: {_passage_text(corpus_index[cid])}",
                    question="How relevant is this passage to the query?",
                    labels=GRADE_LEVELS,
                    gold_index=grade,
                    option_order_seed=seed,
                    qtype=QuestionType.SCORE,
                    lead="1",
                    meta={"query_id": qid, "corpus_id": cid, "qrel_grade": grade},
                )
            )
            n_score += 1

    result.notes.append(f"noul items built: {n_noul}")
    result.notes.append(f"graded score items built: {n_score}")
    if not graded:
        result.notes.append("this source's qrels are binary; no score template is built")
    return result


def _all_grades(qrels_map: dict[str, dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for judgements in qrels_map.values():
        for cid, grade in judgements.items():
            merged[cid] = max(grade, merged.get(cid, grade))
    return merged


def _subsample(rows: list[Any], cap: int, seed: int, *, salt: str) -> tuple[list[Any], int]:
    from meddecide.bench.tier1.common import subsample

    return subsample(rows, cap, seed, salt=salt)


def _caps(cfg: dict[str, Any]) -> tuple[int, int]:
    from meddecide.bench.tier1.common import caps

    return caps(cfg)


def build_trec_covid(
    *,
    queries: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    cfg: dict[str, Any],
    revision: str,
) -> LoadResult:
    """TREC-COVID: 50 topics, graded qrels, no official dev/test split.

    The 50 topics are divided by a stable hash of the topic id (80/20) so the dev/test
    boundary is deterministic and no topic appears in both; the ad-hoc nature of the
    collection means there is no official split to preserve.
    """
    dev_queries = [
        q for q in queries if split_by_record_hash(str(q["_id"]), dev_fraction=0.2, salt="trec-covid") is Split.DEV
    ]
    test_queries = [
        q for q in queries if split_by_record_hash(str(q["_id"]), dev_fraction=0.2, salt="trec-covid") is Split.TEST
    ]
    merged = LoadResult(
        source="trec_covid", dataset_id=TREC_COVID_ID, dataset_revision=revision, license=BEIR_LICENSE
    )
    for split, subset, graded in ((Split.DEV, dev_queries, True), (Split.TEST, test_queries, True)):
        part = load_beir_relevance(
            source="trec_covid",
            dataset_id=TREC_COVID_ID,
            queries=subset,
            corpus=corpus,
            qrels=qrels,
            cfg=cfg,
            revision=revision,
            split=split,
            record_date=date(2020, 3, 1),
            graded=graded,
        )
        merged.rows.extend(part.rows)
        for reason, count in part.dropped.items():
            merged.drop(reason, count)
    max_test, max_dev = _caps(cfg)
    merged.rows, counts, extra = finalize_splits(
        merged.rows, cfg=cfg, source="trec_covid", salt="trec_covid", cap_test=max_test, cap_dev=max_dev
    )
    for reason, n in counts.items():
        merged.drop(reason, n)
    merged.notes = [
        "TREC-COVID has no official dev/test split; the 50 topics are hash-split 80/20 "
        "(deterministic, disjoint) and both halves are exported",
        *extra,
        "qrels grades 0/1/2 map to score levels 1/2/3",
        f"noul items: {sum(1 for i in merged.rows if i.qtype is QuestionType.NOUL)}, "
        f"score items: {sum(1 for i in merged.rows if i.qtype is QuestionType.SCORE)}",
    ]
    merged.split_map = {"test": f"{TREC_COVID_ID}-qrels:test (hash-split topics)",
                        "dev": f"{TREC_COVID_ID}-qrels:test (hash-split topics)"}
    return merged


def build_nfcorpus(
    *,
    queries_test: list[dict[str, Any]],
    queries_train: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    qrels_test: list[dict[str, Any]],
    qrels_train: list[dict[str, Any]],
    cfg: dict[str, Any],
    revision: str,
) -> LoadResult:
    """NFCorpus: official qrels test split is tier-1 test; the official train qrels are dev.

    The official train qrels are graded 1 only, so ``score`` items are built for the test
    split (grades 1/2 plus a level-0 pool); the dev split carries ``noul`` items.
    """
    dev = load_beir_relevance(
        source="nfcorpus", dataset_id=NFCORPUS_ID, queries=queries_train, corpus=corpus,
        qrels=qrels_train, cfg=cfg, revision=revision, split=Split.DEV,
        record_date=date(2015, 1, 1), graded=False,
    )
    test = load_beir_relevance(
        source="nfcorpus", dataset_id=NFCORPUS_ID, queries=queries_test, corpus=corpus,
        qrels=qrels_test, cfg=cfg, revision=revision, split=Split.TEST,
        record_date=date(2015, 1, 1), graded=True,
    )
    dev.rows.extend(test.rows)
    for reason, count in test.dropped.items():
        dev.drop(reason, count)
    max_test, max_dev = _caps(cfg)
    dev.rows, counts, extra = finalize_splits(
        dev.rows, cfg=cfg, source="nfcorpus", salt="nfcorpus", cap_test=max_test, cap_dev=max_dev
    )
    for reason, n in counts.items():
        dev.drop(reason, n)
    dev.notes = [
        "official qrels train split used as dev (2590 queries), official qrels test split "
        "as tier-1 test (323 queries)",
        *extra,
        "official dev qrels are binary (grade 1 only); the graded score template is built "
        "on the test split where grades 1 and 2 both occur",
        f"noul items: {sum(1 for i in dev.items if i.qtype is QuestionType.NOUL)}, "
        f"score items: {sum(1 for i in dev.items if i.qtype is QuestionType.SCORE)}",
    ]
    dev.split_map = {"test": f"{NFCORPUS_ID}-qrels:test", "dev": f"{NFCORPUS_ID}-qrels:train"}
    return dev


def build_scifact(
    *,
    queries: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    cfg: dict[str, Any],
    revision: str,
) -> LoadResult:
    """SciFact relevance (binary qrels): ``noul`` items on the official test qrels.

    SciFact's *claim-verification* label (SUPPORT/CONTRADICT/NOT ENOUGH INFO) is not part of
    the BEIR release (which ships corpus, queries and binary qrels only), so it is not built
    here — deriving a verdict from the relevance qrels would be inventing gold.
    """
    result = load_beir_relevance(
        source="scifact", dataset_id=SCIFACT_ID, queries=queries, corpus=corpus, qrels=qrels,
        cfg=cfg, revision=revision, split=Split.TEST, record_date=date(2020, 5, 1), graded=False,
    )
    result.notes.append(
        "claim-verification labels are not in the BEIR release; only relevance items are built"
    )
    return result
