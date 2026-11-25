#!/usr/bin/env python
"""Fetch Hub metadata (via the huggingface_hub client) for the dataset catalog (S13).

Documentation only: this script reads dataset *metadata* from the Hub (sizes, licence,
tags, split names/counts). It never downloads rows and never calls a model. The catalog
table in ``docs/benchmark/dataset_catalog.md`` is generated from the JSON this script
writes, so every id, size, licence and split count in the catalog comes from the Hub API
rather than from memory.

    uv run python scripts/bench/catalog_datasets.py --out /workspace/tmp/s13/catalog.json
    uv run python scripts/bench/catalog_datasets.py --from-json /workspace/tmp/s13/catalog.json \\
        --markdown-out /workspace/tmp/s13/table.md

``--from-json`` regenerates the markdown from a previous fetch (no network); ``--ids``
fetches an ad-hoc list instead of the curated candidate list.

The judgement columns (label provenance, convertibility, overlap risk, verdict, reason)
are curation, not measurement: they are authored in ``CANDIDATES`` below, with the
``provenance_note`` / ``overlap_note`` evidence for the claim. Sizes and licences are
always read from the Hub or the dataset card.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

# --------------------------------------------------------------------------------------
# Curation. `provenance` is one of human / structured / heuristic / llm / unknown.
# `convertible` lists the typed-decision qtypes a gold item could use (noul/choice/score),
# or is empty when no structured/human gold field exists. `overlap` names the
# MedDecide-Bench tier-1 source it touches, or "none".
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One candidate dataset: the Hub id plus the curated judgement columns."""

    dataset_id: str
    category: str
    provenance: str
    provenance_note: str
    convertible: tuple[str, ...]
    convert_field: str
    overlap: str
    overlap_note: str
    verdict: str
    reason: str
    license_override: str = ""
    license_note: str = ""
    rank: int = 99
    in_table: bool = True
    exclude_reason: str = ""


CANDIDATES: tuple[Candidate, ...] = (
    # --- medical QA / exam -------------------------------------------------------------
    Candidate(
        dataset_id="GBaker/MedQA-USMLE-4-options",
        category="exam QA",
        provenance="human",
        provenance_note="USMLE-style questions with a keyed answer; author-collected.",
        convertible=("choice",),
        convert_field="answer_idx over the 4 options",
        overlap="medqa",
        overlap_note="This *is* the tier-1 medqa source (test split).",
        verdict="reject",
        reason="Tier-1 test source; training on it is barred by D13.",
    ),
    Candidate(
        dataset_id="openlifescienceai/medmcqa",
        category="exam QA",
        provenance="human",
        provenance_note="Postgraduate entrance-exam questions with a keyed option.",
        convertible=("choice",),
        convert_field="cop over opa..opd",
        overlap="medmcqa",
        overlap_note="Tier-1 source; only the official train split is allowed (S5).",
        verdict="train-candidate",
        reason="Official train split is explicitly allowed by S5; keep test/validation out.",
        rank=3,
    ),
    Candidate(
        dataset_id="qiaojin/PubMedQA",
        category="abstract QA",
        provenance="human",
        provenance_note="pqa_labeled: expert yes/no/maybe decisions on abstract conclusions; "
        "pqa_artificial labels are heuristic.",
        convertible=("choice",),
        convert_field="final_decision (pqa_labeled only)",
        overlap="pubmedqa",
        overlap_note="Tier-1 source; pqa_labeled is the tier-1 test split.",
        verdict="reject",
        reason="Tier-1 test split; pqa_artificial is heuristic and explicitly not gold (S5).",
    ),
    Candidate(
        dataset_id="cais/mmlu",
        category="exam QA",
        provenance="human",
        provenance_note="Graduate/exam questions with a keyed answer.",
        convertible=("choice",),
        convert_field="answer over the 4 options",
        overlap="mmlu_medical",
        overlap_note="Tier-1 source; the medical subsets are test-only.",
        verdict="reject",
        reason="Tier-1 test source and MMLU has no medical train split (S5).",
    ),
    Candidate(
        dataset_id="lavita/MedQuAD",
        category="consumer health QA",
        provenance="structured",
        provenance_note="Question/answer pairs keyed to a source document and its section.",
        convertible=("choice", "noul"),
        convert_field="question type / focus section of the source document",
        overlap="medquad",
        overlap_note="Tier-1 source; S5 allows the rows not used by any tier-1 split.",
        verdict="train-candidate",
        reason="Rows not consumed by tier-1 are allowed by S5; gold is the source field.",
        license_note="no licence declared on the Hub; S5 already approved the source",
        rank=14,
    ),
    Candidate(
        dataset_id="HiTZ/MedExpQA",
        category="exam QA",
        provenance="human",
        provenance_note="Expert-written exam questions with gold answers and explanations.",
        convertible=("choice",),
        convert_field="correct option index",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="eval-candidate",
        reason="Human expert gold, but released as a test-only benchmark.",
    ),
    Candidate(
        dataset_id="TsinghuaC3I/MedXpertQA",
        category="exam QA",
        provenance="human",
        provenance_note="Expert-authored, expert-reviewed questions (text split).",
        convertible=("choice",),
        convert_field="label over the answer options",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="eval-candidate",
        reason="Human expert test set; no train split published, so evaluation only.",
    ),
    Candidate(
        dataset_id="LangAGI-Lab/medbullets_op5",
        category="exam QA",
        provenance="human",
        provenance_note="Board-style questions with a keyed answer (mirror of a question bank).",
        convertible=("choice",),
        convert_field="answer letter",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="eval-candidate",
        reason="Human gold, but a scraped question bank with no declared licence.",
        license_note="no licence declared on the Hub",
    ),
    Candidate(
        dataset_id="TIGER-Lab/MMLU-Pro",
        category="exam QA",
        provenance="human",
        provenance_note="Exam questions with a keyed answer; distractors model-filtered; has "
        "a health subcategory.",
        convertible=("choice",),
        convert_field="answer over 10 options",
        overlap="mmlu_medical",
        overlap_note="Successor benchmark to MMLU; overlaps in style, not items.",
        verdict="eval-candidate",
        reason="Human answer keys and a distinct item pool, but test/validation only.",
    ),
    Candidate(
        dataset_id="bigbio/head_qa",
        category="exam QA",
        provenance="human",
        provenance_note="Professional-examination questions with a keyed answer (Spanish).",
        convertible=("choice",),
        convert_field="answer index",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="eval-candidate",
        reason="Human gold in a non-English language; no train split published.",
    ),
    # --- clinical records / notes ------------------------------------------------------
    Candidate(
        dataset_id="zhengyun21/PMC-Patients",
        category="case reports",
        provenance="human",
        provenance_note="Patient summaries extracted from published case reports with "
        "author-written structured fields (age, sex, PMID).",
        convertible=("choice", "noul"),
        convert_field="age / sex / publication-type fields on the patient record",
        overlap="pubmed",
        overlap_note="Derived from PubMed Central case reports; restrict to pre-window PMIDs.",
        verdict="eval-candidate",
        reason="Human records with structured fields, but the non-commercial share-alike "
        "licence bars use in openly-released training data.",
    ),
    Candidate(
        dataset_id="bigbio/pmc_patients",
        category="case reports",
        provenance="human",
        provenance_note="BigBio packaging of the same case-report patient summaries.",
        convertible=("choice", "noul"),
        convert_field="structured patient fields",
        overlap="pubmed",
        overlap_note="Same underlying corpus as zhengyun21/PMC-Patients.",
        verdict="reject",
        reason="Duplicate packaging of PMC-Patients; use one id, not both.",
    ),
    Candidate(
        dataset_id="zhengyun21/PMC-Patients-ReCDS",
        category="case reports",
        provenance="llm",
        provenance_note="Recommendation/diagnosis fields generated over PMC-Patients.",
        convertible=(),
        convert_field="none - generated targets",
        overlap="pubmed",
        overlap_note="PMC-Patients records with generated targets.",
        verdict="reject",
        reason="LLM-generated labels; not gold under D13.",
    ),
    Candidate(
        dataset_id="AGBonnet/augmented-clinical-notes",
        category="clinical notes",
        provenance="llm",
        provenance_note="Synthetic notes with generated annotations.",
        convertible=(),
        convert_field="none - synthetic",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="Synthetic (LLM-generated) notes and labels.",
    ),
    Candidate(
        dataset_id="starmpcc/Asclepius-Synthetic-Clinical-Notes",
        category="clinical notes",
        provenance="llm",
        provenance_note="Generated clinical notes with generated section labels.",
        convertible=(),
        convert_field="none - synthetic",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="Synthetic (LLM-generated) notes and labels.",
    ),
    Candidate(
        dataset_id="NHSEDataScience/synthetic_clinical_notes",
        category="clinical notes",
        provenance="llm",
        provenance_note="Generated notes; the card states they are synthetic.",
        convertible=(),
        convert_field="none - synthetic",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="Synthetic (LLM-generated) notes and labels.",
    ),
    Candidate(
        dataset_id="birgermoell/icd10-clinical-notes",
        category="coding",
        provenance="unknown",
        provenance_note="Note/code pairs; the card does not document how notes or codes were "
        "produced.",
        convertible=(),
        convert_field="unverified code field",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="Provenance undocumented; cannot be verified as human or structured gold.",
    ),
    Candidate(
        dataset_id="harishnair04/mtsamples",
        category="clinical notes",
        provenance="human",
        provenance_note="Transcribed notes carrying a specialty/description field.",
        convertible=("choice",),
        convert_field="medical_specialty",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="Useful structured specialty field, but the upstream transcription corpus has "
        "no declared licence (the Hub tag is the uploader's claim only).",
        license_note="Hub tag only; upstream corpus unlicensed",
    ),
    Candidate(
        dataset_id="omi-health/medical-dialogue-to-soap-summary",
        category="clinical notes",
        provenance="llm",
        provenance_note="SOAP summaries generated from dialogue.",
        convertible=(),
        convert_field="none - generated summaries",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="Generated summaries; the dialogue side has no structured gold either.",
        license_note="no licence declared on the Hub",
    ),
    Candidate(
        dataset_id="thbndi/Mimic4Dataset",
        category="EHR",
        provenance="structured",
        provenance_note="Mirror of a credentialed EHR database.",
        convertible=(),
        convert_field="n/a",
        overlap="none",
        overlap_note="n/a",
        verdict="reject",
        reason="Credentialed data (MIMIC); barred by the data rules, and the mirror is not "
        "an authorised distribution.",
        license_note="no licence declared on the Hub",
    ),
    Candidate(
        dataset_id="Perle-ai/multimodal-ct-radiology-reports",
        category="radiology",
        provenance="unknown",
        provenance_note="Report text paired with imaging; label derivation undocumented.",
        convertible=(),
        convert_field="n/a - requires imaging",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="Text-only state cannot be built (the decision needs the image) and the "
        "licence is a bespoke data-use agreement.",
        license_note="bespoke data-use agreement named on the card",
    ),
    # --- drug / label / pharmacovigilance ----------------------------------------------
    Candidate(
        dataset_id="alexcpn/fda-faers-parquet",
        category="pharmacovigilance",
        provenance="structured",
        provenance_note="Spontaneous adverse-event reports; outcome and drug-role fields are "
        "coded regulatory fields.",
        convertible=("choice", "noul", "score"),
        convert_field="serious/outcome code, drug role, report counts",
        overlap="none",
        overlap_note="Not a tier-1 source (openFDA labels are, FAERS is not).",
        verdict="train-candidate",
        reason="Structured regulatory gold with no tier-1 overlap; use the non-terminology "
        "fields only (reaction terms are MedDRA-coded and licence-restricted).",
        rank=8,
    ),
    Candidate(
        dataset_id="jtviegas/faers_bronze_demo",
        category="pharmacovigilance",
        provenance="structured",
        provenance_note="FAERS extract; coded outcome/reaction fields.",
        convertible=("choice", "noul"),
        convert_field="outcome code",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="eval-candidate",
        reason="Structured gold but a small demo extract with no declared licence; prefer "
        "the larger FAERS id for training.",
        license_note="no licence declared on the Hub",
    ),
    Candidate(
        dataset_id="chrisvoncsefalvay/vaers-outcomes",
        category="pharmacovigilance",
        provenance="structured",
        provenance_note="Vaccine adverse-event reports with coded outcome fields and official "
        "train/val/test splits.",
        convertible=("choice", "noul"),
        convert_field="outcome / seriousness field",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="train-candidate",
        reason="Structured regulatory gold with an official train split; restrict to the "
        "coded outcome fields (symptom terms are MedDRA-coded).",
        rank=5,
    ),
    Candidate(
        dataset_id="sixuexing/FAERS-NLP",
        category="pharmacovigilance",
        provenance="human",
        provenance_note="Adverse-event reports with annotated entity/relation spans.",
        convertible=("noul", "choice"),
        convert_field="annotated reaction/relation label",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="eval-candidate",
        reason="Human annotation, but small, span-shaped and non-commercial licensed.",
    ),
    Candidate(
        dataset_id="Oduwo/drug_label_approved_openfda",
        category="drug labels",
        provenance="structured",
        provenance_note="Label sections from the openFDA label distribution.",
        convertible=("noul", "choice"),
        convert_field="label section presence (e.g. boxed warning)",
        overlap="openfda",
        overlap_note="Same openFDA label records as the fresh-tier openfda source.",
        verdict="reject",
        reason="Same records as the tier-1 fresh openfda source; overlap is record-level.",
    ),
    Candidate(
        dataset_id="um-ids/dailymed-annotations",
        category="drug labels",
        provenance="structured",
        provenance_note="Annotated sections of structured product labels.",
        convertible=("noul", "choice"),
        convert_field="section type / presence",
        overlap="openfda",
        overlap_note="openFDA label content is derived from the same label distribution.",
        verdict="reject",
        reason="Label records overlap the fresh openfda source; record-level leakage risk.",
    ),
    Candidate(
        dataset_id="just-dna-seq/clinpgx_drug_labels",
        category="drug labels",
        provenance="structured",
        provenance_note="Regulator-derived pharmacogenomic dosing statements with an ordered "
        "testing level and a regulator field.",
        convertible=("score", "choice"),
        convert_field="testing_level (ordered) / regulator",
        overlap="openfda",
        overlap_note="Some statements come from the same FDA label documents as the fresh "
        "openfda source (different label sections); restrict to non-FDA regulators or "
        "verify no overlap.",
        verdict="train-candidate",
        reason="Ordered structured gold from regulator labels; share-alike licence and a "
        "record-level check against the fresh openfda source are required.",
        license_override="cc-by-sa-4.0",
        license_note="LICENSE.txt in the repo; card field empty",
        rank=9,
    ),
    Candidate(
        dataset_id="bigbio/ddi_corpus",
        category="drug interactions",
        provenance="human",
        provenance_note="Human-annotated drug-drug interaction sentences with an "
        "interaction type.",
        convertible=("choice", "noul"),
        convert_field="interaction type annotation",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="eval-candidate",
        reason="Clean human relation gold, but the non-commercial licence bars use in "
        "openly-released training data.",
    ),
    Candidate(
        dataset_id="ade-benchmark-corpus/ade_corpus_v2",
        category="adverse events",
        provenance="human",
        provenance_note="Human-annotated adverse-drug-event mentions and relations.",
        convertible=("noul", "choice"),
        convert_field="ADE relation annotation",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="eval-candidate",
        reason="Human-annotated drug-event relation, but the Hub declares no licence.",
        license_note="no licence declared on the Hub",
    ),
    # --- biomedical NER / relation ------------------------------------------------------
    Candidate(
        dataset_id="bigbio/bc5cdr",
        category="NER/relation",
        provenance="human",
        provenance_note="Human-annotated chemical and disease mentions with a relation.",
        convertible=("noul", "choice"),
        convert_field="chemical-induced-disease relation annotation",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="train-candidate",
        reason="Human relation gold under a public-domain mark; a yes/no item can hide the "
        "relation sentence.",
        rank=11,
    ),
    Candidate(
        dataset_id="bigbio/ncbi_disease",
        category="NER",
        provenance="human",
        provenance_note="Human-annotated disease mentions in abstracts.",
        convertible=("noul",),
        convert_field="disease-mention annotation",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="eval-candidate",
        reason="Human gold but only mention spans; a typed decision is a stretch.",
    ),
    Candidate(
        dataset_id="bigbio/chemprot",
        category="NER/relation",
        provenance="human",
        provenance_note="Human-annotated chemical-protein relations with a CPR class.",
        convertible=("choice",),
        convert_field="CPR relation class",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="train-candidate",
        reason="Human multi-class relation gold under a public-domain mark; maps directly to "
        "a choice item.",
        rank=12,
    ),
    Candidate(
        dataset_id="bigbio/gad",
        category="relation",
        provenance="human",
        provenance_note="Human-annotated gene-disease association sentences.",
        convertible=("noul",),
        convert_field="association annotation",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="train-candidate",
        reason="Human yes/no relation gold; the abstract text can be the state.",
        rank=13,
    ),
    Candidate(
        dataset_id="bigbio/biored",
        category="NER/relation",
        provenance="human",
        provenance_note="Human-annotated biomedical relations with novelty flags.",
        convertible=("choice", "noul"),
        convert_field="relation type / novelty annotation",
        overlap="pubmed",
        overlap_note="Abstracts come from PubMed; pre-window PMIDs only.",
        verdict="eval-candidate",
        reason="Human relation gold, but the card declares no licence; also needs the "
        "pre-window PMID restriction.",
    ),
    Candidate(
        dataset_id="spyysalo/bc2gm_corpus",
        category="NER",
        provenance="human",
        provenance_note="Human-annotated gene mentions.",
        convertible=("noul",),
        convert_field="gene-mention annotation",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="Mention spans only, no licence declared, and no decision-shaped gold without "
        "inventing one.",
    ),
    Candidate(
        dataset_id="bigbio/jnlpba",
        category="NER",
        provenance="human",
        provenance_note="Human-annotated cell-type/protein/DNA mentions.",
        convertible=("noul",),
        convert_field="entity-type annotation",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="Span typing is not a medical decision; low value for this schema.",
    ),
    # --- evidence / verification / relevance --------------------------------------------
    Candidate(
        dataset_id="bigbio/evidence_inference",
        category="evidence",
        provenance="human",
        provenance_note="Human-annotated intervention/comparator/outcome prompts and the "
        "direction of the result.",
        convertible=("choice", "noul", "score"),
        convert_field="significance/direction annotation over a PICO prompt",
        overlap="pubmed",
        overlap_note="PubMed abstracts; pre-window PMIDs only.",
        verdict="train-candidate",
        reason="Human structured gold (PICO + direction) that maps to choice and score; "
        "restrict to pre-window PMIDs.",
        rank=1,
    ),
    Candidate(
        dataset_id="bigbio/mednli",
        category="clinical NLI",
        provenance="human",
        provenance_note="Clinician-written entailment labels over clinical sentences.",
        convertible=("choice", "noul"),
        convert_field="entailment/contradiction/neutral label",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="The card declares the PhysioNet 1.5 licence (credentialed data), which the "
        "data rules bar.",
        license_note="card declares PHYSIONET_LICENSE_1p5",
    ),
    Candidate(
        dataset_id="dwadden/healthver_entailment",
        category="claim verification",
        provenance="human",
        provenance_note="Human entailment labels for health claims against abstracts.",
        convertible=("choice", "noul"),
        convert_field="entailment label",
        overlap="pubmed",
        overlap_note="PubMed abstracts; pre-window PMIDs only.",
        verdict="eval-candidate",
        reason="Human claim-verification gold that matches the consistency template family, "
        "but the non-commercial licence bars training use.",
    ),
    Candidate(
        dataset_id="bigbio/pubhealth",
        category="claim verification",
        provenance="human",
        provenance_note="Human fact-check verdicts on health claims with cited evidence.",
        convertible=("choice", "noul"),
        convert_field="claim veracity label",
        overlap="none",
        overlap_note="Not a tier-1 source; evidence passages are news/policy articles.",
        verdict="train-candidate",
        reason="Human-verified claim gold under a permissive licence; needs the pre-window "
        "PMID restriction.",
        rank=7,
    ),
    Candidate(
        dataset_id="bigbio/bioasq_task_b",
        category="abstract QA",
        provenance="human",
        provenance_note="Expert yes/no answer with the supporting abstract snippets.",
        convertible=("choice", "noul"),
        convert_field="exact answer (yes/no) with evidence",
        overlap="pubmed",
        overlap_note="PubMed abstracts; pre-window PMIDs only.",
        verdict="eval-candidate",
        reason="Human expert yes/no gold, but the Hub declares only 'other' licensing and "
        "the release is test-shaped.",
    ),
    Candidate(
        dataset_id="bigbio/mediqa_qa",
        category="abstract QA",
        provenance="human",
        provenance_note="Clinician-written questions with relevance judgements.",
        convertible=("choice", "noul"),
        convert_field="relevance judgement",
        overlap="pubmed",
        overlap_note="PubMed abstracts; pre-window PMIDs only.",
        verdict="eval-candidate",
        reason="Human gold, but the Hub declares no licence.",
        license_note="no licence declared on the Hub",
    ),
    Candidate(
        dataset_id="BeIR/nfcorpus",
        category="relevance",
        provenance="human",
        provenance_note="Human qrels for query-document relevance.",
        convertible=("choice", "noul"),
        convert_field="qrels relevance grade",
        overlap="nfcorpus",
        overlap_note="Tier-1 source; S5 allows the official train qrels pool.",
        verdict="train-candidate",
        reason="Official train qrels are allowed by S5 and are human-judged.",
        rank=6,
    ),
    Candidate(
        dataset_id="BeIR/scifact",
        category="claim verification",
        provenance="human",
        provenance_note="Expert-written claims with evidence and support labels.",
        convertible=("choice", "noul"),
        convert_field="claim label (SUPPORT/REFUTE) and cited evidence",
        overlap="scifact",
        overlap_note="Tier-1 source; S5 allows the official train claims.",
        verdict="train-candidate",
        reason="Official train split is allowed by S5 and is human-expert gold.",
        rank=4,
    ),
    Candidate(
        dataset_id="BeIR/trec-covid",
        category="relevance",
        provenance="human",
        provenance_note="Human qrels for COVID-19 literature search.",
        convertible=("choice",),
        convert_field="qrels relevance grade",
        overlap="trec_covid",
        overlap_note="Tier-1 source; the release has no train split.",
        verdict="reject",
        reason="Tier-1 test source with no official train split.",
    ),
    Candidate(
        dataset_id="MedRAG/pubmed",
        category="corpus",
        provenance="structured",
        provenance_note="PubMed abstract chunks with ids and metadata; no task labels.",
        convertible=(),
        convert_field="none - retrieval corpus, no gold label",
        overlap="pubmed",
        overlap_note="Same PubMed source as the fresh-tier pubmed builder.",
        verdict="reject",
        reason="Corpus without labels and no declared licence; also overlaps the tier-1 "
        "pubmed source.",
        license_note="no licence declared on the Hub",
    ),
    Candidate(
        dataset_id="MedRAG/statpearls",
        category="corpus",
        provenance="human",
        provenance_note="Human-written reference chapters; no task labels.",
        convertible=(),
        convert_field="none - no gold label",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="Reference corpus with no gold field and no declared licence; retrieval is "
        "out of scope for this loop.",
        license_note="no licence declared on the Hub",
    ),
    # --- trials / study design ----------------------------------------------------------
    Candidate(
        dataset_id="araag2/EBM_NLP",
        category="study design",
        provenance="human",
        provenance_note="Crowd-annotated PICO spans over RCT abstracts with an "
        "expert-validated subset (human), with train/test splits.",
        convertible=("choice", "noul"),
        convert_field="PICO element annotation",
        overlap="pubmed",
        overlap_note="PubMed abstracts; pre-window PMIDs only.",
        verdict="train-candidate",
        reason="Human PICO gold with an official train split; share-alike licence and the "
        "pre-window PMID restriction apply.",
        rank=10,
    ),
    Candidate(
        dataset_id="pietrolesci/pubmed-200k-rct",
        category="study design",
        provenance="human",
        provenance_note="Human sentence-level labels for RCT abstract sections.",
        convertible=("choice",),
        convert_field="sentence role label",
        overlap="pubmed",
        overlap_note="PubMed abstracts; pre-window PMIDs only.",
        verdict="eval-candidate",
        reason="Large human section-label set, but the Hub declares no licence; the smaller "
        "RCT mirror has the same problem.",
        license_note="no licence declared on the Hub",
    ),
    Candidate(
        dataset_id="armanc/pubmed-rct20k",
        category="study design",
        provenance="human",
        provenance_note="Human sentence-level labels for RCT abstract sections.",
        convertible=("choice",),
        convert_field="sentence role label",
        overlap="pubmed",
        overlap_note="PubMed abstracts; pre-window PMIDs only.",
        verdict="reject",
        reason="Duplicate packaging of the same human labels as pietrolesci/pubmed-200k-rct; "
        "use one id.",
    ),
    Candidate(
        dataset_id="bigbio/chia",
        category="trial eligibility",
        provenance="human",
        provenance_note="Annotated eligibility criteria from trial records (human "
        "annotators).",
        convertible=("choice", "noul"),
        convert_field="criterion scope annotation",
        overlap="clinicaltrials",
        overlap_note="Trial text may come from the same registry as the fresh clinicaltrials "
        "source; record-level check required.",
        verdict="eval-candidate",
        reason="Human criterion gold, but the registry overlap with the tier-1 fresh source "
        "needs a record-level check before any training use.",
    ),
    Candidate(
        dataset_id="Parexel/clinical-trials-protocols",
        category="trial protocols",
        provenance="unknown",
        provenance_note="Protocol documents; the card does not state how they were assembled.",
        convertible=("choice", "noul"),
        convert_field="protocol section fields (if verifiable)",
        overlap="clinicaltrials",
        overlap_note="Trial protocols may correspond to registered records in the fresh-tier "
        "clinicaltrials source.",
        verdict="reject",
        reason="Provenance and licence unclear, and record-level overlap with the tier-1 "
        "clinicaltrials source cannot be excluded.",
        license_note="no licence declared on the Hub",
    ),
    Candidate(
        dataset_id="chemNLP/clinical-trials-v2",
        category="trial records",
        provenance="structured",
        provenance_note="Registry fields (phase, status, eligibility) from trial records.",
        convertible=("choice", "noul"),
        convert_field="phase / status / eligibility fields",
        overlap="clinicaltrials",
        overlap_note="Same registry as the fresh-tier clinicaltrials source.",
        verdict="reject",
        reason="Record-level overlap with the tier-1 clinicaltrials source; training on it "
        "would leak test records.",
        license_note="no licence declared on the Hub",
    ),
    Candidate(
        dataset_id="domenicrosati/clinical_trial_texts",
        category="trial records",
        provenance="unknown",
        provenance_note="Trial text fields; the card does not state label provenance.",
        convertible=(),
        convert_field="unverified",
        overlap="clinicaltrials",
        overlap_note="Registry-derived text.",
        verdict="reject",
        reason="Undocumented provenance, no licence, and registry overlap.",
        license_note="no licence declared on the Hub",
    ),
    # --- structured clinical / calculation ----------------------------------------------
    Candidate(
        dataset_id="ncbi/MedCalc-Bench-v1.2",
        category="clinical calculation",
        provenance="structured",
        provenance_note="Calculator inputs and outputs computed by a rule, with official "
        "train and test files.",
        convertible=("score", "choice"),
        convert_field="calculator output value band",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="train-candidate",
        reason="Deterministic structured gold with an official train split; a score item can "
        "hide the computed result (share-alike licence).",
        rank=2,
    ),
    Candidate(
        dataset_id="ncbi/MedCalc-Bench",
        category="clinical calculation",
        provenance="structured",
        provenance_note="Earlier release of the same calculator-derived gold.",
        convertible=("score", "choice"),
        convert_field="calculator output value band",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="Superseded release of MedCalc-Bench-v1.2; use one id.",
    ),
    Candidate(
        dataset_id="nsk7153/MedCalc-Bench-Verified",
        category="clinical calculation",
        provenance="human",
        provenance_note="Human-verified subset of the calculator outputs.",
        convertible=("score", "choice"),
        convert_field="verified calculator output",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="eval-candidate",
        reason="Human-verified and therefore the cleanest slice; keep it as evaluation, not "
        "training.",
    ),
    Candidate(
        dataset_id="songlab/clinvar",
        category="variant interpretation",
        provenance="structured",
        provenance_note="ClinVar-derived variant table; the card documents no fields, but "
        "the upstream release carries a clinical-significance field with review status.",
        convertible=("choice", "noul"),
        convert_field="clinical significance classification",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="eval-candidate",
        reason="Structured expert-curated gold, but the Hub repo ships a test split only and "
        "documents no fields; verify against the upstream release first.",
    ),
    Candidate(
        dataset_id="HuggingFaceBio/clinvar-vep",
        category="variant interpretation",
        provenance="structured",
        provenance_note="Variant effect predictions joined to ClinVar significance.",
        convertible=("choice",),
        convert_field="clinical significance",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="Model-derived features and unclear packaging of the upstream significance "
        "field; use the ClinVar release directly.",
    ),
    Candidate(
        dataset_id="HHS-Official/behavioral-risk-factor-surveillance-system-brfss-p",
        category="survey",
        provenance="structured",
        provenance_note="Self-reported survey responses with coded condition fields.",
        convertible=("noul", "choice", "score"),
        convert_field="reported condition/behaviour field",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="eval-candidate",
        reason="Structured population-survey gold, but self-reported and far from the "
        "record/claim decision shapes; calibration-style checks at most.",
        license_override="ODbL-1.0",
        license_note="card text",
    ),
    # --- calibration / uncertainty ------------------------------------------------------
    Candidate(
        dataset_id="facebook/AbstentionBench",
        category="abstention",
        provenance="llm",
        provenance_note="Answerability labels derived from model behaviour and dataset "
        "construction rules.",
        convertible=(),
        convert_field="none - derived answerability",
        overlap="none",
        overlap_note="Aggregates several public sets, some medical.",
        verdict="reject",
        reason="Labels are constructed/derived rather than human gold, and it aggregates "
        "tier-1 test sets.",
    ),
    # --- considered but deliberately not listed -----------------------------------------
    Candidate(
        dataset_id="bigbio/med_qa",
        category="exam QA",
        provenance="human",
        provenance_note="BigBio packaging of the tier-1 medqa questions.",
        convertible=(),
        convert_field="n/a",
        overlap="medqa",
        overlap_note="Same items as GBaker/MedQA-USMLE-4-options.",
        verdict="reject",
        reason="",
        in_table=False,
        exclude_reason="mirror of a tier-1 test source; adds no new information",
    ),
    Candidate(
        dataset_id="bigbio/pubmed_qa",
        category="abstract QA",
        provenance="human",
        provenance_note="BigBio packaging of PubMedQA.",
        convertible=(),
        convert_field="n/a",
        overlap="pubmedqa",
        overlap_note="Same items as qiaojin/PubMedQA.",
        verdict="reject",
        reason="",
        in_table=False,
        exclude_reason="mirror of a tier-1 test source",
    ),
    Candidate(
        dataset_id="bigbio/scifact",
        category="claim verification",
        provenance="human",
        provenance_note="BigBio packaging of SciFact.",
        convertible=(),
        convert_field="n/a",
        overlap="scifact",
        overlap_note="Same items as BeIR/scifact.",
        verdict="reject",
        reason="",
        in_table=False,
        exclude_reason="mirror of a tier-1 test source",
    ),
    Candidate(
        dataset_id="keivalya/MedQuad-MedicalQnADataset",
        category="consumer health QA",
        provenance="structured",
        provenance_note="Derived mirror of MedQuAD.",
        convertible=(),
        convert_field="n/a",
        overlap="medquad",
        overlap_note="Same rows as lavita/MedQuAD.",
        verdict="reject",
        reason="",
        in_table=False,
        exclude_reason="derived mirror of a listed id; adds no new information",
    ),
    Candidate(
        dataset_id="MedRAG/textbooks",
        category="corpus",
        provenance="human",
        provenance_note="Reference-textbook corpus with no task labels.",
        convertible=(),
        convert_field="n/a",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="",
        in_table=False,
        exclude_reason="same family as MedRAG/statpearls (no labels); one corpus example is "
        "enough",
    ),
    Candidate(
        dataset_id="MedRAG/wikipedia",
        category="corpus",
        provenance="human",
        provenance_note="Encyclopaedia corpus with no task labels.",
        convertible=(),
        convert_field="n/a",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="",
        in_table=False,
        exclude_reason="corpus with no labels; not a decision dataset",
    ),
    Candidate(
        dataset_id="lavita/ChatDoctor-HealthCareMagic-100k",
        category="consumer health chat",
        provenance="human",
        provenance_note="Consumer questions with clinician-written free-text replies.",
        convertible=(),
        convert_field="none - free-text replies, no structured field",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="",
        in_table=False,
        exclude_reason="free-text answers with no structured or discrete gold field",
    ),
    Candidate(
        dataset_id="PrashantRGore/synthetic-faers-1m-v3",
        category="pharmacovigilance",
        provenance="llm",
        provenance_note="Fully synthetic case reports with injected signals.",
        convertible=(),
        convert_field="n/a",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="",
        in_table=False,
        exclude_reason="synthetic reports with injected (generated) signals; no source gold",
    ),
    Candidate(
        dataset_id="RubyIntelligence/faers-cardiovascular-drug-safety",
        category="pharmacovigilance",
        provenance="structured",
        provenance_note="Derived organ-system slice of FAERS.",
        convertible=(),
        convert_field="n/a",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="",
        in_table=False,
        exclude_reason="derived slice of the FAERS release already listed",
    ),
    Candidate(
        dataset_id="NickyNicky/medical_mtsamples",
        category="clinical notes",
        provenance="human",
        provenance_note="Mirror of the transcription corpus.",
        convertible=(),
        convert_field="n/a",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="",
        in_table=False,
        exclude_reason="mirror of a listed id with the same licence problem",
    ),
    Candidate(
        dataset_id="SetFit/ade_corpus_v2_classification",
        category="adverse events",
        provenance="human",
        provenance_note="Derived classification mirror of the ADE corpus.",
        convertible=(),
        convert_field="n/a",
        overlap="none",
        overlap_note="Not a tier-1 source.",
        verdict="reject",
        reason="",
        in_table=False,
        exclude_reason="derived mirror of the ADE corpus already listed",
    ),
    Candidate(
        dataset_id="dmacres/mimiciii-hospitalcourse-meta",
        category="EHR",
        provenance="structured",
        provenance_note="Mirror of a credentialed EHR database.",
        convertible=(),
        convert_field="n/a",
        overlap="none",
        overlap_note="n/a",
        verdict="reject",
        reason="",
        in_table=False,
        exclude_reason="credentialed-data mirror; barred by the data rules",
    ),
)

SEARCH_QUERIES: tuple[str, ...] = (
    "medical question answering",
    "clinical notes",
    "pharmacovigilance",
    "clinical trial",
    "MedNLI",
    "clinical NLI",
    "evidence inference",
    "pubmed rct",
    "MedCalc",
    "ClinVar",
    "CIViC",
    "FAERS",
    "ICD coding clinical",
    "drug label",
    "adverse drug event",
    "medical claim verification",
    "MedlinePlus",
    "PMC-Patients",
    "VAERS",
    "DailyMed",
    "medical dialogue",
    "biomedical relation extraction",
    "radiology report",
    "MedExpQA",
    "MedXpertQA",
    "Medbullets",
    "medication question answering",
    "medical exam questions",
    "mtsamples",
    "EBM NLP",
    "HealthVer",
    "MedRAG",
    "medical uncertainty",
    "abstention",
    "SIDER side effect",
    "biored",
    "MIMIC",
    "ehrshot",
    "BRFSS",
    "drug-drug interaction",
    "clinical outcome prediction",
    "PubMed abstract",
    "PICO",
    "GRADE evidence quality",
    "confidence calibration",
    "clinical text classification",
    "ade corpus",
    "bc2gm",
    "bigbio",
)


# --------------------------------------------------------------------------------------
# Hub metadata
# --------------------------------------------------------------------------------------


@dataclass
class HubMeta:
    """What the Hub reports for one dataset id. Empty/zero values mean 'not reported'."""

    dataset_id: str
    ok: bool
    error: str = ""
    sha: str = ""
    last_modified: str = ""
    gated: bool = False
    downloads: int = 0
    likes: int = 0
    license: str = "UNKNOWN"
    license_source: str = ""
    size_categories: str = ""
    download_size: int = 0
    dataset_size: int = 0
    num_examples: int = 0
    size_source: str = ""
    configs: list[str] = field(default_factory=list)
    splits: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _normalise(text: str) -> str:
    """Lowercase alphanumerics only, so CC_BY_4p0 and cc-by-4.0 compare equal."""
    text = re.sub(r"(\d)p(\d)", r"\1.\2", text)  # bigbio writes 4p0 for 4.0
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _license_from(card: dict[str, Any], tags: list[str]) -> tuple[str, str]:
    """Return (licence, where it came from). Never guesses: missing means UNKNOWN."""
    card_license = _as_list(card.get("license"))
    if card_license:
        label = ", ".join(card_license)
        shortname = (
            card.get("bigbio_license_shortname")
            or card.get("bigbio_license_short_name")
            or card.get("license_name")
        )
        if shortname and _normalise(str(shortname)) not in _normalise(label):
            label = f"{label} ({shortname})"  # adds information (e.g. a named licence)
        if _normalise(label) == "unknown":
            return "UNKNOWN", "card_data.license"
        if _normalise(label) == "other":
            return "UNKNOWN (card says 'other' with no name)", "card_data.license"
        return label, "card_data.license"
    tag_licenses = [t.split(":", 1)[1] for t in tags if t.startswith("license:")]
    if tag_licenses:
        return ", ".join(sorted(set(tag_licenses))), "tags"
    return "UNKNOWN", "none"


def _dataset_info_entries(card: dict[str, Any]) -> list[dict[str, Any]]:
    """``dataset_info`` is a dict for single-config cards and a list for multi-config ones."""
    raw = card.get("dataset_info")
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    return []


def fetch_meta(api: HfApi, dataset_id: str) -> HubMeta:
    """Read one dataset's metadata from the Hub. No rows are downloaded."""
    meta = HubMeta(dataset_id=dataset_id, ok=False)
    try:
        info = api.dataset_info(dataset_id, files_metadata=True)
    except Exception as exc:
        meta.error = f"{type(exc).__name__}: {exc}"
        return meta
    card: dict[str, Any] = {}
    if info.card_data is not None:
        try:
            card = info.card_data.to_dict()
        except Exception:
            card = {}
    meta.ok = True
    meta.sha = info.sha or ""
    meta.last_modified = str(info.last_modified or "")
    meta.gated = bool(info.gated)
    meta.downloads = int(info.downloads or 0)
    meta.likes = int(info.likes or 0)
    meta.license, meta.license_source = _license_from(card, list(info.tags or []))
    meta.size_categories = ", ".join(_as_list(card.get("size_categories")))
    meta.tags = [t for t in (info.tags or []) if not t.startswith("license:")]
    for entry in _dataset_info_entries(card):
        if entry.get("config_name"):
            meta.configs.append(str(entry["config_name"]))
        meta.download_size += int(entry.get("download_size") or 0)
        meta.dataset_size += int(entry.get("dataset_size") or 0)
        for split in entry.get("splits") or []:
            if isinstance(split, dict) and split.get("name"):
                name = str(split["name"])
                if name not in meta.splits:
                    meta.splits.append(name)
                meta.num_examples += int(split.get("num_examples") or 0)
    if meta.download_size:
        meta.size_source = "card dataset_info"
    else:
        # No dataset_info on the card: the Hub's file listing is the only size it reports.
        total = sum(int(s.size or 0) for s in (info.siblings or []))
        if total:
            meta.download_size = total
            meta.size_source = "repo file sizes"
    meta.configs = sorted(set(meta.configs))
    return meta


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def human_bytes(n: int) -> str:
    """Compact byte string; 0 means 'not reported'."""
    if n <= 0:
        return "UNKNOWN"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} TB"


def size_cell(meta: HubMeta) -> str:
    """Download size / row bytes / examples, from the Hub card or file listing only."""
    parts = [f"{human_bytes(meta.download_size)} dl"]
    if meta.dataset_size:
        parts.append(f"{human_bytes(meta.dataset_size)} rows")
    if meta.num_examples:
        parts.append(f"{meta.num_examples:,} ex")
    if not meta.dataset_size and not meta.num_examples:
        parts.append(meta.size_categories or "rows UNKNOWN")
    if meta.size_source == "repo file sizes":
        parts.append("repo files")
    return " / ".join(parts)


def license_cell(cand: Candidate, meta: HubMeta) -> str:
    """Hub licence; a curated override is used only when the card field is empty/other."""
    if cand.license_override:
        note = f" ({cand.license_note})" if cand.license_note else ""
        return f"{cand.license_override}{note}"
    if cand.license_note:
        return f"{meta.license} ({cand.license_note})"
    return meta.license


def convert_cell(cand: Candidate) -> str:
    if not cand.convertible:
        return "no"
    return ", ".join(f"`{q}`" for q in cand.convertible)


def verdict_cell(cand: Candidate) -> str:
    return {"train-candidate": "**train-candidate**"}.get(cand.verdict, cand.verdict)


VERDICT_ORDER = {"train-candidate": 0, "eval-candidate": 1, "reject": 2}


def ordered(rows: list[tuple[Candidate, HubMeta]]) -> list[tuple[Candidate, HubMeta]]:
    """Strongest train-candidates first, then eval candidates, then rejects."""
    return sorted(rows, key=lambda r: (VERDICT_ORDER.get(r[0].verdict, 3), r[0].rank))


def markdown_table(rows: list[tuple[Candidate, HubMeta]]) -> str:
    """The catalog table: every cell except the judgement columns comes from the Hub."""
    rows = ordered(rows)
    head = (
        "| # | dataset id | size (Hub) | labels | licence | convertible | overlap risk | "
        "verdict | reason |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for i, (cand, meta) in enumerate(rows, start=1):
        note = " ".join(f"{cand.provenance_note} {cand.overlap_note}".split())
        lines.append(
            f"| {i} | `{cand.dataset_id}` | {size_cell(meta)} | {cand.provenance} | "
            f"{license_cell(cand, meta)} | {convert_cell(cand)} | {note.replace('|', '/')} | "
            f"{verdict_cell(cand)} | {cand.reason.replace('|', '/')} |"
        )
    return head + "\n".join(lines) + "\n"


def counts_block(all_rows: list[tuple[Candidate, HubMeta]]) -> dict[str, Any]:
    table = [(c, m) for c, m in all_rows if c.in_table]
    excluded = [(c, m) for c, m in all_rows if not c.in_table]
    verdicts: dict[str, int] = {}
    provenance: dict[str, int] = {}
    licences: dict[str, int] = {}
    for cand, meta in table:
        verdicts[cand.verdict] = verdicts.get(cand.verdict, 0) + 1
        provenance[cand.provenance] = provenance.get(cand.provenance, 0) + 1
        licences[meta.license] = licences.get(meta.license, 0) + 1
    return {
        "n_search_queries": len(SEARCH_QUERIES),
        "n_fetched": len(all_rows),
        "n_table_rows": len(table),
        "n_excluded": len(excluded),
        "verdicts": verdicts,
        "provenance": provenance,
        "licences": licences,
        # "other (NAME)" names a licence; only a bare unknown/other is genuinely undeclared.
        "n_unknown_license": sum(
            n
            for lic, n in licences.items()
            if "(" not in lic and lic.lower() in {"unknown", "other"}
        ),
        "fetch_failures": [c.dataset_id for c, m in all_rows if not m.ok],
        "excluded": [
            {"dataset_id": c.dataset_id, "reason": c.exclude_reason} for c, _ in excluded
        ],
    }


def build_rows(
    candidates: tuple[Candidate, ...], metas: dict[str, HubMeta]
) -> list[tuple[Candidate, HubMeta]]:
    return [(c, metas.get(c.dataset_id, HubMeta(c.dataset_id, ok=False))) for c in candidates]


def load_metas(path: Path) -> dict[str, HubMeta]:
    payload = json.loads(path.read_text())
    metas: dict[str, HubMeta] = {}
    for entry in payload.get("datasets", []):
        meta = entry.get("hub", entry)
        metas[meta["dataset_id"]] = HubMeta(**meta)
    return metas


def ad_hoc_candidate(dataset_id: str) -> Candidate:
    return Candidate(dataset_id, "ad-hoc", "unknown", "", (), "", "none", "", "reject", "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write the fetched metadata JSON here")
    parser.add_argument("--markdown-out", type=Path, help="write the catalog table here")
    parser.add_argument(
        "--from-json", type=Path, help="reuse metadata from a previous --out run (no network)"
    )
    parser.add_argument("--ids", nargs="*", help="fetch these ids instead of CANDIDATES")
    parser.add_argument("--limit", type=int, default=0, help="debug: fetch only N candidates")
    args = parser.parse_args()

    candidates = CANDIDATES
    if args.ids:
        by_id = {c.dataset_id: c for c in CANDIDATES}
        candidates = tuple(by_id.get(i, ad_hoc_candidate(i)) for i in args.ids)
    if args.limit:
        candidates = candidates[: args.limit]

    if args.from_json:
        metas = load_metas(args.from_json)
    else:
        api = HfApi()
        metas = {}
        for i, cand in enumerate(candidates, start=1):
            metas[cand.dataset_id] = fetch_meta(api, cand.dataset_id)
            status = "ok" if metas[cand.dataset_id].ok else "FAIL"
            print(f"[{i}/{len(candidates)}] {cand.dataset_id} {status}", flush=True)

    rows = build_rows(candidates, metas)
    counts = counts_block(rows)
    print(json.dumps(counts, indent=2, sort_keys=True))

    if args.out:
        payload = {
            "source": "huggingface_hub.HfApi.dataset_info(files_metadata=True)",
            "search_queries": list(SEARCH_QUERIES),
            "datasets": [{"curation": asdict(c), "hub": asdict(m)} for c, m in rows],
            "counts": counts,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown_table([r for r in rows if r[0].in_table]))
        print(f"wrote {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
