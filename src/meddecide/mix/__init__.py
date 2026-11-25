"""Training mix v2 for osler_v0 (O4): student_v1 rows plus generator rows, permissive replay, catalog candidates and
robustness augmentation, with leakage checks against every evaluation set.

Modules: ``rows`` (training-row schema and dates), ``converters`` (source records to rows, each with a kept or dropped
reason), ``augment`` (gold-preserving robustness transforms), ``leakage`` (text and record overlap). No language model
writes any text or label here.
"""
