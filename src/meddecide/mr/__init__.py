"""Medical-record audit decisions (osler_v1): page categories, clinical-indicator evidence, ICD-10-CM code support.

Labels are computed by code from structured facts (D20). Text comes from three sources: real MIMIC notes with structured
labels, real or simulated values rendered into record-page layouts, and teacher prose written from code-built facts
(D27). MIMIC text never reaches a hosted model (``meddecide.mr.teacher_prose`` refuses it).
"""
