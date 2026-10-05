"""MedDecide — an open, calibrated medical decision model.

Loop 0 (``bench_v0``) builds the measurement foundation: MedDecide-Bench v0, the eval
harness, zero-shot baselines, and the teacher pipeline gate. No training happens here.
"""

from meddecide.utils.provenance import __version__

__all__ = ["__version__"]
