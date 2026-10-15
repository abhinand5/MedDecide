"""Gate O2 (PROGRAM D31) for Osler-4B: the verdict rules, as pure functions.

PASS iff every primary check passes:
  (i)   the MR audit suite, against both baselines: the accuracy lower bound is above 0 and the Brier upper bound is below 0
        (``Comparison.passes``, the same rule the O1 gate uses for its scopes);
  (ii)  the external clinical panel, against both baselines: non-inferiority, the accuracy lower bound is above -0.02
        (superiority is reported separately and never enters the verdict);
  (iii) the knowledge guard (D24), Osler-4B against zero-shot Qwen3.5-4B: the accuracy lower bound is above -0.02.
A missing primary comparison makes the verdict NOT MEASURED. It is never counted as a failure.
Additional rows (held-out, d1-3B, osler_v0 models) never enter a verdict.
"""

from __future__ import annotations

from collections.abc import Mapping

from meddecide.eval.gate import KNOWLEDGE_GUARD_FLOOR, Comparison, knowledge_guard
from meddecide.eval.paired import Difference

PANEL_NON_INFERIORITY_FLOOR = -0.02  # D31 (ii)


def panel_non_inferior(accuracy: Difference) -> bool:
    """D31 (ii): the accuracy lower bound, Osler minus the baseline, is above -0.02."""
    return accuracy.lo > PANEL_NON_INFERIORITY_FLOOR


def panel_superior(accuracy: Difference) -> bool:
    """Additional, never in the verdict (D31 (ii): superiority is reported separately): the accuracy lower bound is above zero."""
    return accuracy.lo > 0.0


def o2_verdict(
    mr_suite: Mapping[str, Comparison | None],
    panel: Mapping[str, Comparison | None],
    guard: Difference | None,
) -> tuple[str, str]:
    """The Gate O2 verdict for one model. ``mr_suite`` and ``panel`` map each baseline's name to its comparison, and
    ``guard`` is the knowledge difference against zero-shot Qwen3.5-4B (None when it could not be measured).

    Returns (verdict, reason); the verdict is PASS, FAIL or NOT MEASURED.
    """
    missing = [f"MR suite vs {name}" for name, comp in mr_suite.items() if comp is None]
    missing += [f"external panel vs {name}" for name, comp in panel.items() if comp is None]
    if guard is None:
        missing.append("knowledge guard")
    if missing:
        return "NOT MEASURED", "not measured: " + "; ".join(missing)
    failed = [f"MR suite vs {name}" for name, comp in mr_suite.items() if not comp.passes]  # type: ignore[union-attr]
    failed += [f"external panel vs {name} (accuracy lower bound must exceed {PANEL_NON_INFERIORITY_FLOOR:+.2f})"
               for name, comp in panel.items() if not panel_non_inferior(comp.accuracy)]  # type: ignore[union-attr]
    if not knowledge_guard(guard):
        failed.append(f"knowledge guard (accuracy lower bound must exceed {KNOWLEDGE_GUARD_FLOOR:+.2f})")
    if failed:
        return "FAIL", "failed: " + "; ".join(failed)
    return "PASS", "every primary check passes: the MR suite against both baselines, the panel non-inferior to both, the knowledge guard"
