"""synth_v1 final scoring wrapper: the aggregate summary and the Gate O4 wiring comparison (pure functions; no GPU, no data)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("v1_final_score", REPO / "scripts" / "synth" / "v1_final_score.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gate(values: dict[str, float], panel_pplx: str = "FAIL", panel_meddecider: str = "NOT MEASURED") -> dict:
    def check(name: str, point: float | None, verdict: str = "PASS") -> dict:
        acc = None if point is None else {"point": point, "lower": point - 0.01, "upper": point + 0.01}
        return {"check": name, "verdict": verdict, "reason": "", "accuracy": acc, "brier": None, "items": 10}

    return {"mode": "validate", "sets_scored": ["mr_verdict"], "results": {
        "pplx_flash": {"verdict": "FAIL", "checks": {
            "i": check("i mr_verdict", values["pp_mr"]),
            "ii": check("ii mimic_suite", values["pp_mimic"]),
            "iii": check("iii panel", values["pp_panel"], panel_pplx),
            "iv": check("iv knowledge", values["pp_know"]),
        }},
        "meddecider_4b": {"verdict": "NOT MEASURED", "checks": {
            "i": check("i mr_verdict", values["md_mr"]),
            "ii": check("ii mimic_suite", values["md_mimic"]),
            "iii": check("iii panel", None, panel_meddecider),
            "iv": check("iv knowledge", 0.0),
        }},
    }}


OSLER_V4 = {"pp_mr": 0.1387, "pp_mimic": 0.1654, "pp_panel": -0.0421, "pp_know": -0.0067,
            "md_mr": 0.2449, "md_mimic": 0.1651}


def test_wiring_reproduces_osler_v4_figures() -> None:
    module = _load()
    assert module.wiring_mismatches(_gate(OSLER_V4)) == []


def test_wiring_flags_a_changed_figure_and_a_changed_verdict() -> None:
    module = _load()
    changed = dict(OSLER_V4, pp_mr=0.20)
    problems = module.wiring_mismatches(_gate(changed, panel_pplx="PASS"))
    assert any("pplx_flash mr_verdict" in p for p in problems)
    assert any("pplx_flash panel" in p and "verdict" in p for p in problems)


def test_summary_is_aggregate_only() -> None:
    module = _load()
    gate = _gate(OSLER_V4)
    gate["results"]["pplx_flash"]["checks"]["i"]["item_text"] = "secret"
    summary = module.summarise(gate)
    text = repr(summary)
    assert "secret" not in text
    assert summary["baselines"]["pplx_flash"]["verdict"] == "FAIL"
    assert summary["baselines"]["meddecider_4b"]["checks"]["iii"]["verdict"] == "NOT MEASURED"
    assert summary["baselines"]["pplx_flash"]["checks"]["i"]["accuracy"]["point"] == 0.1387
