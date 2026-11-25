"""O0 smoke test: run a model card's own Python examples verbatim and compare with its printed values.

The card's ```python blocks are executed in order in one namespace (the first block typically
downloads or loads the model and defines the handle; later blocks call it). Each block's own
comment lines give the card's printed result, e.g. "# Yes  0.000" or "# correct: No"; the
probabilities returned by the call are compared with those printed values
(meddecide.eval.model_card does the parsing and the comparison).

PASS for an example: every printed probability is within TOL of the returned value, no printed
label is missing from the returned options, and the argmax of the returned values is the argmax of
the printed values. The card's gold label is recorded beside it but is not part of the PASS
criterion: a card example the model answers differently is a result, not a smoke failure.
The card's numbers are rounded to 3 dp, so the default TOL of 0.01 leaves room for rounding and
bf16 noise, and nothing more.

Usage:
    uv run --frozen python scripts/osler/o0_smoke_card.py --card README.md --id md4b \
        --out outputs/osler_v0/O0/smoke/md4b.json
"""

from __future__ import annotations

import argparse
import ast
import time
from pathlib import Path
from typing import Any

from meddecide.eval.model_card import argmax_label, card_blocks, compare_printed
from meddecide.utils.io import write_json
from meddecide.utils.provenance import gpu_name, utcnow

TOL = 0.01


def run(card_path: Path, out_path: Path, model_id: str, tol: float) -> dict[str, Any]:
    blocks = card_blocks(card_path.read_text())
    ns: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    t0 = time.time()
    for block in blocks:
        tree = ast.parse(block.code)
        last_value: Any = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = tree.body.pop()
            exec(compile(tree, f"<card block {block.index}>", "exec"), ns)  # card code, by design
            last_value = eval(compile(ast.Expression(last.value), "<card expr>", "eval"), ns)
        else:
            exec(compile(tree, f"<card block {block.index}>", "exec"), ns)  # card code, by design
        if not isinstance(last_value, dict):
            results.append({"block": block.index, "returned_type": type(last_value).__name__,
                            "status": "NO_PROBABILITIES"})
            continue
        returned = {str(k): float(v) for k, v in last_value.items()}
        comparison = compare_printed(returned, block.printed)
        argmax = argmax_label(returned)
        printed_argmax = argmax_label(block.printed)
        argmax_ok = None if printed_argmax is None else str(argmax).startswith(printed_argmax)
        gold_ok = None if block.correct is None else str(argmax).startswith(block.correct[:20])
        worst = comparison.worst_abs_diff
        status = "PASS" if (worst is not None and worst <= tol and not comparison.missing
                            and argmax_ok is not False) else "FAIL"
        results.append({"block": block.index, "returned": returned,
                        "printed_vs_returned": comparison.diffs, "argmax": argmax,
                        "printed_argmax": printed_argmax, "argmax_matches_printed": argmax_ok,
                        "card_gold_label": block.correct, "argmax_matches_card_gold": gold_ok,
                        "worst_abs_diff": worst, "status": status})
    prob_results = [r for r in results if "printed_vs_returned" in r]
    overall = ("PASS" if prob_results and all(r["status"] == "PASS" for r in prob_results)
               else "FAIL")
    payload = {
        "id": model_id,
        "card": str(card_path),
        "task": "O0 smoke (card examples run verbatim)",
        "tolerance": tol,
        "examples_with_probabilities": len(prob_results),
        "overall": overall,
        "wall_clock_s": round(time.time() - t0, 1),
        "utc": utcnow(),
        "gpu": gpu_name(),
        "results": results,
    }
    write_json(out_path, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--card", type=Path, required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()
    payload = run(args.card, args.out, args.id, args.tol)
    print(f"{args.id}: overall={payload['overall']} examples_with_probs={payload['examples_with_probabilities']}")
    for r in payload["results"]:
        print(f"  block {r['block']}: status={r['status']} worst_abs_diff={r.get('worst_abs_diff')} "
              f"argmax_ok={r.get('argmax_matches_printed')} gold_ok={r.get('argmax_matches_card_gold')}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
