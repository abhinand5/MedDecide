"""d3_distill step 3 (CPU): build the test-side knowledge evaluation sets that exist on this machine.

The frozen Gate O7 sets (the Y9 knowledge/panel/mr_verdict files) are not present here (STATE.md constraint 2), so the gate
itself is NOT MEASURED. This script builds the public test splits that *are* available and can be scored as an additional,
clearly-labelled diagnostic:

- ``medmcqa_val``: the official MedMCQA validation split (the bench's tier-1 MedMCQA test side, template
  ``medmcqa_4opt_v1``, the same prompt as training);
- ``medqa_test``: the official MedQA-USMLE-4-options test split, template ``medqa_usmle4_v1`` (downloaded once; the
  dataset card is checked and recorded).

No item text is committed: the output goes under the gitignored ``data/`` tree, the counts and hashes under
``outputs/d3_distill/step3/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")  # this builder is the one explicit network user; datasets come from cache

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from meddecide.utils.hashing import file_sha256  # noqa: E402

OUT = REPO / "data" / "bench" / "d3_distill"
MANIFEST = REPO / "outputs" / "d3_distill" / "step3" / "eval_sets_manifest.json"
MEDMCQA_QUESTION = "Which of the following is the best answer?"
LETTERS = "ABCDE"


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_rows(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"file": str(path.relative_to(REPO)), "rows": len(rows), "sha256": file_sha256(path)}


def medmcqa_val() -> tuple[list[dict], dict]:
    from datasets import load_dataset

    split = load_dataset("openlifescienceai/medmcqa", split="validation")
    rows, dropped = [], 0
    for i, r in enumerate(split):
        labels = [r.get("opa"), r.get("opb"), r.get("opc"), r.get("opd")]
        cop = r.get("cop")
        if not all(isinstance(x, str) and x.strip() for x in labels) or cop not in (0, 1, 2, 3):
            dropped += 1
            continue
        rows.append(
            {
                "eval_id": f"medmcqa_val:{i}",
                "source": "medmcqa_val",
                "template_id": "medmcqa_4opt_v1",
                "qtype": "choice",
                "state": str(r["question"]).strip(),
                "question": MEDMCQA_QUESTION,
                "options": [{"key": k, "label": str(lab).strip()} for k, lab in zip("ABCD", labels, strict=True)],
                "gold": "ABCD"[int(cop)],
            }
        )
    return rows, {"dropped_malformed": dropped, "dataset": "openlifescienceai/medmcqa", "split": "validation"}


def medqa_test() -> tuple[list[dict], dict]:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download("GBaker/MedQA-USMLE-4-options", "phrases_no_exclude_test.jsonl", repo_type="dataset")
    )
    rows, dropped = [], 0
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            rec = json.loads(line)
            options = rec.get("options") or {}
            answer_idx = str(rec.get("answer_idx", "")).strip()
            if not isinstance(options, dict) or len(options) < 2 or answer_idx not in options:
                dropped += 1
                continue
            keys = sorted(options)
            letters = list(LETTERS[: len(keys)])
            rows.append(
                {
                    "eval_id": f"medqa_test:{i}",
                    "source": "medqa_test",
                    "template_id": "medqa_usmle4_v1",
                    "qtype": "choice",
                    "state": str(rec["question"]).strip(),
                    "question": "Which of the following is the best answer?",
                    "options": [
                        {"key": letter, "label": str(options[key])}
                        for letter, key in zip(letters, keys, strict=True)
                    ],
                    "gold": letters[keys.index(answer_idx)],
                }
            )
    return rows, {"dropped_malformed": dropped, "dataset": "GBaker/MedQA-USMLE-4-options", "split": "test",
                  "file": path.name}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--with-medqa", action="store_true", help="also build the MedQA test set (network download)")
    args = ap.parse_args()

    manifest = {"generated_utc": utcnow(), "sets": {}}
    rows, info = medmcqa_val()
    manifest["sets"]["medmcqa_val"] = {"counts": info, **write_rows(OUT / "medmcqa_val.jsonl", rows)}
    print(f"medmcqa_val: {len(rows)} rows ({info['dropped_malformed']} malformed dropped)", flush=True)
    if args.with_medqa:
        rows, info = medqa_test()
        manifest["sets"]["medqa_test"] = {"counts": info, **write_rows(OUT / "medqa_test.jsonl", rows)}
        print(f"medqa_test: {len(rows)} rows ({info['dropped_malformed']} malformed dropped)", flush=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
