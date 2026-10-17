"""osler_v2: the pplx-decider-flash port (``meddecide.model.pplx``) against pplx_decider's own reference path.

GPU only; skipped unless CUDA, the downloaded checkpoint and torchvision (needed by pplx's processor) are present. Run it
with torchvision added for the run only:
    uv run --frozen --with pillow --with "torchvision==0.29.*" pytest -q tests/test_pplx_parity.py
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

torch = pytest.importorskip("torch")

from meddecide.bench.schema import Option, QuestionType, Tier, make_item  # noqa: E402
from meddecide.train.config import StudentConfig  # noqa: E402


def test_config_validates_the_backbone() -> None:
    assert StudentConfig.from_dict({"backbone": "pplx", "readout": "option_code"}).backbone == "pplx"
    with pytest.raises(ValueError, match="backbone"):
        StudentConfig.from_dict({"backbone": "llama"})
    with pytest.raises(ValueError, match="pplx backbone"):
        StudentConfig.from_dict({"backbone": "pplx", "readout": "pointer"})


def _items():
    from datetime import date

    out = []
    for i, (state, q, labels, gold) in enumerate([
        ("Lactate 3.4 mmol/L on admission; MAP 58 on norepinephrine.", "Does the record show septic shock criteria?",
         ["Yes", "No"], 0),
        ("Discharge diagnosis: acute on chronic hypoxic respiratory failure.", "Which code applies?",
         ["J96.01", "J96.11", "J96.21", "None of these"], 2),
        ("Itemized statement: room and board, ceftriaxone 1 g IV, CBC.", "Which page type is this?",
         ["Discharge summary", "Bill or itemised charges", "Radiology report"], 1),
    ]):
        opts = [Option(key=chr(65 + j), label=lab) for j, lab in enumerate(labels)]
        out.append(make_item(tier=Tier.FRESH, source="t", source_record_id=f"p{i}", source_url="https://example.org",
                             source_license="test", record_date=date(2026, 1, 1), split="dev", template_id="t",
                             skill="t", qtype=QuestionType.CHOICE, state=state, question=q, options=opts,
                             gold=opts[gold].key, option_order_seed=i))
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_port_matches_the_reference_path() -> None:
    if importlib.util.find_spec("torchvision") is None:
        pytest.skip("pplx's processor needs torchvision (add it for the run only)")
    from meddecide.model.pplx import REPO, PplxDecisionModel, checkpoint_dir

    try:
        ckpt = checkpoint_dir(REPO)
    except Exception as exc:
        pytest.skip(f"checkpoint not available: {exc}")
    sys.path.insert(0, str(ckpt / "source" / "src"))
    from pplx_decider import Decider

    ours = PplxDecisionModel(device="cuda:0", lora=None)
    ours.eval_mode()
    ref = Decider.from_pretrained(ckpt, device="cuda")
    for item in _items():
        enc = ours.encode_item(item)
        request = {"state": item.state, "questions": {"q": {"type": "choice", "instructions": item.question,
                                                          "criteria": {o.label: None for o in item.options}}}}
        assert ref.encoder.encode(request).input_ids.tolist() == enc.input_ids
        with torch.no_grad():
            flat, _ = ours.batch_logits(ours.collate([enc], round_to_chunk=False))
        theirs = torch.tensor(ref.logits([request])[0]["q"])
        assert float((flat.float().cpu() - theirs).abs().max()) < 0.15
        assert int(flat.argmax()) == int(theirs.argmax())
