"""A trained checkpoint's logits must not depend on how the item is padded (strict fp32, 8 dev items).

The test runs with cuDNN TF32 switched off, so the fp32 check uses full-precision convolutions.
PyTorch enables TF32 for cuDNN by default. Measured in V2 (SELF_AUDIT), the setting does not change
the padded-batch residual (identical values either way), so this is a precision pin, not a fix.
The previous setting is restored afterwards. The tolerance is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "outputs" / "student_v0" / "S9_run3" / "checkpoints" / "step_1000"
BENCH = ROOT / "data" / "bench" / "v0.2"
TOLERANCE = 1e-3


def _dev_sample(n: int) -> list:
    from meddecide.train.data import read_items

    items = []
    for directory in ("tier1", "fresh"):
        for path in sorted((BENCH / directory).glob("*.jsonl")):
            items.extend(i for i in read_items(path) if str(i.split) == "dev")
    items.sort(key=lambda i: (len(i.state) + len(i.question), i.item_id))
    step = max(1, (len(items) - 1) // (n - 1))
    return [items[k * step] for k in range(n)]


@pytest.mark.skipif(not CHECKPOINT.is_dir(), reason="trained checkpoint not present")
@pytest.mark.xfail(
    strict=True,
    reason="V2 BLOCKED (acceptance item 1): padded-batch residual 1.92e-3 > 1e-3 in strict fp32; "
           "the sequence-length shape control reaches 1.50e-3. Tolerance not changed; see "
           "loops/student_v1/V2_padding.md. strict=True: a pass here must be reviewed.",
)
def test_trained_checkpoint_logits_are_padding_invariant() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device for the fp32 trained-checkpoint check")
    from meddecide.model.meddecide_model import MedDecideModel

    previous_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        sample = _dev_sample(8)
        model = MedDecideModel.load(CHECKPOINT, device="cuda:0", dtype="float32")
        kwargs = {"max_batch_tokens": 10**9}
        alone = model.score_items(sample, batch_size=1, **kwargs)
        padded = model.score_items(sample, batch_size=len(sample), **kwargs)
    finally:
        torch.backends.cudnn.allow_tf32 = previous_tf32
    assert len(set(alone.prompt_tokens)) > 1, "the sample must span different prompt lengths"
    deltas = [
        float(np.max(np.abs(a - b))) for a, b in zip(alone.logits, padded.logits, strict=True)
    ]
    assert max(deltas) <= TOLERANCE, deltas
