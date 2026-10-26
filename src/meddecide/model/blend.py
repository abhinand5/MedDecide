"""Weight-space blend toward the base (osler_v5 D53; WiSE-FT style).

For a trained option-code model and alpha in [0, 1]:
- the LoRA update is scaled by alpha through the adapter scaling of every LoRA layer (not by a merged bf16 copy), so the adapted
  projection is ``W + alpha · (B A) · (alpha / r)``;
- the readout is ``(1 - alpha) · R_base + alpha · R_trained``, where R_base is the reference readout (the readout at step 0, which
  :meth:`MedDecideModel.snapshot_reference_readout` takes when the model is built from the base LM head).

alpha = 0 is therefore the base model: its scores equal :meth:`MedDecideModel.reference_option_probs` (the adapter off, the base readout).
alpha = 1 is the trained model exactly (the scalings and the readout are restored bit for bit). The temperatures are not touched here:
the caller refits them on dev for each alpha (D53).
"""

from __future__ import annotations

from typing import Any

import torch

from meddecide.model.meddecide_model import MedDecideModel


class Blend:
    """Holds the trained readout and the LoRA scalings of one model, and sets the blend weight in place."""

    def __init__(self, model: MedDecideModel) -> None:
        if model.optcode is None:
            raise ValueError("the blend needs the option-code readout")
        reference = getattr(model, "reference_readout", None)
        if reference is None:
            raise ValueError("the blend needs the reference readout (R_base) of the model")
        self.model = model
        self.trained_readout = model.optcode.weight.detach().clone().to(torch.float32)
        self.base_readout = reference.detach().clone().to(torch.float32)
        self.base_scaling: list[tuple[Any, dict[str, float]]] = []
        if model.peft_model is not None:
            for module in model.peft_model.modules():
                scaling = getattr(module, "scaling", None)
                if isinstance(scaling, dict) and scaling:
                    self.base_scaling.append((module, {name: float(value) for name, value in scaling.items()}))
        if not self.base_scaling and model.peft_model is not None:
            raise ValueError("no LoRA layer with a scaling was found")
        self.alpha = 1.0

    def apply(self, alpha: float) -> None:
        """Set the blend weight. Raises for alpha outside [0, 1]."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        with torch.no_grad():
            readout = (1.0 - alpha) * self.base_readout + alpha * self.trained_readout
            self.model.optcode.weight.copy_(readout.to(device=self.model.optcode.weight.device,
                                                       dtype=self.model.optcode.weight.dtype))
            for module, scales in self.base_scaling:
                for name, scale in scales.items():
                    module.scaling[name] = scale * alpha
        self.alpha = float(alpha)


def export_blend(blend: Blend, alpha: float) -> None:
    """Make the model itself carry the blend at ``alpha`` (osler_v5 Z4): the LoRA B matrices are multiplied by alpha, so with the
    scalings unchanged the update is alpha times the trained one, exactly as the adapter scaling gives; the readout is the blend.
    The model then saves and reloads as an ordinary checkpoint. After this call the Blend object must not be applied again."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    blend.apply(1.0)  # the trained scalings and readout first
    model = blend.model
    with torch.no_grad():
        if model.peft_model is not None:
            for name, parameter in model.peft_model.named_parameters():
                if "lora_B" in name:
                    parameter.mul_(alpha)
        readout = (1.0 - alpha) * blend.base_readout + alpha * blend.trained_readout
        model.optcode.weight.copy_(readout.to(device=model.optcode.weight.device, dtype=model.optcode.weight.dtype))
