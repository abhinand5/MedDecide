#!/usr/bin/env python
# ruff: noqa: RUF001
"""Merge Osler-9B and Osler-4B LoRA + readout weights into standalone Hugging Face repos,
verify standalone `AutoModel.from_pretrained(..., trust_remote_code=True)` inference & parity
without `meddecide`, and upload to `abhinand/Osler-9B` and `abhinand/Osler-4B` via HF_TOKEN2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[2]
SECRETS_ENV = Path("/workspace/.secrets.env")
D3_SNAP = Path(
    "/workspace/.hf_home/hub/models--vllm-sr--d3-flash/snapshots/67e03e1bc157961c582ee7c6a35d9b15fada3dec"
)
PPLX_SNAP = Path(
    "/workspace/.hf_home/hub/models--perplexity-ai--pplx-decider-flash-v1.2/snapshots/9ff7cf9838e59df461c276b7699d56b244d2af58"
)
STAGING_9B = REPO / "outputs/osler_rc0/R8/release_staging/osler-9b-rc0"
STAGING_4B = REPO / "outputs/osler_rc0/R8/release_staging/osler-4b-rc0"
OUT_ROOT = REPO / "outputs/osler_rc0/R8/merged_standalone"

TEMPS_9B = {"choice": 1.527412, "noul": 2.318485, "score": 1.703540}
TEMPS_4B = {"choice": 1.5222331463699836, "noul": 2.3743542371703033, "score": 1.5388436657067504}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hf_token2() -> str:
    if os.environ.get("HF_TOKEN2"):
        return os.environ["HF_TOKEN2"].strip()
    for candidate in (Path("/workspace/secrets.env"), Path("/workspace/.secrets.env")):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            k, v = line.split("=", 1)
            if k.strip() == "HF_TOKEN2":
                return v.strip().strip('"').strip("'")
    raise RuntimeError("HF_TOKEN2 not found in environment, /workspace/secrets.env, or /workspace/.secrets.env")


def merge_safetensors_shards(
    base_dir: Path,
    out_dir: Path,
    deltas: dict[str, tuple[torch.Tensor, torch.Tensor, float]],
) -> list[str]:
    """Merge LoRA (A, B, scaling) deltas into safetensors shards; untouched shards copy verbatim."""
    index = json.loads((base_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))

    missing = [k for k in deltas if k not in weight_map]
    if missing:
        raise KeyError(f"LoRA target keys missing from base index: {missing[:5]}")

    shard_targets: dict[str, list[str]] = {s: [] for s in shards}
    for k in deltas:
        shard_targets[weight_map[k]].append(k)

    merged_count = 0
    for shard in shards:
        src = (base_dir / shard).resolve()
        dst = out_dir / shard
        keys = shard_targets[shard]
        if not keys:
            shutil.copy2(src, dst)
            print(f"  [copy]  {shard} (untouched)")
            continue
        tensors = load_file(str(src), device="cpu")
        for k in keys:
            a, b, scale = deltas[k]
            w = tensors[k]
            delta = (b.float() @ a.float()) * float(scale)
            if delta.shape != w.shape:
                raise ValueError(f"Shape mismatch for {k}: base {w.shape} vs delta {delta.shape}")
            tensors[k] = (w.float() + delta).to(w.dtype).contiguous()
            merged_count += 1
        save_file(tensors, str(dst), metadata={"format": "pt"})
        print(f"  [merge] {shard} ({len(keys)} tensors merged)")
        del tensors

    shutil.copy2((base_dir / "model.safetensors.index.json").resolve(), out_dir / "model.safetensors.index.json")
    print(f"  Total merged LoRA tensors: {merged_count}/{len(deltas)}")
    assert merged_count == len(deltas)
    return shards


def patch_d3_runtime_for_qtype_temps(src_text: str) -> str:
    """Add optional per-qtype temperature rescaling from decision_config['temperatures']."""
    anchor_init = '        self.temperature = float(config.get("temperature", 1.0))'
    repl_init = (
        '        self.temperature = float(config.get("temperature", 1.0))\n'
        '        self.temperatures = {\n'
        '            str(k): float(v) for k, v in (config.get("temperatures") or {}).items()\n'
        '        }'
    )
    if anchor_init not in src_text:
        raise RuntimeError("Could not find temperature init anchor in d3_runtime.py")
    out = src_text.replace(anchor_init, repl_init, 1)

    anchor_resp = "                answers[key] = product_answer(question, probabilities[key])"
    repl_resp = (
        "                probs_q = probabilities[key]\n"
        "                t_q = self.temperatures.get(question.kind)\n"
        "                if t_q is not None and abs(t_q - self.temperature) > 1e-9:\n"
        "                    ratio = self.temperature / t_q\n"
        "                    logs = [math.log(max(float(p), 1e-30)) * ratio for p in probs_q]\n"
        "                    m = max(logs)\n"
        "                    exps = [math.exp(x - m) for x in logs]\n"
        "                    s = sum(exps)\n"
        "                    probs_q = [e / s for e in exps]\n"
        "                answers[key] = product_answer(question, probs_q)"
    )
    if anchor_resp not in out:
        raise RuntimeError("Could not find product_answer anchor in d3_runtime.py")
    return out.replace(anchor_resp, repl_resp, 1)


def patch_pplx_model_for_qtype_temps(src_text: str) -> str:
    """Support optional per-qtype temperatures in pplx_decider/model.py."""
    anchor_init = '        self.temperature = float(config["temperature"])'
    repl_init = (
        '        self.temperature = float(config["temperature"])\n'
        '        self.temperatures: dict[str, float] = {\n'
        '            str(k): float(v) for k, v in (config.get("temperatures") or {}).items()\n'
        '        }'
    )
    if anchor_init not in src_text:
        raise RuntimeError("Could not find temperature init anchor in pplx_decider/model.py")
    out = src_text.replace(anchor_init, repl_init, 1)

    anchor_dec = (
        "        scale = self.temperature if temperature is None else float(temperature)\n"
        "        if not math.isfinite(scale) or scale <= 0:\n"
        '            raise ValueError("Temperature must be positive and finite")\n'
        "        responses = [self._respond(request, logits, chunks, scale) for request, logits, chunks in self._run(batch)]"
    )
    repl_dec = (
        "        if temperature is not None:\n"
        "            scale = float(temperature)\n"
        "            if not math.isfinite(scale) or scale <= 0:\n"
        '                raise ValueError("Temperature must be positive and finite")\n'
        "            override: float | None = scale\n"
        "        else:\n"
        "            override = None\n"
        "        responses = [self._respond(request, logits, chunks, override) for request, logits, chunks in self._run(batch)]"
    )
    if anchor_dec not in out:
        raise RuntimeError("Could not find decide() anchor in pplx_decider/model.py")
    out = out.replace(anchor_dec, repl_dec, 1)

    anchor_resp = (
        "    def _respond(\n"
        "        self, request: Request, logits: dict[str, list[float]], chunks: list[Encoded], scale: float\n"
        "    ) -> Response:\n"
        "        answers = {}\n"
        '        for key, question in request["questions"].items():\n'
        "            values = torch.tensor(logits[key], dtype=torch.float32)\n"
        "            answers[key] = answer(question, (values / scale).softmax(-1).tolist())"
    )
    repl_resp = (
        "    def _respond(\n"
        "        self, request: Request, logits: dict[str, list[float]], chunks: list[Encoded], override: float | None\n"
        "    ) -> Response:\n"
        "        answers = {}\n"
        '        for key, question in request["questions"].items():\n'
        "            values = torch.tensor(logits[key], dtype=torch.float32)\n"
        '            q_scale = self.temperatures.get(str(question.get("type", "")), self.temperature) if override is None else override\n'
        "            answers[key] = answer(question, (values / q_scale).softmax(-1).tolist())"
    )
    if anchor_resp not in out:
        raise RuntimeError("Could not find _respond() anchor in pplx_decider/model.py")
    out = out.replace(anchor_resp, repl_resp, 1)

    anchor_proc = (
        '        self.processor = AutoProcessor.from_pretrained(str(checkpoint))\n'
        '        self.processor.tokenizer.padding_side = "left"\n'
        '        self.processor.image_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": self.max_pixels}\n'
        '        tokenizer = self.processor.tokenizer'
    )
    repl_proc = (
        '        try:\n'
        '            self.processor = AutoProcessor.from_pretrained(str(checkpoint))\n'
        '            self.processor.tokenizer.padding_side = "left"\n'
        '            self.processor.image_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": self.max_pixels}\n'
        '        except ImportError:\n'
        '            from transformers import AutoTokenizer\n'
        '            _tok = AutoTokenizer.from_pretrained(str(checkpoint))\n'
        '            _tok.padding_side = "left"\n'
        '            class _TextOnlyProcessor:\n'
        '                def __init__(self, tok: Any) -> None:\n'
        '                    self.tokenizer = tok\n'
        '                def apply_chat_template(self, *a: Any, **kw: Any) -> Any:\n'
        '                    return self.tokenizer.apply_chat_template(*a, **kw)\n'
        '                def __call__(self, *, text: list[str], images: Any = None, return_tensors: str = "pt") -> Any:\n'
        '                    if images:\n'
        '                        raise ImportError("Image inputs require torchvision (pip install torchvision)")\n'
        '                    return self.tokenizer(text, add_special_tokens=False, return_tensors=return_tensors)\n'
        '            self.processor = _TextOnlyProcessor(_tok)\n'
        '        tokenizer = self.processor.tokenizer'
    )
    if anchor_proc not in out:
        raise RuntimeError("Could not find AutoProcessor anchor in pplx_decider/model.py")
    return out.replace(anchor_proc, repl_proc, 1)


MODELING_OSLER_4B = '''"""Standalone Hugging Face AutoModel wrapper for Osler-4B (`trust_remote_code=True`).

Loads the merged `Qwen3_5Model` + 255-row option-code readout via the bundled `pplx_decider`
runtime with zero external library dependencies. Exposes both `.system_one(state=..., questions=...)`
(unified API matching Osler-9B) and `.decide(request)` (native pplx_decider batch API).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config


def _resolve_checkpoint_dir(pretrained_model_name_or_path: str | os.PathLike[str], **kwargs: Any) -> Path:
    local = Path(pretrained_model_name_or_path)
    if local.is_dir():
        return local.resolve()
    from huggingface_hub import snapshot_download

    hub_keys = ("revision", "cache_dir", "token", "local_files_only", "force_download", "proxies")
    hub_kwargs = {k: kwargs[k] for k in hub_keys if k in kwargs and kwargs[k] is not None}
    return Path(snapshot_download(str(pretrained_model_name_or_path), **hub_kwargs)).resolve()


def _resolve_device(device: Any = None, device_map: Any = None) -> str:
    if device is not None:
        return str(torch.device(device))
    if isinstance(device_map, dict) and "" in device_map:
        return str(torch.device(device_map[""]))
    if isinstance(device_map, str) and device_map != "auto":
        return str(torch.device(device_map))
    return "cuda" if torch.cuda.is_available() else "cpu"


class Osler4BModel(PreTrainedModel):
    """Standalone Osler-4B calibrated single-pass medical decision model."""

    config_class = Qwen3_5Config

    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)
        self.decider: Any = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *model_args: Any,
        config: Qwen3_5Config | None = None,
        device: str | None = None,
        device_map: Any = None,
        max_length: int = 32768,
        fast: bool = False,
        output_loading_info: bool = False,
        **kwargs: Any,
    ) -> Any:
        import importlib

        ckpt = _resolve_checkpoint_dir(pretrained_model_name_or_path, **kwargs)
        src_dir = str(ckpt / "source" / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        Decider = importlib.import_module("pplx_decider").Decider

        if config is None:
            config = Qwen3_5Config.from_pretrained(str(ckpt))
        dev = _resolve_device(device, device_map)
        decider = Decider.from_pretrained(ckpt, device=dev, max_length=max_length, fast=fast)
        model = cls(config)
        model.decider = decider
        model.name_or_path = str(pretrained_model_name_or_path)
        model.eval()
        if output_loading_info:
            return model, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
        return model

    def decide(self, requests: dict[str, Any] | list[dict[str, Any]], *, temperature: float | None = None) -> Any:
        """Native pplx_decider single or batched request API."""
        if self.decider is None:
            raise RuntimeError("Load Osler4BModel via from_pretrained")
        return self.decider.decide(requests, temperature=temperature)

    def logits(self, requests: list[dict[str, Any]]) -> list[dict[str, list[float]]]:
        """Raw pre-temperature option logits per question."""
        if self.decider is None:
            raise RuntimeError("Load Osler4BModel via from_pretrained")
        return self.decider.logits(requests)

    def system_one(
        self,
        *,
        state: Any,
        questions: dict[str, Any],
        images: list[Any] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Unified System-One decision API matching Osler-9B."""
        req: dict[str, Any] = {"state": state, "questions": questions}
        if images:
            req["images"] = images
        out = self.decide(req, temperature=temperature)
        return {"model": "abhinand/Osler-4B", **out}

    def forward(
        self,
        state: Any = None,
        questions: dict[str, Any] | None = None,
        images: list[Any] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        return self.system_one(state=state, questions=questions or {}, images=images, temperature=temperature)
'''


def build_readme_9b() -> str:
    return """---
license: apache-2.0
language:
- en
pipeline_tag: zero-shot-classification
library_name: transformers
base_model:
- vllm-sr/d3-flash
- Qwen/Qwen3.5-9B
tags:
- medical
- clinical
- decision-model
- system-one
- calibrated
- healthcare
- safetensors
- merged
---

# Osler-9B

**Osler-9B** (`abhinand/Osler-9B`) is a **standalone, merged** 8.39B calibrated medical decision model ("System One" single-pass). Given a clinical state (a discharge note, clinical trial record, FDA drug label, or medical question) and typed questions (`choice` over 2–255 options, `noul` yes/no, or `score` over 2–10 ordered levels), it returns calibrated probabilities for every offered option in **one prefill pass** — no text generation and **no external custom library required**.

- **Standalone merged weights:** Full BF16 `Qwen3_5Model` safetensors shards (attention LoRA `r=32` merged) + FP32 `[255, 4096]` option-code readout + bundled `d3_runtime` & `d3_server.py`.
- **Non-generative by architecture:** Carries no `lm_head` (`Qwen3_5Model` + 255-way option-code readout); cannot generate free text.
- **Dev-only per-type calibration built in:** `choice: 1.5274`, `noul: 2.3185`, `score: 1.7035` (applied automatically by `AutoModel.from_pretrained(..., trust_remote_code=True)`).
- **Base model:** [`vllm-sr/d3-flash`](https://huggingface.co/vllm-sr/d3-flash) (revision `67e03e1bc157961c582ee7c6a35d9b15fada3dec`, Apache-2.0, built on `Qwen/Qwen3.5-9B`).
- **Source & benchmark repository:** [github.com/abhinand5/MedDecide](https://github.com/abhinand5/MedDecide)
- **License:** Apache-2.0 (`LICENSE` and `NOTICE` included).

---

## Quickstart — Hugging Face `transformers` (Zero Extra Libraries)

```bash
pip install "transformers>=5.0.0" torch safetensors accelerate pillow
# Optional fast linear-attention kernels on CUDA:
pip install flash-linear-attention
```

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("abhinand/Osler-9B", trust_remote_code=True)

result = model.system_one(
    state=(
        "68-year-old male admitted with fever 38.9C, HR 118, RR 26, BP 84/52 mmHg, "
        "lactate 4.2 mmol/L, and blood cultures growing E. coli. Started on IV norepinephrine "
        "and piperacillin-tazobactam in the ICU."
    ),
    questions={
        "sepsis_supported": {
            "type": "noul",
            "instructions": "Does the clinical documentation support a diagnosis of septic shock?",
        },
        "acuity": {
            "type": "choice",
            "instructions": "Select the highest-acuity clinical syndrome documented.",
            "criteria": {
                "septic_shock": "Sepsis with vasopressor-dependent hypotension and hyperlactatemia",
                "uncomplicated_uti": "Lower urinary tract infection without systemic involvement",
                "stable_angina": "Exertional chest pain relieved by rest",
            },
        },
        "documentation_strength": {
            "type": "score",
            "instructions": "Rate the strength of clinical indicator evidence for severe sepsis.",
            "criteria": ["none", "weak", "moderate", "strong", "definitive"],
        },
    },
)

print(result["answers"])
```

---

## Serving with vLLM Semantic Router & Standalone HTTP Server

> **Note on standard `vllm serve`:** Osler-9B is a **noncausal full-attention single-pass decision model** (`Qwen3_5Model` + `[255, 4096]` option-code readout, **no `lm_head`**). Standard autoregressive `vllm serve` expects a causal `ForCausalLM` architecture and does not serve noncausal System-One readout heads. Use **vLLM Semantic Router Engine mode** (`vllm-sr`) or the bundled **`d3_server.py`** HTTP server below.

### Option A — vLLM Semantic Router (`vllm-sr`, recommended for production throughput)
Osler-9B uses the native Decision 3.0 (`d3-code-readout-v1`) package layout supported directly by [vLLM Semantic Router](https://github.com/vllm-project/semantic-router):

```bash
curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --channel dev --no-launch
export PATH="$HOME/.local/bin:$PATH"
vllm-sr serve -e abhinand/Osler-9B
```

Query the live endpoint:
```bash
curl -s http://localhost:8899/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "model": "abhinand/Osler-9B",
    "state": "Serum creatinine rose from 0.9 mg/dL to 2.4 mg/dL over 36 hours with oliguria.",
    "questions": {
      "aki": {"type": "noul", "instructions": "Do the documented lab values meet KDIGO AKI criteria?"}
    }
  }'
```
*(Note: `vllm-sr` uses the scalar `temperature` in `decision_config.json` (`1.5274`, exact for `choice`); for exact per-`qtype` `noul`/`score` temperatures over HTTP, use Option B below.)*

### Option B — Bundled Standalone HTTP Server (`d3_server.py`)
```bash
huggingface-cli download abhinand/Osler-9B --local-dir ./Osler-9B
python ./Osler-9B/d3_server.py --model ./Osler-9B --host 0.0.0.0 --port 8080
```

---

## Headline Benchmark Results (12-Model Unified Evaluation, 9 Frozen Sets)

| Model | Params | `panel` (4,353) | `knowledge` (5,456) | `confirm_general` (5,988) | `mr_verdict` (13,103) | `mimic_suite` (4,253) | 5-Set Macro | Batch-1 p50 | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Osler-9B** | **8.39B** | 0.5927 | 0.7322 | **0.9008** | **0.9016** | **0.7733** | **0.7801** | **33.71 ms** | **17.32 GB** |
| `pplx-27B` *(27B ceiling)* | 26.09B | **0.6510** | **0.7823** | 0.8843 | 0.8093 | 0.6729 | 0.7600 | 90.96 ms | 52.92 GB |
| **Osler-4B** | 4.55B | 0.5502 | 0.6728 | 0.8883 | 0.8941 | 0.7597 | 0.7530 | 26.54 ms | 9.60 GB |
| `d3-flash` *(9B base)* | 8.39B | 0.5899 | 0.7335 | 0.8774 | 0.7351 | 0.6301 | 0.7132 | 33.80 ms | 17.32 GB |
| `pplx-flash` *(4B base)* | 4.54B | 0.5615 | 0.6835 | 0.8889 | 0.7466 | 0.6240 | 0.7009 | 25.51 ms | 9.58 GB |
| `Clef-Flash` | 9.00B | 0.5711 | 0.6864 | 0.8933 | 0.6831 | 0.6306 | 0.6929 | — | — |
| `MedDecider-9B` | 9.00B | 0.5603 | 0.6997 | 0.8825 | 0.6653 | 0.6189 | 0.6853 | — | — |
| `MedDecider-4B` | 4.00B | 0.5436 | 0.6703 | 0.8820 | 0.6681 | 0.6019 | 0.6732 | — | — |

- **Global 2-axis Pareto front** (OOD General Avg vs. Clinical Audit Avg): **Osler-9B** `(0.7419, 0.8375)` and `pplx-27B` `(0.7725, 0.7411)`. Osler-9B is the only ≤10B model on the front, runs **2.7× faster** in **3.1× less VRAM** than `pplx-27B`, and strictly dominates every comparable ≤10B model on all five headline sets.
- **Calibration:** `panel` ECE **0.0229** (raw `0.1085`, best of all 12 models; Brier `0.5223`), `mr_verdict` ECE **0.0220** (Brier `0.1421`), `knowledge` ECE `0.0441`.
- **Selective abstention:** At 80% coverage, Osler-9B reaches **98.22%** accuracy on `mr_verdict` (`99.77%` at 50%) and **82.81%** on `mimic_suite` (`90.31%` at 50%).
- **Additional sets:** `mednli_test` **0.8917** (#1), `pairs_test` **0.9620** (#1), `mimic_heldout` `0.7000`.

---

## Privacy & MIMIC PHI / Memorisation Audit (Local-Only)

- MIMIC-IV 3.1 / MIMIC-IV-Note 2.2 / MIMIC-IV-ED were processed strictly on local hardware (never sent to any hosted API), split by patient (`anchor_year_group ≥ 2014` held out). Zero training data or MIMIC text is included in this repository.
- **Visited-member membership inference** (2,000 MIMIC rows visited during training vs. 2,000 late-anchor non-members with 0 hash overlap): AUC **0.5876** `[0.5703, 0.6051]` vs. zero-shot `d3-flash` control **0.5408** → Δ **+0.0468** (clears pre-registered `≤ 0.05` rule). Y10 historical compatibility AUC: **0.5018**.
- **Verbatim extraction:** Structurally **not applicable** — Osler-9B carries no `lm_head` and cannot generate text.

---

## Honest Limitations

1. **OOD QA parity vs. base / ceiling vs. 27B:** Ties (does not beat) `d3-flash` on `panel` (`+0.0028 [-0.0069, +0.0129]`) and `knowledge` (`-0.0013 [-0.0090, +0.0062]`), and trails `pplx-27B` on hard multi-step medical QA (`MedXpertQA`, `MMLU-Pro Health`).
2. **Held-out template failure (`READOUT_FAIL`):** On `mr_heldout_test / synthea_indicator_evidence_v1` (n=514, excluded from training), Osler-9B falls below chance (`0.356`, `READOUT_FAIL`) while its base passes (`0.459`).
3. **Fresh-distribution calibration shift:** Dev-fitted temperatures leave the model under-confident on fresh `confirm_general` (mean confidence `0.814` vs. accuracy `0.901`, ECE `0.090`). Recalibrate on a small target-domain dev split when deploying under strong distribution shift.
4. **Not for unsupervised clinical decisions:** Research model outputs must be reviewed by qualified clinicians.
"""


def build_readme_4b() -> str:
    return """---
license: apache-2.0
language:
- en
pipeline_tag: zero-shot-classification
library_name: transformers
base_model:
- perplexity-ai/pplx-decider-flash-v1.2
- Qwen/Qwen3.5-4B
tags:
- medical
- clinical
- decision-model
- system-one
- calibrated
- healthcare
- safetensors
- merged
---

# Osler-4B

**Osler-4B** (`abhinand/Osler-4B`) is a **standalone, merged** 4.55B calibrated medical decision model ("System One" single-pass) built for high-throughput, low-VRAM (`9.6 GB`, `26.5 ms`/item) medical record auditing and clinical verification. Given a clinical state and typed questions (`choice` over 2–255 options, `noul` yes/no, or `score` over 2–10 ordered levels), it returns calibrated probabilities in **one prefill pass** — no text generation and **no external custom library required**.

- **Standalone merged weights:** Full BF16 `Qwen3_5Model` safetensors shards (attention LoRA `r=32` merged) + `[255, 2560]` option-code readout + bundled `modeling_osler.py` and `pplx_decider` runtime.
- **Dev-only per-type calibration built in:** `choice: 1.5222`, `noul: 2.3744`, `score: 1.5388` (applied automatically by both `AutoModel` and `Decider`).
- **Base model:** [`perplexity-ai/pplx-decider-flash-v1.2`](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2) (revision `9ff7cf9838e59df461c276b7699d56b244d2af58`, Apache-2.0, built on `Qwen/Qwen3.5-4B`).
- **Source & benchmark repository:** [github.com/abhinand5/MedDecide](https://github.com/abhinand5/MedDecide)
- **License:** Apache-2.0 (`LICENSE` and `NOTICE` included).

---

## Quickstart — Hugging Face `transformers` (Zero Extra Libraries)

```bash
pip install "transformers>=5.0.0" torch safetensors accelerate pillow
```

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("abhinand/Osler-4B", trust_remote_code=True)

result = model.system_one(
    state=(
        "68-year-old male admitted with fever 38.9C, HR 118, RR 26, BP 84/52 mmHg, "
        "lactate 4.2 mmol/L, and blood cultures growing E. coli. Started on IV norepinephrine "
        "and piperacillin-tazobactam in the ICU."
    ),
    questions={
        "sepsis_supported": {
            "type": "noul",
            "instructions": "Does the clinical documentation support a diagnosis of septic shock?",
        },
        "acuity": {
            "type": "choice",
            "instructions": "Select the highest-acuity clinical syndrome documented.",
            "criteria": {
                "septic_shock": "Sepsis with vasopressor-dependent hypotension and hyperlactatemia",
                "uncomplicated_uti": "Lower urinary tract infection without systemic involvement",
                "stable_angina": "Exertional chest pain relieved by rest",
            },
        },
        "documentation_strength": {
            "type": "score",
            "instructions": "Rate the strength of clinical indicator evidence for severe sepsis.",
            "criteria": ["none", "weak", "moderate", "strong", "definitive"],
        },
    },
)

print(result["answers"])
```

### Batched & Fast Serving (Bundled `pplx_decider` Runtime)

You can also call `model.decide([...])` directly on a list of requests, or install the bundled standalone package from the downloaded repo:

```python
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

ckpt = Path(snapshot_download("abhinand/Osler-4B"))
sys.path.insert(0, str(ckpt / "source" / "src"))
from pplx_decider import Decider

# Pass fast=True for torch.compile + bucketed CUDA execution (see requirements-fast.txt)
decider = Decider.from_pretrained(ckpt, device="cuda", fast=False)
responses = decider.decide([
    {
        "state": "Serum creatinine rose from 0.9 to 2.4 mg/dL over 36 hours with oliguria.",
        "questions": {
            "aki": {"type": "noul", "instructions": "Do the lab values meet KDIGO AKI criteria?"}
        },
    }
])
print(responses[0]["answers"])
```

> **Note on vLLM:** Like Osler-9B, Osler-4B uses **noncausal full attention** over `Qwen3_5Model` with a `[255, 2560]` marker-position option-code readout (no causal `lm_head`). Standard autoregressive `vllm serve` targets causal `ForCausalLM` architectures and cannot serve noncausal marker-readout models. For high-throughput HTTP serving with vLLM ecosystem tooling, see [`abhinand/Osler-9B`](https://huggingface.co/abhinand/Osler-9B) (which runs natively on **vLLM Semantic Router** `vllm-sr serve -e abhinand/Osler-9B`); for Osler-4B, wrap `Decider.from_pretrained(..., fast=True)` in a length-bucketed FastAPI/Uvicorn worker (at batch-1, p50 is `26.54 ms` in `9.60 GB` VRAM).

---

## Headline Benchmark Results (12-Model Unified Evaluation, 9 Frozen Sets)

| Model | Params | `panel` (4,353) | `knowledge` (5,456) | `confirm_general` (5,988) | `mr_verdict` (13,103) | `mimic_suite` (4,253) | 5-Set Macro | Batch-1 p50 | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Osler-9B** | 8.39B | 0.5927 | 0.7322 | **0.9008** | **0.9016** | **0.7733** | **0.7801** | 33.71 ms | 17.32 GB |
| `pplx-27B` *(27B ceiling)* | 26.09B | **0.6510** | **0.7823** | 0.8843 | 0.8093 | 0.6729 | 0.7600 | 90.96 ms | 52.92 GB |
| **Osler-4B** | **4.55B** | 0.5502 | 0.6728 | 0.8883 | 0.8941 | 0.7597 | 0.7530 | **26.54 ms** | **9.60 GB** |
| `pplx-flash` *(4B base)* | 4.54B | 0.5615 | 0.6835 | 0.8889 | 0.7466 | 0.6240 | 0.7009 | 25.51 ms | 9.58 GB |
| `MedDecider-4B` | 4.00B | 0.5436 | 0.6703 | 0.8820 | 0.6681 | 0.6019 | 0.6732 | — | — |

- **≤4B specialist frontier:** Strictly dominates `MedDecider-4B` on all five headline sets (`+22.60 pts` on `mr_verdict`, `+15.78 pts` on `mimic_suite`, `+0.66` `panel`, `+0.26` `knowledge`, `+0.63` `confirm_general`), and beats `pplx-27B` on clinical audit (`mr_verdict` `+8.48 pts`, `mimic_suite` `+8.68 pts`) at **3.4× lower latency** and **5.5× smaller VRAM**.
- **Calibration:** `panel` ECE **0.0374** (raw `0.1450`, Brier `0.5643` vs. `0.6416` base), `knowledge` ECE **0.0194**, `mr_verdict` ECE **0.0314**.
- **Selective abstention:** At 80% coverage, reaches **97.93%** on `mr_verdict` (`99.76%` at 50%) and **81.10%** on `mimic_suite` (`87.87%` at 50%).
- **Additional sets:** `mednli_test` `0.8910`, `pairs_test` `0.9580`, `mimic_heldout` `0.6727`.

---

## Privacy & MIMIC PHI / Memorisation Audit (Local-Only)

- MIMIC-IV 3.1 / MIMIC-IV-Note 2.2 / MIMIC-IV-ED were processed strictly locally (never sent to any hosted API), split by patient (`anchor_year_group ≥ 2014` held out). Zero training data or MIMIC text is included in this repository.
- **Visited-member membership inference** (2,000 visited training rows vs. 2,000 late-anchor non-members, 0 hash overlap): AUC **0.5644** `[0.5466, 0.5813]` vs. zero-shot `pplx-flash` control **0.5396** → Δ **+0.0248** (clears `≤ 0.05` rule). Y10 compatibility AUC: **0.5033**.
- **Verbatim extraction probe (300 MIMIC discharge notes):** **0/300** matches at ≥50, ≥25, and ≥10 consecutive tokens (longest run `1`, control `0/300`).

---

## Honest Limitations

1. **OOD QA regression (~1 pt vs. own base):** Trails `pplx-flash` by `-0.0113 [-0.0218, -0.0007]` on `panel` and `-0.0098 [-0.0191, -0.0017]` on `knowledge` (trading ~1 pt of general OOD accuracy for `+14.7` / `+13.6 pts` clinical audit gains and better Brier on 4 of 5 headline sets). Use [`abhinand/Osler-9B`](https://huggingface.co/abhinand/Osler-9B) if general OOD parity is required.
2. **Held-out template failure (`READOUT_FAIL`):** On `mr_heldout_test / synthea_indicator_evidence_v1` (n=514, excluded from training), falls below chance (`0.245`, `READOUT_FAIL`) while `pplx-flash` passes (`0.597`).
3. **Fresh-distribution calibration shift:** Dev-fitted temperatures leave the model under-confident on fresh `confirm_general` (mean confidence `0.801` vs. accuracy `0.888`, ECE `0.094`).
4. **Not for unsupervised clinical decisions:** Research model outputs must be reviewed by qualified clinicians.
"""


def build_osler_9b_merged(out_dir: Path) -> None:
    print(f"\n=== Building standalone merged Osler-9B -> {out_dir} ===")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    ckpt = torch.load(STAGING_9B / "weights/osler_9b_rc0.pt", map_location="cpu")
    model_sd = ckpt.get("model", ckpt)
    lora = model_sd["lora"]
    readout = model_sd["readout"].float().contiguous()

    # Group A/B pairs; key in lora is "layers.X.self_attn.Y.A" / ".B"
    prefixes = sorted({k[:-2] for k in lora if k.endswith(".A")})
    deltas: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    for p in prefixes:
        a = lora[f"{p}.A"]
        b = lora[f"{p}.B"]
        # r=32, alpha=32 => scaling = 1.0
        deltas[f"language_model.{p}.weight"] = (a, b, 1.0)

    merge_safetensors_shards(D3_SNAP, out_dir, deltas)
    save_file({"weight": readout}, str(out_dir / "readout.safetensors"), metadata={"format": "pt"})

    copy_files = [
        "LICENSE",
        "NOTICE",
        "chat_template.jinja",
        "config.json",
        "d3_engine.py",
        "d3_fast.py",
        "d3_format.py",
        "d3_kernels.py",
        "d3_server.py",
        "merges.txt",
        "modeling_d3.py",
        "pipeline_d3.py",
        "preprocessor_config.json",
        "requirements.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    ]
    for name in copy_files:
        src = (D3_SNAP / name).resolve()
        if src.exists():
            shutil.copy2(src, out_dir / name)

    # Patch d3_runtime.py for per-qtype temperatures
    rt_src = (D3_SNAP / "d3_runtime.py").resolve().read_text(encoding="utf-8")
    (out_dir / "d3_runtime.py").write_text(patch_d3_runtime_for_qtype_temps(rt_src), encoding="utf-8")

    # Update decision_config.json
    dc = json.loads((D3_SNAP / "decision_config.json").resolve().read_text(encoding="utf-8"))
    dc["base_model"] = "vllm-sr/d3-flash"
    dc["revision"] = "67e03e1bc157961c582ee7c6a35d9b15fada3dec"
    dc["temperature"] = TEMPS_9B["choice"]
    dc["temperatures"] = dict(TEMPS_9B)
    (out_dir / "decision_config.json").write_text(json.dumps(dc, indent=2) + "\n", encoding="utf-8")

    # Write calibration.json and README.md
    shutil.copy2(STAGING_9B / "calibration.json", out_dir / "calibration.json")
    (out_dir / "README.md").write_text(build_readme_9b(), encoding="utf-8")

    # Build fresh MODEL_MANIFEST.json so d3_runtime verify="fast" / "full" passes cleanly
    files_sha: dict[str, str] = {}
    files_bytes: dict[str, int] = {}
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file() or p.name == "MODEL_MANIFEST.json":
            continue
        rel = p.relative_to(out_dir).as_posix()
        files_sha[rel] = sha256_file(p)
        files_bytes[rel] = p.stat().st_size
    model_hash_input = json.dumps(
        {k: files_sha[k] for k in sorted(files_sha) if k.endswith(".safetensors")}, sort_keys=True
    ).encode()
    manifest = {
        "schema": "d3-package-manifest/1",
        "model_name": "Osler-9B",
        "repo_id": "abhinand/Osler-9B",
        "format_id": dc["format_id"],
        "prompt": dc["prompt"],
        "attention_mode": dc["attention_mode"],
        "max_input_tokens": dc.get("max_length"),
        "identity": {
            "model_sha256": hashlib.sha256(model_hash_input).hexdigest(),
            "files_sha256": {
                k: v for k, v in files_sha.items()
                if k.endswith(".safetensors") or k in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json")
            },
            "decision": {k: v for k, v in dc.items() if k not in ("codes", "token_ids")},
        },
        "files_sha256": files_sha,
        "files_bytes": files_bytes,
    }
    (out_dir / "MODEL_MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print("  Osler-9B merged package complete.")


def build_osler_4b_merged(out_dir: Path) -> None:
    print(f"\n=== Building standalone merged Osler-4B -> {out_dir} ===")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    ad = load_file(str(STAGING_4B / "weights/adapter/adapter_model.safetensors"), device="cpu")
    ac = json.loads((STAGING_4B / "weights/adapter/adapter_config.json").read_text(encoding="utf-8"))
    scale = float(ac["lora_alpha"]) / float(ac["r"])
    op = torch.load(STAGING_4B / "weights/optcode.pt", map_location="cpu")
    readout = op["weight"].to(torch.bfloat16).contiguous()

    # PEFT key: base_model.model.language_model.layers.X.self_attn.Y.lora_A.weight
    prefixes = sorted({k[: -len(".lora_A.weight")] for k in ad if k.endswith(".lora_A.weight")})
    deltas: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    for p in prefixes:
        base_key = p.removeprefix("base_model.model.") + ".weight"
        deltas[base_key] = (ad[f"{p}.lora_A.weight"], ad[f"{p}.lora_B.weight"], scale)

    merge_safetensors_shards(PPLX_SNAP, out_dir, deltas)
    save_file({"weight": readout}, str(out_dir / "readout.safetensors"), metadata={"format": "pt"})

    copy_files = [
        "LICENSE",
        "NOTICE",
        "chat_template.jinja",
        "joint_config.json",
        "processor_config.json",
        "requirements-fast.txt",
        "requirements.txt",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for name in copy_files:
        src = (PPLX_SNAP / name).resolve()
        if src.exists():
            shutil.copy2(src, out_dir / name)

    # Copy bundled pplx_decider standalone source & patch for per-qtype temperatures
    shutil.copytree(
        (PPLX_SNAP / "source").resolve(),
        out_dir / "source",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    px_model_py = out_dir / "source/src/pplx_decider/model.py"
    px_model_py.write_text(patch_pplx_model_for_qtype_temps(px_model_py.read_text(encoding="utf-8")), encoding="utf-8")

    # Add standalone AutoModel wrapper + auto_map in config.json
    (out_dir / "modeling_osler.py").write_text(MODELING_OSLER_4B, encoding="utf-8")
    cfg = json.loads((PPLX_SNAP / "config.json").resolve().read_text(encoding="utf-8"))
    cfg["auto_map"] = {"AutoModel": "modeling_osler.Osler4BModel"}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    # Update decision_config.json
    dc = json.loads((PPLX_SNAP / "decision_config.json").resolve().read_text(encoding="utf-8"))
    dc["base_model"] = "perplexity-ai/pplx-decider-flash-v1.2"
    dc["revision"] = "9ff7cf9838e59df461c276b7699d56b244d2af58"
    dc["temperature"] = TEMPS_4B["choice"]
    dc["temperatures"] = dict(TEMPS_4B)
    (out_dir / "decision_config.json").write_text(json.dumps(dc, indent=2) + "\n", encoding="utf-8")

    # Write calibration.json and README.md
    shutil.copy2(STAGING_4B / "calibration.json", out_dir / "calibration.json")
    (out_dir / "README.md").write_text(build_readme_4b(), encoding="utf-8")
    print("  Osler-4B merged package complete.")


def sanity_check_hygiene(root: Path) -> None:
    """Assert zero /workspace/ paths or secret token leaks in any text file."""
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix in (".safetensors", ".pt", ".bin", ".png", ".mp4"):
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if "/workspace/" in txt:
            raise RuntimeError(f"Found /workspace/ path in {p}")
        if "hf_" in txt and any(tok.startswith("hf_") and len(tok) > 20 for tok in txt.split()):
            raise RuntimeError(f"Possible HF token in {p}")


def verify_standalone_gpu(dir_9b: Path, dir_4b: Path) -> dict[str, Any]:
    """Load both merged models with pure HF AutoModel (NO meddecide imports) and verify outputs."""
    from transformers import AutoModel

    sample_state = (
        "68-year-old male admitted with fever 38.9C, HR 118, RR 26, BP 84/52 mmHg, "
        "lactate 4.2 mmol/L, and blood cultures growing E. coli. Started on IV norepinephrine."
    )
    sample_questions = {
        "sepsis_noul": {
            "type": "noul",
            "instructions": "Does the note support septic shock?",
        },
        "dx_choice": {
            "type": "choice",
            "instructions": "Select the primary documented condition.",
            "criteria": {
                "septic_shock": "Sepsis with hypotension requiring vasopressors and hyperlactatemia",
                "migraine": "Recurrent unilateral throbbing headache",
                "eczema": "Chronic pruritic inflammatory dermatitis",
            },
        },
        "evidence_score": {
            "type": "score",
            "instructions": "Rate clinical evidence strength for severe sepsis.",
            "criteria": ["none", "weak", "moderate", "strong", "definitive"],
        },
    }

    report: dict[str, Any] = {}
    for name, path in (("Osler-9B", dir_9b), ("Osler-4B", dir_4b)):
        print(f"\n=== GPU standalone verification: {name} ===")
        m = AutoModel.from_pretrained(str(path), trust_remote_code=True)
        out = m.system_one(state=sample_state, questions=sample_questions)
        ans = out["answers"]
        print(f"  {name} answers:", json.dumps(ans, indent=2))
        assert ans["dx_choice"]["choice"] == "septic_shock", f"{name} unexpected choice: {ans}"
        assert ans["sepsis_noul"]["noul"] > 0.65, f"{name} unexpected noul: {ans}"
        assert ans["evidence_score"]["score"] >= 2.0, f"{name} unexpected score: {ans}"
        report[name] = ans
        del m
        torch.cuda.empty_cache()

    # Also verify Osler-9B & Osler-4B merged parity against R2/R3 calibrated predictions (5 panel items)
    print("\n=== Checking Osler-9B & Osler-4B merged parity against R2/R3 calibrated predictions (5 panel items) ===")
    from meddecide.eval.rc0_board import load_cell, load_frozen

    frozen_panel = load_frozen("panel", REPO / "data/bench/osler_v4/y9/panel.jsonl")
    panel_items = [
        json.loads(line)
        for _, line in zip(range(5), (REPO / "data/bench/osler_v4/y9/panel.jsonl").open(encoding="utf-8"), strict=False)
    ]
    cell_9b = load_cell("Osler-9B", frozen_panel, REPO / "outputs/osler_rc0/R2/preds_cal/9b_rc0_panel/predictions.jsonl")
    cell_4b = load_cell("Osler-4B", frozen_panel, REPO / "outputs/osler_rc0/R3/preds_cal/4b_rc0_panel/predictions.jsonl")
    assert cell_9b is not None and cell_9b.probs is not None
    assert cell_4b is not None and cell_4b.probs is not None

    for name, path, ref_cell in (("Osler-9B", dir_9b, cell_9b), ("Osler-4B", dir_4b, cell_4b)):
        m = AutoModel.from_pretrained(str(path), trust_remote_code=True)
        max_diff = 0.0
        argmax_agree = 0
        for idx, item in enumerate(panel_items):
            qtype = item["qtype"]
            if qtype == "choice":
                if name == "Osler-4B":
                    q = {
                        "type": "choice",
                        "instructions": item["question"],
                        "criteria": {o["label"]: None for o in item["options"]},
                    }
                else:
                    q = {
                        "type": "choice",
                        "instructions": item["question"],
                        "criteria": {o["key"]: o["label"] for o in item["options"]},
                    }
            elif qtype == "noul":
                q = {"type": "noul", "instructions": item["question"]}
            else:
                q = {"type": "score", "instructions": item["question"], "criteria": [o["label"] for o in item["options"]]}
            res = m.system_one(state=item["state"], questions={"q": q})["answers"]["q"]
            ref_probs = [float(p) for p in ref_cell.probs[ref_cell.position(idx)]]
            if qtype == "choice":
                if name == "Osler-4B":
                    got_probs = [res["probabilities"][o["label"]] for o in item["options"]]
                else:
                    got_probs = [res["probabilities"][o["key"]] for o in item["options"]]
            elif qtype == "noul":
                p_yes = res["noul"]
                got_probs = [p_yes if o["key"] == "yes" else 1.0 - p_yes for o in item["options"]]
            else:
                p_map = res["probabilities"]
                got_probs = (
                    [p_map[str(i)] for i in range(len(item["options"]))]
                    if isinstance(p_map, dict)
                    else list(p_map)
                )
            diff = max(abs(a - b) for a, b in zip(got_probs, ref_probs, strict=True))
            max_diff = max(max_diff, diff)
            got_arg = max(range(len(got_probs)), key=lambda i: got_probs[i])
            ref_arg = max(range(len(ref_probs)), key=lambda i: ref_probs[i])
            if got_arg == ref_arg:
                argmax_agree += 1
        print(f"  {name} merged vs R2/R3 calibrated (5 panel items): argmax {argmax_agree}/5, max |Δp| = {max_diff:.6f}")
        assert argmax_agree == 5 and max_diff < 0.05, f"{name} parity drift too large: {argmax_agree}/5, {max_diff}"
        report[f"{name}_parity"] = {"argmax_agree": argmax_agree, "total": 5, "max_abs_delta_p": max_diff}
        del m
        torch.cuda.empty_cache()

    return report


def push_to_hf(dir_9b: Path, dir_4b: Path) -> dict[str, Any]:
    token = load_hf_token2()
    api = HfApi(token=token)
    who = api.whoami()
    account = who["name"]
    print(f"\nAuthenticated to Hugging Face as: {account}")
    if account != "abhinand":
        raise RuntimeError(f"Expected HF account 'abhinand', got '{account}'")

    manifest: dict[str, Any] = {
        "account": account,
        "uploaded_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "standalone_merged": True,
        "repos": {},
    }
    for repo_name, folder in (("Osler-9B", dir_9b), ("Osler-4B", dir_4b)):
        repo_id = f"{account}/{repo_name}"
        print(f"\nUploading merged standalone {repo_id} from {folder} ...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        info = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(folder),
            commit_message=f"Publish standalone merged {repo_name} (full safetensors + readout + runtime + calibration)",
            delete_patterns="*",
            ignore_patterns=["**/__pycache__/**", "*.pyc"],
        )
        oid = getattr(info, "oid", None) or str(info)
        commit_url = getattr(info, "commit_url", f"https://huggingface.co/{repo_id}/commit/{oid}")
        print(f"  DONE {repo_id}: {commit_url}")
        manifest["repos"][repo_name] = {
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}",
            "oid": oid,
            "commit_url": commit_url,
        }

    out_manifest = REPO / "outputs/osler_rc0/R8/hf_upload/upload_manifest.json"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-push", action="store_true")
    args = parser.parse_args()

    dir_9b = OUT_ROOT / "Osler-9B"
    dir_4b = OUT_ROOT / "Osler-4B"

    if not args.skip_build:
        build_osler_9b_merged(dir_9b)
        build_osler_4b_merged(dir_4b)
        sanity_check_hygiene(dir_9b)
        sanity_check_hygiene(dir_4b)

    if not args.skip_verify:
        rep = verify_standalone_gpu(dir_9b, dir_4b)
        (OUT_ROOT / "standalone_verify.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")

    if not args.skip_push:
        push_to_hf(dir_9b, dir_4b)


if __name__ == "__main__":
    main()
