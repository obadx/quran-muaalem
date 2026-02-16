from __future__ import annotations

from trt_runtime import _prepare_features, _WrappedModel, _attach_levels

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

try:
    import torch_tensorrt
except Exception:  # pragma: no cover - optional dependency
    torch_tensorrt = None

from transformers import AutoFeatureExtractor

from quran_muaalem.modeling.modeling_multi_level_ctc import (
    Wav2Vec2BertForMultilevelCTC,
)
from quran_muaalem.modeling.multi_level_tokenizer import MultiLevelTokenizer
from quran_muaalem.decode import ctc_decode

DEFAULT_MODEL_ID = "obadx/muaalem-model-v3_2"
DEFAULT_SAMPLING_RATE = 16000   


def export_torchscript_for_triton(
    model_id: str = DEFAULT_MODEL_ID,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
    example_seconds: float = 3.0,
    out_path: str | Path = "model.pt",
):
    device = torch.device(device)
    dtype = dtype or (torch.float16 if device.type == "cuda" else torch.float32)

    processor = AutoFeatureExtractor.from_pretrained(
        model_id, sampling_rate=sampling_rate
    )

    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(model_id)
    model.to(device=device, dtype=dtype)
    model.eval()

    # ---- dummy input (THIS defines Triton shape contract) ----
    dummy_wave = np.zeros(
        int(example_seconds * sampling_rate), dtype=np.float32
    )
    input_features, _ = _prepare_features(
        dummy_wave,
        processor,
        sampling_rate,
        device,
        dtype,
        padding=True,
    )

    wrapper = _WrappedModel(model).to(device).eval()
    _attach_levels(wrapper, list(model.level_to_lm_head.keys()))

    with torch.no_grad():
        scripted = torch.jit.trace(
            wrapper,
            (input_features,),
            strict=False,
            check_trace=False,
        )

    scripted = torch.jit.freeze(scripted)
    scripted.eval()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))

    print(f"✅ TorchScript saved to {out_path}")
    print(f"Input shape example: {tuple(input_features.shape)}")
    print(f"Outputs: {len(wrapper.levels)} levels")

    return scripted


# if __name__ == "__main__":

#     sampling_rate = 16000

#     # audio_path = "./assets/test.wav"
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model_id = "obadx/muaalem-model-v3_2"
#     dtype = torch.float16
            
#     mulit_level_tokenizer = MultiLevelTokenizer(model_id)

#     model = Wav2Vec2BertForMultilevelCTC.from_pretrained(model_id)
#     model.to(device, dtype=dtype)
#     export_torchscript_for_triton(model_id,
#     sampling_rate,
#     device, dtype,3)



if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sampling-rate", type=int, default=DEFAULT_SAMPLING_RATE)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--example-seconds", type=float, default=3.0)
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    export_torchscript_for_triton(
        model_id=args.model_id,
        sampling_rate=args.sampling_rate,
        device=args.device,
        dtype=dtype,
        example_seconds=args.example_seconds,
        out_path=args.output,   # ✅ THIS is the missing link
    )