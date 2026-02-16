from __future__ import annotations

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


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_ID = "obadx/muaalem-model-v3_2"
DEFAULT_SAMPLING_RATE = 16000


@dataclass
class EngineResources:
    model: torch.nn.Module
    processor: AutoFeatureExtractor
    tokenizer: MultiLevelTokenizer
    device: torch.device
    dtype: torch.dtype
    sampling_rate: int
    using_trt: bool
    trt_path: Path | None = None


from typing import Tuple
import torch

class _WrappedModel(torch.nn.Module):
    """
    Returns per level:
      logits: (B, T, V)
      ids:    (B, T)  int64
      probs:  (B, T)  float32  (prob of the chosen id at each frame)
    """
    def __init__(self, model, levels=None):
        super().__init__()
        self.model = model
        # IMPORTANT: fix ordering to match OUTPUT__* in config.pbtxt
        self.levels = ["phonemes"]

    def forward(self, input_features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        logits_dict = self.model(
            input_features=input_features,
            attention_mask=None,
            return_dict=False,
        )[0]  # dict[level] -> (B, T, V)

        outs = []
        
        logits = logits_dict["phonemes"]                 # (B, T, V)
        ids = logits.argmax(dim=-1).to(torch.int64) # (B, T)

        # probs of the chosen token per frame
        probs = torch.softmax(logits.to(torch.float32), dim=-1) \
                    .gather(dim=-1, index=ids.unsqueeze(-1)) \
                    .squeeze(-1)                      # (B, T) float32

        outs.extend([logits, ids, probs])



        return tuple(outs)



def _get_input_tensor(features: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "input_features" in features:
        return features["input_features"]
    raise ValueError(f"Unsupported processor output keys: {list(features.keys())}")


def _prepare_features(
    wave: np.ndarray | Sequence[float] | List[float] | List[Sequence[float]],
    processor: AutoFeatureExtractor,
    sampling_rate: int,
    device: torch.device,
    dtype: torch.dtype,
    padding: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    features = processor(
        wave,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding="longest" if padding else False,
    )

    input_features = _get_input_tensor(features).to(device=device, dtype=dtype)
    if input_features.dim() < 3:
        raise ValueError(
            f"`input_features` must be 3D (batch, frames, feature_size), got shape {tuple(input_features.shape)}. "
            "Check the feature extractor configuration."
        )
    attention_mask = features.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)
    return input_features, attention_mask


def _decode_phonemes(
    level_to_ids: Dict[str, torch.Tensor],
    level_to_probs: Dict[str, torch.Tensor],
    tokenizer: MultiLevelTokenizer,
) -> Tuple[List[int], str]:
    decode_out = ctc_decode(level_to_ids["phonemes"], level_to_probs["phonemes"])[0]
    ids = decode_out.ids.tolist()
    id_to_ph = {v: k for k, v in tokenizer.vocab["phonemes"].items()}
    decoded = "".join(id_to_ph[idx] for idx in ids)
    return ids, decoded


def _run_logits(
    model: torch.nn.Module,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    with torch.no_grad():
        if isinstance(model, _WrappedModel) or isinstance(
            model, (torch.jit.ScriptModule, torch.jit.RecursiveScriptModule)
        ):
            logits_out = model(input_features)
        else:
            logits_out = model(input_features, attention_mask)
    if isinstance(logits_out, dict):
        level_to_logits = logits_out
    elif isinstance(logits_out, (tuple, list)) and hasattr(model, "levels"):
        level_to_logits = {lvl: logits_out[idx] for idx, lvl in enumerate(model.levels)}
    else:
        raise TypeError(
            f"Unexpected logits output type {type(logits_out)}; "
            "expected dict or tuple with `levels` attribute on model."
        )
    level_to_probs = {}
    level_to_ids = {}
    for k, logits in level_to_logits.items():
        probs = torch.softmax(logits, dim=-1)
        ids = probs.argmax(dim=-1)
        gathered = probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        level_to_probs[k] = gathered.cpu()
        level_to_ids[k] = ids.cpu()
    return level_to_ids, level_to_probs


def _attach_levels(module: torch.nn.Module, levels: List[str]) -> None:
    try:
        setattr(module, "levels", levels)
    except Exception:
        LOGGER.warning("Could not attach levels attribute to module of type %s", type(module))


def _merge_lists_with_overlap(
    A: List[int], B: List[int], max_B_offset: int = 2
) -> List[int]:
    a_orig_offset = max(len(A) - len(B) - max_B_offset, 0)
    b_span = min(len(A), len(B))
    curr_match = {"longest": None, "start_a": None, "start_b": None}
    best_match = {"longest": None, "start_a": None, "start_b": None}

    for a_offset in range(a_orig_offset, len(A)):
        for ptr in range(b_span):
            a_idx = a_offset + ptr
            b_idx = ptr
            if A[a_idx] == B[b_idx]:
                if curr_match["longest"] is None:
                    curr_match.update(
                        {"longest": 1, "start_a": a_idx, "start_b": b_idx}
                    )
                else:
                    curr_match["longest"] += 1
            elif curr_match["longest"] is not None:
                if (
                    best_match["longest"] is None
                    or curr_match["longest"] > best_match["longest"]
                ):
                    best_match.update(curr_match)
                curr_match["longest"] = None
        if curr_match["longest"] is not None and (
            best_match["longest"] is None
            or curr_match["longest"] > best_match["longest"]
        ):
            best_match.update(curr_match)
        curr_match["longest"] = None
        if (a_offset + len(B)) >= len(A):
            b_span -= 1

    if best_match["longest"] is None:
        return A + B
    start_a = best_match["start_a"]
    start_b = best_match["start_b"]
    assert start_a is not None and start_b is not None
    return A[:start_a] + B[start_b:]


def sliding_window_inference(
    wave: np.ndarray,
    resources: EngineResources,
    window_ms: int,
    chunk_ms: int,
) -> Tuple[List[int], str]:
    window_samples = int(window_ms / 1000 * resources.sampling_rate)
    chunk_samples = int(chunk_ms / 1000 * resources.sampling_rate)

    merged_ids: List[int] = []
    total_len = len(wave)
    if total_len < window_samples:
        # Pad to window length for short clips
        pad_len = window_samples - total_len
        chunk = np.pad(wave, (0, pad_len), mode="constant")
        start_positions = [0]
    else:
        start_positions = list(range(0, total_len - window_samples + 1, chunk_samples))

    for start in start_positions:
        end = start + window_samples
        chunk = wave[start:end] if end <= total_len else np.pad(wave[start:], (0, end - total_len), mode="constant")
        input_features, attention_mask = _prepare_features(
            chunk,
            resources.processor,
            resources.sampling_rate,
            resources.device,
            resources.dtype,
            padding=False,
        )
        level_to_ids, level_to_probs = _run_logits(
            resources.model, input_features, attention_mask
        )
        curr_ids, _ = _decode_phonemes(
            level_to_ids,
            level_to_probs,
            resources.tokenizer,
        )
        if curr_ids and curr_ids[-1] == 0:
            curr_ids = curr_ids[:-1]
        if merged_ids:
            merged_ids = _merge_lists_with_overlap(merged_ids, curr_ids)
        else:
            merged_ids = curr_ids

    decoded = "".join(
        {v: k for k, v in resources.tokenizer.vocab["phonemes"].items()}[idx]
        for idx in merged_ids
    )
    return merged_ids, decoded


def offline_inference(
    waves: List[np.ndarray],
    resources: EngineResources,
) -> List[Tuple[List[int], str]]:
    input_features, attention_mask = _prepare_features(
        waves,
        resources.processor,
        resources.sampling_rate,
        resources.device,
        resources.dtype,
        padding=True,
    )
    level_to_ids, level_to_probs = _run_logits(
        resources.model, input_features, attention_mask
    )
    outs: List[Tuple[List[int], str]] = []
    for batch_idx in range(input_features.shape[0]):
        ids = level_to_ids["phonemes"][batch_idx : batch_idx + 1]
        probs = level_to_probs["phonemes"][batch_idx : batch_idx + 1]
        decoded_ids, decoded = _decode_phonemes(
            {"phonemes": ids},
            {"phonemes": probs},
            resources.tokenizer,
        )
        outs.append((decoded_ids, decoded))
    return outs


def _trace_wrapper_for_trt(
    model: Wav2Vec2BertForMultilevelCTC,
    processor: AutoFeatureExtractor,
    sampling_rate: int,
    device: torch.device,
    dtype: torch.dtype,
    example_seconds: float = 3.0,
) -> Tuple[torch.jit.ScriptModule, torch.Tensor, torch.Tensor | None]:
    dummy_wave = np.zeros(int(sampling_rate * example_seconds), dtype=np.float32)
    # Force at least one frame; use padding to ensure processor returns expected dims
    input_features, attention_mask = _prepare_features(
        dummy_wave,
        processor,
        sampling_rate,
        device,
        dtype,
        padding=True,
    )
    wrapper = _WrappedModel(model)
    scripted = torch.jit.trace(
        wrapper, (input_features,), check_trace=False, strict=False
    )
    scripted.eval()
    return scripted, input_features, attention_mask


def build_or_load_trt(
    model_id: str = DEFAULT_MODEL_ID,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
    trt_path: str | Path = "engine/muaalem_trt.ts",
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    min_seconds: float = 1.0,
    opt_seconds: float = 3.0,
    max_seconds: float = 6.0,
    force_rebuild: bool = False,
    max_batch_size: int = 4,
    use_trt: bool = True,
) -> EngineResources:
    device = torch.device(device)
    dtype = dtype or (torch.float16 if device.type == "cuda" else torch.float32)
    trt_path = Path(trt_path)

    processor = AutoFeatureExtractor.from_pretrained(
        model_id, sampling_rate=sampling_rate
    )
    tokenizer = MultiLevelTokenizer(model_id)

    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(model_id)
    model.to(device, dtype=dtype)
    model.eval()
    levels = list(model.level_to_lm_head.keys())

    if trt_path.exists() and not force_rebuild and use_trt:
        LOGGER.info("Loading existing TensorRT engine from %s", trt_path)
        trt_module = torch.jit.load(trt_path, map_location=device)
        _attach_levels(trt_module, getattr(trt_module, "levels", levels))
        return EngineResources(
            model=trt_module,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            sampling_rate=sampling_rate,
            using_trt=True,
            trt_path=trt_path,
        )

    if torch_tensorrt is None or not use_trt:
        LOGGER.warning(
            "TensorRT disabled or unavailable; falling back to plain TorchScript."
        )
        wrapper = _WrappedModel(model)
        return EngineResources(
            model=wrapper,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            sampling_rate=sampling_rate,
            using_trt=False,
            trt_path=None,
        )

    LOGGER.info("Tracing model for TensorRT compilation...")
    scripted, example_input, example_mask = _trace_wrapper_for_trt(
        model, processor, sampling_rate, device, dtype
    )



    batch_dim, opt_len, feature_dim = example_input.shape
    min_len = max(1, int(opt_len * min_seconds / opt_seconds))
    max_len = int(opt_len * max_seconds / opt_seconds)

    LOGGER.info("Compiling TensorRT engine...")
    inputs = [
        torch_tensorrt.Input(
            min_shape=(1, min_len, feature_dim),
            opt_shape=(min(max_batch_size, 2), opt_len, feature_dim),
            max_shape=(max_batch_size, max_len, feature_dim),
            dtype=dtype,
        )
    ]
    try:
        trt_module = torch_tensorrt.compile(
            scripted,
            inputs=inputs,
            ir="torchscript",
            require_full_compilation=False,  # allow fallback for unsupported ops
            enabled_precisions={dtype if dtype != torch.bfloat16 else torch.float16},
            workspace_size=1 << 28,
        )
        _attach_levels(trt_module, levels)
        trt_module.eval()
        trt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(trt_module, trt_path)
        LOGGER.info("Saved TensorRT engine to %s", trt_path)
        return EngineResources(
            model=trt_module,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            sampling_rate=sampling_rate,
            using_trt=True,
            trt_path=trt_path,
        )
    except Exception as exc:
        LOGGER.warning(
            "TensorRT compilation failed (%s). Falling back to TorchScript.", exc
        )
        wrapper = _WrappedModel(model)
        wrapper.eval()
        _attach_levels(scripted, levels)
        torch.jit.save(scripted, trt_path)
        return EngineResources(
            model=wrapper,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            sampling_rate=sampling_rate,
            using_trt=False,
            trt_path=trt_path,
        )
