from __future__ import annotations

import logging
import os
from pathlib import Path
from time import perf_counter
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .trt_runtime import (
    DEFAULT_MODEL_ID,
    DEFAULT_SAMPLING_RATE,
    EngineResources,
    build_or_load_trt,
    offline_inference,
    sliding_window_inference,
)


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
ENGINE_PATH = Path("engine/muaalem_trt.ts")
TARGET_SR = DEFAULT_SAMPLING_RATE

USE_TRT = os.getenv("MUALEM_DISABLE_TRT", "0").lower() not in {"1", "true", "yes"}

LOGGER.info("Loading TRT resources on device=%s (use_trt=%s)...", DEVICE, USE_TRT)
RESOURCES: EngineResources = build_or_load_trt(
    model_id=DEFAULT_MODEL_ID,
    device=DEVICE,
    dtype=DEFAULT_DTYPE,
    trt_path=ENGINE_PATH,
    sampling_rate=DEFAULT_SAMPLING_RATE,
    use_trt=USE_TRT,
)
LOGGER.info(
    "Model ready (TensorRT=%s, engine=%s)",
    RESOURCES.using_trt,
    RESOURCES.trt_path,
)


class SlidingWindowRequest(BaseModel):
    audio: List[float]
    sampling_rate: Optional[int] = None
    window_ms: int = 6000
    chunk_ms: int = 300


class OfflineRequest(BaseModel):
    audio: Optional[List[float]] = None
    audios: Optional[List[List[float]]] = None
    sampling_rate: Optional[int] = None


app = FastAPI(title="Muaalem TensorRT Inference API")


def _validate_sampling_rate(req_rate: Optional[int]) -> None:
    if req_rate is not None and req_rate != RESOURCES.sampling_rate:
        raise HTTPException(
            status_code=400,
            detail=f"Sampling rate must be {RESOURCES.sampling_rate}, got {req_rate}",
        )


def _resample_to_target_sr(wave: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return wave
    # Simple linear resample without extra deps
    src_len = len(wave)
    target_len = int(round(src_len * target_sr / src_sr))
    if target_len <= 0:
        raise HTTPException(status_code=400, detail="Audio too short after resample.")
    x_old = np.linspace(0, 1, src_len, endpoint=False)
    x_new = np.linspace(0, 1, target_len, endpoint=False)
    return np.interp(x_new, x_old, wave).astype(np.float32)


@app.post("/inference/sliding-window")
async def sliding_window_endpoint(body: SlidingWindowRequest):
    src_sr = body.sampling_rate or TARGET_SR
    wave = np.array(body.audio, dtype=np.float32)
    if wave.ndim != 1:
        raise HTTPException(status_code=400, detail="Audio must be mono 1D array.")
    wave = _resample_to_target_sr(wave, src_sr, TARGET_SR)
    start = perf_counter()
    ids, decoded = sliding_window_inference(
        wave,
        RESOURCES,
        window_ms=body.window_ms,
        chunk_ms=body.chunk_ms,
    )
    duration_ms = (perf_counter() - start) * 1000
    LOGGER.info(
        "Sliding window request processed in %.2f ms (len=%d)",
        duration_ms,
        len(wave),
    )
    return {
        "phonemes_ids": ids,
        "decoded": decoded,
        "latency_ms": duration_ms,
        "using_tensorrt": RESOURCES.using_trt,
    }


@app.post("/inference/offline")
async def offline_endpoint(body: OfflineRequest):
    _validate_sampling_rate(body.sampling_rate)
    if body.audios:
        waves = [np.array(w, dtype=np.float32) for w in body.audios]
    elif body.audio:
        waves = [np.array(body.audio, dtype=np.float32)]
    else:
        raise HTTPException(
            status_code=400, detail="Provide `audio` or `audios` in the request body."
        )

    start = perf_counter()
    outs = offline_inference(waves, RESOURCES)
    duration_ms = (perf_counter() - start) * 1000

    response = [
        {"phonemes_ids": ids, "decoded": decoded} for ids, decoded in outs
    ]
    LOGGER.info(
        "Offline request (batch=%d) processed in %.2f ms", len(waves), duration_ms
    )
    return {
        "results": response,
        "batch_size": len(waves),
        "latency_ms": duration_ms,
        "using_tensorrt": RESOURCES.using_trt,
    }


@app.get("/")
async def root():
    return {
        "message": "Quran Muaalem inference service is live.",
        "endpoints": ["/inference/sliding-window", "/inference/offline"],
        "using_tensorrt": RESOURCES.using_trt,
    }
