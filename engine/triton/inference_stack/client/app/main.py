import io
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tritonclient.grpc as grpcclient
from fastapi import FastAPI, UploadFile
import librosa
from transformers import AutoFeatureExtractor

from quran_muaalem.modeling.multi_level_tokenizer import MultiLevelTokenizer
from quran_muaalem.decode import ctc_decode

app = FastAPI()

SAMPLE_RATE = 16000
MODEL_NAME = "muaalem"

mulit_level_tokenizer = MultiLevelTokenizer("obadx/muaalem-model-v3_2")

processor = AutoFeatureExtractor.from_pretrained(
    "obadx/muaalem-model-v3_2",
    sampling_rate=SAMPLE_RATE,
)

triton = grpcclient.InferenceServerClient(url="triton:8001")
CPU_POOL = ThreadPoolExecutor(max_workers=4)

def decode_phonemes(ids: list[int], vocab) -> str:
    id_to_ph = {v: k for k, v in vocab["phonemes"].items()}
    return "".join(id_to_ph[i] for i in ids if i in id_to_ph)

def preprocess_waveform(wave: np.ndarray) -> np.ndarray:
    """
    Runs on a worker thread.
    Input: wave float32 shape (n_samples,)
    Output: features float16 shape (1, Tin, 160) matching Triton INPUT__0
    """
    # IMPORTANT: return_tensors="np" so we get numpy arrays
    feats = processor(
        wave,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
        padding=False,
    )["input_features"]  # usually float32, shape (1, Tin, 160)

    # Cast to float16 for Triton FP16 input
    feats = np.ascontiguousarray(feats.astype(np.float16))
    return feats

@app.post("/infer")
async def infer_audio(file: UploadFile):
    audio_bytes = await file.read()

    # ✅ Correct way to use librosa.load with uploaded bytes
    wave, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)

    loop = asyncio.get_running_loop()
    features = await loop.run_in_executor(CPU_POOL, preprocess_waveform, wave)

    # Sanity checks (temporary)
    # features should be (1, Tin, 160)
    # print("features:", features.shape, features.dtype, features.min(), features.max(), features.mean())

    inp = grpcclient.InferInput("INPUT__0", features.shape, "FP16")
    inp.set_data_from_numpy(features)

    outputs = [
        grpcclient.InferRequestedOutput("OUTPUT__0"),
        grpcclient.InferRequestedOutput("OUTPUT__1"),
        grpcclient.InferRequestedOutput("OUTPUT__2"),
    ]

    t0 = time.perf_counter()
    res = triton.infer(model_name=MODEL_NAME, inputs=[inp], outputs=outputs)
    latency_ms = (time.perf_counter() - t0) * 1000

    logits = res.as_numpy("OUTPUT__0")  # (1,T,43)
    ids    = res.as_numpy("OUTPUT__1")  # (1,T)
    probs  = res.as_numpy("OUTPUT__2")  # (1,T)

    # Debug (temporary)
    # print("logits shape:", logits.shape)
    # print("ids unique:", sorted(set(ids.flatten().tolist()))[:20])

    # ✅ ctc_decode expects (B,T) ids and (B,T) probs
    decoded0 = ctc_decode(ids, probs, blank_id=0)[0]  # consider setting blank_id to model.config.pad_token_id
    ph_ids = decoded0.ids.tolist()

    decoded_ph = decode_phonemes(ph_ids, mulit_level_tokenizer.vocab)

    return {"latency_ms": latency_ms, "phonemes": decoded_ph}