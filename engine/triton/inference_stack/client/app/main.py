import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tritonclient.grpc as grpcclient
from fastapi import FastAPI, UploadFile
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

# Size this to your CPU cores / workload. Start with something modest.
CPU_POOL = ThreadPoolExecutor(max_workers=4)

def decode_phonemes(ids: list, vocab):
    id_to_ph = {v: k for k, v in vocab["phonemes"].items()}
    out_str = ""
    for idx in ids:
        out_str += id_to_ph[idx]
    return out_str

def preprocess_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """
    Runs on a worker thread.
    Returns FP16 features ready for Triton.
    """
    audio = np.frombuffer(audio_bytes, dtype=np.float32)

    features = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
        padding=True,
    )["input_features"].astype(np.float16)

    return features


@app.post("/infer")
async def infer_audio(file: UploadFile):
    audio_bytes = await file.read()

    loop = asyncio.get_running_loop()

    # Offload CPU-heavy preprocessing so the event loop isn't blocked
    features = await loop.run_in_executor(CPU_POOL, preprocess_audio_bytes, audio_bytes)

    inp = grpcclient.InferInput("INPUT__0", features.shape, "FP16")
    inp.set_data_from_numpy(features)

    out0 = grpcclient.InferRequestedOutput("OUTPUT__0")
    out1 = grpcclient.InferRequestedOutput("OUTPUT__1")
    out2 = grpcclient.InferRequestedOutput("OUTPUT__2")


    t0 = time.perf_counter()

    # If this blocks noticeably too, you can offload it as well (see below).
    res = triton.infer(
        model_name=MODEL_NAME,
        inputs=[inp],
        outputs=[out0, out1, out2],
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    
    logits = res.as_numpy("OUTPUT__0")
    ids    = res.as_numpy("OUTPUT__1")
    probs  = res.as_numpy("OUTPUT__2")

    print("logits shape:", logits.shape)   # must end with 43 for phonemes
    print("ids shape:", ids.shape)
    print("probs shape:", probs.shape)
    print("ids unique:", sorted(set(ids.flatten().tolist()))[:20], "...")
    ph_ids = ctc_decode(res.as_numpy("OUTPUT__1"), res.as_numpy("OUTPUT__2"))[0].ids.tolist()
    decoded_ph = decode_phonemes(ph_ids, mulit_level_tokenizer.vocab)

    return{
        "latency_ms" : latency_ms,
        "phonemes" : decoded_ph,
    }
    # return {
    #     "latency_ms": latency_ms,
    #     "output0_shape": res.as_numpy("OUTPUT__0").shape,
    #     "output1_shape": res.as_numpy("OUTPUT__1").shape,
    #     "output2_shape": res.as_numpy("OUTPUT__2").shape,
    # }

