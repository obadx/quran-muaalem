import time
import numpy as np
import tritonclient.grpc as grpcclient

from fastapi import FastAPI, UploadFile
from transformers import AutoFeatureExtractor

app = FastAPI()

SAMPLE_RATE = 16000
MODEL_NAME = "muaalem"

processor = AutoFeatureExtractor.from_pretrained(
    "obadx/muaalem-model-v3_2",
    sampling_rate=SAMPLE_RATE,
)

triton = grpcclient.InferenceServerClient(
    url="triton:8001"
)

@app.post("/infer")
async def infer_audio(file: UploadFile):
    audio = np.frombuffer(await file.read(), dtype=np.float32)

    features = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
        padding=True,
    )["input_features"].astype(np.float16)

    inp = grpcclient.InferInput(
        "INPUT__0",
        features.shape,
        "FP16"
    )
    inp.set_data_from_numpy(features)

    out0 = grpcclient.InferRequestedOutput("OUTPUT__0")
    out1 = grpcclient.InferRequestedOutput("OUTPUT__1")

    t0 = time.perf_counter()
    res = triton.infer(
        model_name=MODEL_NAME,
        inputs=[inp],
        outputs=[out0, out1],
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "latency_ms": latency_ms,
        "output0_shape": res.as_numpy("OUTPUT__0").shape,
        "output1_shape": res.as_numpy("OUTPUT__1").shape,
    }
