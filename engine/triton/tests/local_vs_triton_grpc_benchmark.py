
import time
import numpy as np
import torch
import tritonclient.grpc as grpcclient
from transformers import AutoFeatureExtractor

# =====================
# Config
# =====================
MODEL_ID = "obadx/muaalem-model-v3_2"
TRITON_MODEL_NAME = "muaalem"
TRITON_URL = "localhost:8001"   # gRPC port

SAMPLING_RATE = 16000
DURATION_SEC = 15
DTYPE = torch.float16

WARMUP_ITERS = 10
BENCH_ITERS = 1024

# =====================
# Input preparation
# =====================
processor = AutoFeatureExtractor.from_pretrained(
    MODEL_ID,
    sampling_rate=SAMPLING_RATE,
)

wave = np.random.randn(SAMPLING_RATE * DURATION_SEC).astype(np.float32)

features = processor(
    wave,
    sampling_rate=SAMPLING_RATE,
    return_tensors="np",
    padding=True,
)

input_features = features["input_features"].astype(np.float16)
print("Input shape:", input_features.shape)

# =====================
# Triton gRPC client
# =====================
client = grpcclient.InferenceServerClient(TRITON_URL)

inputs = [
    grpcclient.InferInput(
        "INPUT__0",
        input_features.shape,
        "FP16",
    )
]
inputs[0].set_data_from_numpy(input_features)

# Auto-discover outputs
metadata = client.get_model_metadata(TRITON_MODEL_NAME)
outputs = [
    grpcclient.InferRequestedOutput(o.name)
    for o in metadata.outputs
]

# =====================
# Warmup
# =====================
for _ in range(WARMUP_ITERS):
    client.infer(
        model_name=TRITON_MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
    )

# =====================
# Benchmark
# =====================
start = time.time()
for _ in range(BENCH_ITERS):
    client.infer(
        model_name=TRITON_MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
    )
end = time.time()

latency_ms = (end - start) * 1000 / BENCH_ITERS

print(f"\nTriton gRPC latency: {latency_ms:.2f} ms")
