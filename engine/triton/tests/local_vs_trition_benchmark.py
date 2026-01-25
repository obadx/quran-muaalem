
import time
import numpy as np
import torch
import tritonclient.http as httpclient
from transformers import AutoFeatureExtractor

# =====================
# Config
# =====================
MODEL_ID = "obadx/muaalem-model-v3_2"
TRITON_MODEL_NAME = "muaalem"
TRITON_URL = "localhost:8000"

TORCHSCRIPT_PATH = "../model.pt"

SAMPLING_RATE = 16000
DURATION_SEC = 15          # audio length
DTYPE = torch.float16

WARMUP_ITERS = 10
BENCH_ITERS = 1024

# =====================
# Input preparation
# =====================
print("Preparing input...")

processor = AutoFeatureExtractor.from_pretrained(
    MODEL_ID,
    sampling_rate=SAMPLING_RATE,
)

wave = np.random.randn(SAMPLING_RATE * DURATION_SEC).astype(np.float32)

features = processor(
    wave,
    sampling_rate=SAMPLING_RATE,
    return_tensors="pt",
    padding=True,
)

input_features_pt = features["input_features"].cuda().to(DTYPE)
input_features_np = input_features_pt.cpu().numpy()

print("Input shape:", input_features_pt.shape)

# =====================
# Local TorchScript benchmark
# =====================
print("\n===== Local TorchScript =====")

model = torch.jit.load(TORCHSCRIPT_PATH).cuda().eval()

# Warmup
with torch.no_grad():
    for _ in range(WARMUP_ITERS):
        _ = model(input_features_pt)
torch.cuda.synchronize()

# Benchmark
start = time.time()
with torch.no_grad():
    for _ in range(BENCH_ITERS):
        _ = model(input_features_pt)
torch.cuda.synchronize()
end = time.time()

local_latency_ms = (end - start) * 1000 / BENCH_ITERS
print(f"Local TorchScript latency: {local_latency_ms:.2f} ms")

# =====================
# Triton benchmark
# =====================
print("\n===== Triton HTTP =====")

client = httpclient.InferenceServerClient(TRITON_URL)

inputs = [
    httpclient.InferInput(
        "INPUT__0",
        input_features_np.shape,
        "FP16",
    )
]
inputs[0].set_data_from_numpy(input_features_np)

# Auto-discover outputs
metadata = client.get_model_metadata(TRITON_MODEL_NAME)
outputs = [
    httpclient.InferRequestedOutput(o["name"])
    for o in metadata["outputs"]
]

# Warmup
for _ in range(WARMUP_ITERS):
    client.infer(
        model_name=TRITON_MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
    )

# Benchmark
start = time.time()
for _ in range(BENCH_ITERS):
    client.infer(
        model_name=TRITON_MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
    )
end = time.time()

triton_latency_ms = (end - start) * 1000 / BENCH_ITERS
print(f"Triton HTTP latency: {triton_latency_ms:.2f} ms")

# =====================
# Summary
# =====================
print("\n===== Summary =====")
print(f"Local TorchScript : {local_latency_ms:.2f} ms")
print(f"Triton HTTP       : {triton_latency_ms:.2f} ms")
print(f"Overhead factor   : {triton_latency_ms / local_latency_ms:.2f}×")
