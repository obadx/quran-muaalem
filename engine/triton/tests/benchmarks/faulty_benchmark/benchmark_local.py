import time
import json
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt

from datetime import datetime
from transformers import AutoFeatureExtractor

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "model.pt"
PROCESSOR_ID = "obadx/muaalem-model-v3_2"   # or your exact processor repo
DEVICE = "cuda" 
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SAMPLE_RATE = 16000

AUDIO_DURATIONS = [3, 5, 15]   # seconds
BATCH_SIZES = [1, 4, 8, 16]
REQUEST_COUNTS = [512]

OUT_DIR = pathlib.Path("benchmark_results/local")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================
def generate_audio(duration_s: int, batch_size: int):
    samples = duration_s * SAMPLE_RATE
    return np.random.randn(batch_size, samples).astype(np.float32)

def latency_stats(lat):
    if len(lat) == 0:
        return dict(avg_ms=None, p50_ms=None, p95_ms=None,
                    min_ms=None, max_ms=None)

    lat = np.array(lat)
    return dict(
        avg_ms=float(lat.mean()),
        p50_ms=float(np.percentile(lat, 50)),
        p95_ms=float(np.percentile(lat, 95)),
        min_ms=float(lat.min()),
        max_ms=float(lat.max()),
    )

def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

def plot_latency(path, title):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "latency_ms" in r:
                rows.append(r)

    if not rows:
        return

    lat = [r["latency_ms"] for r in rows]

    plt.figure()
    plt.plot(lat, label="latency (ms)")
    plt.xlabel("request")
    plt.ylabel("latency (ms)")
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"))
    plt.close()

# ============================================================
# LOAD MODEL + PREPROCESSOR
# ============================================================
print("Loading feature extractor...")
processor = AutoFeatureExtractor.from_pretrained(PROCESSOR_ID)

print("Loading TorchScript model...")
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# Warmup (important!)
with torch.no_grad():
    dummy_audio = generate_audio(3, 1)
    features = processor(
        dummy_audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    x = features["input_features"].to(DEVICE, dtype=DTYPE)
    _ = model(x)

# ============================================================
# BENCHMARK LOOP
# ============================================================
def run():
    for duration in AUDIO_DURATIONS:
        for batch in BATCH_SIZES:
            for requests in REQUEST_COUNTS:

                tag = f"{duration}s_batch{batch}_req{requests}"
                print(f"\n▶ {tag}")

                out_file = OUT_DIR / f"{tag}.jsonl"
                latencies = []

                wall_start = time.time()

                try:
                    for i in range(requests):
                        audio = generate_audio(duration, batch)

                        t0 = time.perf_counter()

                        # -------- PREPROCESSING --------
                        features = processor(
                            audio,
                            sampling_rate=SAMPLE_RATE,
                            return_tensors="pt",
                            padding=True,
                        )
                        x = features["input_features"].to(DEVICE, dtype=DTYPE)

                        # -------- MODEL --------
                        with torch.no_grad():
                            _ = model(x)

                        if DEVICE == "cuda":
                            torch.cuda.synchronize()

                        t1 = time.perf_counter()
                        lat_ms = (t1 - t0) * 1000
                        latencies.append(lat_ms)

                        append_jsonl(out_file, {
                            "timestamp": datetime.utcnow().isoformat(),
                            "iteration": i,
                            "audio_duration_s": duration,
                            "batch_size": batch,
                            "latency_ms": lat_ms,
                        })

                except Exception as e:
                    print(f"❌ Error during {tag}: {e}")

                total_time = time.time() - wall_start
                stats = latency_stats(latencies)

                summary = {
                    "tag": tag,
                    "audio_duration_s": duration,
                    "batch_size": batch,
                    "requests": requests,
                    "total_time_s": total_time,
                    "throughput_rps": requests / total_time if total_time > 0 else None,
                    **stats,
                }

                append_jsonl(out_file, summary)
                plot_latency(out_file, tag)

                print("✔ Summary:", summary)

# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":
    run()
