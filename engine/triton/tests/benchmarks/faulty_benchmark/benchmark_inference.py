import time
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tritonclient.grpc as grpcclient
from transformers import AutoFeatureExtractor
from triton_metrics import scrape_metrics, extract_model_metrics

# ---------------- CONFIG ----------------
TRITON_URL = "localhost:8001"
MODEL_IDS = [ "muaalem_dyn_off"]

AUDIO_DURATIONS = [3, 5, 15]
BATCH_SIZES = [1, 4, 8, 16]
CONCURRENCY = [1, 2, 4]

WARMUP_RUNS = 5
MEASURE_RUNS = 512
SAMPLING_RATE = 16000

OUT = pathlib.Path("results")
(OUT / "plots").mkdir(parents=True, exist_ok=True)

# ---------------- UTILS ----------------
def append(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()

def latency_stats(lat):
    if not lat:
        return dict(avg=None, p50=None, p95=None, p99=None, min=None, max=None)

    a = np.array(lat)
    return dict(
        avg=float(a.mean()),
        p50=float(np.percentile(a, 50)),
        p95=float(np.percentile(a, 95)),
        p99=float(np.percentile(a, 99)),
        min=float(a.min()),
        max=float(a.max()),
    )

# ---------------- INFER ----------------
def infer(client, model, inp):
    t0 = time.perf_counter()
    client.infer(model, inp)
    return (time.perf_counter() - t0) * 1000

# ---------------- MAIN ----------------
def main():
    client = grpcclient.InferenceServerClient(TRITON_URL)
    proc = AutoFeatureExtractor.from_pretrained(
        "obadx/muaalem-model-v3_2",
        sampling_rate=SAMPLING_RATE,
    )

    for model in MODEL_IDS:
        for sec in AUDIO_DURATIONS:
            wave = np.random.randn(sec * SAMPLING_RATE).astype(np.float32)
            feats = proc(wave, sampling_rate=SAMPLING_RATE, return_tensors="np")[
                "input_features"
            ].astype(np.float16)

            for bs in BATCH_SIZES:
                batched = np.repeat(feats, bs, axis=0)

                for conc in CONCURRENCY:
                    print(f"{model} | {sec}s | bs={bs} | conc={conc}")

                    inp = [grpcclient.InferInput("INPUT__0", batched.shape, "FP16")]
                    inp[0].set_data_from_numpy(batched)

                    # -------- Warmup --------
                    for _ in range(WARMUP_RUNS):
                        infer(client, model, inp)

                    # -------- Measure --------
                    lat = []
                    start = time.time()

                    for i in range(MEASURE_RUNS):
                        try:
                            l = infer(client, model, inp)
                            lat.append(l)
                            append(OUT / "requests.jsonl", {
                                "model": model,
                                "audio_sec": sec,
                                "batch": bs,
                                "lat_ms": l,
                            })
                        except Exception as e:
                            append(OUT / "requests.jsonl", {
                                "model": model,
                                "audio_sec": sec,
                                "batch": bs,
                                "error": str(e),
                            })

                    dur = time.time() - start
                    stats = latency_stats(lat)

                    # -------- Triton Metrics --------
                    raw = scrape_metrics()
                    m = extract_model_metrics(raw, model)
                    append(OUT / "triton_metrics.jsonl", {
                        "model": model,
                        "audio_sec": sec,
                        "batch": bs,
                        **m,
                    })

                    summary = {
                        "model": model,
                        "audio_sec": sec,
                        "batch": bs,
                        "concurrency": conc,
                        "throughput": len(lat) / dur,
                        **stats,
                        **m,
                    }
                    append(OUT / "summary.jsonl", summary)

                    # -------- Plots --------
                    tag = f"{model}_{sec}s_b{bs}"
                    if lat:
                        plt.hist(lat, bins=30)
                        plt.title(tag)
                        plt.xlabel("Latency (ms)")
                        plt.savefig(OUT / "plots" / f"latency_{tag}.png")
                        plt.close()

                    if "queue_us" in m and "compute_infer_us" in m:
                        plt.bar(["queue", "compute"], [
                            m["queue_us"] / 1000,
                            m["compute_infer_us"] / 1000,
                        ])
                        plt.ylabel("ms")
                        plt.title(tag)
                        plt.savefig(OUT / "plots" / f"queue_vs_compute_{tag}.png")
                        plt.close()

if __name__ == "__main__":
    main()
