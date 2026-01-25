import time
import json
import pathlib
import threading
import numpy as np
import matplotlib.pyplot as plt
import tritonclient.grpc as grpcclient
from transformers import AutoFeatureExtractor
from triton_metrics import scrape_metrics, extract_model_metrics

# ---------------- CONFIG ----------------
TRITON_URL = "localhost:8001"
MODEL_IDS = ["muaalem_dyn_on"]

AUDIO_DURATIONS = [10]
NUM_REQUESTS = 128

WARMUP_RUNS = 5
SAMPLING_RATE = 16000

OUT = pathlib.Path("results")
(OUT / "plots").mkdir(parents=True, exist_ok=True)

# ---------------- UTILS ----------------
def append(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()

def print_summary(summary):
    print("\n=== Benchmark Summary ===")
    print(f"Model        : {summary['model']}")
    print(f"Audio (sec)  : {summary['audio_sec']}")
    print(f"Throughput  : {summary['throughput']:.2f} req/s")

    print("\nLatency (ms):")
    for k in ["avg", "p50", "p95", "p99", "min", "max"]:
        v = summary.get(k)
        if v is not None:
            print(f"  {k:>4}: {v:8.2f}")

    print("\nTriton timing (ms):")
    if "queue_us" in summary:
        print(f"  queue   : {summary['queue_us'] / 1000:.2f}")
    if "compute_infer_us" in summary:
        print(f"  compute : {summary['compute_infer_us'] / 1000:.2f}")
    if "compute_input_us" in summary:
        print(f"  input   : {summary['compute_input_us'] / 1000:.2f}")
    if "compute_output_us" in summary:
        print(f"  output  : {summary['compute_output_us'] / 1000:.2f}")

    print("=========================\n")

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

# ---------------- ASYNC BURST ----------------
def burst_async(client, model, inp, n_requests):
    latencies = []
    completed = 0
    lock = threading.Lock()
    done = threading.Event()

    def callback(start_t, *, result=None, error=None):
        nonlocal completed
        end_t = time.perf_counter()
        lat_ms = (end_t - start_t) * 1000

        with lock:
            completed += 1

            if error is not None:
                append(OUT / "requests.jsonl", {
                    "model": model,
                    "error": str(error),
                })
            else:
                latencies.append(lat_ms)
                append(OUT / "requests.jsonl", {
                    "model": model,
                    "lat_ms": lat_ms,
                })

            if completed == n_requests:
                done.set()

    start_wall = time.time()

    # 🚀 Fire all requests
    for _ in range(n_requests):
        t0 = time.perf_counter()
        client.async_infer(
            model_name=model,
            inputs=inp,
            callback=lambda *, result=None, error=None, t=t0: callback(
                t, result=result, error=error
            ),
        )

    done.wait()
    dur = time.time() - start_wall
    return latencies, dur


# ---------------- MAIN ----------------
def main():
    client = grpcclient.InferenceServerClient(TRITON_URL)
    proc = AutoFeatureExtractor.from_pretrained(
        "obadx/muaalem-model-v3_2",
        sampling_rate=SAMPLING_RATE,
    )

    for model in MODEL_IDS:
        for sec in AUDIO_DURATIONS:
            print(f"\n{model} | {sec}s | BURST")

            wave = np.random.randn(sec * SAMPLING_RATE).astype(np.float32)
            feats = proc(
                wave,
                sampling_rate=SAMPLING_RATE,
                return_tensors="np",
            )["input_features"].astype(np.float16)

            inp = [grpcclient.InferInput(
                "INPUT__0",
                feats.shape,
                "FP16",
            )]
            inp[0].set_data_from_numpy(feats)

            # -------- Warmup --------
            for _ in range(WARMUP_RUNS):
                client.infer(model, inp)

            # -------- Measure --------
            lat, dur = burst_async(
                client,
                model,
                inp,
                NUM_REQUESTS,
            )

            stats = latency_stats(lat)

            # -------- Triton Metrics --------
            raw = scrape_metrics()
            m = extract_model_metrics(raw, model)

            append(OUT / "triton_metrics.jsonl", {
                "model": model,
                "audio_sec": sec,
                **m,
            })

            summary = {
                "model": model,
                "audio_sec": sec,
                "throughput": len(lat) / dur if dur > 0 else 0,
                **stats,
                **m,
            }
            append(OUT / "summary.jsonl", summary)
            print_summary(summary)


            # -------- Plots --------
            tag = f"{model}_{sec}s_burst"

            if lat:
                plt.hist(lat, bins=30)
                plt.title(tag)
                plt.xlabel("Latency (ms)")
                plt.savefig(OUT / "plots" / f"latency_{tag}.png")
                plt.close()

            if "queue_us" in m and "compute_infer_us" in m:
                plt.bar(
                    ["queue", "compute"],
                    [m["queue_us"] / 1000, m["compute_infer_us"] / 1000],
                )
                plt.ylabel("ms")
                plt.title(tag)
                plt.savefig(
                    OUT / "plots" / f"queue_vs_compute_{tag}.png"
                )
                plt.close()

if __name__ == "__main__":
    main()
