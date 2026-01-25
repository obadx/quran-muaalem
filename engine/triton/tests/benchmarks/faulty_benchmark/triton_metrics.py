import requests
import time

TRITON_METRICS_URL = "http://localhost:8002/metrics"

def scrape_metrics():
    r = requests.get(TRITON_METRICS_URL, timeout=2)
    r.raise_for_status()
    return r.text

def extract_model_metrics(metrics_text, model_name):
    out = {}

    for line in metrics_text.splitlines():
        if model_name not in line:
            continue

        if "nv_inference_request_duration_us" in line:
            out["request_us"] = float(line.split()[-1])

        elif "nv_inference_queue_duration_us" in line:
            out["queue_us"] = float(line.split()[-1])

        elif "nv_inference_compute_input_duration_us" in line:
            out["compute_input_us"] = float(line.split()[-1])

        elif "nv_inference_compute_infer_duration_us" in line:
            out["compute_infer_us"] = float(line.split()[-1])

        elif "nv_inference_compute_output_duration_us" in line:
            out["compute_output_us"] = float(line.split()[-1])

    return out
