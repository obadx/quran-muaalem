# Inference Setup & Usage

This guide summarizes the local inference workflow we set up (TensorRT/TorchScript, FastAPI endpoints, and benchmarking).

## 1) Build the TensorRT engine (optional)

If TensorRT is available, compile an optimized engine:

```bash
python engine/convert_model_tensorrt.py --device cuda --output engine/muaalem_trt.ts
```

Notes:
- If TensorRT compilation fails, the script falls back to a TorchScript model and still saves it to `engine/muaalem_trt.ts`.
- You can tune dynamic shapes with `--min-seconds`, `--opt-seconds`, `--max-seconds`, and `--max-batch-size`.

## 2) Run the API server

```bash
.venv/bin/uvicorn engine.inference_server:app --host 127.0.0.1 --port 8000
```

Disable TensorRT and force TorchScript:

```bash
MUALEM_DISABLE_TRT=1 .venv/bin/uvicorn engine.inference_server:app --host 127.0.0.1 --port 8000
```

## 3) Endpoints

### Sliding window (realtime-style)

```bash
POST /inference/sliding-window
{
  "audio": [float, ...],
  "sampling_rate": 16000,
  "window_ms": 6000,
  "chunk_ms": 300
}
```

The server auto-resamples to 16 kHz for sliding-window requests.

### Offline

```bash
POST /inference/offline
{
  "audio": [float, ...],
  "sampling_rate": 16000
}
```

For batch inference:

```bash
POST /inference/offline
{
  "audios": [[float, ...], [float, ...]],
  "sampling_rate": 16000
}
```

## 4) Quick test with a WAV file

Use the helper script (resamples to 16 kHz with `librosa`):

```bash
.venv/bin/python engine/test_endpoint.py --wav assets/test.wav --endpoint sliding --mode realtime --realtime-chunk-ms 1000
```

Single request:

```bash
.venv/bin/python engine/test_endpoint.py --wav assets/test.wav --endpoint sliding --mode single
```

Offline:

```bash
.venv/bin/python engine/test_endpoint.py --wav assets/test.wav --endpoint offline
```

## 5) Benchmark offline endpoint (with/without TensorRT)

With TRT (default):

```bash
.venv/bin/uvicorn engine.inference_server:app --host 127.0.0.1 --port 8000
.venv/bin/python engine/benchmark_client.py --endpoint offline --requests 8 --batch-size 4
```

Without TRT:

```bash
MUALEM_DISABLE_TRT=1 .venv/bin/uvicorn engine.inference_server:app --host 127.0.0.1 --port 8000
.venv/bin/python engine/benchmark_client.py --endpoint offline --requests 8 --batch-size 4
```

## 6) Dependencies

Suggested optional deps (see `pyproject.toml`):

- `fastapi`, `uvicorn`
- `torch-tensorrt`
- `httpx`
- `librosa` (for the WAV test helper)
