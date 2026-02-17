# Inference Stack Docker Guide

This document explains how to run and maintain the Docker-based inference stack in this repository.

Main goals:
- Run everything with `docker-compose.yml`
- Understand what `engine/triton/inference_stack/client/app/main.py` does
- Understand Triton model repository structure
- Understand how `engine/export_model_pt.py` and `engine/tensorrt/trt_runtime.py` define the model contract
- Know exactly what to edit when requirements change

## 1. What Runs in Docker

`docker-compose.yml` starts two services:

- `triton`: NVIDIA Triton Inference Server container
- `processor`: FastAPI container that accepts audio files and calls Triton over gRPC

Ports:
- `8000`: Triton HTTP
- `8001`: Triton gRPC
- `8002`: Triton metrics
- `9000`: FastAPI `/infer` endpoint

## 2. Folder Structure (Inference Stack)

```text
engine/triton/inference_stack/
├── client/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       └── main.py
└── triton/
    ├── Dockerfile
    ├── entrypoint.sh
    └── models/
        └── muaalem/
            ├── config.pbtxt
            └── 1/
                └── model.pt
```

Important:
- Triton requires this exact model repository pattern: `<model_name>/<version>/model_file + config.pbtxt`.
- Here, model name is `muaalem`, version is `1`.

## 3. Docker Compose Usage

From repository root:

```bash
docker compose up --build
```

Run in background:

```bash
docker compose up --build -d
```

Stop:

```bash
docker compose down
```

View logs:

```bash
docker compose logs -f triton
docker compose logs -f processor
```

Health check examples:

```bash
curl http://localhost:8000/v2/health/ready
curl http://localhost:8002/metrics | head
```

Test FastAPI endpoint:

```bash
curl -X POST "http://localhost:9000/infer" \
  -F "file=@assets/test.wav"
```

## 4. Triton Container Startup Flow

`engine/triton/inference_stack/triton/Dockerfile`:
- Uses `nvcr.io/nvidia/tritonserver:24.01-py3`
- Installs Python deps (`torch`, `transformers`, `quran-muaalem`, etc.)
- Copies model repository into `/models`
- Runs `/entrypoint.sh`

`engine/triton/inference_stack/triton/entrypoint.sh`:
1. Creates `/models/muaalem/1`
2. Runs `python3 /workspace/engine/export_model_pt.py ...`
3. Exports TorchScript model to `/models/muaalem/1/model.pt`
4. Starts Triton with `--model-repository=/models`

Key point:
- Even if `model.pt` exists in the repo, container startup regenerates it using `export_model_pt.py`.

## 5. What `main.py` Does (Processor Service)

File: `engine/triton/inference_stack/client/app/main.py`

Request flow of `/infer`:
1. Accept uploaded audio file.
2. Load audio with `librosa` at 16 kHz mono.
3. Extract input features with `AutoFeatureExtractor`.
4. Convert features to `float16` and send to Triton gRPC (`triton:8001`).
5. Request outputs `OUTPUT__0`, `OUTPUT__1`, `OUTPUT__2`.
6. Decode CTC IDs to phoneme string using `MultiLevelTokenizer`.
7. Return JSON: `{latency_ms, phonemes}`.

Input/output contract expected by `main.py`:
- Input tensor name: `INPUT__0`
- Output tensor names: `OUTPUT__0`, `OUTPUT__1`, `OUTPUT__2`
- Model name: `muaalem`

Those names must match `config.pbtxt` and the exported model output order.

## 6. Triton Model Contract (`config.pbtxt`)

File: `engine/triton/inference_stack/triton/models/muaalem/config.pbtxt`

Current contract:
- `platform: "pytorch_libtorch"`
- `max_batch_size: 32`
- Input:
  - `INPUT__0` `TYPE_FP16` dims `[-1, 160]`
- Outputs:
  - `OUTPUT__0` `TYPE_FP16` dims `[-1, 43]` (logits)
  - `OUTPUT__1` `TYPE_INT64` dims `[-1]` (argmax ids)
  - `OUTPUT__2` `TYPE_FP32` dims `[-1]` (selected probs)
- Dynamic batching enabled
- GPU instance group count 1

If you edit names, dtypes, or output count here, you must update:
- `engine/triton/inference_stack/client/app/main.py`
- `engine/tensorrt/trt_runtime.py` (`_WrappedModel.forward`)

## 7. Export Model and TRT Runtime Relationship

`engine/export_model_pt.py`:
- Loads pretrained model and feature extractor.
- Uses helpers from `engine/tensorrt/trt_runtime.py`:
  - `_prepare_features`
  - `_WrappedModel`
  - `_attach_levels`
- Traces the wrapped model and saves TorchScript `model.pt`.

`engine/tensorrt/trt_runtime.py`:
- Defines `_WrappedModel.forward`, which controls output order and semantics.
- For Triton export path, wrapper currently returns:
  1. logits
  2. argmax ids
  3. probs of chosen ids
- Also contains optional TensorRT compile/load flow (`build_or_load_trt`) used outside this Triton Docker path.

Practical rule:
- `config.pbtxt` output definitions must match `_WrappedModel.forward` output order.

## 8. What to Edit When Things Change

### A) Change Hugging Face model ID

Edit:
- `engine/triton/inference_stack/triton/entrypoint.sh` (`--model-id ...`)
- `engine/triton/inference_stack/client/app/main.py`:
  - `MultiLevelTokenizer("...")`
  - `AutoFeatureExtractor.from_pretrained("...")`

Then rebuild:

```bash
docker compose up --build -d
```

### B) Change input feature shape or dtype

Likely edit:
- `engine/export_model_pt.py` (dummy/example settings)
- `engine/triton/inference_stack/client/app/main.py` (preprocess dtype/shape)
- `engine/triton/inference_stack/triton/models/muaalem/config.pbtxt` (input definition)

All three must stay aligned.

### C) Change outputs (add/remove levels, rename tensors, change dtype)

Edit:
- `engine/tensorrt/trt_runtime.py` (`_WrappedModel.forward`)
- `engine/triton/inference_stack/triton/models/muaalem/config.pbtxt` (output blocks)
- `engine/triton/inference_stack/client/app/main.py` (requested outputs + decode logic)

### D) Tune batching/performance

Edit:
- `engine/triton/inference_stack/triton/models/muaalem/config.pbtxt`
  - `max_batch_size`
  - `dynamic_batching.preferred_batch_size`
  - `dynamic_batching.max_queue_delay_microseconds`
  - `instance_group.count`

Then restart stack.

### E) Change service ports or networking

Edit:
- `docker-compose.yml` ports
- `engine/triton/inference_stack/client/app/main.py` if Triton host/port changes from `triton:8001`

## 9. Common Failure Modes

- Triton model load fails:
  - mismatch between `config.pbtxt` and exported TorchScript outputs
- Client inference fails:
  - output tensor names in `main.py` do not match `config.pbtxt`
- GPU not visible in container:
  - Docker NVIDIA runtime/toolkit not configured on host
- Wrong decoding:
  - tokenizer model ID in `main.py` does not match exported model ID

## 10. Quick Change Checklist

When editing inference behavior, validate in this order:
1. `_WrappedModel.forward` output order/types
2. `config.pbtxt` tensor names/dtypes/dims
3. `main.py` request/parse/decode logic
4. `docker compose up --build`
5. `/v2/health/ready` and one `/infer` test request
